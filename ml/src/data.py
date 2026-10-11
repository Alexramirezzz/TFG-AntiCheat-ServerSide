"""Phase A - data layer.

Loads the CSV sessions written by SessionRecorder, validates their schema, normalises legacy
labels, applies the cleaning rules derived from the first real recordings and computes the
per-session click streams/intervals used by every later phase.

Run as a script to (re)write results/data_report.md and data/processed/*.csv:
    python -m src.data
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

log = logging.getLogger(__name__)

SCHEMAS = {
    "clicks": ["session_id", "t_ns", "source", "label"],
    "moves": ["session_id", "t_ns", "dx", "dy", "dz", "on_ground", "tps", "ping", "label"],
    "breaks": ["session_id", "t_ns", "label"],
    "sessions": ["session_id", "participant", "label", "start_utc", "end_utc",
                 "duration_s", "events_dropped"],
}
CLICK_SOURCES = {"swing", "interact_air", "interact_block"}


class SchemaError(ValueError):
    pass


@dataclass
class Dataset:
    clicks: pd.DataFrame
    moves: pd.DataFrame
    breaks: pd.DataFrame
    sessions: pd.DataFrame
    issues: list[str]


# ------------------------------------------------------------------ loading / validation
def normalise_label(label: str) -> str:
    label = str(label).strip()
    return C.LEGACY_LABELS.get(label, label)


def _read(path: Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", dtype=str, keep_default_na=False)
    if list(df.columns) != SCHEMAS[kind]:
        raise SchemaError(f"{path}: expected columns {SCHEMAS[kind]}, got {list(df.columns)}")
    return df


def _coerce(df: pd.DataFrame, kind: str, path: Path) -> pd.DataFrame:
    df = df.copy()
    num = {
        "clicks": {"t_ns": "int64"},
        "moves": {"t_ns": "int64", "dx": "float", "dy": "float", "dz": "float",
                  "on_ground": "int64", "tps": "float", "ping": "int64"},
        "breaks": {"t_ns": "int64"},
        "sessions": {"duration_s": "int64", "events_dropped": "int64"},
    }[kind]
    for col, dtype in num.items():
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(dtype)
        except (ValueError, TypeError) as e:
            raise SchemaError(f"{path}: column {col} is not numeric ({e})") from e
    if "label" in df:
        df["label"] = df["label"].map(normalise_label)
        bad = set(df["label"]) - C.ALL_LABELS
        if bad:
            raise SchemaError(f"{path}: unknown labels {sorted(bad)}")
    if kind == "clicks":
        bad = set(df["source"]) - CLICK_SOURCES
        if bad:
            raise SchemaError(f"{path}: unknown click sources {sorted(bad)}")
    if "t_ns" in df and (df["t_ns"] < 0).any():
        raise SchemaError(f"{path}: negative t_ns")
    return df


def load_raw(raw_dir: Path | None = None, origin: str = "real") -> Dataset:
    """Load every <participant>/ folder under raw_dir. Missing folders -> empty dataset."""
    raw_dir = Path(raw_dir or C.RAW_DIR)
    parts = {k: [] for k in SCHEMAS}
    issues: list[str] = []
    if not raw_dir.exists():
        issues.append(f"raw data directory {raw_dir} does not exist")
    for pdir in sorted(p for p in raw_dir.glob("*") if p.is_dir()) if raw_dir.exists() else []:
        for kind in SCHEMAS:
            f = pdir / f"{kind}.csv"
            if not f.exists():
                issues.append(f"{pdir.name}: missing {kind}.csv")
                continue
            df = _coerce(_read(f, kind), kind, f)
            df["participant_dir"] = pdir.name
            parts[kind].append(df)
    out = {k: (pd.concat(v, ignore_index=True) if v else
               pd.DataFrame(columns=SCHEMAS[k] + ["participant_dir"])) for k, v in parts.items()}
    sessions = out["sessions"].drop_duplicates("session_id")
    # sessions.csv row is written at /record stop. Sessions without it (crash) are kept but flagged.
    seen = set(sessions["session_id"])
    for kind in ("clicks", "moves", "breaks"):
        orphan = sorted(set(out[kind]["session_id"]) - seen)
        for sid in orphan:
            sub = out[kind][out[kind]["session_id"] == sid]
            issues.append(f"session {sid}: present in {kind}.csv but not in sessions.csv (kept, incomplete)")
            sessions = pd.concat([sessions, pd.DataFrame([{
                "session_id": sid, "participant": sub["participant_dir"].iloc[0],
                "label": sub["label"].iloc[0], "start_utc": "", "end_utc": "",
                "duration_s": int(sub["t_ns"].max() // 1e9), "events_dropped": -1,
                "participant_dir": sub["participant_dir"].iloc[0]}])], ignore_index=True)
            seen.add(sid)
    for _, s in sessions.iterrows():
        if s["participant"] != s["participant_dir"]:
            issues.append(f"session {s['session_id']}: participant {s['participant']} stored in folder {s['participant_dir']}")
        if s["events_dropped"] > 0:
            issues.append(f"session {s['session_id']}: {s['events_dropped']} events dropped by recorder backpressure")
    # event label must match the session label
    lab = sessions.set_index("session_id")["label"]
    for kind in ("clicks", "moves", "breaks"):
        df = out[kind]
        if len(df):
            mism = df[df["label"] != df["session_id"].map(lab)]
            if len(mism):
                issues.append(f"{kind}: {len(mism)} rows whose label differs from sessions.csv")
    sessions = sessions.copy()
    sessions["activity"] = sessions["label"].map(C.ACTIVITY)
    sessions["origin"] = origin
    # real sessions: the "level" is the recorded label (auto_fixed / auto_jitter / human_*)
    sessions["level"] = sessions["label"].map(lambda l: "human" if l in C.HUMAN_LABELS else f"real_{l}")
    for kind in ("clicks", "moves", "breaks"):
        out[kind] = out[kind].merge(sessions[["session_id", "participant"]], on="session_id", how="left")
        out[kind]["origin"] = origin
    return Dataset(out["clicks"], out["moves"], out["breaks"], sessions.reset_index(drop=True), issues)


# ------------------------------------------------------------------ click stream cleaning
def clean_click_stream(clicks: pd.DataFrame, breaks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Causal cleaning of ONE session (same algorithm as the Java ClickStreamCleaner).

    1. Only `swing` events are clicks; interact_* are used solely as dig-start markers.
    2. A swing closer than DEDUP_MS to the last *accepted* swing is a duplicate (counted, dropped).
    3. Held-button (mining) state machine. A dig marker (interact_block or break) opens a dig;
       while the button stays held the client sends one swing per tick, so the dig run lasts
       while consecutive swings are <= HELD_MAX_IV_MS apart (a break does NOT end it: in the real
       data the break arrives before the held swings when blocks break instantly):
         interact_block | break -> digging = True, last_dig = t
         swing          -> fresh = t - last_dig <= FRESH_DIG_MS
                           held  = digging and iv <= HELD_MAX_IV_MS and (not fresh or prev_held)
                           if not fresh and iv > RELEASE_GAP_MS: digging = False
       A late held swing (HELD_MAX_IV_MS < iv <= RELEASE_GAP_MS = 85 ms, e.g. one slow server tick) is
       evaluated as a click but does not end the dig, so the rest of the run stays held.
    Returns one row per accepted swing: t_ms, iv_ms (NaN for the first), held, n_dup.
    """
    ev = [(float(t) / 1e6, src) for t, src in zip(clicks["t_ns"], clicks["source"])]
    if breaks is not None and len(breaks):
        ev += [(float(t) / 1e6, "break") for t in breaks["t_ns"]]
    # tie order: dig start, then break, then swing (ties are practically impossible with ns)
    prio = {"interact_block": 0, "interact_air": 1, "break": 2, "swing": 3}
    ev.sort(key=lambda x: (x[0], prio[x[1]]))

    rows: list[list] = []
    digging, prev_held = False, False
    last_dig = -math.inf
    last_acc = None
    for t, src in ev:
        if src in ("interact_block", "break"):
            digging, last_dig = True, t
        elif src == "swing":
            if last_acc is not None and t - last_acc < C.DEDUP_MS:
                rows[-1][3] += 1
                continue
            iv = math.nan if last_acc is None else t - last_acc
            fresh = (t - last_dig) <= C.FRESH_DIG_MS
            held = bool(digging and not math.isnan(iv) and iv <= C.HELD_MAX_IV_MS
                        and (not fresh or prev_held))
            if not fresh and (math.isnan(iv) or iv > C.RELEASE_GAP_MS):
                digging = False
            prev_held = held
            rows.append([t, iv, held, 0])
            last_acc = t
    return pd.DataFrame(rows, columns=["t_ms", "iv_ms", "held", "n_dup"])


def build_click_streams(ds: Dataset) -> pd.DataFrame:
    """Clean every session; returns accepted swings with session metadata."""
    out = []
    br = ds.breaks.groupby("session_id") if len(ds.breaks) else None
    for sid, cl in ds.clicks.groupby("session_id"):
        b = br.get_group(sid) if br is not None and sid in br.groups else None
        s = clean_click_stream(cl, b)
        s.insert(0, "session_id", sid)
        out.append(s)
    cols = ["session_id", "t_ms", "iv_ms", "held", "n_dup"]
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
    meta = ds.sessions[[c for c in ["session_id", "participant", "label", "activity", "origin", "level"]
                        if c in ds.sessions.columns]]
    return df.merge(meta, on="session_id", how="left")


def naive_intervals(clicks: pd.DataFrame, sources: set[str]) -> pd.DataFrame:
    """Intervals between consecutive raw events of the given sources, NO de-duplication.

    sources={'interact_air','interact_block'} reproduces what the Java onPlayerInteract
    detector sees; using all sources shows the effect of one physical click -> 3 events.
    Java uses System.currentTimeMillis(), so differences are floored to whole ms.
    """
    out = []
    for sid, g in clicks[clicks["source"].isin(sources)].groupby("session_id"):
        t = np.sort(g["t_ns"].to_numpy()) / 1e6
        if len(t) < 2:
            continue
        out.append(pd.DataFrame({"session_id": sid, "iv_ms": np.floor(np.diff(t))}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["session_id", "iv_ms"])


def trim_moves(moves: pd.DataFrame, seconds: float = C.TRIM_MOVES_S) -> pd.DataFrame:
    """Drop the first `seconds` after the FIRST move event of every session (spawn gravity drop).

    Relative to the first move, not to session start: in the real data moves begin ~12 s after
    /record start.
    """
    if moves.empty:
        return moves
    first = moves.groupby("session_id")["t_ns"].transform("min")
    return moves[moves["t_ns"] - first >= seconds * 1e9].reset_index(drop=True)


# ------------------------------------------------------------------ tick quantization
def tick_residuals(iv_ms: np.ndarray, tick: float = C.TICK_MS) -> np.ndarray:
    """Signed distance of each interval to the nearest multiple of one tick, in (-25, 25]."""
    iv = np.asarray(iv_ms, dtype=float)
    return iv - tick * np.round(iv / tick)


def quantization_summary(iv_ms: np.ndarray) -> dict:
    iv = np.asarray(iv_ms, dtype=float)
    iv = iv[(iv > 0) & (iv < C.PAUSE_MS)]
    if len(iv) == 0:
        return {"n": 0}
    r = tick_residuals(iv)
    # Under "no quantization" residuals would be ~uniform on (-25, 25]
    ks = stats.kstest(r, stats.uniform(loc=-25, scale=50).cdf)
    return {
        "n": int(len(iv)),
        "frac_within_5ms": float(np.mean(np.abs(r) <= C.TICK_NEAR_MS)),
        "frac_within_5ms_if_uniform": C.TICK_NEAR_MS * 2 / C.TICK_MS,
        "residual_mean_ms": float(np.mean(r)),
        "residual_sd_ms": float(np.std(r, ddof=1)) if len(r) > 1 else float("nan"),
        "residual_mad_ms": float(stats.median_abs_deviation(r)),
        "ks_vs_uniform_stat": float(ks.statistic),
        "ks_vs_uniform_p": float(ks.pvalue),
    }


# ------------------------------------------------------------------ baselines on raw data
def baseline_event_flags(ds: Dataset, streams: pd.DataFrame) -> pd.DataFrame:
    """Per-event application of the ORIGINAL thesis thresholds (Annex I) on real data."""
    rows = []
    thr = C.ORIG_CLICK_MIN_DIFF_MS
    def cnt(iv):  # Java: diff > 0 && diff < 66
        iv = np.asarray(iv, dtype=float)
        iv = iv[~np.isnan(iv)]
        return int(len(iv)), int(np.sum((iv > 0) & (iv < thr)))
    for sid, sess in ds.sessions.set_index("session_id").iterrows():
        cl = ds.clicks[ds.clicks["session_id"] == sid]
        n1, f1 = cnt(naive_intervals(cl, {"interact_air", "interact_block"})["iv_ms"])
        n2, f2 = cnt(naive_intervals(cl, CLICK_SOURCES)["iv_ms"])
        st = streams[streams["session_id"] == sid]
        n3, f3 = cnt(np.floor(st["iv_ms"]))
        act = st[~st["held"].astype(bool)]
        n4, f4 = cnt(np.floor(act["iv_ms"]))
        bt = np.sort(ds.breaks.loc[ds.breaks["session_id"] == sid, "t_ns"].to_numpy()) / 1e6
        bd = np.floor(np.diff(bt)) if len(bt) > 1 else np.array([])
        mv = trim_moves(ds.moves[ds.moves["session_id"] == sid])
        mv_all = ds.moves[ds.moves["session_id"] == sid]
        def mvflags(m):
            h = np.hypot(m["dx"], m["dz"])
            fly = (m["on_ground"] == 0) & (m["dy"].abs() > C.ORIG_FLY_DY)
            return (int(len(m)), int((h > C.ORIG_SPEED_TEXT).sum()),
                    int((h > C.ORIG_SPEED_EFFECTIVE).sum()), int(fly.sum()))
        a = mvflags(mv_all); b = mvflags(mv)
        rows.append({
            "session_id": sid, "participant": sess["participant"], "label": sess["label"],
            "naive_java_interact_n": n1, "naive_java_interact_flags": f1,
            "naive_all_events_n": n2, "naive_all_events_flags": f2,
            "dedup_swing_n": n3, "dedup_swing_flags": f3,
            "cleaned_active_n": n4, "cleaned_active_flags": f4,
            "breaks_intervals_n": int(len(bd)),
            "fastbreak_flags": int(np.sum(bd < C.ORIG_FASTBREAK_MIN_MS)),
            "moves_untrimmed_n": a[0], "speed1.2_flags_untrimmed": a[1],
            "speed0.5_flags_untrimmed": a[2], "fly_flags_untrimmed": a[3],
            "moves_trimmed_n": b[0], "speed1.2_flags": b[1], "speed0.5_flags": b[2],
            "fly_flags": b[3],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.3g}" if isinstance(v, float) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def write_report(ds: Dataset, streams: pd.DataFrame, flags: pd.DataFrame, quant: dict,
                 path: Path = C.RESULTS_DIR / "data_report.md") -> None:
    s = ds.sessions
    lines = ["# Data report (REAL data, generated by `src/data.py`)", "",
             f"Raw directory: `{C.RAW_DIR.name}/` (git-ignored). Regenerate with `run_all.ps1`.", ""]
    if s.empty:
        lines += ["**No real sessions found.** Every later result is synthetic (DEMO).", ""]
    else:
        ev = pd.DataFrame({
            "clicks": ds.clicks.groupby("session_id").size(),
            "swings": ds.clicks[ds.clicks.source == "swing"].groupby("session_id").size(),
            "accepted_swings": streams.groupby("session_id").size(),
            "held_swings": streams.groupby("session_id")["held"].sum(),
            "duplicates_merged": streams.groupby("session_id")["n_dup"].sum(),
            "moves": ds.moves.groupby("session_id").size(),
            "breaks": ds.breaks.groupby("session_id").size(),
        }).fillna(0).astype(int)
        per = s.set_index("session_id")[["participant", "label", "duration_s", "events_dropped"]].join(ev).reset_index()
        lines += ["## Sessions", "", _md_table(per)]
        by_label = s.groupby("label").agg(sessions=("session_id", "size"),
                                          participants=("participant", "nunique"),
                                          duration_s=("duration_s", "sum"),
                                          events_dropped=("events_dropped", "sum")).reset_index()
        by_part = s.groupby("participant").agg(sessions=("session_id", "size"),
                                               labels=("label", lambda x: ",".join(sorted(set(x)))),
                                               duration_s=("duration_s", "sum"),
                                               events_dropped=("events_dropped", "sum")).reset_index()
        src = ds.clicks.groupby("source").size().rename("events").reset_index()
        lines += ["## Counts per label", "", _md_table(by_label),
                  "## Counts per participant", "", _md_table(by_part),
                  "## Click events per source (raw)", "", _md_table(src)]
        raw_gaps = []
        for _, g in ds.clicks.groupby("session_id"):
            t = np.sort(g["t_ns"].to_numpy()) / 1e6
            raw_gaps.append(np.diff(t))
        raw_gaps = np.concatenate(raw_gaps) if raw_gaps else np.array([])
        lines += [f"Raw click events closer than {C.DEDUP_MS:.0f} ms to the previous event (any source): "
                  f"**{int(np.sum(raw_gaps < C.DEDUP_MS))}** of {len(raw_gaps)} gaps "
                  "(one physical click -> interact_block/interact_air/swing).", ""]
        # breaks arriving within 5 ms of a dig start = block broken instantly (creative mode or
        # instant-mine blocks); recorder does not log the game mode, so this is only an indicator
        inst = []
        for sid, g in ds.breaks.groupby("session_id"):
            dig = np.sort(ds.clicks.loc[(ds.clicks.session_id == sid) &
                                        (ds.clicks.source == "interact_block"), "t_ns"].to_numpy()) / 1e6
            for bt in g["t_ns"].to_numpy() / 1e6:
                k = np.searchsorted(dig, bt, "right") - 1
                inst.append(k >= 0 and bt - dig[k] <= 5.0)
        if inst:
            lines += [f"Breaks arriving <= 5 ms after an interact_block (instant breaks): "
                      f"**{int(np.sum(inst))} of {len(inst)}**. Many instant breaks suggest creative mode or "
                      "instant-mine blocks; the recorder does not store the game mode.", ""]
        lines += ["## Issues found during validation", ""]
        lines += [f"- {i}" for i in ds.issues] or ["- none"]
        lines += ["", "## Timing quantization (accepted swings, intervals < 1 s)", "",
                  "`all_*` = held + active swings (held swings are one per client tick by construction);",
                  "`active_*` = non-held swings only, i.e. real clicks. Only the active numbers speak",
                  "about click timing; with few active intervals they are anecdotal.", ""]
        lines += [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in quant.items()]
        lines += ["", "Interpretation: if the server saw continuous time, the distance of each interval to the",
                  "nearest multiple of 50 ms would be ~uniform on (-25, 25] ms and only "
                  f"{quant.get('all_frac_within_5ms_if_uniform', 0):.0%} of intervals would fall within +-5 ms.",
                  "See `figures/real_tick_quantization.png`.", "",
                  "## Original thesis thresholds applied per event (REAL data)", "",
                  "Columns: `naive_java_interact` = consecutive LEFT_CLICK_AIR/BLOCK events exactly as",
                  "`onPlayerInteract` sees them; `naive_all_events` = every click event without de-duplication;",
                  "`dedup_swing` = swings after 15 ms de-duplication; `cleaned_active` = de-duplicated swings",
                  "excluding held-button (mining) swings. Click flag: 0 < diff < 66 ms (1000/15, integer",
                  "division as in Annex I). FastBreak: break interval < 450 ms. Speed: horizontal",
                  "displacement per move event > 1.2 (thesis text) or > 0.5 (effective value: config.yml",
                  "ships 0.1, clamped to 0.5 by loadConfigValues). Fly: not on ground and |dy| > 1.0.",
                  "Moves are reported with and without the 5 s spawn trim.", "",
                  _md_table(flags.T.reset_index().rename(columns={"index": "metric"}) if len(flags) else flags),
                  "All sessions are labelled human: every flag above is a false positive of the",
                  "corresponding rule on legitimate play (tiny sample: anecdotal, not a rate estimate).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_quantization(iv_ms: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    iv = np.asarray(iv_ms, dtype=float)
    iv = iv[(iv > 0) & (iv < C.PAUSE_MS)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(iv, bins=np.arange(0, 1000, 5), color="#3b6ea5")
    for k in range(1, 20):
        ax[0].axvline(k * 50, color="grey", lw=0.5, ls=":")
    ax[0].set(xlabel="interval between accepted swings (ms)", ylabel="count",
              title=f"REAL swing intervals (n={len(iv)})")
    ax[1].hist(tick_residuals(iv), bins=np.arange(-25, 26, 2.5), color="#3b6ea5")
    ax[1].set(xlabel="distance to nearest multiple of 50 ms (ms)", ylabel="count",
              title="REAL: residual to tick grid")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = load_raw()
    streams = build_click_streams(ds)
    flags = baseline_event_flags(ds, streams) if len(ds.sessions) else pd.DataFrame()
    iv = streams["iv_ms"].dropna().to_numpy() if len(streams) else np.array([])
    quant = {f"all_{k}": v for k, v in quantization_summary(iv).items()}
    act = streams.loc[~streams["held"].astype(bool), "iv_ms"].dropna().to_numpy() if len(streams) else iv
    quant.update({f"active_{k}": v for k, v in quantization_summary(act).items()})
    if len(iv):
        plot_quantization(iv, C.FIG_DIR / "real_tick_quantization.png")
    write_report(ds, streams, flags, quant)
    # Derived real data stays local (data/processed is git-ignored)
    streams.to_csv(C.PROCESSED_DIR / "real_click_streams.csv", index=False)
    ds.sessions.to_csv(C.PROCESSED_DIR / "real_sessions.csv", index=False)
    pd.DataFrame([quant]).to_csv(C.TAB_DIR / "real_quantization.csv", index=False)
    if len(flags):
        flags.to_csv(C.TAB_DIR / "real_baseline_event_flags.csv", index=False)
    log.info("sessions=%d accepted swings=%d issues=%d", len(ds.sessions), len(streams), len(ds.issues))


if __name__ == "__main__":
    main()
