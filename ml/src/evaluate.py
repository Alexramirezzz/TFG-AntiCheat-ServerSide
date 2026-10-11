"""Phase D - evaluation protocol.

Splits:   StratifiedGroupKFold over PARTICIPANTS (no participant in train and test).
          Inside each training fold, 25% of the HUMAN participants are held out as a calibration
          set used only to place the decision threshold at the target FPR. The test fold is never
          used for fitting or for thresholds.
Metrics:  recall at FPR <= 0.1% and <= 1% (threshold from calibration humans; the realised test
          FPR is reported next to it), PR-AUC, F1 and confusion matrix at the 1% threshold,
          session-level alarm rates, and 95% bootstrap CIs resampling participants within class.
Real data: if real windows of both classes from >= 2 groups exist the same protocol runs on them
          (grouped by participant, or by session with a flagged limitation). Otherwise the final
          synthetic-trained detectors are applied to the real human windows (FPR only).

    python -m src.evaluate
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import config as C
from .data import build_click_streams, load_raw
from .features import make_windows, scored
from .train import FIXED_RULES, available_models, is_attack, make_detector

log = logging.getLogger(__name__)
TARGETS = (0.001, 0.01)
STANDARD_ATTACKS = ["none", "uniform", "gaussian"]
N_BOOT = 1000


# ------------------------------------------------------------------ window preparation
def _violation_counts(t_start, t_end, ev_t, ev_flag):
    """Number of flagged raw gaps whose end lies within [t_start, t_end] (prefix sums)."""
    cs = np.r_[0, np.cumsum(ev_flag)]
    i0 = np.searchsorted(ev_t, t_start, "left")
    i1 = np.searchsorted(ev_t, t_end, "right")
    return cs[i1] - cs[i0]


def prepare_windows(streams: pd.DataFrame, raw_clicks: pd.DataFrame, window_type: str = "count",
                    moves: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scored windows + the two ORIGINAL-rule violation counts used by the fixed baselines."""
    w = scored(make_windows(streams, window_type, moves=moves, keep_sequences=True))
    if w.empty:
        return w
    thr = C.ORIG_CLICK_MIN_DIFF_MS
    w["clean_violations"] = [int(np.sum((np.floor(s) > 0) & (np.floor(s) < thr))) for s in w["seq"]]
    naive = np.zeros(len(w), dtype=int)
    inter = raw_clicks[raw_clicks["source"].isin(["interact_air", "interact_block"])]
    groups = {sid: np.sort(g["t_ns"].to_numpy()) / 1e6 for sid, g in inter.groupby("session_id")}
    for sid, idx in w.groupby("session_id").groups.items():
        t = groups.get(sid)
        if t is None or len(t) < 2:
            continue
        d = np.floor(np.diff(t))
        flag = ((d > 0) & (d < thr)).astype(int)
        sub = w.loc[idx]
        naive[np.asarray(idx)] = _violation_counts(sub["t_start_ms"].to_numpy(), sub["t_end_ms"].to_numpy(),
                                                   t[1:], flag)
    w["naive_java_violations"] = naive
    return w


# ------------------------------------------------------------------ thresholds
def calibrate(cal_scores: np.ndarray, target: float) -> float:
    """Smallest threshold whose FPR on the calibration humans is <= target (flag: score > thr)."""
    s = np.sort(np.asarray(cal_scores, dtype=float))
    k = int(np.floor(target * len(s)))          # number of calibration humans allowed above thr
    return float(s[len(s) - 1 - k]) if len(s) else float("inf")


def run_cv(win: pd.DataFrame, models: list[str], train_levels: list[str], eval_levels: list[str],
           group_col: str = "participant", n_splits: int = 5, seed: int = C.SEED,
           extra_train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-fold scores and per-fold thresholds for every model.

    win must contain humans (level == 'human') and attackers of every level in train/eval levels.
    """
    win = win.reset_index(drop=True)
    y_all = is_attack(win)
    groups = win[group_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for fold, (tr, te) in enumerate(sgkf.split(win, y_all, groups)):
        trw, tew = win.iloc[tr], win.iloc[te]
        trw = trw[(trw["level"] == "human") | trw["level"].isin(train_levels)]
        tew = tew[(tew["level"] == "human") | tew["level"].isin(eval_levels)]
        hum = trw[trw["level"] == "human"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + fold)
        fit_i, cal_i = next(gss.split(hum, groups=hum[group_col]))
        cal = hum.iloc[cal_i]
        fit = pd.concat([hum.iloc[fit_i], trw[trw["level"] != "human"]] +
                        ([extra_train] if extra_train is not None else []))
        for m in models:
            t0 = time.time()
            det = make_detector(m, seed + fold).fit(fit)
            cs, ts = det.score(cal), det.score(tew)
            thr = {tg: (0.0 if m in FIXED_RULES else calibrate(cs, tg)) for tg in TARGETS}
            o = tew[["session_id", group_col, "label", "level", "origin"]].copy()
            if group_col != "participant":
                o["participant"] = tew["participant"]
            o["model"], o["fold"], o["score"], o["y"] = m, fold, ts, is_attack(tew)
            o["n_cal"] = len(cal)
            for tg in TARGETS:
                o[f"thr_{tg}"] = thr[tg]
                o[f"flag_{tg}"] = (ts > thr[tg]).astype(int)
            out.append(o)
            log.info("fold %d %-20s fit+score %.1fs (train=%d cal=%d test=%d)", fold, m,
                     time.time() - t0, len(fit), len(cal), len(tew))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ metrics + bootstrap
def _rate_ci(per: pd.DataFrame, rng, n_boot=N_BOOT):
    """per: rows = participants with columns n, k. Bootstrap ratio sum(k)/sum(n)."""
    if per.empty or per["n"].sum() == 0:
        return float("nan"), float("nan"), float("nan")
    n, k = per["n"].to_numpy(float), per["k"].to_numpy(float)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    reps = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(k.sum() / n.sum()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def summarize(oof: pd.DataFrame, rng=None, levels: list[str] | None = None,
              group_col: str = "participant") -> pd.DataFrame:
    rng = rng or np.random.default_rng(C.SEED)
    rows = []
    for m, g in oof.groupby("model", sort=False):
        hum = g[g["y"] == 0]
        r = {"model": m, "n_human_windows": len(hum), "n_attack_windows": int(g["y"].sum()),
             "n_human_groups": hum[group_col].nunique(), "n_cal_humans_mean": float(g["n_cal"].mean())}
        for tg in TARGETS:
            ph = hum.groupby(group_col)[f"flag_{tg}"].agg(n="size", k="sum")
            r[f"fpr@{tg}"], r[f"fpr@{tg}_lo"], r[f"fpr@{tg}_hi"] = _rate_ci(ph, rng)
            lv = levels or sorted(set(g.loc[g["y"] == 1, "level"]))
            for L in [None] + list(lv):
                a = g[(g["y"] == 1) & ((g["level"] == L) if L else True)]
                pa = a.groupby(group_col)[f"flag_{tg}"].agg(n="size", k="sum")
                key = f"recall@{tg}" + (f"[{L}]" if L else "")
                r[key], r[key + "_lo"], r[key + "_hi"] = _rate_ci(pa, rng)
        # threshold-free and 1%-threshold metrics
        if g["y"].nunique() == 2:
            r["pr_auc"] = float(average_precision_score(g["y"], g["score"]))
            parts = g[group_col].unique()
            reps = []
            for _ in range(200):
                pick = rng.choice(parts, len(parts), replace=True)
                gg = pd.concat([g[g[group_col] == p] for p in pick]) if len(parts) < 60 else \
                    g.set_index(group_col).loc[pick].reset_index()
                if gg["y"].nunique() == 2:
                    reps.append(average_precision_score(gg["y"], gg["score"]))
            r["pr_auc_lo"], r["pr_auc_hi"] = (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))) if reps else (np.nan, np.nan)
        f = g["flag_0.01"]
        tp = int(((f == 1) & (g["y"] == 1)).sum()); fp = int(((f == 1) & (g["y"] == 0)).sum())
        fn = int(((f == 0) & (g["y"] == 1)).sum()); tn = int(((f == 0) & (g["y"] == 0)).sum())
        r.update({"tp@0.01": tp, "fp@0.01": fp, "fn@0.01": fn, "tn@0.01": tn,
                  "f1@0.01": 2 * tp / max(2 * tp + fp + fn, 1)})
        # session-level alarm: >= 3 flagged windows in a session (VL-like accumulation)
        s = g.groupby("session_id").agg(y=("y", "first"), k=("flag_0.01", "sum"))
        r["session_alarm_human"] = float((s.loc[s.y == 0, "k"] >= 3).mean()) if (s.y == 0).any() else np.nan
        r["session_alarm_attack"] = float((s.loc[s.y == 1, "k"] >= 3).mean()) if (s.y == 1).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str], fmt="{:.3f}") -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (float, np.floating)):
                lo, hi = r.get(c + "_lo"), r.get(c + "_hi")
                cells.append(fmt.format(v) + (f" [{fmt.format(lo)}, {fmt.format(hi)}]"
                                              if lo is not None and not pd.isna(lo) else ""))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def plot_recall(summary: pd.DataFrame, path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(summary))
    for j, tg in enumerate(TARGETS):
        v = summary[f"recall@{tg}"].to_numpy()
        err = np.vstack([v - summary[f"recall@{tg}_lo"], summary[f"recall@{tg}_hi"] - v])
        ax.bar(x + (j - 0.5) * 0.38, v, 0.38, yerr=err, capsize=3,
               label=f"threshold at FPR {tg:.1%} (calibration humans)")
    ax.set_xticks(x, summary["model"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("recall (attack windows)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr(oof: pd.DataFrame, path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for m, g in oof.groupby("model", sort=False):
        if m in FIXED_RULES or g["y"].nunique() < 2:
            continue
        p, r, _ = precision_recall_curve(g["y"], g["score"])
        ax.plot(r, p, label=f"{m} (AP={average_precision_score(g['y'], g['score']):.3f})", lw=1.2)
    ax.set(xlabel="recall", ylabel="precision", title=title)
    ax.title.set_fontsize(8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ data access
def synthetic_windows(window_type: str, levels=None) -> pd.DataFrame:
    streams = pd.read_csv(C.PROCESSED_DIR / "synthetic_click_streams.csv")
    clicks = pd.read_csv(C.PROCESSED_DIR / "synthetic_clicks.csv")
    if levels is not None:
        streams = streams[streams["level"].isin(["human"] + list(levels))]
    return prepare_windows(streams, clicks, window_type)


def real_windows(window_type: str) -> pd.DataFrame:
    from .data import trim_moves
    real = load_raw()
    if real.clicks.empty:
        return pd.DataFrame()
    streams = build_click_streams(real)
    return prepare_windows(streams, real.clicks, window_type,
                           moves=trim_moves(real.moves) if len(real.moves) else None)


def fit_final(win: pd.DataFrame, models: list[str], train_levels: list[str], seed: int = C.SEED):
    """Detectors trained on ALL synthetic participants; thresholds from 25% held-out humans."""
    sub = win[(win["level"] == "human") | win["level"].isin(train_levels)]
    hum = sub[sub["level"] == "human"]
    fit_i, cal_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                        .split(hum, groups=hum["participant"]))
    cal = hum.iloc[cal_i]
    fit = pd.concat([hum.iloc[fit_i], sub[sub["level"] != "human"]])
    final = {}
    for m in models:
        det = make_detector(m, seed).fit(fit)
        cs = det.score(cal)
        final[m] = (det, {tg: (0.0 if m in FIXED_RULES else calibrate(cs, tg)) for tg in TARGETS},
                    len(cal))
    return final


def real_track(models: list[str], final: dict) -> str:
    """Evaluation on REAL windows. Returns a markdown section."""
    lines = ["## REAL data", ""]
    rw = real_windows("count")
    rt = real_windows("time")
    for name, w in (("count (50 clicks)", rw), ("time (5 s)", rt)):
        n_h = int((is_attack(w) == 0).sum()) if len(w) else 0
        n_a = int(is_attack(w).sum()) if len(w) else 0
        lines.append(f"- scored real windows, {name}: humans={n_h}, attacks={n_a}")
    lines.append("")
    if rw.empty or len(rw) == 0:
        lines += ["No real window survives cleaning and routing (too few non-held clicks).",
                  "**No real-data detection metric can be computed yet.** The per-event behaviour of the",
                  "original thresholds on the real session is in `data_report.md`.", ""]
        return "\n".join(lines)
    y = is_attack(rw)
    grp_ok = {c: rw.loc[y == k, c].nunique() for c in ("participant", "session_id") for k in (0, 1)}
    if (y == 1).any() and (y == 0).any():
        gc = "participant" if min(rw.loc[y == 0, "participant"].nunique(), rw.loc[y == 1, "participant"].nunique()) >= 2 else "session_id"
        if min(rw.loc[y == 0, gc].nunique(), rw.loc[y == 1, gc].nunique()) >= 2:
            if gc == "session_id":
                lines.append("LIMITATION: fewer than 2 participants per class -> grouping by SESSION; "
                             "results do not measure generalisation to new players.")
            lvls = sorted(set(rw.loc[y == 1, "level"]))
            oof = run_cv(rw, [m for m in models if m != "cnn1d"], lvls, lvls, group_col=gc,
                         n_splits=min(5, rw.loc[y == 0, gc].nunique(), rw.loc[y == 1, gc].nunique()))
            s = summarize(oof, group_col=gc)
            s.to_csv(C.TAB_DIR / "real_cv_metrics.csv", index=False)
            lines += [md_table(s, ["model", "n_human_windows", "n_attack_windows", "recall@0.01",
                                   "fpr@0.01", "recall@0.001", "fpr@0.001", "pr_auc"])]
    # synthetic-trained detectors applied to real humans
    hum = rw[y == 0]
    rows = []
    for m, (det, thr, ncal) in final.items():
        if m == "cnn1d" or hum.empty:
            continue
        sc = det.score(hum)
        rows.append({"model": m, "real_human_windows": len(hum),
                     **{f"flagged@{tg}": int((sc > thr[tg]).sum()) for tg in TARGETS}})
    if rows:
        lines += ["Synthetic-trained detectors (thresholds from synthetic calibration humans) applied to",
                  "REAL human windows:", "", md_table(pd.DataFrame(rows), list(rows[0].keys()))]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    models = available_models()
    report = ["# Phase D - detection results", "",
              f"**Synthetic sections: {C.DEMO_TAG}.** Humans are placeholder-generator humans "
              "(see README); attack levels: none (auto_fixed), uniform and gaussian jitter (auto_jitter).", ""]
    final = None
    for wt in ("count", "time"):
        win = synthetic_windows(wt, STANDARD_ATTACKS)
        ms = models if wt == "count" else [m for m in models if m not in ("ocsvm", "cnn1d")]
        oof = run_cv(win, ms, STANDARD_ATTACKS, STANDARD_ATTACKS)
        oof.to_csv(C.PROCESSED_DIR / f"oof_synthetic_{wt}.csv", index=False)
        s = summarize(oof, levels=STANDARD_ATTACKS)
        s.insert(0, "window_type", wt)
        s.insert(0, "data", "synthetic (DEMO)")
        s.to_csv(C.TAB_DIR / f"cv_synthetic_{wt}_metrics.csv", index=False)
        report += [f"## Synthetic, {wt} windows (StratifiedGroupKFold by participant, 5 folds)", "",
                   f"Windows: humans={s['n_human_windows'].iloc[0]}, attacks={s['n_attack_windows'].iloc[0]}; "
                   f"calibration humans per fold ~{s['n_cal_humans_mean'].iloc[0]:.0f} windows "
                   "(an FPR target of 0.1% is close to the resolution limit of this calibration set).", "",
                   md_table(s, ["model", "recall@0.001", "fpr@0.001", "recall@0.01", "fpr@0.01", "pr_auc",
                                "f1@0.01"]),
                   "Recall per attack level at the 1% threshold:", "",
                   md_table(s, ["model"] + [f"recall@0.01[{l}]" for l in STANDARD_ATTACKS]),
                   "Session-level alarm (>= 3 flagged windows in a session, 1% threshold) and confusion counts:", "",
                   md_table(s, ["model", "session_alarm_human", "session_alarm_attack", "tp@0.01", "fp@0.01",
                                "fn@0.01", "tn@0.01"]), ""]
        plot_recall(s, C.FIG_DIR / f"recall_at_fpr_synthetic_{wt}.png",
                    f"Recall at fixed FPR, {wt} windows, 95% participant-bootstrap CI\n{C.DEMO_TAG}")
        plot_pr(oof, C.FIG_DIR / f"pr_curves_synthetic_{wt}.png", f"PR curves, {wt} windows\n{C.DEMO_TAG}")
        if wt == "count":
            final = fit_final(win, [m for m in models if m != "cnn1d"], STANDARD_ATTACKS)
            import joblib
            joblib.dump({m: (d, t, n) for m, (d, t, n) in final.items()},
                        C.PROCESSED_DIR / "final_models_standard.joblib")
            json.dump({m: {str(k): v for k, v in t.items()} for m, (_, t, _) in final.items()},
                      open(C.TAB_DIR / "final_thresholds_standard.json", "w"), indent=2)
    report.append(real_track(models, final))
    (C.RESULTS_DIR / "phase_d_results.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
