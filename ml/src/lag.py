"""Phase F - robustness to network lag (SIMULATED; no tc netem / clumsy run here).

Lag is simulated on top of the recorded/observed server timestamps:
  * per-packet extra jitter  ~ Normal(0, sigma) with sigma = JITTER_FRAC * RTT
  * lag spikes (Poisson, rate per second) during which delivery freezes and queued packets are
    released together (bunching), duration ~ U(0.5, 1.5) * RTT
  * TCP order preserved
The mapping from ping class to (sigma, spike rate) is an ASSUMPTION (table PING_CLASSES), not a
measurement. docs/real_lag_test_protocol.md describes how to replace it with real measurements.

Evaluated:
  DEMO   fresh synthetic humans (new seed, never seen in training) and standard attackers, scored by the
         final Phase-D detectors with their clean-data thresholds -> FPR / recall per ping class.
  REAL   the real sessions' timestamps with simulated lag (N_REPEATS draws) -> per-event flags of
         the original thesis rule (naive and cleaned), because no real window is scoreable yet.

    python -m src.lag
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import Dataset, build_click_streams, load_raw, naive_intervals
from .evaluate import TARGETS, md_table, prepare_windows
from .observation import enforce_order, lag_spikes
from .synthetic import build, calibration_from_real

log = logging.getLogger(__name__)
JITTER_FRAC = 0.10
PING_CLASSES = {  # ping (RTT, ms) -> (extra jitter sd ms, spikes per second)
    0: (0.0, 0.0),
    20: (JITTER_FRAC * 20, 0.01),
    80: (JITTER_FRAC * 80, 0.03),
    150: (JITTER_FRAC * 150, 0.06),
    250: (JITTER_FRAC * 250, 0.12),
}
N_REPEATS = 20


def lag_session(t_ms: np.ndarray, rng, ping: int) -> np.ndarray:
    sigma, rate = PING_CLASSES[ping]
    if ping == 0:
        return t_ms
    order = np.argsort(t_ms, kind="stable")
    s = t_ms[order] + rng.normal(0, sigma, len(t_ms))
    s = enforce_order(s)
    s = lag_spikes(s, rng, rate, (0.5 * ping, 1.5 * ping))
    out = np.empty_like(s)
    out[order] = s
    return out


def apply_lag(ds: Dataset, rng, ping: int) -> Dataset:
    """Re-time every click and break event of every session (one shared queue per session)."""
    cl, br = ds.clicks.copy(), ds.breaks.copy()
    cl["_k"], br["_k"] = "c", "b"
    allev = pd.concat([cl[["session_id", "t_ns", "_k"]].assign(_i=cl.index),
                       br[["session_id", "t_ns", "_k"]].assign(_i=br.index)], ignore_index=True)
    new = np.empty(len(allev))
    for _, idx in allev.groupby("session_id").groups.items():
        idx = np.asarray(idx)
        new[idx] = lag_session(allev.loc[idx, "t_ns"].to_numpy() / 1e6, rng, ping)
    allev["t_new"] = (new * 1e6).astype("int64")
    m = allev.set_index(["_k", "_i"])["t_new"]
    cl["t_ns"] = m.loc["c"].reindex(cl.index).to_numpy()
    if len(br):
        br["t_ns"] = m.loc["b"].reindex(br.index).to_numpy()
    cl = cl.drop(columns="_k").sort_values(["session_id", "t_ns"], kind="stable")
    br = br.drop(columns="_k")
    return Dataset(cl, ds.moves, br, ds.sessions, ds.issues)


def demo_lag(final: dict, rng) -> pd.DataFrame:
    net, hyper, _ = calibration_from_real()
    ds, _ = build(seed=C.SEED + 777, n_humans=20, n_attackers=8, levels=["none", "uniform", "gaussian"],
                  net=net, hyper=hyper)
    rows = []
    for ping in PING_CLASSES:
        d = apply_lag(ds, rng, ping)
        w = prepare_windows(build_click_streams(d), d.clicks, "count")
        hum = w["level"] == "human"
        for m, (det, thr, _) in final.items():
            sc = det.score(w)
            for tg in TARGETS:
                f = sc > thr[tg]
                rows.append({"data": "synthetic (DEMO)", "ping_ms": ping, "model": m, "target_fpr": tg,
                             "human_windows": int(hum.sum()), "fpr": float(f[hum].mean()),
                             "attack_windows": int((~hum).sum()), "recall": float(f[~hum].mean())})
        log.info("ping %d done", ping)
    return pd.DataFrame(rows)


def real_lag(rng) -> pd.DataFrame:
    real = load_raw()
    if real.clicks.empty:
        return pd.DataFrame()
    rows = []
    thr = C.ORIG_CLICK_MIN_DIFF_MS
    for ping in PING_CLASSES:
        for rep in range(N_REPEATS if ping else 1):
            d = apply_lag(real, rng, ping)
            st = build_click_streams(d)
            nv = naive_intervals(d.clicks, {"interact_air", "interact_block"})["iv_ms"].to_numpy()
            act = np.floor(st.loc[~st["held"].astype(bool), "iv_ms"].dropna().to_numpy())
            rows.append({"data": "REAL timestamps + SIMULATED lag", "ping_ms": ping, "repeat": rep,
                         "naive_java_flags": int(np.sum((nv > 0) & (nv < thr))), "naive_java_n": len(nv),
                         "cleaned_active_flags": int(np.sum((act > 0) & (act < thr))),
                         "cleaned_active_n": len(act), "held_swings": int(st["held"].sum())})
    return pd.DataFrame(rows)


def plot(demo: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = demo[demo["target_fpr"] == 0.01]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m, g in d.groupby("model", sort=False):
        ls = "--" if m.startswith("baseline") else "-"
        ax[0].plot(g["ping_ms"], g["fpr"], marker="o", ls=ls, label=m)
        ax[1].plot(g["ping_ms"], g["recall"], marker="o", ls=ls, label=m)
    ax[0].axhline(0.01, color="grey", lw=0.8, ls=":")
    ax[0].set(xlabel="simulated ping class (ms)", ylabel="human FPR (window)", yscale="symlog",
              title="human FPR, threshold calibrated at 1% on clean data")
    ax[1].set(xlabel="simulated ping class (ms)", ylabel="recall (standard attackers)", ylim=(-0.02, 1.02),
              title="recall under the same lag")
    for a in ax:
        a.title.set_fontsize(8); a.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.suptitle(f"SIMULATED lag - {C.DEMO_TAG}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(C.SEED + 2)
    final = joblib.load(C.PROCESSED_DIR / "final_models_standard.joblib")
    demo = demo_lag(final, rng)
    demo.to_csv(C.TAB_DIR / "lag_synthetic.csv", index=False)
    plot(demo, C.FIG_DIR / "lag_fpr_synthetic.png")
    real = real_lag(rng)
    rep = [f"# Phase F - lag robustness (SIMULATED lag)", "",
           "Ping class -> (extra per-packet jitter sd, lag spikes per second) is an assumption: " +
           ", ".join(f"{p} ms -> ({s:.0f} ms, {r}/s)" for p, (s, r) in PING_CLASSES.items()) +
           "; spike duration U(0.5, 1.5) x ping, packets released together after a spike.", "",
           f"## Synthetic ({C.DEMO_TAG}) - fresh human population never used for training", "",
           md_table(demo[demo["target_fpr"] == 0.01].pivot(index="model", columns="ping_ms", values="fpr")
                    .reset_index().rename(columns=lambda c: f"fpr@ping{c}" if c != "model" else c),
                    ["model"] + [f"fpr@ping{p}" for p in PING_CLASSES], fmt="{:.4f}"),
           "Recall on standard attackers under the same lag (1% threshold):", "",
           md_table(demo[demo["target_fpr"] == 0.01].pivot(index="model", columns="ping_ms", values="recall")
                    .reset_index().rename(columns=lambda c: f"recall@ping{c}" if c != "model" else c),
                    ["model"] + [f"recall@ping{p}" for p in PING_CLASSES]), ""]
    if len(real):
        real.to_csv(C.TAB_DIR / "lag_real_events.csv", index=False)
        agg = real.groupby("ping_ms").agg(repeats=("repeat", "size"),
                                         naive_java_flags_mean=("naive_java_flags", "mean"),
                                         naive_java_n=("naive_java_n", "mean"),
                                         cleaned_active_flags_mean=("cleaned_active_flags", "mean"),
                                         cleaned_active_n=("cleaned_active_n", "mean"),
                                         held_swings_mean=("held_swings", "mean")).reset_index()
        agg.to_csv(C.TAB_DIR / "lag_real_events_summary.csv", index=False)
        rep += ["## REAL session timestamps + SIMULATED lag - original rule, per event", "",
                "Every flag is a false positive (the session is human). One short session: anecdotal.", "",
                md_table(agg, list(agg.columns), fmt="{:.2f}")]
    (C.RESULTS_DIR / "phase_f_results.md").write_text("\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
