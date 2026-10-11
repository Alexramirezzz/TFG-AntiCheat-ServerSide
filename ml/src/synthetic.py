"""Phase B orchestration: build the synthetic dataset (humans + standard adversary levels).

Every synthetic row carries origin = "synthetic"; files are written as
data/processed/synthetic_*.csv. The optimized adversary is generated later (adversarial.py)
because it needs a trained detector.

    python -m src.synthetic
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config as C
from .adversary import generate_attackers
from .data import Dataset, build_click_streams, load_raw
from .observation import NetParams, calibrate_jitter
from .synth_humans import PLACEHOLDER_HYPER, fit_hyper, generate_humans

log = logging.getLogger(__name__)

N_HUMANS = 40
N_ATTACKERS = 24            # per adversary level
STANDARD_LEVELS = ["none", "uniform", "gaussian", "human_like"]


def calibration_from_real() -> tuple[NetParams, dict, dict]:
    """Jitter and human hyper-parameters from real data when possible (else placeholders)."""
    real = load_raw()
    streams = build_click_streams(real) if len(real.clicks) else pd.DataFrame(
        columns=["iv_ms", "held", "label", "participant"])
    net = calibrate_jitter(streams["iv_ms"].dropna().to_numpy()) if len(streams) else NetParams()
    hyper, info = fit_hyper(streams) if len(streams) else (None, {"active_intervals": 0})
    info["mode"] = "fitted" if hyper else "placeholder"
    info["jitter_sigma_ms"] = net.sigma_ms
    info["jitter_source"] = net.source
    return net, (hyper or PLACEHOLDER_HYPER), info


def concat_ds(parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> Dataset:
    cl = pd.concat([p[0] for p in parts], ignore_index=True)
    nonempty = [p[1] for p in parts if len(p[1])]
    br = pd.concat(nonempty, ignore_index=True) if nonempty else         pd.DataFrame({"session_id": pd.Series(dtype=str), "t_ns": pd.Series(dtype="int64"),
                      "label": pd.Series(dtype=str)})
    ss = pd.concat([p[2] for p in parts], ignore_index=True)
    ss["origin"] = "synthetic"
    for df in (cl, br):
        df["origin"] = "synthetic"
        df["participant"] = df["session_id"].map(ss.set_index("session_id")["participant"])
    return Dataset(cl, pd.DataFrame(), br, ss, [])


def build(seed: int = C.SEED, n_humans: int = N_HUMANS, n_attackers: int = N_ATTACKERS,
          levels=STANDARD_LEVELS, net: NetParams | None = None, hyper: dict | None = None):
    rng = np.random.default_rng(seed)
    if net is None or hyper is None:
        net, hyper, _ = calibration_from_real()
    parts, params = [], []
    (h, hp) = generate_humans(n_humans, rng, net, hyper)
    parts.append(h); params.append(hp.assign(level="human"))
    for lvl in levels:
        a, ap = generate_attackers(lvl, n_attackers, rng, net, hyper)
        parts.append(a); params.append(ap)
    ds = concat_ds(parts)
    return ds, pd.concat(params, ignore_index=True)


def plot_overview(streams: pd.DataFrame, real_streams: pd.DataFrame | None, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("human", "synthetic human (combat)")] + [(l, f"adversary: {l}") for l in STANDARD_LEVELS]
    fig, axes = plt.subplots(1, len(groups) + (1 if real_streams is not None else 0),
                             figsize=(3.0 * (len(groups) + 1), 3.0), sharey=False)
    bins = np.arange(0, 400, 5)
    for ax, (lvl, title) in zip(axes, groups):
        s = streams[(streams["level"] == lvl) & ~streams["held"].astype(bool)]
        if lvl == "human":
            s = s[s["label"] == "human_combat"]
        ax.hist(s["iv_ms"].dropna(), bins=bins, color="#888" if lvl == "human" else "#c0504d")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("interval (ms)")
    if real_streams is not None:
        r = real_streams[~real_streams["held"].astype(bool)]["iv_ms"].dropna()
        axes[-1].hist(r, bins=bins, color="#3b6ea5")
        axes[-1].set_title(f"REAL human, non-held (n={len(r)})", fontsize=9)
        axes[-1].set_xlabel("interval (ms)")
    fig.suptitle(f"Observed click intervals after tick quantization - SYNTHETIC panels are {C.DEMO_TAG}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    net, hyper, info = calibration_from_real()
    log.info("calibration: %s", {k: v for k, v in info.items() if k != "fitted_per_participant"})
    ds, params = build(net=net, hyper=hyper)
    streams = build_click_streams(ds)
    ds.clicks.to_csv(C.PROCESSED_DIR / "synthetic_clicks.csv", index=False)
    ds.breaks.to_csv(C.PROCESSED_DIR / "synthetic_breaks.csv", index=False)
    ds.sessions.to_csv(C.PROCESSED_DIR / "synthetic_sessions.csv", index=False)
    streams.to_csv(C.PROCESSED_DIR / "synthetic_click_streams.csv", index=False)
    params.to_csv(C.TAB_DIR / "synthetic_generator_params.csv", index=False)
    (C.TAB_DIR / "synthetic_calibration.json").write_text(json.dumps(
        {"human_generator": info["mode"], "hyper": hyper, "info": info}, indent=2, default=float))
    real = load_raw()
    real_streams = build_click_streams(real) if len(real.clicks) else None
    plot_overview(streams, real_streams, C.FIG_DIR / "synthetic_interval_overview.png")
    summ = streams.groupby(["level", "label"]).agg(
        sessions=("session_id", "nunique"), participants=("participant", "nunique"),
        accepted_swings=("t_ms", "size"), held=("held", "sum"), duplicates=("n_dup", "sum"),
        median_iv_ms=("iv_ms", "median")).reset_index()
    summ.insert(0, "origin", "synthetic")
    summ.to_csv(C.TAB_DIR / "synthetic_summary.csv", index=False)
    log.info("\n%s", summ.to_string(index=False))


if __name__ == "__main__":
    main()
