"""Phase E - adversarial experiment (DEMO: synthetic humans and synthetic adversaries).

Round 0  detectors trained on humans + auto_fixed (level "none") only; evaluated on every level.
         The "optimized" adversary is evolved against the round-0 XGBoost under a minimum click
         rate (MIN_CPS within bouts), i.e. it is adaptive to that specific detector.
Round 1  adversarial training: humans + none, uniform, gaussian, human_like, optimized.
         Re-evaluated on every level PLUS "optimized_r2": a fresh adversary evolved against the
         round-1 XGBoost (otherwise round 1 would be tested on the attack it was trained on).
Checks   human FPR round 0 vs round 1 (same calibration protocol), and an attacker utility
         frontier: recall at 1% FPR vs the minimum click rate the optimizer must keep.

    python -m src.adversarial
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config as C
from .adversary import generate_attackers, optimize_attack
from .data import build_click_streams
from .evaluate import (TARGETS, fit_final, md_table, prepare_windows, run_cv, summarize,
                       synthetic_windows)
from .synthetic import calibration_from_real, concat_ds

log = logging.getLogger(__name__)
MODELS = ["baseline_orig_raw", "baseline_orig_clean", "cv_rule", "iforest", "ocsvm", "rf", "xgb"]
LEVELS = ["none", "uniform", "gaussian", "human_like", "optimized"]
MIN_CPS = 12.0
FRONTIER_CPS = [8.0, 10.0, 12.0, 14.0, 16.0]
N_OPT_ATTACKERS = 24


def windows_from_parts(parts) -> pd.DataFrame:
    ds = concat_ds(parts)
    return prepare_windows(build_click_streams(ds), ds.clicks, "count")


def make_score_fn(det):
    def score_fn(clicks, breaks, sessions):
        sessions = sessions.copy(); sessions["origin"] = "synthetic"
        w = windows_from_parts([(clicks, breaks, sessions)])
        return float(np.mean(det.score(w))) if len(w) else 0.0
    return score_fn


def optimized_windows(det, min_cps: float, rng, net, hyper, level_name: str, n=N_OPT_ATTACKERS,
                      generations=12):
    params, hist = optimize_attack(make_score_fn(det), min_cps, rng, generations=generations, net=net)
    (parts, _) = generate_attackers("optimized", n, rng, net, hyper, opt=params,
                                    prefix=f"SA{level_name}")
    parts[2]["level"] = level_name
    w = windows_from_parts([parts])
    w["level"] = level_name
    return w, params, hist


def plot_main(summ: dict[str, pd.DataFrame], levels_by_round: dict[str, list[str]], path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(summ), figsize=(6.2 * len(summ), 4), sharey=True)
    for ax, (rname, s) in zip(np.atleast_1d(axes), summ.items()):
        lv = levels_by_round[rname]
        x = np.arange(len(lv))
        for _, r in s.iterrows():
            v = np.array([r.get(f"recall@0.01[{l}]", np.nan) for l in lv], float)
            lo = np.array([r.get(f"recall@0.01[{l}]_lo", np.nan) for l in lv], float)
            hi = np.array([r.get(f"recall@0.01[{l}]_hi", np.nan) for l in lv], float)
            fixed = r["model"].startswith("baseline")
            lab = (f"{r['model']} (fixed rule, human FPR {r['fpr@0.01']:.0%})" if fixed
                   else f"{r['model']} (human FPR {r['fpr@0.01']:.1%})")
            ax.errorbar(x, v, yerr=[v - lo, hi - v], marker="o", ms=4, capsize=3,
                        ls="--" if fixed else "-", lw=1.2, label=lab)
        ax.set_xticks(x, lv, rotation=20, fontsize=8)
        ax.set_title(rname, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("recall at 1% FPR (95% participant-bootstrap CI)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Recall vs adversary level - {C.DEMO_TAG}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_frontier(fr: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (rnd, m), g in fr.groupby(["round", "model"]):
        ax.plot(g["min_cps"], g["recall@0.01"], marker="o", label=f"{m} ({rnd})")
    ax.set(xlabel="minimum click rate imposed on the optimizer (CPS within bouts)",
           ylabel="recall at 1% FPR", ylim=(-0.02, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"Attacker utility vs detectability\n{C.DEMO_TAG}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(C.SEED + 1)
    net, hyper, _ = calibration_from_real()
    win = synthetic_windows("count", ["none", "uniform", "gaussian", "human_like"])

    # ---------------- round 0: train on auto_fixed only
    final0 = fit_final(win, ["xgb"], ["none"])
    w_opt, p_opt, h_opt = optimized_windows(final0["xgb"][0], MIN_CPS, rng, net, hyper, "optimized")
    win = pd.concat([win, w_opt], ignore_index=True)
    oof0 = run_cv(win, MODELS, ["none"], LEVELS)
    s0 = summarize(oof0, levels=LEVELS)

    # ---------------- round 1: adversarial training with the harder levels
    final1 = fit_final(win, ["xgb"], LEVELS)
    w_opt2, p_opt2, h_opt2 = optimized_windows(final1["xgb"][0], MIN_CPS, rng, net, hyper, "optimized_r2")
    win1 = pd.concat([win, w_opt2], ignore_index=True)
    levels1 = LEVELS + ["optimized_r2"]
    oof1 = run_cv(win1, MODELS, LEVELS, levels1)
    s1 = summarize(oof1, levels=levels1)

    # ---------------- utility frontier (xgb, both rounds)
    fr_rows = []
    for rnd, fin, train_lv in (("round0", final0, ["none"]), ("round1", final1, LEVELS)):
        det, thr, _ = fin["xgb"]
        for cps in FRONTIER_CPS:
            w, p, _ = optimized_windows(det, cps, rng, net, hyper, f"frontier_{cps:g}", n=12, generations=8)
            sc = det.score(w)
            fr_rows.append({"round": rnd, "model": "xgb", "min_cps": cps, "achieved_cps": p["opt_cps"],
                            "windows": len(w), **{f"recall@{tg}": float(np.mean(sc > thr[tg])) for tg in TARGETS}})
    fr = pd.DataFrame(fr_rows)

    # ---------------- outputs
    for name, s in (("round0", s0), ("round1", s1)):
        s.insert(0, "round", name)
        s.to_csv(C.TAB_DIR / f"adversarial_{name}_metrics.csv", index=False)
    fr.to_csv(C.TAB_DIR / "adversarial_frontier.csv", index=False)
    pd.concat([h_opt.assign(target="round0_xgb"), h_opt2.assign(target="round1_xgb")]).to_csv(
        C.TAB_DIR / "adversarial_optimizer_history.csv", index=False)
    json.dump({"optimized_vs_round0_xgb": p_opt, "optimized_r2_vs_round1_xgb": p_opt2},
              open(C.TAB_DIR / "adversarial_optimized_params.json", "w"), indent=2)
    plot_main({"Round 0: trained on humans + auto_fixed": s0,
               "Round 1: adversarial training (+ uniform, gaussian, human_like, optimized)": s1},
              {"Round 0: trained on humans + auto_fixed": LEVELS,
               "Round 1: adversarial training (+ uniform, gaussian, human_like, optimized)": levels1},
              C.FIG_DIR / "adversarial_recall_vs_level.png")
    plot_frontier(fr, C.FIG_DIR / "adversarial_utility_frontier.png")

    fpr = s0[["model", "fpr@0.01", "fpr@0.01_lo", "fpr@0.01_hi", "fpr@0.001"]].merge(
        s1[["model", "fpr@0.01", "fpr@0.01_lo", "fpr@0.01_hi", "fpr@0.001"]], on="model",
        suffixes=("_round0", "_round1"))
    fpr.to_csv(C.TAB_DIR / "adversarial_human_fpr.csv", index=False)
    rep = [f"# Phase E - adversarial experiment ({C.DEMO_TAG})", "",
           f"Optimized adversary: (mu, lambda) evolution strategy, minimum {MIN_CPS:g} CPS within bouts, "
           "objective = mean XGBoost score of its windows.", "",
           f"- vs round-0 XGBoost: median {p_opt['median_ms']:.1f} ms, spread {p_opt['spread']:.3f}, rho "
           f"{p_opt['rho']:.2f}, pause_prob {p_opt['pause_prob']:.3f}; achieved {p_opt['opt_cps']:.1f} CPS, "
           f"mean score {p_opt['opt_score']:.3f}",
           f"- vs round-1 XGBoost (optimized_r2): median {p_opt2['median_ms']:.1f} ms, spread "
           f"{p_opt2['spread']:.3f}, rho {p_opt2['rho']:.2f}, pause_prob {p_opt2['pause_prob']:.3f}; achieved "
           f"{p_opt2['opt_cps']:.1f} CPS, mean score {p_opt2['opt_score']:.3f}", "",
           "## Round 0 - recall at 1% FPR per level", "",
           md_table(s0, ["model"] + [f"recall@0.01[{l}]" for l in LEVELS]),
           "## Round 1 (adversarial training) - recall at 1% FPR per level", "",
           md_table(s1, ["model"] + [f"recall@0.01[{l}]" for l in levels1]),
           "## Human FPR before/after adversarial training (realised on test humans)", "",
           md_table(fpr, list(fpr.columns[:1]) + ["fpr@0.01_round0", "fpr@0.01_round1",
                                                  "fpr@0.001_round0", "fpr@0.001_round1"]),
           "## Utility frontier (XGBoost)", "", md_table(fr, list(fr.columns)), ""]
    (C.RESULTS_DIR / "phase_e_results.md").write_text("\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
