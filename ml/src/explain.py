"""Phase G - explainability of the final XGBoost click detector (DEMO model).

* TreeSHAP values (shap library; falls back to XGBoost's native pred_contribs, the same algorithm)
* permutation importance (drop in PR-AUC) on a FRESH synthetic hold-out population
* three example decisions rendered as audit-log lines: what the plugin would write to auditoria.log

    python -m src.explain
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from . import config as C
from .data import build_click_streams
from .evaluate import prepare_windows
from .features import FEATURES
from .synthetic import build, calibration_from_real
from .train import X_of, is_attack

log = logging.getLogger(__name__)

DESCRIPTIONS = {
    "mean": "mean click interval (ms)", "std": "std of intervals (ms)", "cv": "coefficient of variation",
    "skew": "skewness", "kurt": "excess kurtosis", "min": "shortest interval (ms)",
    "p05": "5th percentile interval (ms)", "p95": "95th percentile interval (ms)",
    "entropy": "interval histogram entropy (bits)", "ac1": "lag-1 autocorrelation",
    "max_cps_1s": "max clicks in any 1 s", "tick_frac": "fraction of intervals on the 50 ms grid",
    "dup_frac": "fraction of same-tick duplicate swings", "pause_frac": "fraction of pauses > 1 s",
}


def shap_values(model, X: np.ndarray) -> tuple[np.ndarray, float, str]:
    try:
        import shap
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        base = float(np.ravel(ex.expected_value)[0])
        return np.asarray(sv), base, f"shap {shap.__version__} TreeExplainer"
    except Exception as e:  # version mismatch between shap and xgboost
        import xgboost as xgb
        log.warning("shap failed (%s); using xgboost native TreeSHAP", e)
        contrib = model.get_booster().predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)
        return contrib[:, :-1], float(contrib[0, -1]), "xgboost pred_contribs (TreeSHAP)"


def audit_line(row: pd.Series, sv: np.ndarray, score: float, thr: float, human_median: pd.Series,
               k: int = 3) -> str:
    order = np.argsort(-np.abs(sv))[:k]
    reasons = []
    for i in order:
        f = FEATURES[i]
        reasons.append(f"{f}={row[f]:.3g} (typical human {human_median[f]:.3g}) "
                       f"{'raises' if sv[i] > 0 else 'lowers'} suspicion by {abs(sv[i]):.2f} log-odds")
    verdict = "FLAG (+1 VL)" if score > thr else "no action"
    return (f"[ML-AutoClicker] {verdict} | score={score:.3f} threshold={thr:.3f} (1% FPR on calibration "
            f"humans; DEMO model) | window={int(row['n_swings'])} swings, {int(row['n_active'])} active "
            f"intervals, held_frac={row['held_frac']:.2f} | reasons: " + "; ".join(reasons))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = joblib.load(C.PROCESSED_DIR / "final_models_standard.joblib")
    det, thr, _ = final["xgb"]
    model = det.m
    net, hyper, _ = calibration_from_real()
    ds, _ = build(seed=C.SEED + 999, n_humans=16, n_attackers=8, levels=["none", "uniform", "gaussian"],
                  net=net, hyper=hyper)
    w = prepare_windows(build_click_streams(ds), ds.clicks, "count")
    X, y = X_of(w), is_attack(w)
    score = det.score(w)

    sv, base, method = shap_values(model, X)
    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
    pi = permutation_importance(model, X, y, scoring=lambda m, X_, y_: average_precision_score(
        y_, m.predict_proba(X_)[:, 1]), n_repeats=10, random_state=C.SEED, n_jobs=1)
    imp["perm_importance_pr_auc"] = pi.importances_mean
    imp["perm_importance_sd"] = pi.importances_std
    imp["description"] = imp["feature"].map(DESCRIPTIONS)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(C.TAB_DIR / "explain_feature_importance.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#3b6ea5")
    ax[0].set(xlabel="mean |SHAP| (log-odds)", title=f"XGBoost, {method}")
    ax[1].barh(imp["feature"][::-1], imp["perm_importance_pr_auc"][::-1],
               xerr=imp["perm_importance_sd"][::-1], color="#c0504d")
    ax[1].set(xlabel="drop in PR-AUC when permuted", title="permutation importance (fresh hold-out)")
    for a in ax:
        a.title.set_fontsize(8)
    fig.suptitle(f"Feature importance - {C.DEMO_TAG}", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "explain_feature_importance.png", dpi=130)
    plt.close(fig)

    t1 = thr[0.01]
    human_median = w.loc[y == 0, FEATURES].median()
    rng = np.random.default_rng(C.SEED)
    picks = {
        "attack window flagged with high confidence": np.where((y == 1) & (score > t1))[0],
        "attack window missed (below threshold)": np.where((y == 1) & (score <= t1))[0],
        "human window close to the threshold": np.argsort(np.where(y == 0, np.abs(score - t1), np.inf))[:20],
    }
    lines = [f"# Phase G - explainability ({C.DEMO_TAG})", "",
             f"Model: final XGBoost (trained on synthetic humans + none/uniform/gaussian). Explanation "
             f"method: {method}; base value {base:.3f} log-odds. Evaluated on a fresh synthetic population "
             f"({int((y == 0).sum())} human / {int(y.sum())} attack windows).", "",
             "## Feature ranking", "",
             "| feature | description | mean abs SHAP | permutation importance (PR-AUC drop) |",
             "|---|---|---|---|"]
    lines += [f"| {r.feature} | {r.description} | {r.mean_abs_shap:.3f} | {r.perm_importance_pr_auc:.4f} "
              f"± {r.perm_importance_sd:.4f} |" for r in imp.itertuples()]
    lines += ["", "## Three example decisions as audit-log lines", "",
              "Each line is what `auditoria.log` would contain (actor pseudonymised by HMAC in the plugin).", ""]
    for title, idx in picks.items():
        if len(idx) == 0:
            lines += [f"**{title}:** no such window in this hold-out.", ""]
            continue
        i = int(rng.choice(idx))
        lines += [f"**{title}** (true level: {w.iloc[i]['level']})", "", "```",
                  audit_line(w.iloc[i], sv[i], float(score[i]), t1, human_median), "```", ""]
    (C.RESULTS_DIR / "phase_g_results.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("done (%s)", method)


if __name__ == "__main__":
    main()
