"""Phase H - export the best tree-based model to ONNX for the plugin, plus Java parity fixtures.

The best tree model is chosen by recall at 1% FPR (count windows, Phase D CV table). The exported
model is the final model of evaluate.fit_final (trained on ALL synthetic participants) and its
1%-FPR threshold. It is a DEMO model: the plugin ships it in shadow mode by default.

Outputs
  results/models/click_model.onnx + click_model.json   (copied into plugin resources)
  plugin/src/test/resources/click_features_fixture.json (Python features + scores for Java tests)

    python -m src.export_onnx
"""
from __future__ import annotations

import json
import logging
import shutil

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from . import config as C
from .data import clean_click_stream
from .evaluate import TARGETS
from .features import FEATURES, window_features

log = logging.getLogger(__name__)


def to_onnx(name: str, model):
    from onnxmltools.convert.common.data_types import FloatTensorType
    init = [("features", FloatTensorType([None, len(FEATURES)]))]
    if name == "xgb":
        from onnxmltools import convert_xgboost
        onx = convert_xgboost(model, initial_types=init, target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SFT
        onx = convert_sklearn(model, initial_types=[("features", SFT([None, len(FEATURES)]))],
                              options={id(model): {"zipmap": False}}, target_opset=15)
    return onx


def onnx_scores(path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})
    prob = [o for o in outs if getattr(o, "ndim", 0) == 2][0]
    return prob[:, 1]


def fixture_cases(rng) -> list[dict]:
    """Hand-made raw event sequences covering dedup, held runs, pauses and same-tick duplicates."""
    cases = []
    def add(name, ev, breaks=()):
        cl = pd.DataFrame({"t_ns": [int(t * 1e6) for t, _ in ev], "source": [s for _, s in ev]})
        br = pd.DataFrame({"t_ns": [int(t * 1e6) for t in breaks]})
        st = clean_click_stream(cl, br)
        f = window_features(st["t_ms"].to_numpy(), st["held"].to_numpy(), st["n_dup"].to_numpy())
        cases.append({"name": name, "events": [[float(t), s] for t, s in ev],
                      "breaks": [float(b) for b in breaks],
                      "accepted_t_ms": st["t_ms"].tolist(), "held": st["held"].astype(bool).tolist(),
                      "n_dup": st["n_dup"].astype(int).tolist(),
                      "features": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                                   for k, v in f.items() if k in FEATURES + ["held_frac", "n_active", "valid", "n_swings"]}})
    # 1) fixed 12 CPS clicker observed on the tick grid with jitter
    t = np.cumsum(np.tile([100.0, 50.0, 100.0, 83.0], 13)) + rng.normal(0, 2, 52)
    add("autoclicker_like", [(x, "swing") for x in np.sort(t)])
    # 2) human-like with pauses and interact_air duplicates
    t = np.cumsum(rng.lognormal(np.log(140), 0.4, 60)); t[30:] += 1500
    ev = []
    for x in t:
        ev += [(x - 0.3, "interact_air"), (x, "swing")]
    add("human_like_with_pause", ev)
    # 3) mining runs (instant break then held swings) mixed with clicks
    ev, br = [], []
    base = 0.0
    for k in range(6):
        ev.append((base, "interact_block")); br.append(base + 0.2)
        ev += [(base + 0.6 + 50.0 * j + rng.normal(0, 1.5), "swing") for j in range(6)]
        base += 900
        ev += [(base - 400 + 130 * j, "swing") for j in range(3)]
    add("mining_mixed", sorted(ev), br)
    # 4) same-tick double clicks (duplicates within 15 ms)
    t = np.cumsum(np.tile([50.0, 50.0, 100.0], 20))
    ev = []
    for x in t:
        ev += [(x, "swing"), (x + 0.2, "swing")]
    add("same_tick_doubles", ev)
    return cases


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cv = pd.read_csv(C.TAB_DIR / "cv_synthetic_count_metrics.csv")
    trees = cv[cv["model"].isin(["rf", "xgb"])].sort_values(["recall@0.01", "pr_auc"], ascending=False)
    final = joblib.load(C.PROCESSED_DIR / "final_models_standard.joblib")
    # Selection rule: highest recall at 1% FPR; if the runner-up is within 1 percentage point, take
    # the model with the smaller ONNX graph (server impact / TPS is a first-class requirement).
    sizes = {m: len(to_onnx(m, final[m][0].m).SerializeToString()) for m in trees["model"]}
    best = trees["model"].iloc[0]
    if len(trees) > 1 and trees["recall@0.01"].iloc[0] - trees["recall@0.01"].iloc[1] < 0.01:
        best = min(trees["model"].iloc[:2], key=sizes.get)
    log.info("recall@1%%: %s; onnx bytes: %s -> selected %s",
             dict(zip(trees["model"], trees["recall@0.01"].round(4))), sizes, best)
    det, thr, ncal = final[best]
    onx = to_onnx(best, det.m)
    out = C.MODEL_DIR / "click_model.onnx"
    out.write_bytes(onx.SerializeToString())

    # parity python vs onnxruntime on real-shaped windows
    from .evaluate import synthetic_windows
    w = synthetic_windows("count", ["none", "uniform", "gaussian"]).sample(2000, random_state=C.SEED)
    X = w[FEATURES].to_numpy(np.float32)
    py, ox = det.score(w), onnx_scores(out, X)
    maxdiff = float(np.max(np.abs(py - ox)))
    log.info("python vs onnx max |diff| = %.2e", maxdiff)
    if maxdiff > 1e-4:
        raise SystemExit(f"ONNX parity failed: {maxdiff}")

    row = cv[cv["model"] == best].iloc[0]
    meta = {
        "model": best, "features": FEATURES, "threshold_fpr_1pct": thr[0.01],
        "threshold_fpr_0.1pct": thr[0.001], "calibration_human_windows": int(ncal),
        "window_clicks": C.WINDOW_CLICKS, "stride_clicks": C.STRIDE_CLICKS,
        "min_intervals": C.MIN_INTERVALS_PER_WINDOW, "dedup_ms": C.DEDUP_MS,
        "held_max_iv_ms": C.HELD_MAX_IV_MS, "fresh_dig_ms": C.FRESH_DIG_MS, "pause_ms": C.PAUSE_MS,
        "held_route_frac": C.HELD_ROUTE_FRAC, "tick_ms": C.TICK_MS, "tick_near_ms": C.TICK_NEAR_MS,
        "trained_on": "SYNTHETIC humans (placeholder generator) + synthetic none/uniform/gaussian clickers",
        "demo": True, "warning": C.DEMO_TAG,
        "cv_recall_at_1pct_fpr_synthetic": float(row["recall@0.01"]),
        "cv_fpr_at_1pct_synthetic": float(row["fpr@0.01"]),
        "onnx_parity_max_abs_diff": maxdiff,
        "selection": {"recall_at_1pct": dict(zip(trees["model"], trees["recall@0.01"].astype(float))),
                      "onnx_bytes": sizes,
                      "rule": "best recall@1%FPR; within 1 point -> smaller ONNX model"},
    }
    (C.MODEL_DIR / "click_model.json").write_text(json.dumps(meta, indent=2))
    res = C.PLUGIN_RES_DIR / "models"
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, res / "click_model.onnx")
    shutil.copy(C.MODEL_DIR / "click_model.json", res / "click_model.json")

    # Java parity fixtures: features of hand-made cases + model scores of sample windows
    rng = np.random.default_rng(C.SEED)
    # 25 windows spread over the whole score range (not only saturated 0/1 scores)
    order = np.argsort(py)
    sample = w.iloc[order[np.linspace(0, len(order) - 1, 25).astype(int)]]
    fx = {"constants": {k: meta[k] for k in ("dedup_ms", "held_max_iv_ms", "fresh_dig_ms", "pause_ms",
                                              "tick_ms", "tick_near_ms", "min_intervals")},
          "features": FEATURES, "cases": fixture_cases(rng),
          "model_samples": [{"x": [float(v) for v in r], "score": float(s)}
                            for r, s in zip(sample[FEATURES].to_numpy(np.float32), det.score(sample))]}
    tdir = C.REPO_DIR / "plugin" / "src" / "test" / "resources"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "click_features_fixture.json").write_text(json.dumps(fx, indent=1))
    log.info("exported %s (%d bytes), threshold@1%%=%.4f", out.name, out.stat().st_size, thr[0.01])


if __name__ == "__main__":
    main()
