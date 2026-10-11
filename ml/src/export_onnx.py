"""Phase H - export the best tree-based model to ONNX for the plugin, plus Java parity fixtures.

The best tree model is chosen by recall at 1% FPR (count windows, Phase D CV table). The exported
model is the final model of evaluate.fit_final (trained on ALL synthetic participants) and its
1%-FPR threshold. It is a DEMO model: the plugin ships it in shadow mode by default.

Outputs
  results/models/click_model.onnx + click_model.json   (copied into plugin resources)
  results/models/click_model_novelty.onnx              (second detector of the Phase I ensemble)
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


def novelty_to_onnx(det):
    """sklearn Pipeline(StandardScaler, OneClassSVM) or IsolationForest on a feature subset."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SFT
    return convert_sklearn(det.m, initial_types=[("features", SFT([None, len(det.features)]))],
                           target_opset={"": 15, "ai.onnx.ml": 3})


def onnx_named_output(path, X: np.ndarray, output: str, column: int) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    out = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[names.index(output)]
    out = np.asarray(out)
    return out[:, column] if out.ndim == 2 else out


def pvalue(sorted_cal: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = len(sorted_cal)
    ge = n - np.searchsorted(sorted_cal, scores, side="left")
    return (1.0 + ge) / (n + 1.0)


def export_ensemble(primary_path, w_parity: pd.DataFrame) -> dict | None:
    """Second detector + p-value tables of the Phase I policy. Scores are computed with
    onnxruntime (float32), exactly as the plugin will compute them."""
    path = C.PROCESSED_DIR / "decision_final.joblib"
    if not path.exists():
        log.warning("no Phase I policy found: exporting the single model only")
        return None
    d = joblib.load(path)
    pol = d["policy"]
    cal = d["cal_windows"]
    models = []
    for m in pol["models"]:
        if m == "xgb":
            feats, file, output, column, sign = FEATURES, "click_model.onnx", "probabilities", 1, 1.0
            onnx_path = primary_path
        else:
            det = d["ens_dets"][m]
            feats, file = list(det.features), "click_model_novelty.onnx"
            onnx_path = C.MODEL_DIR / file
            onnx_path.write_bytes(novelty_to_onnx(det).SerializeToString())
            # skl2onnx exposes decision_function as "scores"; our detector score is its negation
            output, column, sign = "scores", 0, -1.0
            py = det.score(w_parity)
            ox = sign * onnx_named_output(onnx_path, w_parity[feats].to_numpy(np.float32), output, 0)
            diff = float(np.max(np.abs(py - ox)))
            log.info("%s python vs onnx max |diff| = %.2e", m, diff)
            if diff > 1e-3:
                raise SystemExit(f"ONNX parity failed for {m}: {diff}")
            shutil.copy(onnx_path, C.PLUGIN_RES_DIR / "models" / file)
        cal_scores = np.sort(sign * onnx_named_output(onnx_path, cal[feats].to_numpy(np.float32), output, column))
        models.append({"name": m, "file": file, "features": feats, "output": output, "column": column,
                       "sign": sign, "cal_scores": [float(v) for v in cal_scores]})
    return {"rule": pol["rule"], "alpha": pol["alpha"], "session_k": pol["session_k"],
            "session_n": pol["session_n"], "lag_policy": pol["lag_policy"], "models": models,
            "calibration_human_windows": int(len(cal))}


def ensemble_pvalues(ens: dict, X: np.ndarray) -> np.ndarray:
    cols = []
    for m in ens["models"]:
        idx = [FEATURES.index(f) for f in m["features"]]
        sc = m["sign"] * onnx_named_output(C.MODEL_DIR / m["file"], X[:, idx], m["output"], m["column"])
        cols.append(pvalue(np.asarray(m["cal_scores"]), sc))
    return np.vstack(cols).T


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
    res = C.PLUGIN_RES_DIR / "models"
    res.mkdir(parents=True, exist_ok=True)
    ens = export_ensemble(out, w)
    if ens is not None:
        meta["ensemble"] = ens
    (C.MODEL_DIR / "click_model.json").write_text(json.dumps(meta, indent=2))
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
    if ens is not None:
        Xs = sample[FEATURES].to_numpy(np.float32)
        pv = ensemble_pvalues(ens, Xs)
        fx["ensemble_samples"] = [{"x": [float(v) for v in r], "p": [float(v) for v in pr]}
                                  for r, pr in zip(Xs, pv)]
    tdir = C.REPO_DIR / "plugin" / "src" / "test" / "resources"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "click_features_fixture.json").write_text(json.dumps(fx, indent=1))
    log.info("exported %s (%d bytes), threshold@1%%=%.4f", out.name, out.stat().st_size, thr[0.01])

    # Latency of one window (feature extraction in Python + single-row ONNX inference, 1 thread).
    # Indicative only: measured in Python on this machine, not inside the Minecraft server.
    import platform
    import time
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x1 = X[:1]
    for _ in range(200):
        sess.run(None, {name: x1})
    lat = []
    for i in range(2000):
        t0 = time.perf_counter(); sess.run(None, {name: X[i % len(X):i % len(X) + 1]}); lat.append(time.perf_counter() - t0)
    tw = np.cumsum(np.full(50, 100.0)); hw = np.zeros(50, bool); dw = np.zeros(50)
    fl = []
    for _ in range(2000):
        t0 = time.perf_counter(); window_features(tw, hw, dw); fl.append(time.perf_counter() - t0)
    pd.DataFrame([{"what": "onnx_inference_1_window", "median_us": np.median(lat) * 1e6,
                   "p99_us": np.percentile(lat, 99) * 1e6},
                  {"what": "python_feature_extraction_1_window", "median_us": np.median(fl) * 1e6,
                   "p99_us": np.percentile(fl, 99) * 1e6}]).assign(
        machine=f"{platform.processor() or platform.machine()} / {platform.system()}",
        note="Python onnxruntime, 1 intra-op thread; indicative, not measured in the server").to_csv(
        C.TAB_DIR / "onnx_latency.csv", index=False)


if __name__ == "__main__":
    main()
