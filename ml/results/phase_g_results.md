# Phase G - explainability (DEMO - synthetic, not evidence about real players)

Model: final XGBoost (trained on synthetic humans + none/uniform/gaussian). Explanation method: shap 0.46.0 TreeExplainer; base value 2.120 log-odds. Evaluated on a fresh synthetic population (1837 human / 5488 attack windows).

## Feature ranking

| feature | description | mean abs SHAP | permutation importance (PR-AUC drop) |
|---|---|---|---|
| entropy | interval histogram entropy (bits) | 3.887 | 0.0032 ± 0.0002 |
| mean | mean click interval (ms) | 1.905 | 0.0004 ± 0.0001 |
| ac1 | lag-1 autocorrelation | 1.132 | 0.0002 ± 0.0000 |
| std | std of intervals (ms) | 1.079 | 0.0001 ± 0.0000 |
| p95 | 95th percentile interval (ms) | 0.996 | 0.0001 ± 0.0000 |
| kurt | excess kurtosis | 0.781 | 0.0002 ± 0.0000 |
| cv | coefficient of variation | 0.534 | 0.0000 ± 0.0000 |
| p05 | 5th percentile interval (ms) | 0.440 | 0.0000 ± 0.0000 |
| skew | skewness | 0.431 | 0.0001 ± 0.0000 |
| min | shortest interval (ms) | 0.403 | 0.0000 ± 0.0000 |
| max_cps_1s | max clicks in any 1 s | 0.310 | 0.0001 ± 0.0000 |
| pause_frac | fraction of pauses > 1 s | 0.272 | 0.0002 ± 0.0000 |
| tick_frac | fraction of intervals on the 50 ms grid | 0.133 | -0.0000 ± 0.0000 |
| dup_frac | fraction of same-tick duplicate swings | 0.049 | 0.0000 ± 0.0000 |

## Three example decisions as audit-log lines

Each line is what `auditoria.log` would contain (actor pseudonymised by HMAC in the plugin).

**attack window flagged with high confidence** (true level: none)

```
[ML-AutoClicker] FLAG (+1 VL) | score=1.000 threshold=0.964 (1% FPR on calibration humans; DEMO model) | window=50 swings, 48 active intervals, held_frac=0.00 | reasons: entropy=1.98 (typical human 2.83) raises suspicion by 3.88 log-odds; mean=78.2 (typical human 149) raises suspicion by 2.49 log-odds; ac1=-0.798 (typical human 0.00218) raises suspicion by 1.98 log-odds
```

**attack window missed (below threshold)** (true level: gaussian)

```
[ML-AutoClicker] no action | score=0.699 threshold=0.964 (1% FPR on calibration humans; DEMO model) | window=50 swings, 49 active intervals, held_frac=0.00 | reasons: entropy=2.85 (typical human 2.83) lowers suspicion by 3.06 log-odds; skew=0.0599 (typical human 3.09) raises suspicion by 1.38 log-odds; max_cps_1s=10 (typical human 10) raises suspicion by 0.91 log-odds
```

**human window close to the threshold** (true level: human)

```
[ML-AutoClicker] FLAG (+1 VL) | score=0.986 threshold=0.964 (1% FPR on calibration humans; DEMO model) | window=50 swings, 49 active intervals, held_frac=0.00 | reasons: kurt=38.9 (typical human 10.7) raises suspicion by 2.25 log-odds; std=123 (typical human 102) lowers suspicion by 0.94 log-odds; p95=153 (typical human 252) raises suspicion by 0.79 log-odds
```
