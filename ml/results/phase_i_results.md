# Phase I - decision policy (DEMO - synthetic, not evidence about real players)

p-values from calibration humans; window flagged if min p < 0.01 / M (Bonferroni). Adaptive attackers: evolution strategy against each rule, >= 12 CPS within bouts.

## Window level (human FPR cross-validated; adaptive attackers vs the final ensemble)

| rule | fpr | recall[none] | recall[uniform] | recall[gaussian] | recall[human_like] | recall[opt_vs_xgb] | recall[opt_vs_xgb+ocsvm] | recall[opt_vs_xgb+iforest] | recall[opt_vs_xgb+ocsvm_robust] | recall[opt_vs_xgb+iforest_robust] |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb | 0.013 [0.005, 0.025] | 0.995 [0.990, 0.999] | 0.973 [0.941, 0.996] | 0.988 [0.973, 0.999] | 0.153 [0.059, 0.250] | 0.006 [0.004, 0.008] | 0.065 [0.056, 0.075] | 0.001 [0.000, 0.003] | 0.008 [0.004, 0.011] | 0.008 [0.005, 0.011] |
| xgb+ocsvm | 0.008 [0.004, 0.014] | 0.998 [0.995, 1.000] | 0.945 [0.880, 0.985] | 0.973 [0.938, 0.997] | 0.085 [0.031, 0.152] | 1.000 [1.000, 1.000] | 0.039 [0.031, 0.048] | 0.999 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] |
| xgb+iforest | 0.006 [0.002, 0.012] | 0.978 [0.946, 0.996] | 0.946 [0.874, 0.991] | 0.967 [0.936, 0.990] | 0.081 [0.028, 0.148] | 0.021 [0.016, 0.025] | 0.039 [0.031, 0.047] | 0.009 [0.007, 0.012] | 0.036 [0.027, 0.046] | 0.039 [0.029, 0.050] |
| xgb+ocsvm_robust | 0.008 [0.003, 0.015] | 0.999 [0.999, 1.000] | 0.946 [0.878, 0.990] | 0.974 [0.938, 0.998] | 0.084 [0.032, 0.151] | 0.113 [0.095, 0.133] | 0.041 [0.034, 0.050] | 0.022 [0.018, 0.025] | 0.029 [0.020, 0.039] | 0.053 [0.042, 0.064] |
| xgb+iforest_robust | 0.010 [0.005, 0.017] | 0.992 [0.983, 0.998] | 0.945 [0.881, 0.991] | 0.967 [0.931, 0.993] | 0.079 [0.026, 0.145] | 0.003 [0.002, 0.005] | 0.040 [0.031, 0.048] | 0.000 [0.000, 0.001] | 0.003 [0.001, 0.006] | 0.004 [0.002, 0.006] |

## Choice of the second detector (rule fixed in advance)

Eligible if clean FPR <= 1.5% and simulated-lag FPR at ping <= 80 ms <= 2%; then highest recall against the attacker optimised against that same ensemble. **Chosen: `xgb+ocsvm_robust`.**

The lag condition and the `*_robust` candidates (novelty detectors without the network features `tick_frac` and `dup_frac`) were added after a first run in which the rule without the lag condition chose One-Class SVM on all features; that detector flagged 100% of human windows at a simulated 80 ms ping, because `tick_frac` shifts with network jitter.

| rule | clean_fpr | max_fpr_ping_le_80 | recall_vs_self | eligible |
|---|---|---|---|---|
| xgb+ocsvm | 0.008 | 1.000 | 0.039 | False |
| xgb+iforest | 0.006 | 0.023 | 0.009 | False |
| xgb+ocsvm_robust | 0.008 | 0.017 | 0.029 | True |
| xgb+iforest_robust | 0.010 | 0.015 | 0.004 | True |

## Session decision (alarm if >= K of the last 8 windows are flagged)

K = smallest value with no false alarm on the calibration humans' sessions (rates per K: {"xgb": {"1": 0.23333333333333334, "2": 0.06666666666666667, "3": 0.0}, "xgb+ocsvm": {"1": 0.2, "2": 0.03333333333333333, "3": 0.0}, "xgb+iforest": {"1": 0.16666666666666666, "2": 0.03333333333333333, "3": 0.03333333333333333, "4": 0.0}, "xgb+ocsvm_robust": {"1": 0.2, "2": 0.03333333333333333, "3": 0.0}, "xgb+iforest_robust": {"1": 0.2, "2": 0.03333333333333333, "3": 0.0}}).

| rule | k | level | sessions | alarm_rate | median_windows_to_alarm |
|---|---|---|---|---|---|
| xgb | 3 | gaussian | 72 | 1.000 | 3.000 |
| xgb | 3 | human | 120 | 0.033 | 22.000 |
| xgb | 3 | human_like | 72 | 0.306 | 6.500 |
| xgb | 3 | none | 72 | 1.000 | 3.000 |
| xgb | 3 | opt_vs_xgb | 72 | 0.014 | 5.000 |
| xgb | 3 | opt_vs_xgb+iforest | 72 | 0.000 | nan |
| xgb | 3 | opt_vs_xgb+iforest_robust | 72 | 0.014 | 40.000 |
| xgb | 3 | opt_vs_xgb+ocsvm | 72 | 0.264 | 28.000 |
| xgb | 3 | opt_vs_xgb+ocsvm_robust | 72 | 0.014 | 60.000 |
| xgb | 3 | uniform | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | gaussian | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | human | 120 | 0.008 | 8.000 |
| xgb+ocsvm | 3 | human_like | 72 | 0.236 | 11.000 |
| xgb+ocsvm | 3 | none | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | opt_vs_xgb | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | opt_vs_xgb+iforest | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | opt_vs_xgb+iforest_robust | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | opt_vs_xgb+ocsvm | 72 | 0.111 | 23.500 |
| xgb+ocsvm | 3 | opt_vs_xgb+ocsvm_robust | 72 | 1.000 | 3.000 |
| xgb+ocsvm | 3 | uniform | 72 | 1.000 | 3.000 |
| xgb+iforest | 4 | gaussian | 72 | 1.000 | 4.000 |
| xgb+iforest | 4 | human | 120 | 0.000 | nan |
| xgb+iforest | 4 | human_like | 72 | 0.181 | 20.000 |
| xgb+iforest | 4 | none | 72 | 1.000 | 4.000 |
| xgb+iforest | 4 | opt_vs_xgb | 72 | 0.000 | nan |
| xgb+iforest | 4 | opt_vs_xgb+iforest | 72 | 0.000 | nan |
| xgb+iforest | 4 | opt_vs_xgb+iforest_robust | 72 | 0.028 | 45.500 |
| xgb+iforest | 4 | opt_vs_xgb+ocsvm | 72 | 0.069 | 25.000 |
| xgb+iforest | 4 | opt_vs_xgb+ocsvm_robust | 72 | 0.028 | 36.500 |
| xgb+iforest | 4 | uniform | 72 | 0.986 | 4.000 |
| xgb+ocsvm_robust | 3 | gaussian | 72 | 1.000 | 3.000 |
| xgb+ocsvm_robust | 3 | human | 120 | 0.017 | 17.000 |
| xgb+ocsvm_robust | 3 | human_like | 72 | 0.222 | 10.500 |
| xgb+ocsvm_robust | 3 | none | 72 | 1.000 | 3.000 |
| xgb+ocsvm_robust | 3 | opt_vs_xgb | 72 | 0.528 | 18.500 |
| xgb+ocsvm_robust | 3 | opt_vs_xgb+iforest | 72 | 0.069 | 50.000 |
| xgb+ocsvm_robust | 3 | opt_vs_xgb+iforest_robust | 72 | 0.222 | 13.000 |
| xgb+ocsvm_robust | 3 | opt_vs_xgb+ocsvm | 72 | 0.111 | 23.500 |
| xgb+ocsvm_robust | 3 | opt_vs_xgb+ocsvm_robust | 72 | 0.125 | 23.000 |
| xgb+ocsvm_robust | 3 | uniform | 72 | 1.000 | 3.000 |
| xgb+iforest_robust | 3 | gaussian | 72 | 1.000 | 3.000 |
| xgb+iforest_robust | 3 | human | 120 | 0.025 | 8.000 |
| xgb+iforest_robust | 3 | human_like | 72 | 0.208 | 10.000 |
| xgb+iforest_robust | 3 | none | 72 | 1.000 | 3.000 |
| xgb+iforest_robust | 3 | opt_vs_xgb | 72 | 0.000 | nan |
| xgb+iforest_robust | 3 | opt_vs_xgb+iforest | 72 | 0.000 | nan |
| xgb+iforest_robust | 3 | opt_vs_xgb+iforest_robust | 72 | 0.000 | nan |
| xgb+iforest_robust | 3 | opt_vs_xgb+ocsvm | 72 | 0.111 | 23.500 |
| xgb+iforest_robust | 3 | opt_vs_xgb+ocsvm_robust | 72 | 0.000 | nan |
| xgb+iforest_robust | 3 | uniform | 72 | 1.000 | 3.000 |

## Lag (SIMULATED): always evaluate vs conservative mode

Conservative: ping <= 80 ms normal; <= 150 ms alpha x 0.25; above that the window is not evaluated (coverage 0).

Human window FPR:

| rule | policy | 0 | 20 | 80 | 150 | 250 |
|---|---|---|---|---|---|---|
| xgb | always | 0.0286 | 0.0269 | 0.0259 | 0.0425 | 0.0157 |
| xgb | conservative | 0.0286 | 0.0269 | 0.0259 | 0.0106 | 0.0000 |
| xgb+iforest | always | 0.0156 | 0.0160 | 0.0228 | 0.1605 | 0.2237 |
| xgb+iforest | conservative | 0.0156 | 0.0160 | 0.0228 | 0.1063 | 0.0000 |
| xgb+iforest_robust | always | 0.0152 | 0.0152 | 0.0134 | 0.0202 | 0.0082 |
| xgb+iforest_robust | conservative | 0.0152 | 0.0152 | 0.0134 | 0.0026 | 0.0000 |
| xgb+ocsvm | always | 0.0195 | 0.0520 | 1.0000 | 1.0000 | 1.0000 |
| xgb+ocsvm | conservative | 0.0195 | 0.0520 | 1.0000 | 1.0000 | 0.0000 |
| xgb+ocsvm_robust | always | 0.0160 | 0.0156 | 0.0168 | 0.1180 | 0.1259 |
| xgb+ocsvm_robust | conservative | 0.0160 | 0.0156 | 0.0168 | 0.0392 | 0.0000 |

Recall on standard attackers:

| rule | policy | 0 | 20 | 80 | 150 | 250 |
|---|---|---|---|---|---|---|
| xgb | always | 0.980 | 0.976 | 0.968 | 0.901 | 0.585 |
| xgb | conservative | 0.980 | 0.976 | 0.968 | 0.693 | 0.000 |
| xgb+iforest | always | 0.959 | 0.956 | 0.948 | 0.915 | 0.890 |
| xgb+iforest | conservative | 0.959 | 0.956 | 0.948 | 0.801 | 0.000 |
| xgb+iforest_robust | always | 0.958 | 0.953 | 0.940 | 0.837 | 0.419 |
| xgb+iforest_robust | conservative | 0.958 | 0.953 | 0.940 | 0.498 | 0.000 |
| xgb+ocsvm | always | 0.959 | 0.960 | 1.000 | 1.000 | 1.000 |
| xgb+ocsvm | conservative | 0.959 | 0.960 | 1.000 | 1.000 | 0.000 |
| xgb+ocsvm_robust | always | 0.959 | 0.954 | 0.942 | 0.889 | 0.738 |
| xgb+ocsvm_robust | conservative | 0.959 | 0.954 | 0.942 | 0.737 | 0.000 |

