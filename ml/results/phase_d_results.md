# Phase D - detection results

**Synthetic sections: DEMO - synthetic, not evidence about real players.** Humans are placeholder-generator humans (see README); attack levels: none (auto_fixed), uniform and gaussian jitter (auto_jitter).

## Synthetic, count windows (StratifiedGroupKFold by participant, 5 folds)

Windows: humans=4406, attacks=16902; calibration humans per fold ~950 windows (an FPR target of 0.1% is close to the resolution limit of this calibration set).

| model | recall@0.001 | fpr@0.001 | recall@0.01 | fpr@0.01 | pr_auc | f1@0.01 |
|---|---|---|---|---|---|---|
| baseline_orig_raw | 0.952 [0.907, 0.986] | 0.859 [0.789, 0.917] | 0.952 [0.907, 0.988] | 0.859 [0.791, 0.921] | 0.978 [0.960, 0.990] | 0.875 |
| baseline_orig_clean | 0.952 [0.910, 0.988] | 0.859 [0.789, 0.917] | 0.952 [0.912, 0.987] | 0.859 [0.790, 0.920] | 0.978 [0.960, 0.990] | 0.875 |
| cv_rule | 0.195 [0.124, 0.277] | 0.003 [0.000, 0.006] | 0.286 [0.198, 0.370] | 0.014 [0.006, 0.024] | 0.978 [0.966, 0.986] | 0.444 |
| iforest | 0.360 [0.275, 0.442] | 0.004 [0.001, 0.006] | 0.741 [0.664, 0.813] | 0.022 [0.013, 0.032] | 0.995 [0.991, 0.997] | 0.848 |
| ocsvm | 0.607 [0.495, 0.705] | 0.003 [0.001, 0.006] | 0.916 [0.866, 0.959] | 0.016 [0.009, 0.024] | 0.997 [0.994, 0.999] | 0.954 |
| rf | 0.700 [0.598, 0.794] | 0.001 [0.000, 0.002] | 0.986 [0.970, 0.998] | 0.009 [0.003, 0.016] | 1.000 [0.999, 1.000] | 0.992 |
| xgb | 0.921 [0.880, 0.956] | 0.002 [0.000, 0.005] | 0.984 [0.969, 0.996] | 0.008 [0.003, 0.017] | 1.000 [0.999, 1.000] | 0.991 |
| cnn1d | 0.764 [0.688, 0.838] | 0.002 [0.000, 0.005] | 0.834 [0.768, 0.891] | 0.020 [0.004, 0.038] | 0.993 [0.985, 0.997] | 0.907 |

Recall per attack level at the 1% threshold:

| model | recall@0.01[none] | recall@0.01[uniform] | recall@0.01[gaussian] |
|---|---|---|---|
| baseline_orig_raw | 0.890 [0.772, 0.975] | 0.999 [0.996, 1.000] | 0.966 [0.903, 1.000] |
| baseline_orig_clean | 0.890 [0.763, 0.975] | 0.998 [0.995, 1.000] | 0.966 [0.900, 1.000] |
| cv_rule | 0.468 [0.273, 0.655] | 0.208 [0.111, 0.321] | 0.188 [0.099, 0.302] |
| iforest | 0.587 [0.425, 0.751] | 0.809 [0.687, 0.898] | 0.821 [0.717, 0.907] |
| ocsvm | 0.995 [0.989, 0.999] | 0.851 [0.737, 0.944] | 0.905 [0.819, 0.972] |
| rf | 1.000 [0.999, 1.000] | 0.980 [0.948, 0.998] | 0.979 [0.941, 1.000] |
| xgb | 0.999 [0.998, 1.000] | 0.975 [0.938, 0.998] | 0.978 [0.943, 1.000] |
| cnn1d | 0.793 [0.646, 0.893] | 0.867 [0.747, 0.947] | 0.841 [0.715, 0.927] |

Session-level alarm (>= 3 flagged windows in a session, 1% threshold) and confusion counts:

| model | session_alarm_human | session_alarm_attack | tp@0.01 | fp@0.01 | fn@0.01 | tn@0.01 |
|---|---|---|---|---|---|---|
| baseline_orig_raw | 0.983 | 0.935 | 16097 | 3786 | 805 | 620 |
| baseline_orig_clean | 0.983 | 0.935 | 16098 | 3785 | 804 | 621 |
| cv_rule | 0.067 | 0.560 | 4835 | 62 | 12067 | 4344 |
| iforest | 0.108 | 0.898 | 12516 | 96 | 4386 | 4310 |
| ocsvm | 0.092 | 0.958 | 15478 | 69 | 1424 | 4337 |
| rf | 0.033 | 1.000 | 16669 | 38 | 233 | 4368 |
| xgb | 0.025 | 1.000 | 16635 | 35 | 267 | 4371 |
| cnn1d | 0.075 | 0.852 | 14099 | 87 | 2803 | 4319 |


## Synthetic, time windows (StratifiedGroupKFold by participant, 5 folds)

Windows: humans=8076, attacks=15093; calibration humans per fold ~1699 windows (an FPR target of 0.1% is close to the resolution limit of this calibration set).

| model | recall@0.001 | fpr@0.001 | recall@0.01 | fpr@0.01 | pr_auc | f1@0.01 |
|---|---|---|---|---|---|---|
| baseline_orig_raw | 0.922 [0.855, 0.974] | 0.698 [0.611, 0.785] | 0.922 [0.857, 0.975] | 0.698 [0.606, 0.782] | 0.948 [0.913, 0.976] | 0.803 |
| baseline_orig_clean | 0.922 [0.859, 0.977] | 0.697 [0.607, 0.782] | 0.922 [0.864, 0.974] | 0.697 [0.611, 0.788] | 0.948 [0.919, 0.974] | 0.803 |
| cv_rule | 0.086 [0.036, 0.149] | 0.001 [0.000, 0.002] | 0.196 [0.111, 0.277] | 0.011 [0.005, 0.018] | 0.908 [0.862, 0.944] | 0.326 |
| iforest | 0.086 [0.039, 0.139] | 0.001 [0.000, 0.002] | 0.267 [0.186, 0.352] | 0.010 [0.005, 0.015] | 0.964 [0.941, 0.979] | 0.419 |
| rf | 0.850 [0.777, 0.910] | 0.001 [0.000, 0.001] | 0.950 [0.911, 0.984] | 0.022 [0.009, 0.039] | 0.998 [0.996, 1.000] | 0.969 |
| xgb | 0.857 [0.785, 0.920] | 0.002 [0.001, 0.005] | 0.953 [0.913, 0.984] | 0.021 [0.010, 0.036] | 0.999 [0.997, 1.000] | 0.970 |

Recall per attack level at the 1% threshold:

| model | recall@0.01[none] | recall@0.01[uniform] | recall@0.01[gaussian] |
|---|---|---|---|
| baseline_orig_raw | 0.834 [0.668, 0.958] | 0.989 [0.976, 0.999] | 0.942 [0.847, 1.000] |
| baseline_orig_clean | 0.834 [0.668, 0.958] | 0.989 [0.975, 0.999] | 0.942 [0.842, 1.000] |
| cv_rule | 0.429 [0.238, 0.595] | 0.034 [0.016, 0.056] | 0.124 [0.014, 0.256] |
| iforest | 0.100 [0.012, 0.211] | 0.437 [0.272, 0.593] | 0.262 [0.126, 0.412] |
| rf | 0.994 [0.988, 0.998] | 0.908 [0.813, 0.980] | 0.949 [0.864, 0.996] |
| xgb | 0.995 [0.988, 0.999] | 0.916 [0.829, 0.981] | 0.949 [0.867, 0.997] |

Session-level alarm (>= 3 flagged windows in a session, 1% threshold) and confusion counts:

| model | session_alarm_human | session_alarm_attack | tp@0.01 | fp@0.01 | fn@0.01 | tn@0.01 |
|---|---|---|---|---|---|---|
| baseline_orig_raw | 0.983 | 0.935 | 13915 | 5634 | 1178 | 2442 |
| baseline_orig_clean | 0.983 | 0.935 | 13911 | 5632 | 1182 | 2444 |
| cv_rule | 0.083 | 0.319 | 2953 | 87 | 12140 | 7989 |
| iforest | 0.067 | 0.481 | 4023 | 78 | 11070 | 7998 |
| rf | 0.158 | 1.000 | 14339 | 178 | 754 | 7898 |
| xgb | 0.167 | 1.000 | 14387 | 170 | 706 | 7906 |


## REAL data

- scored real windows, count (50 clicks): humans=0, attacks=0
- scored real windows, time (5 s): humans=0, attacks=0

No real window survives cleaning and routing (too few non-held clicks).
**No real-data detection metric can be computed yet.** The per-event behaviour of the
original thresholds on the real session is in `data_report.md`.
