# Phase F - lag robustness (SIMULATED lag)

Ping class -> (extra per-packet jitter sd, lag spikes per second) is an assumption: 0 ms -> (0 ms, 0.0/s), 20 ms -> (2 ms, 0.01/s), 80 ms -> (8 ms, 0.03/s), 150 ms -> (15 ms, 0.06/s), 250 ms -> (25 ms, 0.12/s); spike duration U(0.5, 1.5) x ping, packets released together after a spike.

## Synthetic (DEMO - synthetic, not evidence about real players) - fresh human population never used for training

| model | fpr@ping0 | fpr@ping20 | fpr@ping80 | fpr@ping150 | fpr@ping250 |
|---|---|---|---|---|---|
| baseline_orig_clean | 0.8863 | 0.8863 | 0.8954 | 0.9359 | 0.9762 |
| baseline_orig_raw | 0.8837 | 0.8837 | 0.7782 | 0.6685 | 0.7327 |
| cv_rule | 0.0294 | 0.0285 | 0.0240 | 0.0090 | 0.0010 |
| iforest | 0.0091 | 0.0277 | 0.1485 | 0.3981 | 0.5400 |
| ocsvm | 0.0134 | 0.0791 | 1.0000 | 1.0000 | 1.0000 |
| rf | 0.0264 | 0.0246 | 0.0214 | 0.0263 | 0.0091 |
| xgb | 0.0277 | 0.0242 | 0.0760 | 0.1230 | 0.0257 |

Recall on standard attackers under the same lag (1% threshold):

| model | recall@ping0 | recall@ping20 | recall@ping80 | recall@ping150 | recall@ping250 |
|---|---|---|---|---|---|
| baseline_orig_clean | 0.973 | 0.973 | 0.974 | 0.994 | 1.000 |
| baseline_orig_raw | 0.973 | 0.973 | 0.977 | 0.999 | 1.000 |
| cv_rule | 0.200 | 0.194 | 0.113 | 0.033 | 0.007 |
| iforest | 0.533 | 0.797 | 0.997 | 0.993 | 0.986 |
| ocsvm | 0.915 | 0.946 | 1.000 | 1.000 | 1.000 |
| rf | 0.996 | 0.995 | 0.983 | 0.860 | 0.421 |
| xgb | 0.994 | 0.995 | 0.991 | 0.925 | 0.585 |


## REAL session timestamps + SIMULATED lag - original rule, per event

Every flag is a false positive (the session is human). One short session: anecdotal.

| ping_ms | repeats | naive_java_flags_mean | naive_java_n | cleaned_active_flags_mean | cleaned_active_n | held_swings_mean |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 33.00 | 0.00 | 20.00 | 56.00 |
| 20.00 | 20.00 | 3.55 | 33.00 | 0.20 | 20.20 | 55.80 |
| 80.00 | 20.00 | 3.25 | 33.00 | 8.70 | 32.90 | 42.85 |
| 150.00 | 20.00 | 3.50 | 33.00 | 16.40 | 48.25 | 24.50 |
| 250.00 | 20.00 | 3.25 | 33.00 | 13.95 | 50.10 | 14.55 |
