import numpy as np
import pytest
from scipy import stats

from src import config as C
from src.features import (interval_stats, max_clicks_in_1s, percentile_linear, window_features)


def test_percentile_matches_numpy():
    a = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=float))
    for q in (0.05, 0.5, 0.95):
        assert percentile_linear(a, q) == pytest.approx(np.percentile(a, q * 100))


def test_moments_match_scipy_biased():
    rng = np.random.default_rng(0)
    A = rng.lognormal(np.log(120), 0.3, 200)
    s = interval_stats(A)
    assert s["mean"] == pytest.approx(A.mean())
    assert s["std"] == pytest.approx(A.std(ddof=0))
    assert s["skew"] == pytest.approx(stats.skew(A, bias=True))
    assert s["kurt"] == pytest.approx(stats.kurtosis(A, fisher=True, bias=True))


def test_constant_intervals():
    s = interval_stats(np.full(20, 50.0))
    assert s["std"] == 0 and s["cv"] == 0 and s["skew"] == 0 and s["kurt"] == 0
    assert s["entropy"] == 0           # one histogram bin
    assert s["ac1"] == 0
    assert s["tick_frac"] == 1.0


def test_alternating_intervals_autocorrelation():
    s = interval_stats(np.array([50.0, 100.0] * 10))
    assert s["ac1"] < -0.9
    assert s["entropy"] == pytest.approx(1.0)  # two equally likely bins -> 1 bit


def test_histogram_overflow_bin():
    s = interval_stats(np.array([600.0, 700.0, 10.0, 12.0]))
    assert s["entropy"] == pytest.approx(1.0)  # bins {0, overflow}


def test_max_cps():
    assert max_clicks_in_1s(np.arange(0, 3000, 100.0)) == 10
    assert max_clicks_in_1s(np.array([0, 999.0])) == 2
    assert max_clicks_in_1s(np.array([0, 1000.0])) == 1


def test_window_features_held_and_pause_handling():
    t = np.array([0, 100, 200, 250, 300, 350, 2000] + [2000 + 100 * k for k in range(1, 12)], float)
    held = np.zeros(len(t), bool); held[3:6] = True       # 250, 300, 350 are held swings
    f = window_features(t, held, np.zeros(len(t)))
    # kept intervals: 100,100,(held x3 dropped),1650(pause),100 x 11
    assert f["n_active"] == 13
    assert f["pause_frac"] == pytest.approx(1 / 14)
    assert f["mean"] == pytest.approx(100.0)
    assert f["held_frac"] == pytest.approx(3 / 18)
    assert f["valid"] and f["routed"] == "click"


def test_window_routed_to_mining_and_invalid():
    t = np.arange(0, 50 * 30, 50.0)
    held = np.ones(len(t), bool); held[0] = False
    f = window_features(t, held, np.zeros(len(t)))
    assert f["routed"] == "mining" and not f["valid"]


def test_dup_frac():
    f = window_features(np.arange(0, 1500, 100.0), np.zeros(15, bool), np.r_[np.ones(5), np.zeros(10)])
    assert f["dup_frac"] == pytest.approx(5 / 20)


def test_calibrate_threshold_respects_target():
    from src.evaluate import calibrate
    s = np.arange(1, 1001, dtype=float)
    thr = calibrate(s, 0.01)
    assert np.mean(s > thr) <= 0.01 and np.mean(s > thr) == pytest.approx(0.01)
    assert calibrate(s[:500], 0.001) == 500.0  # resolution limit: max of calibration set
