import numpy as np

from src.adversary import AttackParams, attack_click_times, effective_cps, sample_attack
from src.data import tick_residuals
from src.observation import NetParams, enforce_order, lag_spikes, observe
from src.synth_humans import human_click_times, sample_params


def test_observation_quantizes_to_client_ticks_without_jitter():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(60, 300, 200))
    s = observe(t, rng, NetParams(sigma_ms=0.0))
    iv = np.diff(s)
    iv = iv[iv > 1]  # same-tick back-to-back sends excluded
    assert np.all(np.abs(tick_residuals(iv)) < 1.0)


def test_observation_keeps_order():
    rng = np.random.default_rng(2)
    s = observe(np.sort(rng.uniform(0, 5000, 300)), rng, NetParams(sigma_ms=10.0))
    assert np.all(np.diff(s) > 0)


def test_fixed_autoclicker_is_strictly_periodic_before_observation():
    rng = np.random.default_rng(3)
    t = attack_click_times(AttackParams("none", 80.0, bout_s=(100, 100), gap_s=(1, 1)), 10, rng)
    assert np.allclose(np.diff(t), 80.0)


def test_effective_cps_matches_fixed_rate():
    assert abs(effective_cps({"median_ms": 100.0, "spread": 0.0, "rho": 0.0, "pause_prob": 0.0,
                              "pause_median_ms": 400.0}, np.random.default_rng(0), 500) - 10.0) < 1e-6


def test_human_generator_rate_in_placeholder_range():
    rng = np.random.default_rng(4)
    p = sample_params(rng)
    t = human_click_times(p, 120, rng)
    iv = np.diff(t)
    assert np.all(iv >= 35.0)
    assert 2 < len(t) / 120 < 15


def test_lag_spikes_bunch_packets():
    rng = np.random.default_rng(5)
    s = np.arange(0, 20000, 100.0)
    out = lag_spikes(s, rng, rate_per_s=0.5, dur_ms=(300, 300))
    assert np.all(np.diff(out) > 0)
    assert np.sum(np.diff(out) < 1.0) > 0  # some packets released together


def test_enforce_order():
    assert np.all(np.diff(enforce_order(np.array([0, 5, 3, 3, 10.0]))) > 0)


def test_sample_levels():
    rng = np.random.default_rng(6)
    for lvl in ("none", "uniform", "gaussian", "human_like"):
        assert sample_attack(lvl, rng).median_ms > 0
