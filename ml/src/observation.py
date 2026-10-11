"""Server-side observation model shared by every synthetic generator (humans and adversaries).

A "true" click time t (what the finger/clicker does) becomes a server event in three steps:

1. Client tick quantization: the Java client handles mouse input once per client tick (20 Hz),
   so a click is sent at the next client tick boundary  c = phase + 50 * ceil((t - phase) / 50).
   Several clicks inside the same client tick are sent back-to-back (sub-millisecond apart).
   This is consistent with the REAL data: accepted swing intervals cluster at multiples of
   50 ms (see results/data_report.md). It is a model, not a measured client trace.
2. Network + server processing jitter: s = c + base_delay + J, J ~ Normal(0, sigma_ms).
   sigma_ms is calibrated from the real interval residuals when there are enough of them
   (an interval residual is the difference of two independent jitters -> sd / sqrt(2)).
3. TCP keeps order: s_i = max(s_i, s_{i-1} + 0.05 ms).

Lag spikes (lag.py) are added on top with `lag_spikes`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

DEFAULT_SIGMA_MS = 2.5   # placeholder when there is not enough real data to calibrate
MIN_CALIBRATION_N = 30


@dataclass
class NetParams:
    sigma_ms: float = DEFAULT_SIGMA_MS
    base_ms: float = 0.0
    source: str = "placeholder"


def calibrate_jitter(real_iv_ms: np.ndarray) -> NetParams:
    """Single-event jitter sd from real interval residuals to the 50 ms grid (robust MAD)."""
    from .data import tick_residuals
    iv = np.asarray(real_iv_ms, dtype=float)
    iv = iv[(iv > 0) & (iv < C.PAUSE_MS)]
    if len(iv) < MIN_CALIBRATION_N:
        return NetParams()
    r = tick_residuals(iv)
    r = r[np.abs(r) <= 15]                      # ignore intervals that are not grid-like at all
    if len(r) < MIN_CALIBRATION_N:
        return NetParams()
    mad_sd = 1.4826 * np.median(np.abs(r - np.median(r)))
    return NetParams(sigma_ms=float(max(mad_sd / np.sqrt(2), 0.5)), source=f"real (n={len(r)})")


def client_quantize(t_ms: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0, C.TICK_MS)
    k = np.ceil((np.asarray(t_ms) - phase) / C.TICK_MS)
    c = phase + C.TICK_MS * k
    # clicks in the same tick are sent back-to-back
    same = np.r_[False, np.diff(c) == 0]
    run = np.zeros(len(c))
    for i in range(1, len(c)):
        run[i] = run[i - 1] + 1 if same[i] else 0
    return c + run * rng.uniform(0.05, 0.4, size=len(c))


def network(c_ms: np.ndarray, rng: np.random.Generator, net: NetParams) -> np.ndarray:
    s = np.asarray(c_ms, dtype=float) + net.base_ms + rng.normal(0, net.sigma_ms, size=len(c_ms))
    return enforce_order(s)


def enforce_order(s: np.ndarray, gap_ms: float = 0.05) -> np.ndarray:
    s = np.array(s, dtype=float)
    for i in range(1, len(s)):
        if s[i] < s[i - 1] + gap_ms:
            s[i] = s[i - 1] + gap_ms
    return s


def lag_spikes(s_ms: np.ndarray, rng: np.random.Generator, rate_per_s: float,
               dur_ms: tuple[float, float]) -> np.ndarray:
    """Freeze delivery during random spikes and release queued packets together (bunching)."""
    s = np.array(s_ms, dtype=float)
    if rate_per_s <= 0 or len(s) == 0:
        return s
    T = s[-1] - s[0] + 1
    n = rng.poisson(rate_per_s * T / 1000.0)
    for start in rng.uniform(s[0], s[-1], size=n):
        end = start + rng.uniform(*dur_ms)
        m = (s >= start) & (s < end)
        s[m] = end + np.arange(m.sum()) * 0.05
    return enforce_order(np.sort(s))


def observe(t_ms: np.ndarray, rng: np.random.Generator, net: NetParams) -> np.ndarray:
    """True click times (sorted, ms) -> server receive times (ms)."""
    if len(t_ms) == 0:
        return np.asarray(t_ms, dtype=float)
    return network(client_quantize(np.sort(t_ms), rng), rng, net)
