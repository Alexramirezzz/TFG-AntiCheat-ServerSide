"""Phase C - window features.

Feature definitions are written out explicitly (no library defaults) because the Java plugin
re-implements them (ClickFeatureExtractor.java) and a parity test compares both on fixtures.

Window input: consecutive ACCEPTED swings of one session (after de-duplication) with their
held-button flag and duplicate count (see data.clean_click_stream).
  intervals  = diffs between consecutive swings of the window, keeping interval i only when
               swing i is not held (held-button mining swings are game-generated, not player clicks)
  pause      = kept interval > PAUSE_MS (excluded from distribution statistics)
  A          = kept intervals <= PAUSE_MS (used for every distribution statistic)
A window is valid when len(A) >= MIN_INTERVALS_PER_WINDOW. A valid window with
held_frac >= HELD_ROUTE_FRAC is routed to "mining" and is NOT scored by click detectors.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C

# Model inputs. held_frac, n_active, ping/tps are kept for routing/audit only (see README).
FEATURES = ["mean", "std", "cv", "skew", "kurt", "min", "p05", "p95", "entropy", "ac1",
            "max_cps_1s", "tick_frac", "dup_frac", "pause_frac"]
AUDIT = ["n_swings", "n_active", "held_frac", "ping_mean", "tps_min"]
N_HIST_BINS = len(C.HIST_EDGES_MS)  # 20 regular bins + 1 overflow bin


def percentile_linear(sorted_a: np.ndarray, q: float) -> float:
    """numpy 'linear' percentile, written explicitly (Java parity)."""
    n = len(sorted_a)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(sorted_a[lo] + (pos - lo) * (sorted_a[hi] - sorted_a[lo]))


def interval_stats(A: np.ndarray) -> dict:
    A = np.asarray(A, dtype=float)
    n = len(A)
    mean = float(A.mean())
    d = A - mean
    m2 = float(np.mean(d ** 2)); m3 = float(np.mean(d ** 3)); m4 = float(np.mean(d ** 4))
    std = math.sqrt(m2)
    skew = m3 / m2 ** 1.5 if m2 > 1e-12 else 0.0
    kurt = m4 / m2 ** 2 - 3.0 if m2 > 1e-12 else 0.0
    s = np.sort(A)
    bins = np.minimum(np.floor(A / 25.0).astype(int), N_HIST_BINS - 1)
    p = np.bincount(bins, minlength=N_HIST_BINS) / n
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    den = float((d ** 2).sum())
    ac1 = float((d[:-1] * d[1:]).sum() / den) if den > 1e-12 and n > 1 else 0.0
    res = A - C.TICK_MS * np.round(A / C.TICK_MS)
    return {"mean": mean, "std": std, "cv": std / mean if mean > 0 else 0.0, "skew": skew,
            "kurt": kurt, "min": float(s[0]), "p05": percentile_linear(s, 0.05),
            "p95": percentile_linear(s, 0.95), "entropy": entropy, "ac1": ac1,
            "tick_frac": float(np.mean(np.abs(res) <= C.TICK_NEAR_MS))}


def max_clicks_in_1s(t_ms: np.ndarray) -> int:
    t = np.sort(np.asarray(t_ms, dtype=float))
    best, j = 0, 0
    for i in range(len(t)):
        while t[i] - t[j] >= 1000.0:
            j += 1
        best = max(best, i - j + 1)
    return best


def window_features(t_ms: np.ndarray, held: np.ndarray, n_dup: np.ndarray) -> dict:
    t = np.asarray(t_ms, dtype=float)
    held = np.asarray(held, dtype=bool)
    n_dup = np.asarray(n_dup, dtype=float)
    n = len(t)
    out = {"n_swings": n, "held_frac": float(held.mean()) if n else float("nan"),
           "dup_frac": float(n_dup.sum() / (n + n_dup.sum())) if n else float("nan")}
    diffs = np.diff(t)
    kept = diffs[~held[1:]] if n > 1 else np.array([])
    A = kept[kept <= C.PAUSE_MS]
    out["pause_frac"] = float(np.mean(kept > C.PAUSE_MS)) if len(kept) else float("nan")
    out["n_active"] = int(len(A))
    out["valid"] = len(A) >= C.MIN_INTERVALS_PER_WINDOW
    if out["valid"]:
        out.update(interval_stats(A))
        out["max_cps_1s"] = float(max_clicks_in_1s(t[~held]))
    else:
        out.update({k: float("nan") for k in FEATURES if k not in out})
    out["routed"] = "mining" if out["held_frac"] >= C.HELD_ROUTE_FRAC else "click"
    return out


def _window_bounds(t: np.ndarray, window_type: str):
    if window_type == "count":
        n = len(t)
        for s in range(0, n - C.WINDOW_CLICKS + 1, C.STRIDE_CLICKS):
            yield s, s + C.WINDOW_CLICKS
    elif window_type == "time":
        if len(t) == 0:
            return
        w, st = C.WINDOW_S * 1000, C.STRIDE_S * 1000
        start = t[0]
        while start + w <= t[-1] + 1e-9:
            i0 = int(np.searchsorted(t, start, "left"))
            i1 = int(np.searchsorted(t, start + w, "left"))
            yield i0, i1
            start += st
    else:
        raise ValueError(window_type)


def make_windows(streams: pd.DataFrame, window_type: str = "count",
                 moves: pd.DataFrame | None = None, keep_sequences: bool = False) -> pd.DataFrame:
    """Windows for every session in a cleaned stream frame (output of data.build_click_streams)."""
    rows = []
    mv = {sid: g for sid, g in moves.groupby("session_id")} if moves is not None and len(moves) else {}
    meta_cols = [c for c in ["session_id", "participant", "label", "activity", "origin", "level"]
                 if c in streams.columns]
    for sid, g in streams.groupby("session_id", sort=False):
        g = g.sort_values("t_ms")
        t = g["t_ms"].to_numpy(); h = g["held"].to_numpy(bool); d = g["n_dup"].to_numpy()
        meta = g.iloc[0][meta_cols].to_dict()
        m = mv.get(sid)
        for w, (i0, i1) in enumerate(_window_bounds(t, window_type)):
            f = window_features(t[i0:i1], h[i0:i1], d[i0:i1])
            f.update(meta)
            f.update({"window_type": window_type, "window_idx": w,
                      "t_start_ms": float(t[i0]) if i1 > i0 else float("nan"),
                      "t_end_ms": float(t[i1 - 1]) if i1 > i0 else float("nan")})
            if m is not None and i1 > i0:
                mm = m[(m["t_ns"] >= t[i0] * 1e6) & (m["t_ns"] <= t[i1 - 1] * 1e6)]
                f["ping_mean"] = float(mm["ping"].mean()) if len(mm) else float("nan")
                f["tps_min"] = float(mm["tps"].min()) if len(mm) else float("nan")
            else:
                f["ping_mean"] = f["tps_min"] = float("nan")
            if keep_sequences:
                diffs = np.diff(t[i0:i1])
                f["seq"] = diffs[~h[i0 + 1:i1]].astype(np.float32)
            rows.append(f)
    return pd.DataFrame(rows)


def scored(windows: pd.DataFrame) -> pd.DataFrame:
    """Windows that click detectors score: valid and not routed to mining."""
    return windows[windows["valid"].astype(bool) & (windows["routed"] == "click")].reset_index(drop=True)


# ------------------------------------------------------------------ movement (report only)
def movement_windows(moves: pd.DataFrame) -> pd.DataFrame:
    """5 s windows of (already trimmed) moves: speed, acceleration and airborne ratio."""
    rows = []
    for sid, g in moves.groupby("session_id"):
        g = g.sort_values("t_ns")
        t = g["t_ns"].to_numpy() / 1e6
        hspd = np.hypot(g["dx"].to_numpy(), g["dz"].to_numpy())
        if len(t) == 0:
            continue
        start = t[0]
        while start + C.WINDOW_S * 1000 <= t[-1] + 1e-9:
            m = (t >= start) & (t < start + C.WINDOW_S * 1000)
            if m.sum() >= 5:
                h = hspd[m]
                rows.append({"session_id": sid, "label": g["label"].iloc[0], "t_start_ms": start,
                             "n": int(m.sum()), "speed_mean": float(h.mean()),
                             "speed_p95": float(np.percentile(h, 95)), "speed_max": float(h.max()),
                             "accel_abs_mean": float(np.abs(np.diff(h)).mean()) if m.sum() > 1 else 0.0,
                             "airborne_ratio": float((g["on_ground"].to_numpy()[m] == 0).mean()),
                             "dy_abs_max": float(np.abs(g["dy"].to_numpy()[m]).max()),
                             "ping_mean": float(g["ping"].to_numpy()[m].mean()),
                             "tps_min": float(g["tps"].to_numpy()[m].min())})
            start += C.STRIDE_S * 1000
    return pd.DataFrame(rows)
