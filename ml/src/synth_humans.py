"""Phase B - synthetic HUMAN click sessions (always origin = synthetic).

Two modes:
* "fitted": used automatically when the real data contain at least MIN_FIT_INTERVALS active
  (non-held) click intervals < 1 s from human sessions. Per-participant lognormal parameters,
  lag-1 autocorrelation and pause rate are estimated from the real data and synthetic
  participants are drawn around them.
* "placeholder": otherwise. A lognormal-with-structure generator whose hyper-parameters are
  ASSUMPTIONS (listed in PLACEHOLDER_HYPER), not measurements. Everything trained or evaluated
  on these humans is a DEMO and says so.

The generator returns true click times; the shared observation model (observation.py) turns them
into what the server would record, including client-tick quantization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C
from .observation import NetParams, observe

MIN_FIT_INTERVALS = 300

# Placeholder hyper-distribution (uniform ranges). Assumptions, documented in README.
PLACEHOLDER_HYPER = {
    "median_ms": (90.0, 170.0),     # ~6-11 CPS while actively clicking
    "sigma": (0.20, 0.45),          # log-sd of intervals
    "rho": (0.10, 0.50),            # AR(1) of log-interval deviations (rhythm drift)
    "bout_mean_clicks": (8.0, 25.0),
    "pause_median_ms": (400.0, 1200.0),
    "fatigue": (0.0, 0.15),         # relative slow-down of the median over a session
}


@dataclass
class HumanParams:
    median_ms: float
    sigma: float
    rho: float
    bout_mean_clicks: float
    pause_median_ms: float
    fatigue: float


def sample_params(rng: np.random.Generator, hyper: dict = PLACEHOLDER_HYPER) -> HumanParams:
    return HumanParams(**{k: float(rng.uniform(*v)) for k, v in hyper.items()})


def fit_hyper(real_streams: pd.DataFrame) -> tuple[dict | None, dict]:
    """Estimate the hyper-distribution from real human data, or (None, info) if not enough."""
    h = real_streams[real_streams["label"].isin(C.HUMAN_LABELS) & ~real_streams["held"].astype(bool)]
    iv = h["iv_ms"].dropna()
    act = h[(h["iv_ms"] > 40) & (h["iv_ms"] < C.PAUSE_MS)]
    info = {"active_intervals": int(len(act)), "participants": int(h["participant"].nunique()),
            "min_required": MIN_FIT_INTERVALS}
    if len(act) < MIN_FIT_INTERVALS:
        return None, info
    rows = []
    for _, g in act.groupby("participant"):
        x = np.log(g["iv_ms"].to_numpy())
        rows.append({"median_ms": float(np.exp(np.median(x))), "sigma": float(np.std(x, ddof=1)),
                     "rho": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 3 else 0.3})
    f = pd.DataFrame(rows)
    pause_frac = float((iv >= C.PAUSE_MS).mean()) if len(iv) else 0.05
    def rng_of(col, lo, hi):
        m, s = f[col].mean(), (f[col].std(ddof=1) if len(f) > 1 else abs(f[col].mean()) * 0.15)
        return (float(max(lo, m - 1.5 * s)), float(min(hi, m + 1.5 * s)))
    hyper = dict(PLACEHOLDER_HYPER)
    hyper["median_ms"] = rng_of("median_ms", 45, 400)
    hyper["sigma"] = rng_of("sigma", 0.05, 1.0)
    hyper["rho"] = rng_of("rho", -0.5, 0.9)
    bout = 1.0 / max(pause_frac, 1e-3)
    hyper["bout_mean_clicks"] = (0.7 * bout, 1.3 * bout)
    info.update({"fitted_per_participant": rows})
    return hyper, info


def human_click_times(p: HumanParams, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """True click times (ms) of one combat-style session."""
    T = duration_s * 1000.0
    t, out, dev = rng.uniform(0, 1000), [], 0.0
    clicks_left = rng.geometric(1.0 / p.bout_mean_clicks)
    while t < T:
        out.append(t)
        clicks_left -= 1
        if clicks_left <= 0:
            t += float(rng.lognormal(np.log(p.pause_median_ms), 0.6))
            clicks_left = rng.geometric(1.0 / p.bout_mean_clicks)
            dev = 0.0
            continue
        dev = p.rho * dev + np.sqrt(1 - p.rho ** 2) * rng.normal(0, p.sigma)
        med = p.median_ms * (1 + p.fatigue * t / T)
        t += max(35.0, med * np.exp(dev))  # physiological floor for one finger
    return np.asarray(out)


def mining_events(duration_s: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-button mining in TRUE time: (dig_start_times, held_swing_times, break_times).

    While the button is held the client sends one swing per client tick. Dig durations are
    assumptions (1-12 ticks, i.e. instant-ish to slow blocks); between digs the player either keeps
    holding (next block starts at the next tick) or releases for a lognormal gap.
    """
    T = duration_s * 1000.0
    t = rng.uniform(0, 1000)
    starts, swings, brk = [], [], []
    while t < T:
        starts.append(t)
        ticks = int(rng.integers(1, 13))
        sw = t + C.TICK_MS * np.arange(ticks + 1)
        swings.extend(sw.tolist())
        brk.append(sw[-1] + 1.0)
        if rng.random() < 0.6:
            t = sw[-1] + C.TICK_MS
        else:
            t = sw[-1] + float(rng.lognormal(np.log(350), 0.5))
    return np.asarray(starts), np.asarray(swings), np.asarray(brk)


def to_recorder_frames(session_id: str, label: str, swings_ms: np.ndarray,
                       rng: np.random.Generator, net: NetParams,
                       dig_starts_ms: np.ndarray | None = None,
                       breaks_ms: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the observation model and emit clicks.csv/breaks.csv-shaped frames.

    All events of a session go through ONE observation call (same tick phase, order kept).
    Combat clicks (no dig) are preceded by an interact_air event (assumption: the server-side
    ray trace misses a block); dig starts are an interact_block just before the first swing.
    """
    mining = dig_starts_ms is not None and len(dig_starts_ms) > 0
    ev = [(t, 3, "swing") for t in np.asarray(swings_ms)]
    if mining:
        ev += [(t - 0.5, 0, "interact_block") for t in dig_starts_ms]
    else:
        ev += [(t - 0.3, 1, "interact_air") for t in np.asarray(swings_ms)]
    if breaks_ms is not None:
        ev += [(t, 2, "break") for t in breaks_ms]
    ev.sort(key=lambda e: (e[0], e[1]))
    obs = observe(np.array([e[0] for e in ev]), rng, net) if ev else np.array([])
    kinds = np.array([e[2] for e in ev])
    keep = obs >= 0
    obs, kinds = obs[keep], kinds[keep]
    is_click = kinds != "break"
    clicks = pd.DataFrame({"session_id": session_id,
                           "t_ns": (obs[is_click] * 1e6).astype("int64"),
                           "source": kinds[is_click], "label": label})
    b = pd.DataFrame({"session_id": session_id, "t_ns": (obs[~is_click] * 1e6).astype("int64"),
                      "label": label})
    return clicks, b


def generate_humans(n_participants: int, rng: np.random.Generator, net: NetParams,
                    hyper: dict | None = None, combat_sessions: int = 3, combat_s: float = 180.0,
                    mining_sessions: int = 1, mining_s: float = 90.0, prefix: str = "SH"):
    """Synthetic human participants. Returns (clicks, breaks, sessions, params) frames."""
    hyper = hyper or PLACEHOLDER_HYPER
    cl, br, ss, pr = [], [], [], []
    for i in range(n_participants):
        pid = f"{prefix}{i:03d}"
        p = sample_params(rng, hyper)
        pr.append({"participant": pid, **asdict(p)})
        for j in range(combat_sessions):
            sid = f"{pid}-c{j}"
            c, b = to_recorder_frames(sid, "human_combat", human_click_times(p, combat_s, rng), rng, net)
            cl.append(c); br.append(b)
            ss.append({"session_id": sid, "participant": pid, "label": "human_combat",
                       "duration_s": int(combat_s), "events_dropped": 0, "level": "human"})
        for j in range(mining_sessions):
            sid = f"{pid}-m{j}"
            starts, swings, brk = mining_events(mining_s, rng)
            c, b = to_recorder_frames(sid, "human_mining", swings, rng, net, starts, brk)
            cl.append(c); br.append(b)
            ss.append({"session_id": sid, "participant": pid, "label": "human_mining",
                       "duration_s": int(mining_s), "events_dropped": 0, "level": "human"})
    return _frames(cl, br, ss), pd.DataFrame(pr)


def _frames(cl, br, ss):
    clicks = pd.concat(cl, ignore_index=True)
    breaks = pd.concat([b for b in br if len(b)], ignore_index=True) if any(len(b) for b in br) \
        else pd.DataFrame(columns=["session_id", "t_ns", "label"])
    sessions = pd.DataFrame(ss)
    sessions["start_utc"] = ""
    sessions["end_utc"] = ""
    sessions["activity"] = sessions["label"].map(C.ACTIVITY)
    sessions["origin"] = "synthetic"
    return clicks, breaks, sessions
