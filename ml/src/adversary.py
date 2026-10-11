"""Phase B - synthetic click-automation adversaries (numeric interval sequences only).

This module only produces arrays of timestamps for offline evaluation. It contains no code that
interacts with a game client or injects input.

Levels (increasing sophistication):
  none       fixed CPS (classic autoclicker)
  uniform    fixed mean + uniform jitter
  gaussian   fixed mean + gaussian jitter
  human_like i.i.d. lognormal intervals with the SAME marginal hyper-distribution as the synthetic
             humans (a recorded-and-resampled human distribution without its temporal structure)
  optimized  parameters found by an evolution strategy that minimises the score of a chosen
             detector subject to a minimum click rate (the attacker wants an advantage)

All levels use the same bout structure (the clicker is switched on during fights) and the same
observation model (client-tick quantization + jitter) as the humans, so only the within-bout
timing differs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .observation import NetParams
from .synth_humans import PLACEHOLDER_HYPER, _frames, to_recorder_frames

LEVELS = ["none", "uniform", "gaussian", "human_like", "optimized"]
LABEL_OF = {"none": "auto_fixed"}  # every other level is recorded as auto_jitter


@dataclass
class AttackParams:
    level: str
    median_ms: float            # centre of the interval distribution
    spread: float = 0.0         # uniform half-width (relative) / gaussian sd (relative) / lognormal sigma
    rho: float = 0.0            # AR(1) of log deviations (optimized only)
    pause_prob: float = 0.0     # probability of a short pause after each click (optimized only)
    pause_median_ms: float = 400.0
    bout_s: tuple = field(default=(3.0, 15.0))
    gap_s: tuple = field(default=(0.5, 4.0))


def sample_attack(level: str, rng: np.random.Generator, hyper: dict = PLACEHOLDER_HYPER,
                  opt: dict | None = None) -> AttackParams:
    cps = rng.uniform(8, 20)
    if level == "none":
        return AttackParams(level, 1000 / cps)
    if level == "uniform":
        return AttackParams(level, 1000 / cps, spread=rng.uniform(0.10, 0.50))
    if level == "gaussian":
        return AttackParams(level, 1000 / cps, spread=rng.uniform(0.05, 0.30))
    if level == "human_like":
        return AttackParams(level, rng.uniform(*hyper["median_ms"]), spread=rng.uniform(*hyper["sigma"]))
    if level == "optimized":
        if opt is None:
            raise ValueError("optimized level needs the parameters found by optimize_attack")
        p = {k: float(opt[k]) for k in OPT_BOUNDS}
        # small per-attacker perturbation around the optimum
        p["median_ms"] *= float(np.exp(rng.normal(0, 0.03)))
        p["spread"] *= float(np.exp(rng.normal(0, 0.05)))
        return AttackParams(level="optimized", **p)
    raise ValueError(level)


def _next_interval(p: AttackParams, rng: np.random.Generator, dev: float) -> tuple[float, float]:
    if p.level == "none":
        return p.median_ms, 0.0
    if p.level == "uniform":
        return p.median_ms * (1 + rng.uniform(-p.spread, p.spread)), 0.0
    if p.level == "gaussian":
        return p.median_ms * (1 + rng.normal(0, p.spread)), 0.0
    # lognormal family (human_like: rho = 0; optimized: free rho)
    dev = p.rho * dev + np.sqrt(max(1 - p.rho ** 2, 0)) * rng.normal(0, p.spread)
    iv = p.median_ms * np.exp(dev)
    if p.pause_prob > 0 and rng.random() < p.pause_prob:
        iv += float(rng.lognormal(np.log(p.pause_median_ms), 0.5))
    return iv, dev


def attack_click_times(p: AttackParams, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    T = duration_s * 1000.0
    t, out = rng.uniform(0, 1000), []
    while t < T:
        bout_end = t + rng.uniform(*p.bout_s) * 1000
        dev = 0.0
        while t < min(bout_end, T):
            out.append(t)
            iv, dev = _next_interval(p, rng, dev)
            t += max(1.0, iv)
        t += rng.uniform(*p.gap_s) * 1000
    return np.asarray(out)


def generate_attackers(level: str, n_participants: int, rng: np.random.Generator, net: NetParams,
                       hyper: dict = PLACEHOLDER_HYPER, opt: dict | None = None,
                       sessions: int = 3, duration_s: float = 180.0, prefix: str = "SA"):
    cl, br, ss, pr = [], [], [], []
    label = LABEL_OF.get(level, "auto_jitter")
    for i in range(n_participants):
        pid = f"{prefix}-{level}-{i:03d}"
        p = sample_attack(level, rng, hyper, opt)
        pr.append({"participant": pid, **asdict(p)})
        for j in range(sessions):
            sid = f"{pid}-s{j}"
            c, b = to_recorder_frames(sid, label, attack_click_times(p, duration_s, rng), rng, net)
            cl.append(c); br.append(b)
            ss.append({"session_id": sid, "participant": pid, "label": label,
                       "duration_s": int(duration_s), "events_dropped": 0, "level": level})
    return _frames(cl, br, ss), pd.DataFrame(pr)


# ------------------------------------------------------------------ optimized adversary
OPT_BOUNDS = {
    "median_ms": (40.0, 250.0),
    "spread": (0.0, 0.8),
    "rho": (-0.5, 0.9),
    "pause_prob": (0.0, 0.3),
    "pause_median_ms": (100.0, 800.0),
}


def effective_cps(p: dict, rng: np.random.Generator, n: int = 4000) -> float:
    """Mean click rate within bouts (Monte Carlo, before quantization)."""
    ap = AttackParams(level="optimized", **p)
    dev, tot = 0.0, 0.0
    for _ in range(n):
        iv, dev = _next_interval(ap, rng, dev)
        tot += max(1.0, iv)
    return 1000.0 * n / tot


def optimize_attack(score_fn, min_cps: float, rng: np.random.Generator, generations: int = 12,
                    mu: int = 4, lam: int = 12, eval_sessions: int = 4, duration_s: float = 60.0,
                    net: NetParams | None = None) -> tuple[dict, pd.DataFrame]:
    """(mu, lambda) evolution strategy in a normalised parameter space.

    score_fn(clicks, breaks, sessions) -> mean detector score (lower = less suspicious).
    Candidates violating the click-rate constraint are penalised. Common random numbers
    (same seeds per generation) reduce noise between candidates.
    """
    net = net or NetParams()
    keys = list(OPT_BOUNDS)
    lo = np.array([OPT_BOUNDS[k][0] for k in keys]); hi = np.array([OPT_BOUNDS[k][1] for k in keys])
    def decode(z):
        return {k: float(v) for k, v in zip(keys, lo + np.clip(z, 0, 1) * (hi - lo))}
    def evaluate(z, seed):
        p = decode(z)
        r = np.random.default_rng(seed)
        cps = effective_cps(p, r, 2000)
        cl, br, ss = [], [], []
        for j in range(eval_sessions):
            ap = AttackParams(level="optimized", **p)
            c, b = to_recorder_frames(f"opt-{j}", "auto_jitter", attack_click_times(ap, duration_s, r), r, net)
            cl.append(c); br.append(b)
            ss.append({"session_id": f"opt-{j}", "participant": "opt", "label": "auto_jitter",
                       "duration_s": int(duration_s), "events_dropped": 0, "level": "optimized"})
        s = score_fn(*_frames(cl, br, ss))
        penalty = max(0.0, min_cps - cps) * 0.5
        return s + penalty, s, cps
    pop = rng.uniform(0, 1, size=(lam, len(keys)))
    step = 0.25
    hist = []
    best = (np.inf, None, None, None)
    for g in range(generations):
        seed = int(rng.integers(1 << 31))
        res = [evaluate(z, seed) for z in pop]
        fit = np.array([r[0] for r in res])
        order = np.argsort(fit)
        for rank, i in enumerate(order[:mu]):
            hist.append({"generation": g, "rank": rank, "fitness": fit[i], "score": res[i][1],
                         "cps": res[i][2], **decode(pop[i])})
        if fit[order[0]] < best[0]:
            best = (fit[order[0]], pop[order[0]].copy(), res[order[0]][1], res[order[0]][2])
        parents = pop[order[:mu]]
        pop = np.vstack([parents[rng.integers(mu)] + rng.normal(0, step, len(keys)) for _ in range(lam - 1)]
                        + [best[1]])
        pop = np.clip(pop, 0, 1)
        step *= 0.85
    p = decode(best[1])
    p.update({"opt_score": float(best[2]), "opt_cps": float(best[3]), "min_cps": float(min_cps)})
    return p, pd.DataFrame(hist)
