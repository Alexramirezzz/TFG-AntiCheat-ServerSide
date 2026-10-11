"""Phase I - decision policy: detector ensemble with p-values, session-level decision and a
conservative mode under lag (DEMO: synthetic humans and synthetic adversaries).

Why: in Phase E an adversary optimised against XGBoost evaded it but was caught by One-Class SVM,
and in Phase F lag raised the false-positive rate. This phase turns both findings into a policy
and measures it.

1. Calibrated p-values. Every detector score s becomes p = (1 + #{calibration human windows with
   score >= s}) / (n + 1): the fraction of human windows that look at least as machine-like.
   It is directly explainable in the audit log.
2. Ensemble. A window is flagged when min_m p_m < ALPHA / M (Bonferroni), which keeps the
   window-level human FPR <= ALPHA for M detectors. Candidates: XGBoost alone (M = 1), and
   XGBoost + one novelty detector (One-Class SVM or Isolation Forest).
   The second detector is chosen by a stated rule: highest recall against an attacker optimised
   against that same ensemble, among the ensembles whose clean human window FPR (cross-validated)
   is <= 1.5% AND whose simulated-lag human FPR at ping <= 80 ms is <= 2%.
   Note: the lag condition was ADDED after the first run of this phase, in which the rule without
   it selected One-Class SVM on all features, which flagged 100% of human windows at 80 ms. The
   *_robust candidates (no network features, see train.NETWORK_FEATURES) were added at the same
   time. Both changes are reported in phase_i_results.md.
3. Session decision. An alarm (one VL in the plugin) is raised when at least K of the last
   N_SESSION windows are flagged. K is chosen on the CALIBRATION humans' sessions (smallest K
   with no false alarm there), never on test data.
4. Conservative mode under lag: ping <= 80 ms normal ALPHA; 80 < ping <= 150 ms ALPHA / 4;
   ping > 150 ms (or low TPS in the plugin) the window is not evaluated (abstention, logged).
   Abstention can be abused by an attacker who fakes lag: it is reported, not hidden.

    python -m src.decision
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import config as C
from .adversary import generate_attackers, optimize_attack
from .adversarial import windows_from_parts
from .data import build_click_streams
from .evaluate import _rate_ci, md_table, prepare_windows, synthetic_windows
from .features import FEATURES
from .lag import PING_CLASSES, apply_lag
from .synthetic import build, calibration_from_real
from .train import is_attack, make_detector

log = logging.getLogger(__name__)
ALPHA = 0.01
N_SESSION = 8
PRIMARY = "xgb"
SECOND_CANDIDATES = ["ocsvm", "iforest", "ocsvm_robust", "iforest_robust"]
TRAIN_ATTACKS = ["none", "uniform", "gaussian"]
EVAL_LEVELS = ["none", "uniform", "gaussian", "human_like"]
MIN_CPS = 12.0
MAX_CLEAN_FPR = 0.015
MAX_LAG80_FPR = 0.02
# strict_alpha_factor: a p-value cannot be smaller than 1 / (n_cal + 1) (~0.0009 with ~1,100 calibration
# windows). With 2 detectors the strict per-detector level is ALPHA * factor / 2, so the factor must keep it
# above that floor: 0.1 gave 0.0005 (impossible to reach, recall 0); 0.25 gives 0.00125.
LAG_POLICY = {"normal_max_ping": 80, "strict_max_ping": 150, "strict_alpha_factor": 0.25}


# ------------------------------------------------------------------ p-values and rules
class PValue:
    def __init__(self, cal_scores):
        self.s = np.sort(np.asarray(cal_scores, dtype=float))

    def __call__(self, scores):
        n = len(self.s)
        ge = n - np.searchsorted(self.s, np.asarray(scores, dtype=float), side="left")
        return (1.0 + ge) / (n + 1.0)


RULES = {"xgb": ["xgb"], **{f"xgb+{m}": ["xgb", m] for m in SECOND_CANDIDATES}}


def rule_flags(pv: pd.DataFrame, rule: str, alpha: float = ALPHA) -> np.ndarray:
    ms = RULES[rule]
    pmin = pv[[f"p_{m}" for m in ms]].min(axis=1).to_numpy()
    return pmin < alpha / len(ms)


def rule_suspicion(pv: pd.DataFrame, rule: str) -> np.ndarray:
    """Smooth objective for the optimiser: max over detectors of -log10(p * M)."""
    ms = RULES[rule]
    return np.max([-np.log10(pv[f"p_{m}"].to_numpy() * len(ms)) for m in ms], axis=0)


class Ensemble:
    """Fitted detectors + p-value tables (from calibration humans)."""
    def __init__(self, fit: pd.DataFrame, cal: pd.DataFrame, seed: int):
        self.dets, self.pv, self.cal_scores = {}, {}, {}
        for m in [PRIMARY] + SECOND_CANDIDATES:
            d = make_detector(m, seed).fit(fit)
            cs = d.score(cal)
            self.dets[m], self.pv[m], self.cal_scores[m] = d, PValue(cs), np.sort(cs)

    def pvalues(self, w: pd.DataFrame) -> pd.DataFrame:
        out = w[["session_id", "participant", "level", "window_idx"]].copy()
        out["y"] = is_attack(w)
        for m, d in self.dets.items():
            out[f"p_{m}"] = self.pv[m](d.score(w))
        return out


def split_fit_cal(win: pd.DataFrame, seed: int):
    hum = win[win["level"] == "human"]
    fi, ci = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                  .split(hum, groups=hum["participant"]))
    fit = pd.concat([hum.iloc[fi], win[win["level"].isin(TRAIN_ATTACKS)]])
    return fit, hum.iloc[ci]


# ------------------------------------------------------------------ session decision
def session_alarms(pv: pd.DataFrame, flags: np.ndarray, k: int, n: int = N_SESSION) -> pd.DataFrame:
    """Per session: alarm if >= k flagged windows among the last n; first alarm position."""
    d = pv[["session_id", "y", "window_idx"]].copy()
    d["f"] = flags.astype(int)
    rows = []
    for sid, g in d.sort_values(["session_id", "window_idx"]).groupby("session_id"):
        f = g["f"].to_numpy()
        roll = np.convolve(f, np.ones(n, int), mode="full")[:len(f)]
        hit = np.flatnonzero(roll >= k)
        rows.append({"session_id": sid, "y": int(g["y"].iloc[0]), "windows": len(f),
                     "alarm": bool(len(hit)), "first_alarm_window": int(hit[0]) + 1 if len(hit) else np.nan})
    return pd.DataFrame(rows)


def choose_k(cal_pv: pd.DataFrame, rule: str) -> tuple[int, dict]:
    """Smallest K with zero false alarms on the calibration humans' sessions."""
    rates = {}
    for k in range(1, N_SESSION + 1):
        s = session_alarms(cal_pv, rule_flags(cal_pv, rule), k)
        rates[k] = float(s["alarm"].mean())
        if rates[k] == 0.0:
            return k, rates
    return N_SESSION, rates


# ------------------------------------------------------------------ experiment
def cv_pvalues(win: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    win = win.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for fold, (tr, te) in enumerate(sgkf.split(win, is_attack(win), win["participant"])):
        fit, cal = split_fit_cal(win.iloc[tr], seed + fold)
        ens = Ensemble(fit, cal, seed + fold)
        p = ens.pvalues(win.iloc[te])
        p["fold"] = fold
        out.append(p)
        log.info("fold %d done", fold)
    return pd.concat(out, ignore_index=True)


def window_table(pv: pd.DataFrame, levels: list[str], rng) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        f = rule_flags(pv, rule)
        r = {"rule": rule}
        hum = pv["y"] == 0
        per = pd.DataFrame({"g": pv.loc[hum, "participant"], "f": f[hum]}).groupby("g")["f"].agg(n="size", k="sum")
        r["fpr"], r["fpr_lo"], r["fpr_hi"] = _rate_ci(per, rng)
        for L in levels:
            m = (pv["level"] == L).to_numpy()
            if m.any():
                per = pd.DataFrame({"g": pv.loc[m, "participant"], "f": f[m]}).groupby("g")["f"].agg(n="size", k="sum")
                r[f"recall[{L}]"], r[f"recall[{L}]_lo"], r[f"recall[{L}]_hi"] = _rate_ci(per, rng)
        rows.append(r)
    return pd.DataFrame(rows)


def optimized_against(ens: Ensemble, rule: str, rng, net, hyper, name: str):
    def score_fn(clicks, breaks, sessions):
        sessions = sessions.copy(); sessions["origin"] = "synthetic"
        w = windows_from_parts([(clicks, breaks, sessions)])
        return float(np.mean(rule_suspicion(ens.pvalues(w), rule))) if len(w) else 0.0
    params, _ = optimize_attack(score_fn, MIN_CPS, rng, generations=12, net=net)
    (parts, _) = generate_attackers("optimized", 24, rng, net, hyper, opt=params, prefix=f"SI{name}")
    parts[2]["level"] = name
    w = windows_from_parts([parts])
    w["level"] = name
    log.info("optimised against %s: %s", rule, {k: round(v, 3) for k, v in params.items()})
    return w, params


def lag_experiment(ens: Ensemble, rules: list[str], rng, net, hyper) -> pd.DataFrame:
    ds, _ = build(seed=C.SEED + 4242, n_humans=20, n_attackers=8, levels=TRAIN_ATTACKS, net=net, hyper=hyper)
    rows = []
    for ping in PING_CLASSES:
        d = apply_lag(ds, rng, ping)
        w = prepare_windows(build_click_streams(d), d.clicks, "count")
        pv = ens.pvalues(w)
        hum = pv["y"].to_numpy() == 0
        for rule in rules:
            for policy in ("always", "conservative"):
                if policy == "always" or ping <= LAG_POLICY["normal_max_ping"]:
                    alpha, covered = ALPHA, True
                elif ping <= LAG_POLICY["strict_max_ping"]:
                    alpha, covered = ALPHA * LAG_POLICY["strict_alpha_factor"], True
                else:
                    alpha, covered = ALPHA, False
                f = rule_flags(pv, rule, alpha) if covered else np.zeros(len(pv), bool)
                rows.append({"ping_ms": ping, "rule": rule, "policy": policy, "coverage": float(covered),
                             "human_windows": int(hum.sum()), "fpr": float(f[hum].mean()),
                             "recall": float(f[~hum].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(C.SEED + 5)
    net, hyper, _ = calibration_from_real()
    win = synthetic_windows("count", EVAL_LEVELS)

    # 1) cross-validated p-values (humans + non-adaptive attackers)
    oof = cv_pvalues(win)
    oof.to_csv(C.PROCESSED_DIR / "decision_oof_pvalues.csv", index=False)

    # 2) final ensemble (all synthetic participants) and adaptive attackers
    fit, cal = split_fit_cal(win, C.SEED)
    ens = Ensemble(fit, cal, C.SEED)
    adaptive, opt_params = [], {}
    for rule in RULES:
        name = f"opt_vs_{rule}"
        w, p = optimized_against(ens, rule, rng, net, hyper, name)
        adaptive.append(ens.pvalues(w)); opt_params[name] = p
    adv = pd.concat(adaptive, ignore_index=True)

    # window-level table: OOF for humans/non-adaptive, final ensemble for adaptive attackers
    t_oof = window_table(oof, EVAL_LEVELS, np.random.default_rng(C.SEED))
    t_adv = window_table(pd.concat([adv, oof[oof["y"] == 0]]), list(adv["level"].unique()),
                         np.random.default_rng(C.SEED)).drop(columns=["fpr", "fpr_lo", "fpr_hi"])
    wt = t_oof.merge(t_adv, on="rule")
    wt.to_csv(C.TAB_DIR / "decision_window_metrics.csv", index=False)

    # 3) lag experiment for every rule, then choose the second detector by the stated rule
    lt = lag_experiment(ens, list(RULES), rng, net, hyper)
    lt.to_csv(C.TAB_DIR / "decision_lag.csv", index=False)
    cand = []
    for second in SECOND_CANDIDATES:
        rule = f"xgb+{second}"
        r = wt[wt["rule"] == rule].iloc[0]
        lag80 = float(lt[(lt["rule"] == rule) & (lt["policy"] == "always") & (lt["ping_ms"] <= 80)]["fpr"].max())
        cand.append({"rule": rule, "clean_fpr": r["fpr"], "max_fpr_ping_le_80": lag80,
                     "recall_vs_self": r[f"recall[opt_vs_{rule}]"],
                     "eligible": bool(r["fpr"] <= MAX_CLEAN_FPR and lag80 <= MAX_LAG80_FPR)})
    cand = pd.DataFrame(cand)
    elig = cand[cand["eligible"]]
    chosen = (elig if len(elig) else cand).sort_values("recall_vs_self", ascending=False)["rule"].iloc[0]
    log.info("chosen ensemble: %s", chosen)

    # 4) session decision: K from calibration humans' sessions
    cal_pv = ens.pvalues(cal)
    k_by_rule = {rule: choose_k(cal_pv, rule) for rule in RULES}
    srows = []
    all_pv = pd.concat([oof, adv], ignore_index=True)
    for rule in RULES:
        k, _ = k_by_rule[rule]
        s = session_alarms(all_pv, rule_flags(all_pv, rule), k)
        s = s.merge(all_pv.groupby("session_id")["level"].first(), on="session_id")
        for lvl, g in s.groupby("level"):
            srows.append({"rule": rule, "k": k, "n": N_SESSION, "level": lvl, "sessions": len(g),
                          "alarm_rate": float(g["alarm"].mean()),
                          "median_windows_to_alarm": float(g["first_alarm_window"].median())})
    st = pd.DataFrame(srows)
    st.to_csv(C.TAB_DIR / "decision_session_metrics.csv", index=False)


    # 6) persist the chosen policy for export_onnx (Java)
    second = chosen.split("+")[1]
    k_chosen, k_rates = k_by_rule[chosen]
    policy = {"rule": chosen, "models": RULES[chosen], "alpha": ALPHA, "session_k": k_chosen,
              "session_n": N_SESSION, "lag_policy": LAG_POLICY,
              "k_selection_calibration_alarm_rates": k_rates,
              "selection": cand.to_dict(orient="records"), "optimized_params": opt_params}
    (C.TAB_DIR / "decision_policy.json").write_text(json.dumps(policy, indent=2, default=float))
    import joblib
    joblib.dump({"ens_dets": {m: ens.dets[m] for m in RULES[chosen]},
                 "cal_scores": {m: ens.cal_scores[m] for m in RULES[chosen]}, "policy": policy,
                 "cal_windows": cal[FEATURES].reset_index(drop=True)},
                C.PROCESSED_DIR / "decision_final.joblib")

    # 7) report
    lv_cols = [c for c in wt.columns if c.startswith("recall[") and not c.endswith(("_lo", "_hi"))]
    lag_piv = lt.pivot_table(index=["rule", "policy"], columns="ping_ms", values="fpr").reset_index()
    lag_piv.columns = [str(c) if not isinstance(c, str) else c for c in lag_piv.columns]
    rec_piv = lt.pivot_table(index=["rule", "policy"], columns="ping_ms", values="recall").reset_index()
    rec_piv.columns = [str(c) if not isinstance(c, str) else c for c in rec_piv.columns]
    rep = [f"# Phase I - decision policy ({C.DEMO_TAG})", "",
           f"p-values from calibration humans; window flagged if min p < {ALPHA} / M (Bonferroni). "
           f"Adaptive attackers: evolution strategy against each rule, >= {MIN_CPS:g} CPS within bouts.", "",
           "## Window level (human FPR cross-validated; adaptive attackers vs the final ensemble)", "",
           md_table(wt, ["rule", "fpr"] + lv_cols),
           "## Choice of the second detector (rule fixed in advance)", "",
           f"Eligible if clean FPR <= {MAX_CLEAN_FPR:.1%} and simulated-lag FPR at ping <= 80 ms <= "
           f"{MAX_LAG80_FPR:.0%}; then highest recall against the attacker optimised against that same "
           f"ensemble. **Chosen: `{chosen}`.**", "",
           "The lag condition and the `*_robust` candidates (novelty detectors without the network features "
           "`tick_frac` and `dup_frac`) were added after a first run in which the rule without the lag condition "
           "chose One-Class SVM on all features; that detector flagged 100% of human windows at a simulated "
           "80 ms ping, because `tick_frac` shifts with network jitter.", "",
           md_table(cand, list(cand.columns)),
           f"## Session decision (alarm if >= K of the last {N_SESSION} windows are flagged)", "",
           "K = smallest value with no false alarm on the calibration humans' sessions "
           f"(rates per K: {json.dumps({r: k_by_rule[r][1] for r in RULES})}).", "",
           md_table(st, ["rule", "k", "level", "sessions", "alarm_rate", "median_windows_to_alarm"]),
           "## Lag (SIMULATED): always evaluate vs conservative mode", "",
           f"Conservative: ping <= {LAG_POLICY['normal_max_ping']} ms normal; <= {LAG_POLICY['strict_max_ping']} ms "
           f"alpha x {LAG_POLICY['strict_alpha_factor']}; above that the window is not evaluated (coverage 0).", "",
           "Human window FPR:", "", md_table(lag_piv, list(lag_piv.columns), fmt="{:.4f}"),
           "Recall on standard attackers:", "", md_table(rec_piv, list(rec_piv.columns)), ""]
    (C.RESULTS_DIR / "phase_i_results.md").write_text("\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
