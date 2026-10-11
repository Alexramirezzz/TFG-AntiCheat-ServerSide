import numpy as np
import pandas as pd

from src.decision import PValue, rule_flags, session_alarms


def test_pvalue_definition():
    p = PValue([1, 2, 3, 4])
    # p = (1 + #cal >= s) / (n + 1)
    assert np.allclose(p([0, 2.5, 4, 10]), [5 / 5, 3 / 5, 2 / 5, 1 / 5])


def test_pvalue_is_uniform_on_its_own_population():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=5000)
    p = PValue(cal)(rng.normal(size=20000))
    assert abs(np.mean(p < 0.01) - 0.01) < 0.003


def test_bonferroni_rule():
    pv = pd.DataFrame({"p_xgb": [0.004, 0.006, 0.5], "p_ocsvm": [0.5, 0.5, 0.004], "p_iforest": 1.0})
    assert rule_flags(pv, "xgb").tolist() == [True, True, False]
    assert rule_flags(pv, "xgb+ocsvm").tolist() == [True, False, True]


def test_k_of_n_session_alarm():
    pv = pd.DataFrame({"session_id": ["a"] * 10 + ["b"] * 10, "y": [0] * 10 + [1] * 10,
                       "window_idx": list(range(10)) * 2})
    flags = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [0, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    s = session_alarms(pv, flags, k=2, n=8).set_index("session_id")
    assert not s.loc["a", "alarm"]               # two flags 9 windows apart: never 2 in 8
    assert s.loc["b", "alarm"] and s.loc["b", "first_alarm_window"] == 3
