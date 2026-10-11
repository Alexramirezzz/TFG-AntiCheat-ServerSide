import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import (SchemaError, _coerce, clean_click_stream, naive_intervals, normalise_label,
                      quantization_summary, tick_residuals, trim_moves)


def clicks(events):
    """events: list of (t_ms, source)."""
    return pd.DataFrame({"t_ns": [int(t * 1e6) for t, _ in events],
                         "source": [s for _, s in events]})


def breaks(ts):
    return pd.DataFrame({"t_ns": [int(t * 1e6) for t in ts]})


def test_legacy_label_normalised():
    assert normalise_label("human") == "human_unspecified"
    assert normalise_label("auto_fixed") == "auto_fixed"


def test_one_physical_click_three_events_is_one_click():
    # interact_block + interact_air + swing within 5 ms, twice, 200 ms apart
    ev = [(0, "interact_block"), (2, "interact_air"), (4, "swing"),
          (200, "interact_air"), (203, "swing"), (205, "swing")]
    s = clean_click_stream(clicks(ev))
    assert len(s) == 2
    assert s["n_dup"].tolist() == [0, 1]
    assert s["iv_ms"].iloc[1] == pytest.approx(199)


def test_dedup_is_relative_to_last_accepted_swing():
    s = clean_click_stream(clicks([(0, "swing"), (10, "swing"), (20, "swing"), (31, "swing")]))
    # 10 merged into 0; 20 accepted (20 >= 15); 31 merged into 20
    assert s["t_ms"].tolist() == [0, 20]
    assert s["n_dup"].tolist() == [1, 1]


def test_held_button_mining_run():
    # dig start, swings every tick for 6 ticks, break, next block continues held
    ev = [(1000, "interact_block"), (1002, "swing")]
    ev += [(1002 + 50 * k, "swing") for k in range(1, 7)]
    ev += [(1305, "interact_block"), (1352, "swing"), (1402, "swing")]
    s = clean_click_stream(clicks(ev), breaks([1303]))
    held = s["held"].tolist()
    assert held[0] is False or held[0] == False        # initial press is a real click
    assert all(held[1:7])                              # held swings during the dig
    assert all(held[7:])                               # continues across a broken block
    # release then click in air 400 ms later -> active again
    ev2 = ev + [(1802, "swing"), (1850, "swing")]
    s2 = clean_click_stream(clicks(ev2), breaks([1303]))
    assert s2["held"].tolist()[-2:] == [False, False]


def test_fast_clicking_in_air_is_never_held():
    ev = [(50 * k, "swing") for k in range(20)]
    s = clean_click_stream(clicks(ev))
    assert not s["held"].any()


def test_naive_interact_intervals_floor_ms():
    ev = clicks([(0.0, "interact_block"), (3.7, "interact_air"), (70.2, "swing"), (90.9, "interact_air")])
    iv = naive_intervals(ev.assign(session_id="s"), {"interact_air", "interact_block"})["iv_ms"]
    assert iv.tolist() == [3.0, 87.0]


def test_tick_residuals_and_summary():
    r = tick_residuals(np.array([48, 52, 99, 151, 175]))
    assert np.allclose(r, [-2, 2, -1, 1, -25]) or np.allclose(r, [-2, 2, -1, 1, 25])
    q = quantization_summary(np.array([50, 100, 150, 49, 101] * 10, dtype=float))
    assert q["frac_within_5ms"] == 1.0


def test_trim_moves_relative_to_first_move():
    m = pd.DataFrame({"session_id": "s", "t_ns": [int(10e9), int(14.9e9), int(15e9), int(16e9)]})
    assert len(trim_moves(m)) == 2


def test_instant_break_then_held_swings():
    # pattern observed in the real data: interact_block, break 0.2 ms later, then the click's
    # swing, then one swing per tick while the button stays held
    ev = [(0, "interact_block"), (0.6, "swing"), (50.1, "swing"), (105.4, "swing"),
          (657.0, "interact_block"), (657.6, "swing"), (705.0, "swing")]
    s = clean_click_stream(clicks(ev), breaks([0.2, 657.2]))
    assert s["held"].tolist() == [False, True, True, False, True]


def test_schema_rejects_unknown_label(tmp_path):
    df = pd.DataFrame({"session_id": ["a"], "t_ns": ["1"], "source": ["swing"], "label": ["cheater"]})
    with pytest.raises(SchemaError):
        _coerce(df, "clicks", tmp_path / "clicks.csv")


def test_original_threshold_is_java_integer_division():
    assert C.ORIG_CLICK_MIN_DIFF_MS == 66


def test_late_tick_does_not_end_held_run():
    # sequence seen in the manual server test: dig start, then held swings where one tick
    # arrived 67 ms late (slow server tick) followed by 33, 48, 49, 50, 50 ms
    t, ev = 0.0, [(0.0, "interact_block"), (0.4, "swing")]
    for iv in (67, 33, 48, 49, 50, 50):
        t += iv
        ev.append((0.4 + t, "swing"))
    s = clean_click_stream(clicks(ev))
    assert s["held"].tolist() == [False, False, True, True, True, True, True]


def test_clicker_alternating_ticks_after_block_hit_is_not_held():
    # attacker hits a block once, then clicks at ~13 CPS: observed as 50/100 ms intervals
    ev, t = [(0.0, "interact_block"), (0.4, "swing")], 0.4
    for k in range(20):
        t += 100.0 if k % 2 == 0 else 50.0
        ev.append((t, "swing"))
    s = clean_click_stream(clicks(ev))
    assert not s["held"].any()
