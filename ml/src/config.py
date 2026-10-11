"""Shared constants, paths and seeds for the ML pipeline.

Every constant that also exists in the Java plugin is marked with (JAVA) so the two
implementations can be kept in sync (see plugin ClickFeatureExtractor / ClickStreamCleaner).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent
# Real data location. Overridable with ML_RAW_DIR (run_all.ps1 -RawDir) so the pipeline can
# read data stored outside the repository (the raw folder is git-ignored).
RAW_DIR = Path(os.environ.get("ML_RAW_DIR", ML_DIR / "data" / "raw"))
PROCESSED_DIR = ML_DIR / "data" / "processed"
RESULTS_DIR = ML_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"
PLUGIN_RES_DIR = REPO_DIR / "plugin" / "src" / "main" / "resources"

for _d in (PROCESSED_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = 20261011

# ---------------------------------------------------------------- cleaning (JAVA)
TICK_MS = 50.0              # one game tick at 20 TPS (client and server)
DEDUP_MS = 15.0             # swings closer than this to the last accepted swing are duplicates
HELD_MAX_IV_MS = 65.0       # swing-to-swing interval compatible with "button held" (1 tick + jitter)
FRESH_DIG_MS = 15.0         # a swing within this time of an interact_block belongs to that dig start
RELEASE_GAP_MS = 85.0       # a gap longer than this means the button was released. One late tick
                            # (e.g. 67 ms on a slow server tick) does not end the dig, but a skipped
                            # tick (~100 ms) does, so a clicker alternating 50/100 ms is not held.
PAUSE_MS = 1000.0           # intervals above this are pauses, excluded from distribution stats
TRIM_MOVES_S = 5.0          # spawn / gravity-drop trim at the start of each session (moves)

# ---------------------------------------------------------------- windows (JAVA)
WINDOW_CLICKS = 50
STRIDE_CLICKS = 25
WINDOW_S = 5.0
STRIDE_S = 2.5
MIN_INTERVALS_PER_WINDOW = 10   # time windows with fewer usable intervals are discarded
HELD_ROUTE_FRAC = 0.5           # windows with >= this fraction of held swings are routed to "mining"
TICK_NEAR_MS = 5.0              # |interval - k*50ms| <= this counts as "near a tick multiple"
HIST_EDGES_MS = [float(x) for x in range(0, 525, 25)]  # 20 bins of 25 ms on [0, 500) + overflow

# ---------------------------------------------------------------- ORIGINAL thesis thresholds
# Annex I: flag if diff > 0 && diff < (1000 / maxClicksPerSecond) with integer division -> 66 ms
ORIG_MAX_CPS = 15
ORIG_CLICK_MIN_DIFF_MS = 1000 // ORIG_MAX_CPS        # = 66 (Java integer division)
ORIG_FASTBREAK_MIN_MS = 450                          # config.yml min_delay_ms
ORIG_SPEED_TEXT = 1.2                                # blocks/tick, thesis text (4.6.1)
# config.yml ships 0.1 but loadConfigValues clamps to [0.5, 5.0] -> effective 0.5
ORIG_SPEED_EFFECTIVE = 0.5
ORIG_FLY_DY = 1.0
ORIG_MAX_VIOLATIONS = 15

# ---------------------------------------------------------------- labels
LEGACY_LABELS = {"human": "human_unspecified"}
HUMAN_LABELS = {"human_combat", "human_mining", "human_explore", "human_unspecified"}
CLICK_AUTO_LABELS = {"auto_fixed", "auto_jitter"}
MOVE_CHEAT_LABELS = {"speed_mod", "fly_mod"}
ALL_LABELS = HUMAN_LABELS | CLICK_AUTO_LABELS | MOVE_CHEAT_LABELS

ACTIVITY = {
    "human_combat": "combat", "human_mining": "mining", "human_explore": "explore",
    "human_unspecified": "unspecified", "auto_fixed": "auto", "auto_jitter": "auto",
    "speed_mod": "movement_cheat", "fly_mod": "movement_cheat",
}

# Banner used on every figure/table that involves synthetic humans
DEMO_TAG = "DEMO - synthetic, not evidence about real players"
