"""Phase D - model definitions.

Every detector exposes the same interface:
    det = make_detector(name); det.fit(train_windows); det.score(windows) -> np.ndarray
Higher score = more suspicious. Thresholds are NOT set here (see evaluate.calibrate).

Detectors
  baseline_orig_raw   ORIGINAL thesis rule on raw LEFT_CLICK_AIR/BLOCK events, no de-dup (fixed rule)
  baseline_orig_clean ORIGINAL rule (0 < diff < 66 ms) on cleaned active intervals (fixed rule)
  cv_rule             one explainable statistic: score = -cv (calibrated threshold)
  iforest, ocsvm      unsupervised, trained on HUMAN windows only
  rf, xgb             supervised
  cnn1d               optional 1D-CNN on raw interval sequences (needs torch)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from . import config as C
from .features import FEATURES

FIXED_RULES = {"baseline_orig_raw", "baseline_orig_clean"}


def is_attack(df: pd.DataFrame) -> np.ndarray:
    return (~df["label"].isin(C.HUMAN_LABELS)).to_numpy().astype(int)


def X_of(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=np.float32)


class FixedRule:
    """Original thesis rule at window level: number of intervals violating 0 < diff < 66 ms.

    A window is flagged when the count is >= 1 (the plugin raises VL on every violating click).
    """
    def __init__(self, column: str):
        self.column = column
    def fit(self, df):
        return self
    def score(self, df):
        return df[self.column].to_numpy(dtype=float)


class CvRule:
    def fit(self, df):
        return self
    def score(self, df):
        return -df["cv"].to_numpy(dtype=float)


class Unsupervised:
    def __init__(self, kind: str, seed: int):
        if kind == "iforest":
            self.m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
        else:
            self.m = make_pipeline(StandardScaler(), OneClassSVM(nu=0.05, gamma="scale"))
        self.kind = kind
    def fit(self, df):
        h = df[is_attack(df) == 0]
        X = X_of(h)
        if self.kind == "ocsvm" and len(X) > 5000:   # O(n^2) kernel: subsample deterministically
            X = X[np.random.default_rng(C.SEED).choice(len(X), 5000, replace=False)]
        self.m.fit(X)
        return self
    def score(self, df):
        return -self.m.decision_function(X_of(df))


class Supervised:
    def __init__(self, kind: str, seed: int):
        self.kind = kind
        if kind == "rf":
            self.m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                            random_state=seed)
        else:
            from xgboost import XGBClassifier
            self.m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                                   colsample_bytree=0.9, tree_method="hist", random_state=seed,
                                   n_jobs=4, eval_metric="logloss")
    def fit(self, df):
        self.m.fit(X_of(df), is_attack(df))
        return self
    def score(self, df):
        return self.m.predict_proba(X_of(df))[:, 1]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class CNN1D:
    """Small 1D-CNN over the window's active intervals (log-scaled, padded to 49)."""
    L = C.WINDOW_CLICKS - 1

    def __init__(self, seed: int, epochs: int = 8):
        self.seed, self.epochs = seed, epochs

    def _tensor(self, df):
        import torch
        out = np.zeros((len(df), 1, self.L), np.float32)
        for i, s in enumerate(df["seq"]):
            s = np.log(np.clip(np.asarray(s, np.float32), 1, 2000)) / np.log(2000)
            out[i, 0, :min(len(s), self.L)] = s[:self.L]
        return torch.from_numpy(out)

    def fit(self, df):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(32, 1))
        X, y = self._tensor(df), torch.tensor(is_attack(df), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        lossf = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = lossf(self.net(X[idx]).squeeze(1), y[idx])
                loss.backward()
                opt.step()
        return self

    def score(self, df):
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.net(self._tensor(df)).squeeze(1)).numpy()


def make_detector(name: str, seed: int = C.SEED):
    if name == "baseline_orig_raw":
        return FixedRule("naive_java_violations")
    if name == "baseline_orig_clean":
        return FixedRule("clean_violations")
    if name == "cv_rule":
        return CvRule()
    if name in ("iforest", "ocsvm"):
        return Unsupervised(name, seed)
    if name in ("rf", "xgb"):
        return Supervised(name, seed)
    if name == "cnn1d":
        return CNN1D(seed)
    raise ValueError(name)


def available_models(include_cnn: bool = True) -> list[str]:
    m = ["baseline_orig_raw", "baseline_orig_clean", "cv_rule", "iforest", "ocsvm", "rf", "xgb"]
    if include_cnn and torch_available():
        m.append("cnn1d")
    return m
