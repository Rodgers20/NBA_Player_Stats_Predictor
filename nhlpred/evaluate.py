"""Scoring helpers shared by the game and prop backtests."""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p) >= 0.5) == (np.asarray(y) >= 0.5)).mean())


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """Equal-count bins of predicted probability vs. what happened."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    out = []
    for idx in np.array_split(order, bins):
        if len(idx):
            out.append({"predicted": round(float(p[idx].mean()), 4), "actual": round(float(y[idx].mean()), 4), "n": int(len(idx))})
    return out


def top_decile(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = max(1, len(p) // 10)
    top = np.argsort(-p)[:k]
    return {"n": int(k), "predicted": round(float(p[top].mean()), 4), "hit_rate": round(float(y[top].mean()), 4),
            "base_rate": round(float(y.mean()), 4)}


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier(y, p), 4)}
