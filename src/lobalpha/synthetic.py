"""Deterministic synthetic BBO stream for smoke tests only, never for edge claims."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_quotes(rows: int = 12_000, seed: int = 7, tick_size: float = 0.25) -> pd.DataFrame:
    if rows < 300:
        raise ValueError("rows must be at least 300")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-02T14:30:00Z", periods=rows, freq="100ms")
    mid_ticks = np.zeros(rows, dtype=int)
    mid_ticks[0] = 80_000
    bid_size = np.zeros(rows, dtype=int)
    ask_size = np.zeros(rows, dtype=int)
    spread_ticks = np.ones(rows, dtype=int)
    trade_size = rng.integers(0, 25, size=rows)
    aggressor = np.full(rows, "unknown", dtype=object)

    latent = 0.0
    for i in range(rows):
        latent = 0.92 * latent + rng.normal(0.0, 0.28)
        if i > 0:
            prob_up = 1.0 / (1.0 + np.exp(-0.55 * latent))
            move = rng.choice([-1, 0, 1], p=[0.16 + 0.12 * (1 - prob_up), 0.56, 0.16 + 0.12 * prob_up])
            mid_ticks[i] = mid_ticks[i - 1] + int(move)
        spread_ticks[i] = int(rng.choice([1, 2, 3], p=[0.82, 0.15, 0.03]))
        base = int(rng.integers(10, 180))
        imbalance = np.tanh(latent + rng.normal(0.0, 0.55))
        bid_size[i] = max(1, int(base * (1.0 + 0.65 * imbalance)))
        ask_size[i] = max(1, int(base * (1.0 - 0.65 * imbalance)))
        if trade_size[i] > 0:
            aggressor[i] = "buy" if rng.random() < 1.0 / (1.0 + np.exp(-latent)) else "sell"

    bid_price = mid_ticks * tick_size
    # Use a half-tick mid for odd one-tick spreads while keeping both quotes on tick grid.
    bid_price = bid_price - np.floor(spread_ticks / 2) * tick_size
    ask_price = bid_price + spread_ticks * tick_size
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price": bid_price,
            "bid_size": bid_size,
            "ask_price": ask_price,
            "ask_size": ask_size,
            "trade_price": (bid_price + ask_price) / 2.0,
            "trade_size": trade_size,
            "aggressor_side": aggressor,
        }
    )
    return frame


def write_synthetic_quotes(path: str | Path, rows: int = 12_000, seed: int = 7, tick_size: float = 0.25) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_quotes(rows=rows, seed=seed, tick_size=tick_size).to_csv(output, index=False)
    return output
