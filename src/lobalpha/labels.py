"""Forward labels, aligned so every target begins after the decision event."""
from __future__ import annotations

import pandas as pd

CLASS_ORDER = ["down", "flat", "up"]


def make_targets(snapshots: pd.DataFrame, horizon_events: int, tick_size: float, neutral_band_ticks: float = 0.0) -> pd.DataFrame:
    """Label the mid-price move from the next quote to `horizon_events` quotes later.

    Using the next quote for entry avoids allowing the decision event to be its own
    fill. The neutral band is specified in ticks and helps suppress tiny/no-change moves.
    """
    if horizon_events < 1:
        raise ValueError("horizon_events must be at least 1.")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")
    if neutral_band_ticks < 0:
        raise ValueError("neutral_band_ticks cannot be negative.")

    mid = (snapshots["bid_price"].astype(float) + snapshots["ask_price"].astype(float)) / 2.0
    entry_mid = mid.shift(-1)
    exit_mid = mid.shift(-(horizon_events + 1))
    move_ticks = (exit_mid - entry_mid) / tick_size
    label = pd.Series("flat", index=snapshots.index, dtype="object")
    label.loc[move_ticks < -neutral_band_ticks] = "down"
    label.loc[move_ticks > neutral_band_ticks] = "up"
    label.loc[move_ticks.isna()] = None
    return pd.DataFrame({"target_move_ticks": move_ticks, "target": label}, index=snapshots.index)
