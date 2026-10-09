"""Event-level limit-order-book features.

OFI follows the best-quote event construction discussed by Cont, Kukanov and
Stoikov (2014). These are BBO-level features; they are not full depth/L3 features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "spread_ticks",
    "queue_imbalance",
    "microprice_edge_ticks",
    "ofi_depth_normalized",
    "ofi_zscore",
    "trade_imbalance",
    "log_total_depth",
    "log_event_rate",
    "mid_return_1_ticks",
    "mid_return_5_ticks",
]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(10, window // 5)
    prior_mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    prior_std = series.rolling(window, min_periods=min_periods).std(ddof=0).shift(1)
    zscore = (series - prior_mean) / prior_std.replace(0, np.nan)
    # A zero prior standard deviation gives no scale for standardization; use a
    # neutral score, while retaining NaN during the initial warm-up prefix.
    return zscore.mask(prior_std.eq(0), 0.0)


def build_features(snapshots: pd.DataFrame, tick_size: float, lookback: int = 100) -> pd.DataFrame:
    """Construct causal features from normalized best-bid/offer snapshots.

    `snapshots` must already have passed `validate_snapshots`. Rolling
    normalization uses prior events only; no future rows are used by features.
    """
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")
    if lookback < 20:
        raise ValueError("lookback must be at least 20 events.")

    data = snapshots.copy().reset_index(drop=True)
    bid = data["bid_price"].astype(float)
    ask = data["ask_price"].astype(float)
    bid_size = data["bid_size"].astype(float)
    ask_size = data["ask_size"].astype(float)
    depth = bid_size + ask_size
    mid = (bid + ask) / 2.0
    spread = ask - bid

    out = pd.DataFrame(index=data.index)
    out["timestamp"] = data["timestamp"]
    out["bid_price"] = bid
    out["ask_price"] = ask
    out["bid_size"] = bid_size
    out["ask_size"] = ask_size
    out["mid_price"] = mid
    out["spread_ticks"] = spread / tick_size
    out["queue_imbalance"] = ((bid_size - ask_size) / depth.replace(0, np.nan)).fillna(0.0)

    # If bid queue is larger, the microprice shifts toward the ask; this is a
    # BBO-derived estimate, not a guarantee of the next trade direction.
    microprice = (ask * bid_size + bid * ask_size) / depth.replace(0, np.nan)
    out["microprice_edge_ticks"] = ((microprice - mid) / tick_size).fillna(0.0)

    prev_bid = bid.shift(1)
    prev_ask = ask.shift(1)
    prev_bid_size = bid_size.shift(1)
    prev_ask_size = ask_size.shift(1)

    bid_component = np.select(
        [bid > prev_bid, bid < prev_bid],
        [bid_size, -prev_bid_size],
        default=bid_size - prev_bid_size,
    )
    ask_component = np.select(
        [ask < prev_ask, ask > prev_ask],
        [-ask_size, prev_ask_size],
        default=prev_ask_size - ask_size,
    )
    ofi = pd.Series(bid_component + ask_component, index=data.index, dtype=float).fillna(0.0)
    out["ofi"] = ofi
    average_depth = (depth + depth.shift(1)) / 2.0
    ofi_depth = ofi / average_depth.replace(0, np.nan)
    out["ofi_depth_normalized"] = ofi_depth.fillna(0.0).clip(-2.0, 2.0)
    out["ofi_zscore"] = _rolling_zscore(ofi, lookback).clip(-8.0, 8.0)

    side_sign = data["aggressor_side"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
    signed_trade_volume = side_sign * data["trade_size"].astype(float)
    rolling_signed = signed_trade_volume.rolling(lookback, min_periods=max(10, lookback // 5)).sum()
    rolling_total = data["trade_size"].astype(float).rolling(
        lookback, min_periods=max(10, lookback // 5)
    ).sum()
    out["trade_imbalance"] = (rolling_signed / rolling_total.replace(0, np.nan)).fillna(0.0).clip(-1.0, 1.0)

    out["log_total_depth"] = np.log1p(depth)
    elapsed_seconds = data["timestamp"].diff().dt.total_seconds()
    valid_elapsed = elapsed_seconds.where(elapsed_seconds > 0)
    event_rate = (1.0 / valid_elapsed).fillna(0.0)
    out["log_event_rate"] = np.log1p(event_rate.clip(upper=1e6))
    out["mid_return_1_ticks"] = mid.diff(1) / tick_size
    out["mid_return_5_ticks"] = mid.diff(5) / tick_size

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out
