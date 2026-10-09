import pandas as pd
import pytest

from lobalpha.data import validate_snapshots
from lobalpha.features import build_features
from lobalpha.synthetic import generate_synthetic_quotes


def test_queue_imbalance_and_microprice_direction():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC"),
            "bid_price": [100.0] * 120,
            "bid_size": [90.0] * 120,
            "ask_price": [100.25] * 120,
            "ask_size": [10.0] * 120,
            "trade_size": [1.0] * 120,
            "aggressor_side": ["buy"] * 120,
        }
    )
    data = validate_snapshots(quotes)
    features = build_features(data, tick_size=0.25, lookback=20)
    assert features.loc[50, "queue_imbalance"] == pytest.approx(0.8)
    assert features.loc[50, "microprice_edge_ticks"] > 0


def test_ofi_matches_same_quote_size_change_formula():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=120, freq="s", tz="UTC"),
            "bid_price": [100.0] * 120,
            "bid_size": [10.0] * 120,
            "ask_price": [100.25] * 120,
            "ask_size": [15.0] * 120,
            "trade_size": [0.0] * 120,
            "aggressor_side": ["unknown"] * 120,
        }
    )
    quotes.loc[1, "bid_size"] = 12.0
    quotes.loc[1, "ask_size"] = 10.0
    data = validate_snapshots(quotes)
    features = build_features(data, tick_size=0.25, lookback=20)
    # Bid component = 12 - 10 = +2; ask component = 15 - 10 = +5.
    assert features.loc[1, "ofi"] == pytest.approx(7.0)


def test_rolling_features_are_finite_after_warmup():
    quotes = validate_snapshots(generate_synthetic_quotes(rows=500, seed=12))
    features = build_features(quotes, tick_size=0.25, lookback=50)
    assert features.loc[200, "ofi_zscore"] == features.loc[200, "ofi_zscore"]
    assert features.loc[200, "spread_ticks"] > 0


def test_quote_only_input_keeps_trade_imbalance_neutral_not_missing():
    quotes = generate_synthetic_quotes(rows=300, seed=9).drop(
        columns=["trade_price", "trade_size", "aggressor_side"]
    )
    data = validate_snapshots(quotes)
    features = build_features(data, tick_size=0.25, lookback=20)
    assert (features["trade_imbalance"] == 0.0).all()
    assert features["queue_imbalance"].notna().all()
