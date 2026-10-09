import pandas as pd
import pytest

from lobalpha.data import DataValidationError, validate_snapshots
from lobalpha.synthetic import generate_synthetic_quotes


def test_synthetic_quotes_validate_and_sort():
    quotes = generate_synthetic_quotes(rows=400, seed=2)
    quotes = quotes.iloc[::-1].reset_index(drop=True)
    result = validate_snapshots(quotes)
    assert len(result) == 400
    assert result["timestamp"].is_monotonic_increasing
    assert (result["bid_price"] < result["ask_price"]).all()


def test_rejects_crossed_market():
    quotes = generate_synthetic_quotes(rows=300, seed=3)
    quotes.loc[12, "bid_price"] = quotes.loc[12, "ask_price"]
    with pytest.raises(DataValidationError, match="locked/crossed"):
        validate_snapshots(quotes)


def test_rejects_missing_required_field():
    quotes = generate_synthetic_quotes(rows=300, seed=3).drop(columns=["ask_size"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_snapshots(quotes)
