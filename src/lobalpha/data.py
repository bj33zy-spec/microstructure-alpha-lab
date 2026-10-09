"""Input normalization and data-quality checks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "bid_price", "bid_size", "ask_price", "ask_size")
OPTIONAL_DEFAULTS = {
    "trade_price": np.nan,
    "trade_size": 0.0,
    "aggressor_side": "unknown",
}


class DataValidationError(ValueError):
    """Raised when input is inconsistent with the normalized quote schema."""


def load_snapshots(path: str | Path) -> pd.DataFrame:
    """Load normalized BBO/trade events from CSV and validate them.

    Required columns: timestamp, bid_price, bid_size, ask_price, ask_size.
    Optional columns: trade_price, trade_size, aggressor_side (buy/sell/unknown).
    Prices are expressed in the instrument's real price units, not integer-scaled
    vendor encodings. Rows represent chronologically ordered BBO updates/events.
    """
    frame = pd.read_csv(path)
    return validate_snapshots(frame)


def validate_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Return validated, stably time-sorted snapshots with normalized columns."""
    if frame.empty:
        raise DataValidationError("Input contains no rows.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")

    data = frame.copy()
    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in data.columns:
            data[column] = default

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce", format="mixed")
    if data["timestamp"].isna().any():
        bad_rows = data.index[data["timestamp"].isna()].tolist()[:5]
        raise DataValidationError(f"Invalid timestamps at row indices: {bad_rows}")

    numeric = ["bid_price", "bid_size", "ask_price", "ask_size", "trade_price", "trade_size"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    core_numeric = ["bid_price", "bid_size", "ask_price", "ask_size"]
    if not np.isfinite(data[core_numeric].to_numpy(dtype=float)).all():
        raise DataValidationError("Required quote fields contain missing or non-finite values.")
    if (data["bid_price"] <= 0).any() or (data["ask_price"] <= 0).any():
        raise DataValidationError("Bid and ask prices must be positive.")
    if (data["bid_price"] >= data["ask_price"]).any():
        raise DataValidationError("Found locked/crossed quotes; clean or investigate these rows first.")
    if (data[["bid_size", "ask_size", "trade_size"]].fillna(0) < 0).any().any():
        raise DataValidationError("Displayed sizes and trade sizes cannot be negative.")

    data["trade_size"] = data["trade_size"].fillna(0.0)
    data["aggressor_side"] = data["aggressor_side"].fillna("unknown").astype(str).str.lower().str.strip()
    allowed: Iterable[str] = ("buy", "sell", "unknown", "", "none", "nan")
    invalid = sorted(set(data["aggressor_side"]) - set(allowed))
    if invalid:
        raise DataValidationError(
            "aggressor_side must use buy/sell/unknown (normalize vendor codes first); "
            f"found {invalid}."
        )
    data["aggressor_side"] = data["aggressor_side"].replace({"": "unknown", "none": "unknown", "nan": "unknown"})
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)

    if len(data) < 100:
        raise DataValidationError("At least 100 rows are required for meaningful rolling features.")
    return data
