"""Non-overlapping market-order backtest using the observed BBO."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_bbo_execution(
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    tick_size: float,
    horizon_events: int,
    min_directional_probability: float = 0.55,
    min_probability_edge: float = 0.10,
    extra_cost_ticks_round_trip: float = 1.0,
) -> pd.DataFrame:
    """Simulate taker entries/exits at subsequent observed quotes.

    Signal at row t enters at row t+1; exit is at row t+1+horizon_events.
    Longs buy the ask and sell the bid; shorts sell the bid and cover at the ask.
    Costs beyond crossing the spread are deducted as a round-trip tick assumption.
    Trades are non-overlapping and this is not a full queue-position simulator.
    """
    if tick_size <= 0 or horizon_events < 1:
        raise ValueError("tick_size and horizon_events must be positive.")
    if not 0.0 <= min_directional_probability <= 1.0:
        raise ValueError("min_directional_probability must be between 0 and 1.")
    if not 0.0 <= min_probability_edge <= 1.0:
        raise ValueError("min_probability_edge must be between 0 and 1.")
    if extra_cost_ticks_round_trip < 0:
        raise ValueError("extra_cost_ticks_round_trip cannot be negative.")

    book = samples.reset_index(drop=True)
    pred = predictions.sort_values("sample_idx").reset_index(drop=True)
    trades: list[dict[str, object]] = []
    next_available_decision = 0

    for row in pred.itertuples(index=False):
        decision_idx = int(row.sample_idx)
        if decision_idx < next_available_decision:
            continue
        entry_idx = decision_idx + 1
        exit_idx = decision_idx + 1 + horizon_events
        if entry_idx >= len(book) or exit_idx >= len(book):
            continue

        p_up = float(row.p_up)
        p_down = float(row.p_down)
        if p_up >= min_directional_probability and p_up - p_down >= min_probability_edge:
            side = 1
        elif p_down >= min_directional_probability and p_down - p_up >= min_probability_edge:
            side = -1
        else:
            continue

        entry = book.iloc[entry_idx]
        exit_ = book.iloc[exit_idx]
        if side == 1:
            entry_price = float(entry["ask_price"])
            exit_price = float(exit_["bid_price"])
            gross_ticks = (exit_price - entry_price) / tick_size
            side_name = "long"
        else:
            entry_price = float(entry["bid_price"])
            exit_price = float(exit_["ask_price"])
            gross_ticks = (entry_price - exit_price) / tick_size
            side_name = "short"

        net_ticks = gross_ticks - extra_cost_ticks_round_trip
        trades.append(
            {
                "decision_idx": decision_idx,
                "decision_timestamp": book.iloc[decision_idx]["timestamp"],
                "entry_timestamp": entry["timestamp"],
                "exit_timestamp": exit_["timestamp"],
                "side": side_name,
                "p_up": p_up,
                "p_down": p_down,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_ticks": gross_ticks,
                "extra_cost_ticks_round_trip": extra_cost_ticks_round_trip,
                "net_ticks": net_ticks,
                "target_move_ticks": getattr(row, "target_move_ticks", np.nan),
            }
        )
        next_available_decision = exit_idx

    result = pd.DataFrame(trades)
    if not result.empty:
        result["cumulative_net_ticks"] = result["net_ticks"].cumsum()
        result["running_peak_ticks"] = result["cumulative_net_ticks"].cummax().clip(lower=0.0)
        result["drawdown_ticks"] = result["cumulative_net_ticks"] - result["running_peak_ticks"]
    return result
