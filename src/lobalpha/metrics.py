"""Prediction and simulated-trade diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix

from .labels import CLASS_ORDER


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    y_true = predictions["target"].astype(str)
    y_pred = predictions["prediction"].astype(str)
    metrics: dict[str, object] = {
        "n_oos_predictions": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "class_order": CLASS_ORDER,
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }
    try:
        probs = predictions[[f"p_{name}" for name in CLASS_ORDER]].to_numpy(dtype=float)
        metrics["multiclass_log_loss"] = float(log_loss(y_true, probs, labels=CLASS_ORDER))
    except (ValueError, KeyError):
        metrics["multiclass_log_loss"] = None
    return metrics


def trade_metrics(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate_net": None,
            "mean_net_ticks": None,
            "median_net_ticks": None,
            "profit_factor_net": None,
            "max_drawdown_ticks": None,
            "total_net_ticks": 0.0,
        }
    net = trades["net_ticks"].astype(float)
    wins = net[net > 0]
    losses = net[net < 0]
    gross_loss_abs = abs(float(losses.sum()))
    profit_factor = float(wins.sum()) / gross_loss_abs if gross_loss_abs > 0 else None
    return {
        "n_trades": int(len(trades)),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "win_rate_net": float((net > 0).mean()),
        "mean_net_ticks": float(net.mean()),
        "median_net_ticks": float(net.median()),
        "profit_factor_net": profit_factor,
        "max_drawdown_ticks": float(trades["drawdown_ticks"].min()),
        "total_net_ticks": float(net.sum()),
        "net_ticks_std": float(net.std(ddof=1)) if len(net) > 1 else 0.0,
    }


def write_json(payload: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
