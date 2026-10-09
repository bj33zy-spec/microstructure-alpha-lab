"""Purged expanding-window model evaluation with train-only scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labels import CLASS_ORDER


def walk_forward_predict(
    samples: pd.DataFrame,
    feature_columns: list[str],
    min_train_size: int = 2_000,
    test_block_size: int = 1_000,
    purge_events: int = 20,
    regularization_c: float = 0.5,
) -> pd.DataFrame:
    """Generate strictly chronological out-of-sample multiclass probabilities.

    For a test block beginning at `start`, model training ends before
    `start - purge_events`, preventing overlapping forward-label windows from
    straddling the train/test boundary. StandardScaler is fitted inside each fold.
    """
    if min_train_size < 100 or test_block_size < 1 or purge_events < 0:
        raise ValueError("Invalid walk-forward sizes: train >= 100, test >= 1, purge >= 0 required.")
    missing = [column for column in feature_columns + ["target"] if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")

    clean = samples.reset_index(drop=True).copy()
    numeric = clean[feature_columns].replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().all(axis=1) & clean["target"].isin(CLASS_ORDER)
    clean = clean.loc[valid].reset_index(drop=True)
    numeric = clean[feature_columns].astype(float)

    predictions: list[pd.DataFrame] = []
    for start in range(min_train_size + purge_events, len(clean), test_block_size):
        train_end = start - purge_events
        if train_end < min_train_size:
            continue
        test_end = min(start + test_block_size, len(clean))
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(start, test_end)
        y_train = clean.loc[train_idx, "target"].astype(str)
        if y_train.nunique() < 2:
            continue

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=regularization_c, max_iter=1_000, solver="lbfgs")),
            ]
        )
        pipeline.fit(numeric.loc[train_idx], y_train)
        model = pipeline.named_steps["model"]
        probabilities = pipeline.predict_proba(numeric.loc[test_idx])

        block = pd.DataFrame(index=test_idx)
        block["sample_idx"] = test_idx
        if "timestamp" in clean.columns:
            block["timestamp"] = clean.loc[test_idx, "timestamp"].to_numpy()
        block["target"] = clean.loc[test_idx, "target"].to_numpy()
        block["target_move_ticks"] = clean.loc[test_idx, "target_move_ticks"].to_numpy() if "target_move_ticks" in clean.columns else np.nan
        block["prediction"] = pipeline.predict(numeric.loc[test_idx])
        for class_name in CLASS_ORDER:
            block[f"p_{class_name}"] = 0.0
        for column_idx, class_name in enumerate(model.classes_):
            block[f"p_{class_name}"] = probabilities[:, column_idx]
        block["train_end_exclusive"] = train_end
        block["test_start_inclusive"] = start
        predictions.append(block.reset_index(drop=True))

    if not predictions:
        raise ValueError(
            "No walk-forward predictions were produced. Provide more data, lower min_train_size, "
            "or verify that target labels contain at least two classes."
        )
    return pd.concat(predictions, ignore_index=True)
