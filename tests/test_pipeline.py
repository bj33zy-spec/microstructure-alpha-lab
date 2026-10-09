import pandas as pd

from lobalpha.backtest import simulate_bbo_execution
from lobalpha.data import validate_snapshots
from lobalpha.features import FEATURE_COLUMNS, build_features
from lobalpha.labels import make_targets
from lobalpha.model import walk_forward_predict
from lobalpha.synthetic import generate_synthetic_quotes


def test_walk_forward_predictions_are_oos_and_backtest_is_non_overlapping():
    raw = validate_snapshots(generate_synthetic_quotes(rows=1800, seed=4))
    features = build_features(raw, tick_size=0.25, lookback=50)
    targets = make_targets(raw, horizon_events=5, tick_size=0.25)
    samples = pd.concat([features, targets], axis=1)
    samples = samples.dropna(subset=FEATURE_COLUMNS + ["target", "target_move_ticks"]).reset_index(drop=True)

    predictions = walk_forward_predict(
        samples,
        feature_columns=FEATURE_COLUMNS,
        min_train_size=400,
        test_block_size=200,
        purge_events=6,
    )
    assert len(predictions) > 0
    assert (predictions["train_end_exclusive"] < predictions["sample_idx"]).all()
    assert (predictions["sample_idx"] - predictions["train_end_exclusive"] >= 6).all()
    assert predictions[["p_down", "p_flat", "p_up"]].notna().all().all()

    trades = simulate_bbo_execution(
        samples,
        predictions,
        tick_size=0.25,
        horizon_events=5,
        min_directional_probability=0.34,
        min_probability_edge=0.0,
        extra_cost_ticks_round_trip=1.0,
    )
    assert trades.empty or (trades["net_ticks"] <= trades["gross_ticks"]).all()
    if len(trades) > 1:
        assert (trades["decision_idx"].iloc[1:].to_numpy() > trades["decision_idx"].iloc[:-1].to_numpy()).all()
