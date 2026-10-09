"""Command-line entrypoints for synthetic smoke tests and walk-forward analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import simulate_bbo_execution
from .data import load_snapshots
from .features import FEATURE_COLUMNS, build_features
from .labels import make_targets
from .metrics import prediction_metrics, trade_metrics, write_json
from .model import walk_forward_predict
from .synthetic import write_synthetic_quotes


def _add_common_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="CSV normalized BBO/trade snapshots")
    parser.add_argument("--output-dir", default="reports/run", help="Directory for CSV/JSON outputs")
    parser.add_argument("--tick-size", type=float, required=True, help="Instrument tick size in price units")
    parser.add_argument("--horizon", type=int, default=20, help="Prediction/holding horizon in quote events")
    parser.add_argument("--lookback", type=int, default=100, help="Rolling feature lookback in events")
    parser.add_argument("--min-train-size", type=int, default=2_000, help="Minimum initial training events")
    parser.add_argument("--test-block-size", type=int, default=1_000, help="Events in each OOS fold")
    parser.add_argument("--min-probability", type=float, default=0.55, help="Minimum long/short probability")
    parser.add_argument("--min-probability-edge", type=float, default=0.10, help="Required P(direction)-P(opposite)")
    parser.add_argument("--extra-cost-ticks-round-trip", type=float, default=1.0, help="Commission + slippage beyond bid/ask crossing, round-trip ticks")
    parser.add_argument("--neutral-band-ticks", type=float, default=0.0, help="Classify moves within +/- this many ticks as flat")
    parser.add_argument("--regularization-c", type=float, default=0.5, help="Logistic-regression inverse regularization strength")


def run_analyze(args: argparse.Namespace) -> dict[str, object]:
    source = load_snapshots(args.input)
    features = build_features(source, tick_size=args.tick_size, lookback=args.lookback)
    targets = make_targets(
        source,
        horizon_events=args.horizon,
        tick_size=args.tick_size,
        neutral_band_ticks=args.neutral_band_ticks,
    )
    samples = pd.concat([features, targets[["target", "target_move_ticks"]]], axis=1)
    samples = samples.dropna(subset=FEATURE_COLUMNS + ["target", "target_move_ticks"]).reset_index(drop=True)
    if len(samples) <= args.min_train_size + args.horizon + 100:
        raise ValueError(
            f"Only {len(samples)} usable samples remain. Need substantially more than "
            f"min_train_size={args.min_train_size}; use more quote events or reduce training size."
        )

    model_specs = {
        "full_model": FEATURE_COLUMNS,
        "ofi_only": ["ofi_zscore"],
        "queue_imbalance_only": ["queue_imbalance"],
    }
    model_predictions: dict[str, pd.DataFrame] = {}
    model_trades: dict[str, pd.DataFrame] = {}
    model_diagnostics: dict[str, dict[str, object]] = {}
    for model_name, columns in model_specs.items():
        model_predictions[model_name] = walk_forward_predict(
            samples,
            feature_columns=columns,
            min_train_size=args.min_train_size,
            test_block_size=args.test_block_size,
            purge_events=args.horizon + 1,
            regularization_c=args.regularization_c,
        )
        model_trades[model_name] = simulate_bbo_execution(
            samples=samples,
            predictions=model_predictions[model_name],
            tick_size=args.tick_size,
            horizon_events=args.horizon,
            min_directional_probability=args.min_probability,
            min_probability_edge=args.min_probability_edge,
            extra_cost_ticks_round_trip=args.extra_cost_ticks_round_trip,
        )
        model_diagnostics[model_name] = {
            "prediction_metrics_oos": prediction_metrics(model_predictions[model_name]),
            "execution_metrics_oos": trade_metrics(model_trades[model_name]),
        }

    predictions = model_predictions["full_model"]
    trades = model_trades["full_model"]
    cost_sensitivity: dict[str, dict[str, object]] = {}
    cost_scenarios = sorted({0.0, 0.5, 1.0, 2.0, 4.0, float(args.extra_cost_ticks_round_trip)})
    for cost_ticks in cost_scenarios:
        sensitivity_trades = simulate_bbo_execution(
            samples=samples,
            predictions=predictions,
            tick_size=args.tick_size,
            horizon_events=args.horizon,
            min_directional_probability=args.min_probability,
            min_probability_edge=args.min_probability_edge,
            extra_cost_ticks_round_trip=cost_ticks,
        )
        cost_sensitivity[str(cost_ticks)] = trade_metrics(sensitivity_trades)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "oos_predictions.csv", index=False)
    trades.to_csv(output / "simulated_trades.csv", index=False)
    model_predictions["ofi_only"].to_csv(output / "oos_predictions_ofi_only.csv", index=False)
    model_predictions["queue_imbalance_only"].to_csv(output / "oos_predictions_queue_imbalance_only.csv", index=False)
    model_trades["ofi_only"].to_csv(output / "simulated_trades_ofi_only.csv", index=False)
    model_trades["queue_imbalance_only"].to_csv(output / "simulated_trades_queue_imbalance_only.csv", index=False)
    metrics = {
        "project": "Microstructure Alpha Lab",
        "input_file": str(Path(args.input)),
        "data_rows_after_validation": int(len(source)),
        "usable_samples": int(len(samples)),
        "tick_size": args.tick_size,
        "horizon_events": args.horizon,
        "lookback_events": args.lookback,
        "walk_forward": {
            "min_train_size": args.min_train_size,
            "test_block_size": args.test_block_size,
            "purge_events": args.horizon + 1,
            "regularization_c": args.regularization_c,
        },
        "execution_assumptions": {
            "entry": "next quote after decision; buy ask / sell bid",
            "exit": "quote horizon_events after entry; sell bid / cover ask",
            "extra_cost_ticks_round_trip": args.extra_cost_ticks_round_trip,
            "overlapping_trades": False,
        },
        "prediction_metrics_oos": prediction_metrics(predictions),
        "execution_metrics_oos": trade_metrics(trades),
        "model_comparison_oos": model_diagnostics,
        "full_model_cost_sensitivity_oos": cost_sensitivity,
        "caveat": "Not a live signal or proof of an edge. Results are only as valid as input data and cost assumptions.",
    }
    write_json(metrics, output / "metrics.json")
    (output / "SUMMARY.md").write_text(_render_summary(metrics), encoding="utf-8")
    return metrics


def _render_summary(metrics: dict[str, object]) -> str:
    pred = metrics["prediction_metrics_oos"]
    trade = metrics["execution_metrics_oos"]
    return "\n".join(
        [
            "# Out-of-Sample Research Summary",
            "",
            "> These figures describe this input file and these assumptions only. They are not evidence of future profitability.",
            "",
            "## Dataset and evaluation",
            f"- Validated source rows: {metrics['data_rows_after_validation']:,}",
            f"- Usable labeled samples: {metrics['usable_samples']:,}",
            f"- Prediction horizon: {metrics['horizon_events']} events",
            f"- Tick size: {metrics['tick_size']}",
            f"- Out-of-sample predictions: {pred['n_oos_predictions']:,}",
            "",
            "## Predictive diagnostics (OOS)",
            f"- Accuracy: {pred['accuracy']:.4f}",
            f"- Balanced accuracy: {pred['balanced_accuracy']:.4f}",
            f"- Macro F1: {pred['macro_f1']:.4f}",
            f"- Multiclass log loss: {pred['multiclass_log_loss']}",
            "",
            "## BBO execution simulation (OOS, full model)",
            f"- Trades: {trade['n_trades']:,}",
            f"- Net mean ticks/trade: {trade['mean_net_ticks']}",
            f"- Total net ticks: {trade['total_net_ticks']}",
            f"- Net win rate: {trade['win_rate_net']}",
            f"- Net profit factor: {trade['profit_factor_net']}",
            f"- Max drawdown (ticks): {trade['max_drawdown_ticks']}",
            "",
            "## Model and feature baselines (same walk-forward design)",
            "| Model | OOS balanced accuracy | OOS macro F1 | Trades | Net mean ticks/trade | Total net ticks |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {name} | {values['prediction_metrics_oos']['balanced_accuracy']:.4f} | "
                f"{values['prediction_metrics_oos']['macro_f1']:.4f} | "
                f"{values['execution_metrics_oos']['n_trades']} | "
                f"{values['execution_metrics_oos']['mean_net_ticks']} | "
                f"{values['execution_metrics_oos']['total_net_ticks']} |"
                for name, values in metrics['model_comparison_oos'].items()
            ],
            "",
            "## Full-model extra-cost sensitivity (round-trip ticks beyond spread)",
            "| Extra cost | Trades | Mean net ticks/trade | Total net ticks |",
            "|---:|---:|---:|---:|",
            *[
                f"| {cost} | {values['n_trades']} | {values['mean_net_ticks']} | {values['total_net_ticks']} |"
                for cost, values in metrics['full_model_cost_sensitivity_oos'].items()
            ],
            "",
            "## Files",
            "- `oos_predictions.csv` and `simulated_trades.csv`: full-model OOS probabilities and executions.",
            "- `oos_predictions_ofi_only.csv` / `simulated_trades_ofi_only.csv`: OFI-only benchmark.",
            "- `oos_predictions_queue_imbalance_only.csv` / `simulated_trades_queue_imbalance_only.csv`: queue-only benchmark.",
            "- `metrics.json`: machine-readable metrics, model comparisons, cost sensitivity and assumptions.",
            "",
            "## Important limitations",
            "This simulator does not model queue position, partial fills, exchange matching priority, latency, market impact, or all exchange fees. Validate the data feed's timestamp ordering, quote quality, roll handling, and event semantics before interpreting results.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lobalpha", description="Limit-order-book alpha research lab")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate synthetic data for a smoke test only")
    demo.add_argument("--output", default="data/synthetic_quotes.csv")
    demo.add_argument("--rows", type=int, default=12_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--tick-size", type=float, default=0.25)

    analyze = sub.add_parser("analyze", help="Run walk-forward OOS analysis")
    _add_common_analyze_args(analyze)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo":
        path = write_synthetic_quotes(args.output, rows=args.rows, seed=args.seed, tick_size=args.tick_size)
        print(f"Wrote SYNTHETIC smoke-test data to {path}. Do not use it as market evidence.")
        return
    metrics = run_analyze(args)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"\nWrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
