# Microstructure Alpha Lab

**A research framework for testing whether limit-order-book events predict short-horizon price moves.** Built for normalized best-bid/offer (BBO) and trade-event data, including data normalized from NQ/MNQ futures feeds.

This is a research project, not a profitable strategy claim, investment recommendation, broker connection, or production execution system. No real-market performance is claimed by this repository. The included generator creates **synthetic test data only** so the pipeline can be exercised without buying data.

## Why this is market microstructure

Unlike a candle indicator, this project works from quote and trade events and computes features from observed bid/ask prices and displayed queues:

- **Order-flow imbalance (OFI):** event-by-event changes in best bid/ask price and displayed size, following the construction studied by Cont, Kukanov & Stoikov.
- **Queue imbalance:** `(bid_size - ask_size) / (bid_size + ask_size)`.
- **Microprice edge:** size-weighted BBO price minus mid-price, measured in ticks.
- **Aggressor-side trade imbalance:** rolling signed trade volume, only when the feed's trade side can be mapped reliably.
- **Depth-adjusted OFI, lagged rolling OFI z-score, spread, total top-level depth, event intensity, and lagged mid-price returns.**

A regularized multinomial logistic regression estimates **down / flat / up** mid-price movement over a configurable event horizon. An expanding walk-forward evaluation uses a purge gap for overlapping labels and fits feature scaling on the training fold only. The execution simulation enters on the next available quote, buys at the ask / sells at the bid, exits using the opposite side of the BBO, subtracts a configurable extra round-trip cost, and avoids overlapping simulated positions.

The implementation uses real quote/trade fields once you supply real data. It does **not** infer order flow from candle bodies. The default data schema is BBO (top of book), not a reconstructed full-depth L2/L3 book; this distinction is intentional and documented.

## Research references

1. Cont, R., Kukanov, A. & Stoikov, S. (2014), “The Price Impact of Order Book Events,” *Journal of Financial Econometrics*, 12(1), 47–88. [Paper / abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822). Their results motivate event-level OFI as a candidate explanatory variable; they do not validate this model or this strategy.
2. Gould, M. D. & Bonart, J. (2016), “Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book.” [arXiv paper](https://arxiv.org/abs/1512.03492). This motivates queue imbalance as a testable feature, not a guaranteed edge across markets.
3. [Databento schema guide](https://databento.com/docs/knowledge-base): describes market-by-price (MBP-1 / MBP-10) and market-by-order (MBO) data schemas. Schema availability and cost depend on the dataset/provider.

## What is included

```text
microstructure-alpha-lab/
├── src/lobalpha/
│   ├── data.py       # normalized data loading and validation
│   ├── features.py   # BBO microstructure features and OFI
│   ├── labels.py     # forward labels aligned after signal time
│   ├── model.py      # purged expanding walk-forward predictions
│   ├── backtest.py   # BBO-crossing, non-overlapping trade simulation
│   ├── metrics.py    # classification and execution diagnostics
│   ├── synthetic.py  # synthetic smoke-test data only
│   └── cli.py        # command-line interface
├── tests/            # feature, validation, leakage, and backtest tests
├── docs/METHODOLOGY.md
├── .github/workflows/tests.yml
├── pyproject.toml
└── Makefile
```

## Requirements

- Python 3.10 or newer.
- Python packages installed by the project (`numpy`, `pandas`, `scikit-learn`).
- Historical event-level quote data and preferably trades for real research. **No exchange data is bundled.**

## Quick start (Windows / Git Bash)

From the project directory:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

If you're using Windows PowerShell rather than Git Bash, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Run the complete smoke test

```bash
lobalpha demo --output data/synthetic_quotes.csv --rows 12000 --seed 7
lobalpha analyze --input data/synthetic_quotes.csv --output-dir reports/demo --tick-size 0.25 --horizon 20 --lookback 100 --min-train-size 2000 --test-block-size 1000
```

This writes synthetic input and diagnostic files under `reports/demo/`. Those metrics only demonstrate that the pipeline executes on generated data. **Never cite synthetic metrics as trading performance or market evidence.**

You can also run the shorter Make target:

```bash
make demo
make analyze
```

## Normalized input data schema

Supply a chronological CSV with these columns:

| Column | Required | Meaning |
|---|---:|---|
| `timestamp` | Yes | Event timestamp; parseable UTC or offset-aware time |
| `bid_price` | Yes | Best bid price, in the instrument's real price units |
| `bid_size` | Yes | Displayed size at best bid |
| `ask_price` | Yes | Best ask price, in the instrument's real price units |
| `ask_size` | Yes | Displayed size at best ask |
| `trade_price` | No | Trade price, if the row represents/includes a trade |
| `trade_size` | No | Trade size; defaults to zero |
| `aggressor_side` | No | `buy`, `sell`, or `unknown`; unknown if side cannot be determined reliably |

Each row should represent a sequenced book update/snapshot or event with the current BBO. Duplicate timestamps can be legitimate if the feed has sequence ordering; preserve original order within ties. Normalize vendor-scaled prices to real price units, normalize aggressor side carefully, remove feed reset/corrupt records, and verify the sequence numbers and timezone before running research.

### Mapping provider fields

For providers exposing MBP-1 style records, map the provider's event timestamp, top bid/ask prices, top bid/ask sizes, trade size, and aggressor side into the normalized names above. Databento describes MBP-1 as BBO/top-level events and MBP-10 as top ten price levels; MBO contains order-by-order messages. Field encodings vary by API/client, so check the vendor's documentation rather than blindly renaming columns. For Databento trade-side conventions, confirm their current schema documentation before mapping `Bid` / `Ask` to `buy` / `sell`.

For NQ/MNQ, obtain data through a provider/feed that actually covers the CME futures product and the required historical resolution. The required depth/schema and history may be paid. A chart or OHLCV candle export is **not** a substitute for event-level quote data.

## Run against your data

Example, assuming you've saved normalized data to `data/nq_bbo.csv`:

```bash
lobalpha analyze \
  --input data/nq_bbo.csv \
  --output-dir reports/nq_walkforward \
  --tick-size 0.25 \
  --horizon 20 \
  --lookback 100 \
  --min-train-size 20000 \
  --test-block-size 5000 \
  --min-probability 0.55 \
  --min-probability-edge 0.10 \
  --extra-cost-ticks-round-trip 1.0
```

Parameter values are starting assumptions, not calibrated optimums. `horizon` is an **event count**, not seconds or bars; event timestamps may be irregular. The `extra-cost-ticks-round-trip` setting represents commission and slippage beyond the spread crossing already charged by buying at ask/selling at bid. Convert your venue/broker costs carefully and test a cost range.

## Outputs

- `oos_predictions.csv` / `simulated_trades.csv`: full-model walk-forward predictions and non-overlapping BBO executions.
- `oos_predictions_ofi_only.csv` / `simulated_trades_ofi_only.csv`: OFI-only benchmark.
- `oos_predictions_queue_imbalance_only.csv` / `simulated_trades_queue_imbalance_only.csv`: queue-imbalance-only benchmark.
- `metrics.json`: predictive and execution diagnostics, same-split baseline comparisons, extra-cost sensitivity, model parameters, and assumptions.
- `SUMMARY.md`: readable summary with caveats.

Predictive metrics include accuracy, balanced accuracy, macro F1, log loss and a confusion matrix. Execution metrics include number of trades, win rate after costs, mean/median net ticks, net profit factor, cumulative net ticks and maximum drawdown in ticks. The report compares the full feature model with OFI-only and queue-imbalance-only baselines on the same walk-forward schedule and runs an additional-cost sensitivity analysis. These do not establish an edge by themselves; confidence intervals, stability, broader baselines and parameter-selection bias remain research responsibilities.

## Important limitations (please read)

1. **BBO, not full L2/L3.** Queue imbalance and microprice are top-of-book measures. This version does not reconstruct individual orders, queue priority, cancellations deeper in the book, or full depth curves.
2. **Simple taker-fill approximation.** It crosses the observed spread but does not model feed-to-order latency, queue position, partial fills, exchange matching, market impact, order rejection, or limit-order fill probability.
3. **Event horizon is data-dependent.** Twenty quote events may cover milliseconds in an active period and much longer in quiet markets.
4. **Data quality is decisive.** Incorrect ordering, stale quotes, rollover gaps, missing trade sides, mixed contract months, vendor-scaled prices, and feed resets can invalidate the results.
5. **Model and selection risk.** Logistic regression is a transparent baseline. Try more complex models only with a pre-registered search plan, nested time-series validation, and a multiple-testing correction.
6. **No empirical findings are included.** Synthetic data is provided only to test code paths. You must run on real, licensed data before describing any result as evidence.
7. **Not production trading software.** No broker connectivity, risk controls, order-routing, or live execution is included.

## Good next research extensions

- Add a separate adapter for the exact licensed feed and retain event sequence numbers / exchange timestamps.
- Add deeper MBP-10 features: multi-level depth imbalance, depth slope, book resiliency, and cancellations/additions by level where records support them.
- Add latency-aware and queue-aware execution simulation, including conservative order acknowledgement delay.
- Compare against simple baselines (unconditional class prior, queue imbalance alone, OFI alone, and always-flat after costs).
- Add block bootstrap confidence intervals, session/time-of-day controls, calibration diagnostics, and regime-sliced OOS metrics.
- Replicate a published OFI result before evaluating a new alpha hypothesis.

## GitHub portfolio framing

Resume-safe description (after you have run the tests):

> Built a Python limit-order-book research pipeline with event-level order-flow imbalance and queue-imbalance features, purged walk-forward validation, multiclass price-move forecasting, and bid/ask-aware cost-adjusted trade simulation.

Only add performance claims after running on real data and documenting the dataset period, instrument/contract, feed schema, costs, and out-of-sample methodology. It is better to show a clean negative result than to present an unverified backtest as profitable.

## License

MIT for the project code. This license does not apply to third-party market data; follow the market-data provider's terms.
