# Methodology and research discipline

## Research question

Does the information in recent best-bid/offer changes and aggressor-side trades add predictive value for the direction of the mid-price move over the next `h` quote events, and does that signal retain positive expectancy after paying the spread and a separate cost allowance?

This is a falsifiable question, not a claim of profitable alpha.

## Features

Let best bid/ask prices be \(P^B_t, P^A_t\), displayed sizes be \(q^B_t, q^A_t\), and mid-price be \(m_t=(P^B_t+P^A_t)/2\).

### Queue imbalance

\[
QI_t = \frac{q^B_t-q^A_t}{q^B_t+q^A_t}
\]

This lies in [-1, 1] when total displayed size is positive. It measures the relative size of the two best queues, not hidden liquidity or queue position.

### Microprice edge

\[
\mu_t = \frac{P^A_t q^B_t + P^B_t q^A_t}{q^B_t+q^A_t}, \quad ME_t=\frac{\mu_t-m_t}{\text{tick size}}
\]

The weighting shifts the estimate toward the opposite quote when one side's displayed queue is larger. It is a feature, not a guaranteed fair value.

### Best-level order-flow imbalance

For consecutive BBO updates, define

\[
 e^B_t=\begin{cases}q^B_t & P^B_t>P^B_{t-1}\\-q^B_{t-1}&P^B_t<P^B_{t-1}\\q^B_t-q^B_{t-1}&P^B_t=P^B_{t-1}\end{cases}
\]

\[
 e^A_t=\begin{cases}-q^A_t & P^A_t<P^A_{t-1}\\q^A_{t-1}&P^A_t>P^A_{t-1}\\q^A_{t-1}-q^A_t&P^A_t=P^A_{t-1}\end{cases},\quad OFI_t=e^B_t+e^A_t
\]

Depth-normalized OFI divides this event imbalance by average current/previous top-level depth. The OFI z-score uses the current OFI relative to rolling moments calculated from prior events only.

This is the BBO event formulation motivated by Cont, Kukanov & Stoikov. It does not include all levels of the book. If input rows are periodically sampled snapshots rather than every BBO update, the feature is the change between snapshots and should not be described as a full message-level OFI series.

### Trade imbalance

Where the feed provides reliable aggressor side, each trade has sign +1 for buy aggressor and -1 for sell aggressor. The rolling signed volume is divided by total trade size. Unknown sides contribute neither signed nor directional volume. A provider's side convention must be verified before normalization.

Other features are spread in ticks, log top-level depth, log event rate, and lagged mid-price returns. Scaling is fitted within each training fold by `StandardScaler` to avoid leaking test distribution moments into training.

## Target and timing

At decision event \(t\), the strategy does not fill at the same row. It enters using the next available BBO quote at \(t+1\) and evaluates the mid-price change through the quote at \(t+1+h\), in ticks. Target labels are down, flat or up according to a configurable neutral band. At default neutral band 0, zero moves belong to flat; positive and negative moves form the directional classes.

The classifier is regularized multinomial logistic regression. Each walk-forward fold trains only on observations whose label horizon has ended before the test fold begins; the `h+1` purge prevents labels from crossing the train/test boundary. Test blocks are predicted once by a model fitted on prior data. The threshold parameters are not optimized inside the test set.

## Execution approximation

- At decision event t, long entry is the ask at t+1 and long exit is the bid at t+1+h.
- Short entry is the bid at t+1 and short cover is the ask at t+1+h.
- The spread is therefore paid directly through the chosen quote sides.
- An additional round-trip ticks cost subtracts commission/slippage/other execution costs beyond spread crossing.
- Trades do not overlap; after a trade exit, the next decision may occur at that exit event or later, but its entry occurs after the previous exit.

This model is still optimistic relative to live execution because it does not model order submission/acknowledgement latency, queue priority, partial fills, market impact, or whether the quote remains available long enough to fill. Sensitivity analysis should vary the added cost and shift execution by additional events.

## Evaluation checklist

1. Verify feed schema, price scale, time ordering, instrument mapping, timestamps, tick size, and market sessions.
2. Compare against prior-probability, always-flat, queue-imbalance-only, and OFI-only baselines.
3. Use a development period and a final untouched holdout. Do not repeatedly tune to the final holdout.
4. Keep all feature transforms and model fits inside each training fold.
5. Purge at least the label horizon between training and test.
6. Evaluate by session, volatility, spread, event intensity, and contract month; include commissions and a cost stress test.
7. Report predictive scores and executable net trade outcomes separately.
8. Use block bootstrap or another dependence-aware uncertainty estimate; event-level outcomes are serially dependent.
9. Report the full number of attempted feature/model/threshold variants and correct for selection where needed.
10. Do not imply statistical significance from a positive backtest point estimate alone.
