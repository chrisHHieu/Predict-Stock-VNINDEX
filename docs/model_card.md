# Model card: VNINDEX next-session direction

## Overview

| Field | Value |
| --- | --- |
| Model artifact | `models/lightgbm/20261031T121542Z` (`models/latest`) |
| Model type | LightGBM classifier (direction) and LightGBM regressor (next-session log return) |
| Task | Binary up/down for session t+1, plus a regression variant of the next-session log return |
| Input | 18 numeric features built from daily OHLCV bars, at least 60 bars |
| Output | Probability of an up session, predicted log return, long/flat decision at the stored threshold |
| Owner | Hoang Hieu Nguyen |
| License | MIT for the code; market data belongs to its source |

## Intended use

Research and engineering demonstration: a reproducible walk-forward pipeline, a
versioned artifact, a serving contract and a drift report. It is a reference
implementation of evaluation discipline for a near-unforecastable target.

## Out of scope

This model is not investment advice, not an execution system, and not validated
for any capital allocation. It trades a single index proxy, long only, with a
fixed 10 basis point cost assumption and no tax, slippage, borrow or liquidity
model. It has no demonstrated edge over the baselines in this repository.

## Data

Daily VNINDEX index history from the public VNDirect dchart endpoint:

```
https://dchart-api.vndirect.com.vn/dchart/history?symbol=VNINDEX&resolution=D&from=<epoch>&to=<epoch>
```

2292 raw observations from 2017-08-24 to 2026-10-30. One row (2020-09-29) is
dropped at clean time because the upstream close sat outside the day low/high.
The processed frame has 2291 rows. The feature frame has 2231 labelled rows,
2017-11-16 to 2026-10-29, with an up share of 0.5603. The raw payload SHA256 is
`c40f4eedd9870a865a65a04d95b04b8f85d7280361c1e419681ecdec2df3d2f2`.

## Features

Log returns at lags 1, 2, 3, 5 and 10; rolling mean and volatility of the log
return over 5, 20 and 60 sessions; RSI(14); ATR(14) normalised by close;
close / MA5, MA20 and MA60 expressed as a ratio minus one; volume z-score over
20 sessions; day of week. All features at row t use data up to the close of t.
Targets are `next_return = log(close[t+1] / close[t])` and
`direction = 1(next_return > 0)`; flat sessions are dropped.

## Training procedure

Walk-forward with an expanding window, minimum 500 training rows and a refit
every 21 sessions, giving 83 refits on 1731 out-of-sample sessions. No shuffling
and no random split anywhere. The served threshold is selected once, on the most
recent 252 training rows, by maximising net strategy return; during the
backtest each fold selects its own threshold on the last 126 rows of its
training window.

LightGBM settings: 300 trees, learning rate 0.03, 15 leaves, minimum 40 samples
per leaf, subsample 0.8, column sample 0.8, L2 penalty 1.0, seed 42.

Chronos-2 settings: `amazon/chronos-2`, 512-session context, one-step horizon, 21
quantile levels, batch size 32, CPU. Weights are downloaded on first use from the
Hugging Face cache; the run recorded here spent 29.3 s loading and 195.2 s on
inference for 1731 points, 112.8 ms per point. The model is zero shot: no
fine-tuning on VNINDEX, no covariate, no per-fold recalibration.

Hyperparameters can be searched with Optuna on the first 500 rows only (five
expanding blocks, so the search never touches the out-of-sample window). A tuned
run is written to `configs/tuned.yaml`; the published comparison above uses
`configs/base.yaml` so that the numbers stay comparable to the earlier report.
Every training run with `--track` logs params, metrics and the model to MLflow,
registers `vnindex_direction` and moves the `champion` alias to the version with
the better walk-forward accuracy.

## Evaluation

Out of sample from 2019-11-20 to 2026-10-29, costs of 10 basis points per
position change.

| Metric | Model | Baseline |
| --- | --- | --- |
| Direction accuracy | 0.5292 | 0.5638 (majority class) |
| ROC AUC | 0.5139 | 0.5000 (constant probability) |
| Brier score | 0.2662 | 0.2463 (constant base rate) |
| MAE next-session log return | 0.008833 | 0.008690 (naive zero) |
| Strategy total return | 0.43% | 74.06% (buy and hold) |
| Annualized Sharpe | 0.004 | 0.397 |
| Max drawdown | -34.02% | -40.34% |
| Hit rate | 56.59% | 56.38% |

Every baseline wins on its own metric. The only column where the model looks
better is drawdown, which is an artefact of being out of the market 35% of the
time, not of timing skill.

### Method comparison, same window and protocol

Run on 2026-10-31 by `python -m vnindex_forecast evaluate`, 1731 sessions, 83
refits, 10 bps per position change.

| Method | Accuracy | AUC | Sharpe | Net return | 80% coverage | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| Naive persistence | 0.5147 | 0.5067 | 0.064 | 6.05% | n/a | 0.0 s |
| Logistic regression | 0.5552 | 0.5191 | 0.278 | 41.01% | n/a | 3.9 s |
| LightGBM (served) | 0.5292 | 0.5139 | 0.044 | 4.95% | 0.7857 | 83.5 s |
| Chronos-2, zero shot | 0.5118 | 0.4952 | -0.229 | -20.96% | 0.7851 | 195.2 s |

The foundation model is the most expensive method and the worst of the four on
this data. It was left in the comparison exactly as measured. The logistic
regression is the best method on every column, which is a statement about the
predictability of daily index direction, not a reason to deploy a linear model
for its own sake.

### Interval forecasts

The conformal band for the next-session log return uses the model's own quantiles
as the nonconformity score, calibrated on the previous 126 sessions. At a nominal
80% it covers 0.7857 of the 1731 walk-forward rows with an average width of
0.0266 log points. On the last 252 sessions, coverage is 0.7262: the band is
calibrated on older volatility and is too narrow after a regime shift. Report the
rolling coverage with the band, never the nominal level alone.

## Calibration

The calibration figure shows predicted probabilities from 0.17 to 0.83 while
the observed up share stays between 0.40 and 0.60 in every bin. The model is
overconfident at both ends and close to useless in the middle. Do not read the
probability as a calibrated probability of an up session.

## Monitoring

`python -m vnindex_forecast drift` computes PSI per feature between the earlier
70% of the feature history and the last 252 sessions. The current run reports 7
stable, 7 moderate and 4 significant features; `atr_14_norm` (PSI 3.60) and
`ret_std_60` (PSI 2.45) show that the 2025-2026 volatility regime barely
overlaps the 2017-2024 reference. PSI is unbounded when reference bins receive
no current observations, so read values above 1.0 as separation rather than as a
calibrated distance.

`drift --evidently` additionally writes `reports/drift_evidently.html` with
per-column statistical tests and a data-drift summary, so the hand-written PSI is
cross-checked by a maintained library instead of being the only signal.

## Ethical and practical notes

Index direction is close to a random walk. Publishing a model like this without
its baseline would be misleading. The artifact metadata records the seed,
config hash, data SHA256 and git commit so any number here can be reproduced or
contradicted.

## Reproduction

Run on 2026-10-31 on Windows 11, Python 3.12, CPU only:

```bash
python -m pip install -e ".[modern,deep]"
python -m vnindex_forecast fetch
python -m vnindex_forecast validate
python -m vnindex_forecast features
python -m vnindex_forecast train --track
python -m vnindex_forecast evaluate
```

The comparison table above comes from that `evaluate` run. It writes
`reports/metrics.json`, `reports/chronos_walk_forward.csv` and the figures. The
Chronos forecast is cached against a hash of the data and the model settings, so
a second `evaluate` reuses it instead of spending another 195 s on the CPU.

Exactly what this card does not claim:

* Not that the model beats buy and hold, or beats a coin flip.
* Not that Chronos-2 helps on this data; it does not.
* Not that the conformal band holds its nominal coverage in a new regime; on the
  last 252 sessions it covered 72.6% at a nominal 80%.
* Not that any of these numbers transfer to intraday data, another index, or a
  tradable instrument with different costs.
