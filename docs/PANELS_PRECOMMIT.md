# Pre-commitment: six further panels (written 2026-11-19, before any of their data were downloaded)

## Question

The paper's findings rest on one panel of 200 US stocks. Do they hold on other stocks, and do they hold outside
stocks? The cross-asset case study (`docs/CASE_STUDY_PRECOMMIT.md`) had 32 instruments and borrowed a model trained on
US stocks; it could not answer either question with any power. Here each panel trains its own pooled model.

## Panels

Instrument lists, start dates and split dates are in `code/panels/panels.json`, written before any download. Data:
yfinance adjusted closes (free). Membership is mechanical: a listed instrument enters if it has at least `min_train`
returns before `cal0` and at least 250 returns from `test0`. Instruments that fail are dropped and counted.

| panel | what | listed | train | calibration | test |
|---|---|---|---|---|---|
| `us_largecap` | US large-cap stocks | 230 | 2005 to 2013 | 2014 to 2015 | 2016-01 to 2026-11 |
| `intl_largecap` | large caps in the UK, Germany, France, Japan, Canada, Australia, Hong Kong; local-currency returns | 350 | same | same | same |
| `etf_equity` | US-listed equity ETFs: broad, style, sector, country | 113 | same | same | same |
| `etf_other` | US-listed bond, credit, commodity and currency ETFs | 32 | same | same | same |
| `fx` | currency pairs against the dollar (pegged currencies not listed) | 29 | same | same | same |
| `crypto` | coins with history from 2018 or earlier | 27 | to 2020 | 2021 | 2022-01 to 2026-11 |

Known limits, accepted in advance. The lists are today's large, liquid instruments, so every panel is tilted to
survivors. The US panel overlaps the paper's CRSP names but not its data source or its test window. Currency data from
this source contain bad ticks; a price that moves more than 3% and reverses by more than 3% the next day is removed and
counted. The international panel pools seven markets with different holidays; inference is by date.

## Protocol

A strict calendar split, the same dates for every name in a panel, so no training date is a test date. Every model is
fitted once on data before `test0` and frozen. Per name: GARCH(1,1)-t, GJR-GARCH-skew-t, Beta-t-EGARCH and SAV-CAViaR on
all data before `test0`. Pooled within the panel: gradient-boosted residual quantiles (the paper's hyperparameters and
features) on the GARCH-t residuals and, separately, on the Beta-t-EGARCH residuals, trained on rows before `cal0`; the
pooled GPD tail at 2.5% on the same rows; and the returns-space neural quantile network of `tools/r70_neural_quantile.py`
(training before `cal0`, early stopping on the calibration window). No conformal step. The score is the trailing 63-day
excess kurtosis of the GARCH-t residuals, lagged one day; deciles are pooled over the panel's test rows.

## Measures and inference

Primary measure: pinball loss at the 2.5% level. Secondary: pinball at 1%, the eleven-level average used in the paper,
and FZ0 at both levels for the models that give an ES. Edges are relative to GARCH-t unless stated. Per-date
Newey-West(10) Diebold-Mariano statistics. Each hypothesis is tested once per panel; Holm across the six panels.

## Hypotheses (each graded per panel on the primary measure)

- **W (where).** The pooled trees' edge over GARCH-t in the top score decile is positive with DM > 1.96, and exceeds
  their edge in deciles 1 to 8.
- **Y (why).** In the top decile Beta-t-EGARCH with a Student-t tail beats GARCH-t (DM > 1.96), and the pooled trees on
  the GARCH-t scale do not beat Beta-t-EGARCH there (DM < 1.96).
- **C (class).** SAV-CAViaR and the neural network each have a positive top-decile edge over GARCH-t (DM > 1.96), and
  neither beats Beta-t-EGARCH there.
- **S (shape).** On the Beta-t-EGARCH scale the pooled trees' top-decile edge over that scale's Student-t tail is
  positive.
- **G (sign).** The pooled trees' top-decile edge over GARCH-t is larger after a positive largest shock than after a
  negative one.

## Predictions

| panel | W | Y | C | S | G |
|---|---|---|---|---|---|
| `us_largecap` | holds | holds | holds | holds | holds |
| `intl_largecap` | holds | holds | holds | holds | holds |
| `etf_equity` | holds, weaker than stocks | fails | fails | no prediction | no prediction |
| `etf_other` | fails | fails | fails | no prediction | no prediction |
| `fx` | fails | fails | fails | no prediction | no prediction |
| `crypto` | fails | fails | fails | no prediction | no prediction |

## What follows

- W, Y and C hold on both stock panels: the paper states its findings for large-cap stocks in eight markets, not for
  one US panel.
- They fail on a stock panel: the paper says which, and the claim narrows to where it holds.
- They hold on a non-stock panel: the scope sentence widens to that asset class.
- They fail on every non-stock panel, as predicted: the paper states the stock-versus-other-assets contrast as a
  result, with these panels as the evidence.

Nothing in this file may change after the first download. Deviations are recorded below with their reason.

## Deviations

(none yet)
