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

1. **Names dropped by the mechanical rule.** `us_largecap`: 230 listed, 201 in the panel (10 with no price file from
   the source, 19 with too few returns before 2014). `intl_largecap`: 350 listed, 338 in the panel (3 with no file, 9
   too short). The other four panels kept every listed instrument. The dropped names and reasons are in each result
   file under `dropped`.
2. **Currency bad ticks.** The 3% spike-and-reverse rule removed 285 prices across the 29 pairs.
3. **International price files contain errors the pre-commitment did not foresee.** Unadjusted splits and
   spin-offs (daily returns of +8,528%, +1,060%, +355% and -78% in four names) and stale feeds (the same close for up
   to 65 consecutive days). In the run as registered they dominate every loss comparison (see
   `docs/SHOCK_RESPONSE_RESULTS.md`, Section 5). A second run applies a screen on the price file alone, switched on by
   `PS_SCREEN=1` in `code/panels/panel_study.py`: a name is dropped if it has a daily return above +100% or below
   -70%, or ten or more consecutive unchanged closes. The screen was chosen after the contaminated run was seen. It
   uses no model output, drops 15 international names, and drops none in `us_largecap`, `etf_equity` or `etf_other`
   (it is not applied to currencies or crypto, where such moves are real). Both runs are committed and both are
   graded below; the screened run is the one to read.
4. **A rerun.** The first `intl_largecap` run was killed with its parent shell before it wrote any output and was
   started again unchanged.
5. **Analyses added after this file.** The closed-form shock response, the SPA tests and the conditional tests on
   these panels (`tools/r71`, `r72`, `r75`) are not part of this pre-commitment. Their predictions are in their own
   script headers.

## Outcome (graded 2026-11-19 on 2.5% pinball loss, as specified)

| panel | names | test rows | W | Y | C | S | G | trees, top decile | trees, deciles 1-8 | Beta-t-EGARCH, top decile | neural network, top decile | trees on Beta-t-EGARCH against its Student-t, top decile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `us_largecap` | 201 | 549333 | holds | holds | holds | holds | holds | +5.1% (8.1) | +0.9% (2.99) | +5.4% (3.35) | +6.0% (9.15) | +0.9% (2.32) |
| `intl_largecap`, as registered | 338 | 923277 | holds | holds | holds | fails | holds | +14.9% (3.11) | +1.2% (3.28) | +19.3% (2.68) | +21.0% (3.2) | -305.5% (-9.07) |
| `intl_largecap`, screened | 323 | 882325 | holds | holds | fails | holds | holds | +4.4% (10.49) | +1.1% (3.0) | +3.4% (2.3) | +6.3% (13.85) | +2.4% (2.61) |
| `etf_equity` | 113 | 308829 | fails | fails | fails | holds | holds | -3.1% (-1.32) | -0.3% (-0.61) | -1.0% (-0.45) | -1.8% (0.73) | +0.3% (-0.16) |
| `etf_other` | 32 | 87456 | fails | fails | fails | holds | fails | +0.8% (-0.79) | +0.3% (1.0) | -2.1% (-0.48) | +0.5% (-0.57) | +3.4% (0.76) |
| `fx` | 29 | 81942 | holds | fails | fails | holds | holds | +1.4% (2.34) | +0.1% (0.88) | -1.7% (-0.63) | +2.4% (3.78) | +3.1% (0.61) |
| `crypto` | 27 | 48006 | holds | holds | holds | fails | holds | +15.4% (3.15) | +1.2% (1.39) | +20.1% (2.18) | +17.1% (3.45) | -0.2% (0.95) |

Edges are over GARCH-t unless stated, with the per-date Newey-West(10) Diebold-Mariano statistic in parentheses.

Against the predictions:

- **`us_largecap`.** All five hold, as predicted.
- **`intl_largecap`.** Predicted: all five hold. Screened run: W, Y, S and G hold; C fails, because the neural
  network beats Beta-t-EGARCH in the top decile (+2.9%, DM 2.05) where the hypothesis said no flexible model would.
  As registered: S fails and the edges are implausibly large, both produced by the data errors of deviation 3.
- **`etf_equity`.** Predicted: W holds but weaker than on stocks, Y and C fail. Outcome: all three fail. The
  prediction for W was wrong; there is no post-shock gain on equity ETFs.
- **`etf_other`.** Predicted: W, Y and C fail. Outcome: all three fail.
- **`fx`.** Predicted: W, Y and C fail. Outcome: W holds (a small gain, +1.4%, DM 2.3); Y and C fail. The prediction
  for W was wrong.
- **`crypto`.** Predicted: W, Y and C fail. Outcome: all three hold, with the largest gains of any panel. The
  predictions were wrong: crypto behaves like single stocks, not like the other non-stock classes.

Holm across the six panels for W (one-sided p-values from the trees' top-decile statistic, screened international
run): `us_largecap`, `intl_largecap`, `crypto` and `fx` remain significant at 5%; the two ETF panels do not.

What follows, by the rules set above. W and Y hold on both stock panels, and C on one of them, so the paper states
the where and the why for large-cap stocks in eight markets and notes that on the international panel one flexible
model also beats the robust filter. W, Y and C hold on crypto, so the scope sentence widens to it. The two ETF
panels fail as a pair, and currencies show only a small gain, so the stock-versus-other contrast is stated as a
result. The closed-form analysis explains the pattern with one number per panel
(`docs/SHOCK_RESPONSE_RESULTS.md`, Table 1).
