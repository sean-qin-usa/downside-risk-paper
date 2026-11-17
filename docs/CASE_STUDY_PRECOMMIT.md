# Pre-commitment: cross-asset case study (written 2026-11-17, before the confirmatory data were downloaded)

## Why

Three exploratory case studies (2026-11-16, `code/case_studies/`, results in `results/case_studies/`) put the paper's
estimator on NVIDIA and on 17 instruments across asset classes. They raised a question the paper does not answer:
is the estimator *comparable* to GARCH-t, and to the other standard benchmarks, as a general rule across asset classes,
or only on the CRSP equity panel? The 17 instruments have been looked at, so any rule fixed now would be tested on data
already seen. This document fixes the question, the tests and the decision rules **before** a fresh set of instruments
is downloaded. The 17 are re-run under the same protocol and reported as **exploratory**; only the new set is
**confirmatory**.

## Instruments

- **Confirmatory (15, not yet examined):** stocks MSFT, PFE, WMT, CAT, BA, NFLX; ETFs IWM (small caps), XLU
  (utilities), HYG (high-yield credit), LQD (investment-grade credit), SLV (silver), DBA (agriculture); FX GBPUSD=X,
  AUDUSD=X; crypto ETH-USD.
- **Exploratory (17, already seen):** KO, JNJ, XOM, JPM, AAPL, AMD, TSLA, NVDA, SPY, EEM, TLT, GLD, USO, UNG,
  EURUSD=X, JPY=X, BTC-USD.

Source: yfinance adjusted closes (free). FX bad ticks are removed by the rule already used: a price that moves more
than 3% and reverses by more than 3% the next day. Every removal is logged. An instrument enters a test year only if it
has at least 750 returns before that year.

## Protocol

- Test window 2020-01-02 to 2026-11-13. Every per-instrument model is refit each January 1 on all of that instrument's
  earlier data and frozen within the year.
- Our pooled parts (gradient-boosted residual quantiles, pooled GPD tail) are trained once on the paper's 200-name CRSP
  panel using pre-2020 rows only, with every case-study ticker that is in the panel excluded. No retuning of any kind.
- Levels: VaR and ES at 1% and 2.5%.

## Models

| Role | Model | Scored on |
|---|---|---|
| benchmark | historical simulation, 500 days | FZ0, pinball |
| benchmark | EWMA (RiskMetrics, λ = 0.94, Gaussian) | FZ0, pinball |
| benchmark | GARCH(1,1)-t | FZ0, pinball |
| benchmark | GJR-GARCH(1,1)-skew-t (Hansen) | FZ0, pinball |
| benchmark | FHS (GARCH-t filter, empirical residuals) | FZ0, pinball |
| benchmark | GARCH-EVT (McNeil-Frey: GARCH-t filter, GPD on the lowest 10% of residuals, closed-form ES) | FZ0, pinball |
| benchmark | Beta-t-EGARCH (Student-t tail) | FZ0, pinball |
| benchmark | SAV-CAViaR (the paper's `frtb_caviar.py` specification) | pinball only (no ES) |
| ours | accuracy layer (the paper's estimator) | FZ0, pinball |
| ours | learned tail on the Beta-t-EGARCH scale, no GPD tail (`body_bteg`) | FZ0, pinball |
| ours | learned tail and GPD tail on the Beta-t-EGARCH scale (`engine_bteg`) | FZ0, pinball |
| ours | score-gated scale (GARCH-t; Beta-t-EGARCH when mk63 exceeds its pre-year 90th percentile) | FZ0, pinball |

## Statistics

- **Pooling.** FZ0 is zero-homogeneous, so a loss *difference* between two models does not depend on an instrument's
  volatility level. For each NYSE trading date (the dates on which SPY has a return), average the difference over the
  instruments with a forecast that date, giving one series; Newey-West(10) variance. Weekend and non-NYSE dates are
  excluded from pooled statistics (the exploratory pooling over all calendar dates over-weighted bitcoin's weekends).
- **Non-inferiority (primary).** "Comparable" means the estimator's pooled mean FZ0 is not worse than the benchmark's
  by more than **δ = 0.01**: the one-sided 95% upper confidence bound of (estimator − benchmark) is below 0.01.
  δ is the size of the estimator's FZ0 advantage over GARCH-t on the paper's panel at 1% (0.011 in the top decile,
  0.009 elsewhere): comparable means it does not give back more than it gains where it wins. δ is unit-free because
  FZ0 is zero-homogeneous.
- **Multiplicity.** The seven FZ0 non-inferiority comparisons at each level (against the seven benchmarks with ES) are
  one family, Holm-adjusted at 5%.
- **Model Confidence Set.** 90% MCS (T_max, stationary bootstrap over dates, mean block 10, B = 1000, seed 0, the
  paper's implementation) on the pooled per-date FZ0 of the 11 models with ES, at each level; and on pooled pinball
  including SAV-CAViaR.
- **Secondary, reported but not deciding:** two-sided superiority DMs; breakdown by asset class; per-instrument DMs
  with Holm across instruments; Kupiec and Christoffersen pass counts; instrument-level sign tests.

## Primary hypotheses (confirmatory set)

- **H1.** The accuracy layer is non-inferior to GARCH-t on FZ0 at both levels.
- **H2.** The accuracy layer is non-inferior to each of the other six FZ0 benchmarks at both levels (Holm).
- **H3.** The accuracy layer is in the 90% FZ0 MCS at both levels.
- **H4.** The accuracy layer is non-inferior to SAV-CAViaR on pinball at both levels (δ = 1% of SAV-CAViaR's mean
  pinball, the pinball analogue).

## Predictions

- H1 holds at both levels.
- H2 holds against HS, EWMA, FHS and GARCH-EVT, and may fail against GJR-skew-t or Beta-t-EGARCH at 1%.
- H3 holds at both levels.
- H4 holds.
- `body_bteg` is not significantly better than the accuracy layer when pooled (two-sided DM below 1.96).

## What follows from each outcome

- **H1 and H3 hold:** the paper may say the estimator is comparable to GARCH-t and in the confidence set of standard
  models across asset classes, citing this study. Otherwise it says the evidence is for US equities, and reports the
  cross-asset result as a limitation.
- **H2 fails against a benchmark:** the paper names that benchmark as better off the equity panel.
- **Recommended configuration:** unchanged unless, on the confirmatory set, `body_bteg` beats the accuracy layer with a
  pooled two-sided DM above 1.96 **and** has the lower mean FZ0 on at least 10 of the 15 instruments, at both levels.
  The exploratory rule (12 of 17) already failed, so this is a second, independent chance and not a lowered bar for
  the same data.

No instrument, model, margin or test may be changed after the confirmatory data are downloaded. Any deviation is
recorded below with its reason.

## Deviations

1. **Implementation details fixed after the confirmatory download, before any model was run on it (2026-11-17).**
   The download printed only date coverage (all 15 instruments have at least 783 returns before 2020).
   - Pinball, unlike FZ0, scales with volatility, so before pooling across instruments each instrument's daily
     pinball is divided by that instrument's mean GARCH-t pinball over the test window at the same level. H4's margin
     is then 1% of SAV-CAViaR's mean normalized pinball.
   - Non-inferiority p-value for each FZ0 comparison: H0 is (estimator − benchmark) ≥ δ; z = (δ − mean)/se with the
     Newey-West(10) se of the pooled series; p = 1 − Φ(z). Holm over the seven benchmarks at each level.
   - MCS: the pooled per-date series of each model's FZ0 (and normalized pinball) averaged over the instruments
     forecasting that date. A model's per-instrument constant cancels in the MCS pairwise differences because every
     model covers the same instrument-dates.
   - One pool of training names serves both sets, excluding every ticker of both sets that is in the panel.
2. **Reporting bug, found after the run (2026-11-17).** The console log `results/case_studies/crossasset.log` printed
   the words "non-inferior" beside every benchmark row whatever the test said. The verdicts in
   `crossasset_results.json` (`noninferior_at_5pct`, `graded`) were computed correctly and are the record; the print
   statement is fixed for future runs. No statistic changed.

## Outcome (confirmatory set, graded 2026-11-17 from `results/case_studies/crossasset_results.json`)

| | 1% | 2.5% |
|---|---|---|
| H1 non-inferior to GARCH-t | **not shown**: estimator lower by 0.010, but upper bound +0.019 exceeds δ | **held**: lower by 0.010, upper bound +0.003 |
| H2 non-inferior to HS / EWMA / FHS / GARCH-EVT | only EWMA shown | **all four held** |
| H2 non-inferior to GJR-skew-t | not shown (estimator higher by 0.014, DM 1.26) | **not shown**: estimator higher by 0.009, two-sided DM 2.06 |
| H2 non-inferior to Beta-t-EGARCH | not shown (wide interval) | not shown (wide interval) |
| H3 in the 90% FZ0 MCS | held (the set keeps all but EWMA) | held (set: GJR-skew-t, Beta-t-EGARCH, estimator, and its three variants) |
| H4 non-inferior to SAV-CAViaR on pinball | not shown | held |
| body_bteg not significantly better than the estimator | held | held |
| Recommended configuration changes | no | no |

Predictions: H1 held at 2.5% and was not shown at 1% (predicted to hold at both). H2 came out exactly as predicted at
2.5% (held against HS, EWMA, FHS, GARCH-EVT; not against GJR-skew-t or Beta-t-EGARCH) and was weaker than predicted
at 1%. H3 held. H4 held at 2.5% only. The body_bteg prediction held.

What the paper may say, under the rules above: at the 2.5% level the estimator is comparable to GARCH-t, FHS, GARCH-EVT
and historical simulation on fifteen instruments outside the equity panel, and sits in the 90% confidence set with
GJR-GARCH-skew-t and Beta-t-EGARCH; at 1% the comparison is inconclusive; off the panel GJR-GARCH-skew-t is the stronger
model at 2.5% (secondary two-sided DM 2.06, not multiplicity-adjusted).

## Post-hoc diagnostic: why GJR-skew-t ties on the panel but leads on the case-study instruments (2026-11-17)

Not part of the confirmatory test. Checked on saved losses first: the gap between the panel result and the case-study
result is not itself significant (z 1.73 at 1%, 1.38 at 2.5%), and restricting the case study to the panel's test window
(2020-07 to 2024-12) makes GJR-skew-t's lead larger, so the 2025–26 period is not the cause. The remaining design
difference is the estimation protocol: on the panel every model is fitted once and frozen; in the case study each
per-instrument model (GJR-skew-t included) is refitted every January while our pooled parts stay frozen at pre-2020.

**Run:** `CS_FROZEN=1 python code/case_studies/crossasset.py` refits nothing after 2020-01-01: every per-instrument
model is fitted once on pre-2020 data and filtered forward, the panel's protocol. Output `crossasset_frozen_results.json`.

**Prediction, written before the run:** if the protocol asymmetry explains the gap, GJR-skew-t's mean advantage over
the estimator across the 32 instruments (instruments as the unit) falls below half of its annual-refit value
(+0.0077 at 1%, +0.0055 at 2.5%) at both levels. If it does not fall, the remaining explanation is the universe
(asset classes and mega-cap stocks outside the panel), checked against the panel's cohort breakdown (r66).

**Outcome (2026-11-17).** Instruments as the unit, all 32 (engine minus GJR-skew-t; positive = GJR-skew-t better):

| protocol | 1% | 2.5% |
|---|---|---|
| annual refits of per-instrument models, pooled parts frozen (the case study as run) | +0.0077 (t 1.30), GJR better on 23/32 | +0.0055 (t 1.79), 18/32 |
| everything fitted once on pre-2020 data (the panel's protocol) | +0.0024 (t 0.34), 20/32 | +0.0023 (t 0.62), 20/32 |

The prediction held at both levels: the lead falls to 31% and 42% of its annual-refit value. Most of GJR-skew-t's
off-panel lead came from the protocol asymmetry, in which the per-instrument benchmark was refitted every year and our
pooled parts were not. Under the panel's protocol the off-panel comparison is a tie, as on the panel. A residual lean of
about +0.002 remains, within noise. The confirmatory verdicts above stand as pre-registered; this diagnostic explains
them and does not replace them. A like-for-like annual comparison would also retrain our pooled parts every year.

## Post-hoc analyses on the saved daily losses, frozen protocol, all 32 instruments (2026-11-17)

Not confirmatory: 17 of the 32 instruments were seen before the pre-commitment and the frozen protocol was adopted
after the first run. No model was refitted. Flagged days are the days the score-gated row switches scale (mk63 above
its pre-2020 90th percentile for that instrument): 10.6% of days.

1. **The post-shock mechanism replicates on stocks and not on other assets.** On flagged days, Beta-t-EGARCH against
   GARCH-t, single-level pinball: 14 stocks +4.8% at 1% (DM 2.5) and +3.4% at 2.5% (DM 2.2); 18 non-stock instruments
   −5.8% (DM −1.6) and −2.0% (DM −1.1). The learned tail on the robust scale (`body_bteg`) on the stocks' flagged days:
   +7.2% (DM 3.3) and +5.0% (DM 2.5), FZ0 DM 2.5 and 2.6. The accuracy layer (GARCH-t scale, pooled GPD tail) shows no
   flagged-day gain on these stocks at the tail levels (0.0% and −0.1%; FZ0 DM −1.9 and −1.8).
2. **Non-inferiority with all 32 pooled (margin 0.01, Holm).** At 2.5% the accuracy layer is non-inferior to every one
   of the seven benchmarks, GJR-skew-t (upper bound +0.0088) and Beta-t-EGARCH (+0.0072) included. At 1% only to EWMA
   (historical simulation p 0.052); the upper bounds against the GARCH family are +0.014 to +0.018.
3. **The score-gated scale does not beat GARCH-t out of sample across the 32:** −0.0038 (DM −1.05) at 1% and −0.0008
   (DM −0.40) at 2.5%. On the stocks alone the flagged-day FZ0 gain is positive and not significant (DM 1.1 and 1.6).
