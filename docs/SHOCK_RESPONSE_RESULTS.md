# The shock response of GARCH(1,1): a closed form, a test, and what flexible models learn (results of 2026-11-19)

This file records five analyses run after R67. None of them is in the manuscript yet. Every script carries its
predictions in its header, written before its first run; the grading is in Section 4. All tables in Section 6 are
printed by `tools/make_r71_summary.py` from the committed result files.

| script | what it does | results |
|---|---|---|
| `tools/r71_shock_response.py` | the closed form, the share lambda, shock age, subgroups, earnings | `results/paper/diagnostics/r71/<universe>.json` |
| `tools/r72_spa_mcs.py` | Hansen's SPA test and model confidence sets, after-shock days and other days | `results/paper/diagnostics/r72/` |
| `tools/r73_simulation.py` | the same code path on simulated panels whose process is known | `results/paper/diagnostics/r73_simulation.json` |
| `tools/r75_conditional_tests.py` | coverage and predictive ability conditional on the shock share | `results/paper/diagnostics/r75/` |
| `code/panels/download_earnings.py` | earnings announcement dates (yfinance) | `results/panels/earnings_coverage_*.json` |

Universes: the canonical CRSP panel (200 stocks, 221,600 test rows, frozen fits) and the six panels of
`docs/PANELS_PRECOMMIT.md`, each with its own models and a strict calendar split.

## 1. The identity

GARCH(1,1) sets `s2[t+1] = omega + alpha * e[t]^2 + beta * s2[t]`. Hold every return fixed except the one on day `s`,
and replace `e[s]^2` by its conditional expectation `s2[s]`, as if day `s` had been an ordinary day. The variance `k`
days later falls by

    Delta_k = alpha * beta^(k-1) * s2[s] * (z[s]^2 - 1),        z[s] = e[s] / sqrt(s2[s]).

The recursion is linear in each squared return, so this is exact for the filter. Three consequences:

- **D.** `D = Delta_k / s2[t]` is the share of today's GARCH variance that is there only because day `s` was large. It
  is known the evening before. "The shock" is the largest standardized residual of the last 63 days.
- **Duration.** The filter's response to one shock decays at rate `beta` per day, half-life `log(0.5)/log(beta)`, not
  at `alpha + beta`. How long GARCH's VaR stays too wide is therefore a property of each name's `beta`.
- **A one-parameter family.** If a share `lambda` of the extra variance is real, true variance is
  `s2[t] * (1 - (1 - lambda) * D)`. `lambda = 1` is GARCH; `lambda = 0` says the shock was a one-off. Each value
  gives, row by row and with nothing further to choose, GARCH-t's breach rate, the VaR that would be right, and a
  corrected GARCH-t.

## 2. Findings

**Stocks: about a tenth of a large shock's variance impact is real.** In the top score decile, lambda estimated by
2.5% pinball loss is 0.10 (95% interval 0.05 to 0.15) on the canonical panel and 0.15 (0.10 to 0.25) on US large caps.
The 1% level and the Student-t likelihood give the same values. Lambda estimated on pre-test rows alone is 0.15 on
both. It is the same in each third of the test period (Table 10).

**The closed form predicts GARCH-t's breach rate by shock age.** On the canonical panel the realized 2.5% breach rate
is 0.36% in the first five days after the shock against 0.37% predicted by lambda = 0, and it returns to nominal as
`beta^(k-1)` decays (Table 3). Over all test rows, not only the top decile, the breach rate falls monotonically in D
as predicted (Table 5), from 3.1% where D is under 0.02 to 0.5% where D is over 0.7. The level is offset: GARCH-t
under-covers by 0.6 to 0.8 points on ordinary days in both stock panels, so its unconditional breach rate looks close
to nominal because two errors partly cancel.

**The shock's real effect lasts about a week.** Residuals standardized by the corrected scale have a 2.5% quantile
of about -2.8 in days 1 to 5 (the Student-t value is -1.9 to -2.0), and are at the Student-t value from day 6 on, in
both stock panels. Volatility is elevated for a week by roughly 40%; GARCH-t has it doubled.

**"Three weeks" was a panel average.** How long GARCH-t stays too wide follows each name's `beta` (Table 4). The
canonical panel's median half-life is 3.2 days and its breach rate is back near nominal after about three weeks. US
large caps have a median half-life of 9.6 days and are still at 1.8% in days 22 to 30. Within each panel the
high-beta tercile recovers later than the low-beta tercile, as the formula says.

**It is not specific to a sector, a size or a volatility level.** On the canonical panel lambda is between 0.05 and
0.20 in each of seven SIC groups, both size cohorts and all three volatility terciles, and the breach rate at shock
ages 6 to 15 is between 0.3% and 1.1% in every one (Table 9).

**Flexible models have learned to discard the shock.** Regressing one minus each model's squared VaR ratio to
GARCH-t on D gives the share of the shock's variance the model discards (Table 6): Beta-t-EGARCH 1.00 and 0.81
(canonical, US large caps), pooled trees 0.67 and 0.68, the neural network 0.92 and 0.72, SAV-CAViaR 0.51 and 0.59.
GJR-skew-t discards 0.12 and 0.01, and the accuracy layer (which takes the minimum with a pooled tail) 0.07 and 0.07.
The data say about 0.85 to 0.90.

**A one-parameter correction of GARCH-t matches them after shocks.** With lambda from pre-test rows the corrected
GARCH-t gains 7.9% (DM 10.1) in the top decile of the canonical panel at 2.5%, against 7.7% for Beta-t-EGARCH and 8.2%
for the pooled trees, and 5.0% on US large caps against 5.4% and 5.1% (Table 2). Over all days it matches them on the
canonical panel (+1.5%) and does not on US large caps (-0.1%), where shocks that are not earnings are mostly real.

**Three quarters of the shocks are earnings announcements.** A shock is an earnings shock if it falls on an
announcement date or the next trading day; about 3% of days do. Of top-decile rows, 75.5% on US large caps (201 of 201
names with dates) and 71.7% on the canonical panel (151 of 200) have an earnings shock as their largest (Table 8).
Lambda is 0.10 after earnings shocks on both panels, and 0.35 and 0.20 after other shocks. Among all rows with D above
0.10 on US large caps it is 0.10 after earnings shocks and 1.2 after others. A correction applied only after earnings
shocks, which needs no parameter and nothing that is not on a calendar, beats GARCH-t over all days on both panels
(+0.6%, DM 4.2 and 4.0) and recovers about four fifths of the pooled trees' top-decile gain on US large caps and
about half on the canonical panel.

**Across asset classes lambda orders the results.** Lambda is 0.20 on international large caps (Section 5), 0.05 on crypto, 0.40 on currencies, 0.85 on equity
ETFs and 1.00 on bond, commodity and currency ETFs (Table 1). Flexible models beat GARCH-t after shocks where lambda
is low and do not where it is near one. On equity ETFs lambda is 0.05 after positive shocks and 1.00 after negative
ones.

**Formal tests.** Hansen's test rejects GARCH-t against the three flexible models in the top decile on every stock
panel at 1%, 2.5% and 5% (Table 11). With Beta-t-EGARCH as the benchmark the same three models do not reject it
there; the learned tail on the Beta-t-EGARCH scale does. On equity and other ETFs nothing rejects GARCH-t on any set
of days, and the model confidence set is every model. GARCH-t's hit rate falls with D with a t-statistic of -16.5 on
the canonical panel and -11.0 on US large caps, and is flat on both ETF panels (Table 12). GARCH-EVT, filtered
historical simulation and GJR-skew-t have the same slope as GARCH-t: changing the tail shape leaves the error
untouched, because the error is in the scale. Per-name Kupiec, Christoffersen and dynamic-quantile tests do not
separate the models.

**Simulation.** On simulated panels the estimator returns 1.03 (0.14) for a pure GARCH-t process, -0.02 (0.03) when
jumps do not enter the variance, 0.27 (0.03) when a quarter of each squared jump does, and 0.84 (0.11) when all of it
does (Table 14). Under pure GARCH-t the breach rate is flat in shock age, and both the lambda = 0 correction and
Beta-t-EGARCH lose 9 to 10% in the top decile: a robust filter is right only when shocks are one-offs. The
correction with lambda from pre-test rows is neutral under GARCH and wins under jumps.

## 3. What is new here and what is not

That additive outliers distort GARCH forecasts, and that correcting for them helps, is established (Franses and
Ghijsels 1999; Carnero, Pena and Ruiz 2007; Grane and Veiga 2014; robust filters in Muler and Yohai 2008 and Harvey
2013). That jumps carry little information about later volatility is established in the realized-volatility
literature (Andersen, Bollerslev and Diebold 2007), and that scheduled-announcement shocks do not persist has been
reported for other markets. What this file adds: (i) the attribution of flexible models' VaR gains to this error,
with the share measured; (ii) a closed-form, parameter-free prediction of GARCH-t's breach rate by shock age and by
name, tested; (iii) lambda as one number that orders seven panels in five asset classes; (iv) the earnings share.
The literature check behind this paragraph was a set of targeted searches, not a systematic review; item 1.9 of
`docs/WORKLIST.md` is to complete it.

## 4. Predictions and how they fared

`r71` (P), `r72` (S), `r73` (M), `r75` (T). "Stock panels" are the canonical panel, US large caps and the
international panel; the international entries are in Section 5.

| | prediction | canonical | US large caps |
|---|---|---|---|
| P1 | lambda below 0.5, interval excludes 1 (crypto below 0.5; ETFs and currencies above the stock value) | holds | holds |
| P2 | lambda = 0 correction recovers half of Beta-t-EGARCH's top-decile edge; does not lose over all days | holds (6.5 of 7.7; +1.1%) | first part holds (4.2 of 5.4); over all days -0.5%, DM -0.7 |
| P3 | breach rate within 0.5 points of the lambda = 0 prediction in every D bin | **fails** (level offset of 0.2 to 0.6) | **fails** (offset of 0.5 to 0.8) |
| P4 | discard shares between 0.5 and 1.1 for Beta-t-EGARCH, trees, CAViaR; GJR-skew-t below 0.25 | holds | holds |
| P5 | at ages 22 to 42 the high-beta tercile has the lower breach rate | holds (2.00 against 2.40) | holds (1.56 against 2.33) |
| P6 | canonical subgroups: lambda below 0.5 and breach at ages 6 to 15 below 1.5% in each | holds | n/a |
| P7 | D locates the gain at least as well as the kurtosis score | holds (+12.9% against +8.2%) | **fails** (+4.8% against +5.1%) |
| P8 | at least half of top-decile rows have an earnings shock | holds (71.7%) | holds (75.5%) |
| P9 | lambda lower after earnings shocks; above 0.3 after others | first part holds; second **fails** (0.20) | holds (0.10, 0.35) |
| P10 | correction and trees gain more after earnings shocks | holds | holds |
| P11 | earnings-only correction beats GARCH-t over all days; top-decile edge at least the all-shock correction's | first part holds; second **fails** (3.8% against 4.5%) | holds (4.2% against 4.2%) |
| S1 | SPA rejects GARCH-t against the flexible set in the top decile at all three levels | holds | holds |
| S2 | SPA does not reject GARCH-t in deciles 1 to 8 at 2.5% | **fails** (p < 0.001; the neural network gains 1.3%) | holds (p = 0.053) |
| S3 | flexible set does not reject Beta-t-EGARCH in the top decile; learned tail on it does | holds | holds |
| S4 | MCS excludes GARCH-t, GJR-skew-t, FHS and includes trees on the robust scale | holds | holds |
| S5 | SPA rejects GARCH-t on crypto and currencies, not on the two ETF panels | holds (all four) | |
| S6 | SPA rejects GARCH-t over all days against all models | holds | holds |
| T1 | GARCH-t's hit rate falls with D (t below -5); Beta-t-EGARCH's and the trees' slopes under half of it | first part and Beta-t-EGARCH hold; trees **fail** (63% of GARCH-t's slope) | same (69%) |
| T2 | gains over GARCH-t rise with D (t above 3); with the kurtosis dummy added, D stays and the dummy falls below 2 | gains hold; dummy **fails** (t = 6.8) | Beta-t-EGARCH **fails** (t = 2.8); dummy **fails** (t = 6.1) |
| T3 | on the ETF panels GARCH-t's slope on D is not significantly negative | holds (both) | |
| T4 | per-name Kupiec rejection shares of GARCH-t and Beta-t-EGARCH within 10 points | holds | holds |
| M1 | pure GARCH: lambda-hat within 0.15 of 1, flat breach profile, corrections lose | holds | |
| M2 | one-off jumps: lambda-hat below 0.2, breach under 1% at ages 1 to 10, corrections gain over 5% | holds | |
| M3 | lambda-hat rises with the share that feeds variance | holds | |
| M4 | one-off jumps raise fitted alpha and lower fitted beta | **fails** (alpha falls to 0.04, beta unchanged) | |

What the failures say. P3 and S2: GARCH-t is also wrong on ordinary days, in the other direction and by less, so
the contrast between after-shock days and other days is one of size, not of kind. P7 and T2: D and the kurtosis
score each carry information the other lacks; D is the better locator only on the canonical panel. T1: the pooled
trees on the GARCH scale remove only about a third of the conditional miscoverage even though their loss gain is
large. M4: the low persistence of the canonical panel's fitted GARCH models is not produced by one-off jumps in this
simulation.

## 5. The international panel

The free price files for the 350 listed international stocks contain errors that the pre-commitment did not
foresee: unadjusted splits and spin-offs (one name shows a daily return of +8,528%, others +1,060%, +355% and -78%)
and stale feeds (the same close repeated for up to 65 trading days). In the run as registered (338 names) these rows
dominate every loss comparison: one name's GARCH-t VaR stays near -3,000% for weeks, the pooled tail fitted on the
Beta-t-EGARCH residuals has a shape parameter of 0.49 against 0.21 for the other panels, and the trees on that scale
lose 227% to GARCH-t. Breach rates and lambda, which depend on quantiles and not on loss sizes, are little affected.

The deviation (recorded in `docs/PANELS_PRECOMMIT.md`): a screen on the price file alone, chosen after seeing the
contaminated run but using no model output. A name is dropped if it has a daily return above +100% or below -70%, or
ten or more consecutive unchanged closes. It drops 15 international names and none in the US stock panel or either ETF panel. Both runs
are committed; the tables below show both.

On the screened panel (323 names, 882,325 test rows, seven markets):

- Lambda is 0.20 (0.10 to 0.40). By market it is 0.05 to 0.25 in Australia, France, Germany, Hong Kong, Japan and the
  United Kingdom, and 0.70 (0.15 to 1.50) in Canada.
- GARCH-t's breach rate by shock age matches the lambda = 0 prediction from day 6 on (0.79% against 0.60%, 0.87%
  against 0.86%, 1.10% against 1.15%, 1.41% against 1.50%). In days 1 to 5 it is 1.4% against 0.3%: the first week
  after a shock carries more real volatility than in the US panels.
- Top-decile gains over GARCH-t at 2.5%: pooled trees 4.4% (DM 10.5), neural network 6.3% (13.9), SAV-CAViaR 3.8%
  (3.7), Beta-t-EGARCH 3.4% (2.3), corrected GARCH-t with pre-test lambda 3.5% (4.4), trees on the Beta-t-EGARCH scale
  5.8% (8.3). The gain is larger after positive shocks (5.0% against 3.9%).
- Graded against the predictions above: P1, P2 (first part; over all days the lambda = 0 correction is -0.3%, DM
  -0.4), P5 and P7 hold. P4 fails narrowly for the pooled trees (discard share 0.49). S1, S4 and S6 hold; S2 fails
  (p = 0.002); S3 fails in its first part, because the neural network beats Beta-t-EGARCH after shocks (p = 0.038).
  T1 holds for GARCH-t (t = -8.8) and Beta-t-EGARCH (slope zero) and fails for the trees; T2 holds for the trees and
  the neural network and fails for SAV-CAViaR and Beta-t-EGARCH; T4 holds.

## 6. Tables

### Table 1. The share of a large shock's variance impact that is real (lambda), by universe

| universe | names | rows | median beta (half-life, days) | largest shock, top decile (SD) | lambda, 2.5% pinball [95%] | lambda, 1% pinball | lambda, likelihood | positive shocks | negative shocks | lambda from pre-test rows |
|---|---|---|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | 200 | 221600 | 0.806 (3.2) | 8.2 | 0.10 [0.05, 0.15] | 0.10 [0.05, 0.10] | 0.15 [0.10, 0.20] | 0.10 [0.05, 0.15] | 0.10 [0.05, 0.15] | 0.15 |
| US large caps (201) | 201 | 549333 | 0.930 (9.6) | 7.0 | 0.15 [0.10, 0.25] | 0.15 [0.05, 0.30] | 0.10 [0.05, 0.15] | 0.10 [0.00, 0.10] | 0.25 [0.15, 0.45] | 0.15 |
| International large caps, screened | 323 | 882325 | 0.921 (8.4) | 6.6 | 0.20 [0.10, 0.40] | 0.30 [0.10, 0.75] | 0.10 [0.05, 0.15] | 0.10 [0.05, 0.15] | 0.30 [0.15, 0.70] | 0.30 |
| International large caps, as registered | 338 | 923277 | 0.920 (8.3) | 6.6 | -0.00 [0.00, 0.10] | -0.00 [0.00, 0.25] | 0.10 [0.05, 0.15] | -0.00 [0.00, -0.00] | 0.30 [0.15, 0.68] | 0.25 |
| Crypto (27) | 27 | 48006 | 0.831 (3.8) | 6.2 | 0.05 [0.00, 0.15] | 0.05 [0.00, 0.10] | 0.20 [0.10, 0.30] | 0.05 [0.00, 0.15] | -0.00 [-0.35, 0.05] | -0.00 |
| Currencies (29) | 29 | 81942 | 0.938 (10.9) | 5.0 | 0.40 [0.15, 0.65] | 0.40 [0.20, 0.85] | 0.40 [0.25, 0.60] | 0.35 [0.10, 0.48] | 0.75 [-0.10, 1.10] | 0.45 |
| Equity ETFs (113) | 113 | 308829 | 0.901 (6.6) | 4.9 | 0.85 [0.40, 1.50] | 1.15 [0.45, 1.50] | 0.55 [0.20, 0.85] | 0.05 [-0.10, 0.30] | 1.00 [0.55, 1.50] | 1.50 |
| Bond, commodity, currency ETFs (32) | 32 | 87456 | 0.946 (12.6) | 4.8 | 1.00 [0.30, 1.50] | 1.50 [0.27, 1.50] | 0.70 [0.35, 1.18] | 0.30 [-0.35, 1.50] | 1.30 [0.32, 1.50] | 0.60 |

### Table 2. Top score decile, 2.5% pinball edge over GARCH-t (DM)

| universe | corrected GARCH-t, lambda = 0 | corrected, lambda from pre-test rows | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t |
|---|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | +6.5% (5.54) | +7.9% (10.08) | +7.7% (7.48) | +8.2% (9.37) | +4.5% (3.35) | +8.7% (8.35) | +10.1% (9.3) | +0.7% (1.06) |
| US large caps (201) | +4.2% (2.28) | +5.0% (4.22) | +5.4% (3.35) | +5.1% (8.1) | +4.9% (9.08) | +6.0% (9.15) | +6.3% (5.74) | +0.4% (0.68) |
| International large caps, screened | +2.6% (1.79) | +3.5% (4.42) | +3.4% (2.3) | +4.4% (10.49) | +3.8% (3.72) | +6.3% (13.85) | +5.8% (8.33) | +0.8% (1.51) |
| International large caps, as registered | +17.3% (2.54) | +11.5% (3.09) | +19.3% (2.68) | +14.9% (3.11) | +17.9% (2.89) | +21.0% (3.2) | -227.2% (-8.4) | +12.0% (2.29) |
| Crypto (27) | +17.6% (2.09) | +17.6% (2.09) | +20.1% (2.18) | +15.4% (3.15) | +21.6% (2.55) | +17.1% (3.45) | +19.9% (2.32) | +2.3% (4.14) |
| Currencies (29) | -1.0% (-0.36) | +1.7% (1.72) | -1.7% (-0.63) | +1.4% (2.34) | +1.6% (0.4) | +2.4% (3.78) | +1.5% (-0.06) | -2.3% (-1.81) |
| Equity ETFs (113) | -5.5% (-1.45) | -0.9% (-1.3) | -1.0% (-0.45) | -3.1% (-1.32) | -1.5% (-0.84) | -1.8% (0.73) | -0.7% (-0.75) | -1.0% (0.31) |
| Bond, commodity, currency ETFs (32) | -5.9% (-1.61) | -0.6% (-0.34) | -2.1% (-0.48) | +0.8% (-0.79) | -1.8% (-0.52) | +0.5% (-0.57) | +1.4% (0.27) | -0.9% (-1.68) |

All days:

| universe | corrected GARCH-t, lambda = 0 | corrected, lambda from pre-test rows | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t |
|---|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | +1.1% (4.07) | +1.5% (7.79) | +1.4% (5.47) | +1.6% (5.1) | +0.3% (1.58) | +2.4% (8.35) | +2.5% (9.89) | +0.3% (1.91) |
| US large caps (201) | -0.5% (-0.74) | -0.1% (-0.21) | +0.9% (3.62) | +1.4% (5.4) | +1.1% (5.43) | +2.2% (3.24) | +2.1% (4.1) | +0.8% (1.71) |
| International large caps, screened | -0.3% (-0.44) | +0.2% (0.61) | +0.4% (0.99) | +1.5% (4.94) | +0.4% (1.24) | +2.3% (6.23) | +1.7% (7.35) | +0.9% (2.48) |
| International large caps, as registered | +1.6% (1.51) | +1.2% (1.97) | +2.5% (2.57) | +3.0% (4.33) | +2.5% (2.86) | +4.4% (4.75) | -30.5% (-7.32) | +2.4% (3.24) |
| Crypto (27) | +2.7% (1.42) | +2.7% (1.42) | +2.2% (1.12) | +3.6% (2.78) | +4.3% (2.16) | +4.7% (2.95) | +3.7% (1.78) | +1.5% (3.7) |
| Currencies (29) | -0.1% (-0.36) | +0.4% (2.92) | -1.6% (-4.64) | +0.3% (2.51) | +0.3% (1.02) | +0.9% (3.73) | -0.7% (-2.64) | -0.6% (-3.01) |
| Equity ETFs (113) | -2.5% (-2.27) | +0.3% (0.81) | -0.3% (-0.59) | -0.7% (-0.76) | -0.2% (-0.38) | +1.2% (1.03) | +0.5% (0.69) | +1.9% (1.48) |
| Bond, commodity, currency ETFs (32) | -1.8% (-1.93) | -0.3% (-1.13) | -0.3% (-0.62) | +0.4% (0.93) | -1.1% (-2.66) | +0.3% (0.47) | +0.6% (1.68) | +0.1% (0.13) |

### Table 3. Top score decile by age of the shock: GARCH-t's 2.5% breach rate, realized / predicted by lambda = 0; then the 2.5% quantile of residuals on the corrected scale (Student-t value in the header row of each universe)

| universe | 1-5 | 6-10 | 11-15 | 16-21 | 22-30 | 31-45 | 46-63 |
|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200): breach | 0.36 / 0.37 | 0.51 / 0.96 | 0.93 / 1.45 | 1.50 / 1.82 | 2.42 / 2.10 | 2.34 / 2.29 | 2.81 / 2.39 |
| corrected-scale quantile (t: -1.92) | -2.79 | -1.80 | -1.81 | -1.81 | -2.02 | -1.92 | -2.05 |
| US large caps (201): breach | 0.74 / 0.38 | 0.72 / 0.60 | 0.48 / 0.82 | 1.14 / 1.08 | 1.75 / 1.41 | 1.97 / 1.82 | 2.85 / 2.14 |
| corrected-scale quantile (t: -1.98) | -2.72 | -2.02 | -1.90 | -1.95 | -2.05 | -2.03 | -2.22 |
| International large caps, screened: breach | 1.36 / 0.33 | 0.79 / 0.60 | 0.87 / 0.86 | 1.10 / 1.15 | 1.41 / 1.50 | 2.11 / 1.91 | 2.70 / 2.23 |
| corrected-scale quantile (t: -1.99) | -2.88 | -2.04 | -1.94 | -1.90 | -1.97 | -2.04 | -2.12 |
| International large caps, as registered: breach | 1.44 / 0.36 | 0.78 / 0.63 | 0.82 / 0.88 | 1.19 / 1.17 | 1.44 / 1.51 | 2.12 / 1.92 | 2.68 / 2.23 |
| corrected-scale quantile (t: -1.98) | -2.87 | -2.05 | -1.93 | -1.90 | -1.96 | -2.03 | -2.12 |
| Crypto (27): breach | 0.00 / 0.50 | 0.75 / 1.04 | 0.74 / 1.47 | 1.74 / 1.83 | 2.54 / 2.19 | 1.77 / 2.42 | 1.86 / 2.49 |
| corrected-scale quantile (t: -1.90) | -2.78 | -1.80 | -1.65 | -1.93 | -1.94 | -1.68 | -1.74 |
| Currencies (29): breach | 3.22 / 0.78 | 0.84 / 1.22 | 2.10 / 1.47 | 1.36 / 1.65 | 1.73 / 1.85 | 1.83 / 2.10 | 1.77 / 2.30 |
| corrected-scale quantile (t: -1.95) | -3.85 | -1.89 | -2.20 | -1.82 | -1.94 | -1.87 | -1.88 |
| Equity ETFs (113): breach | 4.79 / 0.49 | 2.17 / 0.97 | 1.03 / 1.38 | 1.17 / 1.70 | 1.94 / 1.99 | 2.48 / 2.26 | 3.02 / 2.43 |
| corrected-scale quantile (t: -2.00) | -3.97 | -2.56 | -1.97 | -1.85 | -1.99 | -2.03 | -2.19 |
| Bond, commodity, currency ETFs (32): breach | 5.26 / 0.69 | 2.24 / 1.08 | 0.32 / 1.34 | 0.39 / 1.57 | 2.77 / 1.83 | 1.61 / 2.07 | 1.87 / 2.28 |
| corrected-scale quantile (t: -1.99) | -4.60 | -2.30 | -1.91 | -1.67 | -2.21 | -1.81 | -1.81 |

### Table 4. Duration and the persistence parameter: GARCH-t's 2.5% breach rate in the top decile by shock age, names split by beta tercile (realized / predicted by lambda = 0)

| universe | tercile | median beta | half-life (days) | 1-10 | 11-21 | 22-42 | 43-63 |
|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | low beta | 0.690 | 1.9 | 0.42 / 0.88 | 1.97 / 2.23 | 2.40 / 2.49 | 3.02 / 2.50 |
| CRSP stocks (canonical, 200) | mid beta | 0.806 | 3.2 | 0.35 / 0.50 | 0.59 / 1.66 | 2.89 / 2.40 | 2.46 / 2.50 |
| CRSP stocks (canonical, 200) | high beta | 0.899 | 6.5 | 0.53 / 0.55 | 0.90 / 1.03 | 2.00 / 1.75 | 2.64 / 2.17 |
| US large caps (201) | low beta | 0.898 | 6.4 | 0.78 / 0.41 | 1.08 / 1.15 | 2.33 / 2.00 | 3.45 / 2.41 |
| US large caps (201) | mid beta | 0.930 | 9.6 | 0.83 / 0.39 | 0.98 / 0.84 | 1.63 / 1.59 | 2.59 / 2.20 |
| US large caps (201) | high beta | 0.957 | 15.7 | 0.60 / 0.64 | 0.54 / 0.92 | 1.56 / 1.37 | 2.51 / 1.86 |
| International large caps, screened | low beta | 0.891 | 6.0 | 0.92 / 0.52 | 1.51 / 1.37 | 2.18 / 2.14 | 2.64 / 2.45 |
| International large caps, screened | mid beta | 0.921 | 8.4 | 1.06 / 0.38 | 0.76 / 0.91 | 1.94 / 1.71 | 2.74 / 2.28 |
| International large caps, screened | high beta | 0.944 | 12.0 | 1.16 / 0.50 | 0.82 / 0.86 | 1.46 / 1.42 | 2.48 / 1.98 |
| International large caps, as registered | low beta | 0.888 | 5.8 | 0.83 / 0.53 | 1.44 / 1.37 | 2.24 / 2.15 | 2.64 / 2.45 |
| International large caps, as registered | mid beta | 0.920 | 8.3 | 1.10 / 0.39 | 0.77 / 0.91 | 1.92 / 1.72 | 2.72 / 2.29 |
| International large caps, as registered | high beta | 0.944 | 12.0 | 1.28 / 0.56 | 0.92 / 0.90 | 1.48 / 1.44 | 2.48 / 1.99 |
| Crypto (27) | low beta | 0.800 | 3.1 | 1.13 / 0.87 | 2.06 / 1.89 | 1.94 / 2.42 | 2.23 / 2.50 |
| Crypto (27) | mid beta | 0.831 | 3.8 | 0.00 / 0.71 | 0.85 / 1.63 | 2.97 / 2.34 | 0.91 / 2.49 |
| Crypto (27) | high beta | 0.884 | 5.6 | 0.00 / 0.76 | 0.89 / 1.40 | 1.36 / 2.18 | 2.41 / 2.46 |
| Currencies (29) | low beta | 0.862 | 4.7 | 1.64 / 1.15 | 1.79 / 1.84 | 1.82 / 2.22 | 1.77 / 2.43 |
| Currencies (29) | mid beta | 0.938 | 10.9 | 3.56 / 0.92 | 2.58 / 1.44 | 2.07 / 1.94 | 2.12 / 2.30 |
| Currencies (29) | high beta | 0.961 | 17.6 | 0.48 / 0.71 | 0.00 / 1.05 | 1.27 / 1.53 | 1.42 / 1.95 |
| Equity ETFs (113) | low beta | 0.879 | 5.4 | 4.15 / 0.79 | 1.24 / 1.77 | 2.75 / 2.32 | 3.09 / 2.48 |
| Equity ETFs (113) | mid beta | 0.901 | 6.6 | 2.59 / 0.74 | 1.19 / 1.51 | 2.19 / 2.13 | 2.97 / 2.43 |
| Equity ETFs (113) | high beta | 0.916 | 7.9 | 3.46 / 0.67 | 0.85 / 1.32 | 1.85 / 1.90 | 2.52 / 2.33 |
| Bond, commodity, currency ETFs (32) | low beta | 0.915 | 7.8 | 6.12 / 0.91 | 0.77 / 1.76 | 2.34 / 2.23 | 1.59 / 2.41 |
| Bond, commodity, currency ETFs (32) | mid beta | 0.946 | 12.6 | 2.45 / 0.79 | 0.22 / 1.30 | 1.65 / 1.83 | 1.41 / 2.24 |
| Bond, commodity, currency ETFs (32) | high beta | 0.956 | 15.3 | 2.59 / 0.93 | 0.00 / 1.30 | 2.82 / 1.71 | 2.08 / 2.11 |

### Table 5. All test rows by the shock share D: GARCH-t's 2.5% breach rate, realized / predicted by lambda = 0

| universe | 0.00-0.02 | 0.02-0.05 | 0.05-0.10 | 0.10-0.20 | 0.20-0.35 | 0.35-0.50 | 0.50-0.70 | 0.70-1.00 |
|---|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | 3.08 / 2.49 | 2.94 / 2.38 | 2.55 / 2.25 | 2.15 / 2.00 | 1.79 / 1.58 | 1.41 / 1.10 | 0.81 / 0.63 | 0.51 / 0.20 |
| US large caps (201) | 3.30 / 2.47 | 3.15 / 2.37 | 2.90 / 2.22 | 2.73 / 1.94 | 2.34 / 1.49 | 1.81 / 0.99 | 1.23 / 0.53 | 0.61 / 0.16 |
| International large caps, screened | 2.75 / 2.47 | 2.82 / 2.36 | 2.62 / 2.20 | 2.48 / 1.91 | 2.11 / 1.45 | 1.74 / 0.93 | 1.38 / 0.48 | 1.16 / 0.13 |
| International large caps, as registered | 2.73 / 2.47 | 2.82 / 2.36 | 2.63 / 2.20 | 2.48 / 1.91 | 2.11 / 1.45 | 1.76 / 0.94 | 1.40 / 0.49 | 1.17 / 0.13 |
| Crypto (27) | 1.69 / 2.49 | 1.87 / 2.39 | 1.76 / 2.26 | 1.53 / 2.03 | 1.36 / 1.64 | 1.89 / 1.19 | 2.10 / 0.73 | 0.26 / 0.23 |
| Currencies (29) | 2.03 / 2.47 | 2.46 / 2.36 | 2.30 / 2.19 | 1.85 / 1.91 | 2.24 / 1.47 | 1.68 / 1.00 | 2.37 / 0.56 | n/a |
| Equity ETFs (113) | 3.27 / 2.47 | 3.08 / 2.35 | 3.14 / 2.17 | 2.78 / 1.86 | 2.95 / 1.37 | 2.79 / 0.84 | 6.06 / 0.41 | 6.44 / 0.11 |
| Bond, commodity, currency ETFs (32) | 2.79 / 2.46 | 2.52 / 2.35 | 2.45 / 2.18 | 2.61 / 1.87 | 2.10 / 1.40 | 3.23 / 0.88 | 4.67 / 0.46 | n/a |

### Table 6. What each model discards: slope of (1 - squared VaR ratio to GARCH-t) on D, top decile (R-squared)

| universe | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t | accuracy layer |
|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | 1.00 (0.57) | 0.67 (0.49) | 0.51 (0.06) | 0.92 (0.17) | 0.96 (0.56) | 0.12 (0.01) | 0.07 (0.05) |
| US large caps (201) | 0.81 (0.63) | 0.68 (0.44) | 0.59 (0.36) | 0.72 (0.27) | 0.85 (0.58) | 0.01 (0.00) | 0.07 (0.04) |
| International large caps, screened | 0.78 (0.67) | 0.49 (0.24) | 0.60 (0.39) | 0.61 (0.23) | 0.68 (0.40) | -0.00 (0.00) | 0.03 (0.00) |
| International large caps, as registered | 0.83 (0.41) | 0.59 (0.15) | 0.60 (0.36) | 0.68 (0.20) | 0.70 (0.13) | -0.02 (0.00) | 0.38 (0.01) |
| Crypto (27) | 0.88 (0.41) | 0.81 (0.26) | 0.93 (0.40) | 0.81 (0.16) | 0.85 (0.17) | 0.01 (0.00) | 0.24 (0.06) |
| Currencies (29) | 1.18 (0.23) | 0.12 (0.02) | 1.11 (0.15) | 0.52 (0.04) | 1.35 (0.10) | 0.37 (0.02) | 0.08 (0.01) |
| Equity ETFs (113) | 0.54 (0.40) | 0.09 (0.00) | 0.85 (0.23) | 0.37 (0.01) | 0.52 (0.07) | -0.28 (0.03) | -0.13 (0.01) |
| Bond, commodity, currency ETFs (32) | 0.52 (0.24) | -0.27 (0.10) | 0.59 (0.08) | 0.42 (0.01) | 0.20 (0.01) | 0.00 (0.00) | -0.21 (0.10) |

### Table 7. D against the kurtosis score as a locator: pooled trees' 2.5% edge over GARCH-t (DM)

| universe | top D decile | top kurtosis decile | overlap | D only | kurtosis only |
|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | +12.9% (15.4) | +8.2% (9.37) | 0.26 | +7.4% (8.54) | +0.4% (0.75) |
| US large caps (201) | +4.8% (9.91) | +5.1% (8.1) | 0.39 | +1.4% (4.31) | +1.3% (4.62) |
| International large caps, screened | +5.3% (13.92) | +4.4% (10.49) | 0.35 | +3.5% (6.02) | +1.6% (5.02) |
| International large caps, as registered | +14.5% (3.19) | +14.9% (3.11) | 0.35 | +3.4% (4.77) | +1.6% (3.53) |
| Crypto (27) | +17.0% (1.88) | +15.4% (3.15) | 0.27 | +4.6% (2.54) | +0.4% (0.91) |
| Currencies (29) | +1.2% (2.0) | +1.4% (2.34) | 0.29 | +0.6% (0.69) | +0.8% (1.74) |
| Equity ETFs (113) | -3.8% (-2.48) | -3.1% (-1.32) | 0.21 | -2.7% (-2.24) | -1.2% (-1.88) |
| Bond, commodity, currency ETFs (32) | +1.5% (-0.53) | +0.8% (-0.79) | 0.29 | +1.3% (0.13) | +0.1% (0.19) |

### Table 8. Earnings shocks (US stock universes)

| universe | names with dates | days in an earnings window | top-decile rows whose largest shock is an earnings shock | lambda, earnings shocks | lambda, other shocks | corrected GARCH-t edge: earnings / other | pooled trees edge: earnings / other |
|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | 151 | 3.1% | 71.7% | 0.10 [0.05, 0.15] | 0.20 [0.05, 0.25] | +5.7% (3.97) / +2.2% (0.68) | +7.7% (6.25) / +6.1% (2.13) |
| US large caps (201) | 201 | 3.2% | 75.5% | 0.10 [0.05, 0.15] | 0.35 [0.17, 0.70] | +5.8% (6.19) / -0.1% (0.29) | +5.3% (8.07) / +4.7% (4.57) |

### Table 9. Subgroups: lambda (2.5% pinball, top decile) and GARCH-t's 2.5% breach rate at shock ages 1-10 and 6-15

| universe | grouping | group | names | lambda [95%] | breach, ages 1-10 | breach, ages 6-15 | pooled trees' top-decile edge |
|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | sic | business services | 25 | 0.15 [0.05, 0.25] | 0.58 | 0.57 | +6.5% (2.42) |
| CRSP stocks (canonical, 200) | sic | finance, insurance, real estate | 44 | 0.20 [0.00, 0.75] | 0.34 | 0.32 | +6.8% (2.36) |
| CRSP stocks (canonical, 200) | sic | manufacturing, durable | 36 | 0.05 [0.00, 0.10] | 0.54 | 0.99 | +8.5% (3.73) |
| CRSP stocks (canonical, 200) | sic | manufacturing, nondurable | 30 | 0.05 [-0.05, 0.10] | 0.43 | 0.80 | +12.0% (4.4) |
| CRSP stocks (canonical, 200) | sic | other divisions | 24 | 0.10 [0.05, 0.15] | 0.89 | 1.06 | +4.9% (3.12) |
| CRSP stocks (canonical, 200) | sic | trade | 22 | 0.15 [0.00, 0.20] | 0.00 | 0.75 | +10.6% (5.75) |
| CRSP stocks (canonical, 200) | sic | transport, communications, utilities | 15 | 0.15 [-0.05, 0.15] | 0.47 | 0.90 | +8.2% (4.5) |
| CRSP stocks (canonical, 200) | cohort | largecap | 74 | 0.05 [0.05, 0.20] | 0.30 | 0.96 | +7.0% (5.46) |
| CRSP stocks (canonical, 200) | cohort | smallcap | 126 | 0.10 [0.05, 0.15] | 0.49 | 0.63 | +8.5% (8.91) |
| CRSP stocks (canonical, 200) | vol | high volatility tercile | 67 | 0.10 [0.05, 0.15] | 0.71 | 0.82 | +8.2% (6.98) |
| CRSP stocks (canonical, 200) | vol | low volatility tercile | 67 | 0.20 [0.05, 0.63] | 0.47 | 1.11 | +4.8% (1.63) |
| CRSP stocks (canonical, 200) | vol | mid volatility tercile | 66 | 0.10 [0.05, 0.20] | 0.09 | 0.33 | +9.9% (6.28) |
| US large caps (201) | vol | high volatility tercile | 67 | 0.15 [0.10, 0.30] | 0.85 | 0.63 | +5.4% (4.66) |
| US large caps (201) | vol | low volatility tercile | 67 | 0.15 [0.10, 0.20] | 0.95 | 0.74 | +4.8% (5.89) |
| US large caps (201) | vol | mid volatility tercile | 67 | 0.15 [0.05, 0.30] | 0.42 | 0.44 | +5.2% (8.41) |
| International large caps, screened | market | Australia | 38 | 0.20 [0.05, 0.35] | 0.99 | 0.83 | +4.5% (4.12) |
| International large caps, screened | market | Canada | 40 | 0.70 [0.15, 1.50] | 2.00 | 0.45 | +2.8% (3.16) |
| International large caps, screened | market | France | 43 | 0.10 [0.00, 0.30] | 1.03 | 0.69 | +4.7% (6.17) |
| International large caps, screened | market | Germany | 45 | 0.15 [0.05, 0.35] | 0.67 | 1.25 | +4.1% (6.04) |
| International large caps, screened | market | Hong Kong | 37 | 0.05 [-0.05, 0.10] | 0.33 | 0.52 | +5.2% (4.68) |
| International large caps, screened | market | Japan | 57 | 0.20 [0.05, 0.40] | 0.73 | 0.88 | +3.9% (5.6) |
| International large caps, screened | market | United Kingdom | 63 | 0.25 [0.10, 0.55] | 1.28 | 0.94 | +5.0% (6.79) |
| International large caps, screened | vol | high volatility tercile | 108 | 0.25 [0.10, 0.48] | 1.08 | 0.83 | +4.6% (7.55) |
| International large caps, screened | vol | low volatility tercile | 108 | 0.20 [0.10, 0.43] | 1.11 | 0.92 | +4.2% (7.05) |
| International large caps, screened | vol | mid volatility tercile | 107 | 0.15 [0.05, 0.35] | 1.00 | 0.74 | +4.3% (9.12) |
| International large caps, as registered | market | Australia | 40 | 0.20 [0.05, 0.35] | 1.29 | 0.87 | +4.4% (2.9) |
| International large caps, as registered | market | Canada | 40 | 0.70 [0.10, 1.50] | 2.03 | 0.46 | +1.9% (2.04) |
| International large caps, as registered | market | France | 47 | -0.00 [0.00, -0.00] | 1.11 | 0.68 | +38.0% (2.09) |
| International large caps, as registered | market | Germany | 46 | 0.15 [0.05, 0.30] | 0.65 | 1.15 | +4.3% (5.86) |
| International large caps, as registered | market | Hong Kong | 42 | 0.10 [0.00, 0.15] | 0.30 | 0.47 | +5.2% (3.99) |
| International large caps, as registered | market | Japan | 58 | 0.20 [0.05, 0.40] | 0.73 | 0.88 | +3.7% (3.76) |
| International large caps, as registered | market | United Kingdom | 65 | 0.20 [0.10, 0.50] | 1.25 | 0.87 | +5.2% (6.58) |
| International large caps, as registered | vol | high volatility tercile | 113 | 0.25 [0.10, 0.50] | 1.22 | 0.78 | +4.5% (5.78) |
| International large caps, as registered | vol | low volatility tercile | 113 | 0.15 [0.10, 0.35] | 1.11 | 0.87 | +4.7% (5.05) |
| International large caps, as registered | vol | mid volatility tercile | 112 | -0.00 [0.00, -0.00] | 1.00 | 0.74 | +26.8% (2.47) |
| Crypto (27) | vol | high volatility tercile | 9 | -0.00 [0.00, 0.20] | 0.38 | 0.76 | +22.9% (1.91) |
| Crypto (27) | vol | low volatility tercile | 9 | 0.15 [-0.05, 0.15] | 0.00 | 0.43 | +12.0% (2.92) |
| Crypto (27) | vol | mid volatility tercile | 9 | 0.10 [0.05, 0.20] | 0.64 | 0.96 | +7.7% (2.39) |
| Currencies (29) | vol | high volatility tercile | 10 | 0.35 [0.05, 0.80] | 1.58 | 0.76 | +2.1% (2.11) |
| Currencies (29) | vol | low volatility tercile | 10 | 0.75 [0.30, 1.50] | 3.15 | 2.69 | +0.1% (0.64) |
| Currencies (29) | vol | mid volatility tercile | 9 | 0.30 [-0.15, 0.78] | 1.14 | 1.09 | +1.0% (0.78) |
| Equity ETFs (113) | vol | high volatility tercile | 38 | 0.80 [0.10, 1.35] | 3.23 | 1.51 | -4.8% (-1.88) |
| Equity ETFs (113) | vol | low volatility tercile | 38 | 1.10 [0.55, 1.50] | 3.72 | 1.80 | -0.4% (-0.96) |
| Equity ETFs (113) | vol | mid volatility tercile | 37 | 0.90 [0.35, 1.40] | 3.34 | 1.30 | -4.0% (-1.1) |
| Bond, commodity, currency ETFs (32) | vol | high volatility tercile | 11 | 1.00 [0.30, 1.50] | 3.42 | 1.31 | +0.2% (0.08) |
| Bond, commodity, currency ETFs (32) | vol | low volatility tercile | 11 | 1.50 [0.07, 1.50] | 5.36 | 0.82 | +6.7% (0.92) |
| Bond, commodity, currency ETFs (32) | vol | mid volatility tercile | 10 | 0.55 [0.15, 1.50] | 2.85 | 1.75 | -1.1% (-2.0) |

### Table 10. Stability over the test period: lambda by third of the test dates

| universe | first third | second third | last third |
|---|---|---|---|
| CRSP stocks (canonical, 200) | 2020-08-06 to 2022-01-24: 0.10 [0.05, 0.20] | 2022-01-24 to 2023-07-14: 0.10 [0.05, 0.20] | 2023-07-14 to 2024-12-31: 0.10 [0.05, 0.10] |
| US large caps (201) | 2016-01-04 to 2019-08-16: 0.10 [0.05, 0.20] | 2019-08-16 to 2023-03-30: 0.25 [0.05, 0.70] | 2023-03-30 to 2026-11-13: 0.15 [0.05, 0.30] |
| International large caps, screened | 2016-01-04 to 2019-08-16: 0.20 [0.10, 0.55] | 2019-08-16 to 2023-03-31: 0.40 [0.05, 1.50] | 2023-03-31 to 2026-11-13: 0.10 [0.05, 0.20] |
| International large caps, as registered | 2016-01-04 to 2019-08-16: 0.20 [0.10, 0.50] | 2019-08-16 to 2023-03-31: 0.40 [0.05, 1.50] | 2023-03-31 to 2026-11-13: -0.00 [0.00, -0.00] |
| Crypto (27) | 2022-01-01 to 2023-08-16: 0.15 [0.05, 0.15] | 2023-08-16 to 2025-03-31: 0.15 [0.05, 0.35] | 2025-03-31 to 2026-11-13: -0.00 [-0.15, 0.10] |
| Currencies (29) | 2016-01-01 to 2019-08-15: 0.35 [0.05, 0.65] | 2019-08-15 to 2023-03-28: 0.20 [-0.15, 0.80] | 2023-03-28 to 2026-11-13: 0.90 [0.30, 1.50] |
| Equity ETFs (113) | 2016-01-04 to 2019-08-16: 0.70 [-0.10, 1.50] | 2019-08-16 to 2023-03-30: 1.05 [0.27, 1.50] | 2023-03-30 to 2026-11-13: 0.75 [0.00, 1.50] |
| Bond, commodity, currency ETFs (32) | 2016-01-04 to 2019-08-16: 1.30 [0.07, 1.50] | 2019-08-16 to 2023-03-30: 0.85 [0.05, 1.50] | 2023-03-30 to 2026-11-13: 1.50 [-0.40, 1.50] |

### Table 11. Superior predictive ability (consistent p-value), 2.5% pinball; best model in the set and its edge

| universe | days | GARCH-t vs flexible set | GARCH-t vs all | Beta-t-EGARCH vs flexible set | Beta-t-EGARCH vs learned tail on it | model confidence set (10%, T-max) |
|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | top_decile | p = 0.000 (neural, +8.6%) | p = 0.000 (body_bteg, +10.1%) | p = 0.296 (neural, +0.9%) | p = 0.000 (body_bteg, +2.5%) | body, body_bteg, engine_bteg, neural |
| CRSP stocks (canonical, 200) | deciles1to8 | p = 0.000 (neural, +1.3%) | p = 0.000 (neural, +1.3%) | p = 0.011 (neural, +0.9%) | p = 0.002 (body_bteg, +0.8%) | body, body_bteg, engine_bteg, neural |
| CRSP stocks (canonical, 200) | all_days | p = 0.000 (neural, +2.4%) | p = 0.000 (body_bteg, +2.5%) | p = 0.001 (neural, +1.0%) | p = 0.000 (body_bteg, +1.1%) | body_bteg, neural |
| US large caps (201) | top_decile | p = 0.000 (neural, +6.2%) | p = 0.000 (neural, +6.2%) | p = 0.203 (neural, +1.5%) | p = 0.042 (body_bteg, +1.4%) | body, body_bteg, engine_bteg, neural, param_bteg, sav_caviar |
| US large caps (201) | deciles1to8 | p = 0.053 (neural, +1.5%) | p = 0.054 (neural, +1.5%) | p = 0.105 (neural, +1.2%) | p = 0.090 (body_bteg, +1.1%) | body, body_bteg, engine_bteg, gjr_skewt, neural, param_bteg, sav_caviar |
| US large caps (201) | all_days | p = 0.009 (neural, +2.2%) | p = 0.009 (neural, +2.2%) | p = 0.104 (neural, +1.2%) | p = 0.077 (body_bteg, +1.2%) | body, body_bteg, engine_bteg, neural, param_bteg |
| International large caps, screened | top_decile | p = 0.000 (neural, +6.4%) | p = 0.000 (neural, +6.4%) | p = 0.038 (neural, +3.0%) | p = 0.015 (body_bteg, +2.6%) | body, body_bteg, engine_bteg, neural, param_bteg |
| International large caps, screened | deciles1to8 | p = 0.002 (neural, +1.8%) | p = 0.003 (neural, +1.8%) | p = 0.018 (neural, +1.8%) | p = 0.035 (body_bteg, +1.2%) | body, body_bteg, engine, engine_bteg, gjr_skewt, neural, param_bteg |
| International large caps, screened | all_days | p = 0.000 (neural, +2.4%) | p = 0.000 (neural, +2.4%) | p = 0.017 (neural, +1.9%) | p = 0.032 (body_bteg, +1.3%) | body, body_bteg, engine, engine_bteg, gjr_skewt, neural, param_bteg, sav_caviar |
| International large caps, as registered | top_decile | p = 0.019 (neural, +18.0%) | p = 0.021 (neural, +18.0%) | p = 0.221 (neural, +2.5%) | p = 0.521 (body_bteg, -260.2%) | body, neural, param_bteg, sav_caviar |
| International large caps, as registered | deciles1to8 | p = 0.001 (neural, +2.0%) | p = 0.002 (neural, +2.0%) | p = 0.011 (neural, +1.9%) | p = 0.525 (body_bteg, -1.9%) | body, engine, gjr_skewt, neural, param_bteg, sav_caviar |
| International large caps, as registered | all_days | p = 0.001 (neural, +4.4%) | p = 0.002 (neural, +4.4%) | p = 0.019 (neural, +1.9%) | p = 0.523 (body_bteg, -33.4%) | body, gjr_skewt, neural, param_bteg, sav_caviar |
| Crypto (27) | top_decile | p = 0.023 (sav_caviar, +17.4%) | p = 0.036 (sav_caviar, +17.4%) | p = 0.505 (sav_caviar, +1.5%) | p = 0.181 (body_bteg, +1.3%) | body, body_bteg, engine_bteg, neural, param_bteg, sav_caviar |
| Crypto (27) | deciles1to8 | p = 0.068 (neural, +2.0%) | p = 0.081 (neural, +2.0%) | p = 0.020 (neural, +2.8%) | p = 0.015 (body_bteg, +1.8%) | body, body_bteg, fhs, gjr_skewt, neural, sav_caviar |
| Crypto (27) | all_days | p = 0.029 (neural, +4.7%) | p = 0.038 (neural, +4.7%) | p = 0.052 (neural, +2.5%) | p = 0.034 (body_bteg, +1.5%) | body, body_bteg, engine_bteg, fhs, gjr_skewt, neural, param_bteg, sav_caviar |
| Currencies (29) | top_decile | p = 0.032 (neural, +4.3%) | p = 0.115 (neural, +4.3%) | p = 0.045 (neural, +6.0%) | p = 0.345 (body_bteg, +1.6%) | body, body_bteg, engine, fhs, garch_t, gjr_skewt, neural, param_bteg, sav_caviar |
| Currencies (29) | deciles1to8 | p = 0.006 (neural, +0.7%) | p = 0.006 (neural, +0.7%) | p = 0.000 (neural, +2.3%) | p = 0.001 (body_bteg, +0.5%) | body, garch_t, neural, sav_caviar |
| Currencies (29) | all_days | p = 0.001 (neural, +0.9%) | p = 0.001 (neural, +0.9%) | p = 0.000 (neural, +2.5%) | p = 0.001 (body_bteg, +0.9%) | body, neural, sav_caviar |
| Equity ETFs (113) | top_decile | p = 0.397 (neural, +1.5%) | p = 0.535 (neural, +1.5%) | p = 0.261 (neural, +2.1%) | p = 0.571 (engine_bteg, -0.1%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg, sav_caviar |
| Equity ETFs (113) | deciles1to8 | p = 0.207 (neural, +1.3%) | p = 0.093 (gjr_skewt, +2.2%) | p = 0.209 (neural, +1.3%) | p = 0.318 (body_bteg, +0.6%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg, sav_caviar |
| Equity ETFs (113) | all_days | p = 0.253 (neural, +1.2%) | p = 0.128 (gjr_skewt, +1.9%) | p = 0.207 (neural, +1.5%) | p = 0.282 (engine_bteg, +0.8%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg, sav_caviar |
| Bond, commodity, currency ETFs (32) | top_decile | p = 0.979 (body, -0.4%) | p = 0.951 (body_bteg, +0.2%) | p = 0.625 (body, +0.3%) | p = 0.231 (body_bteg, +0.9%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg, sav_caviar |
| Bond, commodity, currency ETFs (32) | deciles1to8 | p = 0.201 (neural, +0.8%) | p = 0.211 (neural, +0.8%) | p = 0.214 (neural, +0.7%) | p = 0.213 (body_bteg, +0.4%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg |
| Bond, commodity, currency ETFs (32) | all_days | p = 0.277 (body, +0.4%) | p = 0.300 (body_bteg, +0.6%) | p = 0.273 (body, +0.7%) | p = 0.153 (body_bteg, +0.9%) | body, body_bteg, engine, engine_bteg, fhs, garch_t, gjr_skewt, neural, param_bteg |

### Table 12. Conditional coverage given D (2.5%): slope of the hit rate on D (t), implied breach rate at D = 0 and D = 0.5

| universe | garch_t | gjr_skewt | param_bteg | sav_caviar | neural | body | body_bteg | cf_pre |
|---|---|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | -0.038 (-16.5); 3.02% to 1.12% | -0.031 (-13.8); 2.89% to 1.33% | -0.007 (-3.0); 2.54% to 2.17% | -0.021 (-9.1); 2.90% to 1.86% | -0.015 (-6.9); 2.77% to 2.04% | -0.024 (-9.7); 3.44% to 2.24% | -0.003 (-1.0); 2.97% to 2.84% | -0.005 (-1.9); 3.03% to 2.79% |
| US large caps (201) | -0.035 (-11.0); 3.26% to 1.49% | -0.033 (-12.4); 3.20% to 1.54% | -0.007 (-1.9); 3.03% to 2.66% | -0.020 (-5.7); 3.22% to 2.22% | -0.017 (-5.8); 3.15% to 2.29% | -0.024 (-7.6); 3.22% to 2.00% | -0.009 (-2.4); 3.13% to 2.69% | +0.001 (0.3); 3.27% to 3.35% |
| International large caps, screened | -0.025 (-8.8); 2.81% to 1.57% | -0.024 (-9.9); 2.88% to 1.66% | +0.001 (0.2); 2.73% to 2.76% | -0.010 (-2.8); 2.91% to 2.43% | -0.012 (-4.2); 2.92% to 2.33% | -0.021 (-7.5); 2.93% to 1.89% | -0.009 (-3.0); 2.92% to 2.46% | +0.001 (0.4); 2.83% to 2.89% |
| International large caps, as registered | -0.024 (-8.6); 2.80% to 1.59% | -0.024 (-9.8); 2.87% to 1.67% | +0.001 (0.4); 2.70% to 2.77% | -0.009 (-2.8); 2.91% to 2.44% | -0.012 (-4.4); 2.90% to 2.30% | -0.019 (-6.9); 2.91% to 1.95% | -0.007 (-2.2); 2.88% to 2.54% | +0.005 (1.3); 2.81% to 3.05% |
| Crypto (27) | -0.005 (-0.4); 1.71% to 1.48% | -0.003 (-0.2); 2.02% to 1.87% | +0.023 (1.7); 1.51% to 2.65% | +0.031 (2.2); 1.59% to 3.14% | +0.019 (1.4); 1.95% to 2.88% | +0.013 (1.1); 1.71% to 2.34% | +0.030 (2.2); 1.83% to 3.34% | +0.054 (3.5); 1.61% to 4.29% |
| Currencies (29) | -0.004 (-0.8); 2.18% to 1.97% | +0.001 (0.3); 2.23% to 2.30% | +0.030 (4.9); 1.71% to 3.24% | +0.022 (3.8); 1.96% to 3.06% | -0.003 (-0.7); 2.13% to 1.95% | -0.004 (-0.9); 2.13% to 1.91% | +0.015 (2.8); 1.92% to 2.67% | +0.021 (3.6); 2.17% to 3.23% |
| Equity ETFs (113) | +0.006 (0.4); 3.10% to 3.42% | -0.003 (-0.3); 2.50% to 2.34% | +0.031 (1.7); 2.98% to 4.55% | +0.035 (2.2); 2.06% to 3.79% | +0.016 (1.2); 2.17% to 2.97% | +0.018 (1.2); 2.37% to 3.30% | +0.028 (1.8); 2.30% to 3.69% | -0.013 (-1.0); 3.11% to 2.47% |
| Bond, commodity, currency ETFs (32) | +0.002 (0.2); 2.56% to 2.68% | +0.002 (0.3); 2.23% to 2.35% | +0.032 (2.5); 2.35% to 3.92% | +0.027 (2.5); 2.05% to 3.39% | +0.013 (1.3); 2.18% to 2.82% | -0.000 (-0.0); 2.33% to 2.32% | +0.015 (1.5); 2.18% to 2.94% | +0.024 (2.1); 2.53% to 3.74% |

### Table 13. Conditional predictive ability given D (2.5% pinball): slope of the gain over GARCH-t on D, t-statistic; then t of D and of the kurtosis-decile dummy when both enter

| universe | param_bteg | sav_caviar | neural | body | body_bteg | gjr_skewt |
|---|---|---|---|---|---|---|
| CRSP stocks (canonical, 200) | t = 8.0; both: 7.8, 6.3 | t = 8.0; both: 7.6, 3.9 | t = 19.9; both: 18.9, 6.5 | t = 20.0; both: 18.9, 6.8 | t = 11.9; both: 11.4, 7.0 | t = 6.3; both: 6.2, -0.6 |
| US large caps (201) | t = 2.8; both: 2.0, 6.5 | t = 6.2; both: 4.9, 5.5 | t = 11.4; both: 8.8, 1.4 | t = 9.4; both: 7.4, 6.1 | t = 6.1; both: 5.5, 3.4 | t = 2.5; both: 2.7, -2.5 |
| International large caps, screened | t = 1.3; both: 0.9, 8.7 | t = 2.7; both: 2.1, 9.3 | t = 14.9; both: 12.6, 3.8 | t = 12.4; both: 10.4, 3.1 | t = 9.7; both: 9.2, 4.6 | t = 2.4; both: 2.5, -3.0 |
| International large caps, as registered | t = 6.7; both: 6.5, 9.5 | t = 7.4; both: 7.2, 9.9 | t = 8.3; both: 8.2, 8.2 | t = 8.1; both: 8.0, 7.4 | t = -13.6; both: -13.0, -15.2 | t = 6.8; both: 6.8, 4.9 |
| Crypto (27) | t = 3.2; both: 3.1, 3.6 | t = 3.2; both: 3.0, 4.3 | t = 4.1; both: 3.9, 4.7 | t = 4.8; both: 4.8, 3.8 | t = 3.2; both: 3.1, 2.7 | t = 3.0; both: 2.7, 1.7 |
| Currencies (29) | t = -0.5; both: -0.5, -0.0 | t = 0.5; both: 0.4, 1.0 | t = 4.3; both: 4.1, 1.5 | t = 3.3; both: 3.0, 2.4 | t = 1.8; both: 1.7, 1.1 | t = 0.1; both: 0.5, -2.4 |
| Equity ETFs (113) | t = -2.0; both: -2.0, 0.7 | t = -1.0; both: -0.9, -1.1 | t = 0.3; both: 0.4, -2.3 | t = -1.7; both: -1.7, -0.9 | t = -1.2; both: -1.2, -0.4 | t = 1.8; both: 1.9, -3.4 |
| Bond, commodity, currency ETFs (32) | t = -2.2; both: -2.4, -0.3 | t = -1.2; both: -1.4, -0.0 | t = 1.3; both: 1.4, -0.3 | t = 1.8; both: 1.9, -0.2 | t = -0.0; both: -0.2, 1.1 | t = 1.1; both: 1.5, -1.8 |

### Table 14. Simulation: mean (standard deviation) over simulated panels

| quantity | garch | jump_0 | jump_0.25 | jump_1 |
|---|---|---|---|---|
| true lambda | 1 | 0 | 0.25 | 1 |
| fitted alpha, median | 0.14 (0.00) | 0.04 (0.00) | 0.04 (0.00) | 0.07 (0.00) |
| fitted beta, median | 0.81 (0.01) | 0.92 (0.01) | 0.93 (0.00) | 0.92 (0.00) |
| lambda-hat, top decile, pinball | 1.03 (0.14) | -0.02 (0.03) | 0.27 (0.03) | 0.84 (0.11) |
| lambda-hat, top decile, likelihood | 1.05 (0.06) | 0.00 (0.00) | 0.42 (0.03) | 1.00 (0.11) |
| lambda-hat, pre-test rows | 1.04 (0.18) | 0.01 (0.02) | 0.31 (0.02) | 0.84 (0.09) |
| GARCH-t breach, top decile (%) | 2.47 (0.15) | 1.39 (0.05) | 1.63 (0.09) | 2.17 (0.15) |
| GARCH-t breach, shock age 1-10 (%) | 2.44 (0.36) | 0.59 (0.17) | 0.83 (0.16) | 2.27 (0.33) |
| predicted by lambda = 0 (%) | 0.89 (0.03) | 0.66 (0.06) | 0.42 (0.01) | 0.48 (0.02) |
| GARCH-t breach, shock age 43-63 (%) | 2.35 (0.33) | 1.83 (0.18) | 1.99 (0.13) | 2.21 (0.19) |
| corrected GARCH-t (lambda = 0) edge, top decile (%) | -8.86 (1.88) | 7.13 (0.39) | 0.98 (0.52) | -10.16 (1.51) |
| corrected (pre-test lambda) edge, top decile (%) | -0.08 (0.13) | 7.10 (0.45) | 3.14 (0.31) | 0.10 (0.18) |
| Beta-t-EGARCH edge, top decile (%) | -9.63 (2.37) | 8.17 (0.76) | 2.55 (0.38) | -6.35 (1.51) |
| Beta-t-EGARCH edge, all days (%) | -2.32 (0.27) | 2.15 (0.13) | 0.42 (0.16) | -1.52 (0.32) |
| corrected (pre-test lambda) edge, all days (%) | -0.02 (0.04) | 0.89 (0.10) | 0.50 (0.05) | 0.06 (0.03) |
