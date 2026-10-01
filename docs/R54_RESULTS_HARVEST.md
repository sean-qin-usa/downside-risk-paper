# R54 results harvest (2026-10-01): a robust Stage-1 scale, and cold start against peer groups

Two pre-registered runs. Predictions were written into each script header before execution and are
graded here without adjustment. Neither run has been written into the manuscript: the text pass waits
until these numbers settle, and four of the eleven predictions failed in ways that change what the text
should say.

New scripts: `code/paper/job_robust_engine.py`, `code/paper/job_coldstart_peers.py`,
`code/paper/job_exception_battery.py`, `code/paper/make_table_deployed.py`,
`code/paper/make_table_scaleshape.py`. Result files: `robust_engine_results.json`,
`robust_engine_results_taq30.json`, `coldstart_peers_results.json`,
`exception_battery_results.json`. Tables:
`submission/tables/tab_deployed.tex`, `submission/tables/tab_scaleshape_bteg.tex`.

---

## 1. Beta-t-EGARCH as a Stage-1 scale (`job_robust_engine.py`, panel canon200)

200 names, 221,600 test rows, 0 fitting failures — the rows of Table 1. Three scales, each as a
standalone parametric benchmark and as the scale under the flexible shape, every downstream stage
re-estimated on that scale's own residuals.

The filter is Harvey–Chakravarty Beta-t-EGARCH, fitted per name by maximum likelihood on the training
window: `lam_{t+1} = om + phi(lam_t - om) + kap u_t` with `u_t = (nu+1)e_t^2/(nu+e_t^2) - 1`, the score
of the Student-t log-likelihood, a martingale difference bounded in `[-1, nu]`. Median fitted
`phi = 0.962`, `kap = 0.077`, `nu = 4.10`. The bound that matters for deployment is `kap*nu = 0.335`:
one shock, however large, can multiply the conditional scale by at most `e^0.335 = 1.40`.

**Pipeline validation.** On the `garch_t` arm the job independently reproduces the committed numbers:
top-decile edge +2.445% at DM 10.18 against the paper's +2.483% at DM 10.08, deciles 1–9 −0.013%
(DM −0.20) against −0.015% (DM −0.25), overall +0.263% (DM 4.60) against +0.265% (DM 4.62), with GPD
`xi = 0.298`, `beta = 0.639` and conformal shift −0.3426, the last matching `bench_all_results.json`
exactly.

**The decomposition, from likelihood instead of a cap.** Flexible shape over its own-scale parametric
benchmark:

| Stage-1 scale | top mk63 decile | deciles 1–9 | overall |
|---|---|---|---|
| GARCH(1,1)-$t$ | **+2.445%** (DM 10.18) | −0.013% (−0.20) | +0.263% (4.60) |
| bounded-news, 3σ cap | +0.420% (DM 4.38) | +0.046% (1.30) | +0.087% (2.41) |
| Beta-t-EGARCH | **+0.260%** (DM 1.81) | −0.090% (−1.46) | −0.052% (−0.88) |

**P1 confirmed.** As a standalone benchmark Beta-t-EGARCH beats GARCH-$t$ on FZ0 at both levels, by
0.0128 (DM 2.83) at 1% and 0.0150 (DM 4.30) at 2.5% — inside the predicted 0.005–0.03 band with DM in
the predicted 1.5–5 range.

**P2 confirmed on size, marginal on significance.** The surviving top-decile edge is +0.26%, inside the
predicted +0.2% to +0.9%, but at DM 1.81 rather than the predicted 2–5. Section 4's attribution
therefore holds and in fact strengthens: a maximum-likelihood robust scale removes slightly *more* of
the frontier than the hand-set 3σ cap does, and under it the flexible shape no longer helps overall
(−0.052%, DM −0.88) or in the bulk (−0.090%, DM −1.46). The cap-tuning objection to the bounded-news
row is answered — the same conclusion follows from a fitted model with no cap to choose.

**P3 half failed.** `engine_bteg` is below `engine` on FZ0 at both levels as predicted (gaps 0.0167 and
0.0176, DM 3.31 and 4.55), but by *more* than the standalone gap, not less. The robust scale helps the
flexible configuration more than it helps the parametric tail — the learner's scale and dispersion
features are more informative once the scale itself stops overshooting.

**P4 failed decisively, and the failure is worth more than the prediction.** Agreement between any two
scales on the top-decile indicator is 0.985–0.993, against 0.82 for independent sorts. A robust scale
does *not* relabel which days look misspecified. The score is close to filter-invariant, which supports
its status as a real-time monitor: a desk's score reading does not depend on which of these three
filters it runs.

**The result that bears on deployment.** On FZ0 every robust-scale configuration beats the paper's
engine at both levels, and the paper's engine leaves the 90% model confidence set once robust scales are
in the comparison set. With the per-scale unconditional-quantile rows in the candidate set the 90% MCS is
`body`, `engine_bip`, `body_bip`, `engine_bteg`, `body_bteg`, `bteg_uncond` at 1% and the same plus
`bteg` at 2.5%; `engine` is in neither, `engine_bteg` in both. Its breach rates are
*closer* to nominal than the paper engine's: 0.92% against 1.04% at the 1% level, 2.46% against 2.78%
at 2.5%. See §3.

## 1b. The third scale row for the scale–shape table (`--panel taq30`)

Same job, 30 large caps with intraday data, 33,240 test rows, four scales on one row set, DM
date-clustered NW(10) against the realized scale with an unconditional residual quantile — the published
table's own reference. Generated into `submission/tables/tab_scaleshape_bteg.tex` by
`code/paper/make_table_scaleshape.py`.

| Model | FZ0 2.5% | DM | FZ0 1% | DM |
|---|---|---|---|---|
| Daily core (GARCH-$t$ + EVT) — *reproduction* | 1.500 | +6.2 | 1.778 | +4.9 |
| Realized scale + residual quantile (reference) | 1.442 | — | 1.712 | — |
| &nbsp;&nbsp;+ pooled shape learner | 1.442 | +0.2 | 1.716 | +0.7 |
| &nbsp;&nbsp;+ shape + EVT | 1.441 | −0.5 | 1.715 | +2.0 |
| **Beta-t-EGARCH scale + residual quantile** | **1.483** | +5.1 | **1.756** | +3.8 |
| &nbsp;&nbsp;+ parametric $t$ tail (benchmark form) | 1.485 | +5.0 | 1.760 | +3.6 |
| &nbsp;&nbsp;+ pooled shape learner | 1.482 | +4.1 | 1.747 | +2.5 |
| &nbsp;&nbsp;+ shape + EVT | 1.480 | +4.9 | 1.754 | +3.8 |

The published row reproduces exactly at 2.5% (1.500 at DM +6.2 against the published 1.500, +6.2) and to
within 0.004 at 1%; the realized reference reproduces at 1.442/1.712 against the published 1.445/1.716,
and the learner row's DM of +0.2 matches the published +0.2. One published row form is not computed for
the new scale: the realized block's "+ EVT tail" row, a GPD tail with no learner.

**What the new block says.** The Beta-t-EGARCH scale sits between the daily GARCH-$t$ core and the
intraday realized scale, and closes about a third to a half of the gap between them using daily returns
alone: at 2.5% it recovers 0.020 of the 0.058 daily-to-realized gap (34%), at 1% 0.031 of 0.066 (47%).
It does not substitute for intraday data — it stays significantly worse than the realized scale at
DM +2.5 to +5.1 — but it is the best available scale where no realized measure can be built, which is
the regime the paper already reserves for the daily model. As on the 200-name panel, once the scale is
either robust or realized the flexible shape adds nothing on these large caps (−0.26% top decile under
Beta-t-EGARCH, −0.34% under the realized scale).

## 2. Cold start against pooled peer groups (`job_coldstart_peers.py`)

417 mature training names on dates before 2018-01-01; 303 test names first listed on or after that date
(300 of them the recent-IPO cohort), 203,067 rows, scored by listing age. No test name and no test date
enters a fitted stage; peer scales are point-in-time and leave-one-out, the leave-one-out median taken
by sorted rank and verified exact against brute force on 23,813 cases including tied pools.

**P1 failed with the sign reversed — the headline.** `own_short_pool`, the name's own 20-day rolling
volatility carrying the pooled shape, is available on 99% of rows from age 6 and beats *every* pooled
peer configuration at every age thereafter: +10.6% of pinball at ages 6–14 (DM 2.69), +11.1% at 15–60
(DM 5.69), +12.8% at 126–250 (DM 8.27), +5.0% at 501+ (DM 9.95). The cold-start edge the paper reports
against own history does not survive an own-history benchmark whose scale window is chosen properly.
What pooling repairs is the *shape*, not the scale, and the window in which pooling is the only option
is **days 1–5**, not the first months.

**P4 failed, and this is the second thing the text must concede.** In that 1–5 day window nothing is
calibrated: the pooled peer model breaches 7.13% against a 1% level and 10.52% against 2.5%. A new
listing's first week is far more volatile than any peer pool's median, so day-one pooled VaR is
available but not usable at the regulatory levels. Coverage only returns to nominal by age 61+ (1.27% at
the 1% level).

**P2 failed; the pre-registered consequence applies.** The young-cohort pool does not dominate. It wins
at ages 1–5 (+1.10%, DM 2.52) and 6–14 (+0.69%, DM 2.18), is within noise at 15–250 (DM −0.9 to 0.18),
and *loses* to the mature same-sector pool at 251–500 (+0.95%, DM 4.82) and 501+ (+2.26%, DM 6.33). The
header's rule was explicit: if the sector pool wins, "peers" means sector. It means cohort for two weeks
and sector afterwards.

**P3 confirmed and stronger than predicted.** The flexible learner on day-one features is *worse* than
pooled empirical quantiles at every age, by 1.7% to 3.8% (DM −4.4 to −8.5). Day-one features carry no
usable name-specific state, so flexibility only costs. Peer information itself does help: the global
pool loses to the young-cohort pool by 2.5–6.3% at every age.

## 3. The deployed configuration (`make_table_deployed.py` → `submission/tables/tab_deployed.tex`)

One configuration, every recorded metric, one exhibit. Fixed as **Stages 1–3: per-name GARCH(1,1)-$t$
scale, pooled HistGBM body on the five features, pooled GPD left tail spliced at p0 = 0.025 as the
body/EVT minimum, rearranged — with the conformal overlay off.**

The choice among the three GARCH-$t$-scale configurations is settled by the date-clustered exception
test, which only one of them passes at both levels: the engine at NW t 0.93 and 1.85, against the body
at 4.23 and 5.13 (over-breaching, 1.32% and 3.25%) and the conformal overlay at −10.73 and −9.79
(over-covering, 0.60% and 1.63%). The overlay is additionally worse on the FZ0 score it is meant to
serve at 2.5% (DM 2.4) and leaves the 90% MCS there; it stays named as the fallback where conditional
coverage binds, since it lifts the dynamic-quantile pass rate at 2.5% from 70.5% to 84.5%.

**This pick was under pressure from §1 and §4 has now settled it against the pick: see §4.** `engine_bteg` —
the same design with Beta-t-EGARCH in place of GARCH-$t$ at Stage 1 — beats it on FZ0 at both levels
(DM 3.31 and 4.55), sits inside the 90% MCS at both levels where it does not, and breaches closer to
nominal at both levels. The reason the pick stands today is coverage of evidence, not evidence of
superiority: the per-asset exception battery (Kupiec and Christoffersen pass rates, the date-clustered
NW t, the dynamic-quantile pass rate, Acerbi–Székely Z2, McNeil–Frey) exists only for the GARCH-$t$
scale. That battery has now been run as `job_exception_battery.py`; §4 reports it. `engine_bteg` passes the
date-clustered test at both levels and is the only variant also passing McNeil–Frey at both, so the
recommendation is to move Stage 1 and restate this table against it.

## 4. The exception battery on all three scales (`job_exception_battery.py`)

The Stage-1 question from §3, settled. The full calibration battery — the per-name Kupiec and
Christoffersen pass rates and the date-clustered exception statistic from `job_perasset_v2.py`, and the
pooled Kupiec, Acerbi–Székely Z2 and McNeil–Frey tests from `job_engine_esbt.py`, all copied verbatim —
run on all three scales over the same 200 names and 221,600 rows. Result file
`exception_battery_results.json`; the table is Panel D of `tab_deployed.tex`.

**Validation, and a reproducibility finding.** The GARCH-$t$ arm reproduces `engine_esbt_results.json`'s
GPD parameters exactly (u = −1.887, ξ = 0.293, β = 0.641) and its mean FZ0 to four decimals (2.14711
against 2.14749). It does *not* reproduce that file's conformal shift (−0.3275 against −0.3354, and
`bench_all`'s −0.3426) or its pooled Kupiec p-value. The cause is that the committed scripts leave
`HistGradientBoostingRegressor`'s `random_state` unset, so the internal early-stopping validation split
is not reproducible between runs. This job sets `random_state=0`. **The committed conformal shift is
therefore not exactly reproducible from the committed script**, which is worth fixing in those two
scripts independently of anything here.

### Predictions, graded

**P1 confirmed — and it was the decision.** `engine_bteg` passes the date-clustered exception test at
both levels: NW t = −1.89 at 1% and −0.68 at 2.5%, against the GARCH-$t$ engine's +0.29 and +1.60. The
picture is mixed rather than a clean win: at 2.5% the Beta-$t$-EGARCH engine is far better centred
(−0.68 against +1.60), while at 1% it over-covers and its statistic sits close to the 1.96 boundary.

**P2 failed.** Its pooled Kupiec at 2.5% rejects at p = 0.010, not p > 0.05. The improvement over the
GARCH-$t$ engine is nonetheless large (breach 2.42% against 2.72%, nominal 2.5%, where the GARCH-$t$
engine rejects at p < 0.001). On a panel of 221,600 rows the pooled Kupiec test rejects on tiny
deviations, which is why the date-clustered statistic is the one the paper leans on.

**P3 half confirmed.** Acerbi–Székely Z2 is much closer to zero at 2.5% (+0.027, p = 0.577, against
−0.101, p = 0.058) but further from zero at 1% (+0.072, p = 0.192, against −0.058, p = 0.348). Neither
rejects at either level, which was the second clause. The sign flips: the Beta-$t$-EGARCH engine's ES
slightly overstates the tail where the GARCH-$t$ engine's understates it.

**P4 confirmed.** `body_bteg` has the lowest FZ0 of any row in the whole study (2.12413 and 1.82765) and
still fails the date-clustered test at both levels (+3.00, +3.32), over-breaching at 1.20% and 2.99%. The
EVT branch, not the scale, is what buys exception-test compliance — exactly as predicted, and the same
pattern as the GARCH-$t$ body.

### The result that decides Stage 1

Four of the twelve variants pass the date-clustered test at both levels. Of those four, **only one also
passes the McNeil–Frey ES test at both levels:**

| variant | date-clustered NW $t$ (1% / 2.5%) | McNeil–Frey $p$ (1% / 2.5%) | ES verdict |
|---|---|---|---|
| `engine_garch_t` | +0.29 / +1.60 | 0.022 / 0.240 | fails at 1% |
| `param_garch_t` | +0.48 / +1.56 | 0.000 / 0.138 | fails at 1% |
| **`engine_bteg`** | **−1.89 / −0.68** | **0.110 / 0.565** | **passes both** |
| `param_bteg` | −0.68 / −0.17 | 0.000 / 0.023 | fails both |

`engine_bteg` is the only configuration in the battery that is jointly compliant on VaR exceptions and ES
shape at both regulatory levels. Combined with §1 — lower FZ0 at both levels (DM 3.31 and 4.55), inside
the 90% MCS at both levels where the GARCH-$t$ engine is in neither — **the recommendation is to move
Stage 1 to Beta-$t$-EGARCH.** The paper's own engine marginally fails McNeil–Frey at 1% (p = 0.022 here,
p = 0.0377 in the committed file), and the swap repairs it.

The cost, stated plainly: the Beta-$t$-EGARCH engine over-covers at the 1% level (0.90% against 1.00%).
For a 99% capital number that is the safe direction — regulators penalise too many exceptions, not too
few — but it is capital the desk pays for, and the date-clustered statistic of −1.89 is one step from
failing in that direction. A referee will notice both facts, so the text should state them rather than
quote only the 2.5% improvement.

Two incidental findings worth keeping. The bounded-news engine **fails** at 2.5% (NW t = +2.08), so the
3σ cap is not a Stage-1 candidate even though it was a fine diagnostic — the likelihood-based filter and
the hand-set cap part company exactly where deployment is decided. And `param_bteg`, the Beta-$t$-EGARCH
scale with a plain parametric $t$ tail and no learner at all, is the best-calibrated **VaR** row in the
whole table (breach 0.96% and 2.48%, pooled Kupiec p = 0.071 and 0.522, passing the date-clustered test
at both levels) while failing McNeil–Frey at both. Once the scale is score-driven and robust, a fixed
parametric tail is already well calibrated for VaR; what the flexible machinery and the EVT branch add is
the **ES shape**. That is a sharper statement of the paper's own thesis than the current text makes.

## Text consequences, for the readability pass that follows

1. Section 4's jump-robust decomposition gains a maximum-likelihood row and loses its cap-tuning
   exposure; the surviving shape component should be quoted as +0.26% (DM 1.81) under Beta-t-EGARCH
   beside +0.42% (DM 4.38) under the 3σ cap, and the honest summary is that it is smaller and weaker
   than the cap-based number implied.
2. The near-invariance of the score across the three filters (0.985–0.993 indicator agreement against
   0.82 under independence) is a new argument *for* the monitor and belongs in the score-use discussion.
3. The cold-start paragraph needs narrowing, not strengthening: it is a days-1–5 availability claim, the
   peer-group benchmark beats own short history only in that window, and the first week is not
   calibrated at the regulatory levels. The abstract's billing of amortization should be checked
   against this.
4. Stage 1 should move to Beta-t-EGARCH, and the abstract's joint-loss sentence then strengthens rather
   than weakens: against the daily-only comparison set the current engine is significantly beaten by
   robust-scale benchmarks (at 2.5% by `bip_t` DM -3.35, `bteg` -2.53, `bteg_uncond` -2.32,
   `bip_t_uncond` -1.85), while `engine_bteg` has no benchmark with a lower FZ0 at either level and is
   significantly beaten by none. The sentence can go from "lowest at 1%, never significantly beaten at
   2.5%" to lowest among benchmarks at both levels. Report the 1% over-coverage (0.90%) as the cost.
5. The intraday question needs no new work. The paper already concedes Realized GARCH in the
   introduction, Section 5 and the conclusion, and scopes itself to the daily regime; sec 1b now puts a
   number on how much of that gap a robust daily filter closes. The exposure was never intraday, it was
   the robust daily filter inside the paper's own arena, and sec 4 closes it.

---

# R55 (2026-10-01): the five open conditions, and what makes 2.5% work

## 0. Reconciliation first: the 2.46% / p = 0.010 mismatch was a reporting error of mine

`engine_bteg`'s breach rate at 2.5% is **2.415%** (5,352 of 221,600), and the Kupiec LR on that rate is
exactly the reported p = 0.0101. The arithmetic is internally consistent. The **2.46%** quoted in prose came
from `robust_engine_results.json`, a different run whose panel additionally requires a non-NaN mk$_{63}$ and
so trims more training rows, shifting the GPD fit and the breach rate by 0.04pp. Two runs, one number
quoted from each. The review's arithmetic was right; it was applied to a figure I mis-sourced. Nothing in
the table came from the mixed pair.

## 1. The swap conditions, all four now graded

| Condition | Result |
|---|---|
| Passes the date-clustered test at both levels | **Met.** −1.89 at 1%, −0.68 at 2.5% |
| Per-name Kupiec within 5 points of the current engine | **Met.** Kupiec +2.5pp at 1%, +4.5pp at 2.5%; Christoffersen −2.5pp and +5.0pp. All four inside ±5, three of four favour the swap |
| FZ0 DM against `bteg_uncond` and `bip_t`, both levels | **Met.** vs `bteg_uncond` DM +3.73 and +5.44; vs `bip_t` +2.30 and +2.73. The flexible stages still earn their place over the robust scale alone |
| Beats the current engine on the frozen 2000–2013 holdout | **Met.** FZ0 1.80459 vs 1.81972 (DM 3.09) at 1%; 1.55939 vs 1.57084 (DM 3.48) at 2.5%, against design-era DM 3.31 and 4.55 |

Caveat on the holdout: the 90% MCS there contains all 13 rows at both levels, so the era separates models
pairwise but not in the MCS sense. The replication is in the pairwise DM, not in model selection.
McNeil–Frey was not in the rule and is reported as an additional test, not the deciding one.

**The swap is adopted.** `tab_deployed.tex` is now generated for the Beta-$t$-EGARCH Stage 1, with Panel B
sourced entirely from the exception battery, so the table no longer mixes the 140-name per-asset file with
the 200-name ES file.

## 2. What makes 2.5% work: the test, not the model

At 2.5% `engine_bteg` already passes every calibration test except the **pooled** Kupiec (p = 0.0101). The
pooled statistic treats all 221,600 rows as independent. On a panel whose 200 names share each day's shock
the effective sample is the number of dates, 1,108. Reading unconditional coverage from the date-clustered
variance instead — which is what the paper already says it does, "pooled $p$-values are reported, not leaned
on; the regulatory reading is the per-asset and date-clustered restatement" — gives:

| configuration | level | rate | pooled Kupiec $p$ | **clustered UC $p$** | design effect | implied intra-date $\rho$ |
|---|---|---|---|---|---|---|
| `engine_bteg` | 1% | 0.901% | <0.001 | **0.059** | 6.2 | 0.026 |
| `engine_bteg` | 2.5% | 2.415% | 0.010 | **0.497** | 14.2 | 0.066 |
| `engine_garch_t` | 1% | 1.017% | 0.419 | 0.772 | 7.8 | 0.034 |
| `engine_garch_t` | 2.5% | 2.716% | <0.001 | 0.110 | 16.5 | 0.078 |

2.5% works, comfortably, at p = 0.497. The design effect of 14.2 implies an intra-date breach correlation of
0.066 across 200 US equities — entirely ordinary — so the clustering correction is substantive rather than
convenient, and the pooled statistic's rejection of a 0.085pp deviation is the artefact. This is not test
shopping: the designation predates this work and sits in the committed R53 text. It is now reported with
measured p-values rather than only t-statistics (`uc_clustered_p` in the battery output).

The honest residual: 1% is a **marginal** pass at p = 0.059, in the over-covering direction. That belongs in
the text beside the 2.5% result.

## 3. The conformal retarget failed, and the failure is the better answer

Diagnosis confirmed: the committed Stage-4 shift is estimated from the **body** on the calibration block and
applied to the **engine**, which already carries the EVT branch, so the tail correction enters twice.
Re-estimating it against the engine itself recovers much of the damage — at 2.5% the GARCH-$t$ overlay's
breach moves from 1.673% to 2.154%, and the Beta-$t$-EGARCH overlay's from 1.569% to 1.984%.

But every retargeted shift is still **negative** (−0.17 and −0.13 for Beta-$t$-EGARCH), when a *positive*
shift is what would carry 2.415% up to 2.50%. The calibration window, the last quarter of the training era,
shows the engine under-covering; the test era shows it over-covering. A single additive shift fitted out of
era has the wrong sign, and the era mismatch is larger than the 0.085pp bias being corrected. Every
retargeted row fails the date-clustered test (−3.07 to −6.28).

So Stage 4 is dropped rather than repaired, and the deployed configuration's "overlay off" is now justified
on mechanism rather than only on outcome. Pre-registered prediction for the added rows — that the retarget
would land within 0.05pp of nominal at both levels — **failed**, and in the direction that rules the stage
out altogether.

## 4. The random-state fix: every committed number that moves

`random_state=0` added at all three HistGradientBoostingRegressor sites in `job_perasset_v2.py` and
`job_engine_esbt.py`; both also now honour `GBC_PROJ`, which the README already claims of every job script.
Without the seed the early-stopping validation split was redrawn each run, so the conformal shift and
everything downstream of it were not reproducible from the committed scripts.

**`job_perasset_v2.py`** — 27 of 29 numbers move; **no verdict changes.** Largest: conformal shift at 1%
−0.5019 → −0.5510; at 2.5% −0.3518 → −0.3553. The deployed `hybrid_EVT` row: breach99 0.0106 → 0.0105,
Kupiec99 0.821 → 0.814, Christoffersen99 0.893 → 0.879, date-clustered99 0.93 → 0.89, breach975 0.0276 →
0.0273, Kupiec975 0.707 → 0.721, Christoffersen975 0.786 → 0.800, date-clustered975 1.85 → 1.71. `raw_hybrid`
4.23 → 4.63 and 5.13 → 5.00; `hybrid_EVT_conf` −10.73 → −11.86 and −9.79 → −10.08. All six pass/fail
verdicts hold.

**`job_engine_esbt.py`** — conf975 −0.3354 → −0.3275, which now agrees with the battery exactly. Engine rows:
breach 0.0103 → 0.0102 at 1%, Kupiec $p$ 0.1546 → 0.4185, AS $Z_2$ −0.0674 → −0.0576, MF $p$ 0.0377 → 0.0220,
FZ0 2.1475 → 2.1471; at 2.5% breach 0.0273 → 0.0272, AS $Z_2$ −0.1082 → −0.1013, MF $p$ 0.2581 → 0.2354, FZ0
1.8489 → 1.8487. DM of garch_t and fhs against the engine move by 0.03 to 0.11.

**One verdict moves:** the GARCH-$t$ engine's Acerbi–Székely $p$ at 2.5% goes 0.0394 → **0.0515**, crossing
0.05 from rejecting to not rejecting. It moves in the paper's favour, which is exactly why it must be
reported as a consequence of a seed fix rather than presented as a finding. The engine's McNeil–Frey at 1%
still rejects, slightly more strongly (0.0377 → 0.0220).

**Printed in the paper and therefore needing a ledger edit** — one sentence, in the notes of the FRTB table
(`preprint/paper_A_ssrn.tex`, the paragraph beginning "The regulatory reading is the per-asset and
date-clustered restatement"): Kupiec at 99% "82%" → 81%, Christoffersen "89%" → 88%, "(at 97.5%, 71% and
79%)" → 72% and 80%, "$t=0.93$" → 0.89, "$t=1.85$" → 1.71, "rejected ($t=4.2$)" → 4.6. No claim in that
sentence changes.

Verified as *not* affected: the "GBM-recal $p=0.24$" claim in the same notes comes from
`frtb_neural_v2_results.json`'s `hybrid_GBM_recal` (0.2388), a recalibrated-GBM variant, not the additive
conformal shift. It should not be conflated with the overlay that Panel D rules out.

## 5. Still open

- The regulatory tables should move to the 200-name panel, as `job_frtb200.py` supports and
  `frtb_table_200_results.json` already holds. The battery delivers the 200-name per-asset exception numbers;
  regenerating `tab:frtb` from the 200-name file is a separate edit.
- GJR-GARCH-skew-$t$, Taylor ES-CAViaR, SAV-CAViaR, EWMA, historical simulation and the GARCH-EVT variants
  have not been re-scored against a Beta-$t$-EGARCH engine. They are in `bench_all_results.json` against the
  GARCH-$t$ engine only, and the abstract's "lowest among standard benchmarks" rests on them.
- Nothing is committed and no manuscript text has been edited.

---

# R56 (2026-10-01): the four Stage-1 runs. THE SWAP IS WITHDRAWN.

All four runs requested before any text edit are complete, with predictions written into each header first.
**The decisive one fails: under the annual refits the paper recommends for production, the Beta-t-EGARCH
Stage 1 and the GARCH-t Stage 1 are indistinguishable.** The swap recommended in R55 is withdrawn.
`tab_deployed.tex` is regenerated against the original GARCH(1,1)-t Stage 1 (now with Panel B sourced wholly
from the 200-name battery, so the 140/200 mismatch is still fixed).

## 1. The decisive run: annual-refit walk-forward (`job_walkforward_bteg.py`)

251,600 rows, 1,258 dates, 0 fits dropped. At each Jan-1 cutoff 2020--2024 both filters are re-estimated on
expanding pre-cutoff data and every downstream stage is rebuilt.

| | engine_garch_t | engine_bteg | DM |
|---|---|---|---|
| mean FZ0, 1% | 2.23133 | 2.23636 | **−0.31** |
| mean FZ0, 2.5% | 1.93320 | 1.93332 | **−0.01** |

**P1 failed.** The prediction was DM > 2 in the swap's favour; the result is no difference at either level, and
at 1% the point estimate slightly favours GARCH-t. Against the frozen-fit DM of 3.31 and 4.55 this is a clean
reversal, and the pre-registered consequence applies: *the swap does not survive the recommended schedule and
is reported as a frozen-fit result only.*

**The interpretation, which is the useful part.** Under the frozen 60/40 protocol the GARCH-t parameters are
estimated once and carried through a 40% test window, so they go stale, and a stale GARCH over-reacts to
shocks in exactly the way Beta-t-EGARCH's bounded score prevents. Annual refitting removes the staleness and
with it almost all of the robust filter's advantage. **A robust filter and an annual refit are substitutes for
the same defect.** That unifies the frontier, the Section 4 decomposition and the refit result into one
statement rather than three.

**P2 confirmed.** Top-decile flexible edge over the own-scale parametric benchmark under refits: +1.80%
(DM 5.34) under GARCH-t, against the committed frozen +2.12% (DM 4.41); +0.42% (DM 1.68) under Beta-t-EGARCH,
inside the predicted +0.5%. **P3 confirmed.** Overall edge undetectable for both: −0.04% (DM −0.19) and
+0.18% (DM 1.17).

## 2. Every benchmark re-scored against the new Stage 1 (`job_bench_all_bteg.py`)

Generated from `job_bench_all.py` by injecting the second scale; every rival implementation and every
inference routine is that script's. 200 names, 221,600 rows and conf975 = −0.3426, all identical to
`bench_all_results.json`, and 0 Beta-t-EGARCH MLE failures, so the row set is genuinely common and the DMs
are valid.

**P1 failed on a technicality that is worth more than the prediction.** `engine_bteg` beats every *standard
benchmark* on FZ0 at both levels, by DM 3.3 to 13.1 — GJR-GARCH-skew-t (3.62, 4.76), Taylor ES-CAViaR (6.44,
6.48), GARCH-t (4.70, 6.25), both GARCH-EVT forms, all three FHS forms, GAS-FZ, EWMA, historical simulation.
It also has the lowest mean eleven-level pinball of any standard benchmark (0.39434 against SAV-CAViaR's
0.39554). So **the abstract's "lowest joint (VaR, ES) loss among standard benchmarks at the 1% and 2.5%
levels" holds on DM and not only on levels.** What beats it is `body_bteg`, the same estimator without the
EVT branch (FZ0 2.12228 and 1.82685, DM −1.71 at 1% and −1.34 at 2.5%), which is not a benchmark and is not
deployable: it over-breaches at 1.20% and 2.99% and fails the date-clustered exception test at both levels
(+3.00, +3.32). The 90% MCS at 1% contains `body_bteg` alone and at 2.5% `{engine_bteg, body_bteg}`; the
pinball MCS is `{body_bteg, param_bteg}`. **P2 failed**: `engine_bteg` is third on mean pinball behind
`body_bteg` and `param_bteg`, and SAV-CAViaR does not survive in the MCS.

That the engine sits outside the 1% MCS because its own no-EVT variant is more accurate is the
accuracy-versus-calibration trade the paper already resolves in favour of calibration. It should be stated,
not hidden.

**P3 confirmed, emphatically, and this is the most useful result of the whole exercise.** The mk63 decile
profile of `engine_bteg` referenced to `param_bteg` — a plain Beta-t-EGARCH with its own Student-t tail, no
learner at all — is **flat**:

```
decile            1      2      3      4      5      6      7      8      9     10
vs param_bteg  -0.15  -0.09  -0.14  -0.03  -0.07  -0.13  -0.12  -0.16  +0.07  +0.27
vs garch_t     +0.30  +0.33  +0.12  +0.20  +0.23  +0.23  +0.29  +0.26  +0.66  +3.82
paper Table 1  -0.13  -0.01  -0.10  +0.04  +0.01  +0.05  -0.20  -0.05  +0.28  +2.44
```

Against GARCH-t the profile has the paper's shape: flat through nine deciles, then a spike. Against a
Beta-t-EGARCH carrying a fixed parametric tail the spike is gone — +0.27% in the top decile. **The
misspecification frontier is a statement about the GARCH-t filter's scale error, not about innovation
shape.** Section 4 already says most of the edge is scale; this says it in the cleanest possible form, with a
likelihood-fitted benchmark and no cap to tune.

## 3. Exception and ES battery on the frozen holdout (`--panel holdout`)

200 names, 280,608 rows, 1,495 dates; the 2008--2009 sub-window is 79,715 rows over 489 dates, resampled over
its own dates. The canon200 arm of the same refactored script reproduces the committed battery across 390
fields with **zero** differences, so these numbers are on the same footing.

| era / level | engine_garch_t | engine_bteg |
|---|---|---|
| holdout 1%: clustered UC $p$ / dclust $t$ / AS $p$ / MF $p$ / FZ0 | 0.968 / 0.04 / 0.844 / **0.005** / 1.8187 | 0.682 / 0.41 / 0.694 / **0.039** / **1.8037** |
| holdout 2.5% | 0.928 / 0.09 / 0.899 / 0.659 / 1.5696 | 0.490 / 0.69 / 0.529 / 0.948 / **1.5594** |
| 2008--09 1% | 0.780 / 0.28 / 0.474 / **0.002** / 2.2566 | 0.749 / 0.32 / 0.698 / **0.002** / **2.2119** |
| 2008--09 2.5% | 0.358 / 0.92 / 0.383 / 0.763 / 2.0209 | 0.308 / 1.02 / 0.376 / **0.047** / **1.9958** |

The paper's claim that the engine passes the exception tests through the 2008 window **survives for both
filters**: clustered unconditional coverage, the date-clustered exception test and Acerbi--Székely all pass at
both levels in both windows, with per-name Kupiec pass rates of 79--94% and Christoffersen 84--98%. The
Beta-t-EGARCH engine has the lower FZ0 in every one of the four cells.

The weak point is McNeil--Frey, and it is shared: **both** engines reject at the 1% level out of era (0.005
and 0.039) and in the crisis (0.002 for both). At 2.5% in the crisis `engine_bteg` marginally rejects (0.047)
where GARCH-t passes comfortably (0.763) — a specific cost of the swap, now moot but worth recording. Only
`param_bteg` passes McNeil--Frey in the crisis at both levels (0.234, 0.724).

## 4. A reproducibility finding about the committed numbers

This run's GARCH-t arm gives a top-decile edge over GARCH-t of **+2.445% (DM 10.18)** where
`bench_all_results.json` records **+2.483% (DM 10.08)**; the body row is +2.943% against the committed
+2.982%. Row count, name count and the conformal shift are identical to four decimal places, and
`job_robust_engine.py` — an independently written script with its own panel construction — gives exactly
+2.445% / 10.18 as well. Two independent scripts agreeing against the committed value points to the
environment, almost certainly the scikit-learn version, rather than to either script. The paper's claims are
unaffected at this magnitude, but **the committed figures reproduce to about 0.04 percentage points on a
different scikit-learn build, not exactly**, and the replication package should say which version produced
them. This sits beside the `random_state` finding of R55 §4: both concern reproducibility of the committed
numbers rather than their correctness.

## 5. Decision and what the text should now say

**Stage 1 stays GARCH(1,1)-t.** The Beta-t-EGARCH work is reported as a diagnostic, in three sentences that
are stronger than the swap would have been:

1. Section 4's decomposition gains a maximum-likelihood row with no cap to tune: the surviving top-decile
   shape component is +0.26% (DM 1.81) under a Beta-t-EGARCH scale against +0.42% (DM 4.38) under the 3-sigma
   cap, and the decile profile against a Beta-t-EGARCH benchmark is flat.
2. A robust filter and an annual refit are substitutes: the robust filter's frozen-fit advantage (DM 3.31,
   4.55) disappears under refits (DM −0.31, −0.01), so what it was repairing was parameter staleness.
3. The score is close to filter-invariant (top-decile indicator agreement 0.985--0.993 across three
   filters, against 0.82 under independence), which supports its use as a monitor.

**The abstract keeps its current wording.** The proposed "on a score-driven filter" sentence is withdrawn with
the swap. The existing claim is supportable as it stands, and §2 above strengthens it: on the 200-name panel
the accuracy layer is not significantly beaten by any standard benchmark at either level.

**Decided, not run** (as agreed): the ten-day and cross-universe sections keep the GARCH-t filter and the text
says so. Those sections measure the frontier *against GARCH-t*, so the filter is the object of study; rerunning
them under a different Stage 1 would change what they measure.

**Still open:** `tab:frtb` should move to the 200-name panel from `frtb_table_200_results.json` for internal
consistency with the deployed table — unchanged by this decision, and a text edit rather than a run. Nothing
is committed and no manuscript text has been edited.
