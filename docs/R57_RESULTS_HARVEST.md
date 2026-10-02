# R57 results harvest (2026-10-01): Stage 1 stays GARCH-t; the disclosure, the seed-fixed regeneration, and tab:frtb on 200 names

Scope set by the R57 work order after the walk-forward result. **Stage 1 is not swapped.** Everything below
is either a correction to R56, a result already in hand that the order asked for, or a new run.

---

## 0. Correction to R56. Two things I got wrong.

**(a) I overstated my own pre-registered rule.** R56 said the swap was "withdrawn". The rule in the header of
`job_walkforward_bteg.py` reads, verbatim:

> `P1. THE DECISION. Under annual refits the Beta-t-EGARCH engine keeps its FZ0 advantage over the GARCH-t`
> `engine at both levels, DM > 2 in its favour, because the advantage comes from the filter's bounded`
> `response to a shock and not from parameter staleness, which refitting is what removes. If P1 fails,`
> `the swap does not survive the recommended schedule and must be reported as a frozen-fit result only.`

The prescribed consequence was *report it as a frozen-fit result*, not *withdraw it*. Escalating one into the
other was an error of mine, not a reading of the evidence.

**(b) R56 §5 made a claim about the wrong configuration.** It said "on the 200-name panel the accuracy layer
is not significantly beaten on FZ0 by any standard benchmark at either level". That result belongs to
`engine_bteg` (from `bench_all_bteg_results.json`). With Stage 1 staying GARCH-t the accuracy layer is
`engine`, and **that row is significantly beaten on frozen-protocol FZ0**, per `robust_engine_results.json`
(DM against `engine`, negative means the row is better):

| level | rows significantly beating the GARCH-t engine (DM < −1.645) |
|---|---|
| 1% | `bip_t_uncond` (−2.02), `bteg_uncond` (−1.75) |
| 2.5% | `bip_t` (−3.35), `bteg` (−2.53), `bteg_uncond` (−2.32), `bip_t_uncond` (−1.85) |

(Within-family rows also beat it: `body_bteg` −5.87/−8.43, `engine_bip` −5.63/−7.73, `engine_bteg`
−3.31/−4.55, `body_bip` −2.30/−3.29, and at 2.5% `body` −1.89.) **"Not significantly beaten by any standard
benchmark" does not hold for the deployed configuration and must not be written.** The paper owes a
disclosure sentence; its strength is what the annual-refit run decides (§1).

---

## 1. The disclosure run — annual refit, engine_garch_t against the robust filters

`job_walkforward_bteg.py` extended to three Stage-1 scales (GARCH-t, bounded-news, Beta-t-EGARCH), with a
pooled-unconditional-residual-quantile row for each (`garch_t_uncond` is FHS on this panel) and the ES tests
for each engine. The bounded-news filter reuses the GARCH-t fit, and nothing was added to the dropna set, so
the existing rows must reproduce exactly — the rerun doubles as a determinism check.

251,600 rows, 1,258 dates, 0 fits dropped. **Determinism check passes**: every value the first run also
produced is bit-identical (14 shared quantities, 0 differences; same rows, dates and drop count), so the
panel is unperturbed and the added rows are trustworthy.

### The verdict is split, and the disclosure sentence must say so

DM against `engine_garch_t`; positive means that row is **worse** than the engine, below −1.96 means it
significantly **beats** the engine.

| level | row | FZ0 | overall DM | top mk$_{63}$ decile DM |
|---|---|---|---|---|
| 1% | bip_t | 2.24880 | +1.17 | −1.42 |
| | bip_t_uncond | 2.23783 | +0.59 | −1.53 |
| | bteg | 2.25573 | +1.21 | **−3.70** |
| | bteg_uncond | 2.24601 | +0.86 | **−3.97** |
| 2.5% | bip_t | 1.94304 | +1.01 | **−2.00** |
| | bip_t_uncond | 1.93774 | +0.59 | **−2.58** |
| | bteg | 1.94402 | +0.85 | **−5.24** |
| | bteg_uncond | 1.94396 | +0.88 | **−5.41** |

**P4 confirmed.** Overall, not one of the four significantly beats the engine under annual refits at either
level. Every overall DM is *positive* — the robust rows sit behind the engine on the point estimate, by
+0.59 to +1.21 — so the overall claim is slightly stronger than a tie.

**P5 confirmed.** In the top misspecification decile the robust rows remain ahead: significantly at 2.5% for
all four (−2.00 to −5.41) and at 1% for the two Beta-t-EGARCH rows (−3.70, −3.97). Refitting removes
parameter staleness, which drove the overall gap; it does not remove post-shock overshoot, which is what the
score's top decile selects on.

**So the disclosure cannot be unconditional.** The supportable sentence is of the form: *measured over all
rows, the robust-filter benchmarks that beat the engine under the frozen fit are no longer ahead under the
annual refits the paper recommends; in the top decile of the score they remain ahead.* Writing only the first
half would be wrong.

### A refinement to R56 that matters

R56 said that under annual refits the two engines are "indistinguishable (DM −0.31 and −0.01)". That is true
**overall and only overall**. By region:

| level | engine_bteg vs engine_garch_t, overall | top mk$_{63}$ decile |
|---|---|---|
| 1% | +0.31 (bteg marginally worse) | **−5.37** (bteg better) |
| 2.5% | +0.01 (dead level) | **−7.25** (bteg better) |

The average hides a large, highly significant top-decile difference. `engine_bip_t` behaves the same way and
is in fact the lowest-FZ0 engine under refits at both levels (2.22939 and 1.92915), beating `engine_garch_t`
by DM −2.18 and −3.48 in the top decile though not significantly overall (−0.20, −0.68). The staleness story
survives, but its scope narrows: **refitting substitutes for a robust filter on average, not in the states
the score flags.**

### P6 failed, and the failure is shared

Predicted: engine_bteg's ES tests do not reject under refits. Acerbi--Székely passes comfortably (p = 0.334
at 1%, 0.279 at 2.5%), but **McNeil--Frey rejects at 1% (p = 0.014)** and barely passes at 2.5% (p = 0.059).
This is not specific to the swap: under annual refits *every* engine rejects McNeil--Frey at 1% —
`engine_garch_t` 0.020, `engine_bip_t` 0.015, `engine_bteg` 0.014 — and the same pattern appeared on the
frozen holdout. The ES-residual test on the 1% tail is the consistently weak diagnostic of this estimator
family under both out-of-era and refit protocols, and the limitations section should say so rather than let
it surface at review.

Exception coverage under refits is otherwise sound: clustered UC p of 0.226, 0.174 and 0.395 at 1% and 0.126,
0.089 and 0.303 at 2.5% for the three engines, none rejecting.

### Flexible shape over its own-scale parametric benchmark, three scales under refit

| scale | top decile | overall |
|---|---|---|
| GARCH-t | +1.80% (DM 5.34) | −0.04% (DM −0.19) |
| bounded-news | +0.58% (DM 3.68) | +0.41% (DM 3.38) |
| Beta-t-EGARCH | +0.42% (DM 1.68) | +0.18% (DM 1.17) |

The same ordering as the frozen protocol: the better the scale, the less the flexible shape adds in the top
decile. Under the bounded-news scale the learner earns a small but significant overall edge (+0.41%,
DM 3.38) that it does not have under GARCH-t.

---

## 2. Already in hand, from the work order's earlier list

**Cold start is Stage-1-independent — nothing reruns.** `job_coldstart_peers.py` calls `arch_model` in exactly
one place, to build the `own_garch_t` *rival* at ages past 250 days. Its pooled models use EWMA(0.94)
peer-group scales and an own 20-day rolling volatility; no GARCH or Beta-t-EGARCH filter enters the estimator
at any age. The cold-start numbers are unaffected by the Stage-1 question.

**engine_bteg beats its own parametric twin on FZ0**, which the order flagged as the decision point for the
abstract clause had the swap gone ahead: DM +4.31 at 1% and +5.35 at 2.5% (`bench_all_bteg_results.json`),
comfortably past the 1.645 threshold. Moot for the text now that Stage 1 stays GARCH-t, but it settles the
question that was asked.

**Holdout, engine_bteg against GARCH-t and FHS**, also already run (`robust_engine_results_holdout.json`):
DM +3.24 and +3.83 against GARCH-t, +2.74 and +3.26 against pooled FHS, at 1% and 2.5%. Exception counts for
both engines, including the 2008--2009 window on its own dates, are in
`exception_battery_results_holdout.json`.

---

## 3. Seed-fixed regeneration, and every number that moves

`random_state=0` is now set at all three HistGradientBoostingRegressor sites in `job_perasset_v2.py` and
`job_engine_esbt.py`, which also honour `GBC_PROJ`. The regenerated result files are installed in
`results/paper/`. **62 numeric fields move; exactly one significance verdict changes.**

### The verdict that changes

`engine_esbt`, 2.5%, accuracy layer: **Acerbi--Székely $Z_2$ $p$ moves 0.0394 to 0.0515**, crossing 0.05 from
rejecting to not rejecting. It moves in the paper's favour, which is why it is recorded here as a consequence
of a seed fix rather than presented as a finding. $Z_2$ itself moves −0.1082 to −0.1013.

### `job_perasset_v2.py` — 27 fields, no verdict changes

Conformal shifts −0.5019 to −0.5510 (1%) and −0.3518 to −0.3553 (2.5%); GPD $\xi$ 0.2809 to 0.2808, $\beta$
0.6353 to 0.6354. By variant:

| variant | breach99 | Kupiec99 | Christoffersen99 | dclust99 | breach975 | Kupiec975 | Christoffersen975 | dclust975 |
|---|---|---|---|---|---|---|---|---|
| raw_hybrid | .0132→.0135 | .686→.714 | .736→.729 | 4.23→4.63 | .0334→.0332 | .564→.571 | .607→.607 | 5.13→5.00 |
| hybrid_EVT | .0106→.0105 | .821→.814 | .893→.879 | 0.93→0.89 | .0276→.0273 | .707→.721 | .786→.800 | 1.85→1.71 |
| hybrid_EVT_conf | .0060→.0057 | .657→.607 | .712→.705 | −10.73→−11.86 | .0163→.0161 | .479→.493 | .529→.536 | −9.79→−10.08 |

All six pass/fail verdicts hold: `hybrid_EVT` passes the date-clustered test at both levels, `raw_hybrid` and
`hybrid_EVT_conf` fail at both.

### `job_engine_esbt.py` — 35 fields

conf975 −0.3354 to −0.3275, which now agrees exactly with `exception_battery_results.json`. Accuracy layer at
1%: breach .0103→.0102, n_breach 2283→2254, Kupiec $p$ .1546→.4185, $Z_2$ −.0674→−.0576 ($p$ .2652→.3390),
MF exres −.1378→−.1524 ($p$ .0377→.0220), FZ0 2.14749→2.14711. At 2.5%: breach .0273→.0272, n_breach
6060→6018, $Z_2$ −.1082→−.1013 (**$p$ .0394→.0515**), MF $p$ .2581→.2354, FZ0 1.84892→1.84871. DM of
garch_norm, garch_t and fhs against the engine move by 0.03 to 0.14.

### Printed in the manuscript and needing a text edit

One sentence, in the notes of the FRTB table (`preprint/paper_A_ssrn.tex`, the paragraph beginning "The
regulatory reading is the per-asset and date-clustered restatement"): Kupiec at 99% "82\%" → 81\%,
Christoffersen "89\%" → 88\%, "(at 97.5\%, 71\% and 79\%)" → 72\% and 80\%, "$t=0.93$" → 0.89, "$t=1.85$" →
1.71, "rejected ($t=4.2$)" → 4.6. No claim in that sentence changes.

### Environment, now pinned in the README

Python 3.10.12, numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, **scikit-learn 1.7.2**, arch 8.0.0. This matters
beyond housekeeping: `bench_all`'s GARCH-t arm reproduces here to about 0.04 percentage points rather than
exactly (top decile +2.445\% at DM 10.18 against the committed +2.483\% at 10.08), with two independently
written scripts agreeing on the former. The committed figures were produced on a different scikit-learn build.

---

## 4. tab:frtb on the 200-name panel

Generated by `code/paper/make_table_frtb200.py` into `submission/tables/tab_frtb200.tex` from the existing
`frtb_table_200_results.json`. 200 names, 221,600 asset-days, 1,108 dates — the panel of Table 1, so the
battery and the frontier finally share one set of rows. Compiles clean.

**The row ordering is unchanged**, exactly as that job's own prediction said. Average pinball rises by about
0.0125 for every row, which is a panel effect and not a model effect. What does change is the **Kupiec$_{99}$
pass set**, because the pooled test gains 40\% more observations:

| model | avg pinball | breach99 | Kupiec99 $p$ |
|---|---|---|---|
| Residual-hybrid (GBM) | 0.3551 → 0.3676 | 1.12 → 1.06\% | 0 → 0.0025 |
| \quad + EVT tail | 0.3555 → 0.3680 | 0.91 → 0.88\% | 0.0001 → 0 |
| GARCH(1,1)-$t$ | 0.3561 → 0.3688 | 1.06 → 1.03\% | **0.026 → 0.167** |
| GJR-GARCH-skew-$t$ | 0.3562 → 0.3688 | 1.05 → 1.02\% | **0.062 → 0.431** |
| GARCH-FHS (per-name) | 0.3562 → 0.3689 | 1.11 → 1.08\% | 0 → 0.0003 |
| GARCH-FHS (pooled) | 0.3566 → 0.3691 | 0.99 → 0.97\% | **0.794 → 0.206** |
| GARCH-FHS (rolling 500d) | 0.3569 → 0.3694 | 1.09 → 1.10\% | 0.0004 → 0 |
| EWMA (RiskMetrics) | 0.3606 → 0.3735 | 1.63 → 1.61\% | 0 → 0 |
| Historical simulation | 0.3619 → 0.3752 | 0.84 → 0.87\% | 0 → 0 |

GARCH-$t$ crosses from rejecting to not rejecting at the 5\% level (0.026 → 0.167). Any sentence that leans on
GARCH-$t$ failing Kupiec at 99\% needs checking against the new panel.

---

## 5. Still open

Nothing is committed and no manuscript text has been edited. §3's text edit and §4's
table swap are both text edits against files that now exist.
