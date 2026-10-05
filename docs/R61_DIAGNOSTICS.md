# R61 diagnostics: is the joint-loss result partly a construction effect?

Read-only. No manuscript edits, no new estimator, nothing written to `results/`. The engine, body and GARCH-$t$
rows are built by `tools/r61_diag.py`, derived from `code/paper/job_garch_evt.py` so the build is the one that
reproduces the canonical accuracy layer bit-for-bit — the converged ES comes back at −3.84505 and −2.85324 and
`conf975` at −0.3426, as in `bench_all_results.json`. Panels: `canon200` (200 names, 221,600 rows, 1,108 dates)
and the 2000–2013 holdout (200 names, 280,608 rows), the latter the same row count as the committed battery.

---

## 1. The envelope hypothesis is confirmed, and the operative variable is the *depth* of the override

**The claim under test.** Both models carry the GARCH-$t$ scale, which is too wide after a shock. The flexible
body learns to pull its standardised tail quantile **in** in those states. The min-envelope of equation (5) then
takes the minimum of that body and an **unconditional** pooled GPD, which can tighten but never loosen. So in
exactly the states where the body corrected the scale error, the envelope puts the unconditional tail back.

canon200, by mk$_{63}$ decile. `body−engine` is the FZ0 differential; **negative means the body beats the
engine**, since DM is signed on `row − engine`. `override` is $z_{\text{body}} - z_{\text{EVT}}$ in the rows
where the EVT branch binds — how much tail the envelope adds back.

| α=1% decile | breach eng | breach body | FZ0 body−eng | DM | EVT binds | $z_{\text{body}}$ | override |
|---|---|---|---|---|---|---|---|
| 1 | 1.17% | 1.38% | **+0.0112** | +1.25 | 50.5% | −2.527 | 0.188 |
| 5 | 1.05% | 1.26% | +0.0017 | +0.19 | 58.4% | −2.477 | 0.232 |
| 8 | 1.07% | 1.44% | +0.0031 | +0.81 | 65.9% | −2.369 | 0.355 |
| 9 | 0.79% | 1.18% | −0.0285 | −3.21 | 64.7% | −2.342 | 0.404 |
| **10** | **0.79%** | **1.14%** | **−0.0680** | **−6.42** | 66.4% | **−2.252** | **0.523** |

$z_{\text{EVT}}$ is a **constant** −2.554 at 1% and −1.880 at 2.5%: the pooled GPD is unconditional, so it
cannot follow the body. Across the ten deciles the body pulls its tail in by 0.275 in $z$ (−2.527 → −2.252)
while the EVT branch does not move, so the override deepens 2.8× (0.188 → 0.523). At 2.5% the same: the body
moves 0.218, the override deepens 3.2×, and `body−engine` goes +0.0103 → −0.0602 (DM +1.75 → −6.77).

**What drives the loss is the depth, not the frequency.** Across deciles:

| correlation with `body−engine` | 1% | 2.5% |
|---|---|---|
| override depth | **−0.840** | **−0.879** |
| $z_{\text{body}}$ at α | −0.786 | −0.875 |
| share of rows where EVT binds | −0.466 | −0.747 |

The bind *share* rises only from 50% to 66% at 1% — it is not where the action is. Conditioning on "the body
shrank relative to the parametric shape **and** EVT binds" adds nothing over "EVT binds" (66.3% against 66.4%),
because whenever the pooled GPD binds, the body has almost always shrunk. **The hypothesis holds in the form
that the override's magnitude, not its incidence, tracks the FZ0 cost.**

**Coverage says the same thing, and locates where the branch earns its place.** The EVT branch exists because
the body over-breaches (1.32% and 3.25% overall). By decile, the engine's breach is closer to nominal in **8 of
10** deciles at 1% and **9 of 10** at 2.5% — the exceptions are deciles 9–10 at 1% and decile 10 at 2.5%. In the
top decile the body breaches 1.14% against nominal 1% while the engine breaches 0.79%: the body is 0.14pp off,
the engine 0.21pp off *on the other side*. So the override fixes over-breaching in the bulk and **over-corrects
into over-coverage in the top decile**, which is exactly where it costs FZ0.

**Holdout replication, same sign and weaker.** `body−engine` −0.0148 (DM −1.46) at 1% and −0.0191 (DM −1.99) at
2.5%, with the correlation against override depth −0.75 and −0.84. The body pulls its tail in by only 0.085
there against 0.275 on canon200, so the override deepens 1.7× rather than 2.8× and costs less. The mechanism is
the same and its size scales with how much the body corrects.

### What this means for the paper

Two things, and the second is the useful one.

- The published statement that the joint-loss edge is **not** concentrated in high-score states is, in part, a
  statement about the splice rather than about state dependence. Without the EVT branch the joint-loss edge over
  GARCH-$t$ *is* concentrated: the body's gap is roughly ten times larger in the top decile than elsewhere. The
  envelope erases most of that top-decile advantage and changes little in the bulk.
- The design was not an error — the branch is what fixes the body's over-breaching across eight or nine deciles.
  But the trade is almost entirely in the top decile, and the paper does not say so.

---

## 2. The holdout design effect of 824 is serial correlation, not cross-name clustering

The committed battery reports, at 1% on the holdout, a design effect of **823.81** for the engine against
**213.2** for GARCH-$t$, and `rho_intradate_implied` of **4.41**. A correlation above 1 is impossible, which is
the tell: the design effect is iid variance over the **Newey–West(10)** date-clustered variance, so it folds two
different things together, and the formula $DE = 1 + (\bar n - 1)\rho$ assumes only one of them.

Decomposing it on the same rows:

| | engine | GARCH-$t$ |
|---|---|---|
| battery design effect (iid / NW(10)-clustered) | 823.81 | 213.2 |
| **within-date design effect only** | **31.3** | **34.4** |
| implied intra-date $\rho$, within-date only | **0.162** (valid) | **0.179** (valid) |
| serial-correlation inflation, $(\mathrm{SE_{NW}}/\mathrm{SE_{within}})^2$ | **25.3** | **5.0** |
| product, against the reported figure | 31 × 25.3 = 792 vs 824 | 34 × 5.0 = 171 vs 213 |

**The hypothesis that a shared pooled GPD makes names breach together is not supported.** Within-date
clustering is essentially identical for the two models — $\rho$ 0.162 for the engine against 0.179 for GARCH-$t$,
and the engine's is the *lower* of the two. The entire fourfold gap in the reported design effect is
**persistence in time**: the engine's per-date breach series carries five times more Newey–West inflation. Its
breaches arrive in runs of consecutive days, which is consistent with item 1 — a tail pinned to an unconditional
level stays pinned for the length of a volatility regime, where GARCH-$t$'s VaR re-adapts daily.

**Power.** Backing the clustered standard error out of the reported breach and $t$:

| holdout, 1% | breach | $t$ | clustered SE | the test rejects only outside |
|---|---|---|---|---|
| engine | 1.08% | +0.15 | **0.53pp** | [0.00%, **2.05%**] |
| GARCH-$t$ | 1.05% | +0.20 | 0.25pp | [0.51%, 1.49%] |
| FHS pooled | 1.20% | +0.96 | 0.21pp | [0.59%, 1.41%] |

At 2.5% the engine's SE is 0.28pp, rejecting outside [1.95%, 3.05%]. **So a 99% VaR breached on 2% of days would
pass this test for the engine**, and the engine's test is 2.1× less powerful than GARCH-$t$'s on identical rows.
The holdout passes at 1% ($t = 0.15$) are weak evidence, and weakest precisely for the model the paper is
defending. Five dates carry 65% of the clustered variance and 28 dates see more than 10% of names breach.

**For the paper.** `tab_deployed`'s design-effect row is annotated "it implies an intra-date breach correlation
of a few per cent". That reading is right on canon200 (DE 8.57, $\rho$ 0.038) and wrong on the holdout, where
the same arithmetic gives 4.41. Either the row should report the within-date design effect, or the note should
say the figure includes serial dependence and is not an intra-date correlation.

---

## 3. McNeil–Frey by decile on the holdout: the over-severity is in the middle, not the tail

Mean exceedance residual, positive = ES **too conservative**, negative = not severe enough.

| α=1% | overall | d1 | d4 | d6 | d7 | d9 | d10 |
|---|---|---|---|---|---|---|---|
| engine | **+0.154** | +0.063 | +0.173 | **+0.280** | **+0.302** | +0.124 | **−0.027** |
| GARCH-$t$ | **−0.175** | −0.289 | −0.169 | −0.053 | −0.061 | −0.199 | −0.303 |
| FHS pooled | +0.124 | +0.070 | +0.129 | +0.238 | +0.238 | +0.054 | −0.030 |

**This does not support the reading that the holdout ES rejection is the envelope pushing the tail deep in
high-score states.** The engine's over-severity peaks in deciles 6–7 (+0.28, +0.30) and is **gone in the top
decile** (−0.027); at 2.5% the top decile is mildly negative (−0.065) while the middle is positive. The engine
and FHS share the same shape, which is the era effect the user identified. GARCH-$t$ fails in the opposite
direction at every decile, worst at the two ends (−0.289 and −0.303).

So three distinct statements, not one: the engine's 1% ES is too conservative in the middle of the score
distribution; GARCH-$t$'s is uniformly not severe enough; and in the top decile — where item 1's override is
largest — the engine's ES is the best calibrated of the three. The envelope's cost in the top decile is a
**VaR**-side and FZ0 cost, not an ES-severity one.

---

## 4. Seeding sweep: the class of error is not closed

R59 seeded five jobs and the commit message said so, but that was scoped to the five it touched. A sweep of
every script, counting a call as seeded if `random_state` appears either on the call or in a `**dict` it splats:

**README-named jobs that still have an unseeded `HistGradientBoostingRegressor`:**

| script | unseeded calls | what it backs |
|---|---|---|
| `frtb_table_canonical.py` | 2 | the 140-name FRTB battery, `frtb_table_results.json` |
| `job_stress_dm.py`, `frtb_stress_exact.py` | 2 each | the ten-day and stress sections |
| `job_wrds_holdout.py` | 1 | **Figure 1**, the 2000–2013 holdout |
| `job_walkforward.py`, `job_walkforward_hybrid.py` | 1 each | the annual-refit paragraph |
| `job_fz_strict_calibration.py` | 1 | the strict-split DMs (M07) |
| `job_pit_universe.py`, `job_calendar_split.py` | 1 each | point-in-time universe, calendar split |
| `job_nurel.py`, `job_tenday_diag.py`, `job_holdout_garch_evt.py` | 1 each | $\nu$-relation, ten-day diagnostics, holdout GARCH-EVT |
| `job_tenday_envelope.py` | 1 of 2 | ten-day envelope |
| `job_synthetic_truth.py` | 1 of 3 | Table OA.16 — **missed in R59** |

Also unseeded, not README-named: 30-odd exploratory scripts under `code/paper/` and `code/amortization/`.
No bare `np.random` or `default_rng()` without a seed in any README-named job.

**Exposure is narrower than the list.** sklearn's `HistGradientBoostingRegressor` draws randomness only when
it subsamples bin thresholds, above 200,000 training rows. Jobs training on canon200 (~800k rows) or the
holdout are exposed; jobs on the 30-name realized-measure panel (~50k) are deterministic regardless. The ones
that matter are therefore `frtb_table_canonical.py`, `job_wrds_holdout.py`, the two walk-forwards,
`job_pit_universe.py`, `job_calendar_split.py` and `job_synthetic_truth.py`.

**Recommendation.** Seed all of them, then rerun only those whose numbers the paper prints, and treat any
movement the way R59 treated Table 4's realized ES — as a reproducibility fix with its old and new values
recorded, not as a new result.

---

## If item 1 is acted on: what a pre-committed state-conditional splice test would look like

Not run here, and deliberately not designed by looking at which variant wins. Written so it can be registered
before anything is estimated.

**One construction, chosen in advance.** The diagnostic above points at a specific defect — an unconditional
tail overriding a conditional body by a margin that grows with the score — so the matching repair is to let the
splice depend on the state. Of the four candidates (score-conditional splice level $p_0(s)$; a cap on the
override depth; a GPD fitted on score-stratified residuals; no override above a score threshold), **the test
should name one before running**, because choosing after seeing four sets of DMs is a specification search over
the same rows that produced the diagnostic. The natural one is the score-conditional splice level, since it
keeps the estimator's form and moves a single scalar.

**Rule fixed before the run.** $p_0(s) = p_0^{\text{low}}$ below a score threshold and $p_0^{\text{high}}$
above it, both the threshold and the two levels chosen **on pre-2020 rows only** — the same discipline the
score gate used, where the threshold was the 90th percentile of mk$_{63}$ on pre-2020 data and was then frozen.

**Predictions, with numbers, written before estimation.**

1. **Top-decile FZ0 improves and the bulk does not degrade.** The engine's top-decile FZ0 falls by at least
   half the current body-minus-engine gap — at least 0.034 of the 0.068 at 1% and 0.030 of the 0.060 at 2.5% —
   while deciles 1–9 move by less than 0.005 with $|DM| < 2$. *Fails if the bulk degrades at all*, because the
   EVT branch is there to fix the bulk's over-breaching.
2. **Top-decile coverage moves toward nominal from below.** The engine's top-decile breach rises from 0.79%
   toward 1% at $\alpha=1\%$ and from 2.04% toward 2.5%, without exceeding nominal in either. *Fails if it
   overshoots*, which would mean the repair has simply become the body.
3. **Overall coverage is preserved.** The pooled breach stays within 0.1pp of 1.04% and 2.78%, and the
   date-clustered exception test still passes at 1%. *Fails otherwise*: a construction that buys top-decile FZ0
   with aggregate coverage is not an improvement.
4. **The mechanism variable moves.** Mean override depth in the top decile falls by at least half, from 0.523
   and 0.336. If FZ0 improves while the override depth does not fall, the gain is coming from somewhere else and
   the explanation in item 1 is wrong.
5. **Nothing on the holdout is claimed.** The holdout is scored once, after the canon200 decision, as a frozen
   check — not used to choose anything.

**Controls, reported as pass/fail before any result is read**, as `job_es_converged.py` does: the parametric and
empirical-tail rows unchanged; every pinball number unchanged, since the splice level touches only the tail
nodes used by VaR and ES; and the ES convention held fixed at the converged integral.

**Decision rule.** Adopted only if predictions 1, 2 and 3 all pass. Otherwise reported as a tested-and-rejected
variant, with its numbers, in the same place the static conformal overlay is reported — the paper already has a
precedent for reporting a repair that did not work.
