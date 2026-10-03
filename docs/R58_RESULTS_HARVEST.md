# R58 results harvest (2026-10-02): the five pre-registered jobs

Stage 1 of the paper's engine stays GARCH-t throughout; the robust-scale configuration is reported as an
additional result, not a swap. Every job carries its predictions and decision rules in its header, written
before it ran. Nothing is committed.

| # | job | status |
|---|---|---|
| 1 | `job_tail_vs_bteg.py` — where the flexible tail adds | done, both protocols |
| 2 | `job_scale_gate.py` — score-gated Stage-1 scale | done |
| 3 | `job_overlay_engine.py` — Stage 4, static and adaptive | done, both panels |
| 4 | `job_coldstart_amort.py` — cold start (a) and (b) | done |
| 5 | `job_tenday_envelope.py` — ten-day envelope | running |

---

## 1. Where the flexible tail adds on a robust scale

**The reframing holds, and the mechanism is the ES, not level dilution.** Pinball reported at each level
rather than averaged. DM against `param_bteg`; negative means the row beats it.

| | frozen 1% | frozen 2.5% | frozen central (0.25/0.5/0.75) | refit 1% | refit 2.5% |
|---|---|---|---|---|---|
| `engine_bteg` pinball | **−5.42** | **−3.39** | +1.52 / +0.54 / +1.71 | **−2.13** | −0.83 |
| `engine_bteg` FZ0 | **−4.29** | **−5.34** | — | **−3.35** | **−2.93** |

Frozen: tail-only pinball (the 1% and 2.5% levels alone) favours `engine_bteg` at **DM −4.67**, while the
eleven-level average favours `param_bteg` at DM +0.87, not significant. That is the dilution effect made
explicit — the flexible tail wins where it acts and is diluted by eight central levels where it cannot.

**P1 (refit) fails as written, 3 of 4.** The rule required DM > 2 on all four of: pinball at 0.01, pinball at
0.025, FZ0 at 1%, FZ0 at 2.5%. Under refits the first, third and fourth pass (−2.13, −3.35, −2.93); **pinball
at the 0.025 level is −0.83, a tie.** The pre-registered consequence is that the claim must be stated for
frozen fits only. The substantive reading is narrower than that: the *joint* (VaR, ES) claim — the one about
capital — passes at both levels under refits; it is the *VaR-only* comparison at 2.5% that is a tie. Both
should be said.

**P2 confirmed** (ties or loses at central levels, all |DM| < 2). **P3 confirmed** under frozen fits.

**Correction to what I reported yesterday.** I said P3 failed and that `param_bteg` beat `engine_bteg` on
tail-only pinball. That came from a run in which I had fitted the frozen-protocol filters on `cp` (45% of the
sample) instead of `sp` (60%), deviating from the canonical pipeline. With the window corrected the
conclusion reverses: `engine_bteg` wins tail-only pinball at DM −4.67. The earlier statement was my bug, not
a finding.

## 2. The score-gated scale works, and beats every fixed choice

Gate: the GARCH-t engine below a frozen mk63 threshold (the 90th percentile over pre-2020 rows, 13.095),
the Beta-t-EGARCH engine at or above it. Fires on 10.0% of rows. Annual refits.

| model | FZ0 1% | FZ0 2.5% | DM vs gate, overall (1% / 2.5%) | DM vs gate, top decile |
|---|---|---|---|---|
| **gate** | **2.22333** | **1.92574** | (reference) | (reference) |
| always engine_garch_t | 2.23133 | 1.93320 | +6.26 / +8.04 | +5.36 / +7.25 |
| always engine_bteg | 2.23636 | 1.93332 | +0.80 / +0.73 | −1.65 / −1.69 |
| always param_garch_t | 2.24204 | 1.94013 | +5.69 / +5.59 | +5.05 / +7.03 |
| always param_bteg | 2.25573 | 1.94402 | +1.63 / +1.45 | +2.28 / +2.58 |

**P1 confirmed.** The gate has the lowest FZ0 of all five rows at both levels, no fixed choice beats it
(every DM against it is positive), and it beats always-GARCH-t in the top decile at DM +5.36 and +7.25.
**P2 confirmed** (within noise of always-bteg overall). **P3 partly fails**: the gate passes the
date-clustered test at both levels, but its breach rate is not strictly between the two fixed engines'
(0.0290 at 2.5% against 0.0288 and 0.0278) — the premise was wrong, since a row-wise partition averages each
engine's rate on its own subset and need not lie between the two overall rates.

This is the result that most directly serves the paper's thesis: the score is good enough to *select the
scale*, and selecting beats any fixed filter. `param_garch_t` fails the date-clustered test at 2.5% (t 2.14).

## 3. Stage 4 repaired — the adaptive overlay

Decision rule set in advance: Stage 4 goes on only if the adaptive overlay lands within 0.3 points of nominal
at 2.5% **and** keeps FZ0 within noise of the accuracy layer, on **both** panels. It does.

| variant | canon200 1% | canon200 2.5% | holdout 1% | holdout 2.5% |
|---|---|---|---|---|
| engine (no shift) | +0.04pt, t 0.62 | +0.28pt, t **2.00 fail** | +0.08pt, t 0.15 | +0.07pt, t 0.25 |
| static, body-targeted (committed) | −0.43pt, t **−11.95** | −0.84pt, t **−9.09** | −0.08pt, t −0.92 | −0.27pt, t −1.00 |
| static, engine-targeted | −0.20pt, t **−4.20** | −0.35pt, t **−3.07** | +0.05pt, t −0.06 | −0.02pt, t −0.07 |
| **adaptive, γ = 0.05** | **+0.03pt, t 0.56** | **+0.10pt, t 0.78** | **+0.04pt, t −0.12** | **−0.12pt, t −0.44** |

FZ0 against the accuracy layer: −0.37 and −1.23 on canon200, +0.82 and −0.10 on the holdout — all within
noise, and the design-era point estimates favour the overlay. **Stage 4 switches on in adaptive form.**

**P3 failed, informatively.** The static engine-targeted shift does *not* stay mis-signed out of era: on the
holdout its shift is −0.02 and it behaves well. The static shift's failure is **era-specific**, not
structural — it breaks when the calibration window and the test era differ in regime (pre-2020 against
COVID) and is harmless when they do not. The adaptive form never has to transfer, which is why it works in
both. This supersedes R55 §3, which inferred a structural defect from design-era evidence alone.

## 4. Cold start

**P1 (as given) fails.** The characteristics-only amortized model loses to own-history at **every** bucket:
−4.21% at ages 1–5 (DM −1.22, against a 2-day own-vol benchmark available on 60% of those rows), −6.05% at
6–14 (DM −1.97), −7.55% at 15–60 (DM −4.28), −10.61% at 61–125 (DM −5.87). The pre-registered consequence
applies: the characteristics-only variant has no age region of its own and the day-one claim rests on
**availability**, not accuracy. **P3 confirmed**: the residual-hybrid is available on 0% of rows in all four
buckets.

**P5 fails at the bucket it was written for.** The age-bucket conformal shift moves `peer_young` from 7.11%
to 5.15% at the 1% level and 10.50% to 7.59% at 2.5% — still four to five points off nominal against a
predicted 0.5. It works from age 6 (0.94%/3.00% at 6–14). The registered precision caveat is the reason: only
72 names enter the panel genuinely before 2018, giving 330 calibration rows at ages 1–5, so the 1% shift is
about three order statistics, and 70 of those 72 are smallcaps where the test names are IPOs. A day-one
correction cannot be calibrated from this panel because the panel contains no comparable day-one listings
before the test period.

**Look-ahead in the published characteristics.** `annvol` correlates 1.0000 with full-sample realized
volatility and matches it exactly for 720 of 720 names; `logmcap` comes from a 2024-12-31 market-cap pull and
`beta` is full-sample. Removing `annvol` costs the characteristics-only model 9.26% of pinball at ages 1–5
and 6.18% overall; removing all three costs 8.15% and 8.73%. **Scope: none of `job_composite.py`,
`job_bench_all.py`, `job_frtb200.py`, `job_engine_esbt.py`, `job_perasset_v2.py` or `job_robust_engine.py`
reads the characteristics file** — the frontier, the battery, the FRTB table, the exception tests and the
holdout are untouched. The paper's own ablation caps all characteristics at +0.5% of pinball in the full
amortized model, so the transfer claim is not materially affected; the day-one claim is, and it was already
being restated. Six other scripts read the file; none of their result files is cited in the manuscript.

## 5. Ten-day envelope — running

---

## Error audit

Errors I made in this round and how each was caught, since several only surfaced because something checked:

| error | caught by | consequence |
|---|---|---|
| frozen filters fit on `cp` not `sp` in `job_tail_vs_bteg.py` | comparing levels against other jobs | **reversed a reported conclusion** (P3); corrected above |
| `TE['bucket']` used before assignment | runtime | job died, no bad numbers |
| calibration predictions read from an empty dict | pre-launch read-through | caught before running |
| `MU*H` broadcasting in the ten-day envelope | runtime | job died after 14 min of fitting |
| sign inverted in my first cold-start print | re-printing with the convention stated | would have reversed the headline; caught before reporting |
| stale global `Dn` after the era refactor | runtime | two battery runs lost ~20 min |
| R56 called the swap "withdrawn" | reviewer | overstated my own pre-registered rule |
| R56 §5 claimed the accuracy layer is unbeaten | reviewer | applied `engine_bteg`'s result to a GARCH-t configuration |

**Unresolved, and flagged rather than papered over.** The plain engine's date-clustered t at 2.5% is **2.00
(fail)** in `job_overlay_engine.py` and **1.60 (pass)** in `job_exception_battery.py`, on identical row counts
(200 names, 221,600 rows, 1,108 dates), with GPD thresholds differing in the third decimal (−1.880 against
−1.887). My first hypothesis — that the battery's stricter dropna shrinks the training pool — is **wrong**:
both rules give identical training rows on a 20-name check. The cause is open. Consequence: **the plain
engine's 2.5% date-clustered verdict straddles 1.96 and should not be leaned on in either direction.** The
adaptive overlay passes comfortably under both implementations, which makes it the safer row to report.
