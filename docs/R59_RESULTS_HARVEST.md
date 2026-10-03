# R59 results harvest (2026-10-02): look-ahead repair, exception comparison, gate verification, ES convergence

Four threads, all opened by objections rather than by a plan: the published amortization features were
partly forward-looking; the engine's unconditional-coverage rejection was mine to explain, not the era's;
the gate needed recomputation by code that shares nothing with the job that produced it; and every
numerically integrated ES in the paper carries a 20-node quadrature error large enough to flip a test.
Stage 1 stays GARCH-t. Nothing is committed.

| # | thread | status |
|---|---|---|
| 1 | point-in-time characteristics (`job_amort_pit.py`) | done, four predictions graded |
| 2 | GARCH-t and FHS exception rows, same battery code | done, both panels |
| 3 | independent gate recomputation (`gate_indep.py`) | done — reproduces to 5e-6, and corrects my framing |
| 4 | converged ES (`job_es_converged.py`) | running, 1 of 6 blocks |
| 5 | ten-day variance-ratio rescaling, recorded as exploratory | done |

---

## 1. The look-ahead is real, and removing it costs almost nothing

The objection was correct and my first scope check was too narrow: I searched `code/paper/` only, and the
cited amortization numbers come from `code/amortization/`. Three features were full-sample: `annvol`,
`beta`, `logmcap`. `job_amort_pit.py` rebuilds the first two point-in-time (trailing 250 days through
t−1; beta against an equal-weighted panel) and drops `logmcap`, which cannot be reconstructed from
returns alone. Same panel as published — 1,318,905 rows, 720 names, 288 held out — so this is like-for-like.

| | published | point-in-time |
|---|---|---|
| transfer win rate, held-out **names** | 99.3% | **99.0%** |
| … names younger than 250d | 95.5% | 93.8% |
| ablation: `no_realized_vol` (% worse than ALL) | 1.02% | 0.54% |
| ablation: `no_chars` (% worse than ALL) | 0.53% | 0.45% |
| edge over own history, d15–30 | 9.68% | **9.17%** |
| … d30–60 / d60–120 / d120–250 | 7.76 / 6.9 / 6.55 | 6.85 / 6.98 / 6.54 |
| mean pinball, ALL | 0.5506 | 0.5510 |
| mean pinball, `chars_only` | 0.5673 | **0.5630** |

**P1, P2, P4 confirmed. P3 failed, in the direction that favours the paper.** I predicted the
characteristics-only arm would lose more than 3% of its accuracy; it instead got *better* in absolute terms
(0.5673 → 0.5630). Replacing a static full-sample volatility with a trailing one hands the quantile model a
time-varying regressor, and that is worth more than the leakage was. The ALL arm moves by 0.0004, or 0.07%.

**So the abstract's "transfers to newly listed assets" stands**, and the first-month 6–10% figure survives as
6–9%. What does not survive is a stronger reading I had briefly entertained: an earlier isolation test of mine
measured what happens when `annvol` is *deleted* (+6.18% worse overall, +9.26% at ages 1–5). That is the
information content of volatility, not the value of the look-ahead. The honest counterfactual is replacement,
because a desk can always compute a trailing estimate — and under replacement the cost is ≈0.

**The 59–79% transfer win rate: searched the full mirror, and it is not there.** Before calling a published
number unsourced I searched `~/sean_dev/GBC_Project` (1.3GB, 3,141 files), not just this repository:

| searched | result |
|---|---|
| every `*amort*` script — `autojobs/`, `_github_private/`, `_github_public/`, `code/amortization/` | no script computes any win rate; `job_amort_agecurve.py` has a categorical `winner` per bucket, not a rate |
| every `*amort*` result file, all three copies | every value in [0.55, 0.85] is a pinball level; no win-rate field exists |
| every JSON in the mirror with a field named `win`/`share`/`frac`/`pct`/`beat`/`rate` and a value in range | 428 hits, all Kupiec/Christoffersen pass rates, edge percentages or coverage shares — none an amortization win rate |
| `autojobs/done/` logs, `logs/ai2_amort_done.txt`, all five `amort_*_console.txt` | age-curve ratios only |
| literal `59–79` across the mirror | five hits, **all prose**: `WEEKLY_WINS_2026-07-21.md:27`, `PAPER_PORTFOLIO.md:56`, `HALLUCINATION_SWEEP_2026-09-03.md:37`, and both `paper_plain*.txt` |

The earliest occurrence is the July weekly note, in a "Supporting wins (mention if asked)" line, already
without a source; that same note records the paper "absorbed all remaining orphan results … transfer rate", so
it entered the manuscript as an orphan. **The figure has no computational origin in either repository.**

Replacing it with the date win rate, which the paper already uses elsewhere ("lower loss on 74% of dates in the
top decile") and which needs no definition to be read:

| statistic | definition |
|---|---|
| **per-date** | share of test dates on which the amortized model's *cross-name mean* pinball beats own history, held-out names only |
| **per-name** | share of held-out *names* whose amortized mean pinball over that name's test rows beats own history |

Both are now computed, overall and by age bucket, carrying `n_dates` and the median names per date — and that
last column changes the recommendation.

| arm | overall | young (<250d) | per-name | per-name young |
|---|---|---|---|---|
| published look-ahead | 92.0% | 79.5% | 99.3% | 95.5% |
| **point-in-time** | **91.7%** | 79.2% | 99.0% | 93.8% |

2,767 test dates. By age bucket, point-in-time:

| bucket | dates | median names/date | date win rate | edge % |
|---|---|---|---|---|
| d15_30 | 1089 | **2** | 68.7% | 9.17 |
| d30_60 | 1473 | **2** | 68.5% | 6.85 |
| d60_120 | 1851 | 3 | 73.8% | 6.98 |
| d120_250 | 2134 | 5 | 78.1% | 6.54 |
| d250_500 | 2189 | 10 | 81.0% | 5.12 |
| d500_1000 | 2189 | 14 | 85.1% | 5.76 |
| d1000_2700 | 1768 | 135 | 86.7% | 5.70 |

**Report the overall figure, not the by-bucket one.** The date win rate rises monotonically with the number of
names in the cross-section (2 → 135) while the edge *falls* (9.17% → 5.70%). That is a sample-size effect, not
a model effect: averaging two names' losses per date is close to a coin flip however good the model is, so the
young buckets' 68–69% measures how thin the cross-section is, not how well transfer works. Pooled, where the
cross-section is wide, it reads **91.7% of 2,767 dates**.

So: **91.7% overall as the date win rate**; the **age curve of edges (9.2% → 6.5%)** carries the cold-start
claim, since it does not degrade with thin cross-sections; and the per-name 99.0% only with its definition
attached.

**Which own-history benchmark, named exactly, so the sentence can name it too.** `job_amort_pit.py:54-57,85`
— at each row the benchmark is the **expanding-window mean and standard deviation of that name's own prior
returns** (`cumsum − current`, `cnt = age`, so strictly out-of-sample), and the quantile is
`own_mean + own_sd × z_τ` with **$z_\tau$ the standard normal quantile** (−1.645 at 0.05, −1.282 at 0.10,
−0.674 at 0.25 — normal, not Student-$t$). The loss is the mean pinball over **seven levels, 0.05 to 0.95**.
Held-out names only, age ≥ 12 days.

**This needs saying in the paper, because a reader will assume otherwise.** Every amortization number — the
91.7% date win rate, the 99.0% per-name rate, the 9.2% first-month edge — is measured against an
**expanding-window Gaussian at moderate quantiles**, not against a fat-tailed model and not at the regulatory
tail. There is no 1% or 2.5% level in that loss at all. So the amortization result is a statement about
learning distributional shape at moderate quantiles from cross-sectional data, not about tail risk; in a paper
whose other sections are entirely about the 1% and 2.5% tail, leaving the benchmark unnamed invites the reader
to transfer the claim to a place it was never tested. The July note already hedged in this direction
("benchmark is own EWMA-$t$/empirical, not full GARCH"); the benchmark actually used is weaker still.

One observation on provenance, offered as a conjecture and not reportable. The by-bucket date win rate spans
68.5–86.7% here, and the July note records the original benchmark as "own EWMA-$t$/empirical, not full GARCH" —
a weaker but better-calibrated comparator, against which win rates would be lower. A by-bucket date win rate
against the empirical benchmark is the most likely shape for "59–79%". The run that produced it does not exist
in either repository, so this stays a conjecture and the figure stays unusable.

**First-month edge, as the point-in-time numbers rather than a blurred range:** **9.2% at 15–30 days, then
6.5–7.0% from 30 to 250 days** (9.17 / 6.85 / 6.98 / 6.54 across d15_30, d30_60, d60_120, d120_250).

---

## 2. The engine's 2.5% over-breach is not the era's fault

Asked for, and the right test: GARCH-t and FHS through the *same fixed battery code* that gives the engine
its 2.00, on the canonical frozen `canon200` rows and on the holdout.

| panel | model | breach 1% | t 1% | breach 2.5% | t 2.5% | |
|---|---|---|---|---|---|---|
| canon200, 221,600 rows | engine | 0.0104 | 0.62 | 0.0278 | **2.00** | fail |
| | GARCH-t | 0.0103 | 0.48 | 0.0271 | 1.56 | pass |
| | FHS (pooled) | 0.0114 | 2.10 | 0.0300 | **3.35** | fail |
| holdout 2000–2013, 280,608 rows | engine | 0.0108 | 0.15 | 0.0257 | 0.25 | pass |
| | GARCH-t | 0.0105 | 0.20 | 0.0248 | −0.06 | pass |
| | FHS (pooled) | 0.0120 | 0.96 | 0.0281 | 0.98 | pass |

**My "the rejection belongs to the era" was wrong, and the numbers say why.** The era is clearly implicated —
all three models pass on the holdout and the two that fail do so only on `canon200`. But GARCH-t survives the
same era on the same rows, so the era cannot be the whole explanation: something in the engine's 2.5% quantile
over-breaches by 7bp more than its own Stage-1 filter does. The agreed sentence, which is both fairer and
carries more information than the one it replaces:

> On 2020–2024 the engine's 97.5% coverage is rejected marginally ($t=2.00$), GARCH-$t$'s is not ($t=1.56$),
> and pooled FHS is rejected more strongly ($t=3.35$); all three pass on the 2000–2013 holdout, and the
> adaptive overlay passes on both panels.

---

## 3. The gate reproduces exactly — and is weaker than I said

`gate_indep.py` imports nothing from `job_scale_gate.py`; FZ0 and the Newey–West t are rewritten from
scratch and read only the saved per-row forecasts. Mean FZ0 agrees with the job to **5e-6** on all five rows
at both levels, which is CSV rounding. The threshold 13.0953 was separately re-derived as the 90th percentile
of `mk63` over the 296,000 pre-2020 rows (latest date 2019-12-31). Fire rate in the test era is 9.97% overall,
but drifts: 6.7% in 2020–22, 13.3% in 2023, 16.4% in 2024.

| alpha | gate | engine_garch_t | engine_bteg | param_garch_t | param_bteg |
|---|---|---|---|---|---|
| 1% mean FZ0 | **2.22333** | 2.23133 | 2.23636 | 2.24204 | 2.25573 |
| 2.5% mean FZ0 | **1.92574** | 1.93320 | 1.93332 | 1.94013 | 1.94402 |

The gate does have the lowest mean FZ0 of all five at both levels, which is what I reported. But the DMs put
that in proportion, and one of them contradicts how I framed it:

| gate vs | 1% | 2.5% |
|---|---|---|
| always-`engine_garch_t`, overall | **+6.26** | **+8.04** |
| always-`param_garch_t`, overall | +5.70 | +5.59 |
| always-`engine_bteg`, overall | +0.80 | +0.73 |
| always-`engine_bteg`, **top mk63 decile** | **−1.65** | **−1.69** |

Positive means the fixed rule is worse than the gate. So the gate beats always-GARCH-t decisively and
always-Beta-t-EGARCH not at all — and in the top score decile, the region the gate exists to serve,
always-`bteg` is *better* than the gate. **The gate is a deployment compromise, not an optimum.** Its case is
that it keeps the paper's Stage-1 filter on 90% of rows while recovering most of the robust scale's gain; if
tail accuracy is the only objective, always-`bteg` dominates it. My earlier summary — "beats every fixed
choice" — was true of the point estimates and misleading about the comparison that matters.

---

## 5. Ten-day variance-ratio rescaling: exploratory

Recorded as exploratory per instruction, not as a result. On the 2014–2024 design (150 names, 164,850 rows,
1,099 dates; 110 non-overlapping):

| | pinball | vs `direct` | coverage 1% / 2.5% |
|---|---|---|---|
| `sqrt_h` (paper) | 1.2511 | +1.39%, DM 6.71 | 0.0124 / 0.0330 |
| `fhs_path` (4,000 paths) | 1.2398 | +0.49%, DM 2.48 (nonoverlap 1.07) | 0.0102 / 0.0258 |
| `direct` (pooled GBM) | 1.2336 | — | 0.0110 / 0.0276 |
| `envelope` = min(direct, √h) | 1.2478 | +1.14%, DM 3.91 | 0.0094 / 0.0245 |
| `rescaled` (variance ratio) | **1.2322** | −0.12%, DM −1.66 (nonoverlap −0.33) | **0.0097 / 0.0244** |

Two things worth saying and neither is a headline. The envelope buys coverage with accuracy: it is the
best-covered rule bar the rescaling and the worst-scoring of the three direct variants. The rescaling is the
best on both axes but its accuracy edge over `direct` is not significant on either the full or the
non-overlapping date set, so it is a tie with better calibration — exploratory, and no claim attached.

---

## 4. Converged ES

### The fix needed fixing first

v1 of `job_es_converged.py` integrated $[0,\alpha]$ with $M$ midpoint nodes throughout. The GPD quantile
diverges as $u^{-\xi}$ at the origin, so that rule converges at order $1-\xi \approx 0.62$:
ES(20), ES(200), ES(2000) came in at −3.78359, −3.83084, −3.84223, leaving $M=2000$ about **0.09% short of
the Richardson limit**. The pre-registered 1e-4 criterion was not met, and calling that column "converged"
would have been wrong. v2 integrates the sub-floor region $[0,\alpha/40]$ **in closed form** — the GPD
integral is elementary — and uses midpoint nodes only on $[\alpha/40,\alpha]$, where the integrand is
bounded. The closed form was unit-tested against a 4×10⁶-node substituted quadrature over 32
$(u,\xi,\beta,F)$ combinations, relative error below 2e-6.

**The two routes agree.** v2's answer lands within 0.02% of the Richardson limit extrapolated from v1's node
sequence, on all four blocks (−3.84505 vs −3.84585; −3.92569 vs −3.92644; −2.85322 vs −2.85388; −2.90670 vs
−2.90728). Two numerically unrelated methods converging on the same value is the cross-check.

### Controls: both pass exactly

| control | result |
|---|---|
| VaR-only statistics unchanged across $M$ | **PASS** — breach rates identical to 6 decimals at $M \in \{20,200,2000,\text{converged},\text{committed}\}$, all four blocks |
| closed-form rows unchanged | **PASS** — `param_garch_t` and `param_bteg` FZ0 identical to the fixed battery, abs diff 0.0e+00 at both levels |

### Results, canon200, 221,600 rows

| row | α | ES 20-node | ES converged | understated by | FZ0 20-node | FZ0 converged | ΔFZ0 |
|---|---|---|---|---|---|---|---|
| `engine_garch_t` | 1% | −3.78359 | **−3.84505** | 1.645% | 2.14731 | **2.14653** | −7.8e-4 |
| `engine_bteg` | 1% | −3.86291 | **−3.92569** | 1.648% | 2.13059 | **2.13077** | +1.8e-4 |
| `engine_garch_t` | 2.5% | −2.80693 | **−2.85322** | 1.661% | 1.84936 | **1.84863** | −7.3e-4 |
| `engine_bteg` | 2.5% | −2.85901 | **−2.90670** | 1.686% | 1.83176 | **1.83198** | +2.2e-4 |

Converged/20-node ES ratio with the cross-name spread, as asked:

| row | α | mean ratio | per-name sd | min | max | names > 1 |
|---|---|---|---|---|---|---|
| `engine_garch_t` | 1% | 1.01645 | 0.00252 | 1.00689 | 1.01893 | 100% |
| `engine_bteg` | 1% | 1.01648 | 0.00248 | 1.00489 | 1.01895 | 100% |
| `engine_garch_t` | 2.5% | 1.01661 | 0.00304 | 1.00542 | 1.01947 | 100% |
| `engine_bteg` | 2.5% | 1.01686 | 0.00191 | 1.00737 | 1.01900 | 100% |

The error is **systematic, not noise**: every one of the 200 names is understated, the spread is a quarter of
a percent, and the four block means sit within 0.04pt of each other.

### Convergence, separated as agreed

| | measured | criterion 1e-4 |
|---|---|---|
| $M$: $\lvert\mathrm{ES}(200)-\mathrm{ES}(2000)\rvert/\lvert\mathrm{ES}\rvert$ | 3.2e-5 – 4.5e-5 | **met** |
| $K$: halving the 60 fitted body levels | 4.3e-4 – 5.8e-4 | **not met** |

The separation held: `relK` was 5.8e-4 in v1 and is 5.8e-4 in v2, untouched by a change that cut the $M$
error by a factor of 500. The residual is 30× smaller than the 1.65% it replaces, and it measures how finely
the body curve is sampled — an estimator choice — rather than the quadrature. The committed estimator fits 20
levels; this runs 60.

### Acerbi–Székely by node count, as asked

| row | α | 20 nodes | 200 | 2000 | published |
|---|---|---|---|---|---|
| `engine_garch_t` | 1% | 0.2536 | 0.3839 | 0.3842 | 0.348 |
| `engine_garch_t` | 2.5% | **0.0256** | 0.0545 | **0.0546** | 0.058 |
| `engine_bteg` | 1% | 0.2489 | 0.1474 | 0.1472 | — |
| `engine_bteg` | 2.5% | 0.7980 | 0.5514 | 0.5508 | — |

McNeil–Frey over the same sweep: `engine_garch_t` 1% **0.0673 → 0.3822 → 0.3832**; 2.5% 0.3914 → 0.5855 →
0.5838; `engine_bteg` 1% 0.1643 → 0.6294 → 0.6307; 2.5% 0.7274 → 0.3469 → 0.3449.

### Predictions graded: P3 confirmed; P1, P2, P4 fail

**P3 confirmed.** The engine's McNeil–Frey $p$ at 1% converges to **0.3832**, far above 0.05, so the A8
limitation sentence shipped in `a1f997f` is **withdrawn, not softened**. Worth noting how the published 0.022
arose: the `mk63` battery bug moved it to 0.0673 and the quadrature moved it to 0.3832 — two independent
errors both pushing the same test toward rejection.

**P1 fails — but only among rows that were already statistical ties.** FZ0 moves by at most 7.8e-4, and every
*significant* comparison keeps its order (at 1%: `engine_bteg` 2.13077 < `param_bteg` 2.14295 <
`engine_garch_t` 2.14653 < `param_garch_t` 2.15572, unchanged; same at 2.5%). But `tab_bench_fz0` at 2.5%
prints the engine at 1.8494 against GJR-GARCH-skew-$t$ 1.8487 and ES-CAViaR 1.8490 — gaps of 7e-4 and 4e-4,
*the same size as the correction*. With the engine at 1.84863 it draws level with GJR and goes marginally
ahead of ES-CAViaR. Both were already DM ties (−0.28 and −0.12). The response should not be to restate the
ordering but to stop printing one: those rows are a tie group, and a four-decimal ranking among gaps smaller
than the convention error was never meaningful.

**A second convention defect, found while checking that.** `tab_bench_fz0`'s note calls the skew-$t$ ES a
closed form. It is not — `job_bench_all.py:155` integrates the skew-$t$ quantile with 200 midpoint nodes.
Measured against the exact Student-$t$ ES, a 200-node midpoint rule understates $\lvert ES \rvert$ by 0.10%
at $\nu=6$, 0.18% at $\nu=4.5$ and 0.04% at $\nu=12$. So the GJR row carries roughly 0.13% of the same error,
about a twelfth of the engine's: the note is wrong, and the row moves slightly in the same direction.

**P2 fails, narrowly, and its point stands anyway.** Acerbi–Székely at 2.5% moves 0.0256 → 0.0546, i.e. by
0.0290, against the 0.0298 spread of the three earlier specification readings (0.0394, 0.0515, 0.0217). The
prediction asked for strictly more and got marginally less. The substantive claim is unaffected: the move
crosses 0.05, and it is the same size as the spread across specifications, so the Acerbi–Székely verdict at
2.5% is not robust to either. Note too that the published 0.058 and the converged 0.0546 agree in verdict —
the printed figure was right by the cancellation of two errors.

**P4 fails half.** $\lvert ES \rvert$ rises for every affected row and every name, as predicted. FZ0 does
*not* fall for every row: it falls for the GARCH-$t$ scale and **rises** for Beta-$t$-EGARCH. The mechanism is
in the loss itself — a deeper ES raises $\log(-e)$ while shrinking both $v/e$ and the breach term, so it helps
a row that over-breaches (`engine_garch_t`, 1.04% / 2.78%) and hurts one that under-breaches (`engine_bteg`,
0.91% / 2.46%).

### The correction is data-dependent, so no factor can be applied by hand

On the synthetic GARCH-$t$ panel the 20-node understatement is **0.57–0.70%**, against 1.65–1.69% on the real
panel, because the fitted GPD tail index is smaller there. The size of the error scales with the tail index
being integrated. Every affected table therefore needs its own rerun; none can be patched with a multiplier.

### The shared module, and the whole battery on the converged convention

The fix now lives in one place, `code/paper/es_integral.py`, rather than in seventeen copies. It self-tests on
two analytic cases: the GPD sub-floor integral against a 4×10⁶-node substituted quadrature (2e-6), and a
pure-GPD end-to-end ES against its closed form (7e-8). It also carries `es_by_substitution` for parametric
quantiles with no elementary integral, which matches the exact Student-$t$ ES to **7.6e-8** for $\nu \in
[3.5, 12]$ — that is the function the GJR-skew-$t$ row needs.

`job_exception_battery.py` imports it and carries `engineC`/`overlayC`/`conf2C` rows beside the committed ones,
so nothing shipped is overwritten and the VaR match between `<row>` and `<row>C` is a built-in control.

**Validation.** `engineC_garch_t` FZ0 reproduces `job_es_converged.py` to **0.0e+00** at both levels, through a
different script and a different code path. All six VaR-only controls pass (breach and date-clustered $t$
identical between `<row>` and `<row>C`).

**One inconsistency found by that validation, worth recording.** `engineC_bteg` at 1% differs from
`job_es_converged.py` by 5e-5, and the *committed* ES differs too (−3.86302 vs −3.86291) — so it is not the
module. The two scripts fit Beta-$t$-EGARCH differently: the battery searches from **three** starting points
with `maxiter=4000`, `job_es_converged.py` from **two** with `maxiter=3000`. The battery takes the minimum NLL
over more starts, so **the battery's `bteg` numbers are the authoritative ones**; the differences are in the
fifth decimal and change nothing. The GARCH-$t$ rows — the deployed configuration — are bit-identical.

**The battery result generalises P3 beyond the deployed row.** Every engine row's McNeil–Frey $p$ rises
sharply under the converged integral:

| row | α | FZ0 20-node | FZ0 conv | ΔFZ0 | AS $p$ 20-node → conv | MF $p$ 20-node → conv |
|---|---|---|---|---|---|---|
| `engine_garch_t` | 1% | 2.14731 | 2.14653 | −7.8e-4 | 0.254 → 0.384 | **0.067 → 0.383** |
| `engine_bip_t` | 1% | 2.13377 | 2.13294 | −8.3e-4 | 0.118 → 0.198 | **0.071 → 0.416** |
| `engine_bteg` | 1% | 2.13065 | 2.13082 | +1.7e-4 | 0.265 → 0.158 | **0.189 → 0.682** |
| `engine_garch_t` | 2.5% | 1.84936 | 1.84863 | −7.3e-4 | **0.026 → 0.055** | 0.391 → 0.583 |
| `engine_bip_t` | 2.5% | 1.83690 | 1.83600 | −9.0e-4 | 0.008 → 0.018 | 0.346 → 0.607 |
| `engine_bteg` | 2.5% | 1.83176 | 1.83198 | +2.2e-4 | 0.798 → 0.551 | 0.725 → 0.346 |

So "McNeil–Frey rejects the engine" was a quadrature artifact on **all three Stage-1 scales**, not just the
deployed one. The FZ0 sign pattern also holds across all three: it falls for the two scales that over-breach
(`garch_t`, `bip_t`) and rises for the one that under-breaches (`bteg`).

**Stage 4 is unaffected, and if anything looks worse.** The overlay and retargeted-conformal rows get *worse*
under the fix (`overlay_garch_t` at 2.5% 1.86553 → 1.86639) and still reject Acerbi–Székely at $p=0.000$ either
way. The decision to drop Stage 4 does not depend on the ES convention. One detail worth noting: the
retargeted row's own McNeil–Frey rejection was also quadrature — `conf2_garch_t` at 2.5% moves 0.010 → 0.169.

### Synthetic truth with the converged integral

Both DGPs are now in, with the converged integral, and they are symmetric. `results/paper/synthetic_truth_results.json`
holds the consolidated record (both DGPs, three placebos, and the quadrature size check). 200 names, 221,600
rows each. This is where the fix mattered most: the engine-versus-truth gaps (1e-5 to 3e-4) were *smaller than
the 20-node error they carried*, so their sign was not identified before.

| DGP | α | scale | parametric | engine conv | engine − param |
|---|---|---|---|---|---|
| GARCH-$t$ | 1% | GARCH-$t$ **(true)** | 1.38795 | 1.38822 | +2.7e-4 |
| | 1% | Beta-$t$-EGARCH | 1.39813 | 1.39172 | **−6.4e-3** |
| | 2.5% | GARCH-$t$ **(true)** | 1.16755 | 1.16756 | **+1.0e-5** |
| | 2.5% | Beta-$t$-EGARCH | 1.17554 | 1.17017 | **−5.4e-3** |
| Beta-$t$-EGARCH | 1% | Beta-$t$-EGARCH **(true)** | 1.74334 | 1.74363 | +2.9e-4 |
| | 1% | GARCH-$t$ | 1.75308 | 1.75146 | **−1.6e-3** |
| | 2.5% | Beta-$t$-EGARCH **(true)** | 1.50774 | 1.50756 | −1.8e-4 |
| | 2.5% | GARCH-$t$ | 1.51721 | 1.51560 | **−1.6e-3** |

**The asymmetry is the result.** Against the true parametric model the flexible engine is indistinguishable
from it — it loses by 2.7e-4 and 1.0e-5 when GARCH-$t$ is true, loses by 2.9e-4 at 1% and edges ahead by
1.8e-4 at 2.5% when Beta-$t$-EGARCH is true. Against a *misspecified* scale it wins by 1.6e-3 to 6.4e-3, six
to five hundred times larger. The estimator costs essentially nothing when the parametric form is right, and
the frontier appears only against misspecification. That is the Online Appendix table.

The three placebos (1/5/20-day shift) keep an edge on the GARCH-$t$ scale (DM 4.4–5.7 overall, 9.5–10.0 in the
top decile), recorded with the limitation stated in the file: both models share the same conditional scale, so
any shift preserving the scale preserves the shape advantage. They bound what a timing artifact could explain;
they do not test for one. Their FZ0 was not computed, and the file says so rather than leaving a gap.

---

## 5b. `job_bench_all` on one converged path: the tie group and the MCS

GJR-skew-$t$ now uses `es_by_substitution` and the engine uses `converged_es`, so every numerically integrated
ES in this table comes from one validated path. `conf975` reproduces at $-0.3426$, the top-decile pinball edge
at $+2.445\%$ (DM 10.18), and the GPD parameterisation guard passed.

**Exactly three rows moved, and nothing else did.** Every other row reproduces its published FZ0 to 5e-5,
which is print rounding. That is the control on the rerun: it changed the ES convention and nothing besides.

| row | α | published | new | Δ |
|---|---|---|---|---|
| engine | 1% | 2.1473 | **2.14653** | −7.7e-4 |
| engine_overlay | 1% | 2.1473 | 2.14653 | −7.7e-4 |
| gjr_skewt | 1% | 2.1506 | **2.15047** | −1.3e-4 |
| engine | 2.5% | 1.8494 | **1.84863** | −7.7e-4 |
| gjr_skewt | 2.5% | 1.8487 | **1.84859** | −1.1e-4 |
| engine_overlay | 2.5% | 1.8656 | 1.86639 | +7.9e-4 |

The superseded values are carried in the result file under `es_convention.superseded_values`
(`engine__ES_20node`, `gjr_skewt__ES_200node`) rather than discarded.

**The 90% MCS is unchanged at both levels** — P1's MCS clause confirmed:

| | 90% MCS on FZ0 |
|---|---|
| 1% | engine, engine_overlay, body, gjr_skewt |
| 2.5% | engine, body, gjr_skewt, taylor |

and the pinball MCS (ES-free, so it must not move) is `body, sav_caviar` as published.

**The 2.5% tie group, to be reported as a group with its significance tests and not as an order:**

| row | FZ0 | DM vs engine |
|---|---|---|
| body (no EVT, internal ablation) | 1.84323 | −1.54 |
| gjr_skewt | 1.84859 | **−0.02** |
| **engine** | 1.84863 | ref |
| taylor (ES-CAViaR) | 1.84899 | +0.13 |

**Nothing significantly beats the engine at 2.5%**, so the abstract's significance statement stands. And the
GJR defect did matter, in the direction that mattered: I had estimated that fixing only the engine would push
it *past* GJR. With GJR's own 200-node error corrected too, the two draw level instead — 4e-5 apart on a
1.8486 level. That is a tie in any sense, and it is only visible because both sides were put on the same path.

**At 1% the engine is the lowest of every external benchmark**, by a margin that the fix widened: the gap to
GJR goes 3.3e-3 → 3.9e-3 and the DM 1.05 → 1.24. It is still not a significant margin, so "lowest at 1%" is a
level statement, not a significance one, and should not be written as though it were. `body` at 2.14112 is
lower than the engine but is the no-EVT ablation of the engine itself, DM −1.09, not significant.

**One DM moves materially and should be restated:** GARCH-$t$ at 2.5%, **4.55 → 5.50**, because the engine
improved while a closed-form row could not. At 1% it barely moves (5.00 → 4.94).

**Stage 4 again looks worse, not better:** `engine_overlay` at 2.5% goes 1.86553 → 1.86639, DM 2.29 → 2.57.

---

## 5c. `job_robust_engine` on the converged path

Done, canon200. Every engine row moved and nothing else did; the ratios match the battery and `job_bench_all`
to five decimals.

| row | α | ES 20-node | ES converged | ratio | FZ0 under 20-node |
|---|---|---|---|---|---|
| `engine` | 1% | −3.78359 | −3.84505 | 1.01645 | 2.14731 |
| `engine` | 2.5% | −2.80693 | −2.85324 | 1.01662 | 1.84936 |
| `engine_bip` | 1% | −3.82571 | −3.89184 | 1.01745 | 2.13377 |
| `engine_bip` | 2.5% | −2.84323 | −2.89194 | 1.01718 | 1.83690 |
| `engine_bteg` | 1% | −3.86302 | −3.92587 | 1.01650 | 2.13065 |
| `engine_bteg` | 2.5% | −2.85890 | −2.90664 | 1.01687 | 1.83176 |

**Three independent scripts now agree bit-for-bit on the deployed row** — `job_exception_battery`,
`job_bench_all` and `job_robust_engine` all give FZ0 2.14653 at 1% and 1.84863 at 2.5%. The MCS on this panel
is unchanged at both levels, and `engine` remains the highest-FZ0 of the three engine variants, so the
Section-4 ordering (robust scales ahead of GARCH-$t$ on the joint score) is untouched by the convention.

---

## 5d. `job_frtb200`: the exact empirical tail mean was the biggest error in the table

Done. 221,600 rows, 200 names — row set identical to published, so this is like-for-like. **Every pinball
number is unchanged to four decimals**, and every closed-form or exact-empirical row's ES moves by ≤0.002,
which is print rounding. Those are the controls.

| model | pinball pub / new | ES pub | ES new | Δ | realized | over % |
|---|---|---|---|---|---|---|
| resid_hybrid_ML | 0.3676 / 0.3676 | −6.63 | **−6.631** | −0.001 | −6.112 | 8.5 |
| hybrid_EVT | 0.3680 / 0.3680 | −7.21 | **−7.326** | −0.116 | −6.138 | 19.3 |
| garch_t | 0.3688 / 0.3688 | −6.98 | −6.982 | −0.002 | −5.916 | 18.0 |
| gjr_skewt | 0.3688 / 0.3688 | −6.85 | **−6.872** | −0.022 | −6.086 | 12.9 |
| fhs_pername | 0.3689 / 0.3689 | −6.90 | −6.900 | +0.000 | −6.061 | 13.8 |
| fhs (pooled) | 0.3691 / 0.3691 | −7.17 | −7.168 | +0.002 | −5.886 | 21.8 |
| **fhs_roll500** | 0.3694 / 0.3694 | −6.71 | **−7.016** | **−0.306** | −6.209 | 13.0 |
| ewma_rm | 0.3735 / 0.3735 | −5.72 | −5.722 | −0.002 | −5.841 | −2.0 |
| **hist_sim** | 0.3752 / 0.3752 | −7.41 | **−7.714** | **−0.304** | −7.081 | 8.9 |

**The two empirical rows were the worst in the table, by a factor of three over the engine.** `hist_sim` moves
4.1% and `fhs_roll500` 4.6%, against the engine's 1.6%. The reason is specific: integrating a rolling
*empirical* quantile function on a 20-node grid over $(0, 0.025]$ is a poor estimator of that window's tail
mean, because with 500 observations only about twelve lie below the 2.5% quantile, so most of the 20 nodes fall
between order statistics and the average drifts toward the quantile rather than the mean below it. The tail
mean is available exactly — `roll_tail_mean` now computes it window by window, the same construction as pooled
FHS — so there was never a reason to approximate it.

**One printed ordering changes.** By severity, `fhs_roll500` moves from 7th to 4th, passing GARCH-$t$,
per-name FHS and GJR-skew-$t$ (−7.016 against −6.982, −6.900, −6.872). Its overstatement against realized goes
8.1% → 13.0%, and `hist_sim`'s 4.7% → 8.9%. Rolling-window benchmarks are more conservative than the table
said, not less.

**The body-only row barely moves** (−6.63 → −6.631), which is itself informative: it localises the engine's
1.6% to the **GPD branch's divergence at the origin**, not to the body. A body curve with a flat sub-floor
extension has nothing to diverge, so there is almost no quadrature error to recover — and conversely, the
`resid_hybrid_ML` ES remains a lower bound on severity, now stated as such in the result file's
`es_convention.converged_above_the_floor_only`.

**The old note's two false claims are corrected in the output**, not just in the paper: that the column was
"the exact tail integral of each model's own quantile function", and that the 200-node Hansen rule was
acceptable because "the same node set" was applied to every empirical model so the bias was common. The GBM
rows used 20 nodes and the parametric rows 200 — an order of magnitude apart, in the same direction.

Two self-tests in the script had to change, because one of them *encoded* the superseded convention: it
asserted that `hansen_es` equals the 200-node midpoint of a standardised $t$. It now asserts agreement with the
**exact** Student-$t$ ES at $\lambda=0$ (where the Hansen skew-$t$ reduces to one) for $\eta \in \{5, 8, 14\}$,
and a second assertion pins the legacy path so the old-vs-new record stays trustworthy. `job_frtb200.py` also
now honours `GBC_PROJ` like the other jobs instead of a hardcoded Windows path.

---

## 6. Every printed number the ES fix moves — FINAL

All five jobs have run. Scope was taken from the generating scripts, not the tables. Every number below is
measured, not inferred. **Nothing is committed.**

### 6.1 What does not move, with the reason

- **Every pinball number anywhere**, and therefore the whole frontier: the +2.44% top decile, DM 10.18, the
  decile profile, the age curve, the amortization results. `job_frtb200` reproduced all nine pinball values to
  four decimals; `job_bench_all` reproduced the top-decile edge exactly.
- **Every VaR-only statistic**: breach rates, Kupiec (pooled and per-name), Christoffersen, Engle–Manganelli
  DQ, the date-clustered exception $t$ and its 2.00, the design effects, the clustered coverage $p$-values.
  Proved, not assumed: `job_exception_battery` carries `<row>` and `<row>C` with identical VaR, and all six
  controls pass with breach and $t$ equal to six decimals.
- **The whole of `tab_bench_tests`**, verified in the code: `cpa()` is called on `PL[m]`, the pinball loss
  (`job_bench_all.py:400`); `murphy()` reads `VE[a][m][0]`, the quantile only (`:418`); DQ is hit-based; the win
  rate is date-averaged pinball. No column touches ES.
- **Closed-form and exact-empirical rows**: GARCH-$t$, EWMA, GARCH-normal, pooled FHS, per-name FHS, HS,
  GARCH-EVT's McNeil–Frey closed form. In `job_frtb200` each moved by ≤0.002, which is print rounding.
- **The 90% MCS, at every level on every panel** — canon200 (both jobs), taq30, holdout — and the pinball MCS.
- **`tab_scaleshape_bteg` in its entirety at printed precision.** All eight FZ0 entries move by ≤0.0005 against
  values printed to three decimals, so every one is unchanged. The taq30 ES ratios are only 1.0065–1.0091
  against canon200's 1.0165, because large-cap 2014–2024 residuals have a thinner fitted GPD tail.

### 6.2 What moves — measured, old → new, with the file

**`tab_deployed`** — `exception_battery_results.json`, `robust_engine_results*.json`

| panel | number | published | new |
|---|---|---|---|
| B | mean FZ0, 1% | 2.14711 | **2.14653** |
| B | mean FZ0, 2.5% | 1.84871 | **1.84863** |
| B | DM vs GARCH(1,1)-$t$, 1% | +5.00 | +4.94 |
| B | DM vs GARCH(1,1)-$t$, 2.5% | +4.57 | **+5.50** |
| B | DM vs robust scale + pooled residual quantile, 1% / 2.5% | −1.75 / −2.32 | −1.64 / −2.24 |
| B | DM vs bounded-news GARCH-$t$, 1% / 2.5% | −1.20 / −3.35 | −0.94 / −2.85 |
| B | Acerbi–Székely $p$, 1% / 2.5% | 0.348 / 0.058 | **0.384 / 0.055** |
| B | McNeil–Frey $p$, 1% / 2.5% | **0.022** / 0.240 | **0.383** / 0.583 |
| C | holdout FZ0 1% / 2.5% | 1.81972 / 1.57084 | 1.81990 / 1.57082 |

The published Panel B FZ0 and test values also absorb the `mk63` battery fix. Both corrections are separable:
the fixed-battery 20-node figures are 2.14731 / 1.84936 / AS 0.2536 / 0.0256 / MF 0.0673 / 0.3914.

**`tab_bench_fz0`** and **`tab_garch_evt_fz0`** — `bench_all_results.json`, `garch_evt_results.json`

| row | α | published | new |
|---|---|---|---|
| Engine | 1% / 2.5% | 2.1473 / 1.8494 | **2.14653 / 1.84863** |
| Engine + conformal overlay | 1% / 2.5% | 2.1473 / 1.8656 | 2.14653 / **1.86639** |
| GJR-GARCH-skew-$t$ | 1% / 2.5% | 2.1506 / 1.8487 | **2.15047 / 1.84859** |

Every DM in both tables is measured against the engine and so moves: at 1% body −1.33→−1.09, GJR
+1.05→+1.24, FHS-pooled +4.15→+3.93, GARCH-EVT-pooled +4.83→+4.51, Taylor +3.53→+3.76, GARCH-$t$ +5.00→+4.94;
at 2.5% body −1.87→−1.54, GJR −0.28→**−0.02**, Taylor −0.12→**+0.13**, FHS-pooled +6.37→+5.89,
GARCH-EVT-pooled +6.85→+6.33, overlay +2.29→+2.57, GARCH-$t$ +4.55→**+5.50**. MCS ticks unchanged.
`tab_garch_evt_fz0`'s note must drop "the engine and body ES are the 20-node integral".

**`tab_frtb200`** — `frtb_table_200_results.json`

| row | published | new | Δ |
|---|---|---|---|
| Residual-hybrid (GBM) | −6.63 | −6.631 | −0.001 |
| \quad + EVT tail | −7.21 | **−7.326** | −0.116 |
| GJR-GARCH-skew-$t$ | −6.85 | −6.872 | −0.022 |
| GARCH-FHS (rolling 500d) | −6.71 | **−7.016** | **−0.306** |
| Historical simulation | −7.41 | **−7.714** | **−0.304** |

Pinball, breach and Kupiec columns unchanged. **The note's claim that this column is "the exact tail integral
of each model's own quantile function" is false and must change**, and so must the 200-node justification now
contradicted in the script's own comment.

**`tab_scaleshape_bteg` DM column** — two entries move at printed precision: the daily-core row at 2.5%
+6.2 → +6.3, and the realized-scale + shape + EVT row at 1% +2.0 → +1.9. The other twelve are unchanged.

**Not in a table but printed in text or carried in result files:** `scale_gate_results.json` FZ0 levels
(2.22333 / 2.23133 / 2.23636 at 1%; 1.92574 / 1.93320 / 1.93332 at 2.5%) and the DMs among them — the gate's
margin over `engine_bteg` is +0.80 / +0.73, small enough that the sign needs rechecking after a gate rerun;
the overlay, walk-forward and cold-start FZ0 figures; `synthetic_truth_results.json` (**done**, both DGPs).

### 6.3 Two claims that change character rather than value

- **A8, McNeil–Frey.** Withdrawn, not softened: $p = 0.383$ at 1%. It was a quadrature artifact on all three
  Stage-1 scales (0.067→0.383, 0.071→0.416, 0.189→0.682), and the published 0.022 compounded it with the
  `mk63` battery bug.
- **The 2.5% ranking.** Report the tie group and the significance tests, not an order: body 1.84323 (DM −1.54),
  gjr_skewt 1.84859 (−0.02), engine 1.84863 (ref), taylor 1.84899 (+0.13). Nothing significantly beats the
  engine, so the abstract's significance statement stands; "lowest at 1%" is a level statement (DM +1.24 vs the
  nearest benchmark) and must not be written as a significance one.

---

## 7. Manuscript-first inventory: every printed FZ0 / ES / Acerbi–Székely / McNeil–Frey number

§6 was scoped from the generating scripts and that was the wrong method — it caught the tables and missed
figures and running text that quote other files. This section starts from `preprint/paper_A_ssrn.tex` and its
online appendix at `a1f997f` and traces each number to a file and a convention. **Nothing is committed.**

### 7.1 Three source files §6 missed entirely

| file | what the manuscript quotes from it | was |
|---|---|---|
| `fz_fullpanel_results.json` | Figure 2 bars and the §5.1 DM list (paper.tex:1087–1095) | 20-node |
| `pzc_taylor_results.json` | §5.1 GAS and Taylor DMs, FHS at 2.5% (paper.tex:1095) | 20-node |
| `garch_evt_results.json` | §5.1 GARCH-EVT DMs, `tab_garch_evt_fz0`, `tab_garch_evt_frontier` | 20-node |

All three are now rerun on `es_integral`. One correction to the hand-off: **`walkforward_hybrid_results.json`
needs no rerun.** It contains no FZ0 and no ES — only `overall_edge_pct`, `overall_DM`, the decile profile and
NW-lag sensitivity, all pinball. `job_walkforward_hybrid.py` builds the body with no EVT branch and no
conformal shift, so there is no ES to integrate. The §5.1 refit DMs attributed to it (3.6/2.7 at 1%, 3.1/2.6 at
2.5%, paper.tex:1097) are **FZ0** numbers and come from `walkforward_bteg_results.json`.

### 7.2 Figure 2 and §5.1 — `fz_fullpanel_results.json`, rerun

| manuscript | number | published | new | printed value |
|---|---|---|---|---|
| 1087, 1095 | GARCH-$t$ DM, 1% | 4.79 | 4.78 | 4.8, **unchanged** |
| 1087, 1095 | GARCH-$t$ DM, 2.5% | 5.06 | **5.40** | 5.1 → **5.4** |
| 1088, 1095 | FHS DM, 1% | 4.93 | 4.87 | 4.9, **unchanged** |
| 1090, 1095 | GAS DM, 1% | 9.55 | **9.93** | 9.5 → **9.9** |
| — | top-decile GARCH-$t$ minus engine, 1% / 2.5% | 3.36 / −6.53 | 3.05 / −6.61 | — |

Engine (no conformal) FZ0 1.84842 at 2.5%, 2.14679 at 1%.

### 7.3 §5.1 dynamic-ES benchmarks — `pzc_taylor`, rerun

**Provenance note first:** no script in the repository writes `pzc_taylor_results.json`. `job_pzc_taylor.py`
writes `pzc_taylor_acc_results.json`. The two share the same note, the same row set and **identical benchmark
FZ0 values** (2.15572 / 2.17241 / 2.23238 / 2.16472 at 1%), so the committed file is an earlier run of this
same script under the previous output name; only the engine reference differs. `apply_R59.py` should point the
text at `pzc_taylor_acc_results.json`.

| manuscript | number | published | new | printed value |
|---|---|---|---|---|
| 1090, 1095 | GAS DM, 1% | 9.51 | **9.99** | 9.5 → **10.0** |
| 1090, 1095 | GAS DM, 2.5% | 8.60 | **9.01** | 8.6 → **9.0** |
| 1095 | Taylor DM, 1% | 3.58 | **3.90** | 3.6 → **3.9** |
| 1095 | Taylor DM, 2.5% | 0.08 | 0.21 | still a tie ✓ |
| 1088, 1095 | FHS DM, 2.5% | 5.27 | **5.13** | 5.3 → **5.1** |

### 7.4 §5.1 GARCH-EVT and `tab_garch_evt_fz0` — `garch_evt`, rerun

Reproduces `bench_all` **exactly** (engine 2.14653 / 1.84863), which is a cross-check on both.

| number | published | new |
|---|---|---|
| GARCH-EVT pooled DM, 1% / 2.5% | 4.83 / 6.85 | **4.51 / 6.33** |
| GARCH-EVT per name DM, 1% / 2.5% | 5.23 / 5.39 | **5.14 / 5.36** |
| GARCH-$t$ DM, 1% / 2.5% | 5.00 / 4.55 | 4.94 / **5.50** |
| pooled body DM, 1% / 2.5% | −1.33 / −1.87 | −1.09 / −1.54 |

So §5.1's "pooled tail by DM 4.8 at 1% and 6.8 at 2.5%, and per name by DM 5.2 and 5.4" becomes 4.5 and 6.3,
5.1 and 5.4.

### 7.5 Stage 4 — `overlay_engine`, both panels, rerun

**The pre-committed rule still passes, and slightly more comfortably.** The rule required the adaptive shift to
move the breach rate toward nominal while keeping FZ0 within noise of the accuracy layer.

| panel | α | engine FZ0 / breach | ACI γ=0.02 | ACI γ=0.05 |
|---|---|---|---|---|
| canon200 | 1% | 2.14653 / 1.04% | 2.14650, DM −0.25, 1.04% | 2.14645, DM −0.29, 1.03% |
| canon200 | 2.5% | 1.84863 / 2.78% | 1.84768, DM −1.31, 2.68% | 1.84726, DM −1.02, 2.60% |
| holdout | 1% | 1.81990 / 1.08% | 1.82083, DM +0.93, 1.06% | 1.82183, DM +0.91, 1.04% |
| holdout | 2.5% | 1.57082 / 2.57% | 1.57124, DM +0.27, 2.43% | 1.57029, DM −0.03, 2.38% |

Every |DM| < 2 on both panels, and on canon200 the adaptive overlay now has *lower* FZ0 than the accuracy layer
at both levels while pulling 2.5% breach from 2.78% to 2.60–2.68%. The static shifts stay bad (body-targeted DM
+3.9 and +2.57). The Stage-4 conclusion is unchanged: static is the wrong instrument, adaptive is free.

### 7.6 The holdout battery — rerun, and it changes what A8 can say

**This is the one place the fix moves a verdict the wrong way, and it would have been invisible under a mixed
convention.** Holdout 2000–2013, 280,608 rows, McNeil–Frey exceedance residual and its $p$:

| row | residual 1% | $p$ | residual 2.5% | $p$ |
|---|---|---|---|---|
| engine, 20-node | +0.0887 | 0.013 | −0.0175 | 0.653 |
| **engine, converged** | **+0.1540** | **0.000** | +0.0278 | 0.477 |
| GARCH-$t$ (closed form, unaffected) | −0.1745 | 0.000 | −0.0986 | 0.022 |
| FHS (exact empirical, unaffected) | +0.1241 | 0.001 | +0.0109 | 0.799 |

A positive residual means realized losses beyond VaR were *less* severe than the predicted ES, i.e. the ES is
too conservative. The 20-node engine was already too conservative at 1% on this panel, so deepening |ES| by
1.85% pushes it from marginal (0.013) to decisive (0.000). **The A8 withdrawal therefore holds for the design
era only.** The correct holdout statement is more informative than the current one: at 1% on 2000–2013 every ES
construction is rejected by McNeil–Frey, and they fail in *opposite directions* — the parametric $t$ tail is
too shallow (−0.17), the EVT and FHS tails too deep (+0.15, +0.12). At 2.5% the engine passes (0.477) and the
parametric tail rejects (0.022). Acerbi–Székely improves on the holdout under the fix: 0.750 → 0.835 at 1% and
0.784 → 0.889 at 2.5%. Panel C FZ0: 1.81972 → 1.81990 and 1.57084 → 1.57082.

### 7.7 The robust-scale tail claim — `tail_vs_bteg`, frozen done

| number | published | new | P1 clause |
|---|---|---|---|
| `engine_bteg` FZ0 vs `param_bteg`, 1% | −4.29 | **−3.85** | passes (DM > 2) |
| `engine_bteg` FZ0 vs `param_bteg`, 2.5% | −5.34 | **−4.68** | passes (DM > 2) |

Both FZ0 clauses of P1 still pass comfortably. The pinball clauses are ES-free and cannot move, so the frozen
grading of P1 is unchanged (3 of 4, with pinball at 0.025 the tie). Refit protocol still running.

### 7.8 A pre-existing inconsistency this inventory exposed, independent of the ES convention

Five scripts each rebuild "the engine" and they do not agree:

| file | engine FZ0 1% | 2.5% | conf975 |
|---|---|---|---|
| `bench_all`, `garch_evt`, `robust_engine`, `exception_battery` | 2.14653 | 1.84863 | −0.3426 |
| `fz_fullpanel` | 2.14679 | 1.84842 | −0.3529 |
| `pzc_taylor_acc` | 2.14622 | 1.84839 | −0.3373 |

The spread is 5.7e-4 at 1% — comparable to the entire quadrature correction (7.7e-4) and larger than several
DM-relevant gaps. It is **not** caused by the ES convention: each script refits the pooled GPD and the
conformal split on its own row set. The paper quotes "lowest FZ0" from more than one of these, so R59 should
either name one file as canonical for that claim or report the range. Flagging, not fixing.

### 7.9 Still running

`job_scale_gate` (through the 2022 cutoff), `job_walkforward_bteg` (through 2021), `job_tail_vs_bteg --protocol
refit` (through 2020). These carry the gate result for §6, the refit DMs at paper.tex:1097 (3.6/2.7, 3.1/2.6,
−0.31/−0.01, 5.37/7.25) and the refit arm of the tail claim. §7 is final once they land; §6 is unchanged by
them.

### 7.10 One convention sentence that must change regardless

`paper.tex:363` states that ES is "computed as the numerical integral of (eq:hybrid) rather than by the GPD
closed form of McNeil–Frey; the Online Appendix reports that the two agree to the third decimal on FZ0." That
agreement was measured with the 20-node rule, which understates |ES| by 1.6–1.9%. The comparison has to be
restated against the converged integral.

---

## 7. Manuscript-first inventory: every printed FZ0 / ES / AS / MF number

§6 was scoped from the scripts and missed sources the text quotes directly. This section is built the other
way: from `preprint/paper_A_ssrn.tex` and its online appendix **at `a1f997f`**, grepping every line that
mentions FZ0, expected shortfall, Acerbi–Székely or McNeil–Frey (47 lines in the paper, 37 in the OA), then
tracing each number to its file, field and convention. **Nothing is committed.**

### 7.0 What the script-first scope missed

Four sources the manuscript quotes that §6 never listed:

| file | what the text quotes from it | found how |
|---|---|---|
| `fz_fullpanel_results.json` | Figure 2's bars; §5.1 GARCH-$t$ 4.8/5.1, FHS 4.9, GAS 9.5 | field names match (`garch_minus_engine_noconf`, `fhs`, `gas_pzc`) |
| `pzc_taylor_results.json` | §5.1 GAS 8.6 at 2.5%, Taylor 3.6, FHS 5.3 at 2.5% | `per_alpha/0.025/gas_pzc/DM_t = 8.6` exactly |
| `garch_evt_results.json` | §5.1 GARCH-EVT 4.8/6.8 pooled, 5.2/5.4 per name | `fz0/0.025/evt_pool/vs_engine/DM_t = 6.85` |
| `engine_esbt_results.json` | Table OA `tab:esbt` and the whole A8 ES-calibration passage | the OA names the file in its source note |

And one the user listed that does **not** need a rerun: `walkforward_hybrid_results.json` carries **no FZ0 and
no ES** — only `overall_DM` and `top_decile_DM` on pinball (4.41, 2.12). The four numbers in the §5.1 refit
sentence (DM 3.6, 2.7 at 1%; 3.1, 2.6 at 2.5%) are FZ0 DMs and come from `walkforward_bteg_results.json`.

### 7.1 One canonical engine forecast — and it needs no per-row dump

Adopted as canonical: the converged accuracy layer at **FZ0 2.14653 / 1.84863, breach 1.04% / 2.78%,
conf975 −0.3426**, which `job_bench_all`, `job_exception_battery`, `job_robust_engine` and `job_garch_evt` all
produce **bit-identically**.

Re-scoring Figure 2's bars, the GAS/Taylor DMs and the GARCH-EVT rows against it turned out to need no new
runs, because the benchmark FZ0 levels are **identical across all four scripts to five decimals** — so the row
sets are the same and the benchmark code agrees, and `bench_all_results.json` already scores every one of them
against the canonical engine on the same 221,600 rows:

| benchmark | published 1% | canonical 1% | published 2.5% | canonical 2.5% |
|---|---|---|---|---|
| GARCH-$t$ | +4.8 | **+4.94** | +5.1 | **+5.50** |
| FHS (per name) | +4.9 | +4.91 | +5.3 | **+5.35** |
| GAS (PZC) | +9.5 | **+9.91** | +8.6 | **+8.78** |
| ES-CAViaR (Taylor) | +3.6 | **+3.76** | — (tie) | +0.13 (still a tie) |
| GARCH-EVT, pooled | +4.8 | **+4.51** | +6.8 | **+6.33** |
| GARCH-EVT, per name | +5.2 | +5.14 | +5.4 | +5.36 |
| pooled body (no EVT) | −1.3 | −1.09 | −1.9 | −1.54 |
| GJR-skew-$t$ | +1.1 | +1.24 | −0.3 | **−0.02** |

**One exception, and it is not about ES.** GAS's own FZ0 differs between runs — 2.23212 / 1.89924 in
`fz_fullpanel` against 2.23238 / 1.89905 in `bench_all` and `pzc_taylor`. GAS is fitted per name by
multi-start Nelder–Mead FZ0 minimisation, so this is optimiser non-determinism, not rows or convention. R59
should name `bench_all_results.json` as the GAS fit of record. Every other benchmark agrees exactly.

**No script needs to save per-row losses.** That was the fallback if the benchmark levels had differed; they do
not.

### 7.2 The four engine forecasts, which is a separate pre-existing defect

Before the convention fix the manuscript was already quoting "lowest FZ0" from runs that disagree about the
engine:

| file | engine FZ0 1% / 2.5% | conf975 | 20-node ES 1% |
|---|---|---|---|
| `bench_all`, battery, `robust_engine`, `garch_evt` | 2.14653 / 1.84863 | −0.3426 | −3.78359 |
| `fz_fullpanel` | 2.14679 / 1.84842 | **−0.3529** | **−3.77788** |
| `pzc_taylor_acc` | 2.14622 / 1.84839 | **−0.3373** | — |
| `engine_esbt` | pending | — | −3.78359 |

Each re-derives the pooled GPD and the conformal split independently. The spread is 5.7e-4 in FZ0 — smaller
than the ES correction but larger than several gaps the paper ranks on. This is **not** created by R59 and is
not fixed by it; §7.1 resolves it for the claims it touches by naming one run.

### 7.3 Orphaned file

`pzc_taylor_results.json` is quoted by §5.1 but **no script in the repository writes it** —
`job_pzc_taylor.py` writes `pzc_taylor_acc_results.json`. The two agree exactly on every benchmark FZ0
(2.15572 / 2.17241 / 2.23238 / 2.16472 and the 2.5% counterparts), so the `_acc` file is the same computation
under its current name. R59 repoints §5.1 to `pzc_taylor_acc_results.json`; the correcting commit should delete
the orphan or mark it superseded in the README.

### 7.4 Holdout McNeil–Frey, by direction — A8 cannot be stated era-free

`exception_battery_results_holdout.json`, 280,608 rows, 2000–2013. Mean exceedance residual (sign is the
direction: **positive = ES too conservative**, negative = ES not severe enough) with its bootstrap $p$:

| row | MF 1% | $p$ | MF 2.5% | $p$ |
|---|---|---|---|---|
| engine, 20-node | +0.0887 | 0.013 | −0.0175 | 0.653 |
| **engine, converged** | **+0.1540** | **0.000** | **+0.0278** | **0.477** |
| GARCH-$t$ (closed form, unaffected) | **−0.1745** | 0.000 | −0.0986 | 0.022 |
| FHS pooled (empirical, unaffected) | +0.1241 | 0.001 | +0.0109 | 0.799 |

**At 1% on the holdout every ES construction is rejected, and they fail in opposite directions**: the
parametric Student-$t$ tail is not severe enough (−0.17), while the engine's EVT-spliced tail and FHS's
empirical tail are too severe (+0.15, +0.12). The convention fix moves the engine *further* into rejection
there (0.013 → 0.000), the opposite of its effect on canon200 (0.067 → 0.383), because the 20-node ES was
already over-conservative on this panel. At 2.5% the engine passes (0.477) and GARCH-$t$ rejects (0.022).

So A8 must be stated **by era and by direction**: on the design era the engine passes McNeil–Frey at both
levels after the fix; on the 2000–2013 holdout its 1% ES is too conservative and rejected, as are GARCH-$t$'s
(too shallow) and FHS's (too deep). That is a fairer statement than the current one and carries more
information, and it is not available under a mixed convention.

### 7.5 Pre-committed rules, re-graded on the converged ES

| rule | verdict | numbers |
|---|---|---|
| adaptive overlay "FZ0 within noise of the accuracy layer", canon200 | **passes** | ACI 0.02/0.05 FZ0 1.84768/1.84726 vs engine 1.84863, DM −1.31/−1.02; breach 2.78% → 2.68%/2.60% |
| same, holdout | **passes** | all four overlay variants \|DM\| ≤ 0.93; breach 1.08% → 1.06%/1.04% and 2.57% → 2.43%/2.38% |
| tail job P1, FZ0 clauses, frozen | **passes** | `engine_bteg` vs `param_bteg` DM −4.29 → **−3.85** (1%), −5.34 → **−4.68** (2.5%) |
| tail job P1, pinball clauses | unchanged | pinball carries no ES |
| gate P1 / P2 | pending | `job_scale_gate` still running |

Static overlays remain wrong in the same way: body-targeted DM +3.90 / +2.57, engine-targeted +1.17 / +0.36,
and both reject Acerbi–Székely at $p=0.000$ at 2.5% under either convention. The Stage-4 decision does not
depend on the ES convention.

### 7.6 The last four sources

**`engine_esbt_results.json` — `tab:esbt` and the whole A8 passage.** The three unaffected rows
(`garch_t`, `fhs`, `garch_norm`) come back **bit-identical**, which is the control.

| row | α | $Z_2$ old → new | MF old → new | MF $p$ old → new |
|---|---|---|---|---|
| engine (no conformal) | 1% | −0.0711 → **−0.0533** | −0.1210 → **−0.0578** | 0.0673 → **0.3832** |
| engine (no conformal) | 2.5% | −0.1216 → −0.1028 | −0.0290 → **+0.0186** | 0.3878 → 0.5807 |
| engine + conformal | 2.5% | +0.2956 → +0.3061 | −0.1942 → −0.1465 | 0.0000 → 0.0017 |

**One A8 sentence reverses, in the paper's favour.** The OA currently says that on the McNeil–Frey residual
"FHS … sits closest to zero and is the only entrant the test does not reject at 1% (−0.10, $p=0.07$)". Under
the converged integral the **engine** is closest to zero (−0.058 against FHS's −0.101) and is the clearest
non-rejection ($p=0.383$ against FHS's 0.070). The $Z_2$ claims survive and strengthen slightly: the engine is
still closest to zero at 1% (−0.05, printed as −0.06) and still tied with GARCH-$t$ at 2.5% (−0.10 both). The
conformal row still rejects at 2.5%, so Stage 4's verdict is unchanged.

**`scale_gate_results.json`.** Threshold 13.0953 and fire rate 9.97% are **identical** — both VaR-only. The
three GBM rows each fall by 1.1–1.3e-3; `param_garch_t` and `param_bteg` are bit-identical.

| α | gate | always-`engine_garch_t` | always-`engine_bteg` |
|---|---|---|---|
| 1% FZ0 | 2.22333 → **2.22206** | 2.23133 → 2.23007 | 2.23636 → 2.23506 |
| 2.5% FZ0 | 1.92574 → **1.92448** | 1.93320 → 1.93197 | 1.93332 → 1.93220 |

| gate vs | 1% old → new | 2.5% old → new |
|---|---|---|
| always-GARCH-$t$, overall | +6.26 → **+6.34** | +8.04 → **+8.12** |
| always-`bteg`, overall | +0.80 → **+0.82** | +0.73 → **+0.76** |
| always-`bteg`, top decile | −1.65 → **−1.63** | −1.69 → **−1.68** |

**The sign I flagged for rechecking holds.** P1 (at least as good as every fixed choice overall) **passes** —
the gate still has the lowest FZ0 at both levels. P2 (within noise of always-`bteg`, $|DM|<2$) **passes** at
+0.82 and +0.76. P3 is VaR-only and unchanged. The agreed description stands verbatim: significantly better
than GARCH-$t$ throughout, statistically equal to always running the robust scale, and worse than it in the
top decile.

**`tail_vs_bteg_results_refit.json`.** FZ0 clauses of P1: −3.35 → **−3.09** (1%), −2.93 → **−2.72** (2.5%),
both still past the DM > 2 bar, so **both FZ0 clauses still pass**. The pinball clauses come back
**bit-identical** (−2.13 at 1%, −0.83 at 2.5%), confirming pinball carries no ES. P1's grading is therefore
unchanged: 3 of 4 clauses pass, the VaR-only pinball comparison at 2.5% remains a tie, and the consequence
stands — the joint $(\VaR,\ES)$ claim passes at both levels under refits while the VaR-only one does not.

**`walkforward_bteg_results.json`.**

| manuscript line | field | published | new |
|---|---|---|---|
| L1095 A3, accuracy layer vs plain Beta-$t$-EGARCH, overall | `fz0/α/param_bteg/vs_engine_garch_t/overall` | +1.21 / +0.85 | **+1.23 / +0.90** |
| L1095 A3, same, top mk$_{63}$ decile | `…/top_mk63_decile` | −3.70 / −5.24 | **−3.65 / −5.12** |
| L1097 A7, engine-vs-engine overall | `fz0/α/DM_engine_garch_t_vs_engine_bteg/overall` | −0.31 / −0.01 | **−0.31 / −0.02** |
| L1097 A7, same, top decile | `…/top_mk63_decile` | +5.37 / +7.25 | **+5.46 / +7.37** |
| L1097, GARCH-$t$ beaten under refits | `fz0/α/param_garch_t/vs_engine_garch_t/overall` | +3.59 / +2.89 | **+3.46 / +3.10** |

Every verdict in both sentences survives: within noise overall, significant in the top decile, and Stage 1
stays GARCH-$t$ under the rule fixed before the run.

**A second unsourced pair, flagged like the 59–79%.** L1097 says the accuracy layer beats "GARCH-$t$ and FHS by
DM 3.6 and 2.7 at 1\% and 3.1 and 2.6 at 2.5\%". The GARCH-$t$ halves map to `param_garch_t` (3.59 → 3.46 and
2.89 → 3.10). **The FHS halves — 2.7 and 2.6 — match no field in `walkforward_bteg_results.json` or
`walkforward_results.json`.** The semantically correct row is `uncond_garch_t`, the unconditional residual
quantile with its tail mean, which is the pooled-FHS construction; it reads 3.04 → 3.16 at 1% and 3.26 → 3.40
at 2.5%, not 2.7/2.6. R59 must either name the field or replace the pair with `uncond_garch_t`'s values.

### 7.7 Convention status, all sources

Every source the manuscript quotes an FZ0, ES, Acerbi–Székely or McNeil–Frey number from is now converged and
carries an `es_convention` field:

| converged this pass | already converged | needs nothing |
|---|---|---|
| `fz_fullpanel`, `pzc_taylor_acc`, `garch_evt`, `engine_esbt`, `scale_gate`, `overlay_engine` (×2), `tail_vs_bteg` (×2), `walkforward_bteg`, holdout battery | `bench_all`, `exception_battery`, `robust_engine` (×3), `frtb_table_200`, `es_converged`, `synthetic_truth` | `walkforward_hybrid` (pinball only), `amort_pit` (no ES) |

Controls held everywhere they could be checked: closed-form and empirical rows bit-identical, VaR-only
statistics identical, pinball identical, and the GPD parameterisation guard passed in every script that uses
the closed-form sub-floor integral.

One labelling note from that audit: `es_converged_results.json` is the only ES-bearing file without an
`es_convention` key, because it is the diagnostic that *measures* the convention rather than adopting one —
every row appears at $M \in \{20, 200, 2000\}$ and at the committed rule, so both conventions are present by
construction. `job_es_converged.py` now says so in its output; the committed file predates that line.
