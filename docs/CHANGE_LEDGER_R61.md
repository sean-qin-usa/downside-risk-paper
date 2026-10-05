# Change ledger R61/R62: the top-decile mechanism, the holdout test's power, and the seeding sweep

Applied by `docs/apply_R61.py` on top of `fd9e5c8`. Seven main-text edits in both builds, one online-appendix
edit in both, one edit to a generated table file. Sources: `docs/R61_DIAGNOSTICS.md`,
`docs/R62_SPLICE_PRECOMMIT.md`, `docs/R62_SPLICE_RESULTS.md`, and the reseeded result files below.

## N01 §5.1: the top-decile joint-loss result is a property of the splice

The paper said the joint-score edge is "less concentrated than the pinball edge" and left it there. It now gives
the mechanism and the price, from `tools/r61_diag_canon200.json` and `tools/r62_splice_canon200.json`:

| statement | source field | value |
|---|---|---|
| body beats the accuracy layer in the top decile | `per_alpha/*/top_vs_bulk/top_decile/body_minus_engine` | 0.068 (DM −6.42) at 1%, 0.060 (DM −6.77) at 2.5% |
| and ties it in deciles 1–9 | `.../deciles_1to9/...` | 0.002 (DM 0.35), 0.001 (DM 0.21) |
| the pooled GPD is a constant in $z$ | `deciles[*]/z_evt_at_alpha` | −2.554, −1.880 |
| the body pulls its tail in as the score rises | `deciles[*]/z_body_at_alpha` | −2.527 → −2.252 |
| so the override widens | `deciles[*]/z_override_depth_when_binding` | 0.188 → 0.523 |
| anchoring removes the deficit | r62 `p0_0.05_anchored` top decile | DM +0.09 against −6.59 unconditional |
| and raises the 97.5% breach rate | r62 `p0_0.05_*` breach at 0.025 | 2.78% → 3.09% |
| so the tail buys coverage at a price | r62 overall FZ0 | 0.013 (2.5%) and 0.019 (1%) |

The anchored variant was **pre-registered** in `docs/R62_SPLICE_PRECOMMIT.md` before it was estimated, and
**rejected** by its own decision rule (`docs/R62_SPLICE_RESULTS.md`): it fails P2 and P3 at 2.5%. The paper
reports only the trade, which is what the rejected run measured.

## N02, N03, P01: the holdout exception test has little power

Backed out of the committed battery's own breach and $t$ (`exception_battery_results_holdout.json`): the
accuracy layer's clustered standard error is **0.53pp at 1%** and 0.28pp at 2.5%, so the test rejects no breach
rate below **2.05%** at 1%. GARCH-$t$'s standard error on the same rows is 0.25pp, so the engine's test is
2.1× less powerful. The online-appendix note adds why: within-date clustering is almost identical for the two
models (implied $\rho$ 0.162 against 0.179), and the gap is serial dependence in the breach series.

## N04: the design-effect note was wrong out of era

`tab_deployed`'s note read the design effect as "an intra-date breach correlation of a few per cent". That holds
on canon200 (8.57, $\rho$ 0.038) and fails on the holdout, where the same arithmetic gives $\rho = 4.41$, which
is not a correlation. Decomposed in `tools/r61_diag_holdout.json`: the reported 824 is a within-date component
of 31 ($\rho = 0.162$) multiplied by a serial-correlation factor of 25. The note now says so.

**`submission/tables/tab_deployed.tex` is a generated candidate table and is `\input` by neither build** (both
have zero `\input`/`\include`; all tables are inlined). The edit is for repository correctness. No reader of the
compiled paper sees that note, which is also why the Panel C numbers below had to be found in the prose.

## N05, N06, N07: printed numbers that moved when five more jobs were seeded

| printed | was | now | source |
|---|---|---|---|
| calendar split, top-decile edge | +2.47%, DM 6.22 | **+2.45%, DM 6.03** | `calendar_split_results.json` |
| point-in-time universe, overall | +0.58%, DM 8.96 | **+0.61%, DM 9.16** | `pit_universe_results.json` |
| point-in-time, frozen top bucket (two mentions) | +2.94%, DM 6.6 | **+3.17%, DM 7.0** | same |
| strict split, accuracy advantage | DM 5.1 and 5.4 | **DM 5.0 and 5.1** | `fz_strict_calibration_results.json` |
| strict split, gap to the full-window benchmark | (−0.6, −2.5) | **(−0.7, −2.6)** | same |

Unchanged at printed precision and therefore not edited: the strict split's short-window benchmark (2.8, 3.8),
the adaptive-conformal walk (−3.2 → −1.5, now −3.18 → −1.52), and the ten-day figures, whose printed home is
the unused generated table rather than the prose.

## The seeding sweep

R59 said "all are now seeded"; that was scoped to the five jobs it touched. All **14** README-named jobs with an
unseeded `HistGradientBoostingRegressor` now carry `random_state=0`: 17 calls across
`frtb_stress_exact`, `frtb_table_canonical`, `job_calendar_split`, `job_fz_strict_calibration`,
`job_holdout_garch_evt`, `job_nurel`, `job_pit_universe`, `job_stress_dm`, `job_synthetic_truth`,
`job_tenday_diag`, `job_tenday_envelope`, `job_walkforward`, `job_walkforward_hybrid`, `job_wrds_holdout`.
A corrected detector confirms none remain in any README-named job.

**Two latent bugs found by trying to run them.**

1. `job_pit_universe.py` could not run at all. A path-portability comment had been inserted mid-line, so
   `t0=time.time(); lg=lambda s:print(...)` sat inside the comment and never executed; the job crashed on its
   first `lg(...)`. Fixed by moving the comment to the end of the line.
2. **`job_synthetic_truth.py` had never run.** R59 created it and added `random_state=0` to calls that already
   received one through `**HGB`, so all three calls raised `TypeError: got multiple values for keyword argument
   'random_state'`. The committed Table OA.16 result file is valid, having been produced by the scratch script it
   was derived from, but the committed generator was not runnable. Fixed by removing the explicit argument.
   The duplicate arose because the dict is defined as `P0=0.025; HGB=dict(...)`, and the first detector's regex
   was anchored to the start of a line. A repo-wide check with the corrected detector finds no other duplicate.

Seven jobs were made path-portable (`GBC_PROJ`/`GBC_PROJECT_DIR`) so they could be rerun off the Windows path:
`frtb_stress_exact`, `frtb_table_canonical`, `job_calendar_split`, `job_fz_strict_calibration`, `job_nurel`,
`job_stress_dm`, `job_walkforward`.

## Not done, and why

- **`job_stress_dm.py`** was not rerun to completion. It reads `stress_es_results.json`, which
  `frtb_stress_exact.py` regenerates, so it has to run after it; its printed claims live in the unused generated
  table rather than the prose, so nothing in the compiled paper depends on it.
- **Figure 1's holdout bars** (`(1,0.078) … (10,1.123)`) match **no committed result file** to three decimals,
  so the reseeded `holdout_frontier_results.json` cannot be mapped onto them. This is a third orphan of the kind
  R59 found in `pzc_taylor_results.json` and the 59–79% transfer rate, and it needs its source named before the
  figure can be updated.
- No new estimator was designed. The anchored splice is reported as rejected; a construction whose anchoring
  strength varies with the state would need its own pre-commitment.
