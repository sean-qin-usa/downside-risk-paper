# Development history

This file records how the manuscript evolved, for readers of the replication package. The paper itself keeps only the lineage that changes how a number should be read (what was frozen before the holdout pull, the composite score being post hoc on the holdout, the Beta(0.3, 0.3) sampling measure being set after the untuned IQN run, the calibration-split leakage check). Everything else that once stood in the text is listed here with a pointer to the ledger that removed it.

## Ledgers

- `docs/CHANGE_LEDGER_R34.md`: text pass on the submitted R32 (notation audit, decision-review fixes, accepted Sep 8 sentences). 77 edits.
- `docs/CHANGE_LEDGER_R54.md`: focus pass removing research artifacts and duplicates from the body. 33 edits, no number changed.
- `docs/CHANGE_LEDGER_R55.md`: the scale-by-shape table (Table 2), the robust-GARCH literature, and the sort behind the robust-engine number.
- `docs/CHANGE_LEDGER_R57.md`: the joint-loss claim scoped to the industry-standard benchmarks of Section 1.2, the Beta-t-EGARCH results disclosed (frozen fits, annual refits, and as Stage 1), the McNeil-Frey limitation, and redundant or misattributed sources removed.
- `docs/CHANGE_LEDGER_R58.md`: Table 4 and its notes moved to the 200-name panel of Table 1, seed-fixed figures, and Beta-t-EGARCH in the Section 4 decomposition.
- `docs/R54_RESULTS_HARVEST.md` and `docs/R57_RESULTS_HARVEST.md`: the pre-registered runs behind R55 to R58, with predictions graded.
- The README's post-submission paragraph lists the empirical runs added after R32 (standalone GARCH-EVT, same-rows benchmark battery, ten-day diagnostics, feature ablation, adaptive-nu GARCH-t, FRTB battery on the 200-name panel) and the scripts that produced them.

## Origins of the estimator

The project began as a generative Bayesian computation (GBC) exercise, in the sense of Polson and Sokolov (2023): an implicit quantile network trained as a quantile generator, with a Gibbs-posterior reading of pinball empirical risk minimization (Jiang and Tanner, 2008; Bissiri, Holmes and Walker, 2016). Three things changed as the work turned into a downside-risk paper.

1. The estimator's role moved from generation to point accuracy at fixed regulatory levels. Gradient-boosted quantile trees beat the tuned network by about 0.75% of pinball loss on tabular state, so the trees carry every reported forecast and the network is a benchmark (Table 5). The inverse-CDF sampler that the network provides, and the lemma that certified it (Lemma 3 in drafts through R53), were never used by a result and were removed in R54, together with the online-appendix paragraph on scenario generation that leaned on it.
2. The Gibbs posterior was dropped as a tool. Earlier drafts used it as a hierarchical shrinkage check and described the own-history correction as an "affine Gibbs update". Code review showed that update to be a deterministic age/(age+80) shrinkage, and no table used the posterior. A naive Gibbs posterior also understated VaR-parameter uncertainty by roughly 1.6 times relative to a block bootstrap, because its temperature is not identified and it ignores serial dependence. The paper reports point forecasts and conformal coverage, so the posterior does not enter; the generalized-Bayes lineage survives as a citation in Section 1.3.
3. The comparison with GBC, once its own subsection (Section 2.5 through R53), was compressed in R54 to one paragraph at the end of Section 2.3, since each of its five differences (tail-weighted sampling measure, standardized-residual target, GPD tail beyond the training support, conformal stage, amortization across the real cross-section) is introduced by the stage it refers to.

## Computational conventions that changed

- Expected Shortfall was first computed by the McNeil-Frey GPD closed form and later as the numerical integral of the min-envelope quantile curve. The two agree to the third decimal on FZ0 (1.849 against 1.851 at 2.5%) and leave every comparison in place; the figures are in the online appendix.
- The FZ-estimated dynamic-ES benchmarks (Patton, Ziegel and Chen, 2019; Taylor, 2019) were first fitted with a single-start optimizer, which did not converge for every name. The reported fits use three starts and converge for all 200 names.
- In the 43-instrument sweep, an apparent electricity win in an early run reversed once the return transformation was corrected for near-zero prices. The reported sweep uses the corrected transformation.
- The transfer pipeline was also scored, as an external check, on the M5 competition data (leakage-safe weighted scaled pinball 0.269, benchmark tier). This is outside the paper's object and is recorded in the online appendix.

## Statements removed as unreported or uncited

- "At hourly frequency the picture inverts" for the crypto series: no hourly result is reported.
- "Many banks have moved back toward the standardized approach": no citation was attached.
- Models named as not run (Zumbach long-memory EWMA; quantile regression forests).
- A reporting convention referring to a 0.05 coverage target and credible intervals, neither of which the paper contains.

## Stage 1 and the Beta-t-EGARCH runs

A Beta-t-EGARCH Stage 1 was tested after submission. It lowers the engine's FZ0 under the frozen 60/40 fits and on the 2000-2013 holdout, but under annual refits the two engines are within noise overall, which failed the prediction written into `job_walkforward_bteg.py` before the run. Under that script's rule Stage 1 stays GARCH-t and the Beta-t-EGARCH engine is reported as a frozen-fit result, with the refit split by region (within noise overall, ahead in the top decile of the score). An earlier harvest called the swap "withdrawn" and claimed the GARCH-t engine was never significantly beaten on FZ0; both statements were corrected in `docs/R57_RESULTS_HARVEST.md`.

## Reproducibility

Result files committed before 2026-10-01 were produced without a fixed seed in the gradient-boosting early-stopping split and are not reproducible from their own scripts to the last digit. The seed is now fixed, scikit-learn 1.7.2 is pinned in the README, and every figure that moved is tabulated in `docs/R57_RESULTS_HARVEST.md`; one test verdict moved (the accuracy layer's Acerbi-Szekely p at 2.5%, 0.0394 to 0.0515).

