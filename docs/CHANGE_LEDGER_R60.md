# Change ledger R60 (scope of the theory; estimation scheme; engine builds)

Applied by `apply_R60.py` on top of `a04779a` to both manuscript builds and the README. No number changes; the abstract and Online Appendix are untouched. Tested on a clean export of `a04779a`: every anchor matches once, the preprint compiles with no undefined references, and `sec:conventions` resolves to Section 3.3.

## M1 intro: scope of Francq-Zakoian

**Why.** The intro said Francq and Zakoian (2015) supply 'the two-step estimator's asymptotic theory', which a reader can take to mean the paper's estimator. Their result is for the filtered empirical quantile.

**Before.**

```
\citep{engle2004caviar}, and \citet{francqzakoian2015} supply the
two-step estimator's asymptotic theory.
```

**After.**

```
\citep{engle2004caviar}, and \citet{francqzakoian2015} supply
asymptotic theory for the filtered empirical quantile.
```

## M2 Section 2.2: no theory for a pooled tree learner

**Why.** Same scope point at the place the estimator is defined. A pooled gradient-boosted second stage with a GPD splice has no asymptotic theory; the paper's claims are forecast comparisons, so the sentence points to the inference conventions.

**Before.**

```
This two-step design, a parametric filter followed by a residual quantile, is the one whose asymptotics \citet{francqzakoian2015} derive, and \citet{mcneilfrey2000} fit the same filter by Gaussian pseudo-likelihood while explicitly declining to believe the innovation law it assumes.
```

**After.**

```
This two-step design, a parametric filter followed by a residual quantile, is the one whose asymptotics \citet{francqzakoian2015} derive for an empirical residual quantile, and \citet{mcneilfrey2000} fit the same filter by Gaussian pseudo-likelihood while explicitly declining to believe the innovation law it assumes. No comparable result covers a pooled tree learner in the second stage, so the paper's inference is on the forecasts (Section~\ref{sec:conventions}).
```

## M3 conventions: Giacomini-White and the estimation scheme

**Why.** Giacomini and White (2006) test forecasting methods with estimation included, for any estimator, under fixed or rolling windows. Every headline comparison uses the fixed 60/40 scheme. The annual-refit runs use expanding windows, which their theory excludes, and the paper already treats them as robustness checks. giacominiwhite2006 and diebold2015 are already cited.

**Before.**

```
the use \citet{diebold2015} endorses.
```

**After.**

```
the use \citet{diebold2015} endorses. Estimation is part of each forecasting method, and the conditional predictive-ability framework of \citet{giacominiwhite2006} holds for any estimator, parametric or not, under the fixed estimation scheme of every headline comparison; the annual-refit runs use expanding windows, which that framework does not cover, and are reported as robustness checks.
```

## R1 README

**Why.** The README names `bench_all_results.json` as the canonical accuracy-layer forecast but not which jobs build their own engine. coherent_results.json gives 1.84806 at 2.5% against the canonical 1.84863. The paragraph says which statements are within-job comparisons.

**Added after the `job_coherent.py` row.**

```
Several jobs rebuild the accuracy layer themselves rather than reading the canonical forecast of `bench_all_results.json`, among them `job_coherent.py`, `job_fz_strict_calibration.py`, `job_fz_fullpanel.py` and `job_pzc_taylor.py`. Each refits the pooled GPD and the body on its own row set, so their engine FZ0 can differ from the canonical 2.14653 / 1.84863 in the fourth decimal (`coherent_results.json`: 1.84806 at 2.5%). Every FZ0 comparison against a benchmark in the paper uses `bench_all_results.json`. The splice-level, closed-form-ES and strict-split statements are comparisons within a single job's own build, and are quoted from that job.
```
