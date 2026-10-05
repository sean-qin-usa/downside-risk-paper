# Semiparametric Value-at-Risk and Expected Shortfall with a Real-Time Misspecification Score

Sean Qin, Northwestern University. Submitted to the *Journal of Financial Econometrics*, September 2026.

## Manuscript

The manuscript is `submission/paper_A_jfec.pdf` and the online appendix is `submission/paper_A_jfec_online_appendix.pdf`, both in the journal's review format, double spaced with endnotes and with the tables and figures collected at the end under "[Table N about here]" markers. The LaTeX sources and `refs_v3.bib` are in the same folder.

The version under review is tagged `r32-submitted`. The files in the repository are ahead of it, and the changes are listed sentence by sentence in `docs/CHANGE_LEDGER_R34.md` and, for the later revisions, `docs/CHANGE_LEDGER_R54.md` through `docs/CHANGE_LEDGER_R58.md` (summarized in `docs/DEVELOPMENT_HISTORY.md`). The revisions add a same-rows comparison against standalone McNeil-Frey GARCH-EVT, per name and with a pooled tail (Sections 1.2, 4.1 and 5.1, Figure 2 and a new online appendix section; `code/paper/job_garch_evt.py`); re-score every model in the comparison set on the same 221,600 test rows with the Giacomini-White conditional predictive ability regression, Murphy diagrams, the Engle-Manganelli dynamic quantile test and a 90% model confidence set (`code/paper/job_bench_all.py`; online appendix Tables OA.10 to OA.12); restate equation (5) and Stages 3 to 4 in the order the code runs them, with the GPD estimator named and two symbol collisions removed; report Christoffersen conditional coverage at 97.5% as well as 99%; and commit a result file for the residual-hybrid annual-refit walk-forward. None of the post-submission runs moved a number in the submitted tables by more than rounding, and the frontier results replicate against every benchmark added since. Standalone GARCH-EVT shows no frontier and loses to the estimator in the top decile and on the joint score, the conditional-predictive-ability regression and the Murphy diagrams confirm the ordering, and the point-in-time, calendar-split and annual-refit checks stand. One claim was narrowed rather than strengthened: the abstract's joint (VaR, ES) sentence now reads "lowest among standard benchmarks at 1% and never significantly beaten at 2.5%", because on the full comparison set GJR-GARCH-skew-t and the Taylor model are within noise of the estimator at 2.5%. Later runs, each with its prediction written into the script header before it ran, measured the ten-day extension against iterated GARCH-t and bootstrapped-path benchmarks (about half of its edge over the sqrt(h) rule is the rule itself; `code/paper/job_tenday_diag.py`), retrained the frontier learner without its residual-dispersion features and with the GARCH scale alone (the top-decile edge keeps +1.6% to +1.9% and its ordering; `job_feature_ablation.py`), scored a GARCH-t with a time-varying degrees-of-freedom law (it closes 9% of the top-decile gap; `job_arcd_bench.py`), and reran the FRTB battery on the 200-name panel (same ordering; `job_frtb200.py`). A whole-paper audit in September fixed mislabelled or mismatched figures between sections without changing any result; every edit is in the ledger.

## Summary

A score built from the excess kurtosis and asymmetry of recent GARCH-standardized residuals predicts when a flexible-shape quantile estimator improves on a parametric VaR/ES model. In the top score decile the pooled gradient-boosted estimator beats GARCH-t by 3.0% of pinball loss (DM 10.5). The pattern holds on an untouched 2000 to 2013 panel under a frozen specification, under strict calendar splits, under an annual-refit walk-forward, and in a point-in-time universe that keeps delisted names. Outside the top decile the advantage shrinks toward zero and no region shows a loss.

Measured against a jump-robust GARCH that caps how far one shock propagates into the variance, the top-decile edge falls to between 0.3% and 0.5% (DM 2.5 to 4.7). Most of the advantage is the standard filter's post-shock scale error, and the part that survives a robust scale is conditional-shape information. The estimator passes the date-clustered exception tests at both regulatory levels through the 2008 window. On the full panel its accuracy layer has a lower joint (VaR, ES) FZ0 score than GARCH-t and every FHS variant at both levels, is within noise of GJR-GARCH-skew-t at both levels and of the Taylor ES-CAViaR model at 2.5%, and on pinball ties SAV-CAViaR, which shows the same frontier.

## Layout

Every result file whose ES is numerically integrated carries an `es_convention` key naming the convention it was computed under. Four files that print FZ0 numbers do not carry the key because their ES needs no quadrature: `fz_strict_calibration_results.json` and `fz_aci_results.json` (McNeil-Frey GPD closed form throughout), `fz_score_results.json` (no gradient-boosted body at all) and `frontier_robust_results.json` (pinball only, no ES).

| Folder | Contents |
|---|---|
| `submission/` | Manuscript, online appendix, bibliography, generated table bodies |
| `code/paper/` | The scripts behind this paper; the table below lists the one behind each exhibit |
| `code/amortization/` | The amortization and IQN study (one pooled fit across names, transfer to unseen names, age curve) |
| `code/gbc/`, `code/strategies/`, `code/data/`, `code/competitions/` | The rest of the research program the paper came out of: generative posterior work, option-selling backtests, data pulls and diagnostics, forecasting competition entries |
| `results/<group>/` | Result files as JSON, one per script run, grouped the same way as `code/`. The numbers in the paper come from `results/paper/` |
| `docs/` | Change ledger, cover letter, journal requirements, SSRN abstract |
| `figures/` | Exported chart for the amortization study |

| Script | Result file | Exhibit |
|---|---|---|
| `code/paper/job_composite.py` | `results/paper/composite_holdout_results.json` | Table 1, the score frontier on the 200-name panel |
| `code/paper/job_wrds_holdout.py` | `results/paper/holdout_frontier_results.json` | Figure 1, the 2000 to 2013 holdout under the frozen specification |
| `code/paper/job_fz_fullpanel.py` | `results/paper/fz_fullpanel_results.json` | Full-panel FZ0 joint loss on its own engine build. **Figure 2's bars and the Section 5.1 DMs are taken from `bench_all_results.json`**, the canonical accuracy-layer forecast (FZ0 2.14653 / 1.84863, conf975 -0.3426) that `job_bench_all.py`, `job_exception_battery.py`, `job_robust_engine.py` and `job_garch_evt.py` all reproduce bit-identically; it is also the GAS fit of record, since GAS is fitted per name by multi-start Nelder-Mead and its own FZ0 differs in the fourth decimal between runs |
| `code/paper/frtb_table_canonical.py` | `results/paper/frtb_table_results.json` | The twelve-level FRTB battery with exact tail-integral ES on the 140-name subpanel (the source of Table 4 through R57) |
| `code/paper/job_pzc_taylor.py` | `results/paper/pzc_taylor_acc_results.json` | GAS and Taylor ES-CAViaR entries of Figure 2, scored against the accuracy layer. `results/paper/pzc_taylor_results.json` is **superseded**: no script in the repository writes that filename, and the two agree exactly on every benchmark FZ0, so the `_acc` file is the same computation under its current name |
| `code/paper/frtb_stress_exact.py`, `code/paper/job_stress_dm.py` | `results/paper/stress_es_results.json` | Ten-day sections in both eras with the boundary purge |
| `code/paper/job_fz_strict_calibration.py` | `results/paper/fz_strict_calibration_results.json` | Strict-split conformal and FZ audit with the matched-information GARCH control |
| `code/paper/job_pit_universe.py` | `results/paper/pit_universe_results.json` | Point-in-time universe with delisting returns |
| `code/paper/job_calendar_split.py`, `code/paper/job_walkforward.py` | `results/paper/calendar_split_results.json`, `results/paper/walkforward_results.json` | Calendar splits and the annual-refit walk-forward |
| `code/paper/job_nurel.py`, `code/paper/job_mechanism.py` | `results/paper/nurel_results.json`, `results/paper/mechanism_results.json` | The nu-relative score and the Fama-MacBeth mechanism test |
| `code/paper/job_coherent.py` | `results/paper/coherent_results.json` | Monotonized curve audit; ES as the converged integral of the same curve |

Several jobs rebuild the accuracy layer themselves rather than reading the canonical forecast of `bench_all_results.json`, among them `job_coherent.py`, `job_fz_strict_calibration.py`, `job_fz_fullpanel.py` and `job_pzc_taylor.py`. Each refits the pooled GPD and the body on its own row set, so their engine FZ0 can differ from the canonical 2.14653 / 1.84863 in the fourth decimal (`coherent_results.json`: 1.84806 at 2.5%). Every FZ0 comparison against a benchmark in the paper uses `bench_all_results.json`. The splice-level, closed-form-ES and strict-split statements are comparisons within a single job's own build, and are quoted from that job.
| `code/paper/es_integral.py` | --- | The one implementation of the converged ES integral, imported by every job that integrates one: the body on an interpolant over `[alpha/40, alpha]`, the GPD branch evaluated exactly at each node, and the sub-floor region in closed form. Self-tests against two analytic cases (`python es_integral.py`) |
| `code/paper/job_es_converged.py` | `results/paper/es_converged_results.json` | The ES-convention diagnostic: every affected row at 20, 200 and 2000 quadrature nodes beside the committed 20-node rule, with the two controls (VaR-only statistics and closed-form rows unchanged) reported as pass/fail |
| `code/paper/job_overlay_engine.py` | `results/paper/overlay_engine_results.json`, `..._holdout.json` | Stage 4: the static shifts and the adaptive (Gibbs-Candes) overlay on the engine's own breaches. The pre-committed accept rule is in the script header |
| `code/paper/job_scale_gate.py`, `tools/gate_indep.py` | `results/paper/scale_gate_results.json` | The score-gated Stage-1 scale under annual refits, and an independent recomputation of its FZ0 and Newey-West statistics that shares no code with the job |
| `code/paper/job_tail_vs_bteg.py` | `results/paper/tail_vs_bteg_results_frozen.json`, `..._refit.json` | Where the flexible tail adds on a robust scale, pinball reported at each level rather than averaged, under frozen fits and annual refits |
| `code/paper/job_tenday_envelope.py` | `results/paper/tenday_envelope_results.json` | The ten-day envelope and the variance-ratio rescaling, both tested and neither recommended |
| `code/paper/job_coldstart_peers.py` | `results/paper/coldstart_peers_results.json` | Cold start against pooled peer-group benchmarks, with the leave-one-out median taken by sorted rank |
| `code/paper/job_amort_pit.py` | `results/paper/amort_pit_results.json` | The amortization numbers recomputed with point-in-time characteristics, and both win-rate definitions stated. The own-history benchmark is an expanding-window Gaussian on the name's own prior returns, scored over seven levels from 0.05 to 0.95 |
| `code/paper/job_synthetic_truth.py`, `code/paper/make_synthetic_truth_results.py` | `results/paper/synthetic_truth_results.json` | Table OA.16: two data-generating processes with a known answer, three shift placebos, and the quadrature size check that motivated the ES convention |
| `code/paper/job_frontier_robust.py` | `results/paper/frontier_robust_results.json` | Table 2 and the jump-robust GARCH decomposition of the top-decile edge (Section 4): the frontier rebuilt on a bounded-news filter at three- and four-sigma caps |
| `code/paper/job_scaleshape_canonical.py` | `results/paper/scaleshape_canonical_results.json` | Realized-variance scale decomposition on large caps |
| `code/paper/job_perasset_v2.py` | `results/paper/perasset_v2_results.json` | Per-asset exception tests at 99% and 97.5% |
| `code/paper/job_garch_evt.py`, `code/paper/job_holdout_garch_evt.py` | `results/paper/garch_evt_results.json`, `results/paper/holdout_garch_evt_results.json` | GARCH-EVT on the same rows, design era and holdout (online appendix) |
| `code/paper/job_bench_all.py` | `results/paper/bench_all_results.json` | Every benchmark on the same rows (online appendix Tables OA.10 to OA.12) |
| `code/paper/job_walkforward_hybrid.py` | `results/paper/walkforward_hybrid_results.json` | Residual-hybrid annual refit |
| `code/paper/job_tenday_diag.py` | `results/paper/tenday_diag_results.json` | Ten-day extension against iterated GARCH-t, simulated GARCH-t paths and bootstrapped filtered paths, both eras, with the variance term-structure ratio and coverage by year |
| `code/paper/job_composite_profiles.py` | `results/paper/composite_profiles_results.json` | Table 1 rerun with the full decile profile of each signal (Online Appendix Figure OA.6) |
| `code/paper/job_feature_ablation.py` | `results/paper/feature_ablation_results.json` | Table 1 learner retrained without the residual-dispersion features and with the GARCH scale alone; decile profiles under each |
| `code/paper/job_arcd_bench.py` | `results/paper/arcd_bench_results.json` | State-adaptive parametric benchmark: GARCH-t with a time-varying degrees-of-freedom law fitted per name, scored on the Table 1 rows |
| `code/paper/job_frtb200.py` | `results/paper/frtb_table_200_results.json` | Table 4 and Figure OA.4, the twelve-level FRTB battery on the 200-name panel of Table 1, and the per-asset pass rates of Figure OA.2 |
| `code/paper/job_robust_engine.py` | `results/paper/robust_engine_results.json`, `..._taq30.json`, `..._holdout.json` | Beta-t-EGARCH (score-driven) Stage-1 scale beside GARCH-t and a bounded-news GARCH, as a benchmark and as the scale under the flexible shape; design era, the 30-name realized-variance panel and the frozen 2000-2013 holdout |
| `code/paper/job_exception_battery.py` | `results/paper/exception_battery_results.json`, `..._holdout.json` | The per-asset and date-clustered exception tests and the ES backtests on every Stage-1 scale and Stage-4 variant, one row set; holdout run reports 2008-2009 on its own dates |
| `code/paper/job_bench_all_bteg.py` | `results/paper/bench_all_bteg_results.json` | Every benchmark of `job_bench_all.py` re-scored against a Beta-t-EGARCH Stage 1 on the identical rows, with the decile profile referenced to that scale's own parametric tail |
| `code/paper/job_walkforward_bteg.py` | `results/paper/walkforward_bteg_results.json` | The annual-refit walk-forward run for both Stage-1 scales side by side, pinball and FZ0 |
| `code/paper/job_coldstart_peers.py` | `results/paper/coldstart_peers_results.json` | Cold start measured against pooled peer-group benchmarks by listing age, with leave-one-out point-in-time peer scales |
| `code/paper/make_table_deployed.py` | `submission/tables/tab_deployed.tex` | Every recorded metric for one deployed configuration as a single exhibit |
| `code/paper/make_table_scaleshape.py` | `submission/tables/tab_scaleshape_bteg.tex` | The Beta-t-EGARCH block for the scale-shape decomposition table |
| `code/paper/make_table_frtb200.py` | `submission/tables/tab_frtb200.tex` | The FRTB battery table regenerated on the 200-name panel of Table 1, so the battery and the frontier share one set of rows |

`code/paper/toy_example.py` runs the whole pipeline on synthetic data and needs no licensed input. Job scripts take the project root, where the licensed panel lives, from the `GBC_PROJ` or `GBC_PROJECT_DIR` environment variable where they read one, and otherwise from a path set at the top of the script.

## Environment

Results committed from 2026-10-01 were produced with Python 3.10.12, numpy 1.26.4, scipy 1.15.3,
pandas 2.3.3, **scikit-learn 1.7.2** and arch 8.0.0. The scikit-learn version is worth pinning rather
than noting: the gradient-boosted residual-quantile stage is the one estimated component in the
pipeline, and figures produced on a different build reproduce to roughly 0.04 percentage points of
pinball rather than exactly. Every job now sets `random_state=0` on that stage, so a rerun on a fixed
build is bit-reproducible; result files committed before that date were produced without the seed and
are not reproducible from their own scripts. `docs/R57_RESULTS_HARVEST.md` lists every figure that
moved when the seed was fixed.

## Data

Returns come from CRSP through WRDS and from Bloomberg under the author's licenses and are not redistributed; no data file is tracked. The WRDS panels rebuild from the queries documented in the scripts for any subscriber. The Bloomberg exhibits are kept as run, since terminal access ended in mid-2026.
