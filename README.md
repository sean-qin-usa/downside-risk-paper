# Semiparametric Value-at-Risk and Expected Shortfall with a Real-Time Misspecification Score

Sean Qin, Northwestern University. Submitted to the *Journal of Financial Econometrics*, September 2026.

## Manuscript

The manuscript is `submission/paper_A_jfec.pdf` and the online appendix is `submission/paper_A_jfec_online_appendix.pdf`, both in the journal's review format, double spaced with endnotes and with the tables and figures collected at the end under "[Table N about here]" markers. The LaTeX sources and `refs_v3.bib` are in the same folder.

The version under review is tagged `r32-submitted`. The files in the repository are ahead of it, and the changes are listed sentence by sentence in `docs/CHANGE_LEDGER_R34.md`. The revisions add a same-rows comparison against standalone McNeil-Frey GARCH-EVT, per name and with a pooled tail (Sections 1.2, 4.1 and 5.1, Figure 2 and a new online appendix section; `code/paper/job_garch_evt.py`); re-score every model in the comparison set on the same 221,600 test rows with the Giacomini-White conditional predictive ability regression, Murphy diagrams, the Engle-Manganelli dynamic quantile test and a 90% model confidence set (`code/paper/job_bench_all.py`; online appendix Tables OA.10 to OA.12); restate equation (5) and Stages 3 to 4 in the order the code runs them, with the GPD estimator named and two symbol collisions removed; report Christoffersen conditional coverage at 97.5% as well as 99%; and commit a result file for the residual-hybrid annual-refit walk-forward. No number in the submitted tables moved by more than rounding.

## Summary

A score built from the excess kurtosis and asymmetry of recent GARCH-standardized residuals predicts when a flexible-shape quantile estimator improves on a parametric VaR/ES model. In the top score decile the pooled gradient-boosted estimator beats GARCH-t by 3.0% of pinball loss (DM 10.5). The pattern holds on an untouched 2000 to 2013 panel under a frozen specification, under strict calendar splits, under an annual-refit walk-forward, and in a point-in-time universe that keeps delisted names. Outside the top decile the advantage shrinks toward zero and no region shows a loss.

Measured against a jump-robust GARCH that caps how far one shock propagates into the variance, the top-decile edge falls to between 0.3% and 0.5% (DM 2.5 to 4.7). Most of the advantage is the standard filter's post-shock scale error, and the part that survives a robust scale is conditional-shape information. The estimator passes the date-clustered exception tests at both regulatory levels through the 2008 window. On the full panel its accuracy layer has a lower joint (VaR, ES) FZ0 score than GARCH-t and every FHS variant at both levels, is within noise of GJR-GARCH-skew-t at both levels and of the Taylor ES-CAViaR model at 2.5%, and on pinball ties SAV-CAViaR, which shows the same frontier.

## Layout

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
| `code/paper/job_fz_fullpanel.py` | `results/paper/fz_fullpanel_results.json` | Figure 2, full-panel FZ0 joint loss |
| `code/paper/frtb_table_canonical.py` | `results/paper/frtb_table_results.json` | Table 3, the twelve-level FRTB battery with exact tail-integral ES |
| `code/paper/job_pzc_taylor.py` | `results/paper/pzc_taylor_results.json` | GAS and Taylor ES-CAViaR entries of Figure 2 |
| `code/paper/frtb_stress_exact.py`, `code/paper/job_stress_dm.py` | `results/paper/stress_es_results.json` | Ten-day sections in both eras with the boundary purge |
| `code/paper/job_fz_strict_calibration.py` | `results/paper/fz_strict_calibration_results.json` | Strict-split conformal and FZ audit with the matched-information GARCH control |
| `code/paper/job_pit_universe.py` | `results/paper/pit_universe_results.json` | Point-in-time universe with delisting returns |
| `code/paper/job_calendar_split.py`, `code/paper/job_walkforward.py` | `results/paper/calendar_split_results.json`, `results/paper/walkforward_results.json` | Calendar splits and the annual-refit walk-forward |
| `code/paper/job_nurel.py`, `code/paper/job_mechanism.py` | `results/paper/nurel_results.json`, `results/paper/mechanism_results.json` | The nu-relative score and the Fama-MacBeth mechanism test |
| `code/paper/job_coherent.py` | `results/paper/coherent_results.json` | Monotonized curve audit; ES as the integral of the same curve |
| `code/paper/job_scaleshape_canonical.py` | `results/paper/scaleshape_canonical_results.json` | Realized-variance scale decomposition on large caps |
| `code/paper/job_perasset_v2.py` | `results/paper/perasset_v2_results.json` | Per-asset exception tests at 99% and 97.5% |
| `code/paper/job_garch_evt.py`, `code/paper/job_holdout_garch_evt.py` | `results/paper/garch_evt_results.json`, `results/paper/holdout_garch_evt_results.json` | GARCH-EVT on the same rows, design era and holdout (online appendix) |
| `code/paper/job_bench_all.py` | `results/paper/bench_all_results.json` | Every benchmark on the same rows (online appendix Tables OA.10 to OA.12) |
| `code/paper/job_walkforward_hybrid.py` | `results/paper/walkforward_hybrid_results.json` | Residual-hybrid annual refit |

`code/paper/toy_example.py` runs the whole pipeline on synthetic data and needs no licensed input. Job scripts take the project root, where the licensed panel lives, from the `GBC_PROJ` or `GBC_PROJECT_DIR` environment variable where they read one, and otherwise from a path set at the top of the script.

## Data

Returns come from CRSP through WRDS and from Bloomberg under the author's licenses and are not redistributed; no data file is tracked. The WRDS panels rebuild from the queries documented in the scripts for any subscriber. The Bloomberg exhibits are kept as run, since terminal access ended in mid-2026.
