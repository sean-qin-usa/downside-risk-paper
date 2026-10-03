# Change ledger R59 (the correcting commit: converged ES, canonical engine, corrected battery, adaptive Stage 4, robust-scale and gate results)

Applied by `apply_R59.py` to both manuscript builds and both online appendices, on top of `a1f997f`. Tested on a clean export of `a1f997f`: every anchor matches exactly once in each file, both documents compile with no undefined references, the abstract is unchanged (97 words), and the new Online Appendix table is OA.16. Every number below was checked against the named result file through the Mac sync on 2026-10-02. Source of the decisions: `docs/R59_RESULTS_HARVEST.md` sections 1-7.

## Preconditions and companion work for ai2 (same commit)

1. **Strict-split DMs (M07).** `fz_strict_calibration_results.json` has no `es_convention` key, so the engine FZ0 behind 'DM 5.1 and 5.4' and the gap '$-0.6$, $-2.5$' is 20-node. Rerun `job_fz_strict_calibration.py` on `es_integral`; if `garchcp_minus_strict_noshift` or `garch_minus_strict_noshift` moves at printed precision, edit M07's replacement text before applying. The 2.8/3.8 pair is closed-form against closed-form and cannot move.
2. **Online Appendix Table on realized measures (`tab:rgarch`) and Section 5.3 'DM 6.2 and 4.5'.** Source `scaleshape_canonical_results.json`, 20-node ES by its own note. Rerun `job_scaleshape_canonical.py` converged; update the table, its note ('ES the 20-node integral') and the Section 5.3 sentence in the same commit if anything moves at printed precision (the converged taq30 reproduction in `robust_engine_results_taq30.json` reads 6.3 and 4.9 for the daily core, so at least one entry likely moves).
3. **Sweep for lines the section-7 grep missed.** Section 7 grepped for FZ0/ES/Acerbi/McNeil. Two FZ0 numbers sat on lines without those words: the strict-split DMs (precondition 1) and 'the top misspecification decile carries DM 2.4 at the 1% level against DM 0.2 in the panel bulk', which matches no result file and is replaced in M25 by bench_all's 3.38/4.71. Please grep both builds and both OAs for 'joint', 'Fissler', 'shortfall', 'concession' and 'DM' near 'FZ' and report any other number without an `es_convention` source.
4. **Realized ES of the two GBM rows of Table 4.** `frtb_table_200_results.json` gives realized −6.112 and −6.138 against the printed −6.12 and −6.15. Realized is VaR-only and should not move. R59 prints the file's values; please say which run the printed ones came from.
5. **Commit the generator of `synthetic_truth_results.json`** (not under `code/paper/` in the sync) and give it a README row, since Table OA.16 cites the result file.
6. **README.** Mark `pzc_taylor_results.json` superseded (no script writes it) or delete it; point Figure 2 and Section 5.1 at `bench_all_results.json` (GAS fit of record); add rows for `job_overlay_engine.py` (with a pointer to the pre-committed rule in its header), `job_scale_gate.py`, `gate_indep.py`, `job_tail_vs_bteg.py`, `job_amort_pit.py`, `job_es_converged.py`, `es_integral.py`, `job_tenday_envelope.py`, `job_coldstart_peers.py`; `.gitignore` for data files; correct the false comment in `job_frtb200.py`.
7. Commit as Sean Qin <44483708+sean-qin-usa@users.noreply.github.com>, no Co-Authored-By or Claude-Session trailers, no data files (no CSV, parquet, per-row panels).

## Edits

### M01 intro: tail on a robust scale (main text)

**Source and reason.** tail_vs_bteg_results_{frozen,refit}.json: engine_bteg vs param_bteg FZ0 DM -3.85/-4.68 (frozen), -3.09/-2.72 (refit). P1's FZ0 clauses pass under both protocols (harvest 7.5, 7.6).

**Before.**

```
re-estimation on realized-volatility residuals leaves the score's edge within noise. This attribution
```

**After.**

```
re-estimation on realized-volatility residuals leaves the score's edge within noise. On a Beta-$t$-EGARCH scale, whose response to a shock is bounded by construction, the flexible tail still lowers the joint $(\VaR,\ES)$ loss against that model's own Student-$t$ tail at both regulatory levels, under frozen fits and under annual refits. This attribution
```

### M02 intro: amortization edge and day-one calibration (main text)

**Source and reason.** amort_pit_results.json, point_in_time arm, age_curve: d15_30 edge 9.17%, d30_60 6.85, d60_120 6.98, d120_250 6.54. Benchmark own_param = expanding-window Gaussian, seven levels 0.05-0.95 (job_amort_pit.py:54-57,85; harvest 1). First-week calibration: coldstart_peers_results.json, peer_young_emp breach 7.13% at 1%, ages 1-5.

**Before.**

```
Its gains over own-history benchmarks are largest in the first trading month, at roughly 6--10\% of pinball loss. In its characteristics-only form it forecasts VaR and ES for assets with no return history at all, the day-one regime in which no per-asset model exists.
```

**After.**

```
Its gains over an own-history benchmark are largest early in a name's life, about 9\% of pinball loss at 15 to 30 trading days and 6.5--7\% through the first year, measured at moderate quantiles. In its characteristics-only form it forecasts VaR and ES for assets with no return history at all, the day-one regime in which no per-asset model exists, though those first-week forecasts are not yet calibrated at the regulatory levels (Section~\ref{sec:applications}).
```

### M03 intro: overlay named (main text)

**Source and reason.** Stage 4 is now the adaptive overlay (overlay_engine_results{,_holdout}.json; decision rule in the header of job_overlay_engine.py).

**Before.**

```
with an extreme-value left tail and an optional split-conformal overlay.
```

**After.**

```
with an extreme-value left tail and an optional adaptive conformal overlay.
```

### M04 design: conformal stage (main text)

**Source and reason.** Same.

**Before.**

```
conformal stage corrects the level of the quantile curve using its
measured miss rate on a held-out split, with the exchangeability-based
guarantee of Proposition~\ref{prop:conformal}.
```

**After.**

```
conformal stage corrects the level of the quantile curve using its
realized miss rate, updated daily, the online form of the split-conformal
construction whose exchangeability-based guarantee is Proposition~\ref{prop:conformal}.
```

### M05 Stage 4 rewritten (main text)

**Source and reason.** The committed static shift was estimated on the BODY's calibration errors and applied to the ENGINE, counting the EVT correction twice (date-clustered t -9.09 at 2.5%). Stage 4 is rebuilt as the Gibbs-Candes update c_{t+1}=c_t-gamma(b_t-alpha), c_0=0, gamma in {0.02,0.05} (job_overlay_engine.py:9, aci_shift). The pre-committed rule (P1 within 0.3pt of nominal at 2.5% on both panels; P2 |DM|<2 vs the accuracy layer) passes on both panels. The sentence 'repairs the 97.5% level that each model in the comparison set otherwise fails' is false on the 200-name panel and is removed.

**Before.**

```
\emph{Stage 4 (conformal finish).} On a held-out calibration split of size $n$, compute the signed errors $\varepsilon_i = \widehat{z}_i - \widehat{q}_\phi(\tau \mid s_i)$ against the Stage-2 body. Let $c_\tau$ be the $\lceil (n{+}1)\tau \rceil$-th order statistic of $\{\varepsilon_i\}$. The finished curve is shifted per level by this scalar, $\widetilde{Q}^{z}_t(\tau) = Q^{z}_t(\tau) + c_\tau$ with $Q^{z}_t$ the envelope of \eqref{eq:hybrid} \citep{vovk2005, lei2018}, the one-sided form of conformalized quantile regression \citep{romano2019cqr}. The shift is the last operation, after the minimum and the rearrangement. The \emph{accuracy layer} that carries the FZ0 comparisons of Section~\ref{sec:frtb} and Table~\ref{tab:frtb} sets $c_\tau\equiv0$; the \emph{conformal overlay} applies the shift at the 97.5\% level. This distribution-free layer repairs
the 97.5\% level that each model in the comparison set otherwise fails. The coverage statement it inherits is standard \citep{vovk2005, lei2018, romano2019cqr} and is restated here in the form the shift uses.
```

**After.**

```
\emph{Stage 4 (conformal finish).} The finished curve is shifted per level by a scalar, $\widetilde{Q}^{z}_t(\tau) = Q^{z}_t(\tau) + c_\tau$ with $Q^{z}_t$ the envelope of \eqref{eq:hybrid}, and the shift is the last operation, after the minimum and the rearrangement. In its static form $c_\tau$ is fixed once: on a held-out calibration split of size $n$, compute the signed errors $\varepsilon_i = \widehat{z}_i - Q^{z}_i(\tau)$ of the envelope and let $c_\tau$ be their $\lceil (n{+}1)\tau \rceil$-th order statistic \citep{vovk2005, lei2018}, the one-sided form of conformalized quantile regression \citep{romano2019cqr}. In the adaptive form used here the shift starts at zero and is updated each day from the panel breach frequency $b_{\tau,t}$ of the shifted forecast, $c_{\tau,t+1} = c_{\tau,t} - \gamma\,(b_{\tau,t}-\tau)$ \citep{gibbs2021aci}, so the shift in force on day $t+1$ uses only breaches observed through day $t$. The \emph{accuracy layer} that carries the FZ0 comparisons of Section~\ref{sec:frtb} and Table~\ref{tab:frtb} sets $c_\tau\equiv0$; the \emph{adaptive overlay} applies the update at both regulatory levels with $\gamma\in\{0.02,0.05\}$, and in era it moves the 97.5\% breach rate toward nominal at no cost in FZ0 (Section~\ref{sec:frtb}). The coverage statement of the static form is standard \citep{vovk2005, lei2018, romano2019cqr} and is restated here for a quantile model $\widehat{q}_\phi$ and its calibration errors.
```

### M06 Proposition 1 scope (main text)

**Source and reason.** 'the conformal stage is the only reason any entrant passes Kupiec at 97.5%' is false (rolling FHS passes pooled Kupiec at 97.5%, R58) and the static shift it described is withdrawn. Static shift on the engine's own errors: breach 2.15%, t -3.07 in era (miscoverage sign flips between calibration block and test era, job header).

**Before.**

```
The proposition's scope is narrower than its use. In practice the calibration split necessarily precedes the test period, and time-ordered residuals are not exchangeable. The proposition therefore motivates the construction. The evidence that it works here is empirical: in the exception-test repairs of Section~\ref{sec:frtb}, the conformal stage is the only reason any entrant passes Kupiec at 97.5\%. The guarantee also attaches to the shifted body quantile alone. The finished forecast adds the same scalar to the envelope of \eqref{eq:hybrid}, whose tail nodes take the minimum of body and EVT branches before rearrangement, so the proposition covers the shifted body while the reported curve is the shifted envelope. The exact finite-sample coverage identity need not survive that composition, so the coverage claim at the reported levels rests on these exception tests rather than on Proposition~\ref{prop:conformal}. Conformal constructions with
validity under dependence exist \citep{chernozhukov2018conformal, barber2023, gibbs2021aci}, but we
prefer the simple split form plus an empirical audit.
```

**After.**

```
The proposition's scope is narrower than its use. In practice the calibration split necessarily precedes the test period, and time-ordered residuals are not exchangeable, so the proposition only motivates the construction. The static shift fails here for that reason, since the calibration block and the 2020--2024 test era miss on opposite sides of nominal and a shift fitted on one over-covers the other (Section~\ref{sec:frtb}). Conformal constructions with
validity under dependence exist \citep{chernozhukov2018conformal, barber2023, gibbs2021aci}, and the adaptive update above is that of \citet{gibbs2021aci}, which gives up the finite-sample statement for long-run coverage under arbitrary distribution shift. Whether it is switched on was decided by the exception tests and the joint score under a rule fixed before the run (Section~\ref{sec:frtb}).
```

### M07 implementation gap: no forecast reads the calibration split (main text)

**Source and reason.** With the adaptive overlay no reported forecast reads the calibration split; the static and strict-split adaptive figures (34%/48%, -0.51 to -0.20, 57%, DM -3.2 to -1.5) described the withdrawn construction. CHECK BEFORE PUSH: DM 5.1/5.4 and -0.6/-2.5 come from fz_strict_calibration_results.json, which has no es_convention key (20-node engine ES). See precondition 1.

**Before.**

```
match. Only the overlay's shift uses the split. Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.1 and 5.4). The strict estimator's gap to the full-window benchmark ($-0.6$, $-2.5$) matches that benchmark's own margin over its short-window twin (2.8, 3.8), so the gap is the cost of a quarter less estimation data and shows no leakage. The static shift widens (per-name pass 34\% against
48\%), consistent with its disclosed conservatism. Under the same strict
split the adaptive conformal update of \citet{gibbs2021aci} walks the shift from
$-0.51$ to $-0.20$, lifts the per-name pass rate to 57\%, and shrinks the
joint-score gap to GARCH-$t$ from DM $-3.2$ to $-1.5$ (not significant at
5\%). The estimator's annual-refit
schedule sits between the static and adaptive poles, and the exception
tests decide between them.
```

**After.**

```
match. The adaptive overlay reads only breaches already observed in the test period, so no reported forecast reads the calibration split. Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.1 and 5.4). The strict estimator's gap to the full-window benchmark ($-0.6$, $-2.5$) matches that benchmark's own margin over its short-window twin (2.8, 3.8), so the gap is the cost of a quarter less estimation data and shows no leakage.
```

### M08 overlay scalar per level (main text)

**Source and reason.** The adaptive overlay shifts both levels.

**Before.**

```
The overlay applies one scalar $c_{0.025}$ to the VaR node and to the ES integral at that level,
```

**After.**

```
The overlay applies one scalar per level to the VaR node and to the ES integral at that level,
```

### M09 ES convention stated (main text)

**Source and reason.** es_integral.converged_es (M=2000, 60 fitted body levels, GPD exact at every node, closed-form sub-floor). Ratio converged/20-node 1.01645 (1%), 1.01661 (2.5%), every one of 200 names > 1 (harvest 4). The 'agree to the third decimal' claim was measured with the 20-node rule (harvest 7.10).

**Before.**

```
ES$_\alpha$ is computed as the numerical integral of \eqref{eq:hybrid} rather than by the GPD closed form of \citet{mcneilfrey2000}; the Online Appendix reports that the two agree to the third decimal on FZ0 and leave each comparison in place.
```

**After.**

```
ES$_\alpha$ is the integral of \eqref{eq:hybrid} over $(0,\alpha]$, computed to convergence: the body is interpolated across sixty fitted levels on $[\alpha/40,\alpha]$, the GPD branch is evaluated exactly at each of 2{,}000 nodes, and below $\alpha/40$, where the GPD quantile diverges, its integral is taken in closed form. A 20-node midpoint rule, used in earlier versions, understates the engine's $|\ES|$ by about 1.6\% for every one of the 200 names and distorts the ES backtests (Online Appendix). The Online Appendix also reports the FZ0 obtained with the GPD closed form of \citet{mcneilfrey2000} in place of the integral.
```

### M10 Algorithm 1 (iv) (main text)

**Source and reason.** Algorithm 1 matches the new Stage 4.

**Before.**

```
(iv)~Learn the conformal shifts $c_\tau$ on a calibration split
disjoint from the training data (overlay only; $c_\tau\equiv0$ in the accuracy layer).
```

**After.**

```
(iv)~Update the shift $c_{\tau,t+1}=c_{\tau,t}-\gamma\,(b_{\tau,t}-\tau)$ daily from the panel breach frequency, starting at zero
(overlay only; $c_\tau\equiv0$ in the accuracy layer).
```

### M11 ablation on point-in-time characteristics (main text)

**Source and reason.** amort_pit_results.json point_in_time ablation: no_realized_vol +0.54%, no_chars +0.45%, chars_only +2.18% vs ALL. The published 1.0%/0.5% used full-sample annvol, beta and logmcap (harvest 1).

**Before.**

```
Feature ablation at full scale (1.32M rows, 720 names, 288 held out) shows the transfer edge is carried overwhelmingly by each name's own recent dynamics, realized vol above all. Removing it costs $+1.0\%$, while removing all cross-sectional characteristics costs only $+0.5\%$. Characteristics matter at cold-start.
```

**After.**

```
Feature ablation at full scale (1.32M rows, 720 names, 288 held out), with the characteristics built point-in-time from data through the previous day (trailing 250-day volatility and market beta; market capitalization, which the returns panel cannot rebuild, is dropped), shows the transfer edge spread across the name's own recent dynamics and its characteristics. Removing short-window realized volatility costs $+0.54\%$ and removing all cross-sectional characteristics $+0.45\%$, while the characteristics-only form is $2.2\%$ worse than the full model. Characteristics matter at cold-start.
```

### M12 transfer win rate replaces the unsourced 59-79% (main text)

**Source and reason.** The 59-79% figure has no computational origin in either repository (harvest 1, full-mirror search). Replaced by date_win_rate_overall 0.9165 over n_test_dates 2767 and transfer_win_rate_all_names 0.9896, with the benchmark named.

**Before.**

```
On held-out names the transfer win rate over own-history benchmarks is
59--79\%.
```

**After.**

```
On held-out names the amortized model has lower cross-name mean pinball loss than the own-history benchmark on 91.7\% of the 2{,}767 test dates, and lower mean loss for 99.0\% of the names. The benchmark is each name's expanding-window mean and standard deviation with a Gaussian quantile, scored at seven levels from 0.05 to 0.95, so the transfer result concerns distributional shape at moderate quantiles; that loss contains no 1\% or 2.5\% level.
```

### M13 GBC relation: conformal stage (main text)

**Source and reason.** Same as M05.

**Before.**

```
The conformal stage adds the coverage statement of Proposition~\ref{prop:conformal}.
```

**After.**

```
The conformal stage adds a coverage correction driven by realized breaches.
```

### M14 protocol: no calibration-split shift (main text)

**Source and reason.** Same as M05: Stage 4 uses no calibration split.

**Before.**

```
the same pooled residuals below their empirical 2.5\% point, and the
conformal shifts on the pooled calibration split. No stage refits
inside the test window
```

**After.**

```
the same pooled residuals below their empirical 2.5\% point. The adaptive shift of Stage~4 starts at zero and is updated from test-period breaches as they are observed. No other stage is updated
inside the test window
```

### M15 synthetic truth in the robustness paragraph (main text)

**Source and reason.** synthetic_truth_results.json: dgps.garch frontier_DM_engine_over_own_parametric garch_t overall -5.2, top -1.85; bteg overall 10.17, top 7.71; pinball 0.198766 vs 0.198724 (0.02%). Mirror design: -4.29/-1.41 and 10.32/8.14.

**Before.**

```
the design-era threshold held fixed the top-decile edge is positive in
every test year from 2020 to 2024.
```

**After.**

```
the design-era threshold held fixed the top-decile edge is positive in
every test year from 2020 to 2024. On simulated panels whose true model is known, the frontier appears only against a misspecified model (Online Appendix Table~OA.16). When the data come from GARCH-$t$, the estimator on the GARCH-$t$ scale trails the true model slightly (pinball $0.02\%$, DM $-5.2$; $-1.9$ in the top decile), while on a misspecified Beta-$t$-EGARCH scale it beats that scale's own Student-$t$ tail (DM 10.2, and 7.7 in the top decile), and data generated from Beta-$t$-EGARCH give the mirror image.
```

### M16 the flexible tail on a robust scale (main text)

**Source and reason.** tail_vs_bteg frozen: FZ0 -3.85/-4.68; pinball_by_level 0.01 -5.42, 0.025 -3.39; eleven-level +0.87. Refit: FZ0 -3.09/-2.72; 0.01 -2.13, 0.025 -0.83; eleven-level -1.17. Central levels 0.25/0.5 within noise.

**Before.**

```
so on a robust scale the frontier is flat.
```

**After.**

```
so on a robust scale the frontier is flat. Against plain Beta-$t$-EGARCH, the estimator on its scale still has lower FZ0 at both levels under the frozen fits (DM 3.85 and 4.68) and under the annual refits (DM 3.09 and 2.72), and lower pinball loss at the 1\% level under both (DM 5.4 and 2.1) and at the 2.5\% level under the frozen fits (DM 3.4; 0.8 under refits). Averaged over the eleven pinball levels the two are within noise, because at the central levels they tie.
```

### M17a Table 4 ES, raw residual-hybrid (main text)

**Source and reason.** frtb_table_200_results.json ES975_pred_true / ES975_realized_ownVaR. resid_hybrid_ML -6.631/-6.112. Realized moves -6.12 -> -6.11 (see precondition 4).

**Before.**

```
& \textbf{0.3676} & $-6.63$ / $-6.12$ &
```

**After.**

```
& \textbf{0.3676} & $-6.63$ / $-6.11$ &
```

### M17b Table 4 ES, EVT tail (main text)

**Source and reason.** hybrid_EVT -7.326/-6.138 (converged; realized -6.15 -> -6.14, precondition 4).

**Before.**

```
& 0.3680 & $-7.21$ / $-6.15$ &
```

**After.**

```
& 0.3680 & $-7.33$ / $-6.14$ &
```

### M17c Table 4 ES, GJR-skew-t (main text)

**Source and reason.** gjr_skewt -6.872 (substitution integral; 200-node understated |ES| 0.10-0.18%).

**Before.**

```
& 0.3688 & $-6.85$ / $-6.09$ &
```

**After.**

```
& 0.3688 & $-6.87$ / $-6.09$ &
```

### M17d Table 4 ES, rolling FHS (main text)

**Source and reason.** fhs_roll500 -7.016 (exact window tail mean; 20-node rolling-quantile integral understated 4.6%).

**Before.**

```
& 0.3694 & $-6.71$ / $-6.21$ &
```

**After.**

```
& 0.3694 & $-7.02$ / $-6.21$ &
```

### M17e Table 4 ES, HS (main text)

**Source and reason.** hist_sim -7.714 (exact window tail mean; 4.1% understated).

**Before.**

```
& 0.3752 & $-7.41$ / $-7.08$ &
```

**After.**

```
& 0.3752 & $-7.71$ / $-7.08$ &
```

### M18 Table 4 note: ES convention (main text)

**Source and reason.** frtb_table_200 es_convention. The old note's two claims (exact tail integral; common node set) were false (harvest 5d). The three-node-shortcut sentence was measured against the 20-node integral and is dropped.

**Before.**

```
Predicted ES is the numerical tail integral
$\alpha^{-1}\!\int_0^\alpha Q(v)\,dv$: closed forms for the $t$ and
normal entrants, 200-node integration of the Hansen skew-$t$ inverse
CDF, and a matched 20-node midpoint rule for the empirical quantile
models and for the residual-hybrid, whose VaR and ES are both read from
one monotonized min-envelope curve $Q^{z}_t$ of \eqref{eq:hybrid}. A three-node tail-quantile average, a common shortcut,
understates predicted ES by 2--6\% relative to this integral.
```

**After.**

```
Predicted ES is the tail integral
$\alpha^{-1}\!\int_0^\alpha Q(v)\,dv$ of each model's quantile function: closed forms for the $t$ and
normal entrants, an exact substitution integral of the Hansen skew-$t$ inverse
CDF, the exact empirical tail mean for historical simulation and the FHS variants, and the converged integral of
Section~\ref{sec:hybrid} for the EVT-tailed residual-hybrid, whose VaR and ES are both read from
one monotonized min-envelope curve $Q^{z}_t$ of \eqref{eq:hybrid}. The raw residual-hybrid has no quantile below its lowest
fitted level, so its ES extends the body flat below $\alpha/40$ and is a lower bound on its severity.
```

### M19 Table 4 note: overstatement range (main text)

**Source and reason.** Overstatement: raw 8.5%, EVT 19.3, garch_t 18.0, gjr 12.9, fhs_pername 13.8, fhs 21.8, roll 13.0, hs 8.9.

**Before.**

```
each fat-tailed entrant exceeds its own-tail comparator by 8\% (raw
residual-hybrid, rolling FHS) to 22\% (pooled FHS).
```

**After.**

```
each fat-tailed entrant exceeds its own-tail comparator by 8\% (raw
residual-hybrid, a lower bound) to 22\% (pooled FHS).
```

### M20 Table 4 note: corrected exception battery (main text)

**Source and reason.** exception_battery_results.json (canon200, battery fixed for the mk63 dropna): engine_garch_t Kupiec per-name 0.835/0.70, Christoffersen 0.90/0.75, date-clustered t 0.62/2.00; body_garch_t t 4.27/4.74; param_garch_t t 1.56 at 2.5%; fhs_garch_t 3.35. Holdout: all three pass (t 0.25, -0.06, 0.98). Overlay: overlay_engine t 1.30/0.78 (canon200) and -0.27/-0.44 (holdout) at 2.5%. 'split-conformal recalibration repairs pooled Kupiec (GBM-recal p=0.24)' described the withdrawn static stage on the old 140-name run; the adaptive overlay still fails pooled Kupiec at 2.5% (p 0.000/0.002), so the sentence is removed.

**Before.**

```
FZ0 comparison passes Kupiec for $84\%$ and Christoffersen conditional coverage for $89\%$ (at 97.5\%, $73\%$ and $78\%$), with the date-clustered test passing at 99\% ($t=0.29$) and at 97.5\% ($t=1.60$) while its pooled body without the EVT tail is rejected ($t=4.1$ and $4.4$): the tail stages carry the regulatory
calibration, and the 2008-era re-run confirms it out of era.
At 97.5\% the split-conformal recalibration is the stage that repairs
pooled Kupiec (GBM-recal $p=0.24$).
```

**After.**

```
FZ0 comparison passes Kupiec for $84\%$ and Christoffersen conditional coverage for $90\%$ (at 97.5\%, $70\%$ and $75\%$). On the date-clustered test it passes at 99\% ($t=0.62$) and is rejected marginally at 97.5\% ($t=2.00$), where GARCH-$t$ passes ($t=1.56$) and pooled FHS is rejected more strongly ($t=3.35$); its pooled body without the EVT tail is rejected at both levels ($t=4.3$ and $4.7$), so the tail stage carries the regulatory
calibration. All three pass on the 2000--2013 holdout, and the adaptive overlay of Stage~4 passes on both panels.
```

### M21 stress-era overlay (main text)

**Source and reason.** overlay_engine_results_holdout.json: aci_0.02 breach 1.06%/2.43%, t 0.01/-0.27; aci_0.05 1.04%/2.38%, t -0.12/-0.44.

**Before.**

```
The conformal layer, calibrated on pre-crisis data, tilts conservative on the 2000--2013 holdout (breach $2.21\%$ against the $2.5\%$ target, $t=-1.09$, not significant). The safety margin widens when the calibration era is calmer than the test era, and the error runs in the direction regulators prefer.
```

**After.**

```
The adaptive overlay of Stage~4 stays near nominal through the crisis window, breaching at $2.43\%$ and $2.38\%$ against the $2.5\%$ target and at $1.06\%$ and $1.04\%$ against $1\%$ for the two update rates (date-clustered $|t|\le0.44$).
```

### M22a Figure 2 bars, 1% (main text)

**Source and reason.** bench_all_results.json fz0 mean_diff vs engine (canonical engine 2.14653/1.84863): garch_t .00919, evt_pool .00861, taylor .01819, fhs_name .02588, gas_pzc .08585.

**Before.**

```
{(GARCH-$t$,0.00848)(GARCH-EVT,0.00784)(Taylor,0.01748)(FHS,0.02517)(GAS,0.08514)};
```

**After.**

```
{(GARCH-$t$,0.00919)(GARCH-EVT,0.00861)(Taylor,0.01819)(FHS,0.02588)(GAS,0.08585)};
```

### M22b Figure 2 bars, 2.5% (main text)

**Source and reason.** Same at 2.5%: .00557, .00963, .00036, .01382, .05042.

**Before.**

```
{(GARCH-$t$,0.00545)(GARCH-EVT,0.00888)(Taylor,0.00024)(FHS,0.01370)(GAS,0.05030)};
```

**After.**

```
{(GARCH-$t$,0.00557)(GARCH-EVT,0.00963)(Taylor,0.00036)(FHS,0.01382)(GAS,0.05042)};
```

### M22c Figure 2 caption (main text)

**Source and reason.** bench_all DM_t: garch_t 4.94/5.50, evt_pool 4.51/6.33, fhs_name 4.91/5.35, gas_pzc 9.91/8.78, taylor 3.76/0.13. One canonical engine forecast (harvest 7.1); GAS fit of record is bench_all.

**Before.**

```
lowest FZ0 at $1\%$ and is never significantly beaten at $2.5\%$, beating GARCH-$t$ (DM $4.8$ at
$1\%$, $5.1$ at $2.5\%$), pooled-tail GARCH-EVT (DM $4.8$ and $6.8$; its bars come from the same-rows run of the Online Appendix, whose accuracy-layer baseline is within $0.0006$ of the one plotted) and FHS (DM $4.9$ and $5.3$), together with the two
FZ-estimated dynamic-ES benchmarks fitted per name by multi-start FZ0
minimization, the generalized autoregressive score (GAS) model of \citet{patton2019} (DM $9.5$ and $8.6$) and the
ES-CAViaR of \citet{taylor2019} (DM $3.6$ at $1\%$, a tie at $2.5\%$). SAV-CAViaR, scored on its own common sample (Table~\ref{tab:noncore}), ties the accuracy layer and is omitted here; the conformal-overlay variant, which concedes FZ0 at $2.5\%$, is discussed in the text.}
```

**After.**

```
lowest FZ0 at $1\%$ and is never significantly beaten at $2.5\%$, beating GARCH-$t$ (DM $4.9$ at
$1\%$, $5.5$ at $2.5\%$), pooled-tail GARCH-EVT (DM $4.5$ and $6.3$) and per-name FHS (DM $4.9$ and $5.4$), together with the two
FZ-estimated dynamic-ES benchmarks fitted per name by multi-start FZ0
minimization, the generalized autoregressive score (GAS) model of \citet{patton2019} (DM $9.9$ and $8.8$) and the
ES-CAViaR of \citet{taylor2019} (DM $3.8$ at $1\%$, a tie at $2.5\%$). Every bar and DM comes from one run on the same rows against the same accuracy-layer forecast. SAV-CAViaR, scored on its own common sample (Table~\ref{tab:noncore}), ties the accuracy layer and is omitted here; the adaptive overlay, within noise of the accuracy layer at both levels, is discussed in the text.}
```

### M23a 5.1 GARCH-t and FHS DMs (main text)

**Source and reason.** bench_all: garch_t 4.94 (1%), 5.50 (2.5%); fhs_name 4.91.

**Before.**

```
At 1\% it beats GARCH-$t$ (DM 4.8) and FHS (DM 4.9), the two standard benchmarks the accuracy claim rests on. At 2.5\% it again wins over GARCH-$t$ (DM 5.1).
```

**After.**

```
At 1\% it beats GARCH-$t$ (DM 4.9) and FHS (DM 4.9), the two standard benchmarks the accuracy claim rests on. At 2.5\% it again wins over GARCH-$t$ (DM 5.5).
```

### M23b 5.1 GARCH-EVT DMs (main text)

**Source and reason.** garch_evt_results.json: evt_pool 4.51/6.33, evt_name 5.14/5.36.

**Before.**

```
with a pooled tail by DM 4.8 at 1\% and 6.8 at 2.5\%, and per name by DM 5.2 and 5.4,
```

**After.**

```
with a pooled tail by DM 4.5 at 1\% and 6.3 at 2.5\%, and per name by DM 5.1 and 5.4,
```

### M23c 5.1 DQ: static-overlay clause removed (main text)

**Source and reason.** The 84% DQ rate is the static overlay's (bench_all dq engine_overlay 0.845); the adaptive overlay's DQ was not computed.

**Before.**

```
83\% and 62\% for the pooled GARCH-EVT; the overlay lifts the 2.5\% rate to 84\%.
```

**After.**

```
83\% and 62\% for the pooled GARCH-EVT.
```

### M23d 5.1 OA.11 sentence: tie group at 2.5% (main text)

**Source and reason.** bench_all fz0: gjr_skewt DM 1.24/-0.02, taylor 0.13, body -1.09/-1.54; tie group body 1.84323, gjr 1.84859, engine 1.84863, taylor 1.84899. MCS unchanged (1%: engine, engine_overlay, body, gjr; 2.5%: engine, body, gjr, taylor). engine_overlay at 1% equals the engine.

**Before.**

```
the accuracy layer has the lowest mean FZ0 of the benchmark models at 1\% and is within noise of GJR-GARCH-skew-$t$ at both levels (DM 1.1 and $-0.3$) and of the Taylor model at 2.5\% (DM $-0.1$); its own pooled body, without the EVT branch, scores lower still (DM $-1.3$ at 1\%, within noise, and $-1.9$ at 2.5\%, marginal under the convention of Section~\ref{sec:conventions}) while breaching at 1.3\% and 3.3\%, the coverage cost the EVT branch removes; the 90\% Model Confidence Set on the per-date FZ0 series contains the accuracy layer, the conformal overlay, the pooled body and GJR-GARCH-skew-$t$ at 1\%
```

**After.**

```
the accuracy layer has the lowest mean FZ0 of the benchmark models at 1\%, a level statement since GJR-GARCH-skew-$t$ is within noise (DM 1.2). At 2.5\% the accuracy layer, GJR-GARCH-skew-$t$ and the Taylor model form a tie group (FZ0 1.8486, 1.8486 and 1.8490; DM $-0.02$ and 0.1), and nothing significantly beats the accuracy layer; its own pooled body, without the EVT branch, scores lower still (DM $-1.1$ at 1\% and $-1.5$ at 2.5\%, both within noise) while breaching at 1.3\% and 3.3\%, the coverage cost the EVT branch removes; the 90\% Model Confidence Set on the per-date FZ0 series contains the accuracy layer, the pooled body and GJR-GARCH-skew-$t$ at 1\% (with the static-shift row, identical to the accuracy layer at that level)
```

### M23e 5.1 GAS and Taylor DMs (main text)

**Source and reason.** bench_all gas_pzc 9.91/8.78, taylor 3.76.

**Before.**

```
The accuracy layer beats the GAS at both levels (DM 9.5 at 1\%, 8.6 at 2.5\%), and it beats the Taylor model at 1\% (DM 3.6) while tying it at 2.5\%,
```

**After.**

```
The accuracy layer beats the GAS at both levels (DM 9.9 at 1\%, 8.8 at 2.5\%), and it beats the Taylor model at 1\% (DM 3.8) while tying it at 2.5\%,
```

### M23f 5.1 Beta-t-EGARCH (A3) converged (main text)

**Source and reason.** robust_engine_results.json fz0 0.025 bteg vs_engine DM -2.48 (1%: -0.74).

**Before.**

```
has lower FZ0 at both levels under the frozen fits, significantly at 2.5\% (DM 2.53).
```

**After.**

```
has lower FZ0 at both levels under the frozen fits, significantly at 2.5\% (DM 2.48).
```

### M23g 5.1 Beta-t-EGARCH top decile under refits (main text)

**Source and reason.** walkforward_bteg fz0 param_bteg vs_engine_garch_t top_mk63_decile -3.65/-5.12; overall 1.23/0.90 unchanged at print.

**Before.**

```
while Beta-$t$-EGARCH stays significantly ahead in the top decile of the score (DM 3.7 and 5.2).
```

**After.**

```
while Beta-$t$-EGARCH stays significantly ahead in the top decile of the score (DM 3.7 and 5.1).
```

### M24a refit ranking: GARCH-t and pooled FHS named (main text)

**Source and reason.** walkforward_bteg fz0 param_garch_t vs_engine_garch_t overall 3.46/3.10; uncond_garch_t 3.16/3.40. The published FHS halves (2.7, 2.6) match no field in walkforward_bteg or walkforward results (harvest 7.6); uncond_garch_t is the pooled-FHS construction and is named in the sentence.

**Before.**

```
the accuracy layer keeps the lowest FZ0 at both levels, beating GARCH-$t$ and FHS by DM 3.6 and 2.7 at 1\% and 3.1 and 2.6 at 2.5\% (one-sided $p<0.005$).
```

**After.**

```
the accuracy layer keeps the lowest FZ0 at both levels, beating GARCH-$t$ by DM 3.5 at 1\% and 3.1 at 2.5\%, and pooled FHS, the unconditional residual quantile on the same refitted scale, by DM 3.2 and 3.4 (one-sided $p<0.005$).
```

### M24b refit: Beta-t-EGARCH engine (A7) converged (main text)

**Source and reason.** robust_engine (frozen) engine vs_engine_bteg 3.16/4.37; holdout 3.14/3.63. walkforward_bteg DM_engine_garch_t_vs_engine_bteg overall -0.31/-0.02, top 5.46/7.37.

**Before.**

```
the engine has lower FZ0 under the frozen fits (DM 3.31 and 4.55, and 3.09 and 3.48 on the 2000--2013 holdout). Under annual refits the two engines are within noise overall (DM $-0.31$ and $-0.01$), and the Beta-$t$-EGARCH engine is ahead in the top decile of the score (DM 5.37 and 7.25),
```

**After.**

```
the engine has lower FZ0 under the frozen fits (DM 3.16 and 4.37, and 3.14 and 3.63 on the 2000--2013 holdout). Under annual refits the two engines are within noise overall (DM $-0.31$ and $-0.02$), and the Beta-$t$-EGARCH engine is ahead in the top decile of the score (DM 5.46 and 7.37),
```

### M25 summary and Stage 4 results (main text)

**Source and reason.** Battery figures as M20. 'DM 2.4 vs 0.2' (FZ0 concentration) matches no result file; bench_all garch_t vs engine at 1%: top-decile mean_diff .01076 DM 3.38, bulk .00901 DM 4.71. Stage 4 paragraphs rewritten from overlay_engine_results{,_holdout}.json (canon200 aci_0.02/0.05: breach 2.68/2.60%, DM vs engine -1.31/-1.02, t 1.30/0.78, 1% DM -0.25/-0.29, vs garch_t -5.15/-4.09, top-decile vs engine 3.40/3.56; holdout breach 2.43/2.38 and 1.06/1.04, |DM| <= 0.93; static body-targeted breach 1.66%, t -9.09, DM 2.57; static engine-targeted 2.15%, t -3.07). The 1.66/1.63%, -1.8/-2.0, 48/47%, 71/72%, -7.1 figures belonged to the withdrawn construction.

**Before.**

```
and passes the aggregate date-clustered exception test at 99\% and marginally at 97.5\%. Per-name diagnostics show residual cross-sectional calibration heterogeneity (84\% and 89\% pass rates at 99\%). The one failed test is the pooled 221.6k-observation Kupiec at 97.5\%, which at this sample size rejects very small deviations. The frontier concentrates the joint-score
edge as it does the pinball edge: the top misspecification decile
carries DM 2.4 at the 1\% level against DM 0.2 in the panel bulk.
Adding the conformal shift at 97.5\%, whose calibration split includes
the 2020 crash, widens the band out of era (breach $1.66\%$ against the $2.5\%$ target) and gives back the FZ0 advantage there. The static-overlay variant loses to GARCH-$t$ on the 2.5\% joint score (DM $-1.8$), and the concession is largest in the top misspecification decile, where the static shift's joint-score deficit reaches DM $-7.1$.

The coverage
property costs sharpness under
a strictly consistent score, and its error direction is conservative
(wider bands, fewer breaches than nominal). Where the conservative
coverage margin is valued, the shift is kept; where FZ0 efficiency is
the criterion, the EVT tail is run alone, and it passes the exception
tests on its own (Online Appendix Figure~OA.2).

Most of the concession reflects a frozen margin. The adaptive form of the shift \citep{gibbs2021aci}, updated daily from the panel breach frequency, shrinks the margin out of era (from $-0.35$ to $-0.10$, breach $1.63\%\to2.06\%$) and erases the overall concession. The adaptive variant ties GARCH-$t$ on 2.5\% FZ0 (DM $-0.3$). The top-decile cost survives the adaptive shift too, at $-4.8$ and $-6.0$ for the two update rates against the static shift's $-7.1$, the price of a margin where the bands are widest. The static shift also worsens per-name coverage (Kupiec$_{97.5}$ pass rate 48\%, against 71\% unshifted and 69\% adaptive), because over-coverage fails the test from the other side. Each ``lowest FZ0'' statement refers to the accuracy layer.
```

**After.**

```
and passes the aggregate date-clustered exception test at 99\%. At 97.5\% it is rejected marginally on 2020--2024 ($t=2.00$, against 1.56 for GARCH-$t$ and 3.35 for pooled FHS) and passes on the 2000--2013 holdout ($t=0.25$). Per-name diagnostics show residual cross-sectional calibration heterogeneity (84\% and 90\% pass rates at 99\%). The pooled 221.6k-observation Kupiec test also rejects at 97.5\%, and at this sample size it rejects very small deviations. The joint-score edge over GARCH-$t$ is less concentrated than the pinball edge: at 1\% the mean FZ0 gap is 0.011 in the top misspecification decile and 0.009 in the other nine (DM 3.4 and 4.7).

\paragraph{The adaptive overlay.} Stage~4 shifts the accuracy layer by a scalar updated each day from the panel breach frequency (Section~\ref{sec:hybrid}). Before the run we fixed the rule under which it would be switched on: on both the 2014--2024 panel and the 2000--2013 holdout, at both update rates, the overlay had to bring the 97.5\% breach rate within 0.3 points of nominal and keep its FZ0 within noise of the accuracy layer ($|\mathrm{DM}|<2$) at both levels. It meets both conditions. In era it moves the 97.5\% breach rate from 2.78\% to 2.68\% and 2.60\% for $\gamma=0.02$ and $0.05$, with FZ0 slightly below the accuracy layer's (DM $-1.3$ and $-1.0$), and the date-clustered test passes ($t=1.30$ and $0.78$); at 99\% it leaves the accuracy layer essentially unchanged (DM $-0.3$ for both rates). On the holdout its breach rates are 2.43\% and 2.38\% at 97.5\% and 1.06\% and 1.04\% at 99\%, with every $|\mathrm{DM}|$ against the accuracy layer below 1. It keeps the joint-score lead over GARCH-$t$ at 2.5\% in era (DM 5.2 and 4.1). In the top misspecification decile it concedes 2.5\% FZ0 to the accuracy layer (DM 3.4 and 3.6), the price of widening the band where it is already widest. A static shift, the construction of earlier versions, fails in both of its forms. Estimated on the body's calibration errors and applied to the envelope, it counts the tail correction twice and over-covers (breach 1.66\%, date-clustered $t=-9.1$, FZ0 DM 2.6 against the accuracy layer); estimated on the envelope's own errors, it still over-covers in era (2.15\%, $t=-3.1$), because the calibration block and the test era miss on opposite sides of nominal. Each ``lowest FZ0'' statement refers to the accuracy layer.
```

### M26 direct ES backtests rewritten (main text)

**Source and reason.** engine_esbt_results.json (canonical engine): MF -0.0578 p 0.383 (1%), +0.0186 p 0.581 (2.5%); FHS -0.1008 p 0.070, +0.0062 p 0.837; garch_t -0.2341 p 0.0002. Refit: walkforward_bteg engine_garch_t MF p 0.214/0.329. Holdout: exception_battery_results_holdout.json engineC +0.154 p 0.000, +0.0278 p 0.477; GARCH-t -0.1745 p 0.000, -0.0986 p 0.022; FHS +0.1241 p 0.001. MF size with 20-node ES 22.5% vs 6% exact (synthetic_truth quadrature_size_check).

**Before.**

```
The residual-hybrid has the $Z_2$ closest to zero at $1\%$ and ties GARCH-$t$ at $2.5\%$, and the residual-hybrid and GARCH-$t$ are the two models that pass the $1\%$ Kupiec test. On the exceedance residual, FHS, which targets the empirical tail directly, is closest to zero. On the exception and $Z_2$ diagnostics the residual-hybrid's tail is calibrated; the exceedance-residual test rejects it at $1\%$ ($p=0.02$, Online Appendix Table~OA.5). The comparative result is the FZ0 loss, where the edge over GARCH-$t$ and FHS is significant at both levels.
```

**After.**

```
The residual-hybrid has the $Z_2$ closest to zero at $1\%$ and ties GARCH-$t$ at $2.5\%$, and the residual-hybrid and GARCH-$t$ are the two models that pass the $1\%$ Kupiec test. On the exceedance residual the residual-hybrid is closest to zero at $1\%$ and FHS, which targets the empirical tail directly, at $2.5\%$. Neither test rejects the residual-hybrid's ES at either level (exceedance residual $p=0.38$ and $0.58$, Online Appendix Table~OA.5), nor under the annual refits ($p=0.21$ and $0.33$), while GARCH-$t$'s $1\%$ ES is rejected as not severe enough ($p<0.001$). On the 2000--2013 holdout the verdict depends on the direction of the error. At $1\%$ the test rejects every construction, GARCH-$t$'s ES as not severe enough (mean exceedance residual $-0.17$) and the residual-hybrid's and FHS's as too severe ($+0.15$ and $+0.12$); at $2.5\%$ the residual-hybrid passes ($p=0.48$) and GARCH-$t$ is rejected ($p=0.02$). These verdicts require an accurate ES: with the 20-node midpoint ES of earlier versions, the exceedance-residual test rejects a correctly specified model in 22.5\% of simulated samples at a nominal 5\% (Online Appendix Table~OA.16). The comparative result is the FZ0 loss, where the edge over GARCH-$t$ and FHS is significant at both levels.
```

### M27 ten-day: envelope tested, not recommended (main text)

**Source and reason.** tenday_envelope_results.json: 2008-09 envelope 1.61%/3.52%, sqrt_h 1.93/3.99, direct 2.15/4.33; in-era envelope cost 1.149%, DM_direct_vs envelope 3.91; rescaled in era DM -1.66 (nonoverlap -0.33), coverage .0097/.0244, holdout DM 2.33. Pre-committed P1 (within 0.5pt in 2008-09) and P2 (in-era cost < 0.5%) both fail, so the envelope recommendation is withdrawn.

**Before.**

```
calibrated through the crisis while the direct tail under-states it. In a
2008-type era the direct ten-day model should therefore be run with the
$\sqrt{h}$-scaled parametric number alongside it as a conservative
envelope. The ten-day evidence is therefore mixed; the benchmark
table is Online Appendix Table~OA.13.
```

**After.**

```
calibrated through the crisis while the direct tail under-states it. A
level-wise minimum of the two curves does not repair this. In 2008--2009
it breaches at 1.61\% and 3.52\% against the 1\% and 2.5\% targets, closer
to nominal than $\sqrt{h}$ scaling (1.93\% and 3.99\%) but short of it, and
in era it costs 1.1\% of pinball against the direct model (DM 3.9), so we
do not recommend it. Rescaling the direct curve by the fitted variance term
structure ties the direct model in era (DM $-1.7$; $-0.3$ on
non-overlapping dates) with breach rates nearer nominal (0.97\% and
2.44\%) and loses to it out of era (DM 2.3); we record it as exploratory.
The ten-day evidence is therefore mixed; the benchmark
table is Online Appendix Table~OA.13.
```

### M28a using the score: switches (main text)

**Source and reason.** The scale gate is a second setting in which the score drives a switch.

**Before.**

```
(Section~\ref{sec:gate}). The one setting in which it drives a switch is
the per-name standalone model, where nonparametric quantiles underperform
on calm days.
```

**After.**

```
(Section~\ref{sec:gate}). It drives a switch in two settings: the per-name
standalone model, where nonparametric quantiles underperform
on calm days, and the choice of Stage-1 scale, described next.
```

### M28b the score-gated scale (main text)

**Source and reason.** scale_gate_results.json: threshold 13.0953 (pre-2020 p90), fire rate 0.0997, n_test 251,600; gate vs engine_garch_t 6.34/8.12; vs engine_bteg 0.82/0.76 overall, -1.63/-1.68 top decile; all date-clustered PASS except param_garch_t at 2.5%. P1, P2, P3 pass (harvest 7.6); independent recomputation gate_indep.py agrees to 5e-6 (harvest 3).

**Before.**

```
component of the charge scales with the ES forecast.

```

**After.**

```
component of the charge scales with the ES forecast.

\paragraph{Choosing the scale with the score.} Under the annual refits the Beta-$t$-EGARCH engine and the GARCH-$t$ engine are within noise overall while the robust one leads in the top decile of the score (Section~\ref{sec:frtb}), which suggests letting the score choose the scale. A gate that runs the GARCH-$t$ engine when the previous day's $\mathrm{mk}_{63}$ is below its pre-2020 90th percentile, a threshold fixed before the first test year, and the Beta-$t$-EGARCH engine otherwise switches on 10.0\% of the 251.6k refit test asset-days. In a run whose predictions were written beforehand, its joint score is significantly better than always running the GARCH-$t$ engine (DM 6.3 at 1\%, 8.1 at 2.5\%) and statistically equal to always running the Beta-$t$-EGARCH engine (DM 0.8 at both levels), which is slightly ahead of it in the top decile (DM $-1.6$ and $-1.7$, within noise); the gate and both fixed engines pass the date-clustered exception test at both levels. The gate keeps the paper's Stage-1 filter on nine asset-days in ten and recovers the robust scale's gain where the score is high. Where tail accuracy is the only objective, running the robust scale everywhere does as well.

```

### M29 cold start: calibration and benchmark (main text)

**Source and reason.** As M02, plus coldstart_peers (303 test names first listed on or after 2018-01-01).

**Before.**

```
Section~\ref{sec:hybrid} attaches once a scale filter is estimable. The
amortized forecast beats own-history benchmarks at each listing age, and
the edge is largest when the name is young ($\sim$6--10\% of pinball in
the first trading month). The ablation
```

**After.**

```
Section~\ref{sec:hybrid} attaches once a scale filter is estimable.
Day-one quantiles exist but are not calibrated at the regulatory levels. In a separate run on
303 names listed from 2018, the best pooled peer forecast breaches its 1\%
VaR on 7.1\% of asset-days in the first five trading days. Once the name
has a few weeks of history the amortized forecast beats its own-history
benchmark, an expanding-window Gaussian scored at seven levels from 0.05
to 0.95, by about 9\% of pinball at 15 to 30 trading days and
6.5--7\% through the first year, and on 91.7\% of test dates overall. The ablation
```

### M30a conclusion: first-week caveat (main text)

**Source and reason.** As M02.

**Before.**

```
produces a forecast for a newly listed asset from its first day, and the residual-hybrid attaches once a scale filter is estimable (Section~\ref{sec:applications}).
```

**After.**

```
produces a forecast for a newly listed asset from its first day, though not yet a calibrated one in its first week, and the residual-hybrid attaches once a scale filter is estimable (Section~\ref{sec:applications}).
```

### M30b conclusion: ES backtests by era and direction (A8) (main text)

**Source and reason.** A8 rewritten by era and direction (harvest 6.3, 7.4); refit MF from walkforward_bteg; 97.5% marginal rejection from the corrected battery.

**Before.**

```
The McNeil--Frey exceedance-residual test rejects the 1\% ES on the full panel, on the 2000--2013 holdout and under annual refits, for every Stage-1 filter tried and for GARCH-$t$ itself; filtered historical simulation, which fits the empirical tail directly, is the one entrant that passes it on the full panel.
```

**After.**

```
The accuracy layer's 97.5\% coverage is rejected marginally on 2020--2024 (date-clustered $t=2.00$, against 1.56 for GARCH-$t$), which the adaptive overlay repairs at no cost in FZ0. The McNeil--Frey exceedance-residual test rejects none of the residual-hybrid's ES forecasts on the design panel or under annual refits, but on the 2000--2013 holdout it rejects the 1\% ES of every construction tried, the residual-hybrid's and FHS's as too severe and GARCH-$t$'s as not severe enough.
```

### M30c conclusion: what survives a robust scale (main text)

**Source and reason.** As M16.

**Before.**

```
and what survives is real but modest.
```

**After.**

```
and what survives is real but modest; on a robust scale it sits in the tail, where the flexible tail still lowers the joint $(\VaR,\ES)$ loss at both regulatory levels under frozen fits and annual refits.
```

### M30d conclusion: the score chooses the scale (main text)

**Source and reason.** As M28b.

**Before.**

```
rather than a day-to-day switching rule, since the day-ahead oracle gap is largely unforecastable (Section~\ref{sec:gate}).
```

**After.**

```
rather than a day-to-day switch between parametric and flexible shapes, since the day-ahead oracle gap is largely unforecastable (Section~\ref{sec:gate}). It can still choose the Stage-1 scale: a gate that moves to Beta-$t$-EGARCH in the score's top decile is significantly better than GARCH-$t$ at both levels and statistically equal to running the robust scale everywhere (Section~\ref{sec:applications}).
```

### O01 deployed pipeline: adaptive shift (online appendix)

**Source and reason.** Stage 4 adaptive.

**Before.**

```
and, where the desk elects the coverage overlay, the conformal shift
$c_\tau$ (static or adaptive); (iv)~report
```

**After.**

```
and, where the desk elects the coverage overlay, the adaptive conformal shift
$c_\tau$, updated from the previous day's panel breach frequency; (iv)~report
```

### O02 deployed pipeline: annual refit list (online appendix)

**Source and reason.** No shifts are re-estimated annually.

**Before.**

```
re-estimate tail and shifts, all on trailing data.
```

**After.**

```
re-estimate the tail, all on trailing data.
```

### O03 summary table: holdout exception row (online appendix)

**Source and reason.** overlay_engine_results_holdout.json, 2.5% t -0.27 and -0.44.

**Before.**

```
at 97.5\%, 0.35 (EVT tail) and $-1.09$ (static overlay, conservative); HS, EWMA fail \\
```

**After.**

```
at 97.5\%, 0.35 (EVT tail) and $-0.27$ to $-0.44$ (adaptive overlay); HS, EWMA fail \\
```

### O04 summary table: FZ0 row (online appendix)

**Source and reason.** As M22c/M23d and M25.

**Before.**

```
DM 4.8--4.9 vs GARCH-$t$, FHS at 1\%, 5.1 at 2.5\%; 8.6--9.5 vs PZC GAS, 3.6-to-tie vs Taylor ES-CAViaR (multi-start); static shift concedes 2.5\% (DM $-2.0$), adaptive shift erases it (tie; top-decile cost remains) \\
```

**After.**

```
DM 4.9 vs GARCH-$t$ and FHS at 1\%, 5.5 vs GARCH-$t$ at 2.5\%; 8.8--9.9 vs PZC GAS, 3.8-to-tie vs Taylor ES-CAViaR (multi-start); tie group with GJR-skew-$t$ and Taylor at 2.5\%; adaptive overlay within noise of the accuracy layer (top-decile cost at 2.5\%) \\
```

### O05 summary table: cold-start row (online appendix)

**Source and reason.** As M02/M29.

**Before.**

```
Cold start / young listings & $+6$--$10\%$ & full-scale panel; no per-asset model applies \\
```

**After.**

```
Cold start / young listings & $+9\%$ at 15--30 days; $+6.5$--$7\%$ to one year & vs an expanding-window Gaussian at levels 0.05--0.95; point-in-time characteristics; first-week VaR not calibrated \\
```

### O06 decision table: Stage 4 (online appendix)

**Source and reason.** Stage 4 adaptive.

**Before.**

```
Residual-hybrid $+$ EVT $+$ conformal & Section~5 of the paper \\
```

**After.**

```
Residual-hybrid $+$ EVT $+$ adaptive overlay & Section~5 of the paper \\
```

### O07 ES backtests: Z2 value (online appendix)

**Source and reason.** engine_esbt engine_noconf AS_Z2 -0.0533 at 1%.

**Before.**

```
carries the statistic closest to zero at $1\%$ ($-0.06$, not rejected) and is
```

**After.**

```
carries the statistic closest to zero at $1\%$ ($-0.05$, not rejected) and is
```

### O08 ES backtests: McNeil-Frey reverses in the engine's favour (online appendix)

**Source and reason.** engine_esbt: engine MF -0.058 (p 0.383) is now closer to zero than FHS -0.101 (p 0.070); the published sentence reverses in the engine's favour (harvest 7.6).

**Before.**

```
McNeil--Frey exceedance residual FHS, which fits the empirical residual tail
directly, sits closest to zero and is the only entrant the test does not
reject at $1\%$ ($-0.10$, $p=0.07$); at $2.5\%$ FHS, GARCH-$t$ and the engine all
fall inside the non-rejection region.
```

**After.**

```
McNeil--Frey exceedance residual the engine sits closest to zero at $1\%$ and is
the clearest non-rejection ($-0.06$, $p=0.38$), ahead of FHS, which fits the empirical
residual tail directly ($-0.10$, $p=0.07$), while GARCH-$t$ and the Gaussian filter are
rejected; at $2.5\%$ FHS, GARCH-$t$ and the engine all
fall inside the non-rejection region.
```

### O09 ES backtests: FHS row bold (online appendix)

**Source and reason.** Bold moves to the engine at 1%.

**Before.**

```
& $-0.29$ & $\mathbf{-0.10\,(0.07)}$ \\
```

**After.**

```
& $-0.29$ & $-0.10\,(0.07)$ \\
```

### O10 ES backtests: engine row converged (online appendix)

**Source and reason.** engine_esbt engine_noconf: 2.5% breach 0.0278, Z2 -0.1028, MF +0.0186 (0.5807); 1% Kupiec 0.0713, Z2 -0.0533, MF -0.0578 (0.3832). Kupiec p at 1% is now the canonical engine's (breach 1.04%).

**Before.**

```
Engine (EVT-tailed) & $0.027$ & $0.00$ & $\mathbf{-0.10}$ & $-0.04\,(0.24)$ & $0.010$ & $\mathbf{0.42}$ & $\mathbf{-0.06}$ & $-0.15\,(0.02)$ \\
```

**After.**

```
Engine (EVT-tailed) & $0.028$ & $0.00$ & $\mathbf{-0.10}$ & $+0.02\,(0.58)$ & $0.010$ & $\mathbf{0.07}$ & $\mathbf{-0.05}$ & $\mathbf{-0.06\,(0.38)}$ \\
```

### O11 ES backtests: note (online appendix)

**Source and reason.** engine_esbt engine_conf (static shift) 2.5%: breach 0.0166, Z2 +0.3061, MF p 0.0017.

**Before.**

```
The deployed engine adds a $97.5\%$ conformal shift and is conservative out of
era (breach $1.66\%$, $Z_2=+0.29$ at $2.5\%$), overstating the tail in the
regulator-preferred direction. Source: \texttt{engine\_esbt\_results.json}.
```

**After.**

```
The engine's ES is the converged integral of Section~2.2 of the paper. The static conformal
shift of earlier versions, estimated on the body and applied to the engine, over-covers
(breach $1.66\%$, $Z_2=+0.31$, McNeil--Frey $p=0.002$ at $2.5\%$); the adaptive overlay that
replaces it is reported in Section~5.1 of the paper. Source: \texttt{engine\_esbt\_results.json}.
```

### O12 GARCH-EVT FZ0 table rows (online appendix)

**Source and reason.** garch_evt_results.json fz0 (reproduces bench_all exactly).

**Before.**

```
Engine (body/EVT minimum) & 2.1473 & 1.04\% & --- & 1.8494 & 2.78\% & --- \\
Engine $+$ conformal overlay (97.5\%) & 2.1473 & 1.04\% & 0.00 & 1.8656 & 1.66\% & 2.29 \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.33 & 1.8433 & 3.25\% & -1.87 \\
GARCH-EVT, pooled tail & 2.1551 & 1.06\% & 4.83 & 1.8583 & 2.89\% & 6.85 \\
GARCH-EVT, per name & 2.1693 & 1.13\% & 5.23 & 1.8596 & 2.93\% & 5.39 \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 5.00 & 1.8542 & 2.71\% & 4.55 \\
```

**After.**

```
Engine (body/EVT minimum) & 2.1465 & 1.04\% & --- & 1.8486 & 2.78\% & --- \\
Engine $+$ static shift (97.5\%) & 2.1465 & 1.04\% & 0.00 & 1.8664 & 1.66\% & 2.57 \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.09 & 1.8432 & 3.25\% & -1.54 \\
GARCH-EVT, pooled tail & 2.1551 & 1.06\% & 4.51 & 1.8583 & 2.89\% & 6.33 \\
GARCH-EVT, per name & 2.1693 & 1.13\% & 5.14 & 1.8596 & 2.93\% & 5.36 \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 4.94 & 1.8542 & 2.71\% & 5.50 \\
```

### O13 GARCH-EVT FZ0 table note (online appendix)

**Source and reason.** evt_pool vs_engine_top_mk63 DM 2.26 (1%), 3.42 (2.5%).

**Before.**

```
the overlay row adds the split-conformal shift at 97.5\% and coincides with the engine at 1\%. GARCH-EVT ES is the McNeil--Frey closed form; the engine and body ES are the 20-node integral of the monotonized quantile curve. Top-$\mathrm{mk}_{63}$-decile DM of pooled GARCH-EVT versus the engine: 2.63 at 1\% and 4.26 at 2.5\%.
```

**After.**

```
the static-shift row adds the split-conformal shift of earlier versions at 97.5\%, estimated on the body's calibration errors, and coincides with the engine at 1\%; Stage~4 now uses the adaptive overlay of Section~5.1 of the paper. GARCH-EVT ES is the McNeil--Frey closed form; the engine ES is the converged integral of the monotonized quantile curve (Section~2.2 of the paper), and the body ES a 20-node integral, since the body has no quantile below its lowest fitted level. Top-$\mathrm{mk}_{63}$-decile DM of pooled GARCH-EVT versus the engine: 2.26 at 1\% and 3.42 at 2.5\%.
```

### O14 DQ table: static-shift row named (online appendix)

**Source and reason.** Row is the static shift (bench_all dq engine_overlay).

**Before.**

```
Engine $+$ conformal overlay & \multicolumn{2}{c}{86\%} & \multicolumn{2}{c}{84\%} \\
```

**After.**

```
Engine $+$ static shift (97.5\%) & \multicolumn{2}{c}{86\%} & \multicolumn{2}{c}{84\%} \\
```

### O15 all-benchmark FZ0 table rows (online appendix)

**Source and reason.** bench_all_results.json fz0; unaffected rows keep their printed FZ0; MCS ticks unchanged. GJR 2.15047/1.84859 (substitution ES).

**Before.**

```
Engine (accuracy layer) & 2.1473 & 1.04\% & --- & $\checkmark$ & 1.8494 & 2.78\% & --- & $\checkmark$ \\
Engine $+$ conformal overlay & 2.1473 & 1.04\% & 0.00 & $\checkmark$ & 1.8656 & 1.66\% & 2.29 &  \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.33 & $\checkmark$ & 1.8433 & 3.25\% & -1.87 & $\checkmark$ \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 5.00 &  & 1.8542 & 2.71\% & 4.55 &  \\
GJR-GARCH-skew-$t$ & 2.1506 & 1.02\% & 1.05 & $\checkmark$ & 1.8487 & 2.64\% & -0.28 & $\checkmark$ \\
GARCH-FHS (pooled) & 2.1557 & 1.14\% & 4.15 &  & 1.8598 & 3.00\% & 6.37 &  \\
GARCH-FHS (per name) & 2.1724 & 1.24\% & 5.02 &  & 1.8624 & 3.08\% & 5.56 &  \\
GARCH-FHS (rolling 500d) & 2.1847 & 1.10\% & 6.62 &  & 1.8611 & 2.50\% & 3.71 &  \\
GARCH-EVT (pooled tail) & 2.1551 & 1.06\% & 4.83 &  & 1.8583 & 2.89\% & 6.85 &  \\
GARCH-EVT (per name) & 2.1693 & 1.13\% & 5.23 &  & 1.8596 & 2.93\% & 5.39 &  \\
GAS-FZ (PZC) & 2.2324 & 0.92\% & 9.50 &  & 1.8990 & 2.53\% & 8.26 &  \\
ES-CAViaR (Taylor) & 2.1647 & 0.98\% & 3.53 &  & 1.8490 & 2.62\% & -0.12 & $\checkmark$ \\
EWMA (RiskMetrics) & 2.3211 & 1.61\% & 6.96 &  & 1.9035 & 2.73\% & 6.30 &  \\
Historical simulation (500d) & 2.2512 & 0.87\% & 6.24 &  & 1.9182 & 2.06\% & 5.16 &  \\
```

**After.**

```
Engine (accuracy layer) & 2.1465 & 1.04\% & --- & $\checkmark$ & 1.8486 & 2.78\% & --- & $\checkmark$ \\
Engine $+$ static shift (97.5\%) & 2.1465 & 1.04\% & 0.00 & $\checkmark$ & 1.8664 & 1.66\% & 2.57 &  \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.09 & $\checkmark$ & 1.8432 & 3.25\% & -1.54 & $\checkmark$ \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 4.94 &  & 1.8542 & 2.71\% & 5.50 &  \\
GJR-GARCH-skew-$t$ & 2.1505 & 1.02\% & 1.24 & $\checkmark$ & 1.8486 & 2.64\% & -0.02 & $\checkmark$ \\
GARCH-FHS (pooled) & 2.1557 & 1.14\% & 3.93 &  & 1.8598 & 3.00\% & 5.89 &  \\
GARCH-FHS (per name) & 2.1724 & 1.24\% & 4.91 &  & 1.8624 & 3.08\% & 5.35 &  \\
GARCH-FHS (rolling 500d) & 2.1847 & 1.10\% & 6.61 &  & 1.8611 & 2.50\% & 4.10 &  \\
GARCH-EVT (pooled tail) & 2.1551 & 1.06\% & 4.51 &  & 1.8583 & 2.89\% & 6.33 &  \\
GARCH-EVT (per name) & 2.1693 & 1.13\% & 5.14 &  & 1.8596 & 2.93\% & 5.36 &  \\
GAS-FZ (PZC) & 2.2324 & 0.92\% & 9.91 &  & 1.8990 & 2.53\% & 8.78 &  \\
ES-CAViaR (Taylor) & 2.1647 & 0.98\% & 3.76 &  & 1.8490 & 2.62\% & 0.13 & $\checkmark$ \\
EWMA (RiskMetrics) & 2.3211 & 1.61\% & 6.89 &  & 1.9035 & 2.73\% & 6.25 &  \\
Historical simulation (500d) & 2.2512 & 0.87\% & 6.34 &  & 1.9182 & 2.06\% & 5.28 &  \\
```

### O16 all-benchmark FZ0 table note (online appendix)

**Source and reason.** bench_all es_convention; harvest 5b (tie group).

**Before.**

```
ES: closed forms for the $t$, skew-$t$ and Gaussian entrants; empirical tail means for HS and FHS; McNeil--Frey closed form for GARCH-EVT; the FZ0-estimated dynamic pair for GAS and Taylor; the 20-node integral of the monotonized curve for the engine and body.
```

**After.**

```
ES: closed forms for the $t$ and Gaussian entrants; an exact substitution integral for the skew-$t$; empirical tail means for HS and FHS; McNeil--Frey closed form for GARCH-EVT; the FZ0-estimated dynamic pair for GAS and Taylor (the GAS fit of record is this run's); the converged integral of the monotonized curve for the engine (Section~2.2 of the paper), and a 20-node integral for the body, which has no quantile below its lowest fitted level. The static-shift row is the split-conformal shift of earlier versions, estimated on the body's calibration errors; Stage~4 now uses the adaptive overlay of Section~5.1 of the paper. At 2.5\% the engine, the body, GJR-GARCH-skew-$t$ and the Taylor model form a tie group (FZ0 1.8432--1.8490, $|$DM$|\le1.54$), reported as a group rather than an order.
```

### O17 tests table: static-shift row named (online appendix)

**Source and reason.** Static-shift row named.

**Before.**

```
Engine $+$ conformal overlay & --- & --- & 86\% & 84\% & 0\% & 45\% & --- & --- \\
```

**After.**

```
Engine $+$ static shift (97.5\%) & --- & --- & 86\% & 84\% & 0\% & 45\% & --- & --- \\
```

### O18 ten-day table: envelope and rescaling rows (online appendix)

**Source and reason.** tenday_envelope_results.json, design and holdout blocks.

**Before.**

```
Direct model & --- & 1.13 / 2.80 & --- & 1.24 / 2.82 \\
```

**After.**

```
Envelope, $\min$(direct, $\sqrt{h}$) & $+1.14$ (3.9; 2.4) & 0.94 / 2.45 & $+0.15$ ($-0.32$; 0.04) & 1.01 / 2.38 \\
Variance-ratio rescaling & $-0.12$ ($-1.7$; $-0.3$) & 0.97 / 2.44 & $+0.07$ (2.3; 2.0) & 1.23 / 2.79 \\
Direct model & --- & 1.13 / 2.80 & --- & 1.24 / 2.82 \\
```

### O19 ten-day table: note (online appendix)

**Source and reason.** Same; job_tenday_envelope.py header P1/P2.

**Before.**

```
so the out-of-era accuracy comparison is uninformative; see \texttt{tenday\_diag\_results.json}.
```

**After.**

```
so the out-of-era accuracy comparison is uninformative; see \texttt{tenday\_diag\_results.json}. The envelope (the level-wise minimum of the direct and $\sqrt{h}$ curves, rearranged) and the variance-ratio rescaling (the direct curve rescaled by the square root of the iterated-to-$\sqrt{h}$ variance ratio) come from a rerun on the same rows, \texttt{tenday\_envelope\_results.json}, in which the $\sqrt{h}$ row reads $+1.39$ (DM 6.7) in era and $+0.51$ (DM 0.05) out of era. In 2008--2009 the envelope breaches at 1.61\% and 3.52\%, against 1.93\% and 3.99\% for $\sqrt{h}$ scaling and 2.15\% and 4.33\% for the direct model. Both constructions were run with predictions written beforehand; the envelope did not meet its two, and the rescaling is exploratory.
```

### O20 synthetic-truth table (online appendix)

**Source and reason.** synthetic_truth_results.json (both DGPs, FZ0 engine_minus_param_converged, quadrature_size_check).

**Before.**

```
Result file \texttt{arcd\_bench\_results.json}; predictions in the script header.
\end{tablenotes}
\end{threeparttable}
\end{table}

```

**After.**

```
Result file \texttt{arcd\_bench\_results.json}; predictions in the script header.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Synthetic truth (Section~4 of the paper): the estimator against the parametric model on its own scale, on panels simulated from a known model.}
\label{tab:synth}
\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
& & \multicolumn{2}{c}{Pinball DM} & \multicolumn{2}{c}{FZ0 difference ($\times10^{3}$)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Data generated by & Scale of both models & Overall & Top mk$_{63}$ decile & $1\%$ & $2.5\%$ \\
\midrule
GARCH-$t$ & GARCH-$t$ (true) & $-5.20$ & $-1.85$ & $+0.27$ & $+0.01$ \\
 & Beta-$t$-EGARCH & $10.17$ & $7.71$ & $-6.41$ & $-5.37$ \\
Beta-$t$-EGARCH & Beta-$t$-EGARCH (true) & $-4.29$ & $-1.41$ & $+0.29$ & $-0.18$ \\
 & GARCH-$t$ & $10.32$ & $8.14$ & $-1.62$ & $-1.61$ \\
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item Each panel is simulated per name from the named model, 200 names and 221,600 test rows. In each row the estimator (pooled shape learner and EVT tail on the stated scale) is compared with the parametric Student-$t$ model on the same scale. Pinball DM is the per-date Diebold--Mariano statistic of the estimator over that model on the eleven-level grid (positive favors the estimator); the FZ0 columns give mean FZ0 of the estimator minus that of the parametric model (negative favors the estimator), with the estimator's ES the converged integral and the parametric ES in closed form. Against the true model the estimator is slightly worse on pinball (by $0.02\%$ in both designs) and within $3\times10^{-4}$ on FZ0; against the misspecified scale's parametric tail it wins by $1.6$--$6.4\times10^{-3}$. The ES convention matters here: these gaps are smaller than the error of a 20-node midpoint ES, and under a correctly specified model the McNeil--Frey test with a 20-node ES rejects in $22.5\%$ of simulated samples at a nominal $5\%$ ($6\%$ with the exact ES). Result file \texttt{synthetic\_truth\_results.json}.
\end{tablenotes}
\end{threeparttable}
\end{table}

```

### O21 ES integral paragraph (online appendix)

**Source and reason.** Closed-form FZ0 1.851 is convention-free; integral 1.84863; GARCH-t DMs bench_all. The closed-form-vs-integral DM (5.3) was measured against the 20-node integral and is not restated. Ratios: es_converged 1.01645/1.01661; frtb200 hist_sim 4.1%, fhs_roll500 4.6%.

**Before.**

```
(i)~Computing ES$_\alpha$ as the numerical integral of the min-envelope curve rather than by the GPD closed form of \citet{mcneilfrey2000} leaves each comparison in place: at 2.5\% the accuracy layer's FZ0 is 1.849 under the integral against 1.851 under the closed form (the integral marginally better, DM 5.3), and the estimator-over-GARCH-$t$ DM is 4.8 and 5.1 at the 1\% and 2.5\% levels under either convention.
```

**After.**

```
(i)~ES$_\alpha$ is the converged integral of the min-envelope curve (Section~2.2 of the paper). Replacing it by the GPD closed form of \citet{mcneilfrey2000} gives the accuracy layer a 2.5\% FZ0 of 1.851, against 1.8486 under the integral, and under the integral the estimator-over-GARCH-$t$ DM is 4.9 and 5.5 at the 1\% and 2.5\% levels. The 20-node midpoint rule of earlier versions understated the engine's $|\ES|$ by $1.65\%$ at 1\% and $1.66\%$ at 2.5\% on average, for every one of the 200 names, and the rolling-window empirical rows of Table~4 of the paper by 4--5\%; those rows now use the exact tail mean.
```

### O22 FRTB figure: predicted ES (online appendix)

**Source and reason.** As M17b.

**Before.**

```
{(Hyb,-6.63) (Hyb+EVT,-7.21) (GARCH-t,-6.98) (FHS,-7.17)};
```

**After.**

```
{(Hyb,-6.63) (Hyb+EVT,-7.33) (GARCH-t,-6.98) (FHS,-7.17)};
```

### O23 FRTB figure: realized ES (online appendix)

**Source and reason.** As M17a/b (precondition 4).

**Before.**

```
{(Hyb,-6.12) (Hyb+EVT,-6.15) (GARCH-t,-5.92) (FHS,-5.89)};
```

**After.**

```
{(Hyb,-6.11) (Hyb+EVT,-6.14) (GARCH-t,-5.92) (FHS,-5.89)};
```

### O24 FRTB figure: caption (online appendix)

**Source and reason.** Convention named.

**Before.**

```
hybrid. Right: ES$_{97.5}$ predicted (exact tail integral) vs realized
```

**After.**

```
hybrid. Right: ES$_{97.5}$ predicted (the tail integral of Table~4 of the paper) vs realized
```

### O25 pipeline figure: breaches feed Stage 4 (online appendix)

**Source and reason.** Stage 4 reads breaches, not a calibration split.

**Before.**

```
{held-out\\calibration split};
```

**After.**

```
{breaches observed\\through day $t-1$};
```

### O26 pipeline figure: caption (online appendix)

**Source and reason.** Same.

**Before.**

```
or held out from training (the calibration split behind the conformal overlay).
```

**After.**

```
or observed as the forecasts are scored (the breaches behind the adaptive overlay).
```

### O27 trading paragraph (online appendix)

**Source and reason.** Prose: the two verdict clauses ('a direction and not a result', 'not a market-timing signal') replaced.

**Before.**

```
with the model supplying tail control rather than alpha on the mean; the
paper evaluates no prices, premia, costs or trading outcomes, so this is a
direction and not a result. The gating experiment provides no evidence that the score times
individual crash days, so its role there is continuous tail-risk
measurement, not a market-timing signal.
```

**After.**

```
with the model supplying tail control rather than alpha on the mean. The
paper evaluates no prices, premia, costs or trading outcomes, so the
application is untested here. The gating experiment provides no evidence that the score times
individual crash days, so its role there would be continuous tail-risk
measurement.
```

## Not changed, with the reason

- Abstract: every clause still holds (lowest among industry-standard benchmarks at 1% is a level statement, nothing significantly beats the engine at 2.5%, transfer to new listings survives point-in-time characteristics).
- Every pinball number, the frontier, the decile profiles and the MCS on pinball: ES-free and reproduced to four decimals (harvest 6.1).
- The 2008 stress-era replication's battery figures (94%/95%, t 0.17/0.35, HS 1.53%, EWMA 1.57%): a separate implementation (frtb_holdout) on the same panel; the battery reproduction gives 93%/95% and t 0.15/0.25 for the engine, the same verdicts.
- Proposition 1 and its proof: stated for the static split, which the text now presents as motivation.
- The ten-day ES-calibration sentence ('almost exactly calibrated through the crisis'): horizon_holdout_results.json, predicted vs realized ES −16.63/−16.57 for sqrt(h) and −14.45/−17.60 for the direct model.
- 'ratios 1.003-1.043' for the two hierarchical corrections: a separate cold-start job, not part of the look-ahead fix.

---

## Precondition findings on ai2 (appended before applying)

**1. M07 / L345 — no change. The premise was wrong, not the numbers.**
`job_fz_strict_calibration.py` contains **no numerical ES integral at all**: every ES is a closed form —
`t_es` for the GARCH arms (`:40`) and the McNeil–Frey GPD formula `evt_es` for the engine arms (`:118`),
used at `:123–125`. `grep` for `SUBN`, `mean(axis=1)`, `star`, `(j+0.5)` returns nothing. So 'DM 5.1 and
5.4', the gap '$-0.6$, $-2.5$', the 2.8/3.8 pair and L345's '$-3.2$ to $-1.5$'
(`garch_minus_strict_static` = −3.2, `garch_minus_strict_aci` = −1.49) **cannot move**. M07 is applied as
written. The absence of an `es_convention` key is not evidence of the 20-node convention — it means the file
predates the labelling.

**2. `tab:rgarch` and Section 5.3 — both move. Two edits added (O28, O29, M32).**
`job_scaleshape_canonical.py` did integrate, on a 20-node grid, in four of its five rows. Rerun converged:
the two GPD-tailed rows now use the closed-form tail integral, the unconditional row the exact empirical tail
mean, the hybrid row the converged integral, and the body-only shape row keeps a flat sub-floor extension
(stated in the new note).

| row | FZ0 2.5% | DM | FZ0 1% | DM |
|---|---|---|---|---|
| Daily core | 1.500 → 1.500 | +6.2 → +6.2 | 1.774 → 1.774 | **+4.5 → +4.6** |
| Realized scale + residual quantile | 1.445 → 1.445 | — | **1.716 → 1.715** | — |
| \quad + EVT tail | **1.444 → 1.445** | **−0.3 → −0.1** | 1.716 → 1.716 | **+0.5 → +1.3** |
| \quad + pooled shape learner | **1.445 → 1.442** | **+0.2 → −0.8** | **1.710 → 1.708** | **−1.3 → −1.4** |
| \quad + shape + EVT | 1.447 → 1.447 | +0.7 → +0.7 | **1.718 → 1.717** | **+0.6 → +0.5** |

The shape row's 2.5% DM changes sign (+0.2 → −0.8), both marginal. Section 5.3's 'DM $6.2$ and $4.5$' becomes
**'$6.2$ and $4.6$'** (M32).

**3. Sweep for 'joint' / 'Fissler' / 'shortfall' / 'concession' / DM-near-FZ.** Four files back printed FZ0
numbers and have no `es_convention` key; three of them cannot move:

| file | printed numbers | verdict |
|---|---|---|
| `fz_strict_calibration_results.json` | M07's DMs, L345 | closed-form ES throughout — cannot move |
| `fz_aci_results.json` | L1113 'adaptive variant ties GARCH-$t$ (DM $-0.3$)' = `aci_gamma_0.05/garch_minus_this_DM` = −0.29 | ES is `evt_es(A)`, GPD closed form (`:68,73`); zero numerical integrals — cannot move |
| `fz_score_results.json` | L1064 'DM 5.4 and 6.0' = `garch_norm` 5.35 / 5.97 | `code/paper/fz_score.py` has **no GBM at all**; all four models closed form or empirical — cannot move |
| `frontier_robust_results.json` | — | no FZ0 and no ES in the file; pinball only — unaffected |
| `coherent_results.json` | **L363 'splice level is immaterial … DM 4.6--4.8'** | **moves**: 1% DMs 4.82/4.78/4.60 → 4.70/4.67/4.50, so the range becomes **4.5–4.7** (M31). The 2.5% DMs move more (3.52/4.76/4.72 → 4.24/5.45/5.42); the printed range was the 1% one, now said so explicitly. OA's 'FZ0 1.849 under the integral' becomes 1.848 at $p_0=0.025$; O21 already states 1.8486 from the canonical engine, which is consistent. |

**4. Realized ES of the two GBM rows — and a reproducibility defect that changed four of R59's own numbers.**
The printed $-6.12$ / $-6.15$ are HEAD's own `frtb_table_200_results.json` ($-6.122$, $-6.146$); they did not
come from an earlier panel. Realized ES is VaR-only and **did not move for any of the seven closed-form or
empirical rows** — all seven are identical to the digit. It moved only for the two GBM rows because
`job_frtb200.py` left `random_state` unset on its three `HistGradientBoostingRegressor` calls, and sklearn
subsamples bin thresholds above 200k rows (training here is ~800k), so the body quantiles and hence those two
rows' VaR differ run to run.

The same omission was in `job_fz_fullpanel.py`, `job_pzc_taylor.py`, `job_scaleshape_canonical.py` and
`job_coherent.py` — in `job_bench_all.py`, `job_exception_battery.py` and `job_robust_engine.py` every fit was
already seeded. All are now seeded with `random_state=0` and all five jobs were rerun. HEAD's values were one
unseeded draw and cannot be reproduced by seeding; from R59 on they are reproducible. **Four numbers in
`apply_R59.py` were taken from the unseeded run and have been corrected here:** M17a predicted $-6.63 \to
-6.64$ and realized $-6.11 \to -6.12$ (i.e. unchanged from HEAD), M17b realized $-6.14 \to -6.16$, O22 Hyb
$-6.63 \to -6.64$, O23 Hyb $-6.11 \to -6.12$ and Hyb+EVT $-6.14 \to -6.16$.

Seeding also closed part of the four-engine spread of harvest §7.2: `fz_fullpanel` and `pzc_taylor` now agree
with each other exactly (20-node ES $-3.77489$ / $-2.81378$ both), though both still differ from `bench_all`'s
$-3.78359$ because their training pools and conformal splits are derived independently. §7.1's decision to
score every FZ0 comparison against `bench_all`'s canonical engine is unaffected and is what M22/M23 use.

---

## R59 follow-ups (second pass, after 0018718)

Twelve edits in four files, every value read from the seeded result files committed in 0018718 and verified
against them before the edit was written. All are in the same two categories as the commit itself: numbers that
moved when a job was reseeded, and numbers that moved with the ES convention.

| # | where | before → after | source |
|---|---|---|---|
| F1 | Table 4 note, DM against the best model | GARCH-$t$ 6.3→**6.2**, GJR-skew-$t$ 5.6→5.6, FHS pooled 7.0→**6.9**, per-name 6.4→**6.3**, rolling 6.0→6.0, HS 8.5→8.5, EWMA 9.7→**9.6** | `frtb_table_200_results.json` `DM_vs_best`: 6.18, 5.62, 6.91, 6.30, 5.97, 8.52, 9.61 |
| F2 | §5.2, the EVT variant's pinball concession | DM 6.9 → **7.0** | `DM_vs_best/hybrid_EVT/DM_stat` = 6.99 |
| F3a | Table 4, EVT-tail row, Breach$_{99}$ | 0.88\% → **0.87\%** | `per_model/hybrid_EVT/breach99` = 0.0087 |
| F3b | Table 4, raw residual-hybrid row, Kupiec$_{99}$ $p$ | 0.00 → **0.01** | `per_model/resid_hybrid_ML/kupiec99_p` = 0.0053 |
| F4 | Table 4 note, the EVT variant's breach | 0.88\% → **0.87\%** | same as F3a |
| F5 | §4.1 splice-level range (M31) | adds "and 4.2--5.5 at $2.5\%$" | `coherent_results.json` 2.5\% `garch_minus_coherent_DM`: 4.24, 5.45, 5.42 |
| F6 | §4.1, the body branch as the minimum | 38\% → **about 40\%** | `coherent_results.json` `bind_frac_body_min` at $p_0=0.025$: 0.3952 (1\%), 0.4168 (2.5\%) |
| F7 | Figure OA.2, raw-hybrid bar | 0.86 → **0.84** | `passrate99_perasset/resid_hybrid_ML` = 0.84 |
| F8 | Figure OA.4 caption, DM range | 5.6--9.7 → **5.6--9.6** | as F1 (EWMA 9.61 is the maximum) |
| F9a | `tab:rgarch` prose | "is below $1.4$" → "**is at most** $1.4$" | `scaleshape_canonical_results.json`: the largest $\lvert DM\rvert$ is 1.36, so "below 1.4" was false by rounding |
| F9b | `tab:rgarch` prose, shape-row DM at 1\% | $-1.3$ → $\mathbf{-1.4}$ | `DM_vs_rg_uncond/rg_shape` at $\alpha=0.01$ = −1.36 |
| F10 | ES-integral paragraph (O21) | the integral-vs-closed-form sentence restated with both levels and both DMs | `coherent_results.json` `p0_0.025`: 2.5\% FZ0 1.84806 vs 1.85084, `shipped_minus_coherent_DM` 4.23, `garch_minus_coherent` 5.45 vs `garch_minus_shipped` 3.98; 1\% 2.1464 vs 2.14865, DM 2.53, 4.67 vs 4.31 |

**Where 38\% came from (F6).** Not another file — the same one, from its pre-R59 run. At `a1f997f`
`coherent_results.json` gave `bind_frac_body_min` 0.3849 at 1\% and 0.3876 at 2.5\% for $p_0=0.025$, which is
the printed 38\%. The seeded rerun gives 0.3952 and 0.4168. The cause is the **seeding, not the ES convention**:
the binding fraction counts the nodes at which the gradient-boosted body sits below the GPD branch, so it is a
VaR-side quantity that shifts when the body is refit, and `job_coherent.py` was one of the five jobs whose
`HistGradientBoostingRegressor` calls were unseeded. A threshold-crossing count amplifies small quantile
shifts, which is why it moves by one to three percentage points where the FZ0 levels move in the fourth
decimal. "About 40\%" is the honest form, since the figure is not stable to the third digit across refits.

**On the ES convention and F10.** The restated sentence is also more careful than the one it replaces: the two
conventions are not equivalent. On the same rows the closed form scores worse than the integral at both levels
(DM 4.2 at 2.5\%, 2.5 at 1\%), and it costs the estimator part of its margin over GARCH-$t$ (5.5 → 4.0 and
4.7 → 4.3). The old sentence's "leaves each comparison in place" was true of the orderings and not of the
margins; the new one gives both directions and both levels.

**Sequencing note.** These twelve edits are **not** folded into 0018718. That commit had already been pushed to
`origin/main` (`0018718` confirmed on the remote) when the instruction to amend arrived, so amending it would
rewrite published history. They are committed separately instead, leaving the choice of a follow-up commit or a
squash-and-force-push open.
