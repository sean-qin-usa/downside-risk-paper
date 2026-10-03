# apply_R59.py -- R59: the correcting commit on top of a1f997f. Converged ES everywhere the manuscript prints
# an FZ0, ES, Acerbi-Szekely or McNeil-Frey number; one canonical engine forecast for every FZ0 comparison
# (bench_all_results.json); the corrected exception battery; Stage 4 rebuilt as the adaptive overlay on the
# engine's own breaches under its pre-committed rule; the tail-on-robust-scale and score-gated-scale results;
# synthetic-truth tests; point-in-time amortization numbers with the own-history benchmark named; the ten-day
# envelope recommendation withdrawn. Sources: docs/R59_RESULTS_HARVEST.md sections 1-7 and the result files
# named in CHANGE_LEDGER_R59.md. Exact-unique replacements; every anchor is checked before anything is written.
# Run from the paper repo root:  python apply_R59.py        (add --dry to check without writing)
import sys, io

FILES = ["submission/paper_A_jfec.tex", "preprint/paper_A_ssrn.tex"]
OA = ["submission/paper_A_jfec_online_appendix.tex", "preprint/paper_A_ssrn_online_appendix.tex"]
EDITS, OAED = [], []
E = lambda n, b, a: EDITS.append((n, b, a))
O = lambda n, b, a: OAED.append((n, b, a))

# ---------------------------------------------------------------- introduction
E("M01 intro: tail on a robust scale",
r"""re-estimation on realized-volatility residuals leaves the score's edge within noise. This attribution""",
r"""re-estimation on realized-volatility residuals leaves the score's edge within noise. On a Beta-$t$-EGARCH scale, whose response to a shock is bounded by construction, the flexible tail still lowers the joint $(\VaR,\ES)$ loss against that model's own Student-$t$ tail at both regulatory levels, under frozen fits and under annual refits. This attribution""")

E("M02 intro: amortization edge and day-one calibration",
r"""Its gains over own-history benchmarks are largest in the first trading month, at roughly 6--10\% of pinball loss. In its characteristics-only form it forecasts VaR and ES for assets with no return history at all, the day-one regime in which no per-asset model exists.""",
r"""Its gains over an own-history benchmark are largest early in a name's life, about 9\% of pinball loss at 15 to 30 trading days and 6.5--7\% through the first year, measured at moderate quantiles. In its characteristics-only form it forecasts VaR and ES for assets with no return history at all, the day-one regime in which no per-asset model exists, though those first-week forecasts are not yet calibrated at the regulatory levels (Section~\ref{sec:applications}).""")

E("M03 intro: overlay named",
r"""with an extreme-value left tail and an optional split-conformal overlay.""",
r"""with an extreme-value left tail and an optional adaptive conformal overlay.""")

# ---------------------------------------------------------------- Section 2: Stage 4 and the ES integral
E("M04 design: conformal stage",
r"""conformal stage corrects the level of the quantile curve using its
measured miss rate on a held-out split, with the exchangeability-based
guarantee of Proposition~\ref{prop:conformal}.""",
r"""conformal stage corrects the level of the quantile curve using its
realized miss rate, updated daily, the online form of the split-conformal
construction whose exchangeability-based guarantee is Proposition~\ref{prop:conformal}.""")

E("M05 Stage 4 rewritten",
r"""\emph{Stage 4 (conformal finish).} On a held-out calibration split of size $n$, compute the signed errors $\varepsilon_i = \widehat{z}_i - \widehat{q}_\phi(\tau \mid s_i)$ against the Stage-2 body. Let $c_\tau$ be the $\lceil (n{+}1)\tau \rceil$-th order statistic of $\{\varepsilon_i\}$. The finished curve is shifted per level by this scalar, $\widetilde{Q}^{z}_t(\tau) = Q^{z}_t(\tau) + c_\tau$ with $Q^{z}_t$ the envelope of \eqref{eq:hybrid} \citep{vovk2005, lei2018}, the one-sided form of conformalized quantile regression \citep{romano2019cqr}. The shift is the last operation, after the minimum and the rearrangement. The \emph{accuracy layer} that carries the FZ0 comparisons of Section~\ref{sec:frtb} and Table~\ref{tab:frtb} sets $c_\tau\equiv0$; the \emph{conformal overlay} applies the shift at the 97.5\% level. This distribution-free layer repairs
the 97.5\% level that each model in the comparison set otherwise fails. The coverage statement it inherits is standard \citep{vovk2005, lei2018, romano2019cqr} and is restated here in the form the shift uses.""",
r"""\emph{Stage 4 (conformal finish).} The finished curve is shifted per level by a scalar, $\widetilde{Q}^{z}_t(\tau) = Q^{z}_t(\tau) + c_\tau$ with $Q^{z}_t$ the envelope of \eqref{eq:hybrid}, and the shift is the last operation, after the minimum and the rearrangement. In its static form $c_\tau$ is fixed once: on a held-out calibration split of size $n$, compute the signed errors $\varepsilon_i = \widehat{z}_i - Q^{z}_i(\tau)$ of the envelope and let $c_\tau$ be their $\lceil (n{+}1)\tau \rceil$-th order statistic \citep{vovk2005, lei2018}, the one-sided form of conformalized quantile regression \citep{romano2019cqr}. In the adaptive form used here the shift starts at zero and is updated each day from the panel breach frequency $b_{\tau,t}$ of the shifted forecast, $c_{\tau,t+1} = c_{\tau,t} - \gamma\,(b_{\tau,t}-\tau)$ \citep{gibbs2021aci}, so the shift in force on day $t+1$ uses only breaches observed through day $t$. The \emph{accuracy layer} that carries the FZ0 comparisons of Section~\ref{sec:frtb} and Table~\ref{tab:frtb} sets $c_\tau\equiv0$; the \emph{adaptive overlay} applies the update at both regulatory levels with $\gamma\in\{0.02,0.05\}$, and in era it moves the 97.5\% breach rate toward nominal at no cost in FZ0 (Section~\ref{sec:frtb}). The coverage statement of the static form is standard \citep{vovk2005, lei2018, romano2019cqr} and is restated here for a quantile model $\widehat{q}_\phi$ and its calibration errors.""")

E("M06 Proposition 1 scope",
r"""The proposition's scope is narrower than its use. In practice the calibration split necessarily precedes the test period, and time-ordered residuals are not exchangeable. The proposition therefore motivates the construction. The evidence that it works here is empirical: in the exception-test repairs of Section~\ref{sec:frtb}, the conformal stage is the only reason any entrant passes Kupiec at 97.5\%. The guarantee also attaches to the shifted body quantile alone. The finished forecast adds the same scalar to the envelope of \eqref{eq:hybrid}, whose tail nodes take the minimum of body and EVT branches before rearrangement, so the proposition covers the shifted body while the reported curve is the shifted envelope. The exact finite-sample coverage identity need not survive that composition, so the coverage claim at the reported levels rests on these exception tests rather than on Proposition~\ref{prop:conformal}. Conformal constructions with
validity under dependence exist \citep{chernozhukov2018conformal, barber2023, gibbs2021aci}, but we
prefer the simple split form plus an empirical audit.""",
r"""The proposition's scope is narrower than its use. In practice the calibration split necessarily precedes the test period, and time-ordered residuals are not exchangeable, so the proposition only motivates the construction. The static shift fails here for that reason, since the calibration block and the 2020--2024 test era miss on opposite sides of nominal and a shift fitted on one over-covers the other (Section~\ref{sec:frtb}). Conformal constructions with
validity under dependence exist \citep{chernozhukov2018conformal, barber2023, gibbs2021aci}, and the adaptive update above is that of \citet{gibbs2021aci}, which gives up the finite-sample statement for long-run coverage under arbitrary distribution shift. Whether it is switched on was decided by the exception tests and the joint score under a rule fixed before the run (Section~\ref{sec:frtb}).""")

E("M07 implementation gap: no forecast reads the calibration split",
r"""match. Only the overlay's shift uses the split. Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.1 and 5.4). The strict estimator's gap to the full-window benchmark ($-0.6$, $-2.5$) matches that benchmark's own margin over its short-window twin (2.8, 3.8), so the gap is the cost of a quarter less estimation data and shows no leakage. The static shift widens (per-name pass 34\% against
48\%), consistent with its disclosed conservatism. Under the same strict
split the adaptive conformal update of \citet{gibbs2021aci} walks the shift from
$-0.51$ to $-0.20$, lifts the per-name pass rate to 57\%, and shrinks the
joint-score gap to GARCH-$t$ from DM $-3.2$ to $-1.5$ (not significant at
5\%). The estimator's annual-refit
schedule sits between the static and adaptive poles, and the exception
tests decide between them.""",
r"""match. The adaptive overlay reads only breaches already observed in the test period, so no reported forecast reads the calibration split. Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.1 and 5.4). The strict estimator's gap to the full-window benchmark ($-0.6$, $-2.5$) matches that benchmark's own margin over its short-window twin (2.8, 3.8), so the gap is the cost of a quarter less estimation data and shows no leakage.""")

E("M08 overlay scalar per level",
r"""The overlay applies one scalar $c_{0.025}$ to the VaR node and to the ES integral at that level,""",
r"""The overlay applies one scalar per level to the VaR node and to the ES integral at that level,""")

E("M09 ES convention stated",
r"""ES$_\alpha$ is computed as the numerical integral of \eqref{eq:hybrid} rather than by the GPD closed form of \citet{mcneilfrey2000}; the Online Appendix reports that the two agree to the third decimal on FZ0 and leave each comparison in place.""",
r"""ES$_\alpha$ is the integral of \eqref{eq:hybrid} over $(0,\alpha]$, computed to convergence: the body is interpolated across sixty fitted levels on $[\alpha/40,\alpha]$, the GPD branch is evaluated exactly at each of 2{,}000 nodes, and below $\alpha/40$, where the GPD quantile diverges, its integral is taken in closed form. A 20-node midpoint rule, used in earlier versions, understates the engine's $|\ES|$ by about 1.6\% for every one of the 200 names and distorts the ES backtests (Online Appendix). The Online Appendix also reports the FZ0 obtained with the GPD closed form of \citet{mcneilfrey2000} in place of the integral.""")

E("M10 Algorithm 1 (iv)",
r"""(iv)~Learn the conformal shifts $c_\tau$ on a calibration split
disjoint from the training data (overlay only; $c_\tau\equiv0$ in the accuracy layer).""",
r"""(iv)~Update the shift $c_{\tau,t+1}=c_{\tau,t}-\gamma\,(b_{\tau,t}-\tau)$ daily from the panel breach frequency, starting at zero
(overlay only; $c_\tau\equiv0$ in the accuracy layer).""")

E("M11 ablation on point-in-time characteristics",
r"""Feature ablation at full scale (1.32M rows, 720 names, 288 held out) shows the transfer edge is carried overwhelmingly by each name's own recent dynamics, realized vol above all. Removing it costs $+1.0\%$, while removing all cross-sectional characteristics costs only $+0.5\%$. Characteristics matter at cold-start.""",
r"""Feature ablation at full scale (1.32M rows, 720 names, 288 held out), with the characteristics built point-in-time from data through the previous day (trailing 250-day volatility and market beta; market capitalization, which the returns panel cannot rebuild, is dropped), shows the transfer edge spread across the name's own recent dynamics and its characteristics. Removing short-window realized volatility costs $+0.54\%$ and removing all cross-sectional characteristics $+0.45\%$, while the characteristics-only form is $2.2\%$ worse than the full model. Characteristics matter at cold-start.""")

E("M12 transfer win rate replaces the unsourced 59-79%",
r"""On held-out names the transfer win rate over own-history benchmarks is
59--79\%.""",
r"""On held-out names the amortized model has lower cross-name mean pinball loss than the own-history benchmark on 91.7\% of the 2{,}767 test dates, and lower mean loss for 99.0\% of the names. The benchmark is each name's expanding-window mean and standard deviation with a Gaussian quantile, scored at seven levels from 0.05 to 0.95, so the transfer result concerns distributional shape at moderate quantiles; that loss contains no 1\% or 2.5\% level.""")

E("M13 GBC relation: conformal stage",
r"""The conformal stage adds the coverage statement of Proposition~\ref{prop:conformal}.""",
r"""The conformal stage adds a coverage correction driven by realized breaches.""")

E("M14 protocol: no calibration-split shift",
r"""the same pooled residuals below their empirical 2.5\% point, and the
conformal shifts on the pooled calibration split. No stage refits
inside the test window""",
r"""the same pooled residuals below their empirical 2.5\% point. The adaptive shift of Stage~4 starts at zero and is updated from test-period breaches as they are observed. No other stage is updated
inside the test window""")

# ---------------------------------------------------------------- Section 4
E("M15 synthetic truth in the robustness paragraph",
r"""the design-era threshold held fixed the top-decile edge is positive in
every test year from 2020 to 2024.""",
r"""the design-era threshold held fixed the top-decile edge is positive in
every test year from 2020 to 2024. On simulated panels whose true model is known, the frontier appears only against a misspecified model (Online Appendix Table~OA.16). When the data come from GARCH-$t$, the estimator on the GARCH-$t$ scale trails the true model slightly (pinball $0.02\%$, DM $-5.2$; $-1.9$ in the top decile), while on a misspecified Beta-$t$-EGARCH scale it beats that scale's own Student-$t$ tail (DM 10.2, and 7.7 in the top decile), and data generated from Beta-$t$-EGARCH give the mirror image.""")

E("M16 the flexible tail on a robust scale",
r"""so on a robust scale the frontier is flat.""",
r"""so on a robust scale the frontier is flat. Against plain Beta-$t$-EGARCH, the estimator on its scale still has lower FZ0 at both levels under the frozen fits (DM 3.85 and 4.68) and under the annual refits (DM 3.09 and 2.72), and lower pinball loss at the 1\% level under both (DM 5.4 and 2.1) and at the 2.5\% level under the frozen fits (DM 3.4; 0.8 under refits). Averaged over the eleven pinball levels the two are within noise, because at the central levels they tie.""")

# ---------------------------------------------------------------- Section 5.1: Table 4
E("M17a Table 4 ES, raw residual-hybrid", r"""& \textbf{0.3676} & $-6.63$ / $-6.12$ &""", r"""& \textbf{0.3676} & $-6.64$ / $-6.12$ &""")
E("M17b Table 4 ES, EVT tail", r"""& 0.3680 & $-7.21$ / $-6.15$ &""", r"""& 0.3680 & $-7.33$ / $-6.16$ &""")
E("M17c Table 4 ES, GJR-skew-t", r"""& 0.3688 & $-6.85$ / $-6.09$ &""", r"""& 0.3688 & $-6.87$ / $-6.09$ &""")
E("M17d Table 4 ES, rolling FHS", r"""& 0.3694 & $-6.71$ / $-6.21$ &""", r"""& 0.3694 & $-7.02$ / $-6.21$ &""")
E("M17e Table 4 ES, HS", r"""& 0.3752 & $-7.41$ / $-7.08$ &""", r"""& 0.3752 & $-7.71$ / $-7.08$ &""")

E("M18 Table 4 note: ES convention",
r"""Predicted ES is the numerical tail integral
$\alpha^{-1}\!\int_0^\alpha Q(v)\,dv$: closed forms for the $t$ and
normal entrants, 200-node integration of the Hansen skew-$t$ inverse
CDF, and a matched 20-node midpoint rule for the empirical quantile
models and for the residual-hybrid, whose VaR and ES are both read from
one monotonized min-envelope curve $Q^{z}_t$ of \eqref{eq:hybrid}. A three-node tail-quantile average, a common shortcut,
understates predicted ES by 2--6\% relative to this integral.""",
r"""Predicted ES is the tail integral
$\alpha^{-1}\!\int_0^\alpha Q(v)\,dv$ of each model's quantile function: closed forms for the $t$ and
normal entrants, an exact substitution integral of the Hansen skew-$t$ inverse
CDF, the exact empirical tail mean for historical simulation and the FHS variants, and the converged integral of
Section~\ref{sec:hybrid} for the EVT-tailed residual-hybrid, whose VaR and ES are both read from
one monotonized min-envelope curve $Q^{z}_t$ of \eqref{eq:hybrid}. The raw residual-hybrid has no quantile below its lowest
fitted level, so its ES extends the body flat below $\alpha/40$ and is a lower bound on its severity.""")

E("M19 Table 4 note: overstatement range",
r"""each fat-tailed entrant exceeds its own-tail comparator by 8\% (raw
residual-hybrid, rolling FHS) to 22\% (pooled FHS).""",
r"""each fat-tailed entrant exceeds its own-tail comparator by 8\% (raw
residual-hybrid, a lower bound) to 22\% (pooled FHS).""")

E("M20 Table 4 note: corrected exception battery",
r"""FZ0 comparison passes Kupiec for $84\%$ and Christoffersen conditional coverage for $89\%$ (at 97.5\%, $73\%$ and $78\%$), with the date-clustered test passing at 99\% ($t=0.29$) and at 97.5\% ($t=1.60$) while its pooled body without the EVT tail is rejected ($t=4.1$ and $4.4$): the tail stages carry the regulatory
calibration, and the 2008-era re-run confirms it out of era.
At 97.5\% the split-conformal recalibration is the stage that repairs
pooled Kupiec (GBM-recal $p=0.24$).""",
r"""FZ0 comparison passes Kupiec for $84\%$ and Christoffersen conditional coverage for $90\%$ (at 97.5\%, $70\%$ and $75\%$). On the date-clustered test it passes at 99\% ($t=0.62$) and is rejected marginally at 97.5\% ($t=2.00$), where GARCH-$t$ passes ($t=1.56$) and pooled FHS is rejected more strongly ($t=3.35$); its pooled body without the EVT tail is rejected at both levels ($t=4.3$ and $4.7$), so the tail stage carries the regulatory
calibration. All three pass on the 2000--2013 holdout, and the adaptive overlay of Stage~4 passes on both panels.""")

E("M21 stress-era overlay",
r"""The conformal layer, calibrated on pre-crisis data, tilts conservative on the 2000--2013 holdout (breach $2.21\%$ against the $2.5\%$ target, $t=-1.09$, not significant). The safety margin widens when the calibration era is calmer than the test era, and the error runs in the direction regulators prefer.""",
r"""The adaptive overlay of Stage~4 stays near nominal through the crisis window, breaching at $2.43\%$ and $2.38\%$ against the $2.5\%$ target and at $1.06\%$ and $1.04\%$ against $1\%$ for the two update rates (date-clustered $|t|\le0.44$).""")

# ---------------------------------------------------------------- Figure 2
E("M22a Figure 2 bars, 1%",
r"""{(GARCH-$t$,0.00848)(GARCH-EVT,0.00784)(Taylor,0.01748)(FHS,0.02517)(GAS,0.08514)};""",
r"""{(GARCH-$t$,0.00919)(GARCH-EVT,0.00861)(Taylor,0.01819)(FHS,0.02588)(GAS,0.08585)};""")
E("M22b Figure 2 bars, 2.5%",
r"""{(GARCH-$t$,0.00545)(GARCH-EVT,0.00888)(Taylor,0.00024)(FHS,0.01370)(GAS,0.05030)};""",
r"""{(GARCH-$t$,0.00557)(GARCH-EVT,0.00963)(Taylor,0.00036)(FHS,0.01382)(GAS,0.05042)};""")
E("M22c Figure 2 caption",
r"""lowest FZ0 at $1\%$ and is never significantly beaten at $2.5\%$, beating GARCH-$t$ (DM $4.8$ at
$1\%$, $5.1$ at $2.5\%$), pooled-tail GARCH-EVT (DM $4.8$ and $6.8$; its bars come from the same-rows run of the Online Appendix, whose accuracy-layer baseline is within $0.0006$ of the one plotted) and FHS (DM $4.9$ and $5.3$), together with the two
FZ-estimated dynamic-ES benchmarks fitted per name by multi-start FZ0
minimization, the generalized autoregressive score (GAS) model of \citet{patton2019} (DM $9.5$ and $8.6$) and the
ES-CAViaR of \citet{taylor2019} (DM $3.6$ at $1\%$, a tie at $2.5\%$). SAV-CAViaR, scored on its own common sample (Table~\ref{tab:noncore}), ties the accuracy layer and is omitted here; the conformal-overlay variant, which concedes FZ0 at $2.5\%$, is discussed in the text.}""",
r"""lowest FZ0 at $1\%$ and is never significantly beaten at $2.5\%$, beating GARCH-$t$ (DM $4.9$ at
$1\%$, $5.5$ at $2.5\%$), pooled-tail GARCH-EVT (DM $4.5$ and $6.3$) and per-name FHS (DM $4.9$ and $5.4$), together with the two
FZ-estimated dynamic-ES benchmarks fitted per name by multi-start FZ0
minimization, the generalized autoregressive score (GAS) model of \citet{patton2019} (DM $9.9$ and $8.8$) and the
ES-CAViaR of \citet{taylor2019} (DM $3.8$ at $1\%$, a tie at $2.5\%$). Every bar and DM comes from one run on the same rows against the same accuracy-layer forecast. SAV-CAViaR, scored on its own common sample (Table~\ref{tab:noncore}), ties the accuracy layer and is omitted here; the adaptive overlay, within noise of the accuracy layer at both levels, is discussed in the text.}""")

# ---------------------------------------------------------------- Section 5.1 text
E("M23a 5.1 GARCH-t and FHS DMs",
r"""At 1\% it beats GARCH-$t$ (DM 4.8) and FHS (DM 4.9), the two standard benchmarks the accuracy claim rests on. At 2.5\% it again wins over GARCH-$t$ (DM 5.1).""",
r"""At 1\% it beats GARCH-$t$ (DM 4.9) and FHS (DM 4.9), the two standard benchmarks the accuracy claim rests on. At 2.5\% it again wins over GARCH-$t$ (DM 5.5).""")
E("M23b 5.1 GARCH-EVT DMs",
r"""with a pooled tail by DM 4.8 at 1\% and 6.8 at 2.5\%, and per name by DM 5.2 and 5.4,""",
r"""with a pooled tail by DM 4.5 at 1\% and 6.3 at 2.5\%, and per name by DM 5.1 and 5.4,""")
E("M23c 5.1 DQ: static-overlay clause removed",
r"""83\% and 62\% for the pooled GARCH-EVT; the overlay lifts the 2.5\% rate to 84\%.""",
r"""83\% and 62\% for the pooled GARCH-EVT.""")
E("M23d 5.1 OA.11 sentence: tie group at 2.5%",
r"""the accuracy layer has the lowest mean FZ0 of the benchmark models at 1\% and is within noise of GJR-GARCH-skew-$t$ at both levels (DM 1.1 and $-0.3$) and of the Taylor model at 2.5\% (DM $-0.1$); its own pooled body, without the EVT branch, scores lower still (DM $-1.3$ at 1\%, within noise, and $-1.9$ at 2.5\%, marginal under the convention of Section~\ref{sec:conventions}) while breaching at 1.3\% and 3.3\%, the coverage cost the EVT branch removes; the 90\% Model Confidence Set on the per-date FZ0 series contains the accuracy layer, the conformal overlay, the pooled body and GJR-GARCH-skew-$t$ at 1\%""",
r"""the accuracy layer has the lowest mean FZ0 of the benchmark models at 1\%, a level statement since GJR-GARCH-skew-$t$ is within noise (DM 1.2). At 2.5\% the accuracy layer, GJR-GARCH-skew-$t$ and the Taylor model form a tie group (FZ0 1.8486, 1.8486 and 1.8490; DM $-0.02$ and 0.1), and nothing significantly beats the accuracy layer; its own pooled body, without the EVT branch, scores lower still (DM $-1.1$ at 1\% and $-1.5$ at 2.5\%, both within noise) while breaching at 1.3\% and 3.3\%, the coverage cost the EVT branch removes; the 90\% Model Confidence Set on the per-date FZ0 series contains the accuracy layer, the pooled body and GJR-GARCH-skew-$t$ at 1\% (with the static-shift row, identical to the accuracy layer at that level)""")
E("M23e 5.1 GAS and Taylor DMs",
r"""The accuracy layer beats the GAS at both levels (DM 9.5 at 1\%, 8.6 at 2.5\%), and it beats the Taylor model at 1\% (DM 3.6) while tying it at 2.5\%,""",
r"""The accuracy layer beats the GAS at both levels (DM 9.9 at 1\%, 8.8 at 2.5\%), and it beats the Taylor model at 1\% (DM 3.8) while tying it at 2.5\%,""")
E("M23f 5.1 Beta-t-EGARCH (A3) converged",
r"""has lower FZ0 at both levels under the frozen fits, significantly at 2.5\% (DM 2.53).""",
r"""has lower FZ0 at both levels under the frozen fits, significantly at 2.5\% (DM 2.48).""")
E("M23g 5.1 Beta-t-EGARCH top decile under refits",
r"""while Beta-$t$-EGARCH stays significantly ahead in the top decile of the score (DM 3.7 and 5.2).""",
r"""while Beta-$t$-EGARCH stays significantly ahead in the top decile of the score (DM 3.7 and 5.1).""")

E("M24a refit ranking: GARCH-t and pooled FHS named",
r"""the accuracy layer keeps the lowest FZ0 at both levels, beating GARCH-$t$ and FHS by DM 3.6 and 2.7 at 1\% and 3.1 and 2.6 at 2.5\% (one-sided $p<0.005$).""",
r"""the accuracy layer keeps the lowest FZ0 at both levels, beating GARCH-$t$ by DM 3.5 at 1\% and 3.1 at 2.5\%, and pooled FHS, the unconditional residual quantile on the same refitted scale, by DM 3.2 and 3.4 (one-sided $p<0.005$).""")
E("M24b refit: Beta-t-EGARCH engine (A7) converged",
r"""the engine has lower FZ0 under the frozen fits (DM 3.31 and 4.55, and 3.09 and 3.48 on the 2000--2013 holdout). Under annual refits the two engines are within noise overall (DM $-0.31$ and $-0.01$), and the Beta-$t$-EGARCH engine is ahead in the top decile of the score (DM 5.37 and 7.25),""",
r"""the engine has lower FZ0 under the frozen fits (DM 3.16 and 4.37, and 3.14 and 3.63 on the 2000--2013 holdout). Under annual refits the two engines are within noise overall (DM $-0.31$ and $-0.02$), and the Beta-$t$-EGARCH engine is ahead in the top decile of the score (DM 5.46 and 7.37),""")

E("M25 summary and Stage 4 results",
r"""and passes the aggregate date-clustered exception test at 99\% and marginally at 97.5\%. Per-name diagnostics show residual cross-sectional calibration heterogeneity (84\% and 89\% pass rates at 99\%). The one failed test is the pooled 221.6k-observation Kupiec at 97.5\%, which at this sample size rejects very small deviations. The frontier concentrates the joint-score
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

Most of the concession reflects a frozen margin. The adaptive form of the shift \citep{gibbs2021aci}, updated daily from the panel breach frequency, shrinks the margin out of era (from $-0.35$ to $-0.10$, breach $1.63\%\to2.06\%$) and erases the overall concession. The adaptive variant ties GARCH-$t$ on 2.5\% FZ0 (DM $-0.3$). The top-decile cost survives the adaptive shift too, at $-4.8$ and $-6.0$ for the two update rates against the static shift's $-7.1$, the price of a margin where the bands are widest. The static shift also worsens per-name coverage (Kupiec$_{97.5}$ pass rate 48\%, against 71\% unshifted and 69\% adaptive), because over-coverage fails the test from the other side. Each ``lowest FZ0'' statement refers to the accuracy layer.""",
r"""and passes the aggregate date-clustered exception test at 99\%. At 97.5\% it is rejected marginally on 2020--2024 ($t=2.00$, against 1.56 for GARCH-$t$ and 3.35 for pooled FHS) and passes on the 2000--2013 holdout ($t=0.25$). Per-name diagnostics show residual cross-sectional calibration heterogeneity (84\% and 90\% pass rates at 99\%). The pooled 221.6k-observation Kupiec test also rejects at 97.5\%, and at this sample size it rejects very small deviations. The joint-score edge over GARCH-$t$ is less concentrated than the pinball edge: at 1\% the mean FZ0 gap is 0.011 in the top misspecification decile and 0.009 in the other nine (DM 3.4 and 4.7).

\paragraph{The adaptive overlay.} Stage~4 shifts the accuracy layer by a scalar updated each day from the panel breach frequency (Section~\ref{sec:hybrid}). Before the run we fixed the rule under which it would be switched on: on both the 2014--2024 panel and the 2000--2013 holdout, at both update rates, the overlay had to bring the 97.5\% breach rate within 0.3 points of nominal and keep its FZ0 within noise of the accuracy layer ($|\mathrm{DM}|<2$) at both levels. It meets both conditions. In era it moves the 97.5\% breach rate from 2.78\% to 2.68\% and 2.60\% for $\gamma=0.02$ and $0.05$, with FZ0 slightly below the accuracy layer's (DM $-1.3$ and $-1.0$), and the date-clustered test passes ($t=1.30$ and $0.78$); at 99\% it leaves the accuracy layer essentially unchanged (DM $-0.3$ for both rates). On the holdout its breach rates are 2.43\% and 2.38\% at 97.5\% and 1.06\% and 1.04\% at 99\%, with every $|\mathrm{DM}|$ against the accuracy layer below 1. It keeps the joint-score lead over GARCH-$t$ at 2.5\% in era (DM 5.2 and 4.1). In the top misspecification decile it concedes 2.5\% FZ0 to the accuracy layer (DM 3.4 and 3.6), the price of widening the band where it is already widest. A static shift, the construction of earlier versions, fails in both of its forms. Estimated on the body's calibration errors and applied to the envelope, it counts the tail correction twice and over-covers (breach 1.66\%, date-clustered $t=-9.1$, FZ0 DM 2.6 against the accuracy layer); estimated on the envelope's own errors, it still over-covers in era (2.15\%, $t=-3.1$), because the calibration block and the test era miss on opposite sides of nominal. Each ``lowest FZ0'' statement refers to the accuracy layer.""")

E("M26 direct ES backtests rewritten",
r"""The residual-hybrid has the $Z_2$ closest to zero at $1\%$ and ties GARCH-$t$ at $2.5\%$, and the residual-hybrid and GARCH-$t$ are the two models that pass the $1\%$ Kupiec test. On the exceedance residual, FHS, which targets the empirical tail directly, is closest to zero. On the exception and $Z_2$ diagnostics the residual-hybrid's tail is calibrated; the exceedance-residual test rejects it at $1\%$ ($p=0.02$, Online Appendix Table~OA.5). The comparative result is the FZ0 loss, where the edge over GARCH-$t$ and FHS is significant at both levels.""",
r"""The residual-hybrid has the $Z_2$ closest to zero at $1\%$ and ties GARCH-$t$ at $2.5\%$, and the residual-hybrid and GARCH-$t$ are the two models that pass the $1\%$ Kupiec test. On the exceedance residual the residual-hybrid is closest to zero at $1\%$ and FHS, which targets the empirical tail directly, at $2.5\%$. Neither test rejects the residual-hybrid's ES at either level (exceedance residual $p=0.38$ and $0.58$, Online Appendix Table~OA.5), nor under the annual refits ($p=0.21$ and $0.33$), while GARCH-$t$'s $1\%$ ES is rejected as not severe enough ($p<0.001$). On the 2000--2013 holdout the verdict depends on the direction of the error. At $1\%$ the test rejects every construction, GARCH-$t$'s ES as not severe enough (mean exceedance residual $-0.17$) and the residual-hybrid's and FHS's as too severe ($+0.15$ and $+0.12$); at $2.5\%$ the residual-hybrid passes ($p=0.48$) and GARCH-$t$ is rejected ($p=0.02$). These verdicts require an accurate ES: with the 20-node midpoint ES of earlier versions, the exceedance-residual test rejects a correctly specified model in 22.5\% of simulated samples at a nominal 5\% (Online Appendix Table~OA.16). The comparative result is the FZ0 loss, where the edge over GARCH-$t$ and FHS is significant at both levels.""")

# ---------------------------------------------------------------- ten-day
E("M27 ten-day: envelope tested, not recommended",
r"""calibrated through the crisis while the direct tail under-states it. In a
2008-type era the direct ten-day model should therefore be run with the
$\sqrt{h}$-scaled parametric number alongside it as a conservative
envelope. The ten-day evidence is therefore mixed; the benchmark
table is Online Appendix Table~OA.13.""",
r"""calibrated through the crisis while the direct tail under-states it. A
level-wise minimum of the two curves does not repair this. In 2008--2009
it breaches at 1.61\% and 3.52\% against the 1\% and 2.5\% targets, closer
to nominal than $\sqrt{h}$ scaling (1.93\% and 3.99\%) but short of it, and
in era it costs 1.1\% of pinball against the direct model (DM 3.9), so we
do not recommend it. Rescaling the direct curve by the fitted variance term
structure ties the direct model in era (DM $-1.7$; $-0.3$ on
non-overlapping dates) with breach rates nearer nominal (0.97\% and
2.44\%) and loses to it out of era (DM 2.3); we record it as exploratory.
The ten-day evidence is therefore mixed; the benchmark
table is Online Appendix Table~OA.13.""")

# ---------------------------------------------------------------- Section 6
E("M28a using the score: switches",
r"""(Section~\ref{sec:gate}). The one setting in which it drives a switch is
the per-name standalone model, where nonparametric quantiles underperform
on calm days.""",
r"""(Section~\ref{sec:gate}). It drives a switch in two settings: the per-name
standalone model, where nonparametric quantiles underperform
on calm days, and the choice of Stage-1 scale, described next.""")
E("M28b the score-gated scale",
r"""component of the charge scales with the ES forecast.
""",
r"""component of the charge scales with the ES forecast.

\paragraph{Choosing the scale with the score.} Under the annual refits the Beta-$t$-EGARCH engine and the GARCH-$t$ engine are within noise overall while the robust one leads in the top decile of the score (Section~\ref{sec:frtb}), which suggests letting the score choose the scale. A gate that runs the GARCH-$t$ engine when the previous day's $\mathrm{mk}_{63}$ is below its pre-2020 90th percentile, a threshold fixed before the first test year, and the Beta-$t$-EGARCH engine otherwise switches on 10.0\% of the 251.6k refit test asset-days. In a run whose predictions were written beforehand, its joint score is significantly better than always running the GARCH-$t$ engine (DM 6.3 at 1\%, 8.1 at 2.5\%) and statistically equal to always running the Beta-$t$-EGARCH engine (DM 0.8 at both levels), which is slightly ahead of it in the top decile (DM $-1.6$ and $-1.7$, within noise); the gate and both fixed engines pass the date-clustered exception test at both levels. The gate keeps the paper's Stage-1 filter on nine asset-days in ten and recovers the robust scale's gain where the score is high. Where tail accuracy is the only objective, running the robust scale everywhere does as well.
""")
E("M29 cold start: calibration and benchmark",
r"""Section~\ref{sec:hybrid} attaches once a scale filter is estimable. The
amortized forecast beats own-history benchmarks at each listing age, and
the edge is largest when the name is young ($\sim$6--10\% of pinball in
the first trading month). The ablation""",
r"""Section~\ref{sec:hybrid} attaches once a scale filter is estimable.
Day-one quantiles exist but are not calibrated at the regulatory levels. In a separate run on
303 names listed from 2018, the best pooled peer forecast breaches its 1\%
VaR on 7.1\% of asset-days in the first five trading days. Once the name
has a few weeks of history the amortized forecast beats its own-history
benchmark, an expanding-window Gaussian scored at seven levels from 0.05
to 0.95, by about 9\% of pinball at 15 to 30 trading days and
6.5--7\% through the first year, and on 91.7\% of test dates overall. The ablation""")

# ---------------------------------------------------------------- conclusion
E("M30a conclusion: first-week caveat",
r"""produces a forecast for a newly listed asset from its first day, and the residual-hybrid attaches once a scale filter is estimable (Section~\ref{sec:applications}).""",
r"""produces a forecast for a newly listed asset from its first day, though not yet a calibrated one in its first week, and the residual-hybrid attaches once a scale filter is estimable (Section~\ref{sec:applications}).""")
E("M30b conclusion: ES backtests by era and direction (A8)",
r"""The McNeil--Frey exceedance-residual test rejects the 1\% ES on the full panel, on the 2000--2013 holdout and under annual refits, for every Stage-1 filter tried and for GARCH-$t$ itself; filtered historical simulation, which fits the empirical tail directly, is the one entrant that passes it on the full panel.""",
r"""The accuracy layer's 97.5\% coverage is rejected marginally on 2020--2024 (date-clustered $t=2.00$, against 1.56 for GARCH-$t$), which the adaptive overlay repairs at no cost in FZ0. The McNeil--Frey exceedance-residual test rejects none of the residual-hybrid's ES forecasts on the design panel or under annual refits, but on the 2000--2013 holdout it rejects the 1\% ES of every construction tried, the residual-hybrid's and FHS's as too severe and GARCH-$t$'s as not severe enough.""")
E("M30c conclusion: what survives a robust scale",
r"""and what survives is real but modest.""",
r"""and what survives is real but modest; on a robust scale it sits in the tail, where the flexible tail still lowers the joint $(\VaR,\ES)$ loss at both regulatory levels under frozen fits and annual refits.""")
E("M30d conclusion: the score chooses the scale",
r"""rather than a day-to-day switching rule, since the day-ahead oracle gap is largely unforecastable (Section~\ref{sec:gate}).""",
r"""rather than a day-to-day switch between parametric and flexible shapes, since the day-ahead oracle gap is largely unforecastable (Section~\ref{sec:gate}). It can still choose the Stage-1 scale: a gate that moves to Beta-$t$-EGARCH in the score's top decile is significantly better than GARCH-$t$ at both levels and statistically equal to running the robust scale everywhere (Section~\ref{sec:applications}).""")

# ================================================================ ONLINE APPENDIX
O("O01 deployed pipeline: adaptive shift",
r"""and, where the desk elects the coverage overlay, the conformal shift
$c_\tau$ (static or adaptive); (iv)~report""",
r"""and, where the desk elects the coverage overlay, the adaptive conformal shift
$c_\tau$, updated from the previous day's panel breach frequency; (iv)~report""")
O("O02 deployed pipeline: annual refit list",
r"""re-estimate tail and shifts, all on trailing data.""",
r"""re-estimate the tail, all on trailing data.""")
O("O03 summary table: holdout exception row",
r"""at 97.5\%, 0.35 (EVT tail) and $-1.09$ (static overlay, conservative); HS, EWMA fail \\""",
r"""at 97.5\%, 0.35 (EVT tail) and $-0.27$ to $-0.44$ (adaptive overlay); HS, EWMA fail \\""")
O("O04 summary table: FZ0 row",
r"""DM 4.8--4.9 vs GARCH-$t$, FHS at 1\%, 5.1 at 2.5\%; 8.6--9.5 vs PZC GAS, 3.6-to-tie vs Taylor ES-CAViaR (multi-start); static shift concedes 2.5\% (DM $-2.0$), adaptive shift erases it (tie; top-decile cost remains) \\""",
r"""DM 4.9 vs GARCH-$t$ and FHS at 1\%, 5.5 vs GARCH-$t$ at 2.5\%; 8.8--9.9 vs PZC GAS, 3.8-to-tie vs Taylor ES-CAViaR (multi-start); tie group with GJR-skew-$t$ and Taylor at 2.5\%; adaptive overlay within noise of the accuracy layer (top-decile cost at 2.5\%) \\""")
O("O05 summary table: cold-start row",
r"""Cold start / young listings & $+6$--$10\%$ & full-scale panel; no per-asset model applies \\""",
r"""Cold start / young listings & $+9\%$ at 15--30 days; $+6.5$--$7\%$ to one year & vs an expanding-window Gaussian at levels 0.05--0.95; point-in-time characteristics; first-week VaR not calibrated \\""")
O("O06 decision table: Stage 4",
r"""Residual-hybrid $+$ EVT $+$ conformal & Section~5 of the paper \\""",
r"""Residual-hybrid $+$ EVT $+$ adaptive overlay & Section~5 of the paper \\""")

O("O07 ES backtests: Z2 value",
r"""carries the statistic closest to zero at $1\%$ ($-0.06$, not rejected) and is""",
r"""carries the statistic closest to zero at $1\%$ ($-0.05$, not rejected) and is""")
O("O08 ES backtests: McNeil-Frey reverses in the engine's favour",
r"""McNeil--Frey exceedance residual FHS, which fits the empirical residual tail
directly, sits closest to zero and is the only entrant the test does not
reject at $1\%$ ($-0.10$, $p=0.07$); at $2.5\%$ FHS, GARCH-$t$ and the engine all
fall inside the non-rejection region.""",
r"""McNeil--Frey exceedance residual the engine sits closest to zero at $1\%$ and is
the clearest non-rejection ($-0.06$, $p=0.38$), ahead of FHS, which fits the empirical
residual tail directly ($-0.10$, $p=0.07$), while GARCH-$t$ and the Gaussian filter are
rejected; at $2.5\%$ FHS, GARCH-$t$ and the engine all
fall inside the non-rejection region.""")
O("O09 ES backtests: FHS row bold",
r"""& $-0.29$ & $\mathbf{-0.10\,(0.07)}$ \\""",
r"""& $-0.29$ & $-0.10\,(0.07)$ \\""")
O("O10 ES backtests: engine row converged",
r"""Engine (EVT-tailed) & $0.027$ & $0.00$ & $\mathbf{-0.10}$ & $-0.04\,(0.24)$ & $0.010$ & $\mathbf{0.42}$ & $\mathbf{-0.06}$ & $-0.15\,(0.02)$ \\""",
r"""Engine (EVT-tailed) & $0.028$ & $0.00$ & $\mathbf{-0.10}$ & $+0.02\,(0.58)$ & $0.010$ & $\mathbf{0.07}$ & $\mathbf{-0.05}$ & $\mathbf{-0.06\,(0.38)}$ \\""")
O("O11 ES backtests: note",
r"""The deployed engine adds a $97.5\%$ conformal shift and is conservative out of
era (breach $1.66\%$, $Z_2=+0.29$ at $2.5\%$), overstating the tail in the
regulator-preferred direction. Source: \texttt{engine\_esbt\_results.json}.""",
r"""The engine's ES is the converged integral of Section~2.2 of the paper. The static conformal
shift of earlier versions, estimated on the body and applied to the engine, over-covers
(breach $1.66\%$, $Z_2=+0.31$, McNeil--Frey $p=0.002$ at $2.5\%$); the adaptive overlay that
replaces it is reported in Section~5.1 of the paper. Source: \texttt{engine\_esbt\_results.json}.""")

O("O12 GARCH-EVT FZ0 table rows",
r"""Engine (body/EVT minimum) & 2.1473 & 1.04\% & --- & 1.8494 & 2.78\% & --- \\
Engine $+$ conformal overlay (97.5\%) & 2.1473 & 1.04\% & 0.00 & 1.8656 & 1.66\% & 2.29 \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.33 & 1.8433 & 3.25\% & -1.87 \\
GARCH-EVT, pooled tail & 2.1551 & 1.06\% & 4.83 & 1.8583 & 2.89\% & 6.85 \\
GARCH-EVT, per name & 2.1693 & 1.13\% & 5.23 & 1.8596 & 2.93\% & 5.39 \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 5.00 & 1.8542 & 2.71\% & 4.55 \\""",
r"""Engine (body/EVT minimum) & 2.1465 & 1.04\% & --- & 1.8486 & 2.78\% & --- \\
Engine $+$ static shift (97.5\%) & 2.1465 & 1.04\% & 0.00 & 1.8664 & 1.66\% & 2.57 \\
GARCH $+$ pooled body (no EVT) & 2.1411 & 1.32\% & -1.09 & 1.8432 & 3.25\% & -1.54 \\
GARCH-EVT, pooled tail & 2.1551 & 1.06\% & 4.51 & 1.8583 & 2.89\% & 6.33 \\
GARCH-EVT, per name & 2.1693 & 1.13\% & 5.14 & 1.8596 & 2.93\% & 5.36 \\
GARCH(1,1)-$t$ & 2.1557 & 1.03\% & 4.94 & 1.8542 & 2.71\% & 5.50 \\""")
O("O13 GARCH-EVT FZ0 table note",
r"""the overlay row adds the split-conformal shift at 97.5\% and coincides with the engine at 1\%. GARCH-EVT ES is the McNeil--Frey closed form; the engine and body ES are the 20-node integral of the monotonized quantile curve. Top-$\mathrm{mk}_{63}$-decile DM of pooled GARCH-EVT versus the engine: 2.63 at 1\% and 4.26 at 2.5\%.""",
r"""the static-shift row adds the split-conformal shift of earlier versions at 97.5\%, estimated on the body's calibration errors, and coincides with the engine at 1\%; Stage~4 now uses the adaptive overlay of Section~5.1 of the paper. GARCH-EVT ES is the McNeil--Frey closed form; the engine ES is the converged integral of the monotonized quantile curve (Section~2.2 of the paper), and the body ES a 20-node integral, since the body has no quantile below its lowest fitted level. Top-$\mathrm{mk}_{63}$-decile DM of pooled GARCH-EVT versus the engine: 2.26 at 1\% and 3.42 at 2.5\%.""")
O("O14 DQ table: static-shift row named",
r"""Engine $+$ conformal overlay & \multicolumn{2}{c}{86\%} & \multicolumn{2}{c}{84\%} \\""",
r"""Engine $+$ static shift (97.5\%) & \multicolumn{2}{c}{86\%} & \multicolumn{2}{c}{84\%} \\""")

O("O15 all-benchmark FZ0 table rows",
r"""Engine (accuracy layer) & 2.1473 & 1.04\% & --- & $\checkmark$ & 1.8494 & 2.78\% & --- & $\checkmark$ \\
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
Historical simulation (500d) & 2.2512 & 0.87\% & 6.24 &  & 1.9182 & 2.06\% & 5.16 &  \\""",
r"""Engine (accuracy layer) & 2.1465 & 1.04\% & --- & $\checkmark$ & 1.8486 & 2.78\% & --- & $\checkmark$ \\
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
Historical simulation (500d) & 2.2512 & 0.87\% & 6.34 &  & 1.9182 & 2.06\% & 5.28 &  \\""")
O("O16 all-benchmark FZ0 table note",
r"""ES: closed forms for the $t$, skew-$t$ and Gaussian entrants; empirical tail means for HS and FHS; McNeil--Frey closed form for GARCH-EVT; the FZ0-estimated dynamic pair for GAS and Taylor; the 20-node integral of the monotonized curve for the engine and body.""",
r"""ES: closed forms for the $t$ and Gaussian entrants; an exact substitution integral for the skew-$t$; empirical tail means for HS and FHS; McNeil--Frey closed form for GARCH-EVT; the FZ0-estimated dynamic pair for GAS and Taylor (the GAS fit of record is this run's); the converged integral of the monotonized curve for the engine (Section~2.2 of the paper), and a 20-node integral for the body, which has no quantile below its lowest fitted level. The static-shift row is the split-conformal shift of earlier versions, estimated on the body's calibration errors; Stage~4 now uses the adaptive overlay of Section~5.1 of the paper. At 2.5\% the engine, the body, GJR-GARCH-skew-$t$ and the Taylor model form a tie group (FZ0 1.8432--1.8490, $|$DM$|\le1.54$), reported as a group rather than an order.""")
O("O17 tests table: static-shift row named",
r"""Engine $+$ conformal overlay & --- & --- & 86\% & 84\% & 0\% & 45\% & --- & --- \\""",
r"""Engine $+$ static shift (97.5\%) & --- & --- & 86\% & 84\% & 0\% & 45\% & --- & --- \\""")

O("O18 ten-day table: envelope and rescaling rows",
r"""Direct model & --- & 1.13 / 2.80 & --- & 1.24 / 2.82 \\""",
r"""Envelope, $\min$(direct, $\sqrt{h}$) & $+1.14$ (3.9; 2.4) & 0.94 / 2.45 & $+0.15$ ($-0.32$; 0.04) & 1.01 / 2.38 \\
Variance-ratio rescaling & $-0.12$ ($-1.7$; $-0.3$) & 0.97 / 2.44 & $+0.07$ (2.3; 2.0) & 1.23 / 2.79 \\
Direct model & --- & 1.13 / 2.80 & --- & 1.24 / 2.82 \\""")
O("O19 ten-day table: note",
r"""so the out-of-era accuracy comparison is uninformative; see \texttt{tenday\_diag\_results.json}.""",
r"""so the out-of-era accuracy comparison is uninformative; see \texttt{tenday\_diag\_results.json}. The envelope (the level-wise minimum of the direct and $\sqrt{h}$ curves, rearranged) and the variance-ratio rescaling (the direct curve rescaled by the square root of the iterated-to-$\sqrt{h}$ variance ratio) come from a rerun on the same rows, \texttt{tenday\_envelope\_results.json}, in which the $\sqrt{h}$ row reads $+1.39$ (DM 6.7) in era and $+0.51$ (DM 0.05) out of era. In 2008--2009 the envelope breaches at 1.61\% and 3.52\%, against 1.93\% and 3.99\% for $\sqrt{h}$ scaling and 2.15\% and 4.33\% for the direct model. Both constructions were run with predictions written beforehand; the envelope did not meet its two, and the rescaling is exploratory.""")

O("O20 synthetic-truth table",
r"""Result file \texttt{arcd\_bench\_results.json}; predictions in the script header.
\end{tablenotes}
\end{threeparttable}
\end{table}
""",
r"""Result file \texttt{arcd\_bench\_results.json}; predictions in the script header.
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
""")

O("O21 ES integral paragraph",
r"""(i)~Computing ES$_\alpha$ as the numerical integral of the min-envelope curve rather than by the GPD closed form of \citet{mcneilfrey2000} leaves each comparison in place: at 2.5\% the accuracy layer's FZ0 is 1.849 under the integral against 1.851 under the closed form (the integral marginally better, DM 5.3), and the estimator-over-GARCH-$t$ DM is 4.8 and 5.1 at the 1\% and 2.5\% levels under either convention.""",
r"""(i)~ES$_\alpha$ is the converged integral of the min-envelope curve (Section~2.2 of the paper). Replacing it by the GPD closed form of \citet{mcneilfrey2000} gives the accuracy layer a 2.5\% FZ0 of 1.851, against 1.8486 under the integral, and under the integral the estimator-over-GARCH-$t$ DM is 4.9 and 5.5 at the 1\% and 2.5\% levels. The 20-node midpoint rule of earlier versions understated the engine's $|\ES|$ by $1.65\%$ at 1\% and $1.66\%$ at 2.5\% on average, for every one of the 200 names, and the rolling-window empirical rows of Table~4 of the paper by 4--5\%; those rows now use the exact tail mean.""")

O("O22 FRTB figure: predicted ES",
r"""{(Hyb,-6.63) (Hyb+EVT,-7.21) (GARCH-t,-6.98) (FHS,-7.17)};""",
r"""{(Hyb,-6.64) (Hyb+EVT,-7.33) (GARCH-t,-6.98) (FHS,-7.17)};""")
O("O23 FRTB figure: realized ES",
r"""{(Hyb,-6.12) (Hyb+EVT,-6.15) (GARCH-t,-5.92) (FHS,-5.89)};""",
r"""{(Hyb,-6.12) (Hyb+EVT,-6.16) (GARCH-t,-5.92) (FHS,-5.89)};""")
O("O24 FRTB figure: caption",
r"""hybrid. Right: ES$_{97.5}$ predicted (exact tail integral) vs realized""",
r"""hybrid. Right: ES$_{97.5}$ predicted (the tail integral of Table~4 of the paper) vs realized""")

O("O25 pipeline figure: breaches feed Stage 4",
r"""{held-out\\calibration split};""",
r"""{breaches observed\\through day $t-1$};""")
O("O26 pipeline figure: caption",
r"""or held out from training (the calibration split behind the conformal overlay).""",
r"""or observed as the forecasts are scored (the breaches behind the adaptive overlay).""")

O("O27 trading paragraph",
r"""with the model supplying tail control rather than alpha on the mean; the
paper evaluates no prices, premia, costs or trading outcomes, so this is a
direction and not a result. The gating experiment provides no evidence that the score times
individual crash days, so its role there is continuous tail-risk
measurement, not a market-timing signal.""",
r"""with the model supplying tail control rather than alpha on the mean. The
paper evaluates no prices, premia, costs or trading outcomes, so the
application is untested here. The gating experiment provides no evidence that the score times
individual crash days, so its role there would be continuous tail-risk
measurement.""")


# ------------------------------------------- R59 preconditions 2 and 3, added on ai2 (see the ledger appendix)
E("M31 splice-level range on the converged integral",
r"""The splice level is immaterial ($p_0\in\{1.5\%,2.5\%,5\%\}$: DM 4.6--4.8).""",
r"""The splice level is immaterial ($p_0\in\{1.5\%,2.5\%,5\%\}$: DM 4.5--4.7 at the $1\%$ level).""")

E("M32 Section 5.3 daily-core concession on the converged integral",
r"""the daily-data core concedes FZ0 at both levels (DM $6.2$ and $4.5$)""",
r"""the daily-data core concedes FZ0 at both levels (DM $6.2$ and $4.6$)""")

O("O28 tab:rgarch rows on the converged integral",
r"""Daily core (GARCH-$t$ $+$ EVT) & $1.500$ & $+6.2$ & $1.774$ & $+4.5$ \\
Realized scale $+$ residual quantile & $1.445$ & --- & $1.716$ & --- \\
\quad $+$ EVT tail & $1.444$ & $-0.3$ & $1.716$ & $+0.5$ \\
\quad $+$ pooled shape learner & $1.445$ & $+0.2$ & $1.710$ & $-1.3$ \\
\quad $+$ shape $+$ EVT (realized-hybrid) & $1.447$ & $+0.7$ & $1.718$ & $+0.6$ \\""",
r"""Daily core (GARCH-$t$ $+$ EVT) & $1.500$ & $+6.2$ & $1.774$ & $+4.6$ \\
Realized scale $+$ residual quantile & $1.445$ & --- & $1.715$ & --- \\
\quad $+$ EVT tail & $1.445$ & $-0.1$ & $1.716$ & $+1.3$ \\
\quad $+$ pooled shape learner & $1.442$ & $-0.8$ & $1.708$ & $-1.4$ \\
\quad $+$ shape $+$ EVT (realized-hybrid) & $1.447$ & $+0.7$ & $1.717$ & $+0.5$ \\""")

O("O29 tab:rgarch note: ES convention",
r"""\item All five rows are computed in one pipeline on the same test rows, with ES the
$20$-node integral of each row's own quantile curve.""",
r"""\item All five rows are computed in one pipeline on the same test rows. ES is exact wherever a closed
form exists: the two GPD-tailed rows use the closed-form tail integral and the unconditional row the
empirical tail mean. The shape row is a body-only forecast with no tail model below its lowest fitted
level, so its ES keeps a flat extension there and is a lower bound on severity; the realized-hybrid row
uses the converged integral of its min-envelope curve.""")

def main():
    dry = "--dry" in sys.argv
    T, problems = {}, []
    for files, eds in ((FILES, EDITS), (OA, OAED)):
        for f in files:
            t = T.get(f) or io.open(f, encoding="utf-8").read(); T[f] = t
            for name, before, _ in eds:
                c = t.count(before)
                if c != 1: problems.append(f"{f}: {name}: anchor found {c} times")
    if problems:
        print("NOT APPLIED:\n  " + "\n  ".join(problems)); sys.exit(1)
    if dry:
        print("dry run OK: %d main edits x %d files, %d OA edits x %d files" % (len(EDITS), len(FILES), len(OAED), len(OA))); return
    for files, eds in ((FILES, EDITS), (OA, OAED)):
        for f in files:
            for _, before, after in eds: T[f] = T[f].replace(before, after, 1)
    for f, t in T.items(): io.open(f, "w", encoding="utf-8", newline="\n").write(t)
    print("R59 applied: %d main edits x %d files, %d OA edits x %d files" % (len(EDITS), len(FILES), len(OAED), len(OA)))

if __name__ == "__main__":
    main()
