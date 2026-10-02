# Change ledger R54 (focus pass: research artifacts and duplicates removed from the body)

Applied by an exact-unique-replacement script to `submission/paper_A_jfec.tex` and `preprint/paper_A_ssrn.tex` (identical text) and to both online-appendix sources; 33 edits. No reported number changes. Lemma 3 (generative sampler) and Section 2.5 are removed, so the formal results are now Lemma 1 and Propositions 1 and 2, and the online-appendix proofs section counts three. Everything removed is recorded in `docs/DEVELOPMENT_HISTORY.md`.

## M01 intro duplicate cost sentence

**Why.** Intro paragraph 1 and paragraph 4 both said the error is costly in both directions; paragraph 4 carries the FRTB mechanism, so paragraph 1 keeps only the hook.

**Before.**

```
a fixed parametric tail is most likely to be wrong. The error is costly in
both directions. A forecast that runs high ties up capital against risk
that never arrives, and one that runs low leaves too little against the
loss that does. Deciding when
```

**After.**

```
a fixed parametric tail is most likely to be wrong. Deciding when
```

## M02 intro buy-side duplicate

**Why.** The buy-side scope sentence appears again, with specifics, three paragraphs later.

**Before.**

```
complexity is therefore a practical question. The question also applies outside Basel-regulated bank trading desks, including to asset managers and hedge funds that forecast conditional tail risk.
```

**After.**

```
complexity is therefore a practical question.
```

## M03 intro GBC simulator sentences

**Why.** GBC's simulator-to-data reading is explained in Section 1.3; the intro only needs to list the learner.

**Before.**

```
\citep{polson2023gbc}. GBC is a simulation-based posterior method. It
becomes a conditional quantile learner once its forward simulator is
replaced by observed state and return pairs, as
Section~\ref{sec:background} does. These learners
```

**After.**

```
\citep{polson2023gbc}. These learners
```

## M04 intro uncited standardized-approach clause

**Why.** The claim that banks moved back to the standardized approach had no citation.

**Before.**

```
Under FRTB a bank chooses between a prescribed standardized formula and its own internal model, and many banks have moved back toward the standardized approach as internal models became costly to maintain. The buy side
```

**After.**

```
Under FRTB a bank chooses between a prescribed standardized formula and its own internal model. The buy side
```

## M05 Zumbach not-run aside

**Why.** A model that was not run has no place in the comparison-set sentence.

**Before.**

```
\citep{riskmetrics1996} (the long-memory successor in \citealp{zumbach2007} is not run here), GARCH(1,1)
```

**After.**

```
\citep{riskmetrics1996}, GARCH(1,1)
```

## M06 fold generalized-Bayes citations into the GBC sentence

**Why.** Keeps the generalized-Bayes lineage (Jiang and Tanner; Bissiri et al.) as related literature rather than as a method the paper declines to use.

**Before.**

```
the architecture at the core of GBC \citep{polson2023gbc, nareklishvili2025gqbp}. In the IQN
```

**After.**

```
the architecture at the core of GBC \citep{polson2023gbc, nareklishvili2025gqbp}, itself in the generalized-Bayes tradition of \citet{jiang2008gibbs} and \citet{bissiri2016general}. In the IQN
```

## M07 §1.3 generative reading and Gibbs sentence

**Why.** The generative reading of the IQN and the Gibbs-posterior sentence described capabilities no result uses; the cross-reference now points at the paragraph that records the differences from GBC.

**Before.**

```
Read as ERM on observed pairs, GBC's quantile generator coincides
with a conditional quantile network trained on data
\citep{dabney2018iqn, koenker1978}: the forward simulator is not needed,
and the same object applies to observational time series
(Section~\ref{sec:polsondelta}). The GBM is the shape model used
throughout, and the IQN is retained as a benchmark. Its generative reading (draw $\tau \sim
U(0,1)$, output $Q_\phi(\tau \mid s)$) is a capability we note
without using, since sampling and posterior uncertainty are outside
this paper's scope. Pinball ERM can be read as the mode of a Gibbs posterior built from the loss in place of a likelihood \citep{jiang2008gibbs, bissiri2016general}; no result below uses that posterior. Distribution-free
```

**After.**

```
Trained on observed state and residual pairs, GBC's quantile generator is a conditional quantile network \citep{dabney2018iqn, koenker1978}, so the forward simulator of the original method is not needed and the same object applies to observational time series; Section~\ref{sec:gbcmethod} records how the downside-risk target changes it. The GBM is the shape model used throughout, and the IQN is retained as a benchmark. Distribution-free
```

## M08 §2 opener: merge OA signposts

**Why.** Five separate pointers to the Online Appendix (glossary, schematic, Algorithm OA.1, Algorithm OA.2, full pipeline) merged into one sentence at the head of Section 2.

**Before.**

```
Each estimator below can be reimplemented from the formulas and
algorithm boxes given. A glossary of acronyms opens the Online
Appendix.
```

**After.**

```
Each estimator below can be reimplemented from the formulas and
algorithm boxes given. The Online Appendix opens with a glossary of
acronyms and a schematic of the full pipeline, and states the amortized
estimation loop and the monitoring procedure as Algorithms OA.1 and
OA.2.
```

## M09 §2.1 von Neumann sentence

**Why.** The inverse-CDF sampling representation was never used; the sentence now only fixes the notation q_phi and H_phi that later sections need.

**Before.**

```
A quantile level $\tau \sim \lambda$ is drawn from a base distribution $\lambda$. The learned generator $H_\phi(s,\tau)$ maps (state, level) to a quantile in the sense of \citet{polson2023gbc}: $z \overset{D}{=} H(s,\tau)$ with $\tau\sim U(0,1)$ is the inverse-CDF (von Neumann) representation of sampling.
```

**After.**

```
A quantile level $\tau \sim \lambda$ is drawn from a base distribution $\lambda$. The shape learners of Section~\ref{sec:gbcmethod} estimate the map from (state, level) to the residual quantile, written $q_\phi(\tau\mid s)$, or $H_\phi(s,\tau)$ for the network.
```

## M10 §2.2 schematic signpost

**Why.** Signpost merged into the Section 2 opener (M08).

**Before.**

```
The paragraphs below define each stage of the estimator; a schematic of the full pipeline is in the Online Appendix.
```

**After.**

```
(deleted)
```

## M11 §2.2 ES closed-form history

**Why.** The 'earlier convention' comparison is draft history; the body keeps the statement that ES is the numerical integral, and the OA carries the 1.849 vs 1.851 robustness figures (O05).

**Before.**

```
Computing ES$_\alpha$ as the numerical integral of \eqref{eq:hybrid} rather than by the GPD closed form of \citet{mcneilfrey2000}, an earlier convention, leaves each comparison in place. At 2.5\%, FZ0 is 1.849 against 1.851, the integral marginally better, DM 5.3, and the estimator-over-GARCH DM is 4.8 and 5.1 at the 1\% and 2.5\% levels. The splice level is immaterial ($p_0\in\{1.5\%,2.5\%,5\%\}$: DM 4.6--4.8).
```

**After.**

```
ES$_\alpha$ is computed as the numerical integral of \eqref{eq:hybrid} rather than by the GPD closed form of \citet{mcneilfrey2000}; the Online Appendix reports that the two agree to the third decimal on FZ0 and leave each comparison in place. The splice level is immaterial ($p_0\in\{1.5\%,2.5\%,5\%\}$: DM 4.6--4.8).
```

## M12 §2.3 opener: draws sentence

**Why.** The trees carry every reported forecast; 'when draws are needed' described an unused capability, and the OA.3 pointer was to the per-decile action table, not to estimator choice.

**Before.**

```
Two estimators realize $q_\phi(\tau\mid s)$ in \eqref{eq:shapeERM}. We
use trees for point accuracy and the network when draws are needed
(Online Appendix Table~OA.3).
```

**After.**

```
Two estimators realize $q_\phi(\tau\mid s)$ in \eqref{eq:shapeERM}. The
trees carry every reported forecast; the network, the architecture at
the core of GBC, is reported as a benchmark.
```

## M13 quantile regression forests aside

**Why.** Method not used.

**Before.**

```
Quantile regression forests \citep{meinshausen2006} are the other tree-based route and are not used here. Each new tree
```

**After.**

```
Each new tree
```

## M14 IQN role-is-generation sentences

**Why.** The network's generative role is not used; the pointer to Estimator choice stays.

**Before.**

```
The network's role is generation rather than
point accuracy: with $\tau\sim U(0,1)$,
$z^{*} = H_\phi(s,\tau)$ draws from the fitted conditional law by
inverse transform. The trade-off between the two estimators is set out under \emph{Estimator choice} below.
```

**After.**

```
The trade-off between the two estimators is set out under \emph{Estimator choice} below.
```

## M15 amortization generative-Bayes sentence

**Why.** The generative-Bayes phrasing restated the cold-start point made two sentences later.

**Before.**

```
pairs of hundreds of names. In generative-Bayes terms, $H$ is an
amortized map evaluated at any new $(s,\tau)$ without per-asset
refitting. It works, at cold-start in particular, because the mapping is estimated across names instead of within one. A per-asset
```

**After.**

```
pairs of hundreds of names, and it works, at cold-start in particular, because the mapping is estimated across names instead of within one. A per-asset
```

## M16 own-history blends duplicate

**Why.** Stated again, with the numbers, in the next paragraph.

**Before.**

```
$s_t$ and the forecast tracks its own dynamics automatically. For the same reason the explicit own-history blends add nothing: the state vector already carries the name's own recent information.
```

**After.**

```
$s_t$ and the forecast tracks its own dynamics automatically.
```

## M17 M5 clause

**Why.** The M5 result is outside the paper's object; moved to the OA (O05).

**Before.**

```
Two external checks: on held-out names the
transfer win rate over own-history benchmarks is 59--79\%, and the same
pipeline scores a leakage-safe weighted scaled pinball of 0.269
(benchmark tier) on the M5 competition data, outside finance
altogether.
```

**After.**

```
On held-out names the transfer win rate over own-history benchmarks is
59--79\%.
```

## M18 estimator choice: drop the draws half, absorb §2.5

**Why.** Estimator choice loses the scenario-generation half; the five differences from GBC that were Section 2.5 become one paragraph here, next to the stages they refer to, so the subsection and its only incoming reference go.

**Before.**

```
\noindent\emph{Estimator choice by forecasting objective.} The two estimators minimize the same loss over different index sets. The trees solve $\min \sum_{k}\sum_{i}\rho_{\tau_k}(\widehat{z}_i - q_k(s_i))$ on a fixed grid $\{\tau_k\}$, one model per level, while the network solves $\min_\phi \sum_i \mathbb{E}_{\tau\sim\lambda}\, \rho_\tau(\widehat{z}_i - H_\phi(s_i,\tau))$, one model for the continuum. The appropriate estimator depends on whether the application requires fixed quantile levels or the full conditional quantile curve.
At fixed regulatory levels (VaR/ES
reporting, limits, capital) the trees are preferable: best point accuracy, no
GPU, no sampling. Where the application consumes draws (scenario generation, portfolio aggregation by simulation), the network is preferable. It learns one continuous conditional quantile map, so draws come by inverse transform with no grid interpolation, at a measured cost of $\sim$0.75\% in point accuracy. A GARCH-$t$ or FHS model also simulates, from its fixed innovation law, whereas a tree grid requires monotone interpolation first. If both are needed,
run both from the same pooled residual panel. They share each upstream
and downstream stage of \eqref{eq:hybrid}, so swapping one for the
other changes a single component of the pipeline.

\noindent The amortized estimation loop is stated as Online Appendix Algorithm OA.1.
```

**After.**

```
\noindent\emph{Estimator choice.} The two estimators minimize the same loss over different index sets. The trees solve $\min \sum_{k}\sum_{i}\rho_{\tau_k}(\widehat{z}_i - q_k(s_i))$ on a fixed grid $\{\tau_k\}$, one model per level, while the network solves $\min_\phi \sum_i \mathbb{E}_{\tau\sim\lambda}\, \rho_\tau(\widehat{z}_i - H_\phi(s_i,\tau))$, one model for the continuum. At the fixed regulatory levels the trees are preferable: best point accuracy and no GPU, with the grid made monotone by rearrangement. The network learns one continuous conditional quantile map at a measured cost of $\sim$0.75\% in point accuracy. The two share each upstream and downstream stage of \eqref{eq:hybrid}, so swapping one for the other changes a single component of the pipeline.

\noindent\emph{Relation to generative Bayesian computation.} Relative to the generative Bayesian computation of \citet{polson2023gbc}, the downside-risk target changes the estimator in five ways, each introduced above. The sampling measure $\lambda$ is tail-weighted rather than uniform, because uniform sampling starves the extreme levels that downside risk depends on. The object modeled is the GARCH-standardized residual rather than the raw return, so that the conditional-scale dynamics the parametric model already estimates well are kept. Beyond the training support, extrapolation is handed to the GPD tail \eqref{eq:evt} rather than to the learner. The conformal stage adds the coverage statement of Proposition~\ref{prop:conformal}. And the amortization runs across the real cross-section of hundreds of assets instead of across simulations of a single model, which is what produces the cold-start forecasts and transfer results. When such machinery improves on the parametric filter is not a question generative computation addresses, and the score of Section~\ref{sec:score} is built to answer it.
```

## M19 Lemma 3 and FHS-nesting repeat

**Why.** Lemma 3 (sampler validity) was never referenced; the FHS-nesting sentence repeats 'Industry models as special cases'; Section 2.5 is absorbed by M18.

**Before.**

```
\begin{lemma}[validity of the generative sampler (standard)]\label{lem:sampler}
If $\tau \mapsto \widetilde{Q}^{z}_t(\tau) = Q^{z}_t(\tau)+c_\tau$ is nondecreasing and
left-continuous (the rearrangement step guarantees this for $Q^{z}_t$; a level-dependent shift $c_\tau$ requires rearranging the shifted curve again) and
$\tau \sim U(0,1)$, then $Z^* = \widetilde{Q}^{z}_t(\tau)$ has
quantile function exactly $\widetilde{Q}^{z}_t(\cdot)$.
\end{lemma}
\begin{proof} In the Online Appendix. \end{proof}

\medskip\noindent Because the flexible class contains FHS as the constant-in-state special case \citep{baroneadesi1999fhs}, empirical risk minimization attains in-sample pinball risk no worse than FHS, and the out-of-sample comparisons of Section~\ref{sec:frtb} test whether that dominance survives estimation noise; CAViaR \citep{engle2004caviar}, outside this nesting, is the one rival the estimator only ties.

% moved to online appendix (OA Section 1): the two-regime toy example
\subsection{Comparison with generative Bayesian computation}\label{sec:polsondelta}

Relative to the generative Bayesian computation of
\citet{polson2023gbc}, the downside-risk target changes the estimator in five ways. The sampling measure is tail-weighted, $\lambda = \tfrac12 U(0,1) + \tfrac12 \mathrm{Beta}(0.3,0.3)$ in place of $\tau\sim U(0,1)$, because uniform sampling starves the extreme levels that downside risk depends on. The object modeled is the GARCH-standardized residual rather than the raw return, so that the conditional-scale dynamics the parametric model already estimates well are kept. Beyond the training support, extrapolation is handed to the GPD tail \eqref{eq:evt} spliced at $p_0$, so that it is governed by extreme-value theory and not by the network. The conformal stage adds the exchangeability-based coverage statement of Proposition~\ref{prop:conformal}. Finally, the amortization runs across the real cross-section of hundreds of assets instead of across simulations of a single model, which is what produces the cold-start forecasts and transfer results of Section~\ref{sec:gbcmethod}. The question changes as well. When such machinery improves on the parametric filter is not something generative computation addresses, and the score of the next subsection is built to answer it.
```

**After.**

```
(deleted)
```

## M20 OA.2 signpost

**Why.** Signpost merged (M08).

**Before.**

```
\noindent The monitoring procedure is stated as Online Appendix Algorithm OA.2.
```

**After.**

```
(deleted)
```

## M21 stale coverage convention

**Why.** No 0.05 level or credible interval exists in the paper; the convention now describes the breach rates that are reported.

**Before.**

```
Coverage
numbers are realized frequencies against a stated nominal (0.05 target:
0.056 is slight under-coverage of risk, 0.032 over-conservatism). For
credible-interval coverage, closer to nominal is better in both
directions. Correlations
```

**After.**

```
Breach
rates are realized exception frequencies against the stated nominal
level, so a rate above nominal is under-coverage of risk and a rate
below it is conservative. Correlations
```

## M22 crypto hourly unreported result

**Why.** An unreported hourly result should not be asserted.

**Before.**

```
A frequency dimension is left to
future work: the five daily crypto series show no edge, while at hourly
frequency the picture inverts.
```

**After.**

```
The five daily crypto series show no edge; a frequency dimension is
left to future work.
```

## M23 Table 3 note: fold in the entrant detail

**Why.** The body paragraph after Table 4 (M25) is removed, so its entrant-level detail moves into the Table 3 note it duplicated.

**Before.**

```
each fat-tailed entrant exceeds its own-tail comparator by 9--22\%.
```

**After.**

```
each fat-tailed entrant exceeds its own-tail comparator by 9\% (raw
residual-hybrid, rolling FHS) to 22\% (pooled FHS).
```

## M24 second HS most-common clause

**Why.** Second statement of the Perignon-Smith fact; the first is in the introduction.

**Before.**

```
Historical simulation, by far the most common method in bank
disclosures \citep{perignonsmith2010}, and EWMA do not survive
```

**After.**

```
Historical simulation and EWMA do not survive
```

## M25 ES-diagnostic paragraph after Table 4

**Why.** Repeats the Table 3 note.

**Before.**

```
The ES columns of Table~\ref{tab:frtb} are diagnostic only. Under the exact tail integral, each fat-tailed entrant's predicted ES exceeds the realized mean below its own VaR, by 9\% (raw residual-hybrid, rolling FHS) to 22\% (pooled FHS). The comparator conditions on each model's own breach set. The joint FZ0 score below is the ES ranking used in this paper.
```

**After.**

```
(deleted)
```

## M26 single-start clause

**Why.** The earlier attempt's instability is draft history; the three-start convergence statement stays.

**Before.**

```
and under a three-start optimizer both converge for all $200$ names, so the single-start instability of an earlier attempt does not arise.
```

**After.**

```
and under a three-start optimizer both converge for all $200$ names.
```

## M27 electricity reversal

**Why.** The earlier electricity result was never public; the correction moves to the OA (O05).

**Before.**

```
An apparent electricity win reversed once the return transformation was corrected for near-zero prices. Instrument-level results
```

**After.**

```
Instrument-level results
```

## M28 §5 pipeline signpost

**Why.** Signpost merged (M08).

**Before.**

```
\noindent The full estimation and forecasting pipeline is stated in the Online Appendix.
```

**After.**

```
(deleted)
```

## O01 scenario generation paragraph

**Why.** Scenario generation depended on the removed sampler lemma.

**Before.**

```
\section{Trading implications and scenario generation}
```

**After.**

```
\section{Trading implications}
```

## O02 scenario generation body

**Why.** Same as O01.

**Before.**

```
\textbf{Scenario generation.} Because the shape stage carries an exact
inverse-CDF sampler, calibrating intractable scenario generators
(rough volatility, Hawkes, jump-diffusions) for CCAR/ICAAP-style
stress design is a natural extension; it is not developed here.
```

**After.**

```
(deleted)
```

## O03 proofs header count

**Why.** Three results remain (Lemma 1, Propositions 1 and 2), and Proposition 1 sits in Section 2.2.

**Before.**

```
The four formal results of Section~2.4 of the paper are stated with
proof sketches in the body;
```

**After.**

```
The three formal results of Sections~2.2 and~2.4 of the paper are stated in the body;
```

## O04 Lemma 2 proof

**Why.** Proof of the removed lemma.

**Before.**

```
\subsection*{Proof of Lemma~2 of the paper (generative sampler)}
Let $\widetilde{q}(\cdot \mid s)$ be nondecreasing and left-continuous
on $(0,1)$ and $\tau \sim U(0,1)$, and set
$Z^*=\widetilde q(\tau\mid s)$ and
$G(x)=\sup\{t\in(0,1):\widetilde q(t\mid s)\le x\}$ (with
$\sup\emptyset=0$). Monotonicity makes
$\{t:\widetilde q(t\mid s)\le x\}$ an interval with left endpoint $0$,
and left-continuity ensures the supremum is attained when the set is
nonempty, so the set equals $(0,G(x)]$; hence
$\{Z^*\le x\}=\{\tau\le G(x)\}$ and
$\Pr(Z^*\le x)=G(x)$, i.e.\ $Z^*$ has CDF $G$. Because $G$ so defined
is the upper generalized inverse of $\widetilde q$, and a
left-continuous nondecreasing $\widetilde q$ is in turn the lower
generalized inverse of $G$, the quantile function of $Z^*$ is exactly
$\widetilde q(\cdot\mid s)$. When $\widetilde q$ is an estimated curve,
the argument applies verbatim to the estimate: the sampler is exact for
the \emph{fitted} conditional law, which is the claim. $\square$
```

**After.**

```
(deleted)
```

## O05 ES-integral robustness, M5 check, electricity note (new subsection before Frontier robustness)

**Why.** Receives the ES-integral robustness figures (M11), the M5 check (M17), and the electricity correction (M27).

**Before.**

```
\section{Frontier robustness: calendar overlap, family-wise error, and the universe rule}
```

**After.**

```
\subsection*{Expected-shortfall integral, external transfer check, and an instrument-level correction}
Three items recorded for completeness. (i)~Computing ES$_\alpha$ as the numerical integral of the min-envelope curve rather than by the GPD closed form of \citet{mcneilfrey2000} leaves each comparison in place: at 2.5\% the accuracy layer's FZ0 is 1.849 under the integral against 1.851 under the closed form (the integral marginally better, DM 5.3), and the estimator-over-GARCH-$t$ DM is 4.8 and 5.1 at the 1\% and 2.5\% levels under either convention. (ii)~As a transfer check outside finance, the same amortized pipeline scores a leakage-safe weighted scaled pinball of 0.269 (benchmark tier) on the M5 competition data. (iii)~In the 43-instrument sweep, an apparent electricity win in an earlier run reversed once the return transformation was corrected for near-zero prices; the sweep reported here uses the corrected transformation.

\section{Frontier robustness: calendar overlap, family-wise error, and the universe rule}
```
