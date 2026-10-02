# Change ledger R57 (FZ0 claim scope, redundant and misattributed sources, stale references)

Applied by `apply_R57.py` on top of R54 and R55 (exact unique replacements, every anchor checked before anything is written) to `submission/paper_A_jfec.tex`, `preprint/paper_A_ssrn.tex`, both online appendices and both `refs_v3.bib`. Tested on R53 + R54 + R55: all anchors match once, both documents compile with no undefined citations or references, abstract 97 words. No reported number changes except C2, which keeps one of two quoted runs.

## A1 abstract (main text)

**Why.** Plain Beta-t-EGARCH has lower FZ0 than the GARCH-t engine at both levels (DM 0.88 at 1%, 2.53 at 2.5%; robust_engine_results.json, now public). 'Industry-standard' ties the claim to the models banks run, which it holds for. One word, abstract stays at 97.

**Before.**

```
it is lowest among standard benchmarks at 1\% and never significantly beaten at 2.5\%.
```

**After.**

```
it is lowest among industry-standard benchmarks at 1\% and never significantly beaten at 2.5\%.
```

## A2 5.1 claim and set (main text)

**Why.** 'Every benchmark' is false once Beta-t-EGARCH is reported. The set named here is the one Section 1.2 introduces as 'the models banks run' and the intro calls 'the methods in standard use across the industry' (with Perignon and Smith on historical simulation), so the abstract's term has a definition and a defense already in the paper. GARCH-EVT is left out because Section 1.2 treats it as a classical academic method; the same paragraph still reports the engine beating it.

**Before.**

```
attains the lowest FZ0 of every benchmark at 1\% and is never significantly beaten at 2.5\%.
```

**After.**

```
attains the lowest FZ0 of the industry-standard benchmarks of Section~\ref{sec:background} (historical simulation, EWMA, GARCH-$t$, GJR-GARCH-skew-$t$ and FHS) at 1\% and is never significantly beaten by them at 2.5\%.
```

## A3 Taylor DM and Beta-t-EGARCH disclosure (main text)

**Why.** Refit numbers from the R57 disclosure run (split verdict: P4 and P5 confirmed). Stale: the same paragraph quotes the Taylor model at 2.5% as DM -0.1 (OA.11) and then 'DM 0.1 in that run'. The Beta-t-EGARCH result is disclosed where the ranking is stated, and tied to the paper's mechanism. A clause on annual refits is added after the pending refit run.

**Before.**

```
while tying it at 2.5\% (DM 0.1 in that run), the near-tie the fat-tailed benchmarks all show at 2.5\% where the shape edge is thin.
```

**After.**

```
while tying it at 2.5\%, the near-tie the fat-tailed benchmarks all show at 2.5\% where the shape edge is thin. A score-driven Beta-$t$-EGARCH \citep{harvey2013dynamic}, outside the industry-standard set, has lower FZ0 at both levels under the frozen fits, significantly at 2.5\% (DM 2.53). Under the annual refits of Section~\ref{sec:calendarleak} the two are within noise overall, with the accuracy layer slightly ahead (DM 1.2 and 0.9), while Beta-$t$-EGARCH stays significantly ahead in the top decile of the score (DM 3.7 and 5.2). Its bounded response to a shock removes the post-shock variance overstatement of Section~\ref{sec:frontier}, the frontier's own mechanism, and refitting does not.
```

## A4 5.1 summary sentence (main text)

**Why.** Same scope as A2.

**Before.**

```
The accuracy layer is thus one configuration that is never significantly beaten on FZ0 at either level,
```

**After.**

```
The accuracy layer is thus one configuration that is never significantly beaten on FZ0 by the industry-standard benchmarks at either level,
```

## A5 intro Taylor wording (main text)

**Why.** 'Keeps the lowest FZ0' is false at 2.5%, where Taylor's mean loss is slightly lower (DM -0.1); the sentence already says 'tying'. Present since R32.

**Before.**

```
It also keeps the lowest FZ0 against the FZ-estimated dynamic ES models of \citet{patton2019} and \citet{taylor2019}, each estimated per name by FZ0 minimization, winning at 1\% and tying Taylor at 2.5\%.
```

**After.**

```
It also beats the FZ-estimated dynamic ES models of \citet{patton2019} and \citet{taylor2019}, each estimated per name by FZ0 minimization, at 1\% and ties Taylor at 2.5\%.
```

## A6 intro scale caveat (main text)

**Why.** The paper's own decomposition (and the flat profile against Beta-t-EGARCH) says the frontier is mostly post-shock GARCH scale error, so 'the component the ARCH family already estimates well' needs the exception stated. Present since R32. Hansen-Lunde stays cited here (an earlier draft of R57 dropped it, which would have left the claim uncited).

**Before.**

```
once a leverage term is allowed \citep{hansenlunde2005}, and replaces the innovation law where it fails.
```

**After.**

```
once a leverage term is allowed \citep{hansenlunde2005}, except after large shocks (Section~\ref{sec:frontier}), and replaces the innovation law where it fails.
```

## A7 Beta-t-EGARCH as Stage 1, frozen-fit result (main text)

**Why.** The walk-forward header's rule: if P1 fails, the swap 'must be reported as a frozen-fit result only'. Numbers: robust_engine_results.json (frozen), exception_battery_results_holdout.json (holdout FZ0 DMs from the R56 harvest), walkforward_bteg_results.json and the R57 harvest (refit overall and top decile).

**Before.**

```
unlike the average pinball edge, which closes under the same refit.
```

**After.**

```
unlike the average pinball edge, which closes under the same refit. With Beta-$t$-EGARCH in place of GARCH-$t$ as Stage 1, the engine has lower FZ0 under the frozen fits (DM 3.31 and 4.55, and 3.09 and 3.48 on the 2000--2013 holdout). Under annual refits the two engines are within noise overall (DM $-0.31$ and $-0.01$), and the Beta-$t$-EGARCH engine is ahead in the top decile of the score (DM 5.37 and 7.25), so refitting does on average what a robust filter does but leaves the post-shock error in the states the score flags. Stage 1 stays GARCH-$t$ under the rule fixed before the refit run.
```

## A8 McNeil-Frey limitation (main text)

**Why.** P6 of the disclosure run failed: MF rejects at 1% under refits for every engine (garch_t 0.020, bip_t 0.015, bteg 0.014), as on the full panel (0.02 after the seed fix) and the frozen holdout (0.005). GARCH-t rejects too (Table OA.5); FHS passes (p = 0.07). Stated in the limitations before a referee finds it.

**Before.**

```
in credit and freight the accuracy edges sit alongside failed breach-independence tests.
```

**After.**

```
in credit and freight the accuracy edges sit alongside failed breach-independence tests. The McNeil--Frey exceedance-residual test rejects the 1\% ES on the full panel, on the 2000--2013 holdout and under annual refits, for every Stage-1 filter tried and for GARCH-$t$ itself; filtered historical simulation, which fits the empirical tail directly, is the one entrant that passes it on the full panel.
```

## B1 drop pricing-side sentence (R55 E4) (main text)

**Why.** An aside on option prices that nothing else in the paper uses. Andersen, Fusari and Todorov (2020) is kept for the response letter. Its bib entry stays and does not print.

**Before.**

```
diagnostic of shape alone. On the pricing side, index option prices also separate a left-tail factor from diffusive volatility \citep{bollerslevtodorov2011, andersen2020tail}.
```

**After.**

```
diagnostic of shape alone.
```

## B2 Todorov on the jump mechanism (main text)

**Why.** One Todorov citation where it supports the argument: high-frequency evidence that individual-stock jump tails are heavy, which is why one shock inflates GARCH variance. Bollerslev, Todorov and Li (2013), JoE 172(2), 307-324, checked against the Duke record.

**Before.**

```
\paragraph{Scale versus shape.} The top decile follows large
standardized residuals, and since
```

**After.**

```
\paragraph{Scale versus shape.} The top decile follows large
standardized residuals, typically price jumps, whose tails in individual stocks are heavy \citep{bollerslev2013jumptails}, and since
```

## B3 GBM citation (main text)

**Why.** Meinshausen (2006) is quantile regression forests, not gradient boosting. R54 M13 removed the sentence it belonged to and left it attached to GBM.

**Before.**

```
(GBM) \citep{friedman2001, meinshausen2006}
```

**After.**

```
(GBM) \citep{friedman2001}
```

## B5 repeated IQN citation (main text)

**Why.** Dabney et al. and Koenker-Bassett are cited two sentences earlier in the same subsection.

**Before.**

```
is a conditional quantile network \citep{dabney2018iqn, koenker1978}, so the forward
```

**After.**

```
is a conditional quantile network, so the forward
```

## B6 POT footnote citations (main text)

**Why.** The sentence the footnote hangs on cites Balkema-de Haan, Pickands and McNeil-Frey already.

**Before.**

```
\footnote{The peaks-over-threshold formulation \citep{balkema1974,pickands1975} uses tail data more efficiently than block maxima \citep{embrechts1997} and is standard in risk management \citep{mcneilfrey2000}.
```

**After.**

```
\footnote{The peaks-over-threshold formulation uses tail data more efficiently than block maxima \citep{embrechts1997} and is standard in risk management.
```

## C2 ten-day holdout, one run (main text)

**Why.** Two runs of the same quantity were quoted side by side; the one kept has its table (OA.13).

**Before.**

```
(direct edge $+0.49\%$, DM 0.07; the rerun behind Online Appendix Table~OA.13 gives $+0.52\%$, DM 0.09)
```

**After.**

```
(direct edge $+0.52\%$, DM 0.09)
```

## C3 OA ten-day earlier-run note (online appendix)

**Why.** Draft history; goes with C2.

**Before.**

```
 The main text's holdout figure ($+0.49\%$, DM 0.07) is an earlier run of the same purged pipeline with a different learner seed.
```

**After.**

```
(deleted)
```

## C4 OA electricity note (online appendix)

**Why.** Draft history added by R54 O05; the sweep reported is the corrected one.

**Before.**

```
 (iii)~In the 43-instrument sweep, an apparent electricity win in an earlier run reversed once the return transformation was corrected for near-zero prices; the sweep reported here uses the corrected transformation.
```

**After.**

```
(deleted)
```

## C5 OA heading and count (online appendix)

**Why.** Goes with C4.

**Before.**

```
\subsection*{Expected-shortfall integral, external transfer check, and an instrument-level correction}
Three items recorded for completeness.
```

**After.**

```
\subsection*{Expected-shortfall integral and an external transfer check}
Two items recorded for completeness.
```

## C6 OA em-dashes (ES backtests) (online appendix)

**Why.** Em-dash pair and an 'X, not Y' clause.

**Before.**

```
of equation~(5) of the paper --- the exact forecasts the joint loss
scores, not a simpler proxy --- and inference
```

**After.**

```
of equation~(5) of the paper, the forecasts the joint loss
scores, and inference
```

## C7 OA em-dashes (FHS) (online appendix)

**Why.** Em-dash pair.

**Before.**

```
McNeil--Frey exceedance residual FHS --- which fits the empirical residual tail
directly --- sits closest
```

**After.**

```
McNeil--Frey exceedance residual FHS, which fits the empirical residual tail
directly, sits closest
```

## C8 OA em-dash (table caption) (online appendix)

**Why.** Em-dash.

**Before.**

```
panel --- the same held-out rows as the full-panel FZ0 re-scoring;
```

**After.**

```
panel, the same held-out rows as the full-panel FZ0 re-scoring;
```

## Bibliography

Added: harvey2013dynamic (Econometric Society Monographs 52, CUP 2013), bollerslev2013jumptails (JoE 172(2)). Uncited after R57 and harmless: bollerslevtodorov2011, andersen2020tail, meinshausen2006, jiang2008gibbs, bissiri2016general (bibtex prints cited entries only).
## Pending: stale numbers that need a run or a decision first

1. **Seed-fixed regeneration** (R55 harvest §4). Printed numbers that move: §5.1 FRTB-notes sentence (Kupiec 82% → 81%, Christoffersen 89% → 88%, 97.5% pass rates 71%/79% → 72%/80%, date-clustered t 0.93 → 0.89 and 1.85 → 1.71, raw residual-hybrid t 4.2 → 4.6); §5.1 "82% and 89% pass rates"; the ES-backtest paragraph's McNeil–Frey p = 0.04 → 0.02; OA Table OA.5 engine row (Kupiec p 0.15 → 0.42 at 1%, Z2 −0.07 → −0.06 and −0.11 → −0.10, MF p 0.04 → 0.02 and 0.26 → 0.24) and the OA paragraph quoting −0.07 and −0.11. The Acerbi–Székely p at 2.5% crossing 0.05 (0.0394 → 0.0515) is reported as a consequence of the seed fix.
2. **tab:frtb and Figure OA.2 on the 200-name panel.** Figure OA.2's caption says 140 CRSP names and §5.1 calls the pooled Kupiec "155k-observation" inside a paragraph about the 200-name panel (221.6k). The per-name pass rates in item 1 change again when the 200-name file is used, so do item 2 and item 1 together.
3. **Stage 4 (conformal overlay).** §5.1 mixes two runs: static-shift breach 1.66% (fz_fullpanel; also OA lines 510, 621, 751) and 1.63% (fz_aci) two paragraphs apart; static DM −1.8 against −2.01; static Kupiec97.5 pass rate 48% against 47%; unshifted 71% against 72%. The harvest switches the overlay off; that rewrite replaces these sentences, so they are not patched here.
4. **Refit run**: done (R57 harvest). Folded into A3 and A7.
5. **scikit-learn version.** Done: 1.7.2 pinned in the README (R57 harvest).

## Already made (R54)

M01–M28 and O01–O05: duplicate intro sentences (cost, buy-side, GBC simulator), uncited standardized-approach clause, unused-method asides (Zumbach, quantile regression forests, IQN generation, Gibbs posterior, generative reading), five OA signposts merged, Section 2.5 and Lemma 3 removed, ES closed-form history moved to the OA, repeated ES-diagnostic paragraph and Perignon–Smith clause removed, stale coverage convention, unreported crypto-hourly claim. R57 B3, B5 and C4 finish three items R54 left half done. R54 M06 (Jiang and Tanner 2008; Bissiri et al. 2016 as the generalized-Bayes lineage of GBC) is kept: it is the only citation of Wenxin Jiang's work, and GBC sits in that tradition.

## Checked, no change

Cross-references: every \ref and \eqref resolves; every "Table/Figure OA.x" in the paper points at the intended OA float after R55's renumbering; the OA's hard-coded "Table 1/4", "Figure 2", "equation (5)", "Section 2/3/4" match the compiled paper. Repeated statements of the same claim (ES not elicitable, the ninety-percent bulk, +2.98%) are left as they are.
