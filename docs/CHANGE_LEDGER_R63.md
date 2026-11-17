# Change ledger R63: the reframe around where and why flexible tail models beat GARCH

Edits applied directly to `submission/paper_A_jfec.tex` and `submission/paper_A_jfec_online_appendix.tex` on top of
`a4b284d` (R61/R62). The preprint copy is regenerated from them by `code/paper/make_ssrn_build.py`. The pre-R63
sources are recoverable from `a4b284d`.

**Why.** An outside read of SSRN R32 (2026-11) argued that the paper's own decomposition undercuts its framing: the
score is mostly a post-shock indicator, a robust filter captures most of the edge, and "real-time misspecification
score" and "distribution-free tail beats GARCH" oversell the shape component. Most of the evidence was already in
the working build (R55 to R62): Table 2 (scale × shape), the robust-GARCH literature, Beta-t-EGARCH as benchmark and
as Stage 1, the score-gated scale. What had not moved was the headline. With Beta-t-EGARCH in the comparison the
paper cannot claim a better VaR/ES model, so R63 makes the decomposition the headline and the estimator the
instrument.

**The version on SSRN is R32.** It still says "lowest FZ0 at both regulatory levels", which is false at 2.5%
against GJR-skew-t and Taylor, and at both levels against Beta-t-EGARCH under frozen fits. R63 is meant to be
posted once, as a finished version.

## Sources

| claim | source | value |
|---|---|---|
| one residual carries the top-decile window | `results/paper/diagnostics/r63_mk63_anatomy_canon200.json` `top_d10.share1_median`, `frac_share1_gt_half` | 0.972; 0.9996 |
| its size in GARCH units | same, `by_decile.10.zraw_q25_med_q75` | 7.19 / 8.19 / 10.26 |
| its timing | same, `by_decile.10.age_q25_med_q75` | 16 / 31 / 47 days |
| score without it | same, `top_d10.mk_drop1_median` | 1.07 (decile 4 median 0.95, decile 5 median 1.43) |
| the other 62 residuals are compressed | same, `by_decile.*.sd_rest_q25_med_q75[1]` | 0.84 in decile 10; 0.91 to 0.93 in deciles 1 to 8 |
| bulk share of the largest term | same, `bulk_d1to8.share1_median` | 0.386 |
| holdout top-decile shock | `..._holdout.json` `top_d10.zraw_median` | 5.67 |
| one 9.8-sigma draw among 62 normal ones | arithmetic, bias-corrected estimator of eq. (7) | excess kurtosis 21.3 (8.2 sigma alone: 15.2) |
| Beta-t-EGARCH beats the pooled body in the top decile | `bench_all_bteg_results.json` `pinball.mk63_decile_profile_vs_param_bteg.body.10` | −0.641%, DM −2.59 |
| and SAV-CAViaR | same, `.sav_caviar.10` | −0.783%, DM −3.70 |
| SAV-CAViaR below the top decile | `pinball.vs_garch_t.sav_caviar.bulk_mk63_d1to9` | +0.083%, DM 1.60 |
| accuracy layer vs EWMA, overall | `bench_all_results.json` `pinball.engine_vs.ewma.overall` | +1.448%, DM 8.22 (was printed +1.5) |
| accuracy layer DM range vs GARCH-t / EWMA / HS | `pinball.engine_vs.{garch_t,ewma,hs500}.overall.DM` | 4.6, 8.22, 7.92 (was printed 4.4 to 8.3) |

`tools/r63_mk63_anatomy.py` is read-only and wrote five predictions into its header before the first run. Graded:
P1 (top-decile median share ≥ 0.60) **held**, 0.972. P2 (median window-normalized size ≥ 6.5) **failed**, 6.08: the
shock inflates the window SD it is measured against. P3 (median score without it < 3) **held**, 1.07. P4 (bulk median
share < 0.30) **failed**, 0.386. P5 (median age 15 to 45) **held**, 31. Two fields were added after the first run with
no prediction attached and are labelled so in the header: `zraw` (size in GARCH units) and `sd_rest`. The `sd_rest`
field was added to explain why one 8.2-sigma shock (excess kurtosis 15) falls short of the decile median of 22.
The paper says "about eight conditional standard deviations", not the "ten" an earlier note proposed from the
arithmetic alone.

## M01 Title, abstract, keywords

- Title: *Semiparametric Value-at-Risk and Expected Shortfall with a Real-Time Misspecification Score* →
  *When Do Flexible Tail-Risk Models Beat GARCH? Post-Shock Scale Error, Residual Shape, and a Real-Time Score*.
  The online appendix title follows. The README and SSRN long abstract follow.
- Abstract rewritten around the two answers (where; why), 89 words (JFEC limit 100; R62's was 84 but led with the
  estimator). It no longer claims the estimator has the lowest joint loss.
- Keywords: drop *amortized estimation*; add *forecast evaluation* and *robust volatility* (six, the limit).

## M02 Introduction rewritten (1,889 → about 1,150 words)

Structure: question → where (frontier, +2.98%, SAV-CAViaR same profile, robustness list) → what a top-decile score
records (new anatomy numbers) → decomposition (bounded-news cap; Beta-t-EGARCH beats both flexible models) → what
survives (+0.27%, FZ0 at both levels, frozen and refit) → the estimator as instrument with its scorecard stated
narrowly (ties GJR-skew-t at both levels and Taylor at 2.5%; Beta-t-EGARCH lower under frozen fits) → monitor
reading and the score-gated scale → contribution.

Removed from the introduction: the three-class taxonomy paragraph and the FRTB/industry-practice paragraphs (both
covered in Section 1), the four-universe list, the Korea sentence, the cold-start paragraph (pending; see M08),
"lowest joint (VaR, ES) score … at both FRTB levels", "improves significantly on … GJR-skew-t" (true on pinball,
not on FZ0, and the reader was not told which), and the "main contribution is the misspecification frontier"
sentence.

## M03 Table 1: configurations and the claims each carries (new)

After Algorithm 1. Six rows: pooled body, accuracy layer, adaptive overlay, static shift, frozen returns-space
learner, robust-scale estimator. Each row says what was fixed before its test data were scored. Every entry is
taken from text already in the paper or a job header (`job_overlay_engine.py`, `job_walkforward_bteg.py`,
`job_scale_gate.py`, `docs/R62_SPLICE_PRECOMMIT.md`). Existing main tables renumber 1–5 → 2–6; the online
appendix's hard-coded "Table 1/4 of the paper" are renumbered to 2/5 (13 occurrences).

## M04 Section 2.4: the formal results compressed to one paragraph (475 → 190 words)

Lemma 1 (pinball regret) and Proposition 2 (tail wedge) move to the online appendix as Lemma OA.1 and Proposition
OA.1, with their statements alongside the proofs that were already there. The proposition's fixed-level remark goes
with them. The main-text equation numbers after (6) shift down by two; the appendix's "equation (9) of the paper"
(the score) becomes "(7)" (two occurrences). Proposition 1 (conformal) stays in the main text.

**Bookkeeping fix:** the proposition stated τ* ∈ (0, 1/2] and its proof concluded τ* ∈ (0, 1/2). Since
τ* = min(δ, 1/2) can equal 1/2, the half-closed interval is right; the proof now says so.

## M05 Section 4: what a top-decile score records (new paragraph) and the attribution across models

- New paragraph *What a top-decile score records* before *Scale versus shape*, from the anatomy table.
- *Scale versus shape* now opens from it ("Such shocks are typically price jumps …").
- The Beta-t-EGARCH sentence adds the pooled body (−0.64%, DM 2.6) and SAV-CAViaR (−0.78%, DM 3.7) and states that
  the attribution is not specific to one learner.
- *GARCH-EVT on the same rows* no longer calls the frontier "a statement about a state-conditioned residual shape".

## M06 Korea as illustration

"The Korea-2026 crash is the negative control" → "an illustration, not a test", with the reason (23 assets, one
correlated event, examined after the event). Online appendix: Figure OA caption, Table OA.1 Panel B row, Table
OA.4 note and the synthetic-truth discussion no longer say "falsification case" or "control".

## M07 Bookkeeping

- Table OA.1 Panel A, first row: EWMA +1.5 → **+1.4** (1.448), DM range 4.4–8.3 → **4.6–8.2**.
- Table OA.1 Panel A: "Top misspecification decile +2.98%" was the pooled body in a panel titled "the engine".
  The panel is now "main results (configuration in parentheses)"; the row gives the body (+2.98%, DM 10.5) and the
  accuracy layer (+2.48%, DM 10.1); the holdout row names the frozen returns-space learner; the FZ0 row is scoped to
  the industry-standard benchmarks and discloses Beta-t-EGARCH. New row: top-decile edges against Beta-t-EGARCH.
- Walk-forward paragraph: "keeps the lowest FZ0 at both levels" → "keeps a lower FZ0 than both refitted
  benchmarks", since the walk-forward compares only GARCH-t and pooled FHS.
- **Already fixed before R63:** the static-overlay FZ0 concession that the R32 reader saw as −1.8 (Section 5)
  against −2.0 (Table OA.1). R62 prints DM 2.6 in the text and 2.57 in Tables OA.6 and OA.11.
- The R32 reader's point on the first cell of Table OA.1 (+0.3 "should round to +0.2"): the source is 0.263, so
  +0.3 stands.

## M08 New online-appendix sections

- *What a top-decile score records*, Table OA.18 (`tab:anatomy`), both panels, with the graded predictions.
- *Top-decile edges against a robust reference*, Table OA.19 (`tab:attribution`): every flexible model against
  GARCH-t and against Beta-t-EGARCH on the same rows. Bodies generated by `tools/make_r63_oa_tables.py` from
  committed result files only.

## M09 Conclusion, first paragraph

Rewritten to the headline. It no longer says the estimator "improves on the accuracy of the market-risk models in
standard use … at calibration parity with the best of them". The qualifications paragraph is unchanged.

## M10 Build

`code/paper/make_ssrn_build.py` pointed at the GBC_Project layout (`paper/jfec/…`) and could not run in this
repository. It now reads `submission/` and writes `preprint/`.

## Not done in R63, and why

- **Length.** The journal build is 66 pages against a guideline of about 40. The cuts (ten-day and cross-asset
  material, the composite-score and ν-relative paragraphs, IQN detail, protocol) are R64, so that the reframe and the
  cuts can be reviewed separately.
- **A third flexible model on the robust scale** (IQN, or a deep quantile regression after Chronopoulos et al.).
  The two-model attribution rests on committed results; a third model would need a new run with predictions
  written first.
- **CAViaR under annual refits.** The attribution table is frozen fits. The estimator's top-decile loss to
  Beta-t-EGARCH holds under refits (FZ0 DM 3.7 and 5.1, Section 5.1); CAViaR's has not been rerun.
- Figure 1's holdout bars still match no committed result file (flagged in R61). OA.18's text quotes the figure's
  +1.12% and inherits that orphan.

## M11 Figure 2 caption

"The accuracy layer … attains the lowest FZ0 at 1% and is never significantly beaten at 2.5%" is now scoped to
"among the industry-standard benchmarks", and the caption states that Beta-t-EGARCH scores below the accuracy layer
at both levels under the frozen fits (DM 2.48 at 2.5%) and is within noise under annual refits. Source: the
Section 5.1 text it summarizes (R57).

## M12 Cold start: the claim reconciled against a peer proxy

The outside reader asked for a peer or sector proxy as the cold-start comparator. `job_coldstart_peers.py` had run
one, with predictions P1 to P4 written first; all four failed or held only vacuously, and its accuracy comparison was
never cited (only its 7.1% first-week breach rate was). Its learner (`peer_young_flex`, three day-one features) is not
the paper's amortized model. Two findings decided the rewrite:

1. **The paper's "listing age" was not listing age.** `job_amort_pit.py:48` sets `age = groupby('permno').cumcount()`,
   days since the name enters the 2014 panel. 345 of the 720 names start in January 2014, so about half the
   held-out names on the "+9% at 15 to 30 days" age curve were established stocks. The 91.7% and 99.0% figures are
   unaffected (they are not age statements), but the text now says what age means there.
2. **On genuine new listings** (`tools/r63_coldstart_vs_peer.py`, post hoc, no prediction attached, labelled as
   such; rows and reference rebuilt from `job_coldstart_peers.py`, per-bucket loss matching its JSON to five
   decimals), out of time, seven levels, positive = amortized wins:

| age (days since listing) | full amortized model vs peer proxy | Gaussian own-history vs peer proxy |
|---|---|---|
| 2–5 | +5.5%, DM 4.9 | −16.6%, DM −2.2 |
| 6–14 | +12.3%, DM 7.3 | +0.7%, DM 1.2 |
| 15–60 | +11.0%, DM 14.3 | +3.6%, DM −0.1 |
| 61–125 | +13.6%, DM 18.3 | +6.7%, DM 3.1 |
| 126–250 | +13.2%, DM 20.1 | +6.2%, DM −0.1 |

The full model beats the Gaussian own-history benchmark by 7.6% at 15–60 (DM 6.8), which reproduces the paper's
claim on a proper split. The look-ahead-free **characteristics-only** form loses to the peer proxy in every bucket
(`coldstart_amort_results.json` `by_age.*.mean_pinball`, `amort_chars_pit` against `peer_young`: −2.5%, −1.2%,
−2.1%, −1.5%); the published `amort_chars` wins only because three of its five characteristics use future data
(that file's own `lookahead_note`). Pooled own-history rivals beat the proxy by 10.6–12.8% (eleven levels), close
to the full model's 12.7–15.7%, and the full model was not run head to head against them.

Text changes: Section 6's cold-start paragraph rewritten as *New listings* with the numbers above, the day-one form
described as available but not more accurate, and transfer called a secondary result; §2.3 says what "age" means
in the ablation and no longer says the mapping "works, at cold-start in particular"; §1.3 and the conclusion follow;
Table OA.1's cold-start row and the OA monitoring procedure follow.
