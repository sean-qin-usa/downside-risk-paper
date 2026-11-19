# Revision worklist

What is left to do on the paper, by theme. Status: OPEN, RUNNING, DONE, DECIDE (needs the author's call). Evidence for
each DONE item is in the file named beside it. Nothing is cut from the manuscript until the writing-out pass is done;
cutting is a separate, later pass.

Last updated 2026-11-19.

## 0. Where the paper stands

The manuscript in `submission/` is revision R67. The results produced since then (`docs/SHOCK_RESPONSE_RESULTS.md`,
`docs/PANELS_PRECOMMIT.md`, `results/paper/diagnostics/r69` to `r75`) are not in the text yet. They change what the paper
is about, so the next step is a rewrite around them, not an edit of R67.

**The paper the evidence now supports.** One question: when a flexible tail-risk model beats GARCH on stocks, what has
it learned? The answer in five steps, each with its own table:

1. **Where.** The gain sits on the days after one large shock: 8 to 9% of tail loss at the 1% and 2.5% levels on the
   tenth of stock-days with the highest score, against 1 to 2% elsewhere.
2. **Why.** GARCH(1,1) carries a large shock into its variance for as long as its persistence parameter dictates. On
   stocks only about a tenth of that extra variance is real (lambda near 0.1). The closed-form response of the
   GARCH recursion predicts GARCH-t's breach rate by shock age and by name with no free parameter, and with the
   pre-test lambda it predicts the size of the flexible models' gain panel by panel.
3. **What the shocks are.** Three quarters are earnings announcements, and lambda is low for any shock that is the
   firm's own. After market-wide shocks lambda is near 1 or above, which is why nothing beats GARCH-t on equity ETFs.
4. **The other half of the same fact.** On announcement dates and the day after, 3% of days, the 2.5% VaR of every
   model is breached 13 to 15% of the time. A calendar is worth 5 to 8% of tail loss to every model at 2.5% and 9 to
   15% at 1%, two to four times the gap between GARCH-t and the best flexible model.
5. **What is left.** With the calendar in every model the flexible models beat GARCH-t by 1 to 4%, and a GARCH-t fitted
   with announcement days masked by at most about 1% (nothing significant at the 1% level on the CRSP panel).

Evidence for steps 4 and 5 and for the origin split is in `docs/EARNINGS_CALENDAR_RESULTS.md`. DECIDE: whether steps 4
and 5 are the paper's main result or a section of it (see 1.12).

## 1. Framing and contribution

| ID | Point | Status | Action |
|---|---|---|---|
| 1.1 | Abstract and introduction must state the question, the answer and the numbers | OPEN | Rewrite: one question, the four steps above, numbers at 1% and 2.5% |
| 1.2 | The contribution is a finding about flexible models, not a new estimator | OPEN | Introduction paragraph listing the contributions as findings; the estimator is the instrument |
| 1.3 | The scale-versus-shape decomposition must be the first result a reader meets | OPEN | Its own table in the main text, at 1% and 2.5% |
| 1.4 | Cite each ancestor of the estimator at first use (FHS, GARCH-EVT, quantile learners, pooling) | partly DONE (R65) | Sweep again after the rewrite |
| 1.5 | One specification per claim, named every time | partly DONE (configuration table, R63) | Reduce to the two-by-two design: standard or robust scale, Student-t or learned tail |
| 1.6 | Economic size of the effect | DONE (`r69_tail_levels.json`, `r71/`) | GARCH-t's VaR is about twice as wide as the corrected one in the first week; breach rate 0.4% against a nominal 2.5% |
| 1.7 | The attribution must hold for more than one flexible model | DONE (`r70_neural_quantile.json`, panels) | Pooled trees, SAV-CAViaR and a returns-space neural network |
| 1.8 | Scope across asset classes | DONE (`docs/PANELS_PRECOMMIT.md`, `r71/`) | State as a result: lambda by asset class, one table |
| 1.9 | Prior work on outliers and announcements in GARCH must be credited | OPEN | Franses and Ghijsels (1999), Carnero, Pena and Ruiz (2007), Grane and Veiga (2014), Muler and Yohai (2008), Harvey (2013), Andersen, Bollerslev and Diebold (2007); add Jones, Lamont and Lumsdaine (1998) on announcement shocks that do not persist, and the earnings-announcement volatility literature. Say exactly what is new: the attribution of flexible models' VaR gains, lambda measured across asset classes, the closed-form test, the earnings share |
| 1.10 | Title | DECIDE | Keep the R63 question title or shorten |
| 1.12 | The earnings calendar | DECIDE | Steps 4 and 5 above change the headline from "flexible models discard one-off shocks" to "single-stock VaR comparisons are mostly about a missing calendar". Decide which leads. Before either: 7.7, 7.8, 7.9 |
| 1.11 | Structure | DECIDE | Proposed: introduction; framework (location, scale, shape; the two-by-two design; model table); data and evaluation; where; why (closed form, lambda, earnings); other panels; conclusion with limitations |

## 2. Evaluation design

| ID | Point | Status | Action |
|---|---|---|---|
| 2.1 | Main criteria: 1% and 2.5% quantile loss and the joint VaR and ES loss; the eleven-level average second | DONE in results (`r69`, `r72`), text OPEN | Lead every table with the two levels |
| 2.2 | State the quantile grid and weights for every comparison | OPEN | One sentence per table |
| 2.3 | GARCH-EVT, FHS and the pooled model with and without the tail on the same rows and filter | DONE (appendix; `r75` shows EVT and FHS share GARCH-t's conditional miscoverage) | Move to the main text |
| 2.4 | Conformal adjustment reported separately from the tail model | DONE | Consider moving all conformal material to the appendix |
| 2.5 | A robust filter as a primary benchmark | DONE | Row in the main table |
| 2.6 | One paragraph on which family each multiplicity adjustment controls | OPEN | |
| 2.7 | No test statistic before the test is defined | OPEN | Reorder: design, then tests, then results |
| 2.8 | Name the HAC estimator at first use (Newey-West, Bartlett kernel, 10 lags, per-date series) | OPEN | |
| 2.9 | Define the "frontier" with an equation: the conditioning variable, the bins, the loss difference | OPEN | With D as the conditioning variable this becomes one regression (`r75`) |
| 2.10 | Sample description: how the 200 names were chosen | facts found, text OPEN | First 200 in sort order of 335 complete-history names (74 large caps, 126 small caps) from a 720-name CRSP pull; all survive to 2024. Say so, and lean on the public-data panels for replication |
| 2.11 | Holdout selection rule differs from the design panel's | OPEN | State both rules side by side |
| 2.12 | Formal multi-model tests | DONE (`r72/`) | Hansen's SPA and model confidence sets, after-shock days and other days |
| 2.13 | Formal conditional tests | DONE (`r75/`) | Coverage and predictive ability conditional on D |
| 2.14 | Simulation with known processes | DONE (`r73_simulation.json`) | One table |
| 2.15 | International stock panel | DONE (`docs/PANELS_PRECOMMIT.md`, deviation 3) | Price files contain unadjusted corporate actions and stale feeds; the screened rerun is the one to report, with the as-registered run beside it |

## 3. The score

| ID | Point | Status | Action |
|---|---|---|---|
| 3.1 | Why a maximum of three signals | OPEN | Drop the composite from the main text; the kurtosis signal alone was already the headline |
| 3.2 | Why these window lengths | DONE (`r69`, windows 21 to 250 days) | One robustness row |
| 3.3 | "Jump indicator" is neither | OPEN | Rename; reserve "jump" for the literature's use |
| 3.4 | The score mostly records one recent shock | DONE | Lead with it |
| 3.7 | What decides lambda | DONE (`r76/`, block H) | Firm-specific against market-wide shocks, within equities; does not order crypto, currencies or the other ETFs |
| 3.5 | A score derived from the model instead of chosen | DONE (`r71/`, block F; `r75/`) | D, the share of tomorrow's GARCH variance due to one past shock. DECIDE: replace the kurtosis score by D in the main text, or keep kurtosis as the pre-registered score and present D as its explanation. D is the better locator on the CRSP panel, about equal on US large caps |
| 3.6 | The cross-sectional regression of the score needs an equation and a table, or should go | OPEN | Likely superseded by the regressions on D |

## 4. Notation and exposition

| ID | Point | Status | Action |
|---|---|---|---|
| 4.1 | Write the tree estimator as a sum over the fixed level grid; define every symbol before use | OPEN | |
| 4.2 | Tail stage in the paper's own notation | OPEN | One notation table |
| 4.3 | Calibration stage described in full or moved to the appendix | OPEN | |
| 4.4 | Index sets defined once (which rows each stage is fitted on) | OPEN | |
| 4.5 | One algorithm for the whole pipeline in the main text | OPEN | |
| 4.6 | Two passages disagree on which rows the shape and tail fits use | code checked, text to verify | |
| 4.7 | Feature table: name, formula, lag | OPEN | |
| 4.8 | Define the rearrangement operator at first use | OPEN | |
| 4.9 | Spell out every abbreviation at first use | OPEN | |
| 4.10 | "Amortized" | partly DONE | Replace by "pooled" |
| 4.11 | No jargon before its definition | partly DONE | Sweep after the rewrite |
| 4.12 | Every claim names the estimate, its uncertainty and the test | partly DONE | |
| 4.13 | Delete asides that interrupt | OPEN | |
| 4.14 | Plain declarative prose; informative abstract | partly DONE | |
| 4.15 | Topic sentence per paragraph; check transitions | partly DONE | |
| 4.16 | Realized GARCH: "fitted to daily returns and a daily realized measure built from intraday data" | OPEN | |
| 4.17 | Sentence on the innovation law in the tail stage | OPEN | Rewrite or delete |
| 4.18 | Numbers in tables, findings in text | OPEN | |

## 5. Theory

| ID | Point | Status | Action |
|---|---|---|---|
| 5.1 | The elicitability lemma is standard | DONE (appendix, labelled) | Keep in the appendix or delete; DECIDE |
| 5.2 | The conformal proposition is standard | OPEN | Cite, do not state as a result |
| 5.3 | Say beside the estimator that conformal coverage is marginal | OPEN | |
| 5.4 | The closed-form shock response as the paper's formal element | DONE in results (`r71`, `r73`) | State as a proposition with a three-line proof: exact linearity of the recursion in one squared return; half-life log(0.5)/log(beta); breach rate and VaR ratio as functions of D and lambda |

## 6. Audit findings still open (`docs/AUDIT_2026-11.md`)

| ID | Point | Status |
|---|---|---|
| M2 | Capped-GARCH text still carries the unfair number; fair run is +0.35% (DM 2.65) at three sigma | text OPEN |
| M4 | Realized-volatility table's "daily core" is not the accuracy layer | OPEN |
| M5 | Cross-sectional regression is form-dependent | OPEN (see 3.6) |
| M6 | Transfer result holds names out but not dates; weak benchmark | OPEN |
| M7 | Cold-start ablation: the characteristics arm includes own trailing volatility | OPEN |
| M9 | Ten-day comparison mixes ES conventions | OPEN |
| M10 | Synthetic-truth job: VaR level and DM indexing | OPEN |
| L2 to L10 | Script-level items (missing generator for one result file, delisting rows, optimiser budgets, quadrature, labels, Kupiec conventions) | OPEN |

Several of these sit in sections that the rewrite may drop. Fix what stays; for what goes, record that it went.

## 7. Other

| ID | Point | Status | Action |
|---|---|---|---|
| 7.1 | GJR-skew-t: beaten on quantile loss, tied on the joint loss; say which | OPEN in text | `r75` adds the reason: its hit rate falls with D exactly as GARCH-t's does |
| 7.2 | The post-shock gain is smaller on US large caps (5%) than on the CRSP panel (8%) | explained (`r71`) | Smaller shocks and a higher lambda among non-earnings shocks; report both |
| 7.3 | Limitations paragraph | OPEN | Daily data only; lambda is a pooled average; survivor-tilted public panels; earnings dates from a free source and missing for a quarter of the CRSP names; the corrected GARCH-t under-covers because GARCH-t is too narrow on ordinary days |
| 7.4 | Length | deferred | Write out first, cut later |
| 7.5 | Preprint | DECIDE | Replace after the rewrite |
| 7.6 | Citations added in R65 and R67 were checked at abstract level only | OPEN | Read the full texts |
| 7.8 | Flexible models with the calendar as a training feature | OPEN | The like-for-like test so far gives each model two multipliers after the fact (`r78/`); retrain the trees and the network with days-to-announcement features |
| 7.9 | Literature on announcement effects in volatility and VaR models | OPEN | Earnings-day variance is long established; check whether published ML-against-GARCH VaR comparisons on single stocks control for the calendar, and credit GARCH models with announcement dummies |
| 7.10 | Scheduled one-offs outside stocks | OPEN | Central-bank and macro release days on the ETF, currency and bond panels |
| 7.11 | What the flexible models' remaining 1 to 2% on ordinary days is | OPEN | Not explained by shock age, D, the score or the volatility level (`r76/`, block S) |
| 7.7 | Earnings result needs a second source | OPEN | Check the announcement dates against another free calendar on a sample of names; test the earnings-only correction on the canonical names without dates |
