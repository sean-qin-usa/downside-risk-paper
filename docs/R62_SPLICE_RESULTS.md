# R62 results: the state-anchored splice is rejected by its own decision rule, and it proves the mechanism

Graded against `docs/R62_SPLICE_PRECOMMIT.md`, written before the run. canon200, 221,600 rows. Primary arm
$p_0=0.05$ anchored against unconditional **at the same $p_0$**, so anchoring is the only difference. Nothing
committed, no manuscript edit.

## Verdict first

**Not adopted.** The decision rule required P1, P2 and P3 to pass; P2 and P3 fail at 2.5% and P1 fails as
written. **P4 passes decisively, which is the scientifically valuable part**: intervening on the override depth
removes the top-decile deficit, so R61's explanation is now supported by an intervention and not only by a
cross-decile correlation.

| | prediction | outcome | verdict |
|---|---|---|---|
| **P1** | top-decile FZ0 improves ≥0.034 (1%), ≥0.030 (2.5%); deciles 1–9 move <0.005, \|DM\|<2 | improves **0.0717** and **0.0633**; deciles 1–9 move up to **0.035**, all *improving* | **fails as written**, purpose met |
| **P2** | top-decile breach rises toward nominal, not exceeding it | 1%: 0.77%→0.93% ✓. 2.5%: 2.05%→**2.73%**, over nominal | **fails at 2.5%** |
| **P3** | pooled breach within 0.1pp of 1.04%/2.78%; clustered test still passes at 1% | 1%: 1.10%, 0.06pp off, $t=1.56$ ✓. 2.5%: **3.09%**, 0.31pp off, $t$ 2.01→**3.90** | **fails at 2.5%** |
| **P4** | top-decile override depth falls ≥50% | 0.526→**0.151** (−71%) and 0.335→**0.049** (−85%) | **passes** |

## What the intervention shows

| $\alpha=1\%$ | unconditional $p_0{=}0.05$ | anchored $p_0{=}0.05$ |
|---|---|---|
| overall FZ0 | 2.14773 | **2.12865** |
| overall breach | 1.02% | 1.10% |
| date-clustered $t$ | 0.38 | 1.56 |
| top-decile FZ0 | 2.30435 | **2.23263** |
| top-decile `body − engine` | −0.06961 (DM **−6.59**) | +0.00211 (DM **+0.09**) |
| top-decile override depth | 0.526 | 0.151 |

At 2.5%: overall FZ0 1.84941 → 1.83688, top-decile FZ0 1.97677 → 1.91347, `body − engine` −0.05987 (DM −6.86)
→ +0.00342 (DM +1.40), override depth 0.335 → 0.049.

**The top-decile deficit is not halved, it is eliminated.** The anchored engine ties its own body there
(DM +0.09 and +1.40, both inside noise) instead of losing to it by six and a half standard errors. And it does
so while the override depth collapses by 71–85%, which is the mechanism R61 named. Overall FZ0 improves at both
levels too.

**But the 2.5% coverage the branch exists to protect is given up.** Pooled breach goes 2.78% → 3.09% and the
date-clustered exception test goes from marginal ($t=2.01$) to decisive rejection ($t=3.90$). At $p_0=0.05$ the
anchored tail starts at the body's 5% quantile (mean $z$ −1.418) and extends down with the pooled shape, which
leaves the 2.5% quantile close to the body's own — so the arm inherits the body's over-breaching, exactly what
the spec's P2 called "the repair has merely become the body".

## The trade, now measured on both sides

This is the reportable finding, and it is stronger than R61's. The unconditional tail is not simply a defect:

> The pooled GPD buys roughly 0.3pp of aggregate 97.5% coverage at a cost of 0.06–0.07 in top-decile joint loss
> and 0.013–0.019 overall.

Both halves are now quantified on the same rows, under one ES convention, with the splice as the only moving
part. The earlier framing — "the top-decile result is partly a construction effect" — understated it: the
construction effect is the whole of the top-decile deficit, and removing it costs something specific and
measurable rather than nothing.

## Secondary and robustness arms, as pre-registered

| arm | $\alpha$ | overall FZ0 | breach | clustered $t$ | top-decile `body − engine` |
|---|---|---|---|---|---|
| $p_0{=}0.025$ unconditional (**= the shipped engine**) | 1% | 2.14731 | 1.04% | 0.62 | −0.06832 (DM −6.52) |
| | 2.5% | 1.84936 | 2.78% | 2.00 | −0.05984 (DM −6.84) |
| $p_0{=}0.025$ anchored | 1% | 2.12875 | 1.17% | **2.61** | +0.00499 (DM +0.75) |
| $p_0{=}0.05$ anchored, proportional rescale | 1% | 2.12757 | 1.11% | 1.81 | +0.00637 |
| | 2.5% | 1.83665 | 3.10% | 3.97 | +0.00476 |

Two things worth noting. The $p_0=0.025$ unconditional arm reproduces the shipped engine to five decimals
(2.14731 / 1.84936, breach 1.04% / 2.78%, $t$ 0.62 / 2.00) — a fidelity control on the whole apparatus, in
addition to the nesting identity. And **the choice between threshold stability and proportional rescaling does
not matter**: the robustness arm lands within 0.001 of the primary on FZ0 at both levels and fails the same way
at 2.5%. The pre-registered decision to settle on threshold stability cost nothing.

The $p_0=0.025$ anchored arm is worse than the primary at 1% ($t=2.61$, a rejection, against 1.56), which is
why the spec made $p_0=0.05$ primary: with the splice at the evaluation level there is no room for the tail to
act.

## The holdout was deliberately not scored

P5 reserved the holdout for a single frozen check **after** the canon200 decision. The decision resolved on
canon200 — not adopted — so scoring the holdout would spend a one-shot check on a variant already rejected, and
would invite reading it as a second chance. It remains unused.

## Controls

| control | result |
|---|---|
| nesting identity: anchoring at the pooled threshold reproduces the pooled tail | **PASS** to 1e-10 at $\tau\in\{0.025, 0.01, 0.001, 0.00025\}$ |
| levels above $p_0$ untouched | **PASS** — `build()` modifies only $\tau \le p_0$ |
| parametric GARCH-$t$ FZ0 unchanged across arms | 2.15572 / 1.85420 |
| pooled FHS FZ0 unchanged across arms | 2.15570 / 1.85975 |
| ES convention | 20-node in every arm, held fixed so the comparison isolates the splice |

One correction to R61's draft wording, already recorded in the spec: pinball **at 0.01 and 0.025 does move**,
because those are the levels the splice governs. Only $\tau > p_0$ is untouched.

## Where this leaves the top-decile question

The diagnosis is settled and the first repair is rejected with its numbers. What the result points at is a
construction whose anchoring **strength depends on the state** — unconditional where the body over-breaches,
anchored where the body has shrunk its tail — since the two failures are at opposite ends of the score
distribution and of the probability scale. That is a second variant, and under the same discipline it needs its
own pre-commitment before it is estimated. It is not designed here.
