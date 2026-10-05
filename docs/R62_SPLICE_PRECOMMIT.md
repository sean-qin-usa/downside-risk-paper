# R62 pre-commitment: the state-anchored splice

**Written and dated before the variant was estimated.** R61's rows motivated it, so nothing below may be
chosen after seeing a result. One construction, one decision rule, predictions with numbers, and the
conditions under which it is abandoned. Date: 2026-10-05.

---

## 1. The defect being repaired

R61 established that the engine's joint-loss deficit in the top mk$_{63}$ decile is a property of the splice.
The pooled GPD sits at a **constant** level in standardised-residual space (−2.554 at 1%, −1.880 at 2.5%). The
body pulls its tail in as the score rises, from −2.527 to −2.252 across the deciles at 1%, and the
min-envelope's override therefore deepens 2.8× (0.188 → 0.523). The FZ0 gap moves with it, +0.011 → −0.068,
with a cross-decile correlation of −0.84 against the override depth, against only −0.47 against the frequency
with which the branch binds. The branch is not removable: the body over-breaches overall (1.32% and 3.25%) and
the engine is closer to nominal coverage than the body in 8 of 10 deciles at 1%.

So the repair must make the tail **move with the state** while keeping a tail at all.

## 2. The construction, named in advance

**State-anchored GPD splice.** Keep the pooled shape $\hat\xi$. Replace the pooled threshold by each row's own
body quantile at the splice level, and move the scale by the GPD's own threshold-stability relation. In loss
orientation ($Y=-z$, $t=-u$):

$$t'_i = -\,\widehat{q}^{\,\text{body}}_i(p_0), \qquad \beta'_i = \hat\beta + \hat\xi\,(t'_i - t), \qquad
Q^{\text{evt}}_i(\tau) = t'_i + \frac{\beta'_i}{\hat\xi}\Big[(\tau/p_0)^{-\hat\xi} - 1\Big],\ \ \tau \le p_0 .$$

Three properties, all verified numerically before writing this, none of them results:

- **It nests the current construction.** Anchoring at the pooled threshold reproduces the pooled tail exactly
  ($\tau=0.01$: 2.55381 both ways). The variant adds **no free parameter** — $\hat\xi$ and $\hat\beta$ are the
  pooled fitted values and the threshold is read off the body.
- **$\beta'_i>0$ is safe.** It would require the body's $p_0$ loss-quantile below $t-\hat\beta/\hat\xi=-0.26$,
  i.e. a negative loss quantile. Guarded in code regardless.
- **At $\tau=p_0$ the tail equals its own anchor**, hence equals the body. This is arithmetic, not a finding,
  and it fixes the design below.

**Why threshold stability rather than proportional rescaling.** $\beta(t)=\beta(t_0)+\xi(t-t_0)$ is the GPD's
own self-consistency property, so it introduces no assumption beyond the one already made in fitting a GPD. A
proportional rescale $\beta'=\hat\beta\,(t'/t)$ is also scale-equivariant and defensible. **The decision is
taken on threshold stability.** The proportional form is run only as a labelled robustness check and may not
override the primary, whichever way it comes out.

## 3. Design forced by the $\tau=p_0$ degeneracy

Because the anchored tail equals the body at $\tau=p_0$, the variant is informative only at $\alpha<p_0$:

| splice level | informative at |
|---|---|
| $p_0=0.025$ | $\alpha=1\%$ only — at $\alpha=2.5\%$ the engine *is* the body, by construction |
| $p_0=0.05$ | $\alpha=1\%$ and $\alpha=2.5\%$ |

So **$p_0=0.05$ is the primary run** and $p_0=0.025$ a consistency check at 1%. Both are compared against the
*unconditional* splice at the **same** $p_0$, so the anchoring is the only thing that differs. The paper already
sweeps $p_0\in\{1.5\%,2.5\%,5\%\}$ unconditionally, so the comparator exists (`coherent_results.json`,
`p0_0.05`: FZ0 2.14692 at 1% and 1.84817 at 2.5%).

## 4. Predictions

Graded on canon200. Signs follow the repository convention: a DM on `row − engine`, negative meaning the row
is better.

- **P1 — the top-decile deficit at least halves, and the bulk does not degrade.** Against the unconditional
  splice at the same $p_0$, the anchored engine's top-decile FZ0 improves by **≥ 0.034 at 1%** and **≥ 0.030 at
  2.5%** (half of R61's 0.068 and 0.060 gaps). Deciles 1–9 move by **< 0.005** with $|DM| < 2$.
  *Fails if the bulk degrades at all*: the branch's whole purpose is the bulk.
- **P2 — top-decile coverage moves toward nominal from below, without overshooting.** Top-decile breach rises
  from 0.79% toward 1.0% and from 2.04% toward 2.5%, and **does not exceed nominal** at either level.
  *Overshooting means the repair has merely become the body.*
- **P3 — aggregate coverage is preserved.** Pooled breach within **0.1pp** of 1.04% and 2.78%, and the
  date-clustered exception test still passes at 1%. A construction that buys top-decile FZ0 with aggregate
  coverage is not an improvement.
- **P4 — the mechanism variable moves.** Mean override depth in the top decile falls by **at least half**, from
  0.523 at 1% and 0.336 at 2.5%. *If FZ0 improves while the override depth does not fall, R61's explanation is
  wrong and the gain is coming from somewhere else.*
- **P5 — the holdout is scored once, frozen, after the canon200 decision**, and is not used to choose anything.

## 5. Controls, reported as pass/fail before any result is read

Following `job_es_converged.py`:

1. **Parametric and empirical-tail rows unchanged** — GARCH-$t$ and pooled FHS carry no splice.
2. **Pinball at $\tau \ge 0.05$ unchanged.** The splice touches only $\tau \le p_0$. Note the correction to
   R61's draft wording: pinball **at 0.01 and 0.025 does move**, because those are the levels the splice
   governs. Claiming all pinball is unchanged would have been wrong.
3. **The nesting identity holds in code**: forcing the anchor to the pooled threshold must reproduce the
   unconditional engine to floating-point.
4. **ES convention fixed** at the converged integral throughout.

## 6. Decision rule

Adopted only if **P1, P2 and P3 all pass** on canon200, with P4 as the mechanism check. Any other outcome is
reported as a tested-and-rejected variant with its numbers, in the same place the static conformal overlay is
reported — the paper already has a precedent for reporting a repair that did not work. The $p_0=0.05$ run
decides; $p_0=0.025$ and the proportional-rescale form are robustness only.

Nothing in the manuscript changes on the strength of this run alone.
