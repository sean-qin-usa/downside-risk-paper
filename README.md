# Downside Risk at the Misspecification Frontier

**The paper: [paper_A_frontier.pdf](paper_A_frontier.pdf)** ([LaTeX](paper_A_frontier.tex) | [bibliography](refs_v3.bib))

A real-time score for when nonparametric models beat industry-standard VaR and ES, with a deployable amortized engine. Sole-authored by Sean Qin; developed with the guidance of Prof. Wenxin Jiang (Northwestern). `code/` holds the study scripts, `results/` the derived statistics behind every number in the paper, `docs/` the submission material. No licensed data (CRSP/WRDS, Bloomberg) are redistributed; panels rebuild from the documented queries for licensed subscribers. Day-to-day working notes live in the companion repository `downside-risk-strategies`.

---
# Downside-Risk: Amortized Conditional-Quantile Forecasting

Research code and results for amortized (transfer) conditional-quantile estimation of financial downside risk, with Prof. Wenxin Jiang (Northwestern).

## Key finding

A single cross-sectionally trained (amortized) quantile model, conditioned on characteristics, beats a name's own return history at every listing age. The edge is largest when the name is brand-new (about 6.4% lower pinball loss at days 15â€“30) and persists at maturity (about 2.6%). In the cold-start regime (under 250 days, where GARCH cannot fit) the amortized model is the only option.

A naive age-weighted blend toward own history does not help. The amortized model works better as a prior (Gibbs / partial pooling) than as something to shrink toward the weaker own-empirical estimate.

## Contents

code/ -- neural IQN and amortized gradient-boosted quantile training and evaluation.
results/ -- pinball-loss results by listing-age bucket and history length (held-out CRSP names).

Trading applications are maintained separately and are not part of this repository.

