# Downside-Risk: Amortized Conditional-Quantile Forecasting

Research code and results for amortized (transfer) conditional-quantile estimation of financial downside risk, with Prof. Wenxin Jiang (Northwestern).

## Key finding

A single cross-sectionally trained (amortized) quantile model, conditioned on characteristics, beats a name's own return history at every listing age. The edge is largest when the name is brand-new (about 6.4% lower pinball loss at days 15–30) and persists at maturity (about 2.6%). In the cold-start regime (under 250 days, where GARCH cannot fit) the amortized model is the only option.

A naive age-weighted blend toward own history does not help. The amortized model works better as a prior (Gibbs / partial pooling) than as something to shrink toward the weaker own-empirical estimate.

## Contents

code/ -- neural IQN and amortized gradient-boosted quantile training and evaluation.
results/ -- pinball-loss results by listing-age bucket and history length (held-out CRSP names).

Trading applications are maintained separately and are not part of this repository.
