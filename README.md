# Semiparametric Value-at-Risk and Expected Shortfall with a Real-Time Misspecification Score

## The paper

The manuscript is [`submission/paper_A_jfec.pdf`](submission/paper_A_jfec.pdf) — the version prepared for the *Journal of Financial Econometrics* (double-spaced, endnotes, floats collected at the end with "[Table N about here]" markers, as the journal's review format requires). Its LaTeX source is [`submission/paper_A_jfec.tex`](submission/paper_A_jfec.tex) and the bibliography is [`submission/refs_v3.bib`](submission/refs_v3.bib).

| File | What it is |
|---|---|
| [`submission/paper_A_jfec.pdf`](submission/paper_A_jfec.pdf) | The manuscript, in journal review format. |
| [`submission/paper_A_jfec_online_appendix.pdf`](submission/paper_A_jfec_online_appendix.pdf) | Online appendix (algorithms, supplementary figures and tables, and proofs deferred from the main text). |
| [`gbc_downside_main.pdf`](gbc_downside_main.pdf) | A convenience copy of the current submission PDF for quick reference. |

An earlier single-spaced reading build (`paper_A_frontier.*`) has been retired: it predated the reframing of the paper and no longer matched the submission in either its numbers or its front matter. The submission build above is now the single source of truth; the retired build remains in the repository history for anyone who needs it.

## What the paper shows

A single measurable quantity — the excess kurtosis and asymmetry of recent GARCH-standardized residuals — orders when flexible-shape quantile methods beat parametric VaR/ES. Where the score is high, an amortized nonparametric estimator wins decisively (top decile +3.0% pinball, DM 10.5; replicated on an untouched 2000–2013 holdout under a frozen specification with predictions written in advance, under strict calendar splits, under a true annual-refit walk-forward, and in a point-in-time universe that keeps delisted names); where it is low the advantage shrinks toward zero. The score orders the magnitude of the edge, which is concentrated in the top decile, never a detectable loss.

The paper's central result is a decomposition of that top-decile edge. Measured against a jump-robust GARCH that does not overstate its variance after a single large shock, the +3.0% edge falls to roughly +0.3% to +0.5% (DM 2.5 to 4.7), so most of the advantage is the standard filter's post-shock scale error and a smaller but statistically significant part is conditional-shape value that a robust scale does not remove. The estimator passes the aggregate date-clustered exception tests at both regulatory levels through a 2008-crisis window, and its accuracy layer attains the lowest joint (VaR, ES) FZ0 score at both levels on the full panel.

## Repository layout

| Location | Contents |
|---|---|
| `submission/` | Journal-format build and online appendix (see `submission/README.md`) |
| `refs_v3.bib` | Bibliography (also mirrored under `submission/`) |
| `code/` | Analysis scripts. Every number in the paper traces to one script here; each script documents its data inputs at the top |
| `results/` | Derived statistics as JSON — one file per script run; these are the numbers quoted in the paper |
| `docs/` | Research notes, review syntheses, cover letter |
| `figures/` | Exported charts |

Root-level `*.py` files (for example `es_backtests.py`, `realized_hybrid_experiment.py`, `rgarch_bench.py`) are standalone experiment drivers with their JSON outputs alongside; the paper's cited numbers come from `code/` and `results/`.

## Data and licensing

Return data derive from CRSP/WRDS and Bloomberg under the author's licenses and are **not** redistributed — no raw data files are tracked. Every WRDS-based panel rebuilds from the documented queries in `code/` for any licensed subscriber; Bloomberg-based exhibits are preserved as-run (terminal access ended mid-2026). A synthetic end-to-end example (`code/toy_example.py`) runs the full pipeline without licensed data.
