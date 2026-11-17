# Case studies (outside the paper's tables)

Single-instrument and cross-asset checks of the paper's estimator against standard benchmarks, on public prices
(yfinance adjusted closes, downloaded to `data/case_studies/`, not committed). Results are in `results/case_studies/`.
Every script writes its predictions into its header before it runs; the graded outcomes are in the result files and
below.

| Script | Status | Question | Result file |
|---|---|---|---|
| `nvda_gjr_case.py` | exploratory | Is GJR-skew-t a significant loss on NVDA, 2020–2026? | `nvda_gjr_case_results.json` |
| `nvda_engine_case.py` | exploratory | The paper's estimator on NVDA (pooled parts trained on the CRSP panel before 2020, NVDA excluded) | `nvda_engine_case_results.json` |
| `nvda_engine_variants.py` | exploratory | Does putting NVDA in the training sample, or retraining yearly, change that? | `nvda_engine_variants_results.json` |
| `multiasset_case.py` | exploratory | 17 instruments across asset classes; decision rule for changing the recommended configuration | `multiasset_case_results.json` |
| `crossasset.py` | **pre-registered** (`docs/CASE_STUDY_PRECOMMIT.md`) | Is the estimator comparable (non-inferior within 0.01 FZ0) to GARCH-t and the other standard benchmarks across asset classes? 15 confirmatory instruments not examined before the pre-commitment, plus the 17 exploratory ones | `crossasset_results.json` |

`nvda_engine_case.py` also holds the shared helpers (Beta-t-EGARCH filter and likelihood copied from
`code/paper/job_robust_engine.py`, the feature construction, the pooled GPD tail). Run from the repo root with
`GBC_PROJ` pointing at the licensed CRSP panel directory.

Known issue in `multiasset_case.py` (exploratory): its pooled DM averages each calendar date over the instruments
trading that day, so bitcoin's weekends enter alone and are over-weighted. `crossasset.py` pools over NYSE dates only.
