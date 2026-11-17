# Change ledger R64: length

Applied by `docs/apply_R64.py` then `docs/apply_R64b.py` on top of R63. **No number changes.** Moved text goes word
for word into a new online-appendix section, *Detail moved from the main text* (`sec:moved`), with main-text
cross-references rewritten as "Section/Table/equation N of the paper"; each script refuses to move a block that
still contains a `\ref` or `\eqref`. The main text keeps a short pointer. Two blocks are deletions of statements
the paper already makes elsewhere (M, O).

JFEC's guideline is about 40 double-spaced pages. The journal build went from **66 pages (R63) to 58**; the text
now ends on page 42, followed by the endnotes and references and then the floats (endfloat).

| id | block | from | words | pointer |
|---|---|---|---|---|
| A | ten-day subsection | §5.3 | 374 → 74 | one paragraph inside §5 |
| B | raw vs ν-relative score (two paragraphs) | §4.1 | 296 → 64 | the regression result and its reading |
| C | FZ0 on the large-cap sandbox subset | §5.1 | 95 → 0 | — |
| D | why the joint-score edge is flat across deciles (R61/R62 splice detail) | §5.1 | 190 → 50 | the trade, stated once |
| E | direct ES calibration backtests | §5.1 | 295 → 88 | the verdicts, both eras |
| F | IQN architecture and training, estimator choice, relation to GBC; equation (6) | §2.3 | 547 → 77 | one paragraph |
| G | the filter window and the calibration split | §2.2 | 129 → 46 | the result |
| H | ES quadrature detail | §2.2 | 149 → 35 | one sentence |
| I | composite score on the holdout | §3 | 74 → 0 | — (the figure and §4 carry it) |
| J | MCS bootstrap provenance sentence | §3 | 24 deleted | — |
| K | choice of scoring function | §2.1 | 295 → 87 | the three reasons in three sentences |
| L | IQN, GBC and conformal background | §1.3 | 215 → 76 | one paragraph, every citation kept |
| M | GARCH-EVT numbers stated in §1.2 and again in §4/§5 | §1.2 | 104 → 36 | deletion of the duplicate numbers |
| N | Table 4, the cross-universe table | §4.2 | float | Online Appendix Table OA.20 |
| O | ten-day sentence in the conclusion's qualifications | §6 | 57 → 19 | — |

## Numbering that moved

- Equation (6), the IQN, left the main text: the score is now (6) and FZ0 (7). The appendix's "equation (7) of
  the paper" (the score, after R63) is now "(6)" (two occurrences).
- Table 4 left the main text: FRTB 5 → 4, non-core 6 → 5. The appendix's hard-coded references were renumbered
  (five × 5 → 4, one × 6 → 5). Main-text references use `\ref` and follow automatically.
- The appendix gains Table OA.20 (cross-universe, moved) after OA.18 and OA.19 (R63).

## Not cut, and why

- **Section 5.1's joint-score paragraph** (about 1,000 words) is the densest remaining block. It carries the
  scorecard the reframe depends on (the tie group, the MCS, GAS and Taylor, Beta-t-EGARCH, the walk-forward), so
  cutting it needs an editorial call about which comparisons stay in the main text.
- **Section 3 conventions** (about 510 words): the inference conventions a JFEC referee checks first.
- **Cold-start paragraphs** in §2.3 and §6: held pending the peer-proxy reconciliation (see R63 ledger).
