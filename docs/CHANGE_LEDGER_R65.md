# Change ledger R65: prior work the R63 framing must be positioned against

A literature check after R63 (2026-11-08) found five papers the paper did not cite. Each piece of the R63 headline has a
precedent; the combination (a panel, a score that flags the days in advance, and the attribution of the flexible
models' advantage to post-shock scale error) was not found. Bibliographic details were checked against
EconPapers/EconBiz/arXiv records before entry; Trucíos, Hotta and Ruiz (robust bootstrap GARCH VaR densities) was left
out because its details could not be confirmed.

| key | reference | why it is needed |
|---|---|---|
| `blazsek2018extreme` | Blazsek, Carrizo, Eskildsen, Gonzalez, *Finance Research Letters* 24 (2018) 193–198 | closest precedent: Beta-t-EGARCH beats t-GARCH on the day after an extreme DJIA return |
| `mancinitrojani2011` | Mancini and Trojani, *JFEC* 9(2) (2011) 281–313 | robust VaR prediction for GARCH-type models, in the target journal |
| `andersen2007roughing` | Andersen, Bollerslev, Diebold, *REStat* 89(4) (2007) 701–720 | jumps less persistent than continuous variation |
| `corsi2010threshold` | Corsi, Pirino, Renò, *J. Econometrics* 159(2) (2010) 276–288 | forecast gains from separating jumps largest after a jump |
| `ormaniec2022lstm` | Ormaniec, Pitera, Safarveisi, Schmidt, arXiv:2207.10539 (2022) | an ML VaR model's advantage comes from faster reaction to volatility changes |

Edits: Section 1.2's robust-GARCH paragraph adds Mancini–Trojani, Blazsek et al., and the two jump papers; Section 1.3
adds Ormaniec et al.; the introduction's novelty sentence ("to our knowledge the attribution by state of the benchmark
filter is new") is replaced by one that names the known ingredients and states what is added. `refs_v3.bib` is copied
to `preprint/`.
