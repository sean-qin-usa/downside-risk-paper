# Manuscript files

`paper_A_jfec.pdf` and `paper_A_jfec.tex` are the main manuscript; `paper_A_jfec_online_appendix.pdf` and `.tex` are the online appendix. Both follow the Journal of Financial Econometrics review format, with double-spaced text, endnotes, and every table and figure collected at the end of the manuscript under a "[Table N about here.]" marker at the point of first mention. `refs_v3.bib` is the bibliography for both. The figures are drawn with pgfplots inside the source, so the two `.tex` files and the bibliography are everything a build needs.

`tables/` holds the table bodies that `code/paper/make_tables_garch_evt.py` and `code/paper/make_tables_bench_all.py` generate from the result files; the online appendix embeds them.
