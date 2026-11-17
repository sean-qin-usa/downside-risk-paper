# apply_R64b.py -- R64, second batch. Same rules as apply_R64.py: moved text goes word for word into the
# online-appendix section "Detail moved from the main text" (created by apply_R64.py), cross-references rewritten,
# a short pointer left in place; deletions are of duplicated statements only. Run after apply_R64.py.
# Run from the repo root:  python docs/apply_R64b.py        (--dry to check without writing)
import sys, re

DRY = "--dry" in sys.argv
MAIN = "submission/paper_A_jfec.tex"
OA = "submission/paper_A_jfec_online_appendix.tex"
m = open(MAIN, encoding="utf-8").read()
o = open(OA, encoding="utf-8").read()
moved, log = [], []

def take(s, start, end, include_end=True):
    assert s.count(start) == 1, start; a = s.index(start)
    assert s.count(end) == 1, end; b = s.index(end, a) + (len(end) if include_end else 0)
    return a, b

def cut(name, start, end, pointer, heading, include_end=True, fix=()):
    global m
    a, b = take(m, start, end, include_end)
    text = m[a:b]; m = m[:a] + pointer + m[b:]
    body = text
    for x, y in fix:
        assert x in body, (name, x); body = body.replace(x, y)
    if heading:
        left = re.findall(r"\\(?:ref|eqref)\{[^}]+\}", body)
        assert not left, (name, left)
        moved.append((heading, body.strip()))
    log.append("%-3s %-58s %5d -> %4d words" % (name, heading or "(deleted duplicate)", len(text.split()), len(pointer.split())))

# ---------------------------------------------------------------- K. choice of scoring function
cut("K", r"\emph{Choice of scoring function.}", "carries an in-sample advantage that the out-of-sample comparison tests.",
    r"""\emph{Choice of scoring function.} Pinball at fixed levels, rather than the continuous ranked probability score that integrates it over all levels \citep{gneitingranjan2011}, keeps the comparison where downside risk lives: the models differ almost entirely in the tail, and a uniform weighting over levels would average those differences away. ES alone is not elicitable \citep{gneiting2011}, so $(\VaR,\ES)$ pairs are scored by the FZ0 loss of Section~\ref{sec:frtb}. Each benchmark is estimated by its own conventional criterion and all are scored out of sample by the same losses (Online Appendix).""",
    "Choice of scoring function",
    fix=[(r"Section~\ref{sec:frtb}", "Section~5 of the paper")])

# ---------------------------------------------------------------- L. Section 1.3: IQN, GBC and conformal background
cut("L", r"The nonparametric family estimates $Q_\phi(\tau \mid s_t)$ directly", "the case for time-ordered residuals.",
    r"""The nonparametric family estimates $Q_\phi(\tau \mid s_t)$ directly by pinball-loss empirical risk minimization \citep{koenker1978, steinwart2011}: gradient-boosted quantile trees \citep{friedman2001}, used throughout, and the implicit quantile network of \citet{dabney2018iqn}, the architecture at the core of generative Bayesian computation \citep{polson2023gbc, nareklishvili2025gqbp} in the generalized-Bayes tradition of \citet{jiang2008gibbs} and \citet{bissiri2016general}, retained as a benchmark. Split-conformal prediction \citep{vovk2005, lei2018, romano2019cqr} and its adaptive form \citep{gibbs2021aci} supply the coverage correction of Stage~4, and \citet{barber2023} bound its coverage loss when exchangeability fails.""",
    "Distribution-free conditional quantiles, GBC, and conformal prediction",
    fix=[(r"Section~\ref{sec:gbcmethod}", "Section~2.3 of the paper")])

# ---------------------------------------------------------------- M. Section 1.2: GARCH-EVT numbers stated twice
cut("M", r"The comparison set of Table~\ref{tab:frtb} does not include standalone GARCH-EVT", "in the top kurtosis decile and on the joint score at both levels.",
    r"""Standalone GARCH-EVT \citep{mcneilfrey2000}, which \citet{kuester2006} rank first among classical methods, is outside the comparison set of Table~\ref{tab:frtb} because the estimator's EVT stage absorbs its filter-then-tail logic; it is compared on the same rows in Sections~\ref{sec:frontier} and~\ref{sec:frtb}.""",
    None)

# ---------------------------------------------------------------- N. the cross-universe table to the appendix
a, b = take(m, r"\begin{table}[t]" + "\n\\centering\n\\begin{threeparttable}\n\\caption{The frontier across universes.", r"\label{tab:universes}")
a2 = a; b2 = m.index(r"\end{table}", b) + len(r"\end{table}")
tab = m[a2:b2]; m = m[:a2] + m[b2:].lstrip("\n")
tab = tab.replace(r"\begin{table}[t]", r"\begin{table}[htbp]", 1)
log.append("N   cross-universe table moved to the appendix (float, %d words)" % len(tab.split()))
x = r"Table~\ref{tab:universes} shows the same axis at work across markets."
assert m.count(x) == 1
m = m.replace(x, r"The same axis is at work across markets (Online Appendix Table~OA.20).")

# ---------------------------------------------------------------- O. conclusion: statements made twice in the same paragraph
cut("O", "The ten-day extension is a separate direct model: it beats", "and the scaled parametric tail is the better calibrated of the two (Section~\\ref{sec:frtb}).",
    r"""The ten-day extension is a separate direct model whose in-era edge does not survive the 2000--2013 rerun (Online Appendix).""", None)

# ---------------------------------------------------------------- the appendix
sec = []
for h, t in moved:
    sec.append("\\subsection*{%s}\n%s\n" % (h, t))
sec.append("\\subsection*{The frontier across universes}\nTable~\\ref{tab:universes} is the cross-market table of Section~4.2 of the paper.\n\n" + tab + "\n")
anchor = r"\section{The two-regime example}"
assert o.count(anchor) == 1
o = o.replace(anchor, "\n".join(sec) + "\n" + anchor)
# main tables after the moved Table 4 renumber: FRTB 5 -> 4, non-core 6 -> 5
n5 = len(re.findall(r"Table~5 of the paper", o)); n6 = len(re.findall(r"Table~6 of the paper", o))
o = o.replace("Table~5 of the paper", "Table~@4 of the paper").replace("Table~6 of the paper", "Table~5 of the paper").replace("Table~@4", "Table~4")
log.append("    appendix references renumbered: %d x Table 5->4, %d x Table 6->5" % (n5, n6))

print("\n".join(log)); print("main words: %d" % len(m.split()))
if not DRY:
    open(MAIN, "w", encoding="utf-8").write(m); open(OA, "w", encoding="utf-8").write(o); print("written")
