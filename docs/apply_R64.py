# apply_R64.py -- R64: length. Moves supporting detail from the main text to a new online-appendix section,
# word for word, with main-text cross-references rewritten for the appendix; leaves a short pointer in place.
# No number changes. Every anchor is checked for uniqueness before anything is written.
# Run from the repo root:  python docs/apply_R64.py        (--dry to check without writing)
# Then: python code/paper/make_ssrn_build.py, and build both PDFs in submission/ and preprint/.
import sys, re

DRY = "--dry" in sys.argv
MAIN = "submission/paper_A_jfec.tex"
OA = "submission/paper_A_jfec_online_appendix.tex"
m = open(MAIN, encoding="utf-8").read()
o = open(OA, encoding="utf-8").read()
moved = []           # (heading, text) appended to the new OA section, in order
log = []

def span(s, start, end, include_end=True, start_back=0):
    a = s.index(start); assert s.count(start) == 1, start
    a -= start_back
    b = s.index(end, a); assert s.count(end) == 1, end
    if include_end: b += len(end)
    return a, b

def cut(name, start, end, pointer, heading, include_end=True, start_back=0, keep_head=None, fix=()):
    """Remove main[start..end], put `pointer` in its place, append the removed text to the OA section."""
    global m
    a, b = span(m, start, end, include_end, start_back)
    text = m[a:b]
    m = m[:a] + pointer + m[b:]
    body = text if keep_head is None else text[len(keep_head):]
    for x, y in fix:
        assert x in body, (name, x); body = body.replace(x, y)
    # main-text cross-references cannot resolve in the appendix
    left = re.findall(r"\\(?:ref|eqref)\{[^}]+\}", body)
    assert not left, (name, left)
    moved.append((heading, body.strip()))
    log.append("%-4s %-55s %5d -> %4d words" % (name, heading, len(text.split()), len(pointer.split())))

# ---------------------------------------------------------------- A. ten-day subsection
cut("A", r"\subsection{The ten-day horizon: a direct multi-day extension}", r"\subsection{Cross-asset evidence}",
    r"""\paragraph{The ten-day horizon.} A separate direct model trained on ten-day returns beats $\sqrt{h}$-scaled GARCH by about 1.4\% of pinball on 2014--2024, and about half of that edge is the scaling rule itself. On the pre-committed 2000--2013 rerun it ties the iterated GARCH benchmarks and the scaled parametric tail is the better calibrated. The ten-day evidence is mixed; it is reported in the Online Appendix (Table~OA.13), and no ten-day claim attaches to the one-day estimator.

""", "The ten-day horizon", include_end=False,
    keep_head=r"\subsection{The ten-day horizon: a direct multi-day extension}",
    fix=[("Algorithm~1", "Algorithm~1 of the paper"), ("(Online Appendix\nFigure~OA.3)", "(Figure~OA.3)"),
         ("the benchmark\ntable is Online Appendix Table~OA.13.", "the benchmark\ntable is Table~OA.13.")])

# ---------------------------------------------------------------- B. the nu-relative score analysis
cut("B", r"The score's content could reflect model-relative", r"\paragraph{What a top-decile score records.}",
    r"""Normalizing the score by each name's fitted $t_{\nu_i}$ does not sharpen it. In a within-date regression of the edge on both, the raw score carries the predictive content ($t=7.8$) and the $\nu$-relative percentile has negative incremental content ($t=-5.2$), so the signal is the absolute level of recent tail activity rather than a per-name surprise (Online Appendix). That is what a single recent shock produces.

""", "Raw versus $\\nu$-relative score", include_end=False,
    fix=[(r"(Section~\ref{sec:score})", "(Section~2.5 of the paper)")])

# ---------------------------------------------------------------- C. the large-cap sandbox FZ0 paragraph
cut("C", "large-cap sandbox subset used to fix", "settles the magnitudes on the residual-hybrid's own forecasts.",
    "", "FZ0 on the large-cap sandbox subset", start_back=len("On the\n"))

# ---------------------------------------------------------------- D. the splice mechanism detail
cut("D", "Dropping the EVT branch and scoring the pooled body alone", "and the paper keeps it for the coverage.",
    r"""The pooled GPD is unconditional, so in the top decile it overrides a body that has pulled its tail in; the unconditional tail buys about 0.3 points of 97.5\% coverage at a cost of 0.06 to 0.07 in top-decile FZ0, and the paper keeps it for the coverage (Online Appendix).""",
    "Why the joint-score edge is flat across deciles")

# ---------------------------------------------------------------- E. direct ES backtests
cut("E", r"\paragraph{Direct ES calibration backtests.}", "where the edge over GARCH-$t$ and FHS is significant at both levels.",
    r"""\paragraph{Direct ES calibration backtests.} Because exception counts test VaR and not ES, the exceedance-residual test of \citet{mcneilfrey2000} and the $Z_2$ statistic of \citet{acerbi2014backtesting} complement the FZ0 comparison \citep{noldeziegel2017elicitability, bayer2022esr}. Neither rejects the residual-hybrid's ES on the design panel or under annual refits, while GARCH-$t$'s 1\% ES is rejected as not severe enough. On the 2000--2013 holdout the 1\% test rejects every construction, in opposite directions: GARCH-$t$'s ES as not severe enough and the residual-hybrid's and FHS's as too severe. The full results are in the Online Appendix (Table~OA.5).""",
    "Direct ES calibration backtests", keep_head=r"\paragraph{Direct ES calibration backtests.}",
    fix=[("(Online Appendix)", ""), ("Online Appendix Table~OA.5", "Table~OA.5"), ("Online Appendix Table~OA.16", "Table~OA.16")])

# ---------------------------------------------------------------- F. IQN, training, estimator choice, GBC relation
a, b = span(m, r"\noindent\emph{Estimator choice.}", r"\subsection{Why shape misspecification", include_end=False)
tail = m[a:b]; m = m[:a] + m[b:]
cut("F", r"\emph{The neural implicit quantile network (IQN), the network at the core of GBC.}", r"\emph{Amortization.}",
    r"""\emph{The neural implicit quantile network (IQN).} The network at the core of GBC \citep{dabney2018iqn, polson2023gbc} takes the level $\tau$ as an input, so one set of weights carries the whole quantile curve. It is trained by the same pinball ERM with tail-weighted level sampling and is reported as a benchmark (Table~\ref{tab:noncore}). Its architecture and training, the choice between the two estimators, and the five ways the downside-risk target changes the GBC estimator are in the Online Appendix.

""", "The implicit quantile network, estimator choice, and relation to GBC", include_end=False,
    fix=[(r"\label{eq:iqn}", r"\label{eq:oaiqn}"), (r"Table~\ref{tab:noncore}", "Table~6 of the paper"),
         (r"\eqref{eq:shapeERM}", "equation~(3) of the paper")])
body = tail
for x, y in [(r"\eqref{eq:hybrid}", "equation~(5) of the paper"), (r"\eqref{eq:evt}", "equation~(4) of the paper"),
             (r"Section~\ref{sec:score}", "Section~2.5 of the paper")]:
    assert x in body, x; body = body.replace(x, y)
assert not re.findall(r"\\(?:ref|eqref)\{[^}]+\}", body), body
moved[-1] = (moved[-1][0], moved[-1][1] + "\n\n" + body.strip())
log.append("F'   estimator choice + GBC relation appended to F                 %5d -> %4d words" % (len(tail.split()), 0))

# ---------------------------------------------------------------- G. the filter-window implementation gap
cut("G", "One further\nimplementation gap", "shows no leakage.",
    r"""The first-stage filter is fit on the full pre-test history, so the calibration errors are held out from the shape learner and the EVT tail but not from the filter; re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (Online Appendix).""",
    "The filter window and the calibration split")

# ---------------------------------------------------------------- H. ES quadrature detail
cut("H", "The body branch is the minimum on about", "and 4.2--5.5 at $2.5\\%$).",
    r"""ES$_\alpha$ is the integral of \eqref{eq:hybrid} over $(0,\alpha]$, computed to convergence; the quadrature, the cost of the 20-node rule of earlier versions, the McNeil--Frey closed-form alternative, and the splice-level check are in the Online Appendix.""",
    "The ES integral",
    fix=[(r"\eqref{eq:hybrid}", "equation~(5) of the paper"), ("(Online Appendix)", "")])

# ---------------------------------------------------------------- I. composite score on the holdout (protocol)
cut("I", "As\na post-hoc check, the composite score also replicates on this holdout.", "not a frozen-specification test.",
    "", "The composite score on the 2000--2013 holdout",
    fix=[(r"Section~\ref{sec:frontier}", "Section~4 of the paper")])

# ---------------------------------------------------------------- J. one sentence of bootstrap provenance
x = r"""\citet{hansen2011mcs} implement the set with a fixed-length block bootstrap \citep{kunsch1989block}, and the stationary variant is the one used in the SPA test of \citet{hansen2005spa}. """
assert m.count(x) == 1; m = m.replace(x, ""); log.append("J    MCS bootstrap provenance sentence deleted (%d words)" % len(x.split()))

# ---------------------------------------------------------------- the new appendix section
sec = ["\n\\section{Detail moved from the main text}\\label{sec:moved}\n",
       "The paragraphs below were in the main text through R63 and moved here, unchanged except for cross-references, "
       "to shorten the paper. Each heading names the subject; the main text keeps a short pointer.\n"]
for h, t in moved:
    sec.append("\\subsection*{%s}\n%s\n" % (h, t))
anchor = r"\section{The two-regime example}"
assert o.count(anchor) == 1
o = o.replace(anchor, "\n".join(sec) + "\n" + anchor)
# equation (6), the IQN, left the main text: the score is now (6)
assert o.count("equation~(7) of the paper") == 2
o = o.replace("equation~(7) of the paper", "equation~(6) of the paper")

print("\n".join(log))
print("main words: %d" % len(m.split()))
if not DRY:
    open(MAIN, "w", encoding="utf-8").write(m); open(OA, "w", encoding="utf-8").write(o); print("written")
