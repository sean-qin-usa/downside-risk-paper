# apply_R67.py -- R67: corrections from the comparison-fairness audit (docs/AUDIT_2026-11.md) and the checks run after it.
# Exact-unique replacements; every anchor is checked before anything is written. Ledger: docs/CHANGE_LEDGER_R67.md.
# Run from the repo root:  python docs/apply_R67.py   (--dry to check without writing)
import sys, json
DRY = "--dry" in sys.argv
MAIN = "submission/paper_A_jfec.tex"; OA = "submission/paper_A_jfec_online_appendix.tex"; BIB = "submission/refs_v3.bib"
m = open(MAIN, encoding="utf-8").read(); o = open(OA, encoding="utf-8").read(); b = open(BIB, encoding="utf-8").read()
n_edits = [0]
def M(a, r):
    global m
    assert m.count(a) == 1, ("MAIN", a[:70], m.count(a)); m = m.replace(a, r); n_edits[0] += 1
def O(a, r, count=1):
    global o
    assert o.count(a) == count, ("OA", a[:70], o.count(a)); o = o.replace(a, r); n_edits[0] += 1

# ================================================================ abstract and introduction
M(r"in the score's top decile and nothing measurable elsewhere.",
  r"in the score's top decile, more after positive shocks than negative ones, and little elsewhere.")
M(r"and in the other nine deciles it is within noise.",
  r"the ninth decile carries a smaller edge ($+0.46\%$, DM 2.9), and the other eight are within noise.")
M(r"learns to pull its tail in.",
  r"learns to pull its tail in. The sign of the shock matters in the way the jump literature predicts: in individual stocks a negative jump raises subsequent volatility and a positive one lowers it \citep{pattonsheppard2015}, and the flexible model's top-decile edge is $+3.9\%$ after a positive shock (DM 7.1) and $+1.6\%$ after a negative one (DM 5.4).")
a = m.index(r"A rule that runs Beta-$t$-EGARCH on those days and GARCH-$t$ otherwise, with"); e = m.index("everywhere.", a) + len("everywhere.")
m = m[:a] + r"On the panel, a rule that runs Beta-$t$-EGARCH on those days and GARCH-$t$ otherwise is significantly better than GARCH-$t$ on FZ0 at both levels and statistically equal to running the robust filter everywhere. The rule was designed on those rows, and on thirty-two instruments outside the panel it does not beat GARCH-$t$ (Online Appendix), so it describes where the gain sits and is not offered as a tested rule." + m[e:]; n_edits[0] += 1
M(r"across a panel, on days a score flags in advance.",
  r"across a panel, on days a score flags in advance. The evidence is for individual US stocks. On fourteen stocks outside the training pool the robust filter's post-shock advantage replicates, and on eighteen instruments in other asset classes it does not appear (Section~\ref{sec:frontier}).")
M(r"so separating them improves forecasts most after a jump \citep{andersen2007roughing, corsi2010threshold}.",
  r"so separating them improves forecasts most after a jump \citep{andersen2007roughing, corsi2010threshold}. In individual stocks jumps mostly accompany firm-specific news such as earnings \citep{leemykland2008, lee2012jumps}, a negative jump raises subsequent volatility while a positive one lowers it \citep{pattonsheppard2015}, and additive outliers left in the data bias GARCH volatility forecasts \citep{fransesghijsels1999}.")

# ================================================================ Section 3: where the calendar overlap actually is
M("Stale benchmark parameters therefore do not\nproduce the conditional result.",
  "Stale benchmark parameters therefore do not\nproduce the conditional result. On the design panel every name's test window begins on the same date (2020-08-06), so that panel has no such overlap and the calendar split above tests a different thing, the refitting of the benchmark. The overlap is in the 2000--2013 holdout and in the point-in-time universe, whose per-name test windows begin between 2003 and 2009. There the check is to train the pooled learner only on dates before the earliest test start, or to score only dates after the last training date, and both are reported with each result.")
a = m.index(r"carries $+3.17\%$ at DM 7.0, larger than in the survivor-tilted holdout."); e = a + len(r"carries $+3.17\%$ at DM 7.0, larger than in the survivor-tilted holdout.")
m = m[:a] + r"carries $+3.17\%$ at DM 7.0. Those figures include calendar overlap in the pooled training rows (Section~\ref{sec:calendarleak}). With the pooled learner trained only before the earliest test start the top bucket carries $+1.25\%$ (DM 4.1) and the overall edge is $+0.04\%$; scored only on dates after the last training date, $+1.77\%$ (DM 3.1) and $+0.47\%$ (DM 4.4). The first variant also removes 57\% of the pooled training rows, so the two bracket the result." + m[e:]; n_edits[0] += 1

# ================================================================ Section 4: the holdout, restated from the result file and with the strict runs
a = m.index(r"the top-decile edge is $+0.89\%$ with per-date DM $2.16$"); e = m.index(r"and the overall edge is $+0.02\%$.", a) + len(r"and the overall edge is $+0.02\%$.")
m = m[:a] + r"the top-decile edge is $+0.92\%$ with per-date DM $2.19$ ($p=0.014$, 1{,}465 dates), deciles one through nine sit between $-0.16\%$ and $+0.29\%$, and the overall edge is $+0.04\%$. Per-name test windows in this era begin between January 2008 and March 2009, so the pooled learner's training rows overlap other names' test dates (Section~\ref{sec:calendarleak}). Trained only on dates before the earliest test start, the top-decile edge is $+0.82\%$ (DM $1.88$, one-sided $p=0.030$) with the same threshold shape, and scored only on dates after the last training date it is $+0.83\%$ (DM $4.1$). The overlap accounts for about a tenth of the edge." + m[e:]; n_edits[0] += 1
a = m.index(r"$+3.17\%$ at DM 7.0 on the point-in-time universe"); e = m.index(r"Section~\ref{sec:fwer}.", a) + len(r"Section~\ref{sec:fwer}.")
m = m[:a] + r"$+1.25\%$ to $+1.77\%$ (DM 3.1 to 4.1) on the point-in-time universe of Section~\ref{sec:fwer} once the calendar overlap is removed." + m[e:]; n_edits[0] += 1

# ================================================================ Section 4: same-rows numbers from the seeded canonical file; SAV-CAViaR
M(r"carries $+2.48\%$ (DM 10.1) in the top kurtosis decile", r"carries $+2.44\%$ (DM 10.2) in the top kurtosis decile")
M(r"by $+2.64\%$ (DM 8.9) in the top kurtosis decile", r"by $+2.60\%$ (DM 9.0) in the top kurtosis decile")
M(r"by $+2.18\%$ (DM 9.6) and $+0.21\%$ (DM 4.6)", r"by $+2.14\%$ (DM 9.7) and $+0.21\%$ (DM 4.7)")
M(r"beats it by $+2.32\%$ (DM 9.7) in the top decile", r"beats it by $+2.29\%$ (DM 9.7) in the top decile")
a = m.index(r"and the two are within a third of a percent in every region"); e = m.index(r"engine over CAViaR)", a) + len(r"engine over CAViaR)")
m = m[:a] + r"and it is slightly ahead of the engine, by $0.37\%$ in the top kurtosis decile (DM $-2.0$), $0.13\%$ overall (DM $-1.6$) and $0.40\%$ in the top composite decile (DM $-2.4$). That lead depends on the optimiser. Fitted from eight starts with a longer budget, SAV-CAViaR fits the estimation window better and forecasts slightly worse, and the two tie in every region (top decile DM $-0.8$, overall DM $-0.4$; Online Appendix)" + m[e:]; n_edits[0] += 1

# ================================================================ Section 4.2 rewritten: what the evidence beyond the panel is worth
a = m.index(r"\subsection{The frontier across universes}"); e = m.index(r"\subsection{Regime dependence and model selection}")
m = m[:a] + r"""\subsection{Beyond the equity panel}

Two kinds of evidence bear on other markets, and they differ in weight. The first is a study of the paper's estimator on fifteen instruments fixed, with its hypotheses, margins and tests, before their data were examined: six stocks outside the training pool, four exchange-traded funds on equities and credit, two on commodities, two currency pairs and ether, tested from 2020 to November 2026. Seventeen further instruments examined beforehand are reported as exploratory. The pooled parts of the estimator were trained once on the panel before 2020 and applied unchanged. Comparable is defined as a pooled FZ0 no more than 0.01 above the benchmark's, about the size of the estimator's gain over GARCH-$t$ on the panel. At 2.5\% the accuracy layer is comparable to GARCH-$t$, FHS, GARCH-EVT, EWMA and historical simulation on the fifteen, and to every benchmark, GJR-GARCH-skew-$t$ and Beta-$t$-EGARCH included, when all thirty-two are pooled and every model is fitted once before 2020. At 1\% the comparison is inconclusive: the estimator's point estimate is below GARCH-$t$'s, and the upper bounds on its excess loss against the GARCH family (0.014 to 0.018) exceed the margin. When the per-instrument benchmarks are refitted every year and the pooled parts are not, GJR-GARCH-skew-$t$ is ahead at 2.5\% on the fifteen (DM 2.06); most of that lead is the refit asymmetry and it falls within noise under a common protocol.

The post-shock mechanism is specific to stocks. On the days the score flags, Beta-$t$-EGARCH beats GARCH-$t$ by $4.8\%$ of 1\% pinball loss (DM 2.5) and $3.4\%$ at 2.5\% (DM 2.2) on the fourteen stocks, none of which is in the training pool, and it does not on the eighteen instruments in other asset classes ($-5.8\%$, DM $-1.6$, and $-2.0\%$, DM $-1.1$). The flexible tail on the robust scale gains $7.2\%$ and $5.0\%$ on the stocks' flagged days (DM 3.3 and 2.5); the accuracy layer on the GARCH-$t$ scale does not. We therefore claim the mechanism for individual stocks only. The protocol, the graded hypotheses and the deviations from it are in the Online Appendix.

The second kind of evidence is the set of exploratory sweeps of earlier versions: 24 currency pairs, 26 country indices, 43 cross-asset instruments and the Korea-2026 crash (Online Appendix Table~OA.20). They are not tests of the estimator. Each entry compares a standalone per-asset model with GARCH-$t$ on pinball loss; in the cross-asset sweep the entry is the better of FHS and the residual-hybrid on the test sample, FHS on 39 of the 43; and the currency entries come from the last of several specifications. We report them as description and draw no conclusion from their significance counts.

""" + m[e:]; n_edits[0] += 1

# ================================================================ Section 5: how GAS and Taylor are fitted
M(r"Each is estimated per name on the same panel by minimizing the same FZ0 loss, and under a three-start optimizer both converge for all $200$ names.",
  r"Each is estimated per name by minimizing the same FZ0 loss on the last 1{,}200 days of the estimation window, from three starts. The GAS scales are fixed at that window's empirical VaR and ES, so it is a restricted form of the model; it scores below GARCH-$t$ and is a weak benchmark. Refitted on the whole estimation window with a longer optimiser, the Taylor model converges for all $200$ names and the comparison is unchanged (DM 3.2 at 1\%, a tie at 2.5\%), though in the top decile of the score it is ahead of the accuracy layer (DM $-1.8$ and $-2.6$); with its scales estimated the GAS improves at 1\% without closing the gap (DM 10.8) and is unstable out of sample at 2.5\% (Online Appendix).")

# ================================================================ Section 6 and the conclusion: the gate
M(r"In a run whose predictions were written beforehand, its joint score is", r"Its joint score is")
M(r"Where tail accuracy is the only objective, running the robust scale everywhere does as well.",
  r"Where tail accuracy is the only objective, running the robust scale everywhere does as well. The gate was designed after the robust filter's top-decile lead had been seen on these rows, so this is a description of the panel and not an out-of-sample test. On thirty-two instruments outside the panel, with every model fitted before 2020, the same rule does not beat GARCH-$t$ (DM $-1.1$ at 1\% and $-0.4$ at 2.5\%), and on the fourteen stocks among them its gain on flagged days is positive and not significant (Online Appendix).")
M(r"The gate threshold was fixed before the first test year", r"The gate threshold was fixed before the first test year, but the gate was designed after the top-decile result was known")
a = m.index(r"It can still choose the Stage-1 scale"); e = m.index(r"(Section~\ref{sec:applications}).", a) + len(r"(Section~\ref{sec:applications}).")
m = m[:a] + r"On the panel a gate that moves to Beta-$t$-EGARCH in the score's top decile is significantly better than GARCH-$t$ at both levels; it was designed on those rows and does not beat GARCH-$t$ on instruments outside the panel (Section~\ref{sec:applications})." + m[e:]; n_edits[0] += 1
M(r"identifies them a day ahead; elsewhere they do not measurably improve on it.",
  r"identifies them a day ahead; the next tenth carries a smaller edge and elsewhere they do not measurably improve on it.")
M(r"its value is diagnostic, in showing where and why the standard filter is stressed.",
  r"its value is diagnostic, in showing where and why the standard filter is stressed. The evidence is for individual stocks: on other asset classes the estimator is comparable to the standard benchmarks at 2.5\% and the post-shock mechanism does not appear.")

# ================================================================ online appendix
D = json.load(open("results/paper/bench_all_results.json"))["pinball"]; V = D["vs_garch_t"]; EV = D["engine_vs"]
f = lambda d: "$%+.2f$ (%.2f)" % (d["edge_pct"], d["DM"])
def row(label, key, mcs=""):
    v = V[key]; ev = f(EV[key]["top_mk63_decile"]) if key in EV else "---"
    return "%s & %s & %s & %s & %s & %s \\\\" % (label, f(v["top_mk63_decile"]), f(v["bulk_mk63_d1to9"]), f(v["overall"]), ev, mcs)
new_rows = "\n".join([row("Engine (accuracy layer)", "engine"), row("GARCH $+$ pooled body (no EVT)", "body", r"$\checkmark$"),
    "GARCH(1,1)-$t$ & (reference) & (reference) & (reference) & %s &  \\\\" % f(EV["garch_t"]["top_mk63_decile"]),
    row("GJR-GARCH-skew-$t$", "gjr_skewt"), row("GARCH-FHS (pooled)", "fhs_pool"), row("GARCH-FHS (per name)", "fhs_name"),
    row("GARCH-FHS (rolling 500d)", "fhs_roll500"), row("GARCH-EVT (pooled tail)", "evt_pool"), row("GARCH-EVT (per name)", "evt_name"),
    row("SAV-CAViaR", "sav_caviar", r"$\checkmark$"), row("EWMA (RiskMetrics)", "ewma"), row("Historical simulation (500d)", "hs500")])
a = o.index(r"Engine (accuracy layer) & $+2.48$ (10.08)"); e = o.index("\\\\", o.index(r"Historical simulation (500d) & $+0.97$ (1.88)", a)) + 2
o = o[:a] + new_rows + o[e:]; n_edits[0] += 1
O(r"Top misspecification decile (pooled body; accuracy layer) & $+2.98\%$; $+2.48\%$ & DM 10.5; 10.1 \\",
  r"Top misspecification decile (pooled body; accuracy layer) & $+2.98\%$; $+2.44\%$ & DM 10.5; 10.2 \\")
O(r"the GAS and Taylor models and SAV-CAViaR are estimated per name on the estimation window;",
  r"and SAV-CAViaR are estimated per name on the estimation window (SAV-CAViaR from one start with a 300-iteration budget; from eight starts with 1{,}500 iterations it fits that window better, forecasts slightly worse, and ties the accuracy layer in every region); the GAS and Taylor models are estimated on its last 1{,}200 days, the GAS with its scales fixed at that window's empirical VaR and ES;")
O(r"The edge is larger than in the survivor-tilted holdout, so the observation",
  r"These figures include calendar overlap in the pooled training rows (Section~3 of the paper): with the pooled learner trained only before the earliest test start the top bucket carries $+1.25\%$ (DM 4.1), and scored only after the last training date $+1.77\%$ (DM 3.1). With the overlap in, the edge is larger than in the survivor-tilted holdout, so the observation")
O(r"Hyperinflation FX (ARS, EGP) &", r"Hyperinflation FX (ARS, EGP; exploratory sweep, last of several specifications) &")
O(r"Frontier equity indices (4 of 8) &", r"Frontier equity indices (4 of 8; exploratory sweep) &")
O(r"Table~\ref{tab:universes} is the cross-market table of Section~4.2 of the paper.",
  r"Table~\ref{tab:universes} collects the exploratory sweeps of earlier versions. They are not tests of the paper's estimator. Each entry compares a standalone per-asset model with GARCH-$t$ on pinball loss. In the cross-asset block the ``nonparametric'' entry is the better of FHS and the residual-hybrid on the test sample (FHS on 39 of the 43 instruments), so the ``13/43 significant wins'' line counts the better of two models chosen after the fact; four of the seven Holm survivors of Table~OA.4 are FHS wins. The currency block comes from the last of several specifications, an earlier one of which had the learner losing on USDEGP, and the earlier specifications are not in the 93-comparison family. The scripts that produced these sweeps are not in the replication repository. The tested cross-asset evidence is the pre-registered study of the next section.")

R = open("docs/r67_oa_section.tex", encoding="utf-8").read()
anchor = r"\section{The two-regime example}"; assert o.count(anchor) == 1; o = o.replace(anchor, R + "\n" + anchor); n_edits[0] += 1

# ================================================================ bibliography
NEW = r"""
@article{leemykland2008,
  author  = {Lee, Suzanne S. and Mykland, Per A.},
  title   = {Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics},
  journal = {Review of Financial Studies},
  volume  = {21},
  number  = {6},
  pages   = {2535--2563},
  year    = {2008}
}

@article{lee2012jumps,
  author  = {Lee, Suzanne S.},
  title   = {Jumps and Information Flow in Financial Markets},
  journal = {Review of Financial Studies},
  volume  = {25},
  number  = {2},
  pages   = {439--479},
  year    = {2012}
}

@article{pattonsheppard2015,
  author  = {Patton, Andrew J. and Sheppard, Kevin},
  title   = {Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility},
  journal = {Review of Economics and Statistics},
  volume  = {97},
  number  = {3},
  pages   = {683--697},
  year    = {2015}
}

@article{fransesghijsels1999,
  author  = {Franses, Philip Hans and Ghijsels, Hendrik},
  title   = {Additive Outliers, {GARCH} and Forecasting Volatility},
  journal = {International Journal of Forecasting},
  volume  = {15},
  number  = {1},
  pages   = {1--9},
  year    = {1999}
}
"""
assert "leemykland2008" not in b; b = b + NEW
print("edits applied: %d" % n_edits[0])
if not DRY:
    open(MAIN, "w", encoding="utf-8").write(m); open(OA, "w", encoding="utf-8").write(o); open(BIB, "w", encoding="utf-8").write(b)
    open("preprint/refs_v3.bib", "w", encoding="utf-8").write(b); print("written")
