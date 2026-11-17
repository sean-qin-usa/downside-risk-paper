# make_r67_oa_section.py -- writes docs/r67_oa_section.tex, the online-appendix section added in R67, from committed
# result files only. docs/apply_R67.py inserts it. Run from the repo root.
import json
J=lambda p: json.load(open(p))
s68=J("results/paper/diagnostics/r68_shock_sign.json")["regions"]; g=J("results/paper/diagnostics/r66_group_breakdown.json")["groupings"]
ho={k:J("results/paper/diagnostics/holdout_frontier_results_strict_%s.json"%k) for k in ("train","test")}; ho["committed"]=J("results/paper/holdout_frontier_results.json")
pi={k:J("results/paper/diagnostics/pit_universe_results_strict_%s.json"%k) for k in ("train","test")}; pi["committed"]=J("results/paper/pit_universe_results.json")
rf=J("results/paper/diagnostics/r67_refit_benchmarks.json"); ca=J("results/case_studies/crossasset_results.json"); ph=J("results/case_studies/crossasset_posthoc.json")
T=[]; A=T.append
A(r"""\section{Checks added after the comparison audit}\label{sec:r67}
An audit of every comparison in the replication repository (\texttt{docs/AUDIT\_2026-11.md}) looked for estimation and information asymmetries between the models compared, look-ahead, row mismatches, pooling artifacts and inference errors. This section reports the checks that followed. Each run's predictions were written into its script before it ran and are graded in that file.

\subsection*{The sign of the shock}
Table~\ref{tab:sign} splits the rows of Table~2 of the paper by the sign of the largest residual in the 63-day window behind each row's score (\texttt{tools/r68\_shock\_sign.py}). The shock sizes are the same in the two halves of the top decile.

\begin{table}[htbp]\centering\begin{threeparttable}
\caption{Pinball edge over GARCH-$t$ (\%, per-date DM in parentheses) by the sign of the window's largest shock.}\label{tab:sign}\footnotesize
\begin{tabular}{lrccccc}\toprule
Region & Rows & Median shock & Pooled body & Accuracy layer & SAV-CAViaR & GJR-skew-$t$ \\ \midrule""")
lab={"top decile, largest shock POSITIVE":"Top decile, positive","top decile, largest shock NEGATIVE":"Top decile, negative","deciles 1-9, largest shock positive":"Deciles 1--9, positive","deciles 1-9, largest shock negative":"Deciles 1--9, negative"}
c=lambda d:"$%+.2f$ (%.1f)"%(d["edge_pct"],d["DM"])
for k,r in s68.items(): A("%s & %s & %.1f & %s & %s & %s & %s \\\\"%(lab[k],"{:,}".format(r["rows"]),r["median_shock_size"],c(r["body"]),c(r["engine"]),c(r["sav_caviar"]),c(r["gjr_skewt"])))
A(r"""\bottomrule\end{tabular}
\begin{tablenotes}\footnotesize\item Median shock: the largest residual of the window in conditional standard deviations of the GARCH-$t$ filter. Eleven-level pinball.\end{tablenotes}
\end{threeparttable}\end{table}

\subsection*{By kind of stock}
Table~\ref{tab:groups} breaks the canonical comparison down by SIC division, by tercile of each name's estimation-window volatility, and by size cohort (\texttt{tools/r66\_group\_breakdown.py}; the per-row losses come from a rerun of \texttt{job\_bench\_all.py} that reproduces the committed FZ0 to five decimals). A Wald test rejects equal FZ0 gains over GARCH-$t$ across SIC groups ($p=%.3f$ at 1\%%, $%.3f$ at 2.5\%%) and across volatility terciles ($p=%.3f$, $%.3f$). The top-decile pinball edge is positive and significant in every group.

\begin{table}[htbp]\centering\begin{threeparttable}
\caption{The accuracy layer against GARCH-$t$ and GJR-GARCH-skew-$t$ by kind of stock: per-date DM of the FZ0 difference (positive: accuracy layer lower), and the pooled body's top-decile pinball edge over GARCH-$t$.}\label{tab:groups}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrcccccc}\toprule
 & & \multicolumn{2}{c}{vs GARCH-$t$} & \multicolumn{2}{c}{vs GJR-skew-$t$} & Names won & Top-decile \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}
Group & Names & 1\%% & 2.5\%% & 1\%% & 2.5\%% & at 1\%% & edge (DM) \\ \midrule"""%(g["sic"]["heterogeneity_engine_vs_garch_t"]["0.01"]["p"],g["sic"]["heterogeneity_engine_vs_garch_t"]["0.025"]["p"],g["vol"]["heterogeneity_engine_vs_garch_t"]["0.01"]["p"],g["vol"]["heterogeneity_engine_vs_garch_t"]["0.025"]["p"]))
short={"business services (incl. software)":"Business services, software","finance, insurance, real estate":"Finance, insurance, real estate","manufacturing (durable: machinery, electronics, computers)":"Durable manufacturing","manufacturing (nondurable: food, chemicals, pharma)":"Nondurable manufacturing","non-classifiable (mostly missing SIC)":"No SIC division","other (divisions with <10 names)":"Other divisions (under 10 names each)","transport, communications, utilities":"Transport, communications, utilities","wholesale and retail trade":"Wholesale and retail trade","high volatility tercile":"High-volatility tercile","low volatility tercile":"Low-volatility tercile","mid volatility tercile":"Mid-volatility tercile","largecap":"Large caps","smallcap":"Small caps"}
for gi,gn in enumerate(("sic","vol","cohort")):
    if gi: A(r"\midrule")
    for l,r in g[gn]["groups"].items():
        f1,f2=r["fz0_0.01"],r["fz0_0.025"]; td=r["top_decile_body_vs_garch_t_pinball"]
        A("%s & %d & $%+.1f$ & $%+.1f$ & $%+.1f$ & $%+.1f$ & %d/%d & $%+.2f\\%%$ (%.1f) \\\\"%(short[l],r["n_names"],f1["garch_t"]["DM"],f2["garch_t"]["DM"],f1["gjr_skewt"]["DM"],f2["gjr_skewt"]["DM"],f1["garch_t"]["names_engine_lower"],f1["garch_t"]["names"],td["edge_pct"],td["DM"]))
A(r"""\bottomrule\end{tabular}\end{threeparttable}\end{table}

\subsection*{Calendar overlap in the holdout and the point-in-time universe}
Per-name 60/40 splits put test starts between 2008-01-25 and 2009-03-10 in the holdout and between 2003 and 2009 in the point-in-time universe, so the pooled learner trains on dates that are other names' test dates. Table~\ref{tab:strict} repeats both jobs with the pooled learner trained only before the earliest test start, and with scoring restricted to dates after the last training date.

\begin{table}[htbp]\centering\begin{threeparttable}
\caption{Top-cell and overall pinball edge over GARCH-$t$ with the calendar overlap removed (\%, per-date DM).}\label{tab:strict}\footnotesize
\begin{tabular}{llccrr}\toprule
Panel & Run & Top cell & Overall & Training rows & Test rows \\ \midrule""")
def hrow(lab,d,tr): A("Holdout 2000--2013 & %s & $%+.2f$ (%.2f) & $%+.2f$ (%.2f) & %s & %s \\\\"%(lab,d["top_decile_mk63"]["edge_pct"],d["top_decile_mk63"]["DM_stat"],d["overall"]["edge_pct"],d["overall"]["DM_stat"],tr,"{:,}".format(d["n_test"])))
hrow("as published",ho["committed"],"414,616"); hrow("train before earliest test start",ho["train"],"396,531"); hrow("score after last training date",ho["test"],"414,616")
A(r"\midrule")
def prow(lab,d,tr): A("Point-in-time universe & %s & $%+.2f$ (%.2f) & $%+.2f$ (%.2f) & %s & %s \\\\"%(lab,d["top_bucket_frozen"]["edge_pct"],d["top_bucket_frozen"]["DM_stat"],d["overall"]["edge_pct"],d["overall"]["DM_stat"],tr,"{:,}".format(d["n_test"])))
prow("as published",pi["committed"],"307,685"); prow("train before earliest test start",pi["train"],"130,977"); prow("score after last training date",pi["test"],"307,685")
A(r"""\bottomrule\end{tabular}
\begin{tablenotes}\footnotesize\item Top cell: the top $\mathrm{mk}_{63}$ decile (holdout) or the frozen design-era top bucket (point-in-time). Overall percentages are row-weighted and the DMs are per-date, so a small percentage can carry a large statistic.\end{tablenotes}
\end{threeparttable}\end{table}

\subsection*{GAS, Taylor and SAV-CAViaR refitted}
\texttt{tools/r67\_refit\_benchmarks.py} first re-runs the paper's own fitting and reproduces the canonical losses exactly, then refits each model on the whole estimation window with more starts and a longer optimiser (Table~\ref{tab:refit}).

\begin{table}[htbp]\centering\begin{threeparttable}
\caption{Benchmarks as fitted in the paper and refitted, against the accuracy layer (DM positive: accuracy layer lower).}\label{tab:refit}\footnotesize
\begin{tabular}{llcccc}\toprule
 & & \multicolumn{2}{c}{As in the paper} & \multicolumn{2}{c}{Refitted} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}
Benchmark & Level & Loss & DM & Loss & DM \\ \midrule""")
for a,lv in (("0.01","1\\%"),("0.025","2.5\\%")):
    b=rf["fz0"][a]
    A("Taylor ES-CAViaR (FZ0) & %s & %.4f & %.2f & %.4f & %.2f \\\\"%(lv,b["taylor_pub"]["meanFZ0"],b["taylor_pub"]["engine_DM"],b["taylor_fix"]["meanFZ0"],b["taylor_fix"]["engine_DM"]))
    gf=b["gas_fix"]; A("GAS (FZ0) & %s & %.4f & %.2f & %s & %s \\\\"%(lv,b["gas_pub"]["meanFZ0"],b["gas_pub"]["engine_DM"],("%.4f"%gf["meanFZ0"]) if gf["nonfinite"]==0 else "unstable",("%.2f"%gf["engine_DM"]) if gf["nonfinite"]==0 else "---"))
cv=rf["caviar"]
A("SAV-CAViaR (pinball), overall & --- & %.5f & %.2f & %.5f & %.2f \\\\"%(cv["pub"]["mean_pinball"],cv["pub"]["engine_DM"]["overall"],cv["fix"]["mean_pinball"],cv["fix"]["engine_DM"]["overall"]))
A("SAV-CAViaR (pinball), top decile & --- &  & %.2f &  & %.2f \\\\"%(cv["pub"]["engine_DM"]["top_mk63_decile"],cv["fix"]["engine_DM"]["top_mk63_decile"]))
A(r"""\bottomrule\end{tabular}
\begin{tablenotes}\footnotesize\item Accuracy layer: FZ0 %.4f at 1\%% and %.4f at 2.5\%%, eleven-level pinball %.5f. With its scales estimated, the GAS diverges out of sample on some names at 2.5\%% (%d non-finite test rows).\end{tablenotes}
\end{threeparttable}\end{table}

\subsection*{The estimator outside the equity panel: a pre-registered study}
The protocol was fixed in \texttt{docs/CASE\_STUDY\_PRECOMMIT.md} before the confirmatory data were downloaded. Fifteen instruments are confirmatory (MSFT, PFE, WMT, CAT, BA, NFLX; IWM, XLU, HYG, LQD, SLV, DBA; GBP/USD, AUD/USD; ether) and seventeen, examined earlier, exploratory. The test window is 2020-01-02 to 2026-11-13. The estimator's pooled parts are trained once on the panel's pre-2020 rows, with every case-study ticker excluded. Because FZ0 is zero-homogeneous, loss differences are pooled across instruments by NYSE date. The primary test is non-inferiority with margin 0.01 in FZ0, Holm-adjusted over the seven benchmarks at each level. Table~\ref{tab:crossasset} gives the confirmatory result as registered, in which each per-instrument benchmark is refitted every January, and the result for all thirty-two instruments with every model fitted once before 2020. The second is post hoc: it was run after the first showed GJR-GARCH-skew-$t$ ahead at 2.5\%%, to test whether refitting the benchmarks yearly while the pooled parts stayed frozen explained that lead. It did: across the thirty-two instruments GJR-GARCH-skew-$t$'s mean advantage falls from 0.0077 and 0.0055 (1\%% and 2.5\%%) to 0.0024 and 0.0023 under the common protocol.

\begin{table}[htbp]\centering\begin{threeparttable}
\caption{Accuracy layer minus benchmark, pooled FZ0 (negative: accuracy layer lower), with the one-sided 95\%% upper bound. An asterisk marks non-inferiority at the Holm-adjusted 5\%% level against a margin of 0.01.}\label{tab:crossasset}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{Confirmatory 15, as registered} & \multicolumn{2}{c}{All 32, every model fitted once} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}
Benchmark & 1\%% & 2.5\%% & 1\%% & 2.5\%% \\ \midrule"""%(rf["fz0"]["0.01"]["engine_meanFZ0"],rf["fz0"]["0.025"]["engine_meanFZ0"],cv["engine_mean_pinball"],rf["fz0"]["0.025"]["gas_fix"]["nonfinite"]))
names={"hs500":"Historical simulation","ewma":"EWMA","garch_t":"GARCH-$t$","gjr_skewt":"GJR-GARCH-skew-$t$","fhs":"FHS","garch_evt":"GARCH-EVT","bteg":"Beta-$t$-EGARCH"}
for bkey,nm in names.items():
    cells=[]
    for a in ("0.01","0.025"):
        v=ca["confirmatory"]["pooled"][a]["fz0"]["engine_vs"][bkey]; cells.append("$%+.4f$ [$%+.4f$]%s"%(v["engine_minus_this"],v["upper95"],"*" if v["noninferior_at_5pct"] else ""))
    for a in ("0.01","0.025"):
        v=ph["noninferiority_all32"][a][bkey]; cells.append("$%+.4f$ [$%+.4f$]%s"%(v["engine_minus_benchmark"],v["upper95"],"*" if v["noninferior"] else ""))
    A("%s & %s \\\\"%(nm," & ".join(cells)))
mc=ca["confirmatory"]["pooled"]; fl=ph["flagged_days_vs_garch_t"]
A(r"""\bottomrule\end{tabular}
\begin{tablenotes}\footnotesize\item Per-date Newey--West(10) inference on the NYSE calendar. On the confirmatory set the 90\%% Model Confidence Set on FZ0 keeps every model except EWMA at 1\%% and, at 2.5\%%, keeps GJR-GARCH-skew-$t$, Beta-$t$-EGARCH, the accuracy layer and its variants and excludes GARCH-$t$, FHS, GARCH-EVT, historical simulation and EWMA.\end{tablenotes}
\end{threeparttable}\end{table}

Graded hypotheses on the confirmatory set: non-inferiority to GARCH-$t$ held at 2.5\%% and was not shown at 1\%%; non-inferiority to historical simulation, EWMA, FHS and GARCH-EVT held at 2.5\%%, and only against EWMA at 1\%%; it was not shown against GJR-GARCH-skew-$t$ or Beta-$t$-EGARCH at either level; the accuracy layer is in the 90\%% FZ0 Model Confidence Set at both levels; and non-inferiority to SAV-CAViaR on pinball held at 2.5\%% only. The rule for changing the recommended configuration to the flexible tail on the Beta-$t$-EGARCH scale was not met.

Three further analyses of the saved daily losses are post hoc (\texttt{tools/r67\_case\_posthoc.py}). Flagged days are those on which each instrument's $\mathrm{mk}_{63}$ exceeds its pre-2020 90th percentile, %.1f\%% of days. On flagged days Beta-$t$-EGARCH's single-level pinball gain over GARCH-$t$ is $%+.1f\%%$ (DM %.1f) at 1\%% and $%+.1f\%%$ (DM %.1f) at 2.5\%% on the fourteen stocks, and $%+.1f\%%$ (DM %.1f) and $%+.1f\%%$ (DM %.1f) on the eighteen other instruments. The flexible tail on the Beta-$t$-EGARCH scale gains $%+.1f\%%$ (DM %.1f) and $%+.1f\%%$ (DM %.1f) on the stocks' flagged days, and the accuracy layer $%+.1f\%%$ and $%+.1f\%%$. The score-gated scale's FZ0 gain over GARCH-$t$ across the thirty-two instruments is $%+.4f$ (DM %.2f) at 1\%% and $%+.4f$ (DM %.2f) at 2.5\%%.
"""%(ph["flagged_share_pct"],
  *[fl["14 stocks"]["flagged"]["bteg"][a][k] for a in ("0.01","0.025") for k in ("pinball_gain_pct","pinball_DM")],
  *[fl["18 non-stocks"]["flagged"]["bteg"][a][k] for a in ("0.01","0.025") for k in ("pinball_gain_pct","pinball_DM")],
  *[fl["14 stocks"]["flagged"]["body_bteg"][a][k] for a in ("0.01","0.025") for k in ("pinball_gain_pct","pinball_DM")],
  fl["14 stocks"]["flagged"]["engine"]["0.01"]["pinball_gain_pct"],fl["14 stocks"]["flagged"]["engine"]["0.025"]["pinball_gain_pct"],
  ph["gate"]["0.01"]["garch_t"]["gain"],ph["gate"]["0.01"]["garch_t"]["DM"],ph["gate"]["0.025"]["garch_t"]["gain"],ph["gate"]["0.025"]["garch_t"]["DM"]))
open("docs/r67_oa_section.tex","w").write("\n".join(T)+"\n"); print("wrote docs/r67_oa_section.tex", len("\n".join(T).split()), "words")
