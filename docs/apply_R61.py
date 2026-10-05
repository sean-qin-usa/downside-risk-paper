# apply_R61.py -- R61/R62: the mechanism behind the top-decile joint-loss result, the power of the holdout
# exception test, the design-effect note, and the printed numbers that moved when five more jobs were seeded.
# Sources: docs/R61_DIAGNOSTICS.md, docs/R62_SPLICE_PRECOMMIT.md, docs/R62_SPLICE_RESULTS.md and the result
# files named in docs/CHANGE_LEDGER_R61.md. Exact-unique replacements, every anchor checked before any write.
# Run from the repo root:  python docs/apply_R61.py        (--dry to check without writing)
import sys, io

FILES = ["submission/paper_A_jfec.tex", "preprint/paper_A_ssrn.tex"]
OA = ["submission/paper_A_jfec_online_appendix.tex", "preprint/paper_A_ssrn_online_appendix.tex"]
EDITS, OAED = [], []
E = lambda n, b, a: EDITS.append((n, b, a))
O = lambda n, b, a: OAED.append((n, b, a))

# ---------------------------------------------------------------- 1. the mechanism, in 5.1
E("N01 5.1 mechanism: the EVT envelope explains the top-decile joint-loss result",
r"""at 1\% the mean FZ0 gap is 0.011 in the top misspecification decile and 0.009 in the other nine (DM 3.4 and 4.7).""",
r"""at 1\% the mean FZ0 gap is 0.011 in the top misspecification decile and 0.009 in the other nine (DM 3.4 and 4.7). That flatness is a property of the splice. Dropping the EVT branch and scoring the pooled body alone, the body beats the accuracy layer in the top decile by 0.068 at 1\% (DM $-6.4$) and 0.060 at 2.5\% (DM $-6.8$), and ties it in the other nine deciles (0.002 and 0.001, DM 0.4 and 0.2). The pooled GPD is unconditional, so it sits at a fixed level in standardized-residual terms, $-2.55$ at 1\% and $-1.88$ at 2.5\%, while the body pulls its own tail quantile in from $-2.53$ to $-2.25$ across the deciles as the score rises, and the minimum of the two therefore overrides the body by a margin that widens from 0.19 to 0.52. Anchoring the same GPD to each row's own body quantile at the splice level, under a rule fixed before the run, removes the top-decile deficit (DM $+0.1$ against $-6.6$ for the unconditional tail at the same splice level) and raises the 97.5\% breach rate from 2.78\% to 3.09\%. The unconditional tail therefore buys about 0.3 points of 97.5\% coverage at a cost of 0.06 to 0.07 in top-decile FZ0 and 0.013 to 0.019 overall, and the paper keeps it for the coverage.""")

# ---------------------------------------------------------------- 2. power of the holdout exception test
E("N02 5.1 summary: the holdout exception test has little power",
r"""That ranking survives annual refitting, with exception counts near nominal through a 2008-crisis window.""",
r"""That ranking survives annual refitting, with exception counts near nominal through a 2008-crisis window, although the clustered exception test on that window has little power: its standard error for the accuracy layer is 0.53 points at $1\%$, so it does not reject any breach rate below $2.05\%$, and 0.28 points at $2.5\%$.""")

E("N03 exception summary: the holdout pass has little power",
r"""and passes on the 2000--2013 holdout ($t=0.25$).""",
r"""and passes on the 2000--2013 holdout ($t=0.25$), a test with little power there: its clustered standard error for the accuracy layer is 0.53 points at $1\%$ and 0.28 at $2.5\%$, so it rejects no breach rate below $2.05\%$ at $1\%$ or inside $1.95$--$3.05\%$ at $2.5\%$.""")

# ---------------------------------------------------------------- 3. the design-effect note
TBL = ["submission/tables/tab_deployed.tex"]   # generated candidate table; NOT \\input by either build
TBED = []
TB = lambda n, b, a: TBED.append((n, b, a))
TB("N04 tab_deployed note: the design effect is not an intra-date correlation out of era",
r"""The design-effect row is the ratio of the iid to the clustered variance; it implies an intra-date breach correlation of a few per cent, which is why the pooled statistic rejects on deviations of under a tenth of a percentage point.""",
r"""The design-effect row is the ratio of the iid to the Newey--West(10) clustered variance, which is why the pooled statistic rejects on deviations of under a tenth of a percentage point. It is not an intra-date correlation. It combines clustering within a date with serial dependence across dates, and the two can be separated: on this panel the within-date component alone is 8.6 and implies an intra-date breach correlation of 0.04, while on the 2000--2013 holdout the reported 824 is a within-date component of 31, implying 0.16, multiplied by a serial-correlation factor of 25.""")

# ---------------------------------------------------------------- 4. numbers that moved when the jobs were seeded
E("N05 calendar split top decile, reseeded",
r"""the top-decile edge survives ($+2.47\%$, DM 6.22)""",
r"""the top-decile edge survives ($+2.45\%$, DM 6.03)""")

E("N06a point-in-time universe, reseeded",
r"""the overall edge is $+0.58\%$ (DM 8.96) and the frozen top bucket
carries $+2.94\%$ at DM 6.6""",
r"""the overall edge is $+0.61\%$ (DM 9.16) and the frozen top bucket
carries $+3.17\%$ at DM 7.0""")

E("N06b point-in-time universe, second mention",
r"""the frozen thresholds carry
$+2.94\%$ at DM 6.6 on the point-in-time universe""",
r"""the frozen thresholds carry
$+3.17\%$ at DM 7.0 on the point-in-time universe""")

E("N07 strict-split DMs, reseeded",
r"""Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.1 and 5.4). The strict estimator's gap to the full-window benchmark ($-0.6$, $-2.5$)""",
r"""Re-estimating with the filter stopped at the calibration boundary leaves the accuracy advantage intact (DM 5.0 and 5.1). The strict estimator's gap to the full-window benchmark ($-0.7$, $-2.6$)""")

# ---------------------------------------------------------------- online appendix
O("P01 ES backtests section: the holdout test's power",
r"""\item Inference for the McNeil--Frey and $Z_2$ $p$-values is a stationary block""",
r"""\item On the 2000--2013 holdout the clustered exception test has little power: the accuracy layer's clustered standard error is 0.53 points at $1\%$, so breach rates below $2.05\%$ are not rejected, against 0.25 points for GARCH-$t$ on the same rows. The gap is serial dependence in the breach series rather than cross-name clustering, whose implied intra-date correlation is 0.16 for the accuracy layer and 0.18 for GARCH-$t$.
\item Inference for the McNeil--Frey and $Z_2$ $p$-values is a stationary block""")

def main():
    dry = "--dry" in sys.argv
    T, problems = {}, []
    for files, eds in ((FILES, EDITS), (OA, OAED), (TBL, TBED)):
        for f in files:
            t = T.get(f) or io.open(f, encoding="utf-8").read(); T[f] = t
            for name, before, _ in eds:
                c = t.count(before)
                if c != 1: problems.append(f"{f}: {name}: anchor found {c} times")
    if problems:
        print("NOT APPLIED:\n  " + "\n  ".join(problems)); sys.exit(1)
    if dry:
        print("dry run OK: %d main x %d, %d OA x %d, %d table-file edits" % (len(EDITS), len(FILES), len(OAED), len(OA), len(TBED))); return
    for files, eds in ((FILES, EDITS), (OA, OAED), (TBL, TBED)):
        for f in files:
            for _, before, after in eds: T[f] = T[f].replace(before, after, 1)
    for f, t in T.items(): io.open(f, "w", encoding="utf-8", newline="\n").write(t)
    print("R61 applied: %d main x %d, %d OA x %d, %d table-file" % (len(EDITS), len(FILES), len(OAED), len(OA), len(TBED)))

if __name__ == "__main__":
    main()
