# r89_breach_table.py -- WHERE THE VaR BREACHES ARE: EARNINGS DAYS, MARKET-WIDE DOWN DAYS, EVERYTHING ELSE.
#
# Every test day of every name is one of four types: the earnings reaction day; the day after; a market-wide down day
# (not one of the first two; the equal-weighted return of the panel that day is at or below the 5th percentile of the
# pre-test days); any other day. For GARCH-t, the masked calendar GARCH-t and the neural network with calendar inputs:
# the share of days of each type, the breach rate on it, and its share of all breaches, at 1% and 2.5%.
# The market-wide type is defined by the day's own return, so it is a description of where breaches fall, not a forecast.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r81 (GARCH-t is breached on 24.6% of reaction
# days at 2.5% and 18.8% at 1% on the CRSP panel; reaction days are 1.6% of days).
#  B1. Market-wide down days carry at least 25% of GARCH-t's 2.5% breaches on both panels.
#  B2. Reaction days carry 12% to 16% of GARCH-t's 2.5% breaches and 24% to 32% of its 1% breaches; with the masked
#      calendar GARCH-t their share is under 3% at both levels.
#  B3. More than half of the breaches of the masked calendar GARCH-t are on days that are neither.
# Usage: python tools/r89_breach_table.py canon | us_largecap
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r89"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0)]; R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json")))
mk = A.groupby("date")["y"].mean(); pre_dates = A[~A.test].date.unique(); thr = float(np.quantile(mk.reindex(pre_dates).values, 0.05)); T = A[A.test & A.U].reset_index(drop=True); m = T.date.map(mk).values; R0 = T.r0.values == 1; R1 = (T.r1.values == 1) & ~R0
typ = np.where(R0, "earnings reaction day", np.where(R1, "day after", np.where(m <= thr, "market-wide down day", "any other day"))); c = R82["multipliers_c0_c1"]["masked calendar GARCH-t"]; mult = np.where(R0, c[0], np.where(R1, c[1], 1.0))
OUT = {"universe": WHAT, "rows": int(len(T)), "market_wide_threshold_pct": round(thr, 2), "predictions_written_before_run": "see the script header (B1-B3)", "share_of_days_pct": {t: round(100 * float((typ == t).mean()), 2) for t in np.unique(typ)}}
for a in ("0.01", "0.025"):
    Q = {"GARCH-t": T["GARCH-t__" + a].values, "masked calendar GARCH-t": T.mmu.values + T.msig.values * mult * tq(float(a), T.mnu.values), "neural network, calendar inputs": T["neural network, calendar inputs__" + a].values}; OUT[a] = {}
    for nm, q in Q.items():
        h = T.y.values <= q; OUT[a][nm] = {"breach_rate_all_days_pct": round(100 * float(h.mean()), 2), "by_type": {t: {"breach_rate_pct": round(100 * float(h[typ == t].mean()), 2), "share_of_breaches_pct": round(100 * float(h[typ == t].sum() / h.sum()), 1)} for t in np.unique(typ)}}
        print(WHAT, a, "%-32s all %.2f%% | %s" % (nm, 100 * h.mean(), {t: (v["breach_rate_pct"], v["share_of_breaches_pct"]) for t, v in OUT[a][nm]["by_type"].items()}))
print("share of days", OUT["share_of_days_pct"], "threshold", OUT["market_wide_threshold_pct"]); json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
