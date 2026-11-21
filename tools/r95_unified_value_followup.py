# r95_unified_value_followup.py -- FOLLOW-UP TO r94: THE FORMULA CHECKED AFTER THE FACT, AND ESTIMATED BEFORE THE FACT ON ALL PRE-TEST ROWS.
#
# WRITTEN AFTER r94's FIRST RESULTS AND A DIAGNOSIS (2026-11-20). r94's recipe under-predicted: on the CRSP panel 6.0%
# against a realized 7.6% for the calendar and about zero against 0.8% for past-return information. Seen in the
# diagnosis: r94 calibrates on one slice of the pre-test rows, and on the CRSP panel that slice is December 2018 to
# August 2020. In it the slope of log|z| on the fitted log scale is 0.31 for past-return information, against 1.05 in
# the test period and 0.8 to 1.5 in every other year; the reaction-day gap in log|z| is 1.16 there against 1.31 before
# and 1.23 after. So r94 tested the formula and one fragile way of estimating its inputs together. This script takes
# the two apart. The general form of the expression: if the truth is z = s * eps and a model applies the multiplier m
# (not necessarily s), its gain over the benchmark is
#     V(m; s) = sum_t sigma_t s_t [ EL(q_t / s_t) - EL(q_t m_t / s_t) ]  /  sum_t sigma_t s_t EL(q_t / s_t),
# which is r94's V when m = s.
# Part A, the formula after the fact. Same regression as r94 (training rows). On the test rows: beta, the slope of
#   log|z| on the fitted value, gives the true scale s = exp(beta * fit); (tau, nu) are fitted to z / s. The expression
#   is then compared with the realized gain of two multipliers: r94's (shrunk on the calibration slice) and the
#   unshrunk fit. Nothing is predicted in advance here; the question is whether one scale family with one tail
#   accounts for gains from zero to ten percent.
# Part B, before the fact, on all pre-test rows. Five blocks of pre-test dates in time order; every pre-test row gets
#   a fitted value from a regression that left its block out; b and (tau, nu) come from all pre-test rows; V is
#   averaged over them. The multiplier applied in the test period is the regression on all pre-test rows times b.
#   Band: block bootstrap over pre-test dates, blocks of 63 trading days, all names together.
#
# PREDICTIONS (written knowing r94's CRSP results and the slopes above, test-period slopes included; before this ran).
#  A1. The expression is within the larger of 0.5 points and 10% of the realized gain in every cell, for both
#      multipliers, at 2.5%; within the larger of 1.0 point and 15% at 1%.
#  A2. Correlation above 0.995 across the comparisons of each panel.
#  B1. Calendar alone: V within 1.0 point of the realized gain at 2.5% on both panels; band under 1.2 points either
#      side.
#  B2. Calendar, days since and past-return state: V below the realized gain, by less than 2.0 points.
#  B3. Past-return state on ordinary days: V at least 0.3 points below the realized gain on the CRSP panel (the low
#      2019-2020 slope is in the pre-test rows), within 0.4 points on US large caps.
#  B4. Realized and V agree within 1.645 combined standard errors in at least 12 of 16 cells at 2.5%.
# Usage: python tools/r95_unified_value_followup.py canon | us_largecap        (needs data/panels/<panel>_calcache.pkl)
import os; os.environ["OMP_NUM_THREADS"] = "4"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r95"); os.makedirs(OUTD, exist_ok=True); NB = int(os.environ.get("R95_BOOT", "200")); t0 = time.time()
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def kappa(a, nu): k = np.sqrt(nu / (nu - 2)); x = stats.t.ppf(a, nu); q = x / k; f = k * stats.t.pdf(x, nu); return q * q * f / ((1 / k) * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1))
def tfit(x, w):   # weighted maximum likelihood for a Student-t with location zero: (nu, tau)
    def nll(p): nu = 2.05 + math.exp(p[0]); tau = math.exp(p[1]); return -float((w * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(tau) - (nu + 1) / 2 * np.log1p((x / tau) ** 2 / nu))).sum()) / float(w.sum())
    r = optimize.minimize(nll, [math.log(3.0), math.log(0.75)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}); return 2.05 + math.exp(r.x[0]), math.exp(r.x[1])
def slope(g, l, w=None):
    w = np.ones(len(g)) if w is None else w; w = w / w.sum(); mg = (w * g).sum(); vg = (w * (g - mg) ** 2).sum(); return float((w * (g - mg) * (l - (w * l).sum())).sum() / vg) if vg > 1e-10 else 0.0
def dm(d, dates):
    s = pd.Series(d, index=dates).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return float(s.mean() / math.sqrt(max(v / len(s), 1e-18)))
HGB = lambda: HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.06, random_state=0)

X5 = ["logsig", "zl1", "absz5", "zstd21", "fracdn5"]; A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A["dow"] = pd.DatetimeIndex(A.date.values).dayofweek.astype(float); A["Dc"] = np.clip(A.Dm.values, -0.5, 0.98)
A = A[np.isfinite(A.r0) & np.isfinite(A[X5 + ["ds", "Dc", "wm"]]).all(1) & (A.U | ~A.test)].reset_index(drop=True)
BLOCKS = {"GARCH-t, all days": dict(sig="sig", mu="mu", nu="nu", rows=np.ones(len(A), bool), sets={"weekday (placebo)": ["dow"], "earnings calendar": ["r0", "r1"], "calendar and days since the last report": ["r0", "r1", "ds"], "calendar, days since and the state of past returns": ["r0", "r1", "ds"] + X5 + ["Dc", "wm"]}),
          "masked calendar GARCH-t, ordinary days": dict(sig="msig", mu="mmu", nu="mnu", rows=((A.r0 != 1) & (A.r1 != 1)).values, sets={"weekday (placebo)": ["dow"], "shock share and origin": ["Dc", "wm"], "state of past returns": X5 + ["ds"], "state of past returns, shock share and origin": X5 + ["ds", "Dc", "wm"]})}
OUT = {"universe": WHAT, "bootstrap_draws": NB, "predictions": "see the script header (A1-A2, B1-B4); written after r94's first results", "A_after_the_fact": {}, "B_before_the_fact": {}}; CA = []; CB = []
for bname, B in BLOCKS.items():
    D = A[B["rows"]].reset_index(drop=True); y, mu, sg, nu = D.y.values, D[B["mu"]].values, D[B["sig"]].values, D[B["nu"]].values; z = (y - mu) / sg; lz = np.log(np.maximum(np.abs(z), 0.02)); tr = D.train.values; te = D.test.values; pr = ~te; ca = pr & ~tr; dates = pd.DatetimeIndex(D.date.values)
    kp = {a: float(kappa(a, D.groupby("name")[B["nu"]].first().values).mean()) for a in AL}; pd_ = np.sort(dates[pr].unique().values); fold = np.searchsorted(pd_[(np.arange(1, 5) * len(pd_)) // 5], dates.values, side="right"); ipr = np.flatnonzero(pr)
    sub = np.random.default_rng(0).choice(ipr, min(80000, len(ipr)), replace=False); sdi = np.searchsorted(pd_, dates.values[sub]); nblk = int(math.ceil(len(pd_) / 63)); OUT["A_after_the_fact"][bname] = {}; OUT["B_before_the_fact"][bname] = {}; Lb = {a: pin(y, mu + sg * tq(a, nu), a) for a in AL}
    def realized(m, a): Li = pin(y, mu + sg * m * tq(a, nu), a); g_ = 100 * float((Lb[a] - Li)[te].sum() / Lb[a][te].sum()); t_ = dm((Lb[a] - Li)[te], dates[te]); return g_, t_
    for cname, cols in B["sets"].items():
        Xv = D[cols].values
        # ---- Part A: the regression of r94, the truth measured on test rows
        g = HGB().fit(Xv[tr], lz[tr]).predict(Xv); gc = g - np.median(g[pr]); bcal = max(slope(gc[ca], lz[ca]), 0.0); beta = slope(gc[te], lz[te]); s = np.exp(beta * gc[te]); nu1, tau1 = tfit(z[te] / s, np.ones(int(te.sum())))
        rowA = {"slope_on_calibration_rows_r94": round(bcal, 3), "slope_on_test_rows": round(beta, 3), "tail_index_test": round(nu1, 2), "tau_test": round(tau1, 3)}
        for lab, bapp in (("multiplier of r94", bcal), ("unshrunk fit", 1.0)):
            m = np.exp(bapp * gc); rowA[lab] = {}
            for a in AL:
                q = tq(a, nu[te]); lu = sg[te] * s * EL(q / s, tau1, nu1, a); Vx = 100 * float((lu - sg[te] * s * EL(q * m[te] / s, tau1, nu1, a)).sum() / lu.sum()); g_, t_ = realized(m, a); rowA[lab][str(a)] = {"expression_pct": round(Vx, 2), "realized_pct": round(g_, 2), "DM": round(t_, 2)}; CA.append((a, Vx, g_))
        OUT["A_after_the_fact"][bname][cname] = rowA
        # ---- Part B: cross-fitted on all pre-test rows
        go = np.full(len(D), np.nan)
        for k in range(5): f = pr & (fold != k); h = pr & (fold == k); go[h] = HGB().fit(Xv[f], lz[f]).predict(Xv[h])
        gf = HGB().fit(Xv[pr], lz[pr]).predict(Xv); gco = go - np.median(go[pr]); gcf = gf - np.median(gf[pr])
        def est(idx, w): b = max(slope(gco[idx], lz[idx], w), 0.0); nu0, tau0 = tfit(z[idx] / np.exp(b * gco[idx]), w); return b, nu0, tau0
        def V(par, idx, w):
            b, nu0, tau0 = par; s_ = np.exp(b * gco[idx]); o = {}
            for a in AL: q = tq(a, nu[idx]); lu = w * sg[idx] * s_ * EL(q / s_, tau0, nu0, a); o[a] = 100 * float((lu - w * sg[idx] * s_ * EL(q, tau0, nu0, a)).sum() / lu.sum())
            return o
        par = est(ipr, np.ones(len(ipr))); Vp = V(par, ipr, np.ones(len(ipr))); Vs = V(est(sub, np.ones(len(sub))), sub, np.ones(len(sub))); rng = np.random.default_rng(1); bs = {a: [] for a in AL}; bb = []
        for _ in range(NB):
            cnt = np.zeros(len(pd_))
            for s_ in rng.integers(0, len(pd_), nblk): cnt[np.arange(s_, s_ + 63) % len(pd_)] += 1
            w = cnt[sdi]; k_ = w > 0; pb = est(sub[k_], w[k_]); vb = V(pb, sub[k_], w[k_]); bb.append(pb[0])
            for a in AL: bs[a].append(vb[a] - Vs[a])
        b = par[0]; m = np.exp(b * gcf); rowB = {"shrink_b": round(b, 3), "b_band_90": [round(float(np.quantile(bb, 0.05)), 3), round(float(np.quantile(bb, 0.95)), 3)], "b_by_block": [round(slope(gco[pr & (fold == k)], lz[pr & (fold == k)]), 2) for k in range(5)], "slope_of_applied_fit_on_test_rows": round(slope(gcf[te], lz[te]), 3), "tail_index": round(par[1], 2), "tau": round(par[2], 3), "sd_of_log_scale_pre_test": round(float((b * gco[pr]).std()), 4)}
        for a in AL:
            g_, t_ = realized(m, a); lo, hi = (Vp[a] + np.quantile(bs[a], q_) for q_ in (0.05, 0.95)); seV = (hi - lo) / 3.29; seR = abs(g_ / t_) if abs(t_) > 1e-6 else float("nan")
            rowB[str(a)] = {"V_pct": round(Vp[a], 2), "V_band_90": [round(float(lo), 2), round(float(hi), 2)], "second_order_pct": round(50 * kp[a] * float((b * gco[pr]).var()), 2), "realized_pct": round(g_, 2), "DM": round(t_, 2), "realized_se": round(seR, 2) if np.isfinite(seR) else None, "z_of_realized_minus_V": round((g_ - Vp[a]) / math.sqrt(seV ** 2 + seR ** 2), 2) if np.isfinite(seR) and seV + seR > 0 else None}; CB.append((a, Vp[a], g_, rowB[str(a)]["z_of_realized_minus_V"]))
        OUT["B_before_the_fact"][bname][cname] = rowB
        print("%5.0fs %-22s | %-50s | A: slope cal %.2f test %.2f; r94 mult %5.2f vs %5.2f, unshrunk %5.2f vs %5.2f | B: b %.2f [%.2f,%.2f] V %5.2f [%5.2f,%5.2f] real %5.2f (DM %4.1f) z %s | 1%%: V %5.2f real %5.2f" % (time.time() - t0, bname[:22], cname[:50], bcal, beta, rowA["multiplier of r94"]["0.025"]["expression_pct"], rowA["multiplier of r94"]["0.025"]["realized_pct"], rowA["unshrunk fit"]["0.025"]["expression_pct"], rowA["unshrunk fit"]["0.025"]["realized_pct"],
              b, *rowB["b_band_90"], rowB["0.025"]["V_pct"], *rowB["0.025"]["V_band_90"], rowB["0.025"]["realized_pct"], rowB["0.025"]["DM"], rowB["0.025"]["z_of_realized_minus_V"], rowB["0.01"]["V_pct"], rowB["0.01"]["realized_pct"]), flush=True)
OUT["summary"] = {"A": {str(a): {"comparisons": sum(1 for c in CA if c[0] == a), "correlation": round(float(np.corrcoef([c[1] for c in CA if c[0] == a], [c[2] for c in CA if c[0] == a])[0, 1]), 4), "mean_absolute_gap_points": round(float(np.mean([abs(c[1] - c[2]) for c in CA if c[0] == a])), 2), "largest_gap_points": round(float(max(abs(c[1] - c[2]) for c in CA if c[0] == a)), 2),
                                 "within_tolerance": sum(1 for c in CA if c[0] == a and abs(c[1] - c[2]) <= max((0.5 if a == 0.025 else 1.0), (0.10 if a == 0.025 else 0.15) * abs(c[2])))} for a in AL},
                  "B": {str(a): {"cells": sum(1 for c in CB if c[0] == a), "correlation": round(float(np.corrcoef([c[1] for c in CB if c[0] == a], [c[2] for c in CB if c[0] == a])[0, 1]), 4), "mean_absolute_gap_points": round(float(np.mean([abs(c[1] - c[2]) for c in CB if c[0] == a])), 2), "cells_where_realized_and_V_agree": sum(1 for c in CB if c[0] == a and (abs(c[3]) < 1.645 if c[3] is not None else abs(c[1] - c[2]) < 0.005))} for a in AL}}
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); print(json.dumps(OUT["summary"]))
