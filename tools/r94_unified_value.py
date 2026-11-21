# r94_unified_value.py -- ONE FORMULA FOR THE VALUE OF ANY INFORMATION TO A VaR MODEL, WITH UNCERTAINTY BANDS.
#
# r83 priced information that sorts days into a few types (exact, tested on the calendar). r90 priced diffuse
# information to second order (a ceiling, too low by more than half for the calendar). One expression covers both.
# Write the benchmark's standardized return as z = s(I) * eps: s(I) is the ratio of the true scale to the benchmark's
# given information I, and eps has one distribution, a Student-t with scale tau and nu degrees of freedom, whatever I
# is. The benchmark forecasts its quantile q every day; the informed model forecasts s(I) * q. With EL the expected
# pinball loss of r83,
#     V(I) = sum_t sigma_t s_t [ EL(q_t / s_t) - EL(q_t) ]  /  sum_t sigma_t s_t EL(q_t / s_t).
# If s takes a few values this is r83's partition form (which also lets the tail differ by type); its second-order
# expansion is r90's (1/2) kappa Var(log s). Estimation, the same for every I and using no test row:
#   1. log s(I): a regression of log|z| on I (boosted trees, squared loss, r90's settings) on the training rows;
#   2. on the calibration rows (pre-test, not used in 1) the slope b of log|z| on the fitted value (never below 0)
#      shrinks it, s = exp(b * (fit - its pre-test median)), and (tau, nu) are fitted to z / s by maximum likelihood;
#   3. V is the expression above averaged over all pre-test rows.
# Band: a block bootstrap of the calibration rows (blocks of 21 trading days, all names together), steps 2 and 3
# redone; 5th to 95th percentile. The regression of step 1 is held fixed, so the band leaves its error out.
# Realized: the same s applied to the benchmark's VaR on the test rows, against the benchmark.
#
# Information sets. Benchmark GARCH-t, all days: a placebo (weekday); the earnings calendar; the calendar and days
# since the last report; those with the state of past returns (the trees' five inputs, the shock share D and its
# market-wide share). Benchmark masked calendar GARCH-t, ordinary days: the placebo; shock share and origin; the
# state of past returns (r90's six inputs); both.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r82-r90 (the calendar: V 7.2% and 5.3% by the
# partition form, realized 7.8% and 5.5%; flexible models with the calendar 10.2% to 10.7% over GARCH-t on the CRSP
# panel and up to 7.7% on US large caps; past-return state on ordinary days: second-order ceiling 1.2% and 0.6%, a
# regression multiplier realized 1.5% and 0.7%; shock share and origin 0.35% and 0.30%; weekday zero).
#  K1. Calendar alone: V is within 1.0 point of the partition form's value and of the realized gain at 2.5%.
#  K2. Calendar, days since and past-return state: V is 8.5% to 11.5% on the CRSP panel and 5.5% to 8.0% on US large
#      caps at 2.5%; the realized gain is within 1.5 points of V; the flexible quantile models given the same
#      information are within 1.5 points of V.
#  K3. Placebo: V under 0.1% and realized within 0.2 points of zero, both benchmarks, both panels.
#  K4. Ordinary days: past-return state V 0.8% to 1.8% (CRSP) and 0.3% to 1.0% (US), realized within 0.4 points;
#      shock share and origin V 0.2% to 0.8%, realized within 0.3 points.
#  K5. Over the 16 cells at 2.5%: correlation of V and realized above 0.99, mean absolute gap under 0.5 points; at 1%
#      the mean absolute gap is under 1.0 point.
#  K6. Realized and V agree (difference under 1.645 combined standard errors) in at least 13 of 16 cells at 2.5%;
#      the band for the calendar alone is narrower than 1.0 point either side.
#  K7. The second-order form is within 25% of V in every cell where V is under 2% and above 0.2%, and under half of V
#      in the cells that contain the calendar.
# Usage: python tools/r94_unified_value.py canon | us_largecap        (needs data/panels/<panel>_calcache.pkl)
import os; os.environ["OMP_NUM_THREADS"] = "4"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r94"); os.makedirs(OUTD, exist_ok=True); NB = int(os.environ.get("R94_BOOT", "200")); t0 = time.time()
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def kappa(a, nu): k = np.sqrt(nu / (nu - 2)); x = stats.t.ppf(a, nu); q = x / k; f = k * stats.t.pdf(x, nu); return q * q * f / ((1 / k) * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1))
def tfit(x, w):   # weighted maximum likelihood for a Student-t with location zero: (nu, tau)
    def nll(p): nu = 2.05 + math.exp(p[0]); tau = math.exp(p[1]); return -float((w * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(tau) - (nu + 1) / 2 * np.log1p((x / tau) ** 2 / nu))).sum()) / float(w.sum())
    r = optimize.minimize(nll, [math.log(3.0), math.log(0.75)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}); return 2.05 + math.exp(r.x[0]), math.exp(r.x[1])
def dm(d, dates):
    s = pd.Series(d, index=dates).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return float(s.mean() / math.sqrt(max(v / len(s), 1e-18)))

X5 = ["logsig", "zl1", "absz5", "zstd21", "fracdn5"]; A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A["dow"] = pd.DatetimeIndex(A.date.values).dayofweek.astype(float); A["Dc"] = np.clip(A.Dm.values, -0.5, 0.98)
A = A[np.isfinite(A.r0) & np.isfinite(A[X5 + ["ds", "Dc", "wm"]]).all(1) & (A.U | ~A.test)].reset_index(drop=True)
BLOCKS = {"GARCH-t, all days": dict(sig="sig", mu="mu", nu="nu", rows=np.ones(len(A), bool), sets={"weekday (placebo)": ["dow"], "earnings calendar": ["r0", "r1"], "calendar and days since the last report": ["r0", "r1", "ds"], "calendar, days since and the state of past returns": ["r0", "r1", "ds"] + X5 + ["Dc", "wm"]}),
          "masked calendar GARCH-t, ordinary days": dict(sig="msig", mu="mmu", nu="mnu", rows=((A.r0 != 1) & (A.r1 != 1)).values, sets={"weekday (placebo)": ["dow"], "shock share and origin": ["Dc", "wm"], "state of past returns": X5 + ["ds"], "state of past returns, shock share and origin": X5 + ["ds", "Dc", "wm"]})}
R83 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r83", WHAT + ".json")))["A"]; OUT = {"universe": WHAT, "bootstrap_draws": NB, "predictions_written_before_run": "see the script header (K1-K7)", "partition_form_r83_calendar": {a: {"V_pct": R83[a]["V_pct"], "realized_pct": R83[a]["realized_garch_t_with_multipliers_pct"]} for a in ("0.01", "0.025")}, "blocks": {}}
for bname, B in BLOCKS.items():
    D = A[B["rows"]].reset_index(drop=True); y, mu, sg, nu = D.y.values, D[B["mu"]].values, D[B["sig"]].values, D[B["nu"]].values; z = (y - mu) / sg; lz = np.log(np.maximum(np.abs(z), 0.02)); tr = D.train.values; te = D.test.values; pr = ~te; ca = pr & ~tr; dates = pd.DatetimeIndex(D.date.values)
    kp = {a: float(kappa(a, D.groupby("name")[B["nu"]].first().values).mean()) for a in AL}; cd = np.sort(dates[ca].unique().values); cdi = np.searchsorted(cd, dates[ca].values); nblk = int(math.ceil(len(cd) / 21)); sub = np.random.default_rng(0).choice(np.flatnonzero(pr), min(60000, int(pr.sum())), replace=False)
    OUT["blocks"][bname] = {"rows_train": int(tr.sum()), "rows_calibration": int(ca.sum()), "rows_test": int(te.sum()), "calibration_from": str(cd[0])[:10], "calibration_to": str(cd[-1])[:10], "kappa": {str(a): round(kp[a], 2) for a in AL}, "cells": {}}
    if bname.startswith("GARCH-t"): OUT["blocks"][bname]["flexible_quantile_models_with_the_calendar_realized_pct"] = {nm: {str(a): round(100 * float((pin(y, mu + sg * tq(a, nu), a) - pin(y, D["%s__%s" % (nm, a)].values, a))[te & np.isfinite(D["%s__%s" % (nm, a)].values)].sum() / pin(y, mu + sg * tq(a, nu), a)[te & np.isfinite(D["%s__%s" % (nm, a)].values)].sum()), 2) for a in AL} for nm in ("trees, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs", "neural network, calendar inputs")}
    for cname, cols in B["sets"].items():
        g = HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.06, random_state=0).fit(D.loc[tr, cols].values, lz[tr]).predict(D[cols].values); gc = g - np.median(g[pr]); gca, lca, zca = gc[ca], lz[ca], z[ca]
        def est(w):   # step 2 on weighted calibration rows
            k = w > 0; ww = w[k] / w[k].sum(); gg, ll = gca[k], lca[k]; mg = (ww * gg).sum(); vg = (ww * (gg - mg) ** 2).sum(); b = max(float((ww * (gg - mg) * (ll - (ww * ll).sum())).sum() / vg), 0.0) if vg > 1e-10 else 0.0; nu0, tau0 = tfit(zca[k] / np.exp(b * gg), w[k]); return b, nu0, tau0
        def V(par, idx):   # step 3
            b, nu0, tau0 = par; s = np.exp(b * gc[idx]); o = {}
            for a in AL: q = tq(a, nu[idx]); lu = sg[idx] * s * EL(q / s, tau0, nu0, a); o[a] = 100 * float((lu - sg[idx] * s * EL(q, tau0, nu0, a)).sum() / lu.sum())
            return o
        par = est(np.ones(int(ca.sum()))); Vp = V(par, np.flatnonzero(pr)); Vs = V(par, sub); rng = np.random.default_rng(1); bs = {a: [] for a in AL}
        for _ in range(NB):
            cnt = np.zeros(len(cd)); st = rng.integers(0, len(cd), nblk)
            for s_ in st: cnt[np.arange(s_, s_ + 21) % len(cd)] += 1
            vb = V(est(cnt[cdi]), sub)
            for a in AL: bs[a].append(vb[a] - Vs[a])
        b = par[0]; s = np.exp(b * gc); row = {"inputs": cols, "shrink_b": round(b, 3), "tail_index": round(par[1], 2), "tau": round(par[2], 3), "sd_of_log_scale_pre_test": round(float((b * gc[pr]).std()), 4), "largest_and_smallest_scale_1st_99th_percentile": [round(float(np.quantile(s[pr], 0.01)), 3), round(float(np.quantile(s[pr], 0.99)), 3)]}
        for a in AL:
            Lb = pin(y, mu + sg * tq(a, nu), a); Li = pin(y, mu + sg * s * tq(a, nu), a); gain = 100 * float((Lb - Li)[te].sum() / Lb[te].sum()); t_ = dm((Lb - Li)[te], dates[te]); lo, hi = (Vp[a] + np.quantile(bs[a], q_) for q_ in (0.05, 0.95)); seV = (hi - lo) / 3.29; seR = abs(gain / t_) if abs(t_) > 1e-6 else float("nan")
            row[str(a)] = {"V_pct": round(Vp[a], 2), "V_band_90": [round(float(lo), 2), round(float(hi), 2)], "second_order_pct": round(50 * kp[a] * float((b * gc[pr]).var()), 2), "realized_pct": round(gain, 2), "DM": round(t_, 2), "realized_se": round(seR, 2), "z_of_realized_minus_V": round((gain - Vp[a]) / math.sqrt(seV ** 2 + seR ** 2), 2) if np.isfinite(seR) else None}
        OUT["blocks"][bname]["cells"][cname] = row; print("%5.0fs %-40s | %-52s b %.2f nu %.1f | 2.5%%: V %5.2f [%5.2f,%5.2f] 2nd %5.2f real %5.2f (DM %5.1f) z %s | 1%%: V %5.2f [%5.2f,%5.2f] real %5.2f" % (time.time() - t0, bname, cname, b, par[1], row["0.025"]["V_pct"], *row["0.025"]["V_band_90"], row["0.025"]["second_order_pct"], row["0.025"]["realized_pct"], row["0.025"]["DM"], row["0.025"]["z_of_realized_minus_V"], row["0.01"]["V_pct"], *row["0.01"]["V_band_90"], row["0.01"]["realized_pct"]), flush=True)
cells = [c for B_ in OUT["blocks"].values() for c in B_["cells"].values()]; OUT["summary"] = {str(a): {"cells": len(cells), "correlation_V_realized": round(float(np.corrcoef([c[str(a)]["V_pct"] for c in cells], [c[str(a)]["realized_pct"] for c in cells])[0, 1]), 4), "mean_absolute_gap_points": round(float(np.mean([abs(c[str(a)]["V_pct"] - c[str(a)]["realized_pct"]) for c in cells])), 2),
                  "cells_where_realized_and_V_agree": int(sum(1 for c in cells if (abs(c[str(a)]["z_of_realized_minus_V"]) < 1.645 if c[str(a)]["z_of_realized_minus_V"] is not None else abs(c[str(a)]["V_pct"] - c[str(a)]["realized_pct"]) < 0.005)))} for a in AL}
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); print(json.dumps(OUT["summary"]), json.dumps(OUT["blocks"]["GARCH-t, all days"].get("flexible_quantile_models_with_the_calendar_realized_pct")))
