# r86_price_partitions.py -- THE EQUATION OF r83 APPLIED TO OTHER INFORMATION: A PLACEBO, FED DAYS, THE SHOCK STATE, THE VOLATILITY STATE.
#
# r83 priced one piece of information, the earnings calendar. The equation prices any information that sorts days into
# types d: V(a) = sum_d p_d m_d [EL_d(q) - EL_d(c_d q)] / sum_d p_d m_d EL_d(q), with (tau_d, nu_d) fitted on pre-test
# standardized residuals of each type. Here the base model is the masked calendar GARCH-t, the days are the ordinary
# days (not a reaction day or the day after), and the partitions are:
#   weekday                  a placebo: five types that should carry nothing
#   Fed decision days        the scheduled decision day, the day after, other days (CRSP panel only: the dates start in
#                            2016, after the US large-cap pre-test window)
#   shock share              six bins of D, the share of today's variance that comes from the largest recent shock (r82)
#   shock share and origin   the same, with D above 0.05 split by the shock's origin (firm-specific, mixed, market-wide)
#   volatility against usual quintiles of today's sigma relative to the name's pre-test level
#   recent residual scale    quintiles of the standard deviation of the last 21 standardized residuals
# For each: V from pre-test rows, and the realized gain on test rows of one multiplier per type chosen on pre-test rows.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r82-r85 (the shock correction added 0.8% to the
# masked model on the CRSP panel and nothing on US large caps; trees given the volatility level gained 1.1% on the CRSP
# panel on ordinary days and nothing on US large caps; trees given the recent residual scale gained 0.6% and 0.4%).
#  Q1. Weekday: V is below 0.15% on both panels at 2.5%, and the realized gain is within 0.15 points of zero.
#  Q2. Fed days (CRSP): V is below 0.3%, and the realized gain is within 0.3 points of V.
#  Q3. Shock share: V is between 0.4% and 1.5% on the CRSP panel, realized within 0.4 points of it; on US large caps V
#      is below 0.6%, realized within 0.3 points.
#  Q4. Knowing the origin raises V by at least 0.1 points on the CRSP panel, and the realized gain rises with it.
#  Q5. The two volatility-state partitions: V between 0.2% and 1.2% on the CRSP panel, realized within 0.4 points.
#  Q6. Over all partitions, both panels and both levels, together with the calendar (r83), V and the realized gain
#      have a correlation above 0.95 and a mean absolute gap under 0.4 points.
# Usage: python tools/r86_price_partitions.py canon | us_largecap
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r86"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0) & (A.r0 != 1) & (A.r1 != 1) & np.isfinite(A.Dm)].reset_index(drop=True); A["z"] = (A.y - A.mmu) / A.msig
A["v"] = A.groupby("name")["z"].transform(lambda s: s.rolling(21, min_periods=8).std().shift(1)); A["lsdev"] = A.logsig - A["name"].map(A[~A.test].groupby("name")["logsig"].mean()); A = A[np.isfinite(A.v) & np.isfinite(A.lsdev) & (A.U | ~A.test)].reset_index(drop=True)
pre = ~A.test.values; te = A.test.values; NU = A.groupby("name")["mnu"].first().values; dates = pd.DatetimeIndex(A.date.values); y, mu, sg, nu = A.y.values, A.mmu.values, A.msig.values, A.mnu.values
qcut = lambda x, k: np.searchsorted(np.quantile(x[pre], np.arange(1, k) / k), x).astype(str)   # bins from pre-test quantiles
Dc = np.clip(A.Dm.values, -0.5, 0.98); db = np.digitize(Dc, [0.02, 0.05, 0.10, 0.20, 0.35]); org = np.where(A.wm.values < 0.1, "f", np.where(A.wm.values < 0.5, "x", "m"))
PARTS = {"weekday": dates.dayofweek.astype(str), "shock share": db.astype(str), "shock share and origin": np.where(db >= 2, np.char.add(db.astype(str), org), db.astype(str)), "volatility against usual": qcut(A.lsdev.values, 5), "recent residual scale": qcut(A.v.values, 5)}
RES = {}
if WHAT == "canon":
    fd = pd.DatetimeIndex(json.load(open(os.path.join(ROOT, "data", "calendars", "fomc_scheduled_decision_dates.json"))))
    ud = np.sort(dates.unique().values); p0 = np.searchsorted(ud, fd.values); p0 = p0[p0 < len(ud)]; d0 = ud[p0]; d1 = ud[np.minimum(p0 + 1, len(ud) - 1)]
    PARTS["Fed decision days"] = np.where(np.isin(dates.values, d0), "decision day", np.where(np.isin(dates.values, d1), "day after", "other"))
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
CG = np.round(np.arange(0.5, 2.0001, 0.02), 2)
for name, lab in PARTS.items():
    pr = pre & (dates >= "2016-01-01") if name == "Fed decision days" else pre; types = [t for t in np.unique(lab) if (pr & (lab == t)).sum() >= 300]; okr = np.isin(lab, types); par = {}; cemp = {}
    for t in types:
        f = pr & (lab == t); nu_, _, tau_ = stats.t.fit(A.z.values[f], floc=0); q25 = tq(0.025, NU)
        par[t] = dict(tau=float(tau_), nu=float(nu_), p=float((lab[te & okr] == t).mean()), m=float(sg[f].mean() / sg[pr & okr].mean()), c=float(optimize.minimize_scalar(lambda c: EL(c * q25, tau_, nu_, 0.025).mean(), bounds=(0.3, 3.0), method="bounded").x))
        b = tq(0.025, nu[f]); cemp[t] = float(CG[int(np.argmin([pin(y[f], mu[f] + c * sg[f] * b, 0.025).sum() for c in CG]))])
    mult = np.ones(len(A));
    for t in types: mult[lab == t] = cemp[t]
    m_ = te & okr; row = {"types": len(types), "implied_multipliers": {t: round(par[t]["c"], 2) for t in types}, "multipliers_chosen_on_pre_test_rows": cemp, "tail_index_by_type": {t: round(par[t]["nu"], 1) for t in types}}
    for a in AL:
        q = tq(a, NU); V = 100 * sum(par[t]["p"] * par[t]["m"] * (EL(q, par[t]["tau"], par[t]["nu"], a) - EL(par[t]["c"] * q, par[t]["tau"], par[t]["nu"], a)).mean() for t in types) / sum(par[t]["p"] * par[t]["m"] * EL(q, par[t]["tau"], par[t]["nu"], a).mean() for t in types)
        Lb = pin(y, mu + sg * tq(a, nu), a); Li = pin(y, mu + sg * mult * tq(a, nu), a); row[str(a)] = {"V_pct": round(float(V), 2), "realized_pct": round(100 * float((Lb - Li)[m_].sum() / Lb[m_].sum()), 2), "DM": dm(Lb - Li, m_)}
    RES[name] = row; print("%-26s %s" % (name, {a: (row[a]["V_pct"], row[a]["realized_pct"], row[a]["DM"]) for a in ("0.025", "0.01")}), "| implied c", row["implied_multipliers"], "| chosen c", cemp)
R83 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r83", WHAT + ".json")))["A"]
json.dump({"universe": WHAT, "rows_pre_test": int(pre.sum()), "rows_test": int(te.sum()), "base_model": "masked calendar GARCH-t, ordinary days", "predictions_written_before_run": "see the script header (Q1-Q6)", "partitions": RES,
           "earnings_calendar_r83": {a: {"V_pct": R83[a]["V_pct"], "realized_pct": R83[a]["realized_garch_t_with_multipliers_pct"]} for a in ("0.01", "0.025")}}, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
