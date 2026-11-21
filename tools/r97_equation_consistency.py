# r97_equation_consistency.py -- IS THE EQUATION OF r83 RIGHT ABOUT ITS OWN SAMPLE? FOUR WAYS OF FITTING THE TAILS.
#
# WRITTEN AFTER r96's CONSISTENCY CHECK AND A DIAGNOSIS (2026-11-20); NO PREDICTIONS. r96 found that the partition
# form says 7.5% (CRSP) and 5.2% (US large caps) on the pre-test rows it is fitted to, where the same multipliers
# gain 6.8% and 3.8% on those rows. An expression that is right returns its own sample's gain. Two assumptions were
# found to fail on reaction days: the standardized return is not symmetric (the loss side is narrower before 2016 on
# US large caps), and it is smaller when the day's volatility is high (the fact behind r85's second clock), while the
# loss sum weights each day by its volatility. This script records the check.
# For the pre-test rows and for the test rows separately, with the multipliers of r96 (chosen on pre-test rows):
#   the gain those multipliers make on the rows, and the expression with each day type's Student-t fitted on the same
#   rows in four ways: both sides or the loss side only (z < 0), equal weights or weights proportional to sigma.
# The expression fitted on pre-test rows is also each variant's forecast for the test period.
# Also recorded: the reaction-day gap in mean log|z| by calendar year, to show how much the calendar effect moves.
# Usage: python tools/r97_equation_consistency.py canon | us_largecap     (needs the calendar cache and r96's result file)
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.025, 0.01); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r97"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def tfit(x, w):
    def nll(p): nu = 2.05 + math.exp(p[0]); tau = math.exp(p[1]); return -float((w * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(tau) - (nu + 1) / 2 * np.log1p((x / tau) ** 2 / nu))).sum()) / float(w.sum())
    r = optimize.minimize(nll, [math.log(3.0), math.log(0.75)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}); return 2.05 + math.exp(r.x[0]), math.exp(r.x[1])
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0) & (A.U | ~A.test)].reset_index(drop=True); y, mu, sg, nu = A.y.values, A.mu.values, A.sig.values, A.nu.values; z = (y - mu) / sg; lz = np.log(np.maximum(np.abs(z), 0.02)); te = A.test.values
typ = np.where(A.r0.values == 1, 0, np.where(A.r1.values == 1, 1, 2)); TY = ("reaction day", "day after", "other days"); C = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r96", WHAT + ".json")))["forms"]["partition form (s = 1)"]["by_type_pre_test"]; c = np.array([C[k]["multiplier"] for k in TY])
VAR = (("both sides, equal weights (r83)", 0, 0), ("both sides, sigma weights", 0, 1), ("loss side, equal weights", 1, 0), ("loss side, sigma weights", 1, 1)); gain = lambda m, a: 100 * float((pin(y, mu + sg * tq(a, nu), a) - pin(y, mu + sg * c[typ] * tq(a, nu), a))[m].sum() / pin(y, mu + sg * tq(a, nu), a)[m].sum())
def expr(fit_rows, eval_rows, left, wsig):   # tails fitted on fit_rows, the expression evaluated over eval_rows
    o = {}; par = {}
    for d in range(3): k = fit_rows & (typ == d) & ((z < 0) if left else True); par[d] = tfit(z[k], sg[k] if wsig else np.ones(int(k.sum())))
    for a in AL:
        num = den = 0.0
        for d in range(3): k = eval_rows & (typ == d); q = tq(a, nu[k]); lu = sg[k] * EL(q, par[d][1], par[d][0], a); den += lu.sum(); num += (lu - sg[k] * EL(c[d] * q, par[d][1], par[d][0], a)).sum()
        o[str(a)] = round(100 * num / den, 2)
    return o, {TY[d]: {"tail_index": round(par[d][0], 2), "tau": round(par[d][1], 3)} for d in range(3)}
yr = pd.DatetimeIndex(A.date.values).year; r0 = typ == 0; o_ = typ == 2
OUT = {"universe": WHAT, "note": "written after r96's consistency check; no predictions", "multipliers_from_r96_pre_test": {TY[d]: float(c[d]) for d in range(3)}, "gain_of_the_multipliers_pct": {"pre-test rows": {str(a): round(gain(~te, a), 2) for a in AL}, "test rows": {str(a): round(gain(te, a), 2) for a in AL}}, "expression_on_its_own_rows_pct": {}, "expression_fitted_on_pre_test_rows_as_a_forecast_for_the_test_rows_pct": {},
       "reaction_day_gap_in_mean_log_abs_z_by_year": {int(y_): round(float(lz[(yr == y_) & r0].mean() - lz[(yr == y_) & o_].mean()), 2) for y_ in sorted(set(yr)) if ((yr == y_) & r0).sum() >= 100},
       "reaction_days": {"correlation_of_sigma_and_abs_z": {"pre-test": round(float(np.corrcoef(sg[~te & r0], np.abs(z[~te & r0]))[0, 1]), 3), "test": round(float(np.corrcoef(sg[te & r0], np.abs(z[te & r0]))[0, 1]), 3)}, "mean_z": {"pre-test": round(float(z[~te & r0].mean()), 3), "test": round(float(z[te & r0].mean()), 3)}}}
for lab, left, wsig in VAR:
    OUT["expression_on_its_own_rows_pct"][lab] = {}
    for per, m in (("pre-test rows", ~te), ("test rows", te)): v, par = expr(m, m, left, wsig); OUT["expression_on_its_own_rows_pct"][lab][per] = {**v, "gap_points": {str(a): round(v[str(a)] - OUT["gain_of_the_multipliers_pct"][per][str(a)], 2) for a in AL}, "tails": par}
    v, _ = expr(~te, te, left, wsig); OUT["expression_fitted_on_pre_test_rows_as_a_forecast_for_the_test_rows_pct"][lab] = {**v, "gap_points": {str(a): round(v[str(a)] - OUT["gain_of_the_multipliers_pct"]["test rows"][str(a)], 2) for a in AL}}
    print("%-32s own rows, pre-test: %5.2f (gap %+.2f)  test: %5.2f (gap %+.2f) | forecast of test from pre-test: %5.2f (gap %+.2f) | 1%%: gaps %+.2f %+.2f, forecast gap %+.2f" % (lab, OUT["expression_on_its_own_rows_pct"][lab]["pre-test rows"]["0.025"], OUT["expression_on_its_own_rows_pct"][lab]["pre-test rows"]["gap_points"]["0.025"], OUT["expression_on_its_own_rows_pct"][lab]["test rows"]["0.025"], OUT["expression_on_its_own_rows_pct"][lab]["test rows"]["gap_points"]["0.025"],
          v["0.025"], OUT["expression_fitted_on_pre_test_rows_as_a_forecast_for_the_test_rows_pct"][lab]["gap_points"]["0.025"], OUT["expression_on_its_own_rows_pct"][lab]["pre-test rows"]["gap_points"]["0.01"], OUT["expression_on_its_own_rows_pct"][lab]["test rows"]["gap_points"]["0.01"], OUT["expression_fitted_on_pre_test_rows_as_a_forecast_for_the_test_rows_pct"][lab]["gap_points"]["0.01"]))
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); print(json.dumps({k: OUT[k] for k in ("gain_of_the_multipliers_pct", "reaction_day_gap_in_mean_log_abs_z_by_year", "reaction_days")}))
