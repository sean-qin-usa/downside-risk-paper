# r90_ceiling.py -- A CEILING ON WHAT ANY MODEL CAN GAIN FROM A SET OF INPUTS, AND WHY IT IS LOW FOR PAST RETURNS.
#
# Write the standardized return of a benchmark model as z = s * eps: eps has the benchmark's shape (a unit-variance
# Student-t) and s is the ratio of the true scale to the benchmark's. A model that knew s would forecast s*q instead of
# q. Expanding the expected pinball loss of r83 to second order around s = 1,
#     loss(q) - loss(s q)  =  (1/2) q^2 f(q) (log s)^2 + ...,        f the density of eps,
# so the relative gain from knowing s is (1/2) kappa E[(log s)^2] with  kappa(a, nu) = q^2 f(q) / EL(q).
# Inputs I reveal only E[log s | I]; projecting, the most ANY model can gain from I, however flexible, is
#     ceiling(I)  =  (1/2) kappa(a, nu) Var( E[log s | I] ).
# The right side needs no quantile model: E[log s | I] is the regression of log|z| on the inputs (up to a constant), a
# mean regression. Three checks.
#   (1) kappa itself, and what it implies: how large must the predictable scale error be for a 1% gain.
#   (2) Finite partitions, where the exact value V is known (r83, r86): the calendar, shock share and origin.
#   (3) Past-return inputs. Ordinary days, base model the masked calendar GARCH-t. log|z| (floored at 0.02) is regressed
#       on the six inputs of the flexible models (boosted trees, squared loss, training rows of r79/r81); on test rows
#       the explained variance gives the ceiling; it is compared with what the fitted scale delivers as a multiplier,
#       and with the margin the flexible QUANTILE models actually made on the same days (r84, r87).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r81-r89.
#  U1. kappa at 2.5% is between 3 and 8 for the panels' tail indices; a 1% gain needs a predictable log-scale error
#      with a standard deviation between 4% and 9%.
#  U2. For the small partitions (shock share and origin) the second-order ceiling is within 30% of the exact V; for
#      the calendar, where the scale changes by a factor of three, it is not.
#  U3. Past-return inputs explain between 0.3% and 3% of the variance of log|z| out of sample on both panels, and the
#      ceiling they imply is between 0.5% and 3% at 2.5%.
#  U4. The fitted scale used as a multiplier gains within 40% of what the formula predicts for it, and the flexible
#      quantile models' ordinary-day margin over the base model is between half and one and a half times the ceiling.
# Usage: python tools/r90_ceiling.py canon | us_largecap
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r90"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def kappa(a, nu): k = np.sqrt(nu / (nu - 2)); x = stats.t.ppf(a, nu); q = x / k; f = k * stats.t.pdf(x, nu); EL = (1 / k) * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1); return q * q * f / EL
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0) & (A.r0 != 1) & (A.r1 != 1)].reset_index(drop=True); X = ["logsig", "zl1", "absz5", "zstd21", "fracdn5", "ds"]; A = A[np.isfinite(A[X]).all(1) & (A.U | ~A.test)].reset_index(drop=True)
A["lz"] = np.log(np.maximum(np.abs((A.y - A.mmu) / A.msig), 0.02)); tr = A.train.values; te = A.test.values; NU = A.groupby("name")["mnu"].first().values; dates = pd.DatetimeIndex(A.date.values)
OUT = {"universe": WHAT, "predictions_written_before_run": "see the script header (U1-U4)", "kappa": {str(a): {"mean_over_names": round(float(kappa(a, NU).mean()), 2), "at_nu_4_6_8": [round(float(kappa(a, v)), 2) for v in (4.0, 6.0, 8.0)]} for a in AL}}
kp = {a: float(kappa(a, NU).mean()) for a in AL}; OUT["sd_of_predictable_log_scale_error_for_a_1pct_gain"] = {str(a): round(math.sqrt(2 * 0.01 / kp[a]), 3) for a in AL}
# (2) finite partitions: Var(E[log s | I]) from the implied multipliers of r83 and r86
R83 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r83", WHAT + ".json"))); R86 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r86", WHAT + ".json"))); OUT["partitions"] = {}
dt = R83["day_types_pre_test"]; c = R83["A"]["implied_multipliers"]; p = np.array([dt["reaction day"]["p"], dt["day after"]["p"], dt["other days"]["p"]]); l = np.log(np.array([c["reaction day"], c["day after"], 1.0])); v = float((p * l * l).sum() - (p * l).sum() ** 2)
OUT["partitions"]["earnings calendar"] = {"var_log_scale": round(v, 5), **{str(a): {"ceiling_pct": round(50 * kp[a] * v, 2), "exact_V_pct": R83["A"][str(a)]["V_pct"]} for a in AL}}
lab = np.where(np.digitize(np.clip(A.Dm.values, -0.5, 0.98), [0.02, 0.05, 0.10, 0.20, 0.35]) >= 2, np.char.add(np.digitize(np.clip(A.Dm.values, -0.5, 0.98), [0.02, 0.05, 0.10, 0.20, 0.35]).astype(str), np.where(A.wm.values < 0.1, "f", np.where(A.wm.values < 0.5, "x", "m"))), np.digitize(np.clip(A.Dm.values, -0.5, 0.98), [0.02, 0.05, 0.10, 0.20, 0.35]).astype(str))
r = R86["partitions"]["shock share and origin"]; cm = r["implied_multipliers"]; ok = te & np.isin(lab, list(cm)); l = np.log(np.array([cm[t] for t in lab[ok]])); v = float(l.var())
OUT["partitions"]["shock share and origin"] = {"var_log_scale": round(v, 5), **{str(a): {"ceiling_pct": round(50 * kp[a] * v, 2), "exact_V_pct": r[str(a)]["V_pct"], "realized_pct": r[str(a)]["realized_pct"]} for a in AL}}
# (3) past-return inputs: regression of log|z| on the six inputs
m = HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.06, random_state=0).fit(A.loc[tr, X].values, A.lz.values[tr]); mh = m.predict(A[X].values); yt = A.lz.values[te]; mt = mh[te] - mh[te].mean(); cov = float(np.mean(mt * (yt - yt.mean()))); vm = float(mt.var()); ev = 2 * cov - vm
OUT["past_return_inputs"] = {"rows_train": int(tr.sum()), "rows_test": int(te.sum()), "var_log_abs_z_test": round(float(yt.var()), 4), "explained_share_out_of_sample": round(ev / float(yt.var()), 4), "var_of_conditional_mean_log_scale": {"covariance_estimate": round(cov, 5), "explained_variance (lower)": round(ev, 5), "variance_of_fitted": round(vm, 5)}}
y, mu, sg, nu = A.y.values, A.mmu.values, A.msig.values, A.mnu.values
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v_ = np.mean(dd * dd)
    for k in range(1, 11): v_ += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v_ / len(s), 1e-18))), 2)
for a in AL:
    Lb = pin(y, mu + sg * tq(a, nu), a); Lm = pin(y, mu + sg * np.exp(mh - mh[tr].mean()) * tq(a, nu), a); g = lambda L_, f=te: 100 * float((Lb - L_)[f].sum() / Lb[f].sum()); fl = {}
    for nm in ("neural network, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs"):
        q = A["%s__%s" % (nm, a)].values; f = te & np.isfinite(q); fl[nm] = round(g(pin(y, np.where(np.isfinite(q), q, 0.0), a), f), 2)
    OUT["past_return_inputs"][str(a)] = {"ceiling_pct": {"from the covariance": round(50 * kp[a] * cov, 2), "from the explained variance (lower)": round(50 * kp[a] * ev, 2)}, "fitted_scale_as_multiplier": {"predicted_pct": round(50 * kp[a] * ev, 2), "realized_pct": round(g(Lm), 2), "DM": dm(Lb - Lm, te)}, "flexible_quantile_models_realized_margin_pct": fl}
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); print(json.dumps(OUT)[:1900])
