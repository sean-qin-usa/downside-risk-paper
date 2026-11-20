# r87_last_percent_2.py -- THE FLEXIBLE MODELS' LAST 1%, TWO OTHER WAYS.
#
# r84 put the margin on the level of volatility on the CRSP panel (93% of it) and found that shrinking sigma toward the
# name's OWN usual level does nothing. r86 priced today's volatility against the name's usual level at zero. What has
# not been tried is the ABSOLUTE level: each name's GARCH-t is fitted alone, so a calm name and a volatile name are
# never compared, while the pooled trees and the network see every name on one scale. Two ways to look:
#   (1) Read it off. Ordinary days are sorted into deciles of the absolute volatility level (logsig, cut points from
#       pre-test rows); one multiplier per decile is chosen on pre-test rows; its gain is measured on test rows. Then a
#       one-parameter version, "panel shrinkage": sigma is multiplied by (sigma / panel level)^delta, the panel level
#       being the pre-test mean of logsig over all names, and the same with the name's usual level in place of today's.
#   (2) Distil the flexible model. On test ordinary days, the log ratio of a flexible model's 2.5% VaR to the masked
#       calendar GARCH-t's is regressed on six state variables using the first half of the test dates; the fitted
#       linear correction is applied to the GARCH-t on the second half.
# Base model throughout: the masked calendar GARCH-t on ordinary days.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r82-r86.
#  N1. CRSP panel: the decile multipliers fall from above 1.03 in the calmest decile to below 0.97 in the most volatile,
#      and they gain at least 0.5% at 2.5%. US large caps: every multiplier is between 0.97 and 1.03, gain below 0.2%.
#  N2. Panel shrinkage: delta between -0.15 and -0.02 on the CRSP panel, gain 0.4% to 1.0% (DM above 2); on US large
#      caps delta within 0.03 of zero and a gain below 0.2%.
#  N3. On the CRSP panel the version that uses only the name's usual level gets at least 40% of that gain.
#  N4. Distillation (CRSP): the first-half regression has an R-squared of at least 0.3, and the linear correction
#      recovers at least half of the flexible model's second-half margin over the masked calendar GARCH-t.
#
# FOLLOW-UP, WRITTEN AFTER N1-N4 WERE RUN AND BEFORE THIS WAS (2026-11-20). Panel shrinkage gained 0.73% on the CRSP
# panel. It is now added to the two-fix GARCH-t of r82 (delta chosen again on pre-test rows) and compared with the best
# flexible model with the calendar, on ordinary days.
#  N5. On the CRSP panel the two-fix GARCH-t with panel shrinkage closes at least 70% of the ordinary-day gap between
#      the masked calendar GARCH-t and the best flexible model, and is within 0.5% of that model.
#
# SECOND FOLLOW-UP, WRITTEN AFTER N5 AND THE DISTILLATION WERE RUN AND BEFORE THIS WAS (2026-11-20). The linear correction
# fitted to the flexible models' own forecasts recovered 84% to 93% of their margin, with a coefficient of about -0.35
# on the firm part of the shock share and -0.07 to -0.19 on today's level. The same six-variable correction is now
# fitted with no flexible model at all: sigma is multiplied by exp(x'beta), beta chosen on pre-test rows by 2.5% pinball
# loss (Powell search from zero).
#  N6. On the CRSP panel this stand-alone correction closes at least 60% of the ordinary-day gap between the masked
#      calendar GARCH-t and the best flexible model at 2.5%; its coefficient on the firm part of the shock share is
#      between -0.6 and -0.1 and its coefficient on today's level is negative. On US large caps it closes at least 30%.
# Usage: python tools/r87_last_percent_2.py canon | us_largecap
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r87"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0) & (A.r0 != 1) & (A.r1 != 1) & np.isfinite(A.Dm)].reset_index(drop=True); A["z"] = (A.y - A.mmu) / A.msig
A["v"] = A.groupby("name")["z"].transform(lambda s: s.rolling(21, min_periods=8).std().shift(1)); A["lsbar"] = A["name"].map(A[~A.test].groupby("name")["logsig"].mean()); A = A[np.isfinite(A.v) & (A.U | ~A.test)].reset_index(drop=True)
pre = ~A.test.values; te = A.test.values; dates = pd.DatetimeIndex(A.date.values); y, mu, sg, nu, ls, lb = A.y.values, A.mmu.values, A.msig.values, A.mnu.values, A.logsig.values, A.lsbar.values; Lbar = float(ls[pre].mean()); b25 = tq(0.025, nu)
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
def score(mult, mask=te): return {str(a): {"gain_pct": round(100 * float((pin(y, mu + sg * tq(a, nu), a) - pin(y, mu + sg * mult * tq(a, nu), a))[mask].sum() / pin(y, mu + sg * tq(a, nu), a)[mask].sum()), 2), "DM": dm(pin(y, mu + sg * tq(a, nu), a) - pin(y, mu + sg * mult * tq(a, nu), a), mask)} for a in AL}
OUT = {"universe": WHAT, "rows_pre_test": int(pre.sum()), "rows_test": int(te.sum()), "predictions_written_before_run": "see the script header (N1-N4)", "panel_level_of_log_sigma": round(Lbar, 3)}
# (1) deciles of the absolute level, and of the name's usual level
CG = np.round(np.arange(0.6, 1.6001, 0.01), 2)
for lab, x in (("volatility level today", ls), ("the name's usual volatility level", lb)):
    b = np.searchsorted(np.quantile(x[pre], np.arange(1, 10) / 10), x); c = [float(CG[int(np.argmin([pin(y[pre & (b == k)], mu[pre & (b == k)] + cc * sg[pre & (b == k)] * b25[pre & (b == k)], 0.025).sum() for cc in CG]))]) for k in range(10)]
    OUT["deciles of " + lab] = {"multipliers_calmest_to_most_volatile": c, "median_sigma_by_decile": [round(float(np.exp(np.median(x[pre & (b == k)]))), 2) for k in range(10)], **score(np.array(c)[b])}
DG = np.round(np.arange(-0.5, 0.3001, 0.01), 2)
for lab, x in (("panel shrinkage, today's level", ls), ("panel shrinkage, the name's usual level", lb)):
    d = float(DG[int(np.argmin([pin(y[pre], mu[pre] + sg[pre] * np.exp(dd * (x[pre] - Lbar)) * b25[pre], 0.025).sum() for dd in DG]))]); OUT[lab] = {"delta": d, **score(np.exp(d * (x - Lbar)))}
# (2) distillation of the flexible models on the test rows
T = te.copy(); cut = np.sort(dates[T].unique())[len(dates[T].unique()) // 2]; h1 = T & (dates < cut); h2 = T & (dates >= cut); X = np.column_stack([np.ones(len(A)), ls - Lbar, lb - Lbar, np.log(np.maximum(A.v.values, 0.05)), np.clip(A.Dm.values, -0.5, 0.98) * (1 - A.wm.values), np.clip(A.Dm.values, -0.5, 0.98) * A.wm.values, A.ds.values / 63.0])
XN = ["constant", "level today less panel level", "usual level less panel level", "log recent residual scale", "shock share, firm part", "shock share, market part", "days since the last reaction day / 63"]; qb = mu + sg * b25; OUT["distillation"] = {"second_half_from": str(pd.Timestamp(cut).date())}
for nm in ("neural network, calendar inputs", "trees on robust scale, calendar inputs"):
    qf = A["%s__0.025" % nm].values; ok = np.isfinite(qf) & (qf < 0) & (qb < 0); t_ = np.where(ok, np.log(np.where(ok, qf / qb, 1.0)), np.nan); f1 = h1 & ok; f2 = h2 & ok; beta, *_ = np.linalg.lstsq(X[f1], t_[f1], rcond=None); r2 = 1 - np.sum((t_[f1] - X[f1] @ beta) ** 2) / np.sum((t_[f1] - t_[f1].mean()) ** 2)
    Lb = pin(y, qb, 0.025); Ld = pin(y, qb * np.exp(X @ beta), 0.025); Lf = pin(y, np.where(ok, qf, qb), 0.025); g = lambda L_: 100 * float((Lb - L_)[f2].sum() / Lb[f2].sum())
    OUT["distillation"][nm] = {"coefficients": {k: round(float(v), 3) for k, v in zip(XN, beta)}, "r2_first_half": round(float(r2), 3), "second_half_gain_of_flexible_pct": round(g(Lf), 2), "second_half_gain_of_linear_correction_pct": round(g(Ld), 2), "DM_linear_correction": dm(Lb - Ld, f2), "share_recovered": round(g(Ld) / g(Lf), 2) if abs(g(Lf)) > 1e-9 else None}
R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json"))); lf, lm = (R82["real_share_pre_test"]["masked filter"][k] for k in ("lambda_f", "lambda_m"))
fx = np.sqrt(np.maximum(1 - (1 - (lf + (lm - lf) * A.wm.values)) * np.clip(A.Dm.values, -0.5, 0.98), 0.02)); d2 = float(DG[int(np.argmin([pin(y[pre], mu[pre] + sg[pre] * fx[pre] * np.exp(dd * (ls[pre] - Lbar)) * b25[pre], 0.025).sum() for dd in DG]))]); OUT["two-fix with panel shrinkage"] = {"delta": d2}
for a in AL:
    fl = {nm: A["%s__%s" % (nm, a)].values for nm in ("neural network, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs")}; ok = te & np.all([np.isfinite(v) for v in fl.values()], axis=0); Lb = pin(y, mu + sg * tq(a, nu), a); best = min(fl, key=lambda k: pin(y, fl[k], a)[ok].sum()); Lbest = pin(y, fl[best], a)
    g = lambda L_: 100 * float((Lb - L_)[ok].sum() / Lb[ok].sum()); L2 = pin(y, mu + sg * fx * tq(a, nu), a); L3 = pin(y, mu + sg * fx * np.exp(d2 * (ls - Lbar)) * tq(a, nu), a); L1 = pin(y, mu + sg * np.exp(OUT["panel shrinkage, today's level"]["delta"] * (ls - Lbar)) * tq(a, nu), a)
    OUT["two-fix with panel shrinkage"][str(a)] = {"best_flexible": best, "gain_over_masked_pct": {"best flexible": round(g(Lbest), 2), "two-fix": round(g(L2), 2), "panel shrinkage alone": round(g(L1), 2), "two-fix with panel shrinkage": round(g(L3), 2)}, "share_of_gap_closed": round(g(L3) / g(Lbest), 2),
                                                   "against_best_flexible": {"edge_pct": round(100 * float((Lbest - L3)[ok].sum() / Lbest[ok].sum()), 2), "DM": dm(Lbest - L3, ok)}}
from scipy import optimize
rng = np.random.default_rng(0); ip = np.where(pre)[0]; ip = rng.choice(ip, min(len(ip), 400000), replace=False); Xp, yp, mp, sp_, bp = X[ip], y[ip], mu[ip], sg[ip], b25[ip]
res = optimize.minimize(lambda b_: float(pin(yp, mp + sp_ * np.exp(Xp @ b_) * bp, 0.025).mean()), np.zeros(X.shape[1]), method="Powell", options={"xtol": 1e-3, "ftol": 1e-7, "maxiter": 40}); bs = res.x; OUT["stand-alone log-linear correction"] = {"coefficients": {k: round(float(v), 3) for k, v in zip(XN, bs)}}
for a in AL:
    fl = {nm: A["%s__%s" % (nm, a)].values for nm in ("neural network, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs")}; ok = te & np.all([np.isfinite(v) for v in fl.values()], axis=0); Lb = pin(y, mu + sg * tq(a, nu), a); best = min(fl, key=lambda k: pin(y, fl[k], a)[ok].sum()); Lbest = pin(y, fl[best], a)
    L4 = pin(y, mu + sg * np.exp(X @ bs) * tq(a, nu), a); g = lambda L_: 100 * float((Lb - L_)[ok].sum() / Lb[ok].sum())
    OUT["stand-alone log-linear correction"][str(a)] = {"gain_over_masked_pct": round(g(L4), 2), "DM": dm(Lb - L4, ok), "best_flexible_gain_pct": round(g(Lbest), 2), "share_of_gap_closed": round(g(L4) / g(Lbest), 2), "against_best_flexible": {"edge_pct": round(100 * float((Lbest - L4)[ok].sum() / Lbest[ok].sum()), 2), "DM": dm(Lbest - L4, ok)},
                                                        "breach_pct": round(100 * float((y <= mu + sg * np.exp(X @ bs) * tq(a, nu))[ok].mean()), 2)}
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
for k, v in OUT.items():
    if isinstance(v, dict): print(k, json.dumps(v)[:620])
