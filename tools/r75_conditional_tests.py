# r75_conditional_tests.py -- FORMAL CONDITIONAL TESTS, with the shock share D of tools/r71_shock_response.py as the
# conditioning variable. D is known the evening before: the share of tomorrow's GARCH variance that is there only
# because the largest shock of the last 63 days was larger than an ordinary day.
#
#  A  Conditional coverage (a dynamic-quantile test in the sense of Engle and Manganelli 2004, with D as the instrument):
#     for each model and level, regress (hit - level) on a constant and D over all test rows; standard errors clustered
#     by date. A correct conditional quantile has both coefficients zero. Reported: the slope, its t-statistic, and the
#     Wald p-value of the joint null.
#  B  Conditional predictive ability (Giacomini and White 2006, with D as the conditioning variable): regress the
#     per-row loss difference, GARCH-t minus the model, on a constant and D; and on a constant, D and the dummy for the
#     top decile of the kurtosis score, to see which of the two carries the result.
#  C  The usual per-name backtests, as the literature reports them: share of names on which Kupiec's unconditional
#     coverage test, Christoffersen's conditional coverage test and the dynamic-quantile test (constant, four lagged
#     hits, the VaR) reject at 5%, per model and level.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: blocks A-F of r71 on the canonical panel and
# us_largecap (GARCH-t's breach rate falls with D there).
#  T1. Stock panels: GARCH-t's hit rate falls with D (slope t below -5 at 2.5%); the absolute slope of Beta-t-EGARCH
#      and of the pooled trees is less than half of GARCH-t's.
#  T2. Stock panels: the loss difference of the pooled trees, SAV-CAViaR, the neural network and Beta-t-EGARCH against
#      GARCH-t rises with D (t above 3 at 2.5%). With D and the kurtosis-decile dummy together, D keeps t above 3 for
#      the pooled trees and the dummy's t falls below 2.
#  T3. Equity ETFs and other ETFs: GARCH-t's slope on D is not significantly negative (t above -2) at 2.5%.
#  T4. Per-name backtests do not separate the models: on the stock panels the share of names rejecting Kupiec at 2.5%
#      differs by less than 10 percentage points between GARCH-t and Beta-t-EGARCH. (The conditional test in A does.)
# Usage:  python tools/r75_conditional_tests.py canon <bench_all_bteg_rows.npz> [r70_preds.npz]   |   python tools/r75_conditional_tests.py <panel>
#         (needs data/panels/<universe>_shock.npz, written by tools/r71_shock_response.py)
import os, sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r75"); os.makedirs(OUTD, exist_ok=True); AL = (0.01, 0.025)
if WHAT == "canon":
    D_ = np.load(sys.argv[2]); rk = D_["rk_mk"]; NAME = D_["permno"]; Q = {k[3:]: D_[k].astype(float) for k in D_.files if k.startswith("q__")}
    if len(sys.argv) > 3:
        N = np.load(sys.argv[3]); kk = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); Q["neural"] = N["q"][kk.loc[list(zip(D_["permno"], D_["date"]))].values].astype(float)
else:
    D_ = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); rk = D_["rk"]; NAME = D_["name"]; Q = {k[3:]: D_[k].astype(float) for k in D_.files if k.startswith("q__")}
S = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_shock.npz")); DD = S["D"].astype(float); Y = D_["y"].astype(float); dates = pd.to_datetime(D_["date"]); TAUS = [round(float(t), 4) for t in D_["taus"]]; J = {a: TAUS.index(a) for a in AL}
assert len(DD) == len(Y); VAR = {a: {m: Q[m][:, J[a]] for m in Q} for a in AL}
for a in AL: VAR[a]["cf0"] = S["cf0__%s" % a].astype(float); VAR[a]["cf_pre"] = S["cf_pre__%s" % a].astype(float)
MS = [m for m in ("garch_t", "cf0", "cf_pre", "gjr_skewt", "param_bteg", "sav_caviar", "neural", "body", "engine", "body_bteg", "engine_bteg", "evt_pool", "fhs", "fhs_pool") if m in VAR[0.025]]
pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q)); dcode = pd.factorize(dates)[0]; G = dcode.max() + 1; TOPK = (rk == 10).astype(float)
def cluster_ols(y, X):
    """OLS with covariance clustered by date."""
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ y); u = y - X @ b; M = np.zeros((G, X.shape[1]))
    for j in range(X.shape[1]): M[:, j] = np.bincount(dcode, weights=X[:, j] * u, minlength=G)
    V = np.linalg.solve(XtX, (M.T @ M) * (G / (G - 1))) @ np.linalg.inv(XtX); return b, V
ONE = np.ones(len(Y)); XD = np.column_stack([ONE, DD]); XDK = np.column_stack([ONE, DD, TOPK])
OUT = {"universe": WHAT, "note": "D = shock share of GARCH variance (tools/r71_shock_response.py). A: (hit - level) on [1, D]; B: loss(GARCH-t) - loss(model) on [1, D] and on [1, D, top kurtosis decile]; date-clustered. C: per-name tests, share rejecting at 5%.",
       "predictions_written_before_run": "see the script header (T1-T4)", "rows": int(len(Y)), "dates": int(G), "mean_D": round(float(DD.mean()), 4), "A_conditional_coverage": {}, "B_conditional_predictive_ability": {}, "C_per_name_backtests": {}}
for a in AL:
    A_ = {}; B_ = {}
    for m in MS:
        h = (Y <= VAR[a][m]).astype(float) - a; b, V = cluster_ols(h, XD); w = float(b @ np.linalg.solve(V, b))
        A_[m] = {"breach_pct": round(100 * float(h.mean() + a), 3), "intercept": round(float(b[0]), 5), "slope_on_D": round(float(b[1]), 5), "t_slope": round(float(b[1] / math.sqrt(V[1, 1])), 2), "wald_p": float("%.3g" % stats.chi2.sf(w, 2)),
                 "implied_breach_pct_at_D_0": round(100 * float(a + b[0]), 2), "implied_breach_pct_at_D_0.5": round(100 * float(a + b[0] + 0.5 * b[1]), 2)}
        if m == "garch_t": continue
        d = pin(Y, VAR[a]["garch_t"], a) - pin(Y, VAR[a][m], a); b, V = cluster_ols(d, XD); b2, V2 = cluster_ols(d, XDK); base = float(pin(Y, VAR[a]["garch_t"], a).mean())
        B_[m] = {"mean_gain_pct_of_garch_t_loss": round(100 * float(d.mean()) / base, 3), "on_D": {"intercept_pct": round(100 * float(b[0]) / base, 3), "t_intercept": round(float(b[0] / math.sqrt(V[0, 0])), 2), "slope_pct": round(100 * float(b[1]) / base, 2), "t_slope": round(float(b[1] / math.sqrt(V[1, 1])), 2)},
                 "on_D_and_kurtosis_decile": {"t_D": round(float(b2[1] / math.sqrt(V2[1, 1])), 2), "t_kurtosis_dummy": round(float(b2[2] / math.sqrt(V2[2, 2])), 2), "slope_D_pct": round(100 * float(b2[1]) / base, 2), "coef_dummy_pct": round(100 * float(b2[2]) / base, 2)}}
    OUT["A_conditional_coverage"][str(a)] = A_; OUT["B_conditional_predictive_ability"][str(a)] = B_
lg("conditional tests done")
# ---------------------------------------------------------------- C. per-name backtests
def kupiec(h, a):
    n, x = len(h), int(h.sum()); p = x / n
    if x == 0: lr = -2 * n * math.log(1 - a)
    else: lr = -2 * ((n - x) * math.log(1 - a) + x * math.log(a) - (n - x) * math.log(1 - p) - x * math.log(p))
    return lr
def christ(h, a):
    h = h.astype(int); n00 = int(((h[:-1] == 0) & (h[1:] == 0)).sum()); n01 = int(((h[:-1] == 0) & (h[1:] == 1)).sum()); n10 = int(((h[:-1] == 1) & (h[1:] == 0)).sum()); n11 = int(((h[:-1] == 1) & (h[1:] == 1)).sum())
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1); ll = lambda k, n_, q: 0.0 if k == 0 and q == 0 else (k * math.log(max(q, 1e-12)) + (n_ - k) * math.log(max(1 - q, 1e-12)))
    ind = -2 * (ll(n01 + n11, n00 + n01 + n10 + n11, p) - ll(n01, n00 + n01, p01) - ll(n11, n10 + n11, p11)); return kupiec(h, a) + max(ind, 0.0)
def dq(h, v, a, L=4):
    y = h[L:] - a; X = np.column_stack([np.ones(len(y))] + [h[L - k:-k] - a for k in range(1, L + 1)] + [v[L:]])
    try: b = np.linalg.lstsq(X, y, rcond=None)[0]; return float(b @ (X.T @ X) @ b / (a * (1 - a))), X.shape[1]
    except Exception: return float("nan"), X.shape[1]
order = np.lexsort((dates.values, NAME)); nm = NAME[order]; cuts = np.r_[0, np.where(nm[1:] != nm[:-1])[0] + 1, len(nm)]
for a in AL:
    C_ = {}
    for m in MS:
        hk = hc = hd = tot = 0
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            ix = order[i0:i1]
            if len(ix) < 250: continue
            h = (Y[ix] <= VAR[a][m][ix]).astype(float); tot += 1; hk += kupiec(h, a) > stats.chi2.ppf(0.95, 1); hc += christ(h, a) > stats.chi2.ppf(0.95, 2); s_, k_ = dq(h, VAR[a][m][ix], a); hd += (s_ > stats.chi2.ppf(0.95, k_)) if np.isfinite(s_) else 0
        C_[m] = {"names": tot, "kupiec_reject_share": round(hk / tot, 3), "christoffersen_reject_share": round(hc / tot, 3), "dq_reject_share": round(hd / tot, 3)}
    OUT["C_per_name_backtests"][str(a)] = C_
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
print("\n######## %s: %d rows, mean D %.3f" % (WHAT, len(Y), DD.mean()))
for a in ("0.025", "0.01"):
    print("--- level %s. model | breach | slope on D (t) | implied breach at D=0, D=0.5 | Wald p || gain over GARCH-t: mean, slope on D (t), with dummy: t_D, t_dummy || per-name reject shares Kupiec/Christoffersen/DQ" % a)
    for m in MS:
        x = OUT["A_conditional_coverage"][a][m]; b = OUT["B_conditional_predictive_ability"][a].get(m); c = OUT["C_per_name_backtests"][a][m]
        print("  %-12s %.2f%% | %+.4f (%6.2f) | %.2f%% %.2f%% | %-8.2g || %s || %.2f %.2f %.2f" % (m, x["breach_pct"], x["slope_on_D"], x["t_slope"], x["implied_breach_pct_at_D_0"], x["implied_breach_pct_at_D_0.5"], x["wald_p"],
              "reference" if b is None else "%+.2f%%, %+.1f%% (%5.2f), %5.2f %5.2f" % (b["mean_gain_pct_of_garch_t_loss"], b["on_D"]["slope_pct"], b["on_D"]["t_slope"], b["on_D_and_kurtosis_decile"]["t_D"], b["on_D_and_kurtosis_decile"]["t_kurtosis_dummy"]), c["kupiec_reject_share"], c["christoffersen_reject_share"], c["dq_reject_share"]))
lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
