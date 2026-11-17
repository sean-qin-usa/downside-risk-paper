# nvda_gjr_case.py -- CASE STUDY (not in the paper): is GJR-GARCH-skew-t a significant loss on NVIDIA?
# Data: NVDA daily adjusted closes from yfinance (free), 2005-01-03 .. 2026-11-13 (the partial 2026-11-16 row dropped).
# Returns in percent (simple). Walk-forward: every model refit each January 1 from 2020 to 2026 on all prior data
# (expanding window from 2005), parameters frozen within the year, filters updated daily. Test 2020-01-02..2026-11-13.
#
# Models (one-day-ahead VaR and ES at 1% and 2.5%):
#   garch_t     GARCH(1,1)-t                        the paper's benchmark
#   gjr_skewt   GJR-GARCH(1,1) with Hansen skewed-t  the benchmark in question
#   gjr_t       GJR-GARCH(1,1)-t                     leverage without skew   (to split GJR-skew-t's two additions)
#   garch_skewt GARCH(1,1) with skewed-t             skew without leverage
#   bteg        Beta-t-EGARCH, the paper's code (job_robust_engine.py), robust to a single shock by construction
#   fhs         GARCH-t filter x empirical training residual quantiles (filtered historical simulation)
#   hs500       historical simulation, trailing 500 days
# Scores: FZ0 joint (VaR, ES) loss at each level; pinball at each level; breach rate, Kupiec and Christoffersen tests.
# Inference: DM with Newey-West(10) variance on the daily loss differential (one name, so per-date = per-day), two-sided p.
# High-score days: the paper's mk63 (63-day bias-corrected excess kurtosis of GARCH-t residuals, lagged one day) above
# its 90th percentile in the data before each refit date (no look-ahead).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-16):
#  P1. On FZ0, GJR-skew-t and GARCH-t are within noise of each other at both levels (|DM| < 1.96): on the paper's
#      panel GJR-skew-t ties the accuracy layer and its pinball edge over GARCH-t is +0.01%; one name has less power.
#  P2. Beta-t-EGARCH has a lower mean FZ0 than GJR-skew-t at both levels, not significantly (DM < 1.96).
#  P3. On high-score days (after a large shock) GJR-skew-t's FZ0 is worse than Beta-t-EGARCH's (point estimate).
#  P4. Every parametric model's 2.5% breach rate passes Kupiec at 5%; at 1% the breach rates lie in [0.5%, 2.0%].
#  P5. NVDA's fitted skew parameter is positive or near zero and its leverage term small, so GJR-skew-t's two additions
#      buy little on this name.
import os, sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
from arch import arch_model
from arch.univariate import SkewStudent
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1]
OUT = os.path.join(HERE, "..", "..", "results", "case_studies", "nvda_gjr_case_results.json")
A = [0.01, 0.025]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)

px = pd.read_csv(DATA, index_col=0, parse_dates=True)["Close"]
px = px[px.index <= "2026-11-13"]
r = (px.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
lg("NVDA %d returns %s..%s" % (n, dates[0].date(), dates[-1].date()))

# ---------------------------------------------------------------- helpers
def nw_t(x, l=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; m = len(x); d = x - x.mean(); v = np.mean(d * d)
    for k in range(1, l + 1): v += 2 * (1 - k / (l + 1)) * np.mean(d[k:] * d[:-k])
    return float(x.mean() / math.sqrt(max(v / m, 1e-16)))
def fz0(ret, v, e, a):
    v = np.minimum(v, -1e-8); e = np.minimum(e, v); hit = (ret <= v).astype(float)
    return -(1.0 / (a * e)) * hit * (v - ret) + v / e + np.log(-e) - 1.0
def pin(ret, q, a): d = ret - q; return np.where(d >= 0, a * d, (a - 1) * d)
def es_from_ppf(ppf, a, S=60.0, m=6000):
    """ES = (1/a) int_0^a Q(u) du with u = a e^{-s}: = int_0^inf Q(a e^{-s}) e^{-s} ds (trapezoid)."""
    s = np.linspace(0, S, m); u = a * np.exp(-s); q = ppf(u); f = q * np.exp(-s)
    return float(np.trapz(f, s))
def t_std_q_es(a, nu):
    sc = math.sqrt(nu / (nu - 2)); q = stats.t.ppf(a, nu)
    es = -stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * a)
    return q / sc, es / sc
# self-test of the quadrature against the closed form
_q, _e = t_std_q_es(0.01, 4.0); _e2 = es_from_ppf(lambda u: stats.t.ppf(u, 4.0) / math.sqrt(2.0), 0.01)
assert abs(_e - _e2) < 1e-3, (_e, _e2)
SK = SkewStudent()
def skewt_q_es(a, eta, lam):
    ppf = lambda u: SK.ppf(np.clip(u, 1e-300, 1.0), np.array([eta, lam]))
    return float(ppf(np.array([a]))[0]), es_from_ppf(ppf, a)
def kupiec(h, a):
    T = len(h); x = int(h.sum()); p = x / T
    if x == 0: lr = -2 * T * math.log(1 - a)
    else: lr = -2 * ((T - x) * math.log(1 - a) + x * math.log(a) - (T - x) * math.log(1 - p) - x * math.log(p))
    return float(1 - stats.chi2.cdf(lr, 1))
def christoffersen(h, a):
    h = h.astype(int); n00 = n01 = n10 = n11 = 0
    for i in range(1, len(h)):
        if h[i-1] == 0 and h[i] == 0: n00 += 1
        elif h[i-1] == 0: n01 += 1
        elif h[i] == 0: n10 += 1
        else: n11 += 1
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll = lambda k, pr: k * math.log(pr) if k > 0 and pr > 0 else 0.0
    lr_ind = -2 * (ll(n00 + n10, 1 - p) + ll(n01 + n11, p) - ll(n00, 1 - p01) - ll(n01, p01) - ll(n10, 1 - p11) - ll(n11, p11))
    lr_uc = -2 * (len(h) * 0 + 0)
    T = len(h); x = int(h.sum()); pp = x / T
    lr_uc = -2 * ((T - x) * math.log(1 - a) + x * math.log(a) - ((T - x) * math.log(1 - pp) + (x * math.log(pp) if x else 0)))
    return float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))

# Beta-t-EGARCH: verbatim from code/paper/job_robust_engine.py (filter and likelihood), warm-started yearly
def _bteg_filter(yy, mu, om, phi, kap, nu):
    m = len(yy); lam = np.empty(m); lam[0] = om
    for k in range(m - 1):
        e = (yy[k] - mu) * math.exp(-lam[k]); e2 = e * e; u = (nu + 1.0) * e2 / (nu + e2) - 1.0
        lam[k + 1] = min(max(om + phi * (lam[k] - om) + kap * u, -12.0), 12.0)
    return lam
def _bteg_nll(th, yy):
    mu, om = th[0], th[1]; phi = 1 / (1 + math.exp(-th[2])); kap = math.exp(th[3]); nu = 2.05 + math.exp(th[4])
    if kap > 2.0 or nu > 200.0: return 1e12
    lam = _bteg_filter(yy, mu, om, phi, kap, nu); e = (yy - mu) * np.exp(-lam)
    c = math.lgamma(0.5 * (nu + 1)) - math.lgamma(0.5 * nu) - 0.5 * math.log(math.pi * nu)
    s = float(np.sum(c - lam - 0.5 * (nu + 1) * np.log1p(e * e / nu))); return -s if np.isfinite(s) else 1e12
def fit_bteg(ytr, x0s):
    best = None
    for x0 in x0s:
        rr = optimize.minimize(_bteg_nll, x0, args=(ytr,), method="Nelder-Mead",
                               options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-5, "fatol": 1e-5})
        if np.isfinite(rr.fun) and (best is None or rr.fun < best.fun): best = rr
    return best.x

# ---------------------------------------------------------------- walk-forward
years = list(range(2020, 2027)); M = ["garch_t", "gjr_skewt", "gjr_t", "garch_skewt", "bteg", "fhs", "hs500"]
V = {m: {a: np.full(n, np.nan) for a in A} for m in M}; E = {m: {a: np.full(n, np.nan) for a in A} for m in M}
MK = np.full(n, np.nan); HI = np.zeros(n, bool); PARAMS = {}; bteg_x = None
SPEC = {"garch_t": dict(o=0, dist="t"), "gjr_skewt": dict(o=1, dist="skewt"), "gjr_t": dict(o=1, dist="t"),
        "garch_skewt": dict(o=0, dist="skewt")}
for yr in years:
    tr = dates < pd.Timestamp("%d-01-01" % yr); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1)))
    if te.sum() == 0: continue
    ntr = int(tr.sum()); last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; PARAMS[yr] = {}
    for m, sp in SPEC.items():
        res = arch_model(ytr, vol="GARCH", p=1, o=sp["o"], q=1, dist=sp["dist"], rescale=False).fit(disp="off", show_warning=False)
        pr = res.params; PARAMS[yr][m] = {k: round(float(v), 4) for k, v in pr.items()}
        sig = arch_model(y[:last], vol="GARCH", p=1, o=sp["o"], q=1, dist=sp["dist"], rescale=False).fix(pr.values).conditional_volatility
        mu = float(pr["mu"])
        for a in A:
            if sp["dist"] == "t": zq, ze = t_std_q_es(a, float(pr["nu"]))
            else: zq, ze = skewt_q_es(a, float(pr["eta"]), float(pr["lambda"]))
            V[m][a][te] = mu + sig[te[:last]] * zq; E[m][a][te] = mu + sig[te[:last]] * ze
        if m == "garch_t":
            z = (y[:last] - mu) / sig; ztr = z[:ntr]
            for a in A:   # FHS on the GARCH-t filter
                qz = float(np.quantile(ztr, a)); ez = float(ztr[ztr <= qz].mean())
                V["fhs"][a][te] = mu + sig[te[:last]] * qz; E["fhs"][a][te] = mu + sig[te[:last]] * ez
            mk = pd.Series(z).rolling(63, min_periods=30).kurt().shift(1).values
            thr = float(np.nanquantile(mk[:ntr], 0.9)); MK[te] = mk[te[:last]]; HI[te] = mk[te[:last]] > thr
            PARAMS[yr]["mk63_threshold"] = round(thr, 2)
    # Beta-t-EGARCH (first year: the paper's three starts; later years: warm start from the previous fit, plus one default)
    mu0 = float(np.mean(ytr)); sd0 = float(np.std(ytr))
    starts = [np.array([mu0, math.log(sd0) - 0.5 * math.log(nu0 / (nu0 - 2)), math.log(p0 / (1 - p0)), math.log(k0), math.log(nu0 - 2.05)])
              for nu0, k0, p0 in ((6.0, 0.08, 0.98), (4.0, 0.15, 0.95), (10.0, 0.04, 0.99))]
    bteg_x = fit_bteg(ytr, starts if bteg_x is None else [bteg_x, starts[0]])
    mu, om = bteg_x[0], bteg_x[1]; phi = 1 / (1 + math.exp(-bteg_x[2])); kap = math.exp(bteg_x[3]); nu = 2.05 + math.exp(bteg_x[4])
    lam = _bteg_filter(y[:last], mu, om, phi, kap, nu); sc = np.exp(lam)
    PARAMS[yr]["bteg"] = dict(mu=round(mu, 4), om=round(om, 4), phi=round(phi, 4), kap=round(kap, 4), nu=round(nu, 3))
    for a in A:
        q = stats.t.ppf(a, nu); es = -stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * a)   # unscaled t on exp(lam)
        V["bteg"][a][te] = mu + sc[te[:last]] * q; E["bteg"][a][te] = mu + sc[te[:last]] * es
    # HS500
    for i in np.where(te)[0]:
        w = y[max(0, i - 500):i]
        for a in A:
            qq = float(np.quantile(w, a)); V["hs500"][a][i] = qq; E["hs500"][a][i] = float(w[w <= qq].mean())
    lg("year %d: %d test days; GJR-skew-t eta=%.2f lambda=%.3f gamma=%.3f; bteg nu=%.2f kap=%.3f" % (
        yr, te.sum(), PARAMS[yr]["gjr_skewt"]["eta"], PARAMS[yr]["gjr_skewt"]["lambda"], PARAMS[yr]["gjr_skewt"]["gamma[1]"], nu, kap))

# ---------------------------------------------------------------- scoring
T = np.isfinite(V["garch_t"][0.01]); yt = y[T]; dt = dates[T]; hi = HI[T]; last3 = dt >= pd.Timestamp("2023-11-14")
out = {"data": "NVDA daily adjusted close, yfinance", "test": [str(dt[0].date()), str(dt[-1].date())], "n_test": int(T.sum()),
       "n_high_score": int(hi.sum()), "params": PARAMS, "by_level": {}}
for a in A:
    L = {m: fz0(yt, V[m][a][T], E[m][a][T], a) for m in M}; P = {m: pin(yt, V[m][a][T], a) for m in M}
    blk = {"mean_fz0": {m: round(float(L[m].mean()), 4) for m in M},
           "mean_pinball": {m: round(float(P[m].mean()), 5) for m in M},
           "breach_pct": {m: round(100 * float((yt <= V[m][a][T]).mean()), 2) for m in M},
           "kupiec_p": {m: round(kupiec((yt <= V[m][a][T]), a), 3) for m in M},
           "christoffersen_cc_p": {m: round(christoffersen((yt <= V[m][a][T]), a), 3) for m in M},
           "gjr_skewt_vs": {}}
    for m in M:   # DM > 0: GJR-skew-t has the HIGHER (worse) loss than m
        if m == "gjr_skewt": continue
        res = {}
        for lab, mask in (("all", np.ones(len(yt), bool)), ("last_3y", last3), ("high_score", hi), ("rest", ~hi)):
            d = L["gjr_skewt"][mask] - L[m][mask]; dp = P["gjr_skewt"][mask] - P[m][mask]
            tf = nw_t(d); tp = nw_t(dp)
            res[lab] = {"fz0_diff": round(float(d.mean()), 4), "fz0_DM": round(tf, 2), "fz0_p2": round(float(2 * (1 - stats.norm.cdf(abs(tf)))), 3),
                        "pin_rel_pct": round(100 * float(dp.mean() / P[m][mask].mean()), 2), "pin_DM": round(tp, 2), "n": int(mask.sum())}
        blk["gjr_skewt_vs"][m] = res
    out["by_level"][str(a)] = blk
json.dump(out, open(OUT, "w"), indent=1)
for a in A:
    b = out["by_level"][str(a)]
    print("\n=== alpha %.3f   (DM > 0: GJR-skew-t worse)" % a)
    print("%-12s %8s %9s %7s %6s %6s" % ("model", "FZ0", "pinball", "breach", "Kup p", "CC p"))
    for m in M: print("%-12s %8.4f %9.5f %6.2f%% %6.3f %6.3f" % (m, b["mean_fz0"][m], b["mean_pinball"][m], b["breach_pct"][m], b["kupiec_p"][m], b["christoffersen_cc_p"][m]))
    for m, res in b["gjr_skewt_vs"].items():
        print("  vs %-11s " % m + "  ".join("%s: dFZ0 %+.4f DM %+.2f" % (k, v["fz0_diff"], v["fz0_DM"]) for k, v in res.items()))
lg("wrote %s" % OUT)
