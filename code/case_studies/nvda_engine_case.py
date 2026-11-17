# nvda_engine_case.py -- CASE STUDY (not in the paper): the paper's estimator on NVIDIA, 2020-01-02 .. 2026-11-13.
# Our model is trained ONCE on the other 199 names of the paper's CRSP panel using only data before 2020-01-01 and then
# frozen (NVDA, permno 86580, is excluded from training, so NVDA is an out-of-sample name in an out-of-sample era).
# NVDA's own Stage-1 filter is refit each January 1 on all prior NVDA data (yfinance, from 2005), exactly as the
# benchmarks are. Construction follows code/paper/job_robust_engine.py: features ZX on the filter's residuals, pooled
# HistGBM quantile body (same hyperparameters, random_state=0), pooled GPD lower tail at p0 = 2.5%, minimum envelope
# and rearrangement, ES as the converged integral (code/paper/es_integral.py). No conformal shift (accuracy layer).
#
# Rows:  garch_t, gjr_skewt, bteg (Beta-t-EGARCH with its Student-t tail), fhs         benchmarks
#        body, engine            our model on the GARCH-t scale (engine = the accuracy layer)
#        body_bteg, engine_bteg  our model on the Beta-t-EGARCH scale
# Scores: FZ0 at 1% and 2.5%, pinball at both levels and over the eleven-level grid, breach rates, Kupiec.
# DM: Newey-West(10) on the daily loss differential, two-sided p. High-score days: NVDA's mk63 (GARCH-t residuals)
# above its 90th percentile in NVDA's data before each refit date.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-16):
#  P1. The engine has a lower mean FZ0 than GARCH-t at both levels; one name, so DM between 1 and 3.
#  P2. The engine and GJR-skew-t are within noise on FZ0 at both levels (|DM| < 1.96).
#  P3. The engine and Beta-t-EGARCH are within noise on FZ0 at both levels.
#  P4. On high-score days the engine beats GARCH-t on FZ0 (DM > 2) and does not beat Beta-t-EGARCH (DM < 1.96).
#  P5. engine_bteg has the lowest mean FZ0 of all rows at both levels, not significantly below Beta-t-EGARCH.
#  P6. The engine's breach rates pass Kupiec at 5% at both levels.
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from arch import arch_model
from arch.univariate import SkewStudent
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper"))
from es_integral import converged_es
HERE = os.path.dirname(os.path.abspath(__file__)); NVDA_CSV = sys.argv[1] if len(sys.argv) > 1 else None; PANEL_DIR = sys.argv[2] if len(sys.argv) > 2 else None
RES = os.path.join(HERE, "..", "..", "results", "case_studies"); OUT = os.path.join(RES, "nvda_engine_case_results.json")
A = [0.01, 0.025]; TAUS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
ZX = ['logsig', 'zl1', 'absz5', 'zstd21', 'fracdn5']; P0 = 0.025; KEXTRA = 40; SUBN = 20
HGB = dict(max_iter=250, max_depth=3, learning_rate=0.06); CUT = pd.Timestamp("2020-01-01"); NV = 86580
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)

# ---------------------------------------------------------------- helpers (as in the paper's jobs)
def nw_t(x, l=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; m = len(x); d = x - x.mean(); v = np.mean(d * d)
    for k in range(1, l + 1): v += 2 * (1 - k / (l + 1)) * np.mean(d[k:] * d[:-k])
    return float(x.mean() / math.sqrt(max(v / m, 1e-16)))
def fz0(ret, v, e, a):
    v = np.minimum(v, -1e-8); e = np.minimum(e, v); hit = (ret <= v).astype(float)
    return -(1.0 / (a * e)) * hit * (v - ret) + v / e + np.log(-e) - 1.0
def pin(ret, q, a): d = ret - q; return np.where(d >= 0, a * d, (a - 1) * d)
def kupiec(h, a):
    T = len(h); x = int(h.sum()); p = x / T
    lr = -2 * T * math.log(1 - a) if x == 0 else -2 * ((T - x) * math.log(1 - a) + x * math.log(a) - (T - x) * math.log(1 - p) - x * math.log(p))
    return float(1 - stats.chi2.cdf(lr, 1))
def t_std_q_es(a, nu):
    sc = math.sqrt(nu / (nu - 2)); q = stats.t.ppf(a, nu); es = -stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1) * a)
    return q / sc, es / sc
SK = SkewStudent()
def es_from_ppf(ppf, a, S=60.0, m=6000):
    s = np.linspace(0, S, m); return float(np.trapz(ppf(a * np.exp(-s)) * np.exp(-s), s))
def feats(y, sig, mu):
    z = (y - mu) / np.maximum(sig, 1e-6); df = pd.DataFrame({'y': y, 'sig': sig, 'z': z})
    df['logsig'] = np.log(np.maximum(df['sig'], 1e-6)); df['zl1'] = df['z'].shift(1)
    df['absz5'] = df['z'].abs().rolling(5, min_periods=3).mean().shift(1); df['zstd21'] = df['z'].rolling(21, min_periods=8).std().shift(1)
    df['fracdn5'] = (df['y'] < 0).rolling(5, min_periods=3).mean().shift(1); df['mk63'] = df['z'].rolling(63, min_periods=30).kurt().shift(1)
    return df
class GPDTail:
    def __init__(self, z, p0):
        zz = -np.asarray(z, float); zz = zz[np.isfinite(zz)]; self.u = float(np.quantile(zz, 1 - p0)); exc = zz[zz > self.u] - self.u
        self.p0 = p0; self.n_exc = len(exc); self.xi, _, self.beta = stats.genpareto.fit(exc, floc=0.0)
    def q(self, tau):
        xi, b, u, p0 = self.xi, self.beta, self.u, self.p0
        return -(u + (b / xi) * ((tau / p0) ** (-xi) - 1.0) if abs(xi) > 1e-6 else u + b * math.log(p0 / tau))
# Beta-t-EGARCH, verbatim filter and likelihood from job_robust_engine.py
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
def fit_bteg(ytr, extra=None):
    mu0 = float(np.mean(ytr)); sd0 = float(np.std(ytr)); best = None
    starts = [np.array([mu0, math.log(sd0) - 0.5 * math.log(nu0 / (nu0 - 2)), math.log(p0 / (1 - p0)), math.log(k0), math.log(nu0 - 2.05)])
              for nu0, k0, p0 in ((6.0, 0.08, 0.98), (4.0, 0.15, 0.95), (10.0, 0.04, 0.99))]
    for x0 in (starts if extra is None else [extra, starts[0]]):
        r = optimize.minimize(_bteg_nll, x0, args=(ytr,), method="Nelder-Mead", options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-5, "fatol": 1e-5})
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun): best = r
    return best.x
def bteg_unpack(x): return x[0], x[1], 1 / (1 + math.exp(-x[2])), math.exp(x[3]), 2.05 + math.exp(x[4])

# ---------------------------------------------------------------- 1. training pool: 199 panel names, pre-2020 only
def panel_name(args):
    pn, y = args
    out = {}
    try:
        res = arch_model(y, vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False); pr = res.params
        mu = float(pr["mu"]); s2 = np.empty(len(y)); s2[0] = np.var(y); e0 = y - mu
        for k in range(1, len(y)): s2[k] = max(float(pr["omega"]) + float(pr["alpha[1]"]) * e0[k - 1] ** 2 + float(pr["beta[1]"]) * s2[k - 1], 1e-8)
        out["garch_t"] = feats(y, np.sqrt(s2), mu)[ZX + ['z']].dropna().values
        x = fit_bteg(y); mu_b, om, phi, kap, nu = bteg_unpack(x); lam = _bteg_filter(y, mu_b, om, phi, kap, nu)
        out["bteg"] = feats(y, np.exp(lam) * math.sqrt(nu / (nu - 2)), mu_b)[ZX + ['z']].dropna().values
    except Exception:
        return pn, None
    return pn, out
if __name__ == "__main__":
    rr = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
    rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); names = [p for p in cnt[cnt >= 1500].index.tolist()[:200] if p != NV]
    pre = rr[rr.date < CUT]; jobs = []
    for pn in names:
        y = pre[pre.permno == pn].sort_values('date')['ret'].values.astype(float)
        if len(y) >= 750 and np.isfinite(y).all(): jobs.append((pn, y))
    with Pool(15) as pool: res = pool.map(panel_name, jobs)
    TRN = {f: np.concatenate([o[f] for _, o in res if o is not None]) for f in ("garch_t", "bteg")}
    lg("training pool: %d names (NVDA excluded), %d rows pre-2020" % (sum(o is not None for _, o in res), len(TRN["garch_t"])))

    # ---------------------------------------------------------------- 2. NVDA filters, refit each January
    px = pd.read_csv(NVDA_CSV, index_col=0, parse_dates=True)["Close"]; px = px[px.index <= "2026-11-13"]
    r = (px.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
    SIG = {f: np.full(n, np.nan) for f in ("garch_t", "gjr_skewt", "bteg")}; MU = {f: np.full(n, np.nan) for f in SIG}
    FEAT = {f: np.full((n, len(ZX)), np.nan) for f in ("garch_t", "bteg")}
    V = {}; E = {}; HI = np.zeros(n, bool); bx = None
    for m in ("garch_t", "gjr_skewt", "bteg", "fhs"): V[m] = {a: np.full(n, np.nan) for a in A}; E[m] = {a: np.full(n, np.nan) for a in A}
    for yr in range(2020, 2027):
        tr = dates < pd.Timestamp("%d-01-01" % yr); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1)))
        if te.sum() == 0: continue
        ntr = int(tr.sum()); last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; tl = te[:last]
        for m, o, dist in (("garch_t", 0, "t"), ("gjr_skewt", 1, "skewt")):
            pr = arch_model(ytr, vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False).params
            sig = arch_model(y[:last], vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fix(pr.values).conditional_volatility
            mu = float(pr["mu"]); SIG[m][te] = sig[tl]; MU[m][te] = mu
            for a in A:
                if dist == "t": zq, ze = t_std_q_es(a, float(pr["nu"]))
                else:
                    ppf = lambda u, pr=pr: SK.ppf(np.clip(u, 1e-300, 1.0), np.array([float(pr["eta"]), float(pr["lambda"])]))
                    zq, ze = float(ppf(np.array([a]))[0]), es_from_ppf(ppf, a)
                V[m][a][te] = mu + sig[tl] * zq; E[m][a][te] = mu + sig[tl] * ze
            if m == "garch_t":
                df = feats(y[:last], sig, mu); FEAT["garch_t"][te] = df[ZX].values[tl]; z = df['z'].values; ztr = z[:ntr]
                for a in A:
                    qz = float(np.quantile(ztr, a)); ez = float(ztr[ztr <= qz].mean()); V["fhs"][a][te] = mu + sig[tl] * qz; E["fhs"][a][te] = mu + sig[tl] * ez
                mk = df['mk63'].values; HI[te] = mk[tl] > float(np.nanquantile(mk[:ntr], 0.9))
        bx = fit_bteg(ytr, bx); mu, om, phi, kap, nu = bteg_unpack(bx); lam = _bteg_filter(y[:last], mu, om, phi, kap, nu)
        sd = np.exp(lam) * math.sqrt(nu / (nu - 2)); SIG["bteg"][te] = sd[tl]; MU["bteg"][te] = mu
        FEAT["bteg"][te] = feats(y[:last], sd, mu)[ZX].values[tl]
        for a in A:
            zq, ze = t_std_q_es(a, nu); V["bteg"][a][te] = mu + sd[tl] * zq; E["bteg"][a][te] = mu + sd[tl] * ze
    lg("NVDA filters done")

    # ---------------------------------------------------------------- 3. our model, frozen, applied to NVDA
    T = np.isfinite(V["garch_t"][0.01]); yt = y[T]; hi = HI[T]
    def lev_for(a):
        floor = a / SUBN / 2.0
        return np.unique(np.concatenate([a * ((np.arange(SUBN) + 0.5) / SUBN), np.exp(np.linspace(math.log(floor), math.log(a), KEXTRA)), [a]]))
    LEVS = sorted(set(TAUS) | set(float(u) for a in A for u in lev_for(a)))
    def fit_pred(args):
        f, t = args
        Xtr = TRN[f][:, :-1]; ztr = TRN[f][:, -1]
        return f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(Xtr, ztr).predict(FEAT[f][T])
    with Pool(15) as pool: preds = pool.map(fit_pred, [(f, t) for f in ("garch_t", "bteg") for t in LEVS])
    QZ = {(f, round(t, 12)): p for f, t, p in preds}
    lg("pooled bodies fitted: %d quantile models" % len(preds))
    DIAG = {}
    for f, lab in (("garch_t", ""), ("bteg", "_bteg")):
        tail = GPDTail(TRN[f][:, -1], P0); DIAG["gpd" + lab] = dict(xi=round(tail.xi, 4), beta=round(tail.beta, 4), u=round(-tail.u, 4))
        mu = MU[f][T]; sg = SIG[f][T]; Q = lambda t: QZ[(f, round(t, 12))]
        eng = {t: (np.minimum(Q(t), tail.q(t)) if t <= P0 else Q(t)) for t in TAUS}
        Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Q(t) for t in TAUS], 1), 1)
        for a in A:
            lev = lev_for(a); QB = np.stack([Q(float(u)) for u in lev], 1)
            zq = Es[:, TAUS.index(a)]
            es = converged_es(a, lev, QB, tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq)
            mids = [a * (j + 0.5) / SUBN for j in range(SUBN)]
            zqb = bod[:, TAUS.index(a)]; esb = np.minimum(np.stack([Q(u) for u in mids], 1).mean(1), zqb - 1e-6)   # body ES: 20 midpoints, as the paper
            for m, vq, ve in (("engine" + lab, zq, es), ("body" + lab, zqb, esb)):
                V.setdefault(m, {a2: np.full(n, np.nan) for a2 in A}); E.setdefault(m, {a2: np.full(n, np.nan) for a2 in A})
                V[m][a][T] = mu + sg * vq; E[m][a][T] = mu + sg * ve
    M_ = ["garch_t", "gjr_skewt", "bteg", "fhs", "body", "engine", "body_bteg", "engine_bteg"]
    out = {"test": [str(dates[T][0].date()), str(dates[T][-1].date())], "n_test": int(T.sum()), "n_high_score": int(hi.sum()),
           "training": "199 CRSP panel names, pre-2020 rows only, NVDA excluded", "diag": DIAG, "by_level": {}}
    last3 = dates[T] >= pd.Timestamp("2023-11-14")
    for a in A:
        L = {m: fz0(yt, V[m][a][T], E[m][a][T], a) for m in M_}; P = {m: pin(yt, V[m][a][T], a) for m in M_}
        blk = {"mean_fz0": {m: round(float(L[m].mean()), 4) for m in M_}, "mean_pinball": {m: round(float(P[m].mean()), 5) for m in M_},
               "breach_pct": {m: round(100 * float((yt <= V[m][a][T]).mean()), 2) for m in M_},
               "kupiec_p": {m: round(kupiec(yt <= V[m][a][T], a), 3) for m in M_}, "engine_vs": {}, "engine_bteg_vs": {}}
        for ref in ("engine", "engine_bteg"):
            for m in M_:
                if m == ref: continue
                res = {}
                for lab, mask in (("all", np.ones(len(yt), bool)), ("last_3y", last3), ("high_score", hi), ("rest", ~hi)):
                    d = L[m][mask] - L[ref][mask]; tf = nw_t(d)   # DM > 0: ref (our model) has the LOWER loss
                    dp = P[m][mask] - P[ref][mask]
                    res[lab] = {"fz0_gain": round(float(d.mean()), 4), "fz0_DM": round(tf, 2), "p2": round(float(2 * (1 - stats.norm.cdf(abs(tf)))), 3),
                                "pin_gain_pct": round(100 * float(dp.mean() / P[m][mask].mean()), 2), "pin_DM": round(nw_t(dp), 2), "n": int(mask.sum())}
                blk[ref + "_vs"][m] = res
        out["by_level"][str(a)] = blk
    json.dump(out, open(OUT, "w"), indent=1)
    np.savez(os.path.join(RES, "nvda_engine_daily.npz"), dates=np.array([str(d.date()) for d in dates[T]]), y=yt, hi=hi,
             feat_g=FEAT["garch_t"][T], sig_g=SIG["garch_t"][T], mu_g=MU["garch_t"][T], sig_b=SIG["bteg"][T],
             **{"V_%s_%s" % (m, a): V[m][a][T] for m in M_ for a in A}, **{"E_%s_%s" % (m, a): E[m][a][T] for m in M_ for a in A},
             trn_feat=TRN["garch_t"][:, :-1], trn_z=TRN["garch_t"][:, -1])
    for a in A:
        b = out["by_level"][str(a)]
        print("\n=== alpha %.3f   (DM > 0: our model has the lower loss)" % a)
        print("%-12s %8s %9s %7s %6s" % ("model", "FZ0", "pinball", "breach", "Kup p"))
        for m in M_: print("%-12s %8.4f %9.5f %6.2f%% %6.3f" % (m, b["mean_fz0"][m], b["mean_pinball"][m], b["breach_pct"][m], b["kupiec_p"][m]))
        for ref in ("engine", "engine_bteg"):
            print(" %s vs:" % ref)
            for m, res in b[ref + "_vs"].items():
                print("   %-11s " % m + "  ".join("%s %+.4f (DM %+.2f)" % (k, v["fz0_gain"], v["fz0_DM"]) for k, v in res.items()))
    lg("wrote %s" % OUT)
