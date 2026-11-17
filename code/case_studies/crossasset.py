# crossasset.py -- the pre-registered cross-asset case study. Protocol, hypotheses, margins, predictions and decision
# rules are fixed in docs/CASE_STUDY_PRECOMMIT.md (written before the confirmatory data were downloaded); nothing in
# this file may change them. Run from anywhere:
#   GBC_PROJ=<CRSP panel dir> python code/case_studies/crossasset.py
# Data: data/case_studies/<TICKER>.csv (yfinance adjusted closes, not committed). Output:
#   results/case_studies/crossasset_results.json   aggregates (committed)
#   data/case_studies/crossasset_daily_<set>.npz     per-instrument daily forecasts and losses (not committed)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "code", "paper"))
import nvda_engine_case as C
from es_integral import converged_es
PANEL_DIR = os.environ["GBC_PROJ"]; DATA = os.path.join(ROOT, "data", "case_studies")
A, TAUS, ZX, P0, SUBN, KEXTRA, HGB = C.A, C.TAUS, C.ZX, C.P0, C.SUBN, C.KEXTRA, C.HGB
SETS = {"confirmatory": ["MSFT", "PFE", "WMT", "CAT", "BA", "NFLX", "IWM", "XLU", "HYG", "LQD", "SLV", "DBA", "GBPUSD=X", "AUDUSD=X", "ETH-USD"],
        "exploratory": ["KO", "JNJ", "XOM", "JPM", "AAPL", "AMD", "TSLA", "NVDA", "SPY", "EEM", "TLT", "GLD", "USO", "UNG", "EURUSD=X", "JPY=X", "BTC-USD"]}
CLASS = {"stocks": ["MSFT", "PFE", "WMT", "CAT", "BA", "NFLX", "KO", "JNJ", "XOM", "JPM", "AAPL", "AMD", "TSLA", "NVDA"],
         "equity ETFs": ["IWM", "XLU", "SPY", "EEM"], "credit and rates": ["HYG", "LQD", "TLT"],
         "commodities": ["SLV", "DBA", "GLD", "USO", "UNG"], "FX": ["GBPUSD=X", "AUDUSD=X", "EURUSD=X", "JPY=X"], "crypto": ["ETH-USD", "BTC-USD"]}
CLS = {t: c for c, ts in CLASS.items() for t in ts}
YEARS = list(range(2020, 2027)); CUT = pd.Timestamp("2020-01-01"); END = "2026-11-13"
DELTA = 0.01; MINTR = 750
FROZEN = os.environ.get("CS_FROZEN") == "1"   # post-hoc diagnostic: fit every per-instrument model once, pre-2020
SFX = "_frozen" if FROZEN else ""; NW_ = int(os.environ.get("CS_WORKERS", "15"))
BENCH = ["hs500", "ewma", "garch_t", "gjr_skewt", "fhs", "garch_evt", "bteg"]
OURS = ["engine", "body_bteg", "engine_bteg", "gate"]
FZROWS = BENCH + OURS; PINROWS = FZROWS + ["caviar"]
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)

def lev_for(a):
    floor = a / SUBN / 2.0
    return np.unique(np.concatenate([a * ((np.arange(SUBN) + 0.5) / SUBN), np.exp(np.linspace(math.log(floor), math.log(a), KEXTRA)), [a]]))
LEVS = sorted(set(TAUS) | set(float(u) for a in A for u in lev_for(a)))

def caviar_sav(rtrain, rfull, tau):     # verbatim specification of code/paper/frtb_caviar.py
    r = rfull; n = len(r); q0 = np.quantile(rtrain[:300], tau); absr = np.abs(r)
    def path(b):
        f = np.empty(n); f[0] = q0
        for i in range(1, n): f[i] = b[0] + b[1] * f[i - 1] + b[2] * absr[i - 1]
        return f
    ntr = len(rtrain)
    def obj(b):
        f = path(b)[:ntr]; d = rtrain - f; return np.mean(np.where(d >= 0, tau * d, (tau - 1) * d))
    b0 = np.array([q0 * 0.1, 0.8, -0.1 if tau < 0.5 else 0.1])
    try: b = optimize.minimize(obj, b0, method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-5}).x
    except Exception: b = b0
    return path(b)

def christoffersen(h, a):
    h = h.astype(int); n00 = n01 = n10 = n11 = 0
    for i in range(1, len(h)):
        if h[i-1] == 0: n00 += h[i] == 0; n01 += h[i] == 1
        else: n10 += h[i] == 0; n11 += h[i] == 1
    T = len(h); x = int(h.sum()); pp = x / T
    ll = lambda k, pr: k * math.log(pr) if k > 0 and pr > 0 else 0.0
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (ll(n00 + n10, 1 - p) + ll(n01 + n11, p) - ll(n00, 1 - p01) - ll(n01, p01) - ll(n10, 1 - p11) - ll(n11, p11))
    lr_uc = -2 * (ll(T - x, 1 - a) + ll(x, a) - ll(T - x, 1 - pp) - ll(x, pp))
    return float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))

def mcs(L, names, alpha=0.10, B=1000, blk=10, seed=0):   # code/paper/frtb_caviar.py, seeded
    rng = np.random.default_rng(seed); surv = list(range(L.shape[1])); pv = {}
    def bidx(T):
        idx = np.empty(T, int); i = 0
        while i < T:
            s = rng.integers(0, T); l = rng.geometric(1 / blk)
            for j in range(l):
                if i < T: idx[i] = (s + j) % T; i += 1
        return idx
    while len(surv) > 1:
        Ls = L[:, surv]; means = Ls.mean(0); M = len(surv); dij = means[:, None] - means[None, :]
        bd = np.zeros((B, M, M))
        for bb in range(B): ix = bidx(Ls.shape[0]); mb = Ls[ix].mean(0); bd[bb] = mb[:, None] - mb[None, :]
        varij = bd.var(0) + 1e-12; TR = np.nanmax(np.abs(dij) / np.sqrt(varij))
        bootTR = np.array([np.nanmax(np.abs(bd[bb] - dij) / np.sqrt(varij)) for bb in range(B)])
        pval = float(np.mean(bootTR >= TR)); worst = surv[int(np.argmax(np.nanmax(dij / np.sqrt(varij), axis=1)))]
        pv[names[worst]] = round(pval, 3)
        if pval >= alpha: break
        surv.remove(worst)
    return {"in_MCS_90": [names[i] for i in surv], "elimination_p": pv}

MODELS = {}; TAILS = {}
def instrument(tk):
    s = pd.read_csv(os.path.join(DATA, tk.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]
    s = s[(s > 0) & (s.index <= END)]; removed = []
    if tk.endswith("=X"):
        r_ = s.pct_change(); bad = ((r_.abs() > 0.03) & (r_.shift(-1).abs() > 0.03) & (np.sign(r_) != np.sign(r_.shift(-1)))).fillna(False).values
        removed = [str(d.date()) for d in s.index[bad]]; s = s[~bad]
    r = (s.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
    V = {m: {a: np.full(n, np.nan) for a in A} for m in PINROWS}; E = {m: {a: np.full(n, np.nan) for a in A} for m in FZROWS}
    SIG = {f: np.full(n, np.nan) for f in ("garch_t", "bteg")}; MU = {f: np.full(n, np.nan) for f in SIG}
    FEAT = {f: np.full((n, len(ZX)), np.nan) for f in SIG}; YR = np.zeros(n, int); HI = np.zeros(n, bool); bx = None
    # EWMA (RiskMetrics): no parameters to refit
    s2 = np.empty(n); s2[0] = np.var(y[:250])
    for k in range(1, n): s2[k] = 0.94 * s2[k - 1] + 0.06 * y[k - 1] ** 2
    for yr in YEARS:
        tr = dates < (CUT if FROZEN else pd.Timestamp("%d-01-01" % yr)); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1)))
        ntr = int(tr.sum())
        if te.sum() == 0 or ntr < MINTR: continue
        last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; tl = te[:last]; YR[te] = yr
        for a in A:
            za = stats.norm.ppf(a); V["ewma"][a][te] = np.sqrt(s2[te]) * za; E["ewma"][a][te] = -np.sqrt(s2[te]) * stats.norm.pdf(za) / a
            for i in np.where(te)[0]:
                w = y[max(0, i - 500):i]; q = float(np.quantile(w, a)); V["hs500"][a][i] = q; E["hs500"][a][i] = float(w[w <= q].mean())
            V["caviar"][a][te] = caviar_sav(ytr, y[:last], a)[tl]
        for m, o, dist in (("garch_t", 0, "t"), ("gjr_skewt", 1, "skewt")):
            pr = arch_model(ytr, vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False).params
            sig = arch_model(y[:last], vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fix(pr.values).conditional_volatility
            mu = float(pr["mu"])
            for a in A:
                if dist == "t": zq, ze = C.t_std_q_es(a, float(pr["nu"]))
                else:
                    ppf = lambda u, pr=pr: C.SK.ppf(np.clip(u, 1e-300, 1.0), np.array([float(pr["eta"]), float(pr["lambda"])]))
                    zq, ze = float(ppf(np.array([a]))[0]), C.es_from_ppf(ppf, a)
                V[m][a][te] = mu + sig[tl] * zq; E[m][a][te] = mu + sig[tl] * ze
            if m == "garch_t":
                SIG["garch_t"][te] = sig[tl]; MU["garch_t"][te] = mu
                df = C.feats(y[:last], sig, mu); FEAT["garch_t"][te] = df[ZX].values[tl]; ztr = df['z'].values[:ntr]; ztr = ztr[np.isfinite(ztr)]
                x = -ztr; u = float(np.quantile(x, 0.90)); exc = x[x > u] - u; xi, _, be = stats.genpareto.fit(exc, floc=0.0)
                for a in A:
                    qz = float(np.quantile(ztr, a)); V["fhs"][a][te] = mu + sig[tl] * qz; E["fhs"][a][te] = mu + sig[tl] * float(ztr[ztr <= qz].mean())
                    vl = u + (be / xi) * ((a / 0.10) ** (-xi) - 1.0) if abs(xi) > 1e-6 else u + be * math.log(0.10 / a)
                    el = (vl + be - xi * u) / (1.0 - xi)               # McNeil-Frey closed form (xi < 1)
                    V["garch_evt"][a][te] = mu - sig[tl] * vl; E["garch_evt"][a][te] = mu - sig[tl] * el
                mk = df['mk63'].values; HI[te] = mk[tl] > float(np.nanquantile(mk[:ntr], 0.9))
        bx = C.fit_bteg(ytr, bx); mu, om, phi, kap, nu = C.bteg_unpack(bx); lam = C._bteg_filter(y[:last], mu, om, phi, kap, nu)
        sd = np.exp(lam) * math.sqrt(nu / (nu - 2)); SIG["bteg"][te] = sd[tl]; MU["bteg"][te] = mu
        FEAT["bteg"][te] = C.feats(y[:last], sd, mu)[ZX].values[tl]
        for a in A:
            zq, ze = C.t_std_q_es(a, nu); V["bteg"][a][te] = mu + sd[tl] * zq; E["bteg"][a][te] = mu + sd[tl] * ze
    T = YR > 0
    for f in ("garch_t", "bteg"):
        X = FEAT[f][T]; ok = np.isfinite(X).all(1); tail = TAILS[f]; Qc = {}
        for t in LEVS:
            q = np.full(len(X), np.nan); q[ok] = MODELS[(f, round(t, 12))].predict(X[ok]); Qc[round(t, 12)] = q
        Q = lambda t: Qc[round(float(t), 12)]
        eng = {t: (np.minimum(Q(t), tail.q(t)) if t <= P0 else Q(t)) for t in TAUS}
        Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Q(t) for t in TAUS], 1), 1)
        mu = MU[f][T]; sg = SIG[f][T]
        for a in A:
            lev = lev_for(a); QB = np.stack([Q(u) for u in lev], 1); zq = Es[:, TAUS.index(a)]; es = np.full(len(zq), np.nan)
            es[ok] = converged_es(a, lev, QB[ok], tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq[ok])
            zqb = bod[:, TAUS.index(a)]; esb = np.minimum(np.stack([Q(a * (j + 0.5) / SUBN) for j in range(SUBN)], 1).mean(1), zqb - 1e-6)
            if f == "garch_t": V["engine"][a][T] = mu + sg * zq; E["engine"][a][T] = mu + sg * es
            else:
                V["engine_bteg"][a][T] = mu + sg * zq; E["engine_bteg"][a][T] = mu + sg * es
                V["body_bteg"][a][T] = mu + sg * zqb; E["body_bteg"][a][T] = mu + sg * esb
    for a in A: V["gate"][a] = np.where(HI, V["bteg"][a], V["garch_t"][a]); E["gate"][a] = np.where(HI, E["bteg"][a], E["garch_t"][a])
    keep = T & np.all([np.isfinite(V[m][a]) for m in PINROWS for a in A] + [np.isfinite(E[m][a]) for m in FZROWS for a in A], axis=0)
    out = {"dates": dates[keep], "y": y[keep], "removed": removed, "n_dropped": int(T.sum() - keep.sum())}
    for a in A:
        for m in FZROWS: out[("fz0", m, a)] = C.fz0(y[keep], V[m][a][keep], E[m][a][keep], a)
        for m in PINROWS:
            out[("pin", m, a)] = C.pin(y[keep], V[m][a][keep], a); out[("hit", m, a)] = (y[keep] <= V[m][a][keep])
    return tk, out

def nw_se(x, l=10):
    x = np.asarray(x, float); m = len(x); d = x - x.mean(); v = np.mean(d * d)
    for k in range(1, l + 1): v += 2 * (1 - k / (l + 1)) * np.mean(d[k:] * d[:-k])
    return math.sqrt(v / m)
def holm(p):
    k = sorted(range(len(p)), key=lambda i: p[i]); adj = [0.0] * len(p); run = 0.0
    for r, i in enumerate(k): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj

def analyse(R, tickers, nyse):
    res = {"per_instrument": {}, "pooled": {}, "by_class": {}}
    for tk in tickers:
        o = R[tk]; blk = {"class": CLS[tk], "n_test": int(len(o["y"])), "fx_ticks_removed": o["removed"], "levels": {}}
        for a in A:
            row = {}
            for m in PINROWS:
                h = o[("hit", m, a)]
                row[m] = {"breach_pct": round(100 * float(h.mean()), 2), "kupiec_p": round(C.kupiec(h, a), 3), "christoffersen_p": round(christoffersen(h, a), 3),
                          "pinball_rel": round(float(o[("pin", m, a)].mean() / o[("pin", "garch_t", a)].mean()), 4)}
                if m in FZROWS:
                    row[m]["fz0"] = round(float(o[("fz0", m, a)].mean()), 4)
                    if m != "engine":
                        d = o[("fz0", "engine", a)] - o[("fz0", m, a)]
                        row[m]["engine_minus_this"] = round(float(d.mean()), 4); row[m]["DM_engine_minus_this"] = round(float(d.mean() / nw_se(d)), 2)
            blk["levels"][str(a)] = row
        res["per_instrument"][tk] = blk
    for a in A:
        # pooled per-date series on the NYSE calendar, equal weight over instruments forecasting that date
        def pooled(kind, m, norm=False):
            parts = []
            for tk in tickers:
                o = R[tk]; v = o[(kind, m, a)]
                if norm: v = v / o[("pin", "garch_t", a)].mean()
                parts.append(pd.Series(v, index=o["dates"]))
            S = pd.concat(parts, axis=1); S = S[S.index.isin(nyse)]
            return S.mean(axis=1)
        P = {"fz0": {}, "pinball": {}}
        FZ = {m: pooled("fz0", m) for m in FZROWS}; PN = {m: pooled("pin", m, True) for m in PINROWS}
        nonin = {}; pvals = []
        for m in FZROWS:
            if m == "engine": continue
            d = (FZ["engine"] - FZ[m]).values; mean = float(d.mean()); se = nw_se(d)
            p_ni = float(1 - stats.norm.cdf((DELTA - mean) / se))
            nonin[m] = {"engine_minus_this": round(mean, 4), "se": round(se, 4), "upper95": round(mean + 1.645 * se, 4),
                        "two_sided_DM": round(mean / se, 2), "p_noninferior": round(p_ni, 4),
                        "instruments_engine_lower": int(sum(R[tk][("fz0", "engine", a)].mean() < R[tk][("fz0", m, a)].mean() for tk in tickers))}
            if m in BENCH: pvals.append((m, p_ni))
        adj = holm([p for _, p in pvals])
        for (m, _), pa in zip(pvals, adj): nonin[m]["p_noninferior_holm"] = round(pa, 4); nonin[m]["noninferior_at_5pct"] = pa < 0.05
        P["fz0"]["engine_vs"] = nonin
        d = (FZ["body_bteg"] - FZ["engine"]).values
        P["fz0"]["body_bteg_minus_engine"] = {"mean": round(float(d.mean()), 4), "two_sided_DM": round(float(d.mean() / nw_se(d)), 2),
                                              "instruments_body_bteg_lower": int(sum(R[tk][("fz0", "body_bteg", a)].mean() < R[tk][("fz0", "engine", a)].mean() for tk in tickers))}
        M = np.column_stack([FZ[m].values for m in FZROWS]); P["fz0"]["mcs"] = mcs(M, FZROWS)
        P["fz0"]["mean"] = {m: round(float(FZ[m].mean()), 4) for m in FZROWS}
        dcv = (PN["engine"] - PN["caviar"]).values; delta_p = 0.01 * float(PN["caviar"].mean()); se = nw_se(dcv)
        P["pinball"]["engine_vs_caviar"] = {"engine_minus_caviar": round(float(dcv.mean()), 5), "margin": round(delta_p, 5), "upper95": round(float(dcv.mean()) + 1.645 * se, 5),
                                            "p_noninferior": round(float(1 - stats.norm.cdf((delta_p - dcv.mean()) / se)), 4), "two_sided_DM": round(float(dcv.mean() / se), 2)}
        P["pinball"]["mcs"] = mcs(np.column_stack([PN[m].values for m in PINROWS]), PINROWS)
        P["pinball"]["mean_rel_to_garch_t"] = {m: round(float(PN[m].mean()), 4) for m in PINROWS}
        P["coverage_pass_count"] = {m: {"kupiec": int(sum(res["per_instrument"][tk]["levels"][str(a)][m]["kupiec_p"] >= 0.05 for tk in tickers)),
                                        "christoffersen": int(sum(res["per_instrument"][tk]["levels"][str(a)][m]["christoffersen_p"] >= 0.05 for tk in tickers))} for m in PINROWS}
        P["n_dates"] = int(len(FZ["engine"])); res["pooled"][str(a)] = P
        bc = {}
        for c in CLASS:
            tks = [t for t in tickers if CLS[t] == c]
            if not tks: continue
            parts = [pd.Series(R[t][("fz0", "engine", a)] - R[t][("fz0", "garch_t", a)], index=R[t]["dates"]) for t in tks]
            S = pd.concat(parts, axis=1); S = S[S.index.isin(nyse)].mean(axis=1).values
            bc[c] = {"n_instruments": len(tks), "engine_minus_garch_t": round(float(S.mean()), 4), "DM": round(float(S.mean() / nw_se(S)), 2)}
        res["by_class"][str(a)] = bc
    return res

if __name__ == "__main__":
    allt = SETS["confirmatory"] + SETS["exploratory"]
    rr = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
    rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    ch = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_chars.csv")); excl = set(ch[ch.ticker.isin(allt)].permno)
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); top = cnt[cnt >= 1500].index.tolist()[:200]
    names = [p for p in top if p not in excl]; pre = rr[rr.date < CUT]
    jobs = [(pn, pre[pre.permno == pn].sort_values('date')['ret'].values.astype(float)) for pn in names]
    jobs = [(pn, yy) for pn, yy in jobs if len(yy) >= 750 and np.isfinite(yy).all()]
    with Pool(NW_) as pool: res = pool.map(C.panel_name, jobs)
    TRN = {f: np.concatenate([o[f] for _, o in res if o is not None]) for f in ("garch_t", "bteg")}
    def _fit(args):
        f, t = args
        return f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(TRN[f][:, :-1], TRN[f][:, -1])
    globals()["_fit"] = _fit
    with Pool(NW_) as pool: fitted = pool.map(_fit, [(f, t) for f in ("garch_t", "bteg") for t in LEVS], chunksize=4)
    MODELS.update({(f, round(t, 12)): m for f, t, m in fitted}); TAILS.update({f: C.GPDTail(TRN[f][:, -1], P0) for f in TRN})
    meta = {"training_names": int(sum(o is not None for _, o in res)), "training_rows": int(len(TRN["garch_t"])),
            "panel_tickers_excluded": sorted(set(ch[ch.permno.isin(excl & set(top))].ticker)),
            "gpd": {f: dict(xi=round(TAILS[f].xi, 4), beta=round(TAILS[f].beta, 4), u=round(-TAILS[f].u, 4)) for f in TAILS}}
    lg("training: %d names, %d rows; excluded panel tickers %s" % (meta["training_names"], meta["training_rows"], meta["panel_tickers_excluded"]))
    with Pool(NW_) as pool: R = dict(pool.map(instrument, allt))
    lg("instruments done")
    spy = pd.read_csv(os.path.join(DATA, "SPY.csv"), index_col=0, parse_dates=True).index; nyse = set(spy[(spy >= CUT) & (spy <= END)])
    OUT = {"precommit": "docs/CASE_STUDY_PRECOMMIT.md", "delta": DELTA, "meta": meta}
    for sname, tks in SETS.items():
        OUT[sname] = analyse(R, tks, nyse)
        np.savez(os.path.join(DATA, "crossasset%s_daily_%s.npz" % (SFX, sname)),
                 **{"%s|%s|%s|%s" % (tk, k[0], k[1], k[2]): v for tk in tks for k, v in R[tk].items() if isinstance(k, tuple)},
                 **{"%s|dates" % tk: np.array([str(d.date()) for d in R[tk]["dates"]]) for tk in tks})
    # pre-registered hypotheses and decision rule, graded on the confirmatory set
    cf = OUT["confirmatory"]["pooled"]; g = {}
    for a in ("0.01", "0.025"):
        ev = cf[a]["fz0"]["engine_vs"]
        g[a] = {"H1_noninferior_to_garch_t": bool(ev["garch_t"]["noninferior_at_5pct"]),
                "H2_noninferior_to_each_other_benchmark": {m: bool(ev[m]["noninferior_at_5pct"]) for m in BENCH if m != "garch_t"},
                "H3_engine_in_fz0_mcs90": "engine" in cf[a]["fz0"]["mcs"]["in_MCS_90"],
                "H4_noninferior_to_caviar_pinball": cf[a]["pinball"]["engine_vs_caviar"]["p_noninferior"] < 0.05,
                "body_bteg_not_significantly_better": abs(cf[a]["fz0"]["body_bteg_minus_engine"]["two_sided_DM"]) < 1.96 or cf[a]["fz0"]["body_bteg_minus_engine"]["mean"] > 0,
                "config_change_condition": cf[a]["fz0"]["body_bteg_minus_engine"]["two_sided_DM"] < -1.96 and cf[a]["fz0"]["body_bteg_minus_engine"]["instruments_body_bteg_lower"] >= 10}
    g["CHANGE_RECOMMENDED_CONFIGURATION"] = all(g[a]["config_change_condition"] for a in ("0.01", "0.025"))
    OUT["graded"] = g
    json.dump(OUT, open(os.path.join(ROOT, "results", "case_studies", "crossasset%s_results.json" % SFX), "w"), indent=1, default=str)
    for sname in SETS:
        print("\n######## %s set (%d instruments)" % (sname.upper(), len(SETS[sname])))
        for a in ("0.01", "0.025"):
            p = OUT[sname]["pooled"][a]; print("--- alpha %s, %d NYSE dates. Accuracy layer minus benchmark (negative = ours lower); margin %.2f" % (a, p["n_dates"], DELTA))
            for m, v in p["fz0"]["engine_vs"].items():
                print("   %-12s %+.4f  upper95 %+.4f  DM %+5.2f  ours lower on %2d/%d  %s" % (m, v["engine_minus_this"], v["upper95"], v["two_sided_DM"], v["instruments_engine_lower"],
                      len(SETS[sname]), (("NON-INFERIOR" if v["noninferior_at_5pct"] else "not shown non-inferior") + " (Holm p %.3f)" % v["p_noninferior_holm"]) if "p_noninferior_holm" in v else ""))
            print("   FZ0 MCS90: %s" % p["fz0"]["mcs"]["in_MCS_90"]); print("   pinball MCS90: %s" % p["pinball"]["mcs"]["in_MCS_90"])
            print("   vs CAViaR (pinball): %s" % p["pinball"]["engine_vs_caviar"]); print("   body_bteg minus engine: %s" % p["fz0"]["body_bteg_minus_engine"])
            print("   coverage passes (Kupiec): %s" % {m: v["kupiec"] for m, v in p["coverage_pass_count"].items()})
            print("   by class (engine minus GARCH-t): %s" % OUT[sname]["by_class"][a])
    print("\nGRADED (confirmatory):", json.dumps(g, indent=1))
    lg("done")
