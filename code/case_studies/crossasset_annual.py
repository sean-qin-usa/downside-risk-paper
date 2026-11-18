# crossasset_annual.py -- the like-for-like annual protocol of docs/CASE_STUDY_PRECOMMIT.md (section of 2026-11-18;
# predictions A1-A5 there). All 32 instruments. Per-instrument models refit each January, and the estimator's pooled
# parts RETRAINED each January on all panel rows before it, as Algorithm 1 of the paper specifies. Not confirmatory.
#   GBC_PROJ=<CRSP panel dir> python code/case_studies/crossasset_annual.py
# Output: results/case_studies/crossasset_annual_results.json; data/case_studies/crossasset_annual_daily.npz
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import crossasset as X
C = X.C; converged_es = X.converged_es
A, TAUS, ZX, P0, SUBN, HGB, LEVS, lev_for = X.A, X.TAUS, X.ZX, X.P0, X.SUBN, X.HGB, X.LEVS, X.lev_for
YEARS, CUT, END, DATA, ROOT, DELTA = X.YEARS, X.CUT, X.END, X.DATA, X.ROOT, X.DELTA
ALLT = X.SETS["confirmatory"] + X.SETS["exploratory"]; STK = X.CLASS["stocks"]
BENCH = ["hs500", "ewma", "garch_t", "gjr_skewt", "fhs", "garch_evt", "bteg"]
OURS = ["body", "engine", "body_bteg", "engine_bteg", "body_frz", "engine_frz"]
FZ = BENCH + OURS; P11 = ["garch_t", "gjr_skewt", "bteg", "body", "engine", "body_bteg", "engine_bteg", "body_frz", "engine_frz"]
PY = lambda yr: min(yr, 2025)          # pool year: the panel ends in 2024
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
POOL = {}; MODELS = {}; TAILS = {}

def name_job(args):
    key, y = args; return key, C.panel_name((key, y))[1]
def fit_job(args):
    c, f, t = args; D = POOL[(c, f)]
    return c, f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(D[:, :-1], D[:, -1])

def instrument(tk):
    s = pd.read_csv(os.path.join(DATA, tk.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]; s = s[(s > 0) & (s.index <= END)]
    if tk.endswith("=X"):
        r_ = s.pct_change(); bad = ((r_.abs() > 0.03) & (r_.shift(-1).abs() > 0.03) & (np.sign(r_) != np.sign(r_.shift(-1)))).fillna(False).values; s = s[~bad]
    r = (s.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
    V = {m: {a: np.full(n, np.nan) for a in A} for m in FZ}; E = {m: {a: np.full(n, np.nan) for a in A} for m in FZ}
    Q11 = {m: np.full((n, len(TAUS)), np.nan) for m in P11}
    SIG = {f: np.full(n, np.nan) for f in ("garch_t", "bteg")}; MU = {f: np.full(n, np.nan) for f in SIG}
    FEAT = {f: np.full((n, len(ZX)), np.nan) for f in SIG}; YR = np.zeros(n, int); HI = np.zeros(n, bool); bx = None
    s2 = np.empty(n); s2[0] = np.var(y[:250])
    for k in range(1, n): s2[k] = 0.94 * s2[k - 1] + 0.06 * y[k - 1] ** 2
    for yr in YEARS:
        tr = dates < pd.Timestamp("%d-01-01" % yr); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1))); ntr = int(tr.sum())
        if te.sum() == 0 or ntr < X.MINTR: continue
        last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; tl = te[:last]; YR[te] = yr
        for a in A:
            za = stats.norm.ppf(a); V["ewma"][a][te] = np.sqrt(s2[te]) * za; E["ewma"][a][te] = -np.sqrt(s2[te]) * stats.norm.pdf(za) / a
            for i in np.where(te)[0]:
                w = y[max(0, i - 500):i]; q = float(np.quantile(w, a)); V["hs500"][a][i] = q; E["hs500"][a][i] = float(w[w <= q].mean())
        for m, o, dist in (("garch_t", 0, "t"), ("gjr_skewt", 1, "skewt")):
            pr = arch_model(ytr, vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False).params
            sig = arch_model(y[:last], vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fix(pr.values).conditional_volatility; mu = float(pr["mu"])
            if dist == "t":
                nu = float(pr["nu"]); sc = math.sqrt(nu / (nu - 2)); zt = stats.t.ppf(TAUS, nu) / sc
            else:
                ppf = lambda u, pr=pr: C.SK.ppf(np.clip(u, 1e-300, 1.0), np.array([float(pr["eta"]), float(pr["lambda"])])); zt = ppf(np.array(TAUS))
            Q11[m][te] = mu + sig[tl][:, None] * zt[None, :]
            for a in A:
                if dist == "t": zq, ze = C.t_std_q_es(a, nu)
                else: zq, ze = float(ppf(np.array([a]))[0]), C.es_from_ppf(ppf, a)
                V[m][a][te] = mu + sig[tl] * zq; E[m][a][te] = mu + sig[tl] * ze
            if m == "garch_t":
                SIG["garch_t"][te] = sig[tl]; MU["garch_t"][te] = mu
                df = C.feats(y[:last], sig, mu); FEAT["garch_t"][te] = df[ZX].values[tl]; ztr = df['z'].values[:ntr]; ztr = ztr[np.isfinite(ztr)]
                x = -ztr; u = float(np.quantile(x, 0.90)); exc = x[x > u] - u; xi, _, be = stats.genpareto.fit(exc, floc=0.0)
                for a in A:
                    qz = float(np.quantile(ztr, a)); V["fhs"][a][te] = mu + sig[tl] * qz; E["fhs"][a][te] = mu + sig[tl] * float(ztr[ztr <= qz].mean())
                    vl = u + (be / xi) * ((a / 0.10) ** (-xi) - 1.0) if abs(xi) > 1e-6 else u + be * math.log(0.10 / a); el = (vl + be - xi * u) / (1.0 - xi)
                    V["garch_evt"][a][te] = mu - sig[tl] * vl; E["garch_evt"][a][te] = mu - sig[tl] * el
                mk = df['mk63'].values; HI[te] = mk[tl] > float(np.nanquantile(mk[:ntr], 0.9))
        bx = C.fit_bteg(ytr, bx); mu, om, phi, kap, nu = C.bteg_unpack(bx); lam = C._bteg_filter(y[:last], mu, om, phi, kap, nu)
        sc = math.sqrt(nu / (nu - 2)); sd = np.exp(lam) * sc; SIG["bteg"][te] = sd[tl]; MU["bteg"][te] = mu
        FEAT["bteg"][te] = C.feats(y[:last], sd, mu)[ZX].values[tl]; Q11["bteg"][te] = mu + sd[tl][:, None] * (stats.t.ppf(TAUS, nu) / sc)[None, :]
        for a in A:
            zq, ze = C.t_std_q_es(a, nu); V["bteg"][a][te] = mu + sd[tl] * zq; E["bteg"][a][te] = mu + sd[tl] * ze
        # ---- our rows for this year: pooled parts of pool year PY(yr) (annual) and of 2020 (frozen)
        for f, lab in (("garch_t", ""), ("bteg", "_bteg")):
            for c, sfx in ((PY(yr), ""), (2020, "_frz")):
                if sfx and f == "bteg": continue
                Xf = FEAT[f][te]; ok = np.isfinite(Xf).all(1); tail = TAILS[(c, f)]; Qc = {}
                for t in LEVS:
                    q = np.full(len(Xf), np.nan); q[ok] = MODELS[(c, f, round(t, 12))].predict(Xf[ok]); Qc[round(t, 12)] = q
                Q = lambda t: Qc[round(float(t), 12)]
                eng = {t: (np.minimum(Q(t), tail.q(t)) if t <= P0 else Q(t)) for t in TAUS}
                Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Q(t) for t in TAUS], 1), 1)
                mu_ = MU[f][te]; sg = SIG[f][te]; bn = "body" + lab + sfx; en = "engine" + lab + sfx
                Q11[bn][te] = mu_[:, None] + sg[:, None] * bod; Q11[en][te] = mu_[:, None] + sg[:, None] * Es
                for a in A:
                    lev = lev_for(a); QB = np.stack([Q(u) for u in lev], 1); zq = Es[:, TAUS.index(a)]; es = np.full(len(zq), np.nan)
                    es[ok] = converged_es(a, lev, QB[ok], tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq[ok])
                    zqb = bod[:, TAUS.index(a)]; esb = np.minimum(np.stack([Q(a * (j + 0.5) / SUBN) for j in range(SUBN)], 1).mean(1), zqb - 1e-6)
                    V[en][a][te] = mu_ + sg * zq; E[en][a][te] = mu_ + sg * es; V[bn][a][te] = mu_ + sg * zqb; E[bn][a][te] = mu_ + sg * esb
    T = YR > 0
    keep = T & np.all([np.isfinite(V[m][a]) & np.isfinite(E[m][a]) for m in FZ for a in A] + [np.isfinite(Q11[m]).all(1) for m in P11], axis=0)
    out = {"dates": dates[keep], "hi": HI[keep], "n_dropped": int(T.sum() - keep.sum())}; yk = y[keep]
    for a in A:
        for m in FZ: out[("fz0", m, a)] = C.fz0(yk, V[m][a][keep], E[m][a][keep], a); out[("hit", m, a)] = yk <= V[m][a][keep]
    for m in P11: out[("pin11", m)] = np.mean([C.pin(yk, Q11[m][keep][:, j], t) for j, t in enumerate(TAUS)], axis=0)
    return tk, out

if __name__ == "__main__":
    rr = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
    rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    ch = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "crsp_panel_chars.csv")); excl = set(ch[ch.ticker.isin(ALLT)].permno)
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); names = [p for p in cnt[cnt >= 1500].index.tolist()[:200] if p not in excl]
    SMOKE = os.environ.get("CS_SMOKE") == "1"
    if SMOKE: names = names[:14]; ALLT[:] = ["MSFT", "HYG", "NVDA", "GLD"]
    G = {pn: g.sort_values('date') for pn, g in rr[rr.permno.isin(names)].groupby('permno')}; jobs = []
    for c in range(2020, 2026):
        for pn in names:
            yy = G[pn][G[pn].date < pd.Timestamp("%d-01-01" % c)]['ret'].values.astype(float)
            if len(yy) >= 750 and np.isfinite(yy).all(): jobs.append(((c, pn), yy))
    with Pool(15) as pool: res = dict(pool.map(name_job, jobs, chunksize=4))
    for c in range(2020, 2026):
        for f in ("garch_t", "bteg"): POOL[(c, f)] = np.concatenate([res[(c, pn)][f] for pn in names if res.get((c, pn)) is not None])
    lg("pools: " + ", ".join("%d: %d rows" % (c, len(POOL[(c, "garch_t")])) for c in range(2020, 2026)))
    with Pool(15) as pool: fitted = pool.map(fit_job, [(c, f, t) for c in range(2020, 2026) for f in ("garch_t", "bteg") for t in LEVS], chunksize=4)
    MODELS.update({(c, f, round(t, 12)): m for c, f, t, m in fitted}); TAILS.update({(c, f): C.GPDTail(POOL[(c, f)][:, -1], P0) for c in range(2020, 2026) for f in ("garch_t", "bteg")})
    lg("%d quantile models fitted" % len(MODELS))
    with Pool(15) as pool: R = dict(pool.map(instrument, ALLT))
    lg("instruments done")
    np.savez(os.path.join(DATA, "crossasset_annual_daily%s.npz" % ("_smoke" if SMOKE else "")), **{"%s|%s" % (tk, "|".join(map(str, k)) if isinstance(k, tuple) else k): (np.array([str(d.date()) for d in v]) if k == "dates" else v)
                                                                   for tk in ALLT for k, v in R[tk].items() if k != "n_dropped"})
    nyse = set(pd.read_csv(os.path.join(DATA, "SPY.csv"), index_col=0, parse_dates=True).index)
    def series(tks, key, m, ref, mask=None, norm=False):       # ref minus m, pooled by NYSE date; > 0: m has the lower loss
        parts = []
        for t in tks:
            o = R[t]
            lm = o[(key[0], m) + key[1:]]; lr = o[(key[0], ref) + key[1:]]; v = lr - lm
            if norm: v = v / o[("pin11", "garch_t")].mean()
            msk = np.ones(len(v), bool) if mask is None else (o["hi"] if mask == "flag" else ~o["hi"])
            parts.append(pd.Series(v[msk], index=o["dates"][msk]))
        S = pd.concat(parts, axis=1); return S[S.index.isin(nyse)].mean(axis=1).dropna().values
    st = lambda s: {"gain": round(float(s.mean()), 4), "DM": round(float(s.mean() / X.nw_se(s)), 2)}
    GR = {"all 32": ALLT, "confirmatory 15": [t for t in X.SETS["confirmatory"] if t in ALLT], "14 stocks": [t for t in ALLT if t in STK], "18 non-stocks": [t for t in ALLT if t not in STK]}
    OUT = {"precommit": "docs/CASE_STUDY_PRECOMMIT.md, like-for-like annual protocol", "note": "gain > 0 and DM > 0: the row model has the lower loss than the reference.",
           "pool_rows": {str(c): int(len(POOL[(c, "garch_t")])) for c in range(2020, 2026)}, "flagged_share_pct": round(100 * float(np.mean(np.concatenate([R[t]["hi"] for t in ALLT]))), 1),
           "fz0": {}, "pin11": {}, "coverage": {}}
    for a in A:
        blk = {}
        for gname, tks in GR.items():
            g = {}
            for m in OURS:
                g[m] = {}
                for b in BENCH + (["engine_frz"] if m == "engine" else []) + (["engine"] if m != "engine" else []):
                    s = series(tks, ("fz0", a), m, b); d = st(s)
                    d["instruments_lower"] = int(sum(R[t][("fz0", m, a)].mean() < R[t][("fz0", b, a)].mean() for t in tks)); g[m][b] = d
            # non-inferiority of the accuracy layer, Holm over the seven benchmarks
            rows = []
            for b in BENCH:
                s = -series(tks, ("fz0", a), "engine", b); se = X.nw_se(s); rows.append((b, float(s.mean()), float(s.mean() + 1.645 * se), float(1 - stats.norm.cdf((DELTA - s.mean()) / se))))
            adj = X.holm([r_[3] for r_ in rows])
            g["engine_noninferiority"] = {r_[0]: {"engine_minus_benchmark": round(r_[1], 4), "upper95": round(r_[2], 4), "p_holm": round(p, 4), "noninferior": bool(p < 0.05)} for r_, p in zip(rows, adj)}
            iv = np.array([R[t][("fz0", "engine", a)].mean() - R[t][("fz0", "gjr_skewt", a)].mean() for t in tks])
            g["gjr_advantage_instruments_as_unit"] = {"mean": round(float(iv.mean()), 4), "t": round(float(iv.mean() / (iv.std(ddof=1) / math.sqrt(len(iv)))), 2), "gjr_better_on": int((iv > 0).sum())}
            blk[gname] = g
        Mrows = BENCH + ["body", "engine", "body_bteg", "engine_bteg"]
        def pooled_level(m):
            S = pd.concat([pd.Series(R[t][("fz0", m, a)], index=R[t]["dates"]) for t in ALLT], axis=1); return S[S.index.isin(nyse)].mean(axis=1)
        L = pd.concat([pooled_level(m).rename(m) for m in Mrows], axis=1).dropna()
        blk["mcs90_all32"] = X.mcs(L.values, Mrows); blk["mean_fz0_all32"] = {m: round(float(L[m].mean()), 4) for m in Mrows}
        OUT["fz0"][str(a)] = blk
        OUT["coverage"][str(a)] = {m: int(sum(C.kupiec(R[t][("hit", m, a)], a) >= 0.05 for t in ALLT)) for m in Mrows}
    for gname, tks in GR.items():
        OUT["pin11"][gname] = {w: {m: st(series(tks, ("pin11",), m, "garch_t", None if w == "all days" else ("flag" if w == "flagged" else "rest"), norm=True))
                                   for m in ("body", "engine", "body_bteg", "engine_bteg", "body_frz", "bteg", "gjr_skewt")} for w in ("all days", "flagged", "other")}
    json.dump(OUT, open(os.path.join(ROOT, "results", "case_studies", "crossasset_annual_results%s.json" % ("_smoke" if SMOKE else "")), "w"), indent=1)
    for a in ("0.01", "0.025"):
        b = OUT["fz0"][a]["all 32"]; print("\n=== alpha %s, all 32 (gain > 0: ours lower)" % a)
        for m in ("engine", "body", "body_bteg", "engine_bteg"): print("  %-12s " % m + "  ".join("%s %+.4f (%+.1f, %d/32)" % (k, v["gain"], v["DM"], v["instruments_lower"]) for k, v in b[m].items()))
        print("  non-inferiority:", {k: (v["upper95"], v["noninferior"]) for k, v in b["engine_noninferiority"].items()})
        print("  GJR advantage (instruments):", b["gjr_advantage_instruments_as_unit"], "| MCS:", OUT["fz0"][a]["mcs90_all32"]["in_MCS_90"], "| Kupiec passes:", OUT["coverage"][a])
    for gname in GR: print("pin11 vs GARCH-t,", gname, {w: {m: (v["gain"], v["DM"]) for m, v in d.items()} for w, d in OUT["pin11"][gname].items() if w != "other"})
    lg("done")
