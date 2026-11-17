# nvda_engine_variants.py -- CASE STUDY (not in the paper): does putting NVIDIA in the training sample help our model?
# Same NVDA test (2020-01-02 .. 2026-11-13), same NVDA filters and benchmarks as nvda_engine_case.py (helpers imported).
# Our model is rebuilt under four training designs; every forecast uses only data dated before its test year:
#   excl_frozen   200 panel names, NVDA excluded, pre-2020 rows, trained once           (= nvda_engine_case.py)
#   incl_frozen   the same pool plus NVDA's 2014-2019 rows, trained once
#   incl_annual   panel names plus NVDA, retrained every January 1 on all rows before it (panel to 2024; NVDA from yfinance)
#   nvda_only     NVDA's own rows only (from 2005), retrained every January 1: a per-name learner, no pooling
# For each: body and engine (accuracy layer) on the GARCH-t scale, and on the Beta-t-EGARCH scale.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-16):
#  P1. incl_frozen moves the engine's mean FZ0 by less than 0.01 at both levels relative to excl_frozen (NVDA's rows
#      are about half a percent of the pool), and the engine still has a higher FZ0 than GARCH-t at 1%.
#  P2. incl_annual lowers the engine's FZ0 relative to incl_frozen, but the engine still does not beat GARCH-t
#      significantly at 1% (DM < 1.96 in its favour), because the pooled GPD tail still overrides the body.
#  P3. In every pooled design the body on the Beta-t-EGARCH scale has the lowest or tied-lowest FZ0 of our rows.
#  P4. nvda_only is worse than the pooled designs at 1% (higher FZ0 for the body), since one name has few tail points.
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvda_engine_case as C            # helpers only; its main block is guarded
from es_integral import converged_es
NVDA_CSV, PANEL_DIR = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__))
A, TAUS, ZX, P0, SUBN, KEXTRA, HGB, NV = C.A, C.TAUS, C.ZX, C.P0, C.SUBN, C.KEXTRA, C.HGB, C.NV
YEARS = list(range(2020, 2027)); t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)

def lev_for(a):
    floor = a / SUBN / 2.0
    return np.unique(np.concatenate([a * ((np.arange(SUBN) + 0.5) / SUBN), np.exp(np.linspace(math.log(floor), math.log(a), KEXTRA)), [a]]))
LEVS = sorted(set(TAUS) | set(float(u) for a in A for u in lev_for(a)))

def name_rows(args):
    """GARCH-t and Beta-t-EGARCH fitted on one name's rows (all before a cut); returns pooled-training arrays."""
    key, y = args
    try: return key, C.panel_name((key, y))[1]
    except Exception: return key, None

def fit_level(args):
    key, f, t = args
    D = POOLS[key][f]
    return key, f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(D[:, :-1], D[:, -1])

POOLS = {}
if __name__ == "__main__":
    # ---------------------------------------------------------------- data
    rr = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
    rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); names = [p for p in cnt[cnt >= 1500].index.tolist()[:200] if p != NV]
    px = pd.read_csv(NVDA_CSV, index_col=0, parse_dates=True)["Close"]; px = px[px.index <= "2026-11-13"]
    r = (px.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
    G = {pn: g.sort_values('date') for pn, g in rr[rr.permno.isin(names)].groupby('permno')}

    # ---------------------------------------------------------------- training rows per (cut, name)
    cuts = sorted(set(YEARS)); jobs = []
    for yr in cuts:
        cut = pd.Timestamp("%d-01-01" % yr)
        for pn in names:
            yy = G[pn][G[pn].date < cut]['ret'].values.astype(float)
            if len(yy) >= 750 and np.isfinite(yy).all(): jobs.append(((yr, pn), yy))
        nv = y[(dates < cut) & (dates >= pd.Timestamp("2014-01-01"))]; jobs.append(((yr, "NVDA14"), nv))      # NVDA, panel window
        jobs.append(((yr, "NVDA05"), y[dates < cut]))                                                       # NVDA, full history
    with Pool(15) as pool: res = dict(pool.map(name_rows, jobs))
    lg("per-name training fits: %d (%d failed)" % (len(res), sum(v is None for v in res.values())))
    def pool_of(yr, incl):
        keys = [(yr, pn) for pn in names] + ([(yr, "NVDA14")] if incl else [])
        return {f: np.concatenate([res[k][f] for k in keys if res.get(k) is not None]) for f in ("garch_t", "bteg")}
    POOLS[("excl_frozen", 2020)] = pool_of(2020, False); POOLS[("incl_frozen", 2020)] = pool_of(2020, True)
    for yr in YEARS:
        POOLS[("incl_annual", yr)] = pool_of(yr, True); POOLS[("nvda_only", yr)] = res[(yr, "NVDA05")]
    lg("pools: " + ", ".join("%s/%d: %d rows" % (k[0], k[1], len(v["garch_t"])) for k, v in POOLS.items() if k[1] in (2020, 2026)))

    # ---------------------------------------------------------------- NVDA filters and benchmarks (same as nvda_engine_case.py)
    SIG = {f: np.full(n, np.nan) for f in ("garch_t", "gjr_skewt", "bteg")}; MU = {f: np.full(n, np.nan) for f in SIG}
    FEAT = {f: np.full((n, len(ZX)), np.nan) for f in ("garch_t", "bteg")}; YR = np.zeros(n, int); HI = np.zeros(n, bool)
    V = {m: {a: np.full(n, np.nan) for a in A} for m in ("garch_t", "gjr_skewt", "bteg", "fhs")}; E = {m: {a: np.full(n, np.nan) for a in A} for m in V}; bx = None
    for yr in YEARS:
        tr = dates < pd.Timestamp("%d-01-01" % yr); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1)))
        ntr = int(tr.sum()); last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; tl = te[:last]; YR[te] = yr
        for m, o, dist in (("garch_t", 0, "t"), ("gjr_skewt", 1, "skewt")):
            pr = arch_model(ytr, vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False).params
            sig = arch_model(y[:last], vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False).fix(pr.values).conditional_volatility
            mu = float(pr["mu"]); SIG[m][te] = sig[tl]; MU[m][te] = mu
            for a in A:
                if dist == "t": zq, ze = C.t_std_q_es(a, float(pr["nu"]))
                else:
                    ppf = lambda u, pr=pr: C.SK.ppf(np.clip(u, 1e-300, 1.0), np.array([float(pr["eta"]), float(pr["lambda"])]))
                    zq, ze = float(ppf(np.array([a]))[0]), C.es_from_ppf(ppf, a)
                V[m][a][te] = mu + sig[tl] * zq; E[m][a][te] = mu + sig[tl] * ze
            if m == "garch_t":
                df = C.feats(y[:last], sig, mu); FEAT["garch_t"][te] = df[ZX].values[tl]; ztr = df['z'].values[:ntr]
                for a in A:
                    qz = float(np.quantile(ztr, a)); V["fhs"][a][te] = mu + sig[tl] * qz; E["fhs"][a][te] = mu + sig[tl] * float(ztr[ztr <= qz].mean())
                mk = df['mk63'].values; HI[te] = mk[tl] > float(np.nanquantile(mk[:ntr], 0.9))
        bx = C.fit_bteg(ytr, bx); mu, om, phi, kap, nu = C.bteg_unpack(bx); lam = C._bteg_filter(y[:last], mu, om, phi, kap, nu)
        sd = np.exp(lam) * math.sqrt(nu / (nu - 2)); SIG["bteg"][te] = sd[tl]; MU["bteg"][te] = mu
        FEAT["bteg"][te] = C.feats(y[:last], sd, mu)[ZX].values[tl]
        for a in A:
            zq, ze = C.t_std_q_es(a, nu); V["bteg"][a][te] = mu + sd[tl] * zq; E["bteg"][a][te] = mu + sd[tl] * ze
    T = YR > 0; yt = y[T]; hi = HI[T]; yrs = YR[T]
    lg("NVDA filters done, %d test days" % T.sum())

    # ---------------------------------------------------------------- fit every quantile model, in parallel
    tasks = [(k, f, t) for k in POOLS for f in ("garch_t", "bteg") for t in LEVS]
    lg("fitting %d quantile models" % len(tasks))
    with Pool(15) as pool: models = {(k, f, round(t, 12)): m for k, f, t, m in pool.map(fit_level, tasks, chunksize=4)}
    lg("quantile models fitted")
    TAILS = {(k, f): C.GPDTail(POOLS[k][f][:, -1], P0) for k in POOLS for f in ("garch_t", "bteg")}

    # ---------------------------------------------------------------- assemble our rows per design
    designs = {"excl_frozen": lambda yr: ("excl_frozen", 2020), "incl_frozen": lambda yr: ("incl_frozen", 2020),
               "incl_annual": lambda yr: ("incl_annual", yr), "nvda_only": lambda yr: ("nvda_only", yr)}
    for dname, keyf in designs.items():
        for f, lab in (("garch_t", ""), ("bteg", "_bteg")):
            for m in ("body", "engine"):
                V["%s%s|%s" % (m, lab, dname)] = {a: np.full(n, np.nan) for a in A}; E["%s%s|%s" % (m, lab, dname)] = {a: np.full(n, np.nan) for a in A}
            for yr in YEARS:
                sel = (YR == yr); X = FEAT[f][sel]; k = keyf(yr); tail = TAILS[(k, f)]
                Q = lambda t: models[(k, f, round(float(t), 12))].predict(X)
                cache = {round(float(t), 12): Q(t) for t in LEVS}; Qc = lambda t: cache[round(float(t), 12)]
                eng = {t: (np.minimum(Qc(t), tail.q(t)) if t <= P0 else Qc(t)) for t in TAUS}
                Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Qc(t) for t in TAUS], 1), 1)
                mu = MU[f][sel]; sg = SIG[f][sel]
                for a in A:
                    lev = lev_for(a); QB = np.stack([Qc(u) for u in lev], 1); zq = Es[:, TAUS.index(a)]
                    es = converged_es(a, lev, QB, tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq)
                    zqb = bod[:, TAUS.index(a)]; esb = np.minimum(np.stack([Qc(a * (j + 0.5) / SUBN) for j in range(SUBN)], 1).mean(1), zqb - 1e-6)
                    V["engine%s|%s" % (lab, dname)][a][sel] = mu + sg * zq; E["engine%s|%s" % (lab, dname)][a][sel] = mu + sg * es
                    V["body%s|%s" % (lab, dname)][a][sel] = mu + sg * zqb; E["body%s|%s" % (lab, dname)][a][sel] = mu + sg * esb
    lg("rows assembled")

    # ---------------------------------------------------------------- score
    bench = ["garch_t", "gjr_skewt", "bteg", "fhs"]; ours = [m for m in V if "|" in m]
    out = {"test": [str(dates[T][0].date()), str(dates[T][-1].date())], "n_test": int(T.sum()), "n_high_score": int(hi.sum()),
           "pool_rows": {"%s/%d" % k: int(len(v["garch_t"])) for k, v in POOLS.items()}, "by_level": {}}
    out["gpd"] = {"%s/%d/%s" % (k[0], k[1], f): dict(xi=round(t.xi, 3), beta=round(t.beta, 3), u=round(-t.u, 3)) for (k, f), t in TAILS.items() if k[1] in (2020, 2026)}
    for a in A:
        L = {m: C.fz0(yt, V[m][a][T], E[m][a][T], a) for m in bench + ours}
        blk = {}
        for m in bench + ours:
            row = {"fz0": round(float(L[m].mean()), 4), "breach_pct": round(100 * float((yt <= V[m][a][T]).mean()), 2),
                   "kupiec_p": round(C.kupiec(yt <= V[m][a][T], a), 3)}
            for ref in ("garch_t", "gjr_skewt", "bteg"):
                if m == ref: continue
                d = L[ref] - L[m]; dh = d[hi]          # > 0: m has the LOWER loss
                row["vs_" + ref] = {"gain": round(float(d.mean()), 4), "DM": round(C.nw_t(d), 2), "gain_high": round(float(dh.mean()), 4), "DM_high": round(C.nw_t(dh), 2)}
            blk[m] = row
        out["by_level"][str(a)] = blk
    json.dump(out, open(os.path.join(HERE, "..", "..", "results", "case_studies", "nvda_engine_variants_results.json"), "w"), indent=1)
    for a in A:
        b = out["by_level"][str(a)]
        print("\n=== alpha %.3f   gain > 0 and DM > 0: the row has the LOWER FZ0 than the reference" % a)
        print("%-28s %7s %7s | %-17s %-17s %-17s" % ("row", "FZ0", "breach", "vs GARCH-t", "vs GJR-skew-t", "vs Beta-t-EGARCH"))
        for m in bench + ours:
            r_ = b[m]; cell = lambda ref: ("%+.4f (%+.2f)" % (r_["vs_" + ref]["gain"], r_["vs_" + ref]["DM"])) if "vs_" + ref in r_ else "--"
            print("%-28s %7.4f %6.2f%% | %-17s %-17s %-17s" % (m, r_["fz0"], r_["breach_pct"], cell("garch_t"), cell("gjr_skewt"), cell("bteg")))
    lg("done")
