# multiasset_case.py -- CASE STUDY (not yet in the paper): which configuration should the paper recommend?
# 17 instruments across volatility levels and asset classes (yfinance, free): stocks KO JNJ XOM JPM AAPL AMD TSLA NVDA;
# ETFs SPY EEM TLT GLD USO UNG; FX EURUSD=X JPY=X; crypto BTC-USD. Test 2020-01-02 .. 2026-11-13, every per-instrument
# filter refit each January 1 on all prior data of that instrument. FX: a price that jumps more than 3% and reverses by
# more than 3% the next day is a known yfinance bad tick (all 2008, in the training window) and is removed; each removal
# is logged in the output.
# Our model's pooled parts (the trees and the GPD tail) are trained ONCE on the paper's 200-name CRSP panel using pre-2020
# rows only, with every test ticker that is in the panel excluded, then frozen. Construction as nvda_engine_case.py.
#
# Rows per instrument:
#   garch_t, gjr_skewt, bteg (Beta-t-EGARCH, Student-t tail), fhs               benchmarks
#   engine        the paper's accuracy layer (GARCH-t scale, pooled trees, pooled GPD tail)    = the paper's model
#   body_bteg     pooled trees on the Beta-t-EGARCH scale, no GPD tail                         = the candidate
#   engine_bteg   pooled trees and GPD tail on the Beta-t-EGARCH scale
#   gate          GARCH-t, switching to Beta-t-EGARCH when mk63 > its pre-year 90th percentile
# Scores: FZ0 at 1% and 2.5%, breach rates, Kupiec. FZ0 is zero-homogeneous, so loss DIFFERENCES between models do not
# depend on an instrument's volatility level and can be pooled across instruments: the pooled DM averages each date's
# differences over the instruments trading that date, Newey-West(10).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-16):
#  P1. The paper's accuracy layer beats GARCH-t on FZ0 on fewer than half the instruments at 1%, and its pooled DM against
#      GARCH-t is within noise at 1% (the pooled GPD tail does not transfer off the equity panel).
#  P2. body_bteg has a lower FZ0 than GARCH-t on at least 12 of 17 instruments at both levels, pooled DM > 1.96 at both.
#  P3. body_bteg beats the accuracy layer with pooled DM > 1.96 at both levels.
#  P4. body_bteg's FZ0 gain over GARCH-t is larger on the eight stocks, UNG and BTC than on FX, TLT and GLD.
#  P5. GJR-skew-t's pooled gain over GARCH-t is positive, and larger on stocks and equity ETFs than on FX and GLD.
#
# DECISION RULE, FIXED BEFORE THE RUN. The paper's recommended configuration changes from the accuracy layer to body_bteg
# only if ALL of the following hold at BOTH levels:
#  (a) body_bteg's pooled FZ0 DM against the accuracy layer exceeds 1.96;
#  (b) body_bteg has the lower mean FZ0 on at least 12 of the 17 instruments;
#  (c) body_bteg passes Kupiec at 5% on at least 12 of the 17 instruments.
# Otherwise the paper keeps the accuracy layer as its estimator and reports body_bteg as a result.
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "unused", sys.argv[2], sys.argv[1]]     # nvda_engine_case reads argv[1:3] at import
import nvda_engine_case as C
from es_integral import converged_es
DATA_DIR, PANEL_DIR = sys.argv[3], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__))
A, TAUS, ZX, P0, SUBN, KEXTRA, HGB = C.A, C.TAUS, C.ZX, C.P0, C.SUBN, C.KEXTRA, C.HGB
TICKERS = ["KO", "JNJ", "XOM", "JPM", "AAPL", "AMD", "TSLA", "NVDA", "SPY", "EEM", "TLT", "GLD", "USO", "UNG", "EURUSD=X", "JPY=X", "BTC-USD"]
CLASS = {**{t: "stock" for t in TICKERS[:8]}, "SPY": "equity ETF", "EEM": "equity ETF", "TLT": "bonds", "GLD": "gold", "USO": "oil",
         "UNG": "gas", "EURUSD=X": "FX", "JPY=X": "FX", "BTC-USD": "crypto"}
YEARS = list(range(2020, 2027)); CUT = pd.Timestamp("2020-01-01"); t0 = time.time()
lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
def lev_for(a):
    floor = a / SUBN / 2.0
    return np.unique(np.concatenate([a * ((np.arange(SUBN) + 0.5) / SUBN), np.exp(np.linspace(math.log(floor), math.log(a), KEXTRA)), [a]]))
LEVS = sorted(set(TAUS) | set(float(u) for a in A for u in lev_for(a)))
TRN = {}
def fit_level(args):
    f, t = args
    return f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(TRN[f][:, :-1], TRN[f][:, -1])

def instrument(tk, models, tails):
    s = pd.read_csv(os.path.join(DATA_DIR, tk.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]
    s = s[(s > 0) & (s.index <= "2026-11-13")]; removed = []
    if tk.endswith("=X"):
        r_ = s.pct_change()
        bad = (r_.abs() > 0.03) & (r_.shift(-1).abs() > 0.03) & (np.sign(r_) != np.sign(r_.shift(-1)))
        removed = [str(d.date()) for d in s.index[bad.fillna(False).values]]; s = s[~bad.fillna(False).values]
    r = (s.pct_change() * 100).dropna(); dates = r.index; y = r.values.astype(float); n = len(y)
    SIG = {f: np.full(n, np.nan) for f in ("garch_t", "bteg")}; MU = {f: np.full(n, np.nan) for f in SIG}
    FEAT = {f: np.full((n, len(ZX)), np.nan) for f in SIG}; YR = np.zeros(n, int); HI = np.zeros(n, bool)
    V = {m: {a: np.full(n, np.nan) for a in A} for m in ("garch_t", "gjr_skewt", "bteg", "fhs")}; E = {m: {a: np.full(n, np.nan) for a in A} for m in V}; bx = None
    for yr in YEARS:
        tr = dates < pd.Timestamp("%d-01-01" % yr); te = (dates >= pd.Timestamp("%d-01-01" % yr)) & (dates < pd.Timestamp("%d-01-01" % (yr + 1)))
        if te.sum() == 0: continue
        ntr = int(tr.sum()); last = int(np.where(te)[0][-1]) + 1; ytr = y[:ntr]; tl = te[:last]; YR[te] = yr
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
                df = C.feats(y[:last], sig, mu); FEAT["garch_t"][te] = df[ZX].values[tl]; ztr = df['z'].values[:ntr]
                for a in A:
                    qz = float(np.quantile(ztr, a)); V["fhs"][a][te] = mu + sig[tl] * qz; E["fhs"][a][te] = mu + sig[tl] * float(ztr[ztr <= qz].mean())
                mk = df['mk63'].values; HI[te] = mk[tl] > float(np.nanquantile(mk[:ntr], 0.9))
        bx = C.fit_bteg(ytr, bx); mu, om, phi, kap, nu = C.bteg_unpack(bx); lam = C._bteg_filter(y[:last], mu, om, phi, kap, nu)
        sd = np.exp(lam) * math.sqrt(nu / (nu - 2)); SIG["bteg"][te] = sd[tl]; MU["bteg"][te] = mu
        FEAT["bteg"][te] = C.feats(y[:last], sd, mu)[ZX].values[tl]
        for a in A:
            zq, ze = C.t_std_q_es(a, nu); V["bteg"][a][te] = mu + sd[tl] * zq; E["bteg"][a][te] = mu + sd[tl] * ze
    T = YR > 0
    for f, lab in (("garch_t", ""), ("bteg", "_bteg")):
        X = FEAT[f][T]; ok = np.isfinite(X).all(1); tail = tails[f]
        Qc = {}
        for t in LEVS:
            q = np.full(len(X), np.nan); q[ok] = models[(f, round(t, 12))].predict(X[ok]); Qc[round(t, 12)] = q
        Q = lambda t: Qc[round(float(t), 12)]
        eng = {t: (np.minimum(Q(t), tail.q(t)) if t <= P0 else Q(t)) for t in TAUS}
        Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Q(t) for t in TAUS], 1), 1)
        mu = MU[f][T]; sg = SIG[f][T]
        for m in ("engine" + lab, "body" + lab): V[m] = {a: np.full(n, np.nan) for a in A}; E[m] = {a: np.full(n, np.nan) for a in A}
        for a in A:
            lev = lev_for(a); QB = np.stack([Q(u) for u in lev], 1); zq = Es[:, TAUS.index(a)]
            es = np.full(len(zq), np.nan)
            es[ok] = converged_es(a, lev, QB[ok], tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq[ok])
            zqb = bod[:, TAUS.index(a)]; esb = np.minimum(np.stack([Q(a * (j + 0.5) / SUBN) for j in range(SUBN)], 1).mean(1), zqb - 1e-6)
            V["engine" + lab][a][T] = mu + sg * zq; E["engine" + lab][a][T] = mu + sg * es
            V["body" + lab][a][T] = mu + sg * zqb; E["body" + lab][a][T] = mu + sg * esb
    V["gate"] = {a: np.where(HI, V["bteg"][a], V["garch_t"][a]) for a in A}; E["gate"] = {a: np.where(HI, E["bteg"][a], E["garch_t"][a]) for a in A}
    keep = T & np.all([np.isfinite(V[m][a]) & np.isfinite(E[m][a]) for m in V for a in A], axis=0)
    return dict(dates=dates[keep], y=y[keep], hi=HI[keep], V={m: {a: V[m][a][keep] for a in A} for m in V},
                E={m: {a: E[m][a][keep] for a in A} for m in E}, removed=removed, n_dropped=int(T.sum() - keep.sum()))

if __name__ == "__main__":
    rr = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
    rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    ch = pd.read_csv(os.path.join(PANEL_DIR, "crsp_panel_chars.csv")); excl = set(ch[ch.ticker.isin(TICKERS)].permno)
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False)
    names = [p for p in cnt[cnt >= 1500].index.tolist()[:200] if p not in excl]
    pre = rr[rr.date < CUT]; jobs = []
    for pn in names:
        yy = pre[pre.permno == pn].sort_values('date')['ret'].values.astype(float)
        if len(yy) >= 750 and np.isfinite(yy).all(): jobs.append((pn, yy))
    with Pool(15) as pool: res = pool.map(C.panel_name, jobs)
    for f in ("garch_t", "bteg"): TRN[f] = np.concatenate([o[f] for _, o in res if o is not None])
    lg("training pool: %d names, %d rows (test tickers in the panel excluded: %d)" % (sum(o is not None for _, o in res), len(TRN["garch_t"]), len(excl & set(cnt.index[:200]))))
    with Pool(15) as pool: fitted = pool.map(fit_level, [(f, t) for f in ("garch_t", "bteg") for t in LEVS], chunksize=4)
    models = {(f, round(t, 12)): m for f, t, m in fitted}; tails = {f: C.GPDTail(TRN[f][:, -1], P0) for f in TRN}
    lg("pooled trees fitted (%d models); GPD xi %.3f / %.3f" % (len(models), tails["garch_t"].xi, tails["bteg"].xi))

    ROWS = ["garch_t", "gjr_skewt", "bteg", "fhs", "engine", "body", "engine_bteg", "body_bteg", "gate"]
    per = {}; diffs = {a: {} for a in A}; out = {"per_instrument": {}, "pooled": {}, "decision": {}}
    for tk in TICKERS:
        R = instrument(tk, models, tails); per[tk] = R
        blk = {"class": CLASS[tk], "n_test": int(len(R["y"])), "n_high_score": int(R["hi"].sum()), "fx_ticks_removed": R["removed"], "levels": {}}
        for a in A:
            L = {m: C.fz0(R["y"], R["V"][m][a], R["E"][m][a], a) for m in ROWS}
            row = {}
            for m in ROWS:
                d = L["garch_t"] - L[m]; de = L["engine"] - L[m]
                row[m] = {"fz0": round(float(L[m].mean()), 4), "breach_pct": round(100 * float((R["y"] <= R["V"][m][a]).mean()), 2),
                          "kupiec_p": round(C.kupiec(R["y"] <= R["V"][m][a], a), 3),
                          "gain_vs_garch_t": round(float(d.mean()), 4), "DM_vs_garch_t": round(C.nw_t(d), 2) if m != "garch_t" else None,
                          "gain_vs_engine": round(float(de.mean()), 4), "DM_vs_engine": round(C.nw_t(de), 2) if m != "engine" else None}
                for ref in ("garch_t", "engine"):
                    diffs[a].setdefault((ref, m), []).append(pd.Series(L[ref] - L[m], index=R["dates"]))
            blk["levels"][str(a)] = row
        out["per_instrument"][tk] = blk
        lg("%-9s %-10s n=%d  FZ0 1%%: garch_t %.3f engine %.3f body_bteg %.3f bteg %.3f | 2.5%%: %.3f %.3f %.3f %.3f" % (
            tk, CLASS[tk], len(R["y"]), *[blk["levels"]["0.01"][m]["fz0"] for m in ("garch_t", "engine", "body_bteg", "bteg")],
            *[blk["levels"]["0.025"][m]["fz0"] for m in ("garch_t", "engine", "body_bteg", "bteg")]))
    for a in A:
        P = {}
        for (ref, m), lst in diffs[a].items():
            if m == ref: continue
            daily = pd.concat(lst, axis=1).mean(axis=1).dropna()     # each date: mean difference over instruments trading
            wins = sum(out["per_instrument"][tk]["levels"][str(a)][m]["gain_vs_" + ref] > 0 for tk in TICKERS)
            P["%s_vs_%s" % (m, ref)] = {"mean_daily_gain": round(float(daily.mean()), 4), "DM": round(C.nw_t(daily.values), 2),
                                        "instruments_won": int(wins), "n_dates": int(len(daily))}
        P["kupiec_pass_count"] = {m: int(sum(out["per_instrument"][tk]["levels"][str(a)][m]["kupiec_p"] >= 0.05 for tk in TICKERS)) for m in ROWS}
        out["pooled"][str(a)] = P
    dec = {}
    for a in A:
        p = out["pooled"][str(a)]; c = p["body_bteg_vs_engine"]
        dec[str(a)] = {"a_pooled_DM_gt_1.96": c["DM"] > 1.96, "b_won_ge_12": c["instruments_won"] >= 12, "c_kupiec_ge_12": p["kupiec_pass_count"]["body_bteg"] >= 12}
    dec["CHANGE_RECOMMENDED_CONFIGURATION"] = all(all(v.values()) for v in dec.values())
    out["decision"] = dec
    json.dump(out, open(os.path.join(HERE, "..", "..", "results", "case_studies", "multiasset_case_results.json"), "w"), indent=1, default=str)
    for a in A:
        print("\n=== alpha %.3f  pooled over 17 instruments (gain > 0: row has the lower FZ0)" % a)
        p = out["pooled"][str(a)]
        for m in ROWS:
            g = p.get(m + "_vs_garch_t"); e = p.get(m + "_vs_engine")
            print("  %-12s vs GARCH-t %s   vs accuracy layer %s   Kupiec pass %2d/17" % (m,
                  ("%+.4f DM %+5.2f won %2d/17" % (g["mean_daily_gain"], g["DM"], g["instruments_won"])) if g else "%27s" % "--",
                  ("%+.4f DM %+5.2f won %2d/17" % (e["mean_daily_gain"], e["DM"], e["instruments_won"])) if e else "--", p["kupiec_pass_count"][m]))
    print("\nDECISION:", json.dumps(dec))
    lg("done")
