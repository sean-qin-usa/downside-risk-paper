# r81_canon_calendar.py -- THE CANONICAL CRSP PANEL WITH THE EARNINGS CALENDAR AS A MODEL INPUT, THE ANNOUNCEMENT TIMED.
#
# tools/r79_calendar_features.py ran this test on us_largecap only. This is the same test on the paper's main panel,
# under that panel's own protocol (code/paper/job_bench_all_bteg.py, tools/r70_neural_quantile.py): per name, the
# first 60% of days is the estimation window (GARCH-t, Beta-t-EGARCH and the masked GARCH-t are fitted there), the
# first 45% trains the pooled trees and the neural network, the next 15% validates the network, the last 40% is the
# test. Everything else is r79's: the reaction day from the announcement time, the three calendar inputs (r0, r1, ds),
# the masked GARCH-t with multipliers c0, c1 chosen on pre-test rows, and the same comparisons.
#
# The calendar. crsp_panel_chars.csv carries an old ticker per permno (LOWE, NIKE, AEXP, CHL ...), so r71-r78 looked
# up some calendars under a ticker that no longer belongs to the company, or found none. Here the calendar is the
# I/B/E/S record of quarterly earnings announcements (date and time, Eastern), linked through the CRSP header CUSIP
# and the I/B/E/S ticker: data/panels/earnings_ibes/<permno>.csv, licensed, not committed. Only the before-or-after
# 16:00 class of each announcement is kept (written as 08:00 or 16:00). 195 of the 200 names have a record; a name is
# used if it has at least 36 announcements between 2014 and 2024 (186 names).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: r76 and r78 on this panel with the untimed
# two-day window and unverified tickers (mean squared GARCH-t residual 6.1 and 8.5 on the two days; GARCH-t 2.5% breach
# 14.4% in the window; masked model 3 to 6% better than every flexible model without a calendar; with multipliers on
# every model, trees on the robust scale 1.3% better than the masked model, DM 3.0) and r79 on us_largecap (reaction
# day: mean squared residual 10.5, breach 23%; masked calendar GARCH-t +6.5% over GARCH-t; best model with calendar
# inputs +7.7%; calendar inputs worth 5.6 to 5.8% to each flexible model; masked model 0.4 to 1.2% behind them).
# The I/B/E/S calendar was not known when these were written.
#  J1. On the reaction day the mean squared GARCH-t residual is above 11 and GARCH-t's 2.5% VaR is breached on more
#      than 20% of days; on the day after the mean squared residual is below 2.5.
#  J2. The calendar inputs give each flexible model at least 4% over its own retrain without them over all days at
#      2.5% (DM above 5); on other days the difference is within 0.5% either way.
#  J3. The masked calendar GARCH-t beats GARCH-t by at least 6% over all days at 2.5%, and beats every flexible model
#      without a calendar (stored forecasts) by at least 3%.
#  J4. With the calendar in both, the trees on the Beta-t-EGARCH scale and the neural network beat the masked calendar
#      GARCH-t over all days at 2.5% by between 0.5% and 2.5%; the DM is above 2 for the trees on the robust scale.
#  J5. The masked calendar GARCH-t's gain over GARCH-t is at least three quarters of the gain of the best model with
#      calendar inputs (2.5%, all days).
#  J6. At 1% the masked calendar GARCH-t is within 1.5% of each flexible model with calendar inputs over all days.
# Usage: GBC_PROJ=<crsp dir> python tools/r81_canon_calendar.py <bench_all_bteg_rows.npz> <r70_preds.npz>
#        (needs data/panels/earnings_ibes/<permno>.csv)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "code", "panels")); sys.path.insert(0, os.path.join(ROOT, "tools"))
ARGV = sys.argv; sys.argv = [ARGV[0], "us_largecap"]   # panel_study and r78 read a panel name at import; only their functions are used here
import panel_study as P
import r78_calendar_garch as R78
sys.argv = ARGV; GP = os.environ["GBC_PROJ"]
C, TAUS, ZX = P.C, P.TAUS, P.ZX; WHAT = "canon"; AL = (0.01, 0.025); JT = [TAUS.index(a) for a in AL]; CG = np.round(np.arange(1.0, 6.001, 0.1), 2); NW = int(os.environ.get("R79_WORKERS", "15"))
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r81"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))

def work(job):
    tk, y, dates = job   # permno, returns in percent, dates
    try:
        n = len(y); sp = int(n * 0.6); cp = int(sp * 0.75); ix = np.arange(n); tr = ix < cp; ca = (ix >= cp) & (ix < sp); te = ix >= sp; ne = sp
        fn = os.path.join(ROOT, "data", "panels", "earnings_ibes", "%d.csv" % tk)
        if not os.path.exists(fn): return tk, "no calendar"
        a = pd.read_csv(fn)["announcement"].astype(str); a = a[a.str.len() >= 16]
        if ((a.str[:10] >= "2014-01-01") & (a.str[:10] <= "2024-12-31")).sum() < 36: return tk, "fewer than 36 announcements in 2014-2024"
        d0 = pd.DatetimeIndex(a.str[:10]); hr = a.str[11:13].astype(int).values + a.str[14:16].astype(int).values / 60.0
        p = np.searchsorted(dates.values, d0.values); inr = p < n; same = np.zeros(len(p), bool); same[inr] = dates.values[p[inr]] == d0.values[inr]; p = p + ((hr >= 16) & same); p = p[p < n]
        r0 = np.zeros(n); r0[p] = 1; r1 = np.zeros(n); r1[p[p + 1 < n] + 1] = 1; last = np.maximum.accumulate(np.where(r0 == 1, np.arange(n), -1)); lastp = np.concatenate([[-1], last[:-1]]); ds = np.minimum(np.where(lastp >= 0, np.arange(n) - lastp, 63), 63).astype(float)
        known = dates >= d0.min(); r0[~known] = np.nan; r1[~known] = np.nan; ds[~known] = np.nan; cal = np.column_stack([r0, r1, ds])
        pr = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params; mu, nu = float(pr["mu"]), float(pr["nu"])
        s2 = np.empty(n); s2[0] = np.var(y[:ne]); e_ = y - mu   # the canonical panel's own filter (job_bench_all_bteg.py)
        for k in range(1, n): s2[k] = max(float(pr["omega"]) + float(pr["alpha[1]"]) * e_[k - 1] ** 2 + float(pr["beta[1]"]) * s2[k - 1], 1e-8)
        sig = np.sqrt(s2)
        bx = C.fit_bteg(y[:ne]); bmu, bom, phi, kap, bnu = C.bteg_unpack(bx); bsd = np.exp(C._bteg_filter(y, bmu, bom, phi, kap, bnu)) * math.sqrt(bnu / (bnu - 2))
        _, fm = R78.fit_one((tk, y, r0, r1, ne, [mu, 0.0, float(pr["alpha[1]"]), float(pr["beta[1]"]), nu]))
        if fm is None: return tk, "fewer than 750 pre-test days with a calendar"
        k0 = fm["k0"]; th = fm["masked"]["th"]; msig = np.full(n, np.nan); msig[k0:] = np.sqrt(fm["masked"]["s2"]); mmu, mnu = float(th[0]), float(th[4]); o = {"dates": np.array([str(d.date()) for d in dates[te]]), "y": y[te], "cal": cal[te]}
        for lab, sg, m_ in (("g", sig, mu), ("b", bsd, bmu), ("m", msig, mmu)):
            df = C.feats(y, np.where(np.isfinite(sg), sg, 1.0), m_); f = df[ZX].values; f[~np.isfinite(sg)] = np.nan; ok = np.isfinite(f).all(1) & known & np.isfinite(sg)
            o[lab] = dict(Xtr=f[tr & ok], Ctr=cal[tr & ok], ztr=df["z"].values[tr & ok], Xte=f[te], sig=sg[te], mu=m_)
        scn = float(np.std(y[tr])); F = P.nfeat(y / scn); okn = np.isfinite(F).all(1) & known
        o["n"] = dict(Ftr=F[tr & okn], Ctr=cal[tr & okn], ytr=y[tr & okn] / scn, Fva=F[ca & okn], Cva=cal[ca & okn], yva=y[ca & okn] / scn, Fte=np.nan_to_num(F[te]), scn=scn)
        i = np.arange(k0 + 126, ne); o["pre"] = pd.DataFrame({"y": y[i], "sig": msig[i], "mu": mmu, "nu": mnu, "r0": r0[i], "r1": r1[i], "zg2": ((y[i] - mu) / sig[i]) ** 2})
        o["par"] = dict(nu=nu, mnu=mnu, alpha=float(pr["alpha[1]"]), malpha=float(th[2]), gsig=sig[te], gmu=mu, gnu=nu, mnu_=mnu)
        return tk, o
    except Exception as e:
        return tk, "failed: %s" % str(e)[:100]

POOLS = {}
def fit_job(a):
    key, t = a; X, z = POOLS[key]
    return key, t, HistGradientBoostingRegressor(loss="quantile", quantile=t, random_state=0, **P.HGB).fit(X, z)

if __name__ == "__main__":
    Rw = np.load(sys.argv[1]); PNS = [int(p) for p in pd.unique(Rw["permno"])]
    rr = pd.read_csv(os.path.join(GP, "crsp_panel_returns.csv"), dtype={"permno": "int32"}); rr["date"] = pd.to_datetime(rr["date"]); rr["ret"] = pd.to_numeric(rr["ret"], errors="coerce") * 100.0; jobs = []
    for pn in PNS:
        g = rr[rr.permno == pn].sort_values("date"); jobs.append((pn, g["ret"].values.astype(float), pd.DatetimeIndex(g["date"].values)))
    jobs = jobs[:int(os.environ.get("R81_MAXN", "100000"))]   # R81_MAXN and R81_TAG: a smoke test on a few names, written to another file
    with Pool(NW) as pool: res = pool.map(work, jobs, chunksize=1)
    Rr = {k: v for k, v in res if isinstance(v, dict)}; names = list(Rr); why = pd.Series([v for k, v in res if not isinstance(v, dict)]).value_counts().to_dict()
    lg("%s: %d names with a calendar and a masked fit (of %d in the panel; left out: %s)" % (WHAT, len(names), len(PNS), why))
    cat = lambda f: np.concatenate([f(Rr[t]) for t in names]); rep = lambda f: np.concatenate([np.full(len(Rr[t]["y"]), f(Rr[t])) for t in names])
    for lab in ("g", "b", "m"):
        X = cat(lambda o: o[lab]["Xtr"]); Cc = cat(lambda o: o[lab]["Ctr"]); z = cat(lambda o: o[lab]["ztr"])
        if lab != "m": POOLS[lab] = (X, z)
        POOLS[lab + "_cal"] = (np.hstack([X, Cc]), z)
    with Pool(NW) as pool: fitted = pool.map(fit_job, [(k, t) for k in POOLS for t in TAUS], chunksize=1)
    MOD = {(k, t): m for k, t, m in fitted}; lg("pooled trees fitted: %s on %d rows" % (list(POOLS), len(POOLS["g"][1])))
    Y = cat(lambda o: o["y"]); DATE = cat(lambda o: o["dates"]); NAME = np.concatenate([np.full(len(Rr[t]["y"]), t) for t in names]); CAL = cat(lambda o: o["cal"]); n = len(Y); Q = {}
    for lab, nm in (("g", "trees"), ("b", "trees on robust scale"), ("m", "trees on masked scale")):
        Xte = cat(lambda o: o[lab]["Xte"]); S = cat(lambda o: o[lab]["sig"]); M_ = rep(lambda o: o[lab]["mu"])
        for key, suf, XX in ((lab, ", retrained", Xte), (lab + "_cal", ", calendar inputs", np.hstack([Xte, CAL]))):
            if key not in POOLS: continue
            ok = np.isfinite(XX).all(1) & np.isfinite(S); zq = np.full((n, len(TAUS)), np.nan)
            for j, t in enumerate(TAUS): zq[ok, j] = MOD[(key, t)].predict(XX[ok])
            Q[nm + suf] = (M_[:, None] + S[:, None] * np.sort(zq, 1))[:, JT]
    ntr = lambda c: (np.hstack([cat(lambda o: o["n"]["Ftr"]), cat(lambda o: o["n"]["Ctr"])]) if c else cat(lambda o: o["n"]["Ftr"]), cat(lambda o: o["n"]["ytr"]))
    nva = lambda c: (np.hstack([cat(lambda o: o["n"]["Fva"]), cat(lambda o: o["n"]["Cva"])]) if c else cat(lambda o: o["n"]["Fva"]), cat(lambda o: o["n"]["yva"]))
    Fte = cat(lambda o: o["n"]["Fte"]); scn = rep(lambda o: o["n"]["scn"])
    Q["neural network, retrained"] = (P.neural(ntr(0), nva(0), Fte) * scn[:, None])[:, JT]; Q["neural network, calendar inputs"] = (P.neural(ntr(1), nva(1), np.hstack([Fte, np.nan_to_num(CAL)])) * scn[:, None])[:, JT]; lg("neural networks fitted")
    pre = pd.concat([Rr[t]["pre"] for t in names], ignore_index=True)
    cfit = lambda df: float(CG[int(np.argmin([pin(df.y.values, df.mu.values + c * df.sig.values * tq(0.025, df.nu.values), 0.025).sum() for c in CG]))])
    c0 = cfit(pre[pre.r0 == 1]); c1 = cfit(pre[(pre.r1 == 1) & (pre.r0 != 1)]); R0 = CAL[:, 0] == 1; R1 = (CAL[:, 1] == 1) & ~R0; mult = np.where(R0, c0, np.where(R1, c1, 1.0))
    SM = cat(lambda o: o["m"]["sig"]); MM = rep(lambda o: o["m"]["mu"]); NM = rep(lambda o: o["par"]["mnu"]); GS = cat(lambda o: o["par"]["gsig"]); GM = rep(lambda o: o["par"]["gmu"]); GN = rep(lambda o: o["par"]["gnu"])
    Q["masked calendar GARCH-t"] = np.column_stack([MM + SM * mult * tq(a, NM) for a in AL]); Q["GARCH-t"] = np.column_stack([GM + GS * tq(a, GN) for a in AL])
    # stored forecasts of the panel run, matched by name and date
    sidx = pd.Series(np.arange(len(Rw["y"])), index=pd.MultiIndex.from_arrays([Rw["permno"].astype(int), Rw["date"]]))
    pos = sidx.reindex(pd.MultiIndex.from_arrays([NAME.astype(int), DATE])).values; has = np.isfinite(pos); pi = np.where(has, pos, 0).astype(int); rk = np.where(has, Rw["rk_mk"][pi], -1)
    for m, nm in (("param_bteg", "Beta-t-EGARCH"), ("body", "trees, stored"), ("body_bteg", "trees on robust scale, stored")):
        q = Rw["q__" + m].astype(float)[:, JT][pi]; q[~has] = np.nan; Q[nm] = q
    Nn = np.load(sys.argv[2]); kk = pd.Series(np.arange(len(Nn["permno"])), index=pd.MultiIndex.from_arrays([Nn["permno"].astype(int), Nn["date"]])); pp = kk.reindex(pd.MultiIndex.from_arrays([NAME.astype(int), DATE])).values
    q = Nn["q"].astype(float)[:, JT][np.where(np.isfinite(pp), pp, 0).astype(int)]; q[~np.isfinite(pp)] = np.nan; Q["neural network, stored"] = q
    gs = Rw["q__garch_t"].astype(float)[:, JT[1]][pi]; lg("GARCH-t refit against the stored forecast: 99.9th percentile of the relative gap %.2e" % float(np.nanquantile(np.abs(Q["GARCH-t"][:, 1] - gs)[has] / np.abs(gs[has]), 0.999)))
    U = has & np.isfinite(CAL).all(1) & np.all([np.isfinite(q).all(1) for q in Q.values()], axis=0); dates = pd.to_datetime(DATE); TOP = rk == 10; RW = R0 | R1
    if os.environ.get("CAL_DUMP"):   # per-row forecasts for tools/r82_two_fix_garch.py (not committed); nothing is scored in this mode
        np.savez_compressed(os.path.join(ROOT, "data", "panels", WHAT + "_calrows.npz"), name=NAME.astype(str), date=DATE.astype(str), y=Y, cal=CAL, rk=rk, U=U, **{"q__" + k: v for k, v in Q.items()}); lg("per-row forecasts written"); sys.exit(0)
    L = {m: {a: pin(Y, q[:, j], a) for j, a in enumerate(AL)} for m, q in Q.items()}
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
        if len(s) < 30: return None
        dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    edge = lambda Lr, Lm, mask: None if mask.sum() < 50 else {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
    CELLS = (("all days", U), ("reaction day", U & R0), ("day after", U & R1), ("other days", U & ~RW), ("top decile", U & TOP), ("top decile, other days", U & TOP & ~RW), ("deciles 1-8, other days", U & (rk <= 8) & (rk > 0) & ~RW))
    zg2 = ((Y - GM) / GS) ** 2; PAIRS = (("trees, calendar inputs", "trees, retrained"), ("trees on robust scale, calendar inputs", "trees on robust scale, retrained"), ("neural network, calendar inputs", "neural network, retrained"),
                                         ("trees, retrained", "trees, stored"), ("trees on robust scale, retrained", "trees on robust scale, stored"), ("neural network, retrained", "neural network, stored"))
    OUT = {"universe": WHAT, "names": len(names), "names_in_panel": len(PNS), "left_out": why, "rows": int(U.sum()), "training_rows_trees": int(len(POOLS["g"][1])), "predictions_written_before_run": "see the script header (J1-J6)", "c0_reaction_day": c0, "c1_day_after": c1,
           "share_of_days": {"reaction day": round(float(R0[U].mean()), 4), "day after": round(float(R1[U].mean()), 4)},
           "mean_squared_garch_residual": {"reaction day": round(float(zg2[U & R0].mean()), 2), "day after": round(float(zg2[U & R1].mean()), 2), "other days": round(float(zg2[U & ~RW].mean()), 2),
                                           "pre-test reaction day": round(float(pre.zg2[pre.r0 == 1].mean()), 2), "pre-test day after": round(float(pre.zg2[(pre.r1 == 1) & (pre.r0 != 1)].mean()), 2)},
           "median_nu": {"GARCH-t": round(float(np.median([Rr[t]["par"]["nu"] for t in names])), 2), "masked": round(float(np.median([Rr[t]["par"]["mnu"] for t in names])), 2)}}
    for j, a in enumerate(AL):
        mean_loss = {m: float(L[m][a][U].mean()) for m in Q}
        OUT[str(a)] = {"vs_garch_t": {m: {lab: edge(L["GARCH-t"][a], L[m][a], c) for lab, c in CELLS} for m in Q if m != "GARCH-t"},
                       "masked_calendar_garch_t_vs": {m: {lab: edge(L[m][a], L["masked calendar GARCH-t"][a], c) for lab, c in CELLS} for m in Q if m != "masked calendar GARCH-t"},
                       "pairs_first_vs_second": {"%s | %s" % p_: {lab: edge(L[p_[1]][a], L[p_[0]][a], c) for lab, c in CELLS} for p_ in PAIRS},
                       "breach_pct": {m: {lab: round(100 * float((Y <= Q[m][:, j])[c].mean()), 2) for lab, c in CELLS} for m in Q}, "lowest_loss_all_days": min(mean_loss, key=mean_loss.get)}
    json.dump(OUT, open(os.path.join(OUTD, os.environ.get("R81_TAG", WHAT) + ".json"), "w"), indent=1)
    f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"])
    print("\n######## %s: %d names, %d rows; c0 %.1f, c1 %.1f; shares %s; mean squared GARCH-t residual %s; median nu %s" % (WHAT, len(names), U.sum(), c0, c1, OUT["share_of_days"], OUT["mean_squared_garch_residual"], OUT["median_nu"]))
    for a in ("0.025", "0.01"):
        print("--- %s: each model against GARCH-t (lowest loss over all days: %s)" % (a, OUT[a]["lowest_loss_all_days"]))
        for m, v in OUT[a]["vs_garch_t"].items(): print("   %-40s %s" % (m, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s: the masked calendar GARCH-t against each model (positive: masked model better)" % a)
        for m, v in OUT[a]["masked_calendar_garch_t_vs"].items(): print("   %-40s %s" % (m, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s: pairs (positive: the first is better)" % a)
        for m, v in OUT[a]["pairs_first_vs_second"].items(): print("   %-75s %s" % (m, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s: breach rates" % a)
        for m, v in OUT[a]["breach_pct"].items(): print("   %-40s %s" % (m, v))
    lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
