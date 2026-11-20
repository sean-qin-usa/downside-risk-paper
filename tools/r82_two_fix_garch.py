# r82_two_fix_garch.py -- A GARCH-t WITH BOTH FIXES, AGAINST EVERY MODEL, EACH GIVEN THE EARNINGS CALENDAR.
#
# r76-r81 found two errors of GARCH-t on single stocks. On the day the market reacts to an earnings announcement its
# VaR is far too narrow, because the model does not know the date. In the weeks after a large shock its VaR is too
# wide, because it treats a one-off shock as lasting; firm-specific shocks fade and market-wide shocks persist. Each
# error has been corrected on its own (the masked calendar GARCH-t of r78; the closed-form correction of r71). This
# script puts both corrections in one parametric model and asks whether that model matches the flexible models once
# every model has the calendar.
#
# The two-fix GARCH-t. Start from the masked calendar GARCH-t (r78: the reaction day and the day after add nothing to
# the likelihood or to the variance recursion; multipliers c0, c1 on those two days). For every day t take the largest
# standardized residual of days t-63..t-1 among the days that feed the recursion, on day s. Its share of today's
# variance is  D = alpha * G(s,t) * s2[s] * (z[s]^2 - 1) / s2[t],  with G the product of the recursion's decay factors
# between s and t (beta on an ordinary day, alpha+beta on a masked day). The corrected variance is
#     s2[t] * (1 - (1 - lambda(w)) * D),      lambda(w) = lambda_f + (lambda_m - lambda_f) * w,
# where w = clip(b*m/r, 0, 1) is the market-wide share of the shock (r76: r the name's return on day s less its mean,
# m the equal-weighted return of the other names of the panel that day, b the name's pre-test slope on m; an unknown w
# counts as 0). lambda_f and lambda_m (the real share of a firm-specific and of a market-wide shock) are two numbers
# per panel, chosen on pre-test rows by 2.5% pinball loss; c0 and c1 are then chosen on pre-test calendar days.
# Two tails: the model's own Student-t, and three empirical quantiles per level (reaction day, day after, other days)
# of the model's pooled pre-test standardized residuals.
# For the 2-by-2 table the same shock correction is also applied to the plain GARCH-t (no calendar).
#
# Every model in the comparison, on the same rows:
#   without the calendar (stored forecasts of the paper): GARCH-t, GJR-skew-t, Beta-t-EGARCH, SAV-CAViaR, FHS, pooled
#       trees, trees on the robust scale, both with the EVT tail, the neural network (and the pooled GARCH-EVT on CRSP)
#   parametric models given the calendar as two multipliers on their VaR, chosen on pre-test rows: GARCH-t,
#       GJR-skew-t, Beta-t-EGARCH, SAV-CAViaR, FHS
#   the masked calendar GARCH-t
#   flexible models retrained with the calendar as an input (r79, r81): trees on three scales, the neural network
#   the shock correction alone, the two-fix GARCH-t, the two-fix GARCH-t with empirical quantiles
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r71-r81. In particular, at 2.5% over all days the
# masked calendar GARCH-t gains 9.2% (CRSP) and 6.5% (US large caps) over GARCH-t, the best flexible model with
# calendar inputs 10.7% and 7.7%, and the flexible models beat the masked model by 1.6-1.7% and 1.2%; that margin is
# 4-6% in the top decile of the score on ordinary days and about 1% in deciles 1 to 8; the closed form predicted a gain
# of 1.4% and 1.2% for the shock correction on the plain filter; masking already removes the earnings shocks.
#  T1. On the masked filter, lambda_f is at most 0.30 and lambda_m exceeds lambda_f by at least 0.30, on both panels.
#  T2. The two-fix GARCH-t beats the masked calendar GARCH-t over all days at 2.5% by 0.3% to 1.2% (DM above 2) on
#      both panels, and by at least 2% in the top decile on ordinary days.
#  T3. The two fixes are less than additive: the gain of both over GARCH-t is below the sum of the two separate gains
#      by 0.3 to 1.2 points (2.5%, all days).
#  T4. The two-fix GARCH-t does NOT match the best flexible models with the calendar. Over all days at 2.5% it stays
#      0.4% to 1.4% behind the trees on the robust scale and the neural network with calendar inputs, DM below -2 on
#      the CRSP panel; it closes between a quarter and two thirds of the gap the masked model leaves.
#  T5. With empirical quantiles the two-fix model is 0.2% to 1.0% better than with its Student-t (2.5%, all days) and
#      within 1.0% of the best flexible model with the calendar on both panels.
#  T6. The two-fix GARCH-t has lower 2.5% loss over all days than every parametric model given calendar multipliers
#      (GARCH-t, GJR-skew-t, Beta-t-EGARCH, SAV-CAViaR, FHS; DM above 2 each) and beats every model without the
#      calendar by at least 5% (CRSP) and 3.5% (US large caps).
#  T7. The 90% model confidence set at 2.5% over all days holds no model without the calendar, holds at least one
#      flexible model with calendar inputs, and does not hold the two-fix GARCH-t (Student-t) on the CRSP panel.
# If T4 holds, "two pieces of information are enough" is false as stated: they give most of the gain, not all of it.
#
# Usage:  GBC_PROJ=<crsp dir> python tools/r82_two_fix_garch.py canon <bench_all_bteg_rows.npz>
#         python tools/r82_two_fix_garch.py us_largecap
#         (needs data/panels/<universe>_calrows.npz: CAL_DUMP=1 python tools/r81_canon_calendar.py ... / r79_calendar_features.py ...)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from arch import arch_model
from arch.bootstrap import MCS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "code", "panels")); sys.path.insert(0, os.path.join(ROOT, "tools"))
ARGV = sys.argv; sys.argv = [ARGV[0], "us_largecap"]   # panel_study and r78 read a panel name at import; only their functions and the us_largecap settings are used
import panel_study as P
import r78_calendar_garch as R78
sys.argv = ARGV; C, X = P.C, P.X; WHAT = sys.argv[1]; AL = (0.01, 0.025); NW = int(os.environ.get("R82_WORKERS", "15")); CG = np.round(np.arange(1.0, 6.001, 0.1), 2); GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 2)
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r82"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
RIV = ("garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "fhs")

def cal_flags(a, dates):   # reaction day and the day after, as in r79 and r81
    n = len(dates); d0 = pd.DatetimeIndex(a.str[:10]); hr = a.str[11:13].astype(int).values + a.str[14:16].astype(int).values / 60.0
    p = np.searchsorted(dates.values, d0.values); inr = p < n; same = np.zeros(len(p), bool); same[inr] = dates.values[p[inr]] == d0.values[inr]; p = p + ((hr >= 16) & same); p = p[p < n]
    r0 = np.zeros(n); r0[p] = 1; r1 = np.zeros(n); r1[p[p + 1 < n] + 1] = 1; known = dates >= d0.min(); r0[~known] = np.nan; r1[~known] = np.nan
    return r0, r1

def shock_D(z, s2, al, be, mask):
    """For every day t: the largest standardized residual of days t-63..t-1 among days that feed the recursion (day s),
    and D = its share of s2[t]."""
    n = len(z); ze = np.where(mask, 0.0, z); win = sliding_window_view(ze, 63)[:-1]; j = np.argmax(np.abs(win - win.mean(1, keepdims=True)), 1); s = np.arange(len(win)) + j
    G = np.cumsum(np.where(mask, math.log(al + be), math.log(be))); t = np.arange(63, n); D = np.full(n, np.nan); sidx = np.zeros(n, int); sidx[63:] = s
    D[63:] = al * np.exp(G[t - 1] - G[s]) * s2[s] * (z[s] ** 2 - 1.0) / s2[t]
    return D, sidx

def work(job):
    key, y, loo, r0, r1, ne, mode = job
    try:
        n = len(y); pr = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params
        mu, nu, om, al, be = float(pr["mu"]), float(pr["nu"]), float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"])
        def filt(o_, a_, g_, b_, m_):   # the CRSP panel's own recursion (job_bench_all_bteg.py); the other panels use the arch filter
            e = y - m_; s2 = np.empty(n); s2[0] = np.var(y[:ne])
            for k in range(1, n): s2[k] = max(o_ + a_ * e[k - 1] ** 2 + g_ * e[k - 1] ** 2 * (e[k - 1] < 0) + b_ * s2[k - 1], 1e-8)
            return np.sqrt(s2)
        sig = filt(om, al, 0.0, be, mu) if mode == "manual" else np.asarray(arch_model(y, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
        z = (y - mu) / sig; Dg, sg = shock_D(z, sig ** 2, al, be, np.zeros(n, bool)); ok = np.isfinite(loo[:ne]); b = float(np.polyfit(loo[:ne][ok], y[:ne][ok], 1)[0]) if ok.sum() > 250 else np.nan
        wfun = lambda sidx, m_: np.nan_to_num(np.clip(b * loo[sidx] / np.where(np.abs(y[sidx] - m_) > 1e-9, y[sidx] - m_, np.nan), 0, 1))
        _, fm = R78.fit_one((key, y, r0, r1, ne, [mu, 0.0, al, be, nu]))
        if fm is None: return key, "fewer than 750 pre-test days with a calendar"
        k0 = fm["k0"]; th = fm["masked"]["th"]; mm, mal, mbe, mnu = float(th[0]), float(th[2]), float(th[3]), float(th[4]); ms2 = np.full(n, np.nan); ms2[k0:] = fm["masked"]["s2"]; mk = (r0 == 1) | (r1 == 1)
        zm = (y - mm) / np.sqrt(ms2); Dm = np.full(n, np.nan); sm = np.zeros(n, int); d_, s_ = shock_D(zm[k0:], ms2[k0:], mal, mbe, mk[k0:]); Dm[k0:] = d_; sm[k0:] = s_ + k0
        Qp = {"garch_t": [mu + sig * tq(a, nu) for a in AL]}; nanq = [np.full(n, np.nan)] * 2
        try:
            p2 = arch_model(y[:ne], vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False).fit(disp="off", show_warning=False).params; m2 = float(p2["mu"])
            s2_ = filt(float(p2["omega"]), float(p2["alpha[1]"]), float(p2["gamma[1]"]), float(p2["beta[1]"]), m2) if mode == "manual" else np.asarray(arch_model(y, vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False).fix(p2.values).conditional_volatility)
            Qp["gjr_skewt"] = [m2 + s2_ * float(C.SK.ppf(np.array([a]), np.array([float(p2["eta"]), float(p2["lambda"])]))[0]) for a in AL]
        except Exception: Qp["gjr_skewt"] = nanq
        try:
            bx = C.fit_bteg(y[:ne]); bmu, bom, phi, kap, bnu = C.bteg_unpack(bx); bsd = np.exp(C._bteg_filter(y, bmu, bom, phi, kap, bnu)) * math.sqrt(bnu / (bnu - 2)); Qp["param_bteg"] = [bmu + bsd * tq(a, bnu) for a in AL]
        except Exception: Qp["param_bteg"] = nanq
        try: Qp["sav_caviar"] = [np.asarray(X.caviar_sav(y[:ne], y, a)) for a in AL]
        except Exception: Qp["sav_caviar"] = nanq
        ze = z[:ne]; ze = ze[np.isfinite(ze)]; Qp["fhs"] = [mu + sig * float(np.quantile(ze, a)) for a in AL]
        cols = lambda i: dict(y=y[i], r0=r0[i], r1=r1[i], sig=sig[i], mu=mu, nu=nu, Dg=Dg[i], wg=wfun(sg, mu)[i], msig=np.sqrt(ms2[i]), mmu=mm, mnu=mnu, Dm=Dm[i], wm=wfun(sm, mm)[i])
        i = np.arange(k0 + 126, ne); i = i[np.isfinite(Dm[i]) & np.isfinite(Dg[i])]; pre = pd.DataFrame(cols(i))
        for m in RIV:
            for j, a in enumerate(AL): pre["%s__%s" % (m, a)] = Qp[m][j][i]
        return key, dict(pre=pre, te=pd.DataFrame(cols(np.arange(ne, n))), ne=ne, par=dict(alpha=al, malpha=mal, nu=nu, mnu=mnu, b=b))
    except Exception as e:
        return key, "failed: %s" % str(e)[:100]

if __name__ == "__main__":
    Dd = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_calrows.npz")); dn = Dd["name"].astype(str); names = list(pd.unique(dn))[:int(os.environ.get("R82_MAXN", "100000"))]; SER = {}
    if WHAT == "canon":
        Rw = np.load(sys.argv[2]); rname = Rw["permno"].astype(int).astype(str); rk_key = "rk_mk"
        rr = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "crsp_panel_returns.csv"), dtype={"permno": "int32"}); rr["date"] = pd.to_datetime(rr["date"]); rr["ret"] = pd.to_numeric(rr["ret"], errors="coerce") * 100.0
        for k in names:
            g = rr[rr.permno == int(k)].sort_values("date"); y = g["ret"].values.astype(float); dates = pd.DatetimeIndex(g["date"].values); a = pd.read_csv(os.path.join(ROOT, "data", "panels", "earnings_ibes", k + ".csv"))["announcement"].astype(str)
            SER[k] = (y, dates, int(len(y) * 0.6), a[a.str.len() >= 16], "manual")
        STORED = {"GJR-skew-t": "gjr_skewt", "SAV-CAViaR": "sav_caviar", "FHS": "fhs_name", "trees with EVT tail": "engine", "trees with EVT tail, robust scale": "engine_bteg", "pooled GARCH-EVT": "evt_pool"}
    else:
        Rw = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); rname = np.array([str(x) for x in Rw["names"]])[Rw["name"]]; rk_key = "rk"
        for k in names:
            s = pd.read_csv(os.path.join(P.DDIR, k.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]; s = s[(s > 0) & (s.index <= P.END)]; r = (s.pct_change() * 100).dropna(); r = r[np.isfinite(r)]
            SER[k] = (r.values.astype(float), r.index, int((r.index < P.TEST0).sum()), pd.read_csv(os.path.join(ROOT, "data", "panels", "earnings", k + ".csv"))["announcement"].astype(str), "arch")
        STORED = {"GJR-skew-t": "gjr_skewt", "SAV-CAViaR": "sav_caviar", "FHS": "fhs", "trees with EVT tail": "engine", "trees with EVT tail, robust scale": "engine_bteg"}
    W = pd.DataFrame({k: pd.Series(v[0], index=v[1]) for k, v in SER.items()}); sm_, ct_ = W.sum(1), W.count(1); jobs = []
    for k in names:
        y, dates, ne, a, mode = SER[k]; c = ct_.reindex(dates).values; loo = np.where(c >= 6, (sm_.reindex(dates).values - y) / np.maximum(c - 1, 1), np.nan); r0, r1 = cal_flags(a, dates); jobs.append((k, y, loo, r0, r1, ne, mode))
    with Pool(NW) as pool: res = pool.map(work, jobs, chunksize=1)
    bad = {k: v for k, v in res if not isinstance(v, dict)}; Rr = {k: v for k, v in res if isinstance(v, dict)}; names = [k for k in names if k in Rr]; lg("%s: %d names fitted (left out: %s)" % (WHAT, len(names), bad))
    PRE = pd.concat([Rr[k]["pre"] for k in names], ignore_index=True); TE = pd.concat([Rr[k]["te"] for k in names], ignore_index=True)
    NAME = np.concatenate([np.full(len(Rr[k]["te"]), k) for k in names]); DATE = np.concatenate([np.array([str(d.date()) for d in SER[k][1][Rr[k]["ne"]:]]) for k in names]); n = len(TE); Y = TE.y.values
    key = pd.MultiIndex.from_arrays([NAME, DATE]); pos = pd.Series(np.arange(len(dn)), index=pd.MultiIndex.from_arrays([dn, Dd["date"].astype(str)])).reindex(key).values; assert np.isfinite(pos).all(); pos = pos.astype(int)
    assert np.allclose(Dd["y"][pos], Y, atol=1e-6) and np.array_equal(np.nan_to_num(Dd["cal"][pos][:, 0], nan=-1), np.nan_to_num(TE.r0.values, nan=-1))
    rpos = pd.Series(np.arange(len(rname)), index=pd.MultiIndex.from_arrays([rname, Rw["date"].astype(str)])).reindex(key).values; has = np.isfinite(rpos); rp = np.where(has, rpos, 0).astype(int); TAUS = [round(float(t), 4) for t in Rw["taus"]]; JT = [TAUS.index(a) for a in AL]
    Q = {}; DQ = {k[3:]: Dd[k][pos] for k in Dd.files if k.startswith("q__")}
    for nm, src in (("GARCH-t", "GARCH-t"), ("Beta-t-EGARCH", "Beta-t-EGARCH"), ("trees", "trees, stored"), ("trees on robust scale", "trees on robust scale, stored"), ("neural network", "neural network, stored")): Q[nm] = DQ[src]
    for nm, m in STORED.items(): q = Rw["q__" + m].astype(float)[:, JT][rp]; q[~has] = np.nan; Q[nm] = q
    NOCAL = list(Q); R0 = TE.r0.values == 1; R1 = (TE.r1.values == 1) & ~R0; P0 = PRE.r0.values == 1; P1 = (PRE.r1.values == 1) & ~P0; PO = ~(P0 | P1)
    cfit = lambda y, qfun: float(CG[int(np.argmin([pin(y, qfun(c), 0.025).sum() for c in CG]))])
    # parametric rivals with calendar multipliers chosen on pre-test rows
    MULT = {}
    for m, nm in zip(RIV, ("GARCH-t", "GJR-skew-t", "Beta-t-EGARCH", "SAV-CAViaR", "FHS")):
        qp = PRE["%s__0.025" % m].values; f = np.isfinite(qp); c0 = cfit(PRE.y.values[P0 & f], lambda c: c * qp[P0 & f]); c1 = cfit(PRE.y.values[P1 & f], lambda c: c * qp[P1 & f]); MULT[nm] = (c0, c1)
        Q[nm + ", calendar multipliers"] = Q[nm] * np.where(R0, c0, np.where(R1, c1, 1.0))[:, None]
    # masked calendar GARCH-t (r78), recomputed here and checked against the r79/r81 forecast
    def tail_t(sig, mu, nu, cc): return np.column_stack([mu + sig * np.where(R0, cc[0], np.where(R1, cc[1], 1.0)) * tq(a, nu) for a in AL])
    pq = lambda sig, c: PRE.mmu.values + c * sig * tq(0.025, PRE.mnu.values); psig = PRE.msig.values
    cm = (cfit(PRE.y.values[P0], lambda c: pq(psig, c)[P0]), cfit(PRE.y.values[P1], lambda c: pq(psig, c)[P1])); Q["masked calendar GARCH-t"] = tail_t(TE.msig.values, TE.mmu.values, TE.mnu.values, cm)
    lg("masked calendar GARCH-t against the stored forecast of the calendar run: largest gap %.2e; multipliers %s" % (float(np.nanmax(np.abs(Q["masked calendar GARCH-t"] - DQ["masked calendar GARCH-t"]))), cm))
    for nm in ("trees, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs", "neural network, calendar inputs"): Q[nm] = DQ[nm]
    # the shock correction: lambda_f and lambda_m on pre-test ordinary days where the correction matters (D above 0.02)
    def lam_fit(sig, mu, nu, D, w):
        f = PO & (D > 0.02); y_, s_, m_, D_, w_, b_ = PRE.y.values[f], sig[f], mu[f], np.clip(D[f], -0.5, 0.98), w[f], tq(0.025, nu[f]); best = (1e30, 0.0, 0.0)
        for lf in GRID:
            for lm in GRID:
                L = pin(y_, m_ + s_ * np.sqrt(np.maximum(1 - (1 - (lf + (lm - lf) * w_)) * D_, 0.02)) * b_, 0.025).sum()
                if L < best[0]: best = (L, float(lf), float(lm))
        return best[1], best[2], int(f.sum())
    fix = lambda sig, D, w, lf, lm: sig * np.sqrt(np.maximum(1 - (1 - (lf + (lm - lf) * w)) * np.clip(D, -0.5, 0.98), 0.02))
    lfm, lmm, nm_ = lam_fit(PRE.msig.values, PRE.mmu.values, PRE.mnu.values, PRE.Dm.values, PRE.wm.values); lfg, lmg, ng_ = lam_fit(PRE.sig.values, PRE.mu.values, PRE.nu.values, PRE.Dg.values, PRE.wg.values)
    lg("pre-test real shares: masked filter lambda_f %.2f, lambda_m %.2f (%d rows); plain filter %.2f, %.2f (%d rows)" % (lfm, lmm, nm_, lfg, lmg, ng_))
    Q["shock correction only"] = np.column_stack([TE.mu.values + fix(TE.sig.values, TE.Dg.values, TE.wg.values, lfg, lmg) * tq(a, TE.nu.values) for a in AL])
    psf = fix(PRE.msig.values, PRE.Dm.values, PRE.wm.values, lfm, lmm); tsf = fix(TE.msig.values, TE.Dm.values, TE.wm.values, lfm, lmm)
    c2 = (cfit(PRE.y.values[P0], lambda c: pq(psf, c)[P0]), cfit(PRE.y.values[P1], lambda c: pq(psf, c)[P1])); Q["two-fix GARCH-t"] = tail_t(tsf, TE.mmu.values, TE.mnu.values, c2)
    zf = (PRE.y.values - PRE.mmu.values) / psf; EQ = {lab: [float(np.quantile(zf[f], a)) for a in AL] for lab, f in (("reaction day", P0), ("day after", P1), ("other days", PO))}
    Q["two-fix GARCH-t, empirical quantiles"] = np.column_stack([TE.mmu.values + tsf * np.where(R0, EQ["reaction day"][j], np.where(R1, EQ["day after"][j], EQ["other days"][j])) for j in range(2)])
    U = Dd["U"][pos] & np.all([np.isfinite(q).all(1) for q in Q.values()], axis=0); rk = Dd["rk"][pos]; dates = pd.to_datetime(DATE); TOP = rk == 10; RW = R0 | R1
    L = {m: {a: pin(Y, q[:, j], a) for j, a in enumerate(AL)} for m, q in Q.items()}
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
        if len(s) < 30: return None
        dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    edge = lambda Lr, Lm, mask: None if mask.sum() < 50 else {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
    CELLS = (("all days", U), ("reaction day", U & R0), ("day after", U & R1), ("other days", U & ~RW), ("top decile", U & TOP), ("top decile, other days", U & TOP & ~RW), ("deciles 1-8, other days", U & (rk <= 8) & (rk > 0) & ~RW))
    TF = "two-fix GARCH-t"; TFE = "two-fix GARCH-t, empirical quantiles"; MK = "masked calendar GARCH-t"
    OUT = {"universe": WHAT, "names": len(names), "rows": int(U.sum()), "predictions_written_before_run": "see the script header (T1-T7)", "models_without_calendar": NOCAL,
           "real_share_pre_test": {"masked filter": {"lambda_f": lfm, "lambda_m": lmm, "rows": nm_}, "plain filter": {"lambda_f": lfg, "lambda_m": lmg, "rows": ng_}},
           "multipliers_c0_c1": {**{k + ", calendar multipliers": v for k, v in MULT.items()}, MK: cm, TF: c2}, "empirical_quantiles_two_fix": {k: [round(x, 3) for x in v] for k, v in EQ.items()},
           "student_t_quantiles_at_median_nu": [round(float(tq(a, np.median(PRE.mnu.values))), 3) for a in AL]}
    for j, a in enumerate(AL):
        g = lambda m: 100 * float((L["GARCH-t"][a] - L[m][a])[U].mean() / L["GARCH-t"][a][U].mean()); mean_loss = {m: float(L[m][a][U].mean()) for m in Q}; best_flex = min([m for m in Q if m.endswith("calendar inputs")], key=mean_loss.get)
        M = pd.DataFrame({m: pd.Series(L[m][a][U], index=dates[U]).groupby(level=0).mean() for m in Q}).dropna(); mcs = {}
        for meth in ("max", "R"):
            mc = MCS(M, size=0.10, reps=1000, block_size=10, method=meth, bootstrap="stationary", seed=0); mc.compute(); mcs[meth] = {"included": list(mc.included), "pvalues": {str(k): round(float(v), 4) for k, v in mc.pvalues["Pvalue"].items()}}
        OUT[str(a)] = {"vs_garch_t": {m: {lab: edge(L["GARCH-t"][a], L[m][a], c) for lab, c in CELLS} for m in Q if m != "GARCH-t"},
                       "two_fix_vs": {m: {lab: edge(L[m][a], L[TF][a], c) for lab, c in CELLS} for m in Q if m != TF},
                       "two_fix_empirical_vs": {m: {lab: edge(L[m][a], L[TFE][a], c) for lab, c in CELLS} for m in Q if m != TFE},
                       "two_by_two_gain_over_garch_t_pct": {"calendar only": round(g(MK), 2), "shock correction only": round(g("shock correction only"), 2), "both": round(g(TF), 2), "sum of the two": round(g(MK) + g("shock correction only"), 2)},
                       "best_flexible_with_calendar": best_flex, "share_of_gap_closed": {"two-fix GARCH-t": round((g(TF) - g(MK)) / (g(best_flex) - g(MK)), 3), TFE: round((g(TFE) - g(MK)) / (g(best_flex) - g(MK)), 3)},
                       "breach_pct": {m: {lab: round(100 * float((Y <= Q[m][:, j])[c].mean()), 2) for lab, c in CELLS} for m in Q}, "lowest_loss_all_days": min(mean_loss, key=mean_loss.get), "mcs_90pct_all_days": mcs}
    json.dump(OUT, open(os.path.join(OUTD, os.environ.get("R82_TAG", WHAT) + ".json"), "w"), indent=1)
    f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"]); SH = ("all days", "reaction day", "other days", "top decile, other days", "deciles 1-8, other days")
    print("\n######## %s: %d names, %d rows; real shares %s; multipliers %s; empirical quantiles %s against Student-t %s" % (WHAT, len(names), U.sum(), OUT["real_share_pre_test"], OUT["multipliers_c0_c1"], OUT["empirical_quantiles_two_fix"], OUT["student_t_quantiles_at_median_nu"]))
    for a in ("0.025", "0.01"):
        print("--- %s: each model against GARCH-t (lowest loss over all days: %s)" % (a, OUT[a]["lowest_loss_all_days"]))
        for m, v in OUT[a]["vs_garch_t"].items(): print("   %-42s %s" % (m, "  ".join("%s %s" % (lab, f(v[lab])) for lab in SH)))
        for blk, ttl in (("two_fix_vs", "the two-fix GARCH-t"), ("two_fix_empirical_vs", "the two-fix GARCH-t with empirical quantiles")):
            print("--- %s: %s against each model (positive: it is better)" % (a, ttl))
            for m, v in OUT[a][blk].items(): print("   %-42s %s" % (m, "  ".join("%s %s" % (lab, f(v[lab])) for lab in SH)))
        print("--- %s: two by two %s; best flexible with calendar: %s; share of the gap closed %s" % (a, OUT[a]["two_by_two_gain_over_garch_t_pct"], OUT[a]["best_flexible_with_calendar"], OUT[a]["share_of_gap_closed"]))
        print("--- %s: 90%% model confidence set, all days: max %s | R %s" % (a, OUT[a]["mcs_90pct_all_days"]["max"]["included"], OUT[a]["mcs_90pct_all_days"]["R"]["included"]))
        print("--- %s: breach rates (all days, reaction day, other days): %s" % (a, {m: (v["all days"], v["reaction day"], v["other days"]) for m, v in OUT[a]["breach_pct"].items()}))
    lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
