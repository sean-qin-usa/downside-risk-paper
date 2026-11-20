# r71_shock_response.py -- THE CLOSED-FORM SHOCK RESPONSE OF GARCH(1,1), TESTED ON EVERY PANEL AND SUBGROUP.
#
# The identity. GARCH(1,1) sets s2[t+1] = omega + alpha * e[t]^2 + beta * s2[t]. Hold every return fixed except the one
# on day s, and replace e[s]^2 by its conditional expectation s2[s] (an ordinary day). The variance k days later falls by
#       Delta_k = alpha * beta^(k-1) * s2[s] * (z[s]^2 - 1),          z[s] = e[s] / sqrt(s2[s]).
# This is exact for the filter (it is linear in e[s]^2), so the share of today's GARCH variance that is there only
# because day s was large is D = Delta_k / s2[t], and the filter's response to one shock decays at rate beta per day
# (half-life log(0.5)/log(beta)), not at alpha+beta.
#
# The hypothesis. Suppose a share lambda of that extra variance is real and the rest is not:
#       true variance = s2[t] * (1 - (1 - lambda) * D).
# lambda = 1 is GARCH. lambda = 0 says the shock was a one-off that says nothing about later variance. Each value gives,
# with no further free parameter and row by row, the breach rate of GARCH-t's VaR, the VaR that would be right, and how
# long the error lasts for a name with a given beta. "The shock" is the largest standardized residual of the trailing
# 63 days (the observation the paper's score records).
#
# Blocks, per universe (a panel, or a subgroup of one):
#  A  top score decile by shock age: GARCH-t's realized breach rate at 1% and 2.5% beside the rate lambda=0 predicts;
#     the 2.5% quantile of the residuals standardized by the corrected scale beside the Student-t quantile; loss edges
#     over GARCH-t of the corrected GARCH-t (lambda = 0), Beta-t-EGARCH and the flexible models
#  B  lambda estimated: by pinball at 2.5% and 1% and by Student-t likelihood on the top decile of the test rows, with
#     a stationary-bootstrap interval over dates; by shock sign and by shock age; and on the pre-test rows, which gives an
#     out-of-sample one-parameter corrected GARCH-t
#  C  what each model has learned: its implied variance ratio rho^2 = ((VaR_m - mu) / (VaR_garch - mu))^2 regressed on D
#     in the top decile; the slope is the share of the shock's variance the model discards (1 - its implied lambda)
#  D  duration: names split by beta tercile; breach rates by shock age in each
#  E  dose-response over ALL rows by D: realized breach against the lambda=0 prediction
#  F  D as a score: the flexible models' edges over GARCH-t in the top D-decile against the top kurtosis decile
#
# Universes: the canonical CRSP panel, split by SIC division, volatility tercile and size cohort; each panel of
# docs/PANELS_PRECOMMIT.md, split by volatility tercile, and the international panel by market.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: the canonical top-decile breach rates by shock age
# (tools/r69_tail_levels.py) and the panels' graded hypotheses. Not known: anything about D, lambda or the corrected model.
#  P1. Stocks (canonical, us_largecap, intl_largecap): lambda-hat (pinball 2.5%, top decile) is below 0.5 and its
#      interval excludes 1. Crypto: below 0.5. Equity ETFs, other ETFs and currencies: above the stock panels' value.
#  P2. On the three stock panels the corrected GARCH-t (lambda = 0, no free parameter) recovers at least half of
#      Beta-t-EGARCH's top-decile edge over GARCH-t at 2.5%, and does not lose to GARCH-t over all days.
#  P3. On the three stock panels the realized GARCH-t breach rate across the D bins of block E stays within 0.5
#      percentage points of the lambda=0 prediction at 2.5% in every bin with at least 2,000 rows.
#  P4. Implied discard shares (block C slope) on the stock panels: between 0.5 and 1.1 for Beta-t-EGARCH, the pooled
#      trees and SAV-CAViaR; GJR-skew-t's is below 0.25.
#  P5. Duration scales with beta: within the top decile, at shock ages 22 to 42 days, the high-beta tercile has a lower
#      GARCH-t breach rate than the low-beta tercile on each stock panel.
#  P6. Canonical subgroups: lambda-hat is below 0.5 in every SIC group, volatility tercile and size cohort with at
#      least 10 names, and the 2.5% breach rate at ages 6 to 15 is below 1.5% in each.
#  P7. D is at least as good a locator as the kurtosis score: on the stock panels the pooled trees' 2.5% edge over
#      GARCH-t in the top D-decile is at least as large as in the top kurtosis decile.
# Block G, added the same day BEFORE any earnings date was read (US stock universes only; dates from
# code/panels/download_earnings.py): a shock is an "earnings shock" if it falls on an announcement date or the next
# trading day. Known beforehand: blocks A-F of the canonical panel, us_largecap and three non-stock panels.
#  P8. On us_largecap and on the canonical panel at least half of the top-decile rows have an earnings shock as their
#      largest shock (about 3% of trading days fall in such windows).
#  P9. lambda-hat is lower for earnings shocks than for other shocks, and for other shocks it is above 0.3.
#  P10. The corrected GARCH-t's and the pooled trees' top-decile edges over GARCH-t are larger for earnings shocks.
#  P11 (added after P8-P10 were graded on us_largecap, before this quantity was computed anywhere). An earnings-only
#      correction -- lambda = 0 when the largest shock is an earnings shock, plain GARCH-t otherwise, so no parameter and
#      nothing that is not on a calendar beforehand -- beats GARCH-t over all days on both US universes (DM > 2), and its
#      top-decile edge is at least that of the all-shock correction.
# Usage:  GBC_PROJ=<crsp dir> python tools/r71_shock_response.py canon <bench_all_bteg_rows.npz> [r70_preds.npz]
#         python tools/r71_shock_response.py <panel>          (needs data/panels/<panel>_rows.npz and the price files)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r71"); os.makedirs(OUTD, exist_ok=True)
AL = (0.01, 0.025); GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 2); NB = int(os.environ.get("R71_BOOT", "500")); NW = int(os.environ.get("R71_WORKERS", "12"))
AGES = (("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16-21", 16, 21), ("22-30", 22, 30), ("31-45", 31, 45), ("46-63", 46, 63))
AGES3 = (("1-10", 1, 10), ("11-21", 11, 21), ("22-42", 22, 42), ("43-63", 43, 63)); AGESL = (("1-5", 1, 5), ("6-10", 6, 10), ("11-21", 11, 21), ("22-42", 22, 42), ("43-63", 43, 63))
DBINS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0)

def shock_arrays(a):
    """One name: GARCH-t fit on the estimation window, filter over the whole history, and for every day t the largest
    standardized residual of days t-63..t-1: its age, sign, size, and D = Delta / s2[t]."""
    key, y, ne, mode = a
    try:
        am = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False); pr = am.fit(disp="off", show_warning=False).params
        om, al, be, mu, nu = float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"]), float(pr["mu"]), float(pr["nu"]); n = len(y)
        if mode == "arch": sig = np.asarray(arch_model(y, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
        else:
            e0 = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:ne])
            for k in range(1, n): s2[k] = max(om + al * e0[k - 1] ** 2 + be * s2[k - 1], 1e-8)
            sig = np.sqrt(s2)
        z = (y - mu) / sig; win = sliding_window_view(z, 63)[:-1]; j = np.argmax(np.abs(win - win.mean(1, keepdims=True)), 1); s = np.arange(len(win)) + j
        age = np.zeros(n, int); D = np.full(n, np.nan); sgn = np.zeros(n); siz = np.full(n, np.nan); sidx = np.zeros(n, int); sidx[63:] = s
        age[63:] = 63 - j; sgn[63:] = np.sign(z[s]); siz[63:] = np.abs(z[s])
        D[63:] = al * be ** (age[63:] - 1.0) * sig[s] ** 2 * (z[s] ** 2 - 1.0) / sig[63:] ** 2
        mk = pd.Series(z).rolling(63, min_periods=30).kurt().shift(1).values
        return key, dict(sidx=sidx, sig=sig, z=z, age=age, D=D, sgn=sgn, siz=siz, mk=mk, mu=mu, nu=nu, al=al, be=be, om=om, sd_est=float(np.std(y[:ne])))
    except Exception as e:
        return key, "failed: %s" % str(e)[:80]

# ---------------------------------------------------------------- load one universe into row arrays
if WHAT == "canon":
    P = os.environ["GBC_PROJ"]; Dm = np.load(sys.argv[2]); PN = Dm["permno"]; TAUS = [round(float(t), 4) for t in Dm["taus"]]
    rr = pd.read_csv(os.path.join(P, "crsp_panel_returns.csv"), dtype={"permno": "int32"}); rr["date"] = pd.to_datetime(rr["date"]); rr["ret"] = pd.to_numeric(rr["ret"], errors="coerce") * 100.0
    ch = pd.read_csv(os.path.join(P, "crsp_panel_chars.csv")).set_index("permno"); names = list(pd.unique(PN)); jobs = []; YS = {}
    for pn in names:
        y = rr[rr.permno == pn].sort_values("date")["ret"].values.astype(float); YS[pn] = y; jobs.append((pn, y, int(len(y) * 0.6), "manual"))
    Y = Dm["y"].astype(float); dates = pd.to_datetime(Dm["date"]); rk = Dm["rk_mk"]; NAME = PN; MODELS = ["garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "body", "engine", "body_bteg", "engine_bteg", "evt_pool"]
    Q = {m: Dm["q__" + m].astype(float) for m in MODELS}; rowsel = {}
    for pn in names: m = np.where(PN == pn)[0]; n = len(YS[pn]); rowsel[pn] = (m, np.arange(n - len(m), n)); assert np.allclose(YS[pn][-len(m):], Y[m], atol=1e-4)
    if len(sys.argv) > 3:
        N = np.load(sys.argv[3]); kk = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); Q["neural"] = N["q"][kk.loc[list(zip(PN, Dm["date"]))].values].astype(float); MODELS.append("neural")
    pre_start = {pn: 126 for pn in names}; test_start = {pn: rowsel[pn][1][0] for pn in names}
    DT = {pn: pd.DatetimeIndex(rr[rr.permno == pn].sort_values("date")["date"].values) for pn in names}; TICK = {pn: str(ch.loc[pn, "ticker"]) for pn in names}
    SICN = {0: "agriculture", 1: "mining and construction", 2: "manufacturing, nondurable", 3: "manufacturing, durable", 4: "transport, communications, utilities",
            5: "trade", 6: "finance, insurance, real estate", 7: "business services", 8: "health and other services", 9: "non-classifiable"}
    sic = {pn: int(ch.loc[pn, "sector"]) for pn in names}; cnt = pd.Series(sic).value_counts()
    GROUPS = {"sic": {pn: (SICN[s] if (cnt[s] >= 10 or s == 9) else "other divisions") for pn, s in sic.items()}, "cohort": {pn: str(ch.loc[pn, "cohort"]) for pn in names}}
else:
    BASE = WHAT[:-9] if WHAT.endswith("_screened") else WHAT   # "<panel>_screened": the data-quality-screened run of code/panels/panel_study.py
    CFG = json.load(open(os.path.join(ROOT, "code", "panels", "panels.json")))[BASE]; Dm = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); TAUS = [round(float(t), 4) for t in Dm["taus"]]
    CAL0, TEST0 = pd.Timestamp(CFG["cal0"]), pd.Timestamp(CFG["test0"]); names = [str(x) for x in Dm["names"]]; NAME = np.array(names)[Dm["name"]]; jobs = []; DT = {}
    for tk in names:
        s = pd.read_csv(os.path.join(ROOT, "data", "panels", BASE, tk.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]; s = s[(s > 0) & (s.index <= "2026-11-13")]
        if tk.endswith("=X"):
            r_ = s.pct_change(); s = s[~((r_.abs() > 0.03) & (r_.shift(-1).abs() > 0.03) & (np.sign(r_) != np.sign(r_.shift(-1)))).fillna(False).values]
        r = (s.pct_change() * 100).dropna(); r = r[np.isfinite(r)]; DT[tk] = r.index; jobs.append((tk, r.values.astype(float), int((r.index < TEST0).sum()), "arch"))
    YS = {a[0]: a[1] for a in jobs}; Y = Dm["y"].astype(float); dates = pd.to_datetime(Dm["date"]); rk = Dm["rk"]; MODELS = ["garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "neural", "body", "engine", "body_bteg", "engine_bteg", "fhs"]
    Q = {m: Dm["q__" + m].astype(float) for m in MODELS}; rowsel = {}; dstr = Dm["date"]
    for tk in names:
        m = np.where(NAME == tk)[0]; pos = pd.Series(np.arange(len(DT[tk])), index=[str(d.date()) for d in DT[tk]]).loc[list(dstr[m])].values; rowsel[tk] = (m, pos); assert np.allclose(YS[tk][pos], Y[m], atol=1e-3), tk
    pre_start = {tk: 126 for tk in names}; test_start = {tk: int((DT[tk] < TEST0).sum()) for tk in names}; GROUPS = {}; TICK = {tk: tk for tk in names}
    if BASE == "intl_largecap":
        MK = {"L": "United Kingdom", "DE": "Germany", "PA": "France", "T": "Japan", "TO": "Canada", "AX": "Australia", "HK": "Hong Kong"}; GROUPS["market"] = {tk: MK.get(tk.split(".")[-1], "other") for tk in names}
with Pool(NW) as pool: res = dict(pool.map(shock_arrays, jobs, chunksize=2))
bad = [k for k, v in res.items() if not isinstance(v, dict)]; assert not bad, bad
lg("%s: %d names filtered" % (WHAT, len(names)))
n = len(Y); J = {a: TAUS.index(a) for a in AL}
SIG, MU, NU, BE, AGE, DD, SGN, SIZ = (np.zeros(n) for _ in range(8)); PRE = []
for k in names:
    m, pos = rowsel[k]; r = res[k]; SIG[m] = r["sig"][pos]; MU[m] = r["mu"]; NU[m] = r["nu"]; BE[m] = r["be"]; AGE[m] = r["age"][pos]; DD[m] = r["D"][pos]; SGN[m] = r["sgn"][pos]; SIZ[m] = r["siz"][pos]
    i = np.arange(pre_start[k], test_start[k]); ok = np.isfinite(r["mk"][i]) & np.isfinite(r["D"][i]); i = i[ok]
    PRE.append(pd.DataFrame({"y": YS[k][i], "sig": r["sig"][i], "mu": r["mu"], "nu": r["nu"], "D": r["D"][i], "mk": r["mk"][i]}))
PRE = pd.concat(PRE, ignore_index=True)
# earnings windows: announcement date and the next trading day; EARN = 1 earnings shock, 0 other shock, nan unknown
EARN = np.full(n, np.nan); EDIR = os.path.join(ROOT, "data", "panels", os.environ.get("R71_EARN_DIR", "earnings")); ecov = 0; ebase = []
if WHAT == "canon" and os.environ.get("R71_EARN_DIR"): TICK = {pn: str(pn) for pn in names}   # R71_EARN_DIR=earnings_ibes36: the I/B/E/S calendar of tools/r81_canon_calendar.py, one file per permno (186 names)
if WHAT in ("canon", "us_largecap") and os.path.isdir(EDIR):
    for k in names:
        fn = os.path.join(EDIR, TICK[k] + ".csv")
        if not os.path.exists(fn): continue
        ed = pd.DatetimeIndex(sorted(set(pd.Timestamp(str(x)[:10]) for x in pd.read_csv(fn)["announcement"])))
        if len(ed) < 8: continue
        dt = DT[k]; pos = np.searchsorted(dt.values, ed.values); pos = pos[pos < len(dt)]; flag = np.zeros(len(dt), bool); flag[pos] = True; flag[np.minimum(pos + 1, len(dt) - 1)] = True
        m, ps = rowsel[k]; sd_ = res[k]["sidx"][ps]; e = flag[sd_].astype(float); e[dt.values[sd_] < ed.values[0]] = np.nan; EARN[m] = e; ecov += 1
        cov_ = (dt >= ed[0]) & (dt <= ed[-1])
        if cov_.any(): ebase.append(float(flag[cov_].mean()))
    lg("earnings dates for %d of %d names; %.1f%% of their trading days are in an earnings window" % (ecov, len(names), 100 * float(np.mean(ebase)) if ebase else float("nan")))
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2))
chk = float(np.nanmax(np.abs(MU + SIG * tq(0.025, NU) - Q["garch_t"][:, J[0.025]]) / np.abs(Q["garch_t"][:, J[0.025]]))); lg("refit check: largest relative gap to the stored GARCH-t 2.5%% VaR = %.2e" % chk)
assert np.nanquantile(np.abs(MU + SIG * tq(0.025, NU) - Q["garch_t"][:, J[0.025]]) / np.abs(Q["garch_t"][:, J[0.025]]), 0.999) < 0.02
DD = np.clip(DD, -0.5, 0.98); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
cfq = lambda a, lam, Dv=None, sig=None, mu=None, nu=None: (MU if mu is None else mu) + (SIG if sig is None else sig) * np.sqrt(np.maximum(1 - (1 - lam) * (DD if Dv is None else Dv), 0.02)) * tq(a, NU if nu is None else nu)
vol_ter = pd.qcut(pd.Series({k: res[k]["sd_est"] for k in names}), 3, labels=["low volatility tercile", "mid volatility tercile", "high volatility tercile"]); GROUPS["vol"] = {k: str(vol_ter[k]) for k in names}
be_ter = pd.qcut(pd.Series({k: res[k]["be"] for k in names}).rank(method="first"), 3, labels=["low beta", "mid beta", "high beta"]); BT = np.array([str(be_ter[k]) for k in NAME])
# out-of-sample lambda: pre-test rows in the top decile of the pre-test score
pre_top = PRE[PRE.mk >= PRE.mk.quantile(0.9)]
def lam_fit(y, mu, sig, nu, Dv, a):
    L = [pin(y, cfq(a, l, Dv, sig, mu, nu), a).mean() for l in GRID]; return float(GRID[int(np.argmin(L))])
LAM_PRE = {a: lam_fit(pre_top.y.values, pre_top.mu.values, pre_top.sig.values, pre_top.nu.values, np.clip(pre_top.D.values, -0.5, 0.98), a) for a in AL}
lg("pre-test lambda (top decile of %d pre-test rows): %s" % (len(PRE), LAM_PRE))
QX = dict(Q); QC0 = {a: cfq(a, 0.0) for a in AL}; QCP = {a: cfq(a, LAM_PRE[0.025]) for a in AL}
LOSS = {a: {m: pin(Y, QX[m][:, J[a]], a) for m in MODELS} for a in AL}
for a in AL: LOSS[a]["cf0"] = pin(Y, QC0[a], a); LOSS[a]["cf_pre"] = pin(Y, QCP[a], a); LOSS[a]["cf_earn"] = np.where(EARN == 1, LOSS[a]["cf0"], LOSS[a]["garch_t"])
if os.environ.get("R71_DUMP"):
    # Per-row file for tools/r76_origin_accounting.py (not committed). Nothing is estimated or graded here: for every test
    # row it stores the GARCH-t fit, the date and return of the largest shock, the return of the other names of the panel
    # on that date (leave-one-out equal-weighted mean; within the name's own market on the international panel), the name's
    # slope on that mean over the pre-test window, and whether the row's own date is an announcement date or the day after.
    grp = GROUPS.get("market", {k: "all" for k in names}); MKT = {}
    for g in sorted(set(grp.values())):
        W = pd.DataFrame({k: pd.Series(YS[k], index=DT[k]) for k in names if grp[k] == g}); MKT[g] = (W.sum(1), W.count(1))
    SDATE = np.empty(n, "datetime64[D]"); SRET, MS, MZ, BM, E0, E1, ALR = (np.full(n, np.nan) for _ in range(7)); PX = []; SER = {}
    for k in names:
        sm, ct = MKT[grp[k]]; dt = DT[k]; own = pd.Series(YS[k], index=dt); c_ = ct.reindex(dt).values; loo = pd.Series(np.where(c_ >= 6, (sm.reindex(dt).values - own.values) / np.maximum(c_ - 1, 1), np.nan), index=dt)
        lv = loo.values; ne_ = test_start[k]; ok = np.isfinite(lv[:ne_]); b = float(np.polyfit(lv[:ne_][ok], YS[k][:ne_][ok], 1)[0]) if ok.sum() > 250 else np.nan
        mz = lv / loo.rolling(250, min_periods=60).std().shift(1).values; f0 = np.full(len(dt), np.nan); f1 = f0.copy(); fn = os.path.join(EDIR, TICK[k] + ".csv")
        if WHAT in ("canon", "us_largecap") and os.path.exists(fn):
            ed = pd.DatetimeIndex(sorted(set(pd.Timestamp(str(x)[:10]) for x in pd.read_csv(fn)["announcement"])))
            if len(ed) >= 8:
                p_ = np.searchsorted(dt.values, ed.values); p_ = p_[p_ < len(dt)]; f0 = np.zeros(len(dt)); f1 = np.zeros(len(dt)); f0[p_] = 1; f1[np.minimum(p_ + 1, len(dt) - 1)] = 1; bf = dt.values < ed.values[0]; f0[bf] = np.nan; f1[bf] = np.nan
        m, pos = rowsel[k]; r = res[k]; sd_ = r["sidx"][pos]; SDATE[m] = dt.values[sd_].astype("datetime64[D]"); SRET[m] = YS[k][sd_] - r["mu"]; MS[m] = lv[sd_]; MZ[m] = mz[sd_]; BM[m] = b; E0[m] = f0[pos]; E1[m] = f1[pos]; ALR[m] = r["al"]
        i = np.arange(pre_start[k], test_start[k]); i = i[np.isfinite(r["mk"][i]) & np.isfinite(r["D"][i])]; fs = np.where(np.isfinite(f0[r["sidx"][i]]), np.maximum(f0, f1)[r["sidx"][i]], np.nan)
        PX.append(pd.DataFrame({"y": YS[k][i], "sig": r["sig"][i], "mu": r["mu"], "nu": r["nu"], "D": r["D"][i], "mk": r["mk"][i], "e0": f0[i], "e1": f1[i], "earn": fs}))
        if np.isfinite(f0).any(): SER[k] = dict(y=YS[k], f0=f0, f1=f1, ts=int(test_start[k]), rows=m, pos=pos)
    PX = pd.concat(PX, ignore_index=True)
    if SER: import pickle; pickle.dump(SER, open(os.path.join(ROOT, "data", "panels", WHAT + "_series.pkl"), "wb"))   # full return series and calendar flags for tools/r78_calendar_garch.py
    np.savez_compressed(os.path.join(ROOT, "data", "panels", WHAT + "_dump.npz"), sig=SIG, mu=MU, nu=NU, alpha=ALR, beta=BE, D=DD, age=AGE.astype("int16"), sign=SGN.astype("int8"), size=SIZ, earn=EARN, sdate=SDATE.astype(str), sret=SRET, mkt_s=MS, mktz_s=MZ, bmk=BM, e0=E0, e1=E1,
                        **{"pre_" + c: PX[c].values for c in PX.columns})
    lg("wrote the per-row dump (%d test rows, %d pre-test rows)" % (n, len(PX))); sys.exit(0)
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
    if len(s) < 30: return None
    dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
def edge(Lr, Lm, mask):
    if mask.sum() < 50: return None
    return {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask)}
rng = np.random.default_rng(0)
def sb_index(T, B, p=0.1):
    idx = np.empty((B, T), int); idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T): new = rng.random(B) < p; idx[:, t] = np.where(new, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx
def lam_hat(mask, a, kind="pin", boot=True):
    """lambda minimizing the summed loss over the grid; interval from a stationary bootstrap over dates."""
    if mask.sum() < 300: return None
    y, mu, sig, nu, Dv = Y[mask], MU[mask], SIG[mask], NU[mask], DD[mask]; cols = []
    for l in GRID:
        if kind == "pin": cols.append(pin(y, cfq(a, l, Dv, sig, mu, nu), a))
        else:
            s = sig * np.sqrt(np.maximum(1 - (1 - l) * Dv, 0.02)) / np.sqrt(nu / (nu - 2)); cols.append(-(stats.t.logpdf((y - mu) / s, nu) - np.log(s)))
    M = pd.DataFrame(np.column_stack(cols), index=dates[mask]).groupby(level=0).sum().values; o = {"lambda": float(GRID[int(np.argmin(M.sum(0)))]), "rows": int(mask.sum())}
    if boot and len(M) >= 60:
        bs = GRID[np.argmin(M[sb_index(len(M), NB)].sum(1), axis=1)]; o["ci95"] = [round(float(np.quantile(bs, 0.025)), 2), round(float(np.quantile(bs, 0.975)), 2)]
    return o
def pred_breach(mask, a, lam): return round(100 * float(stats.t.cdf(stats.t.ppf(a, NU[mask]) / np.sqrt(np.maximum(1 - (1 - lam) * DD[mask], 0.02)), NU[mask]).mean()), 2)
def cell(mask, models):
    if mask.sum() < 200: return {"rows": int(mask.sum())}
    r = {"rows": int(mask.sum()), "mean_D": round(float(DD[mask].mean()), 3), "garch_sigma_over_corrected_sigma": round(float((1 / np.sqrt(1 - DD[mask])).mean()), 3)}
    for a in AL:
        zc = (Y[mask] - MU[mask]) / (SIG[mask] * np.sqrt(1 - DD[mask]))
        r[str(a)] = {"garch_t_breach_pct": round(100 * float((Y <= Q["garch_t"][:, J[a]])[mask].mean()), 2), "predicted_breach_pct_lambda0": pred_breach(mask, a, 0.0), "corrected_breach_pct_lambda0": round(100 * float((Y <= QC0[a])[mask].mean()), 2),
                     "bteg_breach_pct": round(100 * float((Y <= Q["param_bteg"][:, J[a]])[mask].mean()), 2), "body_breach_pct": round(100 * float((Y <= Q["body"][:, J[a]])[mask].mean()), 2),
                     "realized_quantile_corrected_residual": round(float(np.quantile(zc, a)), 3), "realized_quantile_garch_residual": round(float(np.quantile((Y[mask] - MU[mask]) / SIG[mask], a)), 3), "student_t_quantile": round(float(tq(a, NU[mask]).mean()), 3),
                     "edges_over_garch_t": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], mask) for m in models}}
    return r
FLEX = [m for m in ("cf0", "cf_pre", "param_bteg", "body", "sav_caviar", "neural", "body_bteg", "gjr_skewt") if m in LOSS[0.025]]

def analyze(U, full=True):
    """U: boolean mask of the universe's test rows."""
    TOP = U & (rk == 10); o = {"rows": int(U.sum()), "names": int(len(np.unique(NAME[U]))), "top_decile_rows": int(TOP.sum()), "median_beta": round(float(np.median(BE[U])), 3), "median_alpha": round(float(np.median([res[k]["al"] for k in np.unique(NAME[U])])), 3), "median_nu": round(float(np.median([res[k]["nu"] for k in np.unique(NAME[U])])), 2),
                               "median_half_life_days": round(float(np.log(0.5) / np.log(np.median(BE[U]))), 1), "top_decile_share_positive_shock": round(float((SGN[TOP] > 0).mean()), 3), "top_decile_median_shock_size": round(float(np.median(SIZ[TOP])), 2)}
    o["A_top_decile_by_shock_age"] = {lab: cell(TOP & (AGE >= lo) & (AGE <= hi), FLEX if full else ("cf0", "param_bteg", "body")) for lab, lo, hi in (AGES if full else AGES3)}
    o["A_top_decile_ages_6_15"] = cell(TOP & (AGE >= 6) & (AGE <= 15), ("cf0", "param_bteg", "body")); o["A_top_decile_by_sign"] = {"positive": cell(TOP & (SGN > 0), FLEX), "negative": cell(TOP & (SGN < 0), FLEX)}
    o["A_top_decile"] = cell(TOP, FLEX); o["A_all_days"] = {str(a): {m: edge(LOSS[a]["garch_t"], LOSS[a][m], U) for m in FLEX} for a in AL}
    o["A_top_decile_vs_bteg"] = {str(a): {m: edge(LOSS[a]["param_bteg"], LOSS[a][m], TOP) for m in FLEX if m != "param_bteg"} for a in AL}
    B = {"top_decile_pinball_0.025": lam_hat(TOP, 0.025), "top_decile_pinball_0.01": lam_hat(TOP, 0.01, boot=full), "top_decile_likelihood": lam_hat(TOP, None, "lik", boot=full),
         "positive_shock": lam_hat(TOP & (SGN > 0), 0.025, boot=full), "negative_shock": lam_hat(TOP & (SGN < 0), 0.025, boot=full)}
    if full:
        B["by_age"] = {lab: lam_hat(TOP & (AGE >= lo) & (AGE <= hi), 0.025) for lab, lo, hi in AGESL}; B["all_rows"] = lam_hat(U, 0.025); B["decile9"] = lam_hat(U & (rk == 9), 0.025); B["rows_with_D_above_0.10"] = lam_hat(U & (DD > 0.10), 0.025)
        B["by_shock_size"] = {lab: lam_hat(TOP & (SIZ >= lo) & (SIZ < hi), 0.025) for lab, lo, hi in (("under 5 SD", 0, 5), ("5 to 7 SD", 5, 7), ("7 SD and over", 7, 1e9))}
    if full:   # stability over the test period: three equal thirds of the test dates
        ud = np.sort(pd.unique(dates[U])); c1, c2 = ud[len(ud) // 3], ud[2 * len(ud) // 3]
        B["by_test_period"] = {"%s to %s" % (str(pd.Timestamp(lo).date()), str(pd.Timestamp(hi).date())): lam_hat(TOP & (dates >= lo) & (dates <= hi), 0.025) for lo, hi in ((ud[0], c1), (c1, c2), (c2, ud[-1]))}
    o["B_lambda"] = B
    if full and np.isfinite(EARN[TOP]).sum() > 2000:
        TE_, TN_ = TOP & (EARN == 1), TOP & (EARN == 0); key = lambda m_: len(set(zip(NAME[m_].tolist(), (np.where(m_)[0] - AGE[m_]).tolist())))
        o["G_earnings"] = {"names_with_dates": ecov, "share_of_trading_days_in_an_earnings_window": round(float(np.mean(ebase)), 4), "top_decile_rows_classified": int((TE_ | TN_).sum()),
                           "share_of_top_decile_rows_with_earnings_shock": round(float(TE_.sum() / (TE_ | TN_).sum()), 3), "share_of_all_rows_with_earnings_shock": round(float(np.nanmean(EARN[U])), 3),
                           "share_of_rows_with_D_above_0.35_with_earnings_shock": round(float(np.nanmean(EARN[U & (DD > 0.35)])), 3),
                           "lambda_earnings_shock": lam_hat(TE_, 0.025), "lambda_other_shock": lam_hat(TN_, 0.025), "lambda_earnings_shock_D_above_0.10": lam_hat(U & (EARN == 1) & (DD > 0.10), 0.025), "lambda_other_shock_D_above_0.10": lam_hat(U & (EARN == 0) & (DD > 0.10), 0.025),
                           "earnings_only_correction": {str(a): {"top_decile_vs_garch_t": edge(LOSS[a]["garch_t"], LOSS[a]["cf_earn"], TE_ | TN_), "all_days_vs_garch_t": edge(LOSS[a]["garch_t"], LOSS[a]["cf_earn"], U & np.isfinite(EARN)),
                                                                 "all_shock_correction_top_decile_same_rows": edge(LOSS[a]["garch_t"], LOSS[a]["cf0"], TE_ | TN_), "all_shock_correction_all_days_same_rows": edge(LOSS[a]["garch_t"], LOSS[a]["cf0"], U & np.isfinite(EARN)),
                                                                 "all_days_vs_bteg": edge(LOSS[a]["param_bteg"], LOSS[a]["cf_earn"], U & np.isfinite(EARN)), "all_days_vs_pooled_trees": edge(LOSS[a]["body"], LOSS[a]["cf_earn"], U & np.isfinite(EARN)),
                                                                 "top_decile_vs_bteg": edge(LOSS[a]["param_bteg"], LOSS[a]["cf_earn"], TE_ | TN_), "top_decile_vs_pooled_trees": edge(LOSS[a]["body"], LOSS[a]["cf_earn"], TE_ | TN_),
                                                                 "bteg_all_days_same_rows": edge(LOSS[a]["garch_t"], LOSS[a]["param_bteg"], U & np.isfinite(EARN)), "pooled_trees_all_days_same_rows": edge(LOSS[a]["garch_t"], LOSS[a]["body"], U & np.isfinite(EARN))} for a in AL},
                           "earnings_shock": cell(TE_, FLEX), "other_shock": cell(TN_, FLEX),
                           "earnings_shock_by_age": {lab: cell(TE_ & (AGE >= lo) & (AGE <= hi), ("cf0", "param_bteg", "body")) for lab, lo, hi in AGES3}, "other_shock_by_age": {lab: cell(TN_ & (AGE >= lo) & (AGE <= hi), ("cf0", "param_bteg", "body")) for lab, lo, hi in AGES3}}
    if full:
        Cc = {}
        for m in ("param_bteg", "body", "sav_caviar", "neural", "body_bteg", "gjr_skewt", "engine"):
            if m not in Q: continue
            rho2 = ((Q[m][:, J[0.025]] - MU) / (Q["garch_t"][:, J[0.025]] - MU))[TOP] ** 2; x = DD[TOP]; ok = np.isfinite(rho2) & (rho2 < 9); yv = 1 - rho2[ok]; x = x[ok]
            b1, b0 = np.polyfit(x, yv, 1); Cc[m] = {"slope_discard_share": round(float(b1), 3), "intercept": round(float(b0), 3), "slope_no_intercept": round(float((x * yv).sum() / (x * x).sum()), 3),
                                                   "r2": round(float(np.corrcoef(x, yv)[0, 1] ** 2), 3), "spearman": round(float(stats.spearmanr(x, yv)[0]), 3)}
        o["C_what_each_model_discards"] = Cc
        o["D_duration_by_beta_tercile"] = {g: {"median_beta": round(float(np.median(BE[U & (BT == g)])), 3), "half_life_days": round(float(np.log(0.5) / np.log(np.median(BE[U & (BT == g)]))), 1),
                                               "by_age": {lab: (lambda mk_: {"rows": int(mk_.sum()), "mean_D": round(float(DD[mk_].mean()), 3) if mk_.sum() else None,
                                                                              "garch_t_breach_0.025": round(100 * float((Y <= Q["garch_t"][:, J[0.025]])[mk_].mean()), 2) if mk_.sum() >= 200 else None,
                                                                              "predicted_lambda0": pred_breach(mk_, 0.025, 0.0) if mk_.sum() >= 200 else None})(TOP & (BT == g) & (AGE >= lo) & (AGE <= hi)) for lab, lo, hi in AGES3}}
                                           for g in ("low beta", "mid beta", "high beta") if (U & (BT == g)).sum() > 0}
        E = {}
        for lo, hi in zip(DBINS[:-1], DBINS[1:]):
            mk_ = U & (DD >= lo) & (DD < hi)
            if mk_.sum() < 200: E["%.2f-%.2f" % (lo, hi)] = {"rows": int(mk_.sum())}; continue
            E["%.2f-%.2f" % (lo, hi)] = {"rows": int(mk_.sum()), "share_in_top_score_decile": round(float((rk[mk_] == 10).mean()), 3),
                                         **{"breach_%s" % a: {"garch_t": round(100 * float((Y <= Q["garch_t"][:, J[a]])[mk_].mean()), 2), "predicted_lambda0": pred_breach(mk_, a, 0.0), "predicted_lambda_pre": pred_breach(mk_, a, LAM_PRE[0.025]),
                                                              "corrected_lambda0": round(100 * float((Y <= QC0[a])[mk_].mean()), 2), "bteg": round(100 * float((Y <= Q["param_bteg"][:, J[a]])[mk_].mean()), 2)} for a in AL},
                                         "edges_over_garch_t_0.025": {m: edge(LOSS[0.025]["garch_t"], LOSS[0.025][m], mk_) for m in ("cf0", "param_bteg", "body")}}
        o["E_dose_response_all_rows"] = E
        rd = np.full(n, -1); rd[U] = pd.qcut(pd.Series(DD[U]).rank(method="first"), 10, labels=False).values + 1; TD = rd == 10
        o["F_D_as_score"] = {"overlap_top_D_with_top_kurtosis": round(float((TD & TOP).sum() / max(TD.sum(), 1)), 3), "D_threshold_top_decile": round(float(DD[TD].min()), 3),
                             **{str(a): {"top_D_decile": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], TD) for m in FLEX}, "top_kurtosis_decile": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], TOP) for m in FLEX},
                                         "top_D_not_top_kurtosis": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], TD & ~TOP) for m in ("cf0", "param_bteg", "body")},
                                         "top_kurtosis_not_top_D": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], TOP & ~TD) for m in ("cf0", "param_bteg", "body")}} for a in AL}}
    return o

ALLM = np.ones(n, bool)
OUT = {"universe": WHAT, "note": "edge_pct > 0 and DM > 0: the named model has the lower pinball loss than the reference. cf0 = GARCH-t with the largest shock's excess variance removed (lambda = 0, no free parameter); "
                                 "cf_pre = the same with lambda estimated on pre-test rows. D = share of today's GARCH variance due to the largest shock of the last 63 days being larger than an ordinary day.",
       "predictions_written_before_run": "see the script header (P1-P7)", "lambda_pre_test": {str(a): v for a, v in LAM_PRE.items()}, "pre_test_rows": int(len(PRE)), "bootstrap_reps": NB, "refit_check_max_rel_gap": chk,
       "all": analyze(ALLM), "groups": {}}
lg("whole universe done")
for gname, gmap in GROUPS.items():
    G = np.array([gmap[k] for k in NAME]); OUT["groups"][gname] = {}
    for g in sorted(set(gmap.values())):
        U = G == g; nn = len(np.unique(NAME[U]))
        if nn < 5: OUT["groups"][gname][g] = {"names": nn, "skipped": "fewer than 5 names"}; continue
        OUT["groups"][gname][g] = analyze(U, full=False)
    lg("groups by %s done" % gname)
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
np.savez_compressed(os.path.join(ROOT, "data", "panels", WHAT + "_shock.npz"), D=DD.astype("float32"), age=AGE.astype("int16"), sign=SGN.astype("int8"), size=SIZ.astype("float32"), earn=EARN.astype("float32"), beta=BE.astype("float32"),
                    **{"cf0__%s" % a: QC0[a].astype("float32") for a in AL}, **{"cf_pre__%s" % a: QCP[a].astype("float32") for a in AL})   # per-row file for tools/r75_conditional_tests.py, not committed
# ---------------------------------------------------------------- print
f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"]); fl = lambda b: "n/a" if b is None else "%.2f %s" % (b["lambda"], b.get("ci95", ""))
A_ = OUT["all"]; print("\n######## %s: %d names, %d rows; median beta %.3f (half-life %.1f days); top decile: %d rows, largest shock %.1f SD, %.0f%% positive; pre-test lambda %s" % (
    WHAT, A_["names"], A_["rows"], A_["median_beta"], A_["median_half_life_days"], A_["top_decile_rows"], A_["top_decile_median_shock_size"], 100 * A_["top_decile_share_positive_shock"], LAM_PRE))
print("A. top decile by shock age (2.5%): rows | D | GARCH-t breach, predicted (lambda=0), corrected | corrected-residual quantile vs t | edges over GARCH-t: cf0, bteg, body")
for lab, c in A_["A_top_decile_by_shock_age"].items():
    if "0.025" not in c: print("  %-6s %6d rows" % (lab, c["rows"])); continue
    x = c["0.025"]; e = x["edges_over_garch_t"]; print("  %-6s %6d | %.3f | %.2f%% %.2f%% %.2f%% | %.3f vs %.3f | %s  %s  %s | 1%%: %.2f%% pred %.2f%%" % (lab, c["rows"], c["mean_D"], x["garch_t_breach_pct"], x["predicted_breach_pct_lambda0"], x["corrected_breach_pct_lambda0"],
          x["realized_quantile_corrected_residual"], x["student_t_quantile"], f(e["cf0"]), f(e["param_bteg"]), f(e["body"]), c["0.01"]["garch_t_breach_pct"], c["0.01"]["predicted_breach_pct_lambda0"]))
for a in ("0.025", "0.01"):
    print("   top decile edges over GARCH-t at %s:" % a, {m: f(e) for m, e in A_["A_top_decile"][a]["edges_over_garch_t"].items()}); print("   all days:", {m: f(e) for m, e in A_["A_all_days"][a].items()}); print("   top decile vs Beta-t-EGARCH:", {m: f(e) for m, e in A_["A_top_decile_vs_bteg"][a].items()})
B = A_["B_lambda"]; print("B. lambda: pinball 2.5%% %s | pinball 1%% %s | likelihood %s | positive %s | negative %s | decile 9 %s | D>0.10 %s" % tuple(fl(B[k]) for k in ("top_decile_pinball_0.025", "top_decile_pinball_0.01", "top_decile_likelihood", "positive_shock", "negative_shock", "decile9", "rows_with_D_above_0.10")))
sg = A_["A_top_decile_by_sign"]; print("   by sign of the shock (rows, mean D, GARCH-t 2.5%% breach, edges cf0 / bteg / body): %s" % {k: (c["rows"], c.get("mean_D"), c["0.025"]["garch_t_breach_pct"], f(c["0.025"]["edges_over_garch_t"]["cf0"]), f(c["0.025"]["edges_over_garch_t"]["param_bteg"]), f(c["0.025"]["edges_over_garch_t"]["body"])) for k, c in sg.items() if "0.025" in c})
print("   all rows:", fl(B["all_rows"]), "| by age:", {k: fl(v) for k, v in B["by_age"].items()}, "| by size:", {k: fl(v) for k, v in B["by_shock_size"].items()})
print("   by test period:", {k: fl(v) for k, v in B["by_test_period"].items()})
print("C. discard share (slope, intercept, R2):", {m: (v["slope_discard_share"], v["intercept"], v["r2"]) for m, v in A_["C_what_each_model_discards"].items()})
print("D. duration by beta tercile (2.5% breach realized/predicted by age):", {g: (v["median_beta"], v["half_life_days"], {k: (x["garch_t_breach_0.025"], x["predicted_lambda0"]) for k, x in v["by_age"].items()}) for g, v in A_["D_duration_by_beta_tercile"].items()})
print("E. dose-response, all rows: D bin | rows | in top decile | 2.5% breach GARCH-t, predicted lambda0, predicted lambda_pre, corrected | 1% breach, predicted | edges cf0, bteg, body")
for b, c in A_["E_dose_response_all_rows"].items():
    if "breach_0.025" not in c: continue
    x = c["breach_0.025"]; x1 = c["breach_0.01"]; e = c["edges_over_garch_t_0.025"]; print("  %s %7d %.2f | %.2f%% %.2f%% %.2f%% %.2f%% | %.2f%% %.2f%% | %s %s %s" % (b, c["rows"], c["share_in_top_score_decile"], x["garch_t"], x["predicted_lambda0"], x["predicted_lambda_pre"], x["corrected_lambda0"], x1["garch_t"], x1["predicted_lambda0"], f(e["cf0"]), f(e["param_bteg"]), f(e["body"])))
F = A_["F_D_as_score"]; print("F. D as score: overlap %.2f, threshold D %.3f | 2.5%% top D-decile:" % (F["overlap_top_D_with_top_kurtosis"], F["D_threshold_top_decile"]), {m: f(e) for m, e in F["0.025"]["top_D_decile"].items()},
                              "| D only:", {m: f(e) for m, e in F["0.025"]["top_D_not_top_kurtosis"].items()}, "| kurtosis only:", {m: f(e) for m, e in F["0.025"]["top_kurtosis_not_top_D"].items()})
if "G_earnings" in A_:
    G_ = A_["G_earnings"]; print("G. earnings: %d names with dates; %.1f%% of days in a window; top-decile rows with an earnings shock %.1f%% (all rows %.1f%%, D>0.35 rows %.1f%%) | lambda earnings %s, other %s | D>0.10: earnings %s, other %s" % (
        G_["names_with_dates"], 100 * G_["share_of_trading_days_in_an_earnings_window"], 100 * G_["share_of_top_decile_rows_with_earnings_shock"], 100 * G_["share_of_all_rows_with_earnings_shock"], 100 * G_["share_of_rows_with_D_above_0.35_with_earnings_shock"],
        fl(G_["lambda_earnings_shock"]), fl(G_["lambda_other_shock"]), fl(G_["lambda_earnings_shock_D_above_0.10"]), fl(G_["lambda_other_shock_D_above_0.10"])))
    for a in ("0.025", "0.01"): print("   earnings-only correction at %s:" % a, {k: f(v) for k, v in G_["earnings_only_correction"][a].items()})
    for k in ("earnings_shock", "other_shock"):
        c = G_[k]; print("   %-15s rows %d, D %.3f, GARCH-t breach %.2f%% (predicted %.2f%%), edges:" % (k, c["rows"], c["mean_D"], c["0.025"]["garch_t_breach_pct"], c["0.025"]["predicted_breach_pct_lambda0"]), {m: f(e) for m, e in c["0.025"]["edges_over_garch_t"].items()})
        print("      by age (rows, breach, predicted, corrected-residual quantile):", {a_: (x["rows"], x["0.025"]["garch_t_breach_pct"], x["0.025"]["predicted_breach_pct_lambda0"], x["0.025"]["realized_quantile_corrected_residual"]) for a_, x in G_[k + "_by_age"].items() if "0.025" in x})
for gname, gs in OUT["groups"].items():
    print("--- groups by %s: names | lambda (2.5%%) | positive, negative | breach by age (realized/predicted) | ages 6-15 | top-decile edges cf0, bteg, body | all days cf0" % gname)
    for g, v in gs.items():
        if "skipped" in v: print("  %-40s %s" % (g, v["skipped"])); continue
        print("  %-40s %3d | %s | %s, %s | %s | %s | %s %s %s | %s" % (g[:40], v["names"], fl(v["B_lambda"]["top_decile_pinball_0.025"]), fl(v["B_lambda"]["positive_shock"]), fl(v["B_lambda"]["negative_shock"]),
              {k: (c["0.025"]["garch_t_breach_pct"], c["0.025"]["predicted_breach_pct_lambda0"]) if "0.025" in c else None for k, c in v["A_top_decile_by_shock_age"].items()}, v["A_top_decile_ages_6_15"]["0.025"]["garch_t_breach_pct"] if "0.025" in v["A_top_decile_ages_6_15"] else None,
              *[f(v["A_top_decile"]["0.025"]["edges_over_garch_t"][m]) if "0.025" in v["A_top_decile"] else "n/a" for m in ("cf0", "param_bteg", "body")], f(v["A_all_days"]["0.025"]["cf0"])))
lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
