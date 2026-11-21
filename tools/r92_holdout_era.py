# r92_holdout_era.py -- THE EQUATION OF r83 IN A DIFFERENT DECADE: 200 CRSP STOCKS, 2000-2013.
#
# Everything in r79-r91 comes from two panels that end in 2024 and 2026. This is a panel none of it has seen: the 200
# stocks of holdout_panel_2000_2013.csv (the paper's earlier-era sample), with I/B/E/S announcement dates and times
# (code/panels/download_ibes_dates.py). Per name, the first 60% of days estimates GARCH-t (the CRSP panel's protocol) and
# the last 40% is the test, which here runs from about mid-2008 to 2013 and contains the financial crisis. A name is
# used if it has at least 44 announcements. No flexible model is fitted: the test is of the calendar and the equation.
#   (1) GARCH-t's breach rate on the reaction day and the day after.
#   (2) The gain of two calendar multipliers chosen on pre-test rows.
#   (3) V of r83, from pre-test rows only, against that gain; the implied against the chosen multiplier.
#   (4) Portfolios of N names (r83, part C), with the plain GARCH-t variance as the ordinary variance.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r79-r91 on the 2014-2026 panels; of this panel
# only that all 200 names have an I/B/E/S record with a median of 58 announcements.
#  E1. GARCH-t's 2.5% VaR is breached on 15% to 30% of reaction days, and its 1% VaR on 10% to 22%.
#  E2. The calendar multipliers gain 4% to 9% at 2.5% and 8% to 16% at 1% over all days (DM above 5).
#  E3. V is within 1.0 point of that gain at 2.5% and within 2.0 points at 1%; the implied reaction-day multiplier is
#      within 0.3 of the one chosen on pre-test rows.
#  E4. Portfolios: the gain falls with N, is between 0.3% and 3% at N = 10 and below 0.6% at N = 50, and V is within
#      0.5 points or a factor of 1.5 of it at every N.
# Usage: GBC_PROJ=<crsp dir> python tools/r92_holdout_era.py
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "tools")); AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r92"); os.makedirs(OUTD, exist_ok=True)
sys.argv = sys.argv[:1] + ["holdout"]; import r83_information_value as R83
tq, pin, EL = R83.tq, R83.pin, R83.EL
def cal_flags(a, dates):
    n = len(dates); d0 = pd.DatetimeIndex(a.str[:10]); hr = a.str[11:13].astype(int).values + a.str[14:16].astype(int).values / 60.0
    p = np.searchsorted(dates.values, d0.values); inr = p < n; same = np.zeros(len(p), bool); same[inr] = dates.values[p[inr]] == d0.values[inr]; p = p + ((hr >= 16) & same); p = p[p < n]
    r0 = np.zeros(n); r0[p] = 1; r1 = np.zeros(n); r1[p[p + 1 < n] + 1] = 1; return r0, r1
def work(job):
    k, y, dates = job; fn = os.path.join(ROOT, "data", "panels", "earnings_ibes_holdout", "%d.csv" % k)
    if not os.path.exists(fn): return None
    a = pd.read_csv(fn)["announcement"].astype(str); a = a[a.str.len() >= 16]
    if len(a) < 44: return None
    n = len(y); ne = int(n * 0.6); pr = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params; mu, nu = float(pr["mu"]), float(pr["nu"]); e = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:ne])
    for i in range(1, n): s2[i] = max(float(pr["omega"]) + float(pr["alpha[1]"]) * e[i - 1] ** 2 + float(pr["beta[1]"]) * s2[i - 1], 1e-8)
    r0, r1 = cal_flags(a, dates); i = np.arange(126, n); return pd.DataFrame({"name": k, "date": dates[i], "y": y[i], "sig": np.sqrt(s2[i]), "mu": mu, "nu": nu, "r0": r0[i], "r1": r1[i], "test": i >= ne})
if __name__ == "__main__":
    rr = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "holdout_panel_2000_2013.csv"), dtype={"permno": "int32"}); rr["date"] = pd.to_datetime(rr["date"]); rr["ret"] = pd.to_numeric(rr["ret"], errors="coerce") * 100.0; rr = rr.dropna(subset=["ret"])
    jobs = [(int(k), g.sort_values("date")["ret"].values.astype(float), pd.DatetimeIndex(g.sort_values("date")["date"].values)) for k, g in rr.groupby("permno")]
    with Pool(12) as pool: A = pd.concat([d for d in pool.map(work, jobs, chunksize=2) if d is not None], ignore_index=True)
    typ = np.where(A.r0 == 1, "reaction day", np.where(A.r1 == 1, "day after", "other days")); pre = ~A.test.values; te = A.test.values; z = ((A.y - A.mu) / A.sig).values; y, mu, sg, nu = A.y.values, A.mu.values, A.sig.values, A.nu.values; dates = pd.DatetimeIndex(A.date.values); NU = A.groupby("name")["nu"].first().values
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    par = {}; TY = ("reaction day", "day after", "other days")
    for d in TY: f = pre & (typ == d); nu_, _, tau_ = stats.t.fit(z[f], floc=0); par[d] = {"tau": float(tau_), "nu": float(nu_), "p": float((typ[te] == d).mean()), "m": float(sg[f].mean() / sg[pre].mean()), "variance_ratio": float(np.mean(z[f] ** 2)), "rows": int(f.sum())}
    cst = {d: float(optimize.minimize_scalar(lambda c: EL(c * tq(0.025, NU), par[d]["tau"], par[d]["nu"], 0.025).mean(), bounds=(0.5, 8.0), method="bounded").x) for d in TY[:2]}
    CG = np.round(np.arange(1.0, 6.001, 0.1), 2); b = tq(0.025, nu); cemp = {d: float(CG[int(np.argmin([pin(y[pre & (typ == d)], mu[pre & (typ == d)] + c * sg[pre & (typ == d)] * b[pre & (typ == d)], 0.025).sum() for c in CG]))]) for d in TY[:2]}
    mult = np.where(typ == "reaction day", cemp["reaction day"], np.where(typ == "day after", cemp["day after"], 1.0))
    OUT = {"names": int(A["name"].nunique()), "rows_test": int(te.sum()), "test_from": str(dates[te].min().date()), "test_to": str(dates[te].max().date()), "predictions_written_before_run": "see the script header (E1-E4)", "day_types_pre_test": par, "implied_multipliers": {d: round(c, 2) for d, c in cst.items()}, "multipliers_chosen_on_pre_test_rows": cemp,
           "mean_squared_residual_test": {d: round(float(np.mean(z[te & (typ == d)] ** 2)), 2) for d in TY}}
    for a in AL:
        q = tq(a, NU); V = 100 * sum(par[d]["p"] * par[d]["m"] * (EL(q, par[d]["tau"], par[d]["nu"], a) - EL(cst[d] * q, par[d]["tau"], par[d]["nu"], a)).mean() for d in cst) / sum(par[d]["p"] * par[d]["m"] * EL(q, par[d]["tau"], par[d]["nu"], a).mean() for d in TY)
        Lb = pin(y, mu + sg * tq(a, nu), a); Li = pin(y, mu + sg * mult * tq(a, nu), a); h = y <= mu + sg * tq(a, nu); hi = y <= mu + sg * mult * tq(a, nu)
        OUT[str(a)] = {"V_pct": round(float(V), 2), "realized_pct": round(100 * float((Lb - Li)[te].sum() / Lb[te].sum()), 2), "DM": dm(Lb - Li, te), "breach_garch_t_pct": {d: round(100 * float(h[te & (typ == d)].mean()), 2) for d in TY}, "breach_with_multipliers_pct": {d: round(100 * float(hi[te & (typ == d)].mean()), 2) for d in TY},
                       "share_of_garch_t_breaches_on_reaction_days_pct": round(100 * float(h[te & (typ == "reaction day")].sum() / h[te].sum()), 1)}
    # portfolios (names with the full history)
    full = A.groupby("name").size(); keep = full[full == full.max()].index; B = A[A["name"].isin(keep)]; R = B.pivot(index="date", columns="name", values="y"); piv = lambda c: B.pivot(index="date", columns="name", values=c).reindex(R.index).values
    S2 = piv("sig") ** 2; F0 = (piv("r0") == 1).astype(float); F1 = ((piv("r1") == 1) & (F0 == 0)).astype(float); tst = B.groupby("date")["test"].first().reindex(R.index).values.astype(bool); um = (piv("y") - piv("mu")) ** 2 / S2; k0 = float(um[~tst][F0[~tst] == 1].mean()); k1 = float(um[~tst][F1[~tst] == 1].mean()); M = R.shape[1]
    R83.G.update(R=R.values, S2=S2, F0=F0, F1=F1, pre=~tst, k0=k0, k1=k1, k0t=float(um[tst][F0[tst] == 1].mean()), k1t=float(um[tst][F1[tst] == 1].mean())); SIZES = (1, 5, 10, 20, 50)
    with Pool(12) as pool: res = [r for r in pool.map(R83.port, [(N, s) for N in SIZES for s in (range(M) if N == 1 else range(40))], chunksize=4) if r is not None]
    OUT["portfolios"] = {"names": M, "variance_ratio_pre_test": [round(k0, 2), round(k1, 2)], "by_N": {str(N): {str(a): {"gain_pct": round(float(np.mean([r[str(a)]["gain"] for r in res if r["N"] == N])), 2), "V_pct": round(float(np.mean([r[str(a)]["V"] for r in res if r["N"] == N])), 2)} for a in AL} for N in SIZES}}
    R83.check_table(res, SIZES, "holdout", {"variance_ratio_reaction_day_and_day_after": {"pre-test days": [round(k0, 2), round(k1, 2)], "test days": [round(R83.G["k0t"], 2), round(R83.G["k1t"], 2)]}})   # consistency check added 2026-11-20 (r98)
    json.dump(OUT, open(os.path.join(OUTD, "holdout.json"), "w"), indent=1); print(json.dumps({k: v for k, v in OUT.items() if k != "day_types_pre_test"})[:2300]); print({d: (round(v["tau"], 2), round(v["nu"], 1), round(v["variance_ratio"], 1), v["rows"]) for d, v in par.items()})
