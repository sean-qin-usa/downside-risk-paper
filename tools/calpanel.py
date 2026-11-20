# calpanel.py -- one per-row table for a stock panel with the earnings calendar, built once and read by r83-r85.
#
# For every name of the panel (the names of data/panels/<panel>_calrows.npz) and every day from 126 days after the
# first known announcement: the return, the reaction-day flags (r0 = reaction day, r1 = the day after, ds = trading
# days since the last reaction day, capped at 63), the GARCH-t fit, the masked calendar GARCH-t fit (r78), the share D
# of the day's variance that comes from the largest recent shock and that shock's market-wide share w (r82), and the
# five inputs of the pooled trees computed on the masked scale. Nothing is scored here.
#   GBC_PROJ=<crsp dir> python tools/calpanel.py canon        python tools/calpanel.py us_largecap
# Output: data/panels/<panel>_calcache.pkl (not committed).
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "tools"))
import r82_two_fix_garch as R82   # cal_flags, shock_D; it imports panel_study and r78
P, R78, C = R82.P, R82.R78, R82.C; ZX = P.ZX
path = lambda what: os.path.join(ROOT, "data", "panels", what + "_calcache.pkl")
load = lambda what: pd.read_pickle(path(what))

def work(job):
    key, y, dates, loo, r0, r1, ne, ntr, mode = job
    try:
        n = len(y); pr = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params
        mu, nu, om, al, be = float(pr["mu"]), float(pr["nu"]), float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"])
        if mode == "manual":
            e = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:ne])
            for k in range(1, n): s2[k] = max(om + al * e[k - 1] ** 2 + be * s2[k - 1], 1e-8)
            sig = np.sqrt(s2)
        else: sig = np.asarray(arch_model(y, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
        _, fm = R78.fit_one((key, y, r0, r1, ne, [mu, 0.0, al, be, nu]))
        if fm is None: return key, "fewer than 750 pre-test days with a calendar"
        k0 = fm["k0"]; th = fm["masked"]["th"]; mm, mal, mbe, mnu = float(th[0]), float(th[2]), float(th[3]), float(th[4]); ms2 = np.full(n, np.nan); ms2[k0:] = fm["masked"]["s2"]; mk = (r0 == 1) | (r1 == 1)
        zm = (y - mm) / np.sqrt(ms2); Dm = np.full(n, np.nan); sm = np.zeros(n, int); d_, s_ = R82.shock_D(zm[k0:], ms2[k0:], mal, mbe, mk[k0:]); Dm[k0:] = d_; sm[k0:] = s_ + k0
        ok = np.isfinite(loo[:ne]); b = float(np.polyfit(loo[:ne][ok], y[:ne][ok], 1)[0]) if ok.sum() > 250 else np.nan
        wm = np.nan_to_num(np.clip(b * loo[sm] / np.where(np.abs(y[sm] - mm) > 1e-9, y[sm] - mm, np.nan), 0, 1))
        last = np.maximum.accumulate(np.where(r0 == 1, np.arange(n), -1)); lastp = np.concatenate([[-1], last[:-1]]); ds = np.minimum(np.where(lastp >= 0, np.arange(n) - lastp, 63), 63).astype(float)
        f = C.feats(y, np.where(np.isfinite(ms2), np.sqrt(ms2), 1.0), mm); i = np.arange(k0 + 126, n)
        df = pd.DataFrame({"name": key, "date": dates[i], "idx": i, "y": y[i], "r0": r0[i], "r1": r1[i], "ds": ds[i], "sig": sig[i], "mu": mu, "nu": nu, "msig": np.sqrt(ms2[i]), "mmu": mm, "mnu": mnu, "Dm": Dm[i], "wm": wm[i],
                           "test": i >= ne, "train": i < ntr, **{c: f[c].values[i] for c in ZX}})
        return key, df
    except Exception as e:
        return key, "failed: %s" % str(e)[:100]

if __name__ == "__main__":
    WHAT = sys.argv[1]; t0 = time.time(); Dd = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_calrows.npz")); names = list(pd.unique(Dd["name"].astype(str))); SER = {}
    if WHAT == "canon":
        rr = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "crsp_panel_returns.csv"), dtype={"permno": "int32"}); rr["date"] = pd.to_datetime(rr["date"]); rr["ret"] = pd.to_numeric(rr["ret"], errors="coerce") * 100.0
        for k in names:
            g = rr[rr.permno == int(k)].sort_values("date"); y = g["ret"].values.astype(float); a = pd.read_csv(os.path.join(ROOT, "data", "panels", "earnings_ibes", k + ".csv"))["announcement"].astype(str)
            SER[k] = (y, pd.DatetimeIndex(g["date"].values), int(len(y) * 0.6), int(int(len(y) * 0.6) * 0.75), a[a.str.len() >= 16], "manual")
    else:
        for k in names:
            s = pd.read_csv(os.path.join(P.DDIR, k.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]; s = s[(s > 0) & (s.index <= P.END)]; r = (s.pct_change() * 100).dropna(); r = r[np.isfinite(r)]
            SER[k] = (r.values.astype(float), r.index, int((r.index < P.TEST0).sum()), int((r.index < P.CAL0).sum()), pd.read_csv(os.path.join(ROOT, "data", "panels", "earnings", k + ".csv"))["announcement"].astype(str), "arch")
    W = pd.DataFrame({k: pd.Series(v[0], index=v[1]) for k, v in SER.items()}); sm_, ct_ = W.sum(1), W.count(1); jobs = []
    for k in names:
        y, dates, ne, ntr, a, mode = SER[k]; c = ct_.reindex(dates).values; loo = np.where(c >= 6, (sm_.reindex(dates).values - y) / np.maximum(c - 1, 1), np.nan); r0, r1 = R82.cal_flags(a, dates); jobs.append((k, y, dates, loo, r0, r1, ne, ntr, mode))
    with Pool(int(os.environ.get("CALPANEL_WORKERS", "8"))) as pool: res = pool.map(work, jobs, chunksize=1)
    bad = {k: v for k, v in res if not isinstance(v, pd.DataFrame)}; A = pd.concat([v for k, v in res if isinstance(v, pd.DataFrame)], ignore_index=True)
    key = pd.MultiIndex.from_arrays([A["name"].values, A["date"].dt.strftime("%Y-%m-%d").values]); pos = pd.Series(np.arange(len(Dd["y"])), index=pd.MultiIndex.from_arrays([Dd["name"].astype(str), Dd["date"].astype(str)])).reindex(key).values
    has = np.isfinite(pos); pi = np.where(has, pos, 0).astype(int); A["rk"] = np.where(has, Dd["rk"][pi], -1); A["U"] = has & Dd["U"][pi]   # score decile and the common-rows flag of the calendar run (test rows only)
    for k in Dd.files:
        if k.startswith("q__") and ("calendar" in k or k == "q__GARCH-t"):
            for j, a_ in enumerate(("0.01", "0.025")): v = np.full(len(A), np.nan); v[has] = Dd[k][pi[has], j]; A["%s__%s" % (k[3:], a_)] = v
    A.to_pickle(path(WHAT)); print("%s: %d names, %d rows (%d test), left out %s, %.0fs" % (WHAT, A["name"].nunique(), len(A), int(A.test.sum()), bad, time.time() - t0))
