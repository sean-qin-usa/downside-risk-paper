# r67_refit_benchmarks.py -- AUDIT FIX RUN for H4 and H5 (docs/AUDIT_2026-11.md). Read-only with respect to the paper's
# jobs: it refits three benchmarks properly and re-scores them against the accuracy layer's saved per-row losses
# (job_bench_all.py --dump-rows), on the canonical 200 names and 221,600 test rows.
#
# What the paper's job does (job_bench_all.py) and what this does instead:
#   GAS (Patton-Ziegel-Chen one-factor)  trailing 1,200-day window; scales a, b FIXED at the window's empirical VaR/ES;
#                                        Nelder-Mead maxiter 120, xatol 1e-3; no convergence check.
#                                    ->  full estimation window [:sp]; a and b ESTIMATED (a < 0, b = a(1+exp(d)), omega = 0
#                                        as PZC normalise it); the paper's three starts plus three more; maxiter 3000,
#                                        xatol 1e-5; convergence logged.
#   Taylor ES-CAViaR                     trailing 1,200-day window; maxiter 200.
#                                    ->  full window; six starts; maxiter 3000; convergence logged.
#   SAV-CAViaR (eleven levels)           one start, maxiter 300.
#                                    ->  eight starts (the paper's and seven seeded perturbations), maxiter 1500.
# The "as published" fits are re-run here too, to check that this script reproduces the dumped losses before anything
# is compared.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-17):
#  P1. Refitted GAS improves by more than 0.03 FZ0 at 1% (published 2.232) and the accuracy layer still beats it at
#      both levels (DM > 2).
#  P2. Refitted Taylor improves; the accuracy layer's win at 1% (published DM 3.8) falls below 1.96; at 2.5% Taylor is
#      not significantly better than the accuracy layer (DM > -1.96).
#  P3. Refitted SAV-CAViaR lowers its pinball by less than 0.2%, and its top-mk63-decile lead over the accuracy layer
#      is significant (DM < -1.96; canonical file -1.96).
# Usage: GBC_PROJ=<panel dir> python tools/r67_refit_benchmarks.py <bench_all_rows.npz>
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import optimize
P = os.environ["GBC_PROJ"]; DUMP = sys.argv[1]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "paper", "diagnostics", "r67_refit_benchmarks.json")
ALPHAS = [0.01, 0.025]; TAUS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)

def pin(y, q, t): d = y - q; return np.where(d >= 0, t * d, (t - 1) * d)
def fz0(r, v, e, a):
    v = np.minimum(v, -1e-8); e = np.minimum(e, v); hit = (r <= v).astype(float)
    return -(1.0 / (a * e)) * hit * (v - r) + v / e + np.log(-e) - 1.0
def gas_filter(y, a, b, om_, be_, ga_, k0, alpha):
    n = len(y); k = np.empty(n); k[0] = k0; v = np.empty(n); e = np.empty(n)
    for i in range(n):
        ex = math.exp(min(k[i], 6.0)); v[i] = a * ex; e[i] = b * ex
        if i + 1 < n:
            hit = 1.0 if y[i] <= v[i] else 0.0; H = (1.0 / e[i]) * ((1.0 / alpha) * hit * y[i] - e[i]); k[i + 1] = om_ + be_ * k[i] + ga_ * H
    return v, e
def taylor_filter(y, b1, b2, b3, delta, v0):
    n = len(y); v = np.empty(n); e = np.empty(n); v[0] = v0; mult = 1.0 + math.exp(min(delta, 4.0))
    for i in range(n):
        if i > 0: v[i] = b1 + b2 * v[i - 1] + b3 * abs(y[i - 1])
        e[i] = v[i] * mult
    return v, e
def nm(obj, x0, maxiter, xatol, fatol):
    try:
        r = optimize.minimize(obj, x0=x0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
        return r if np.isfinite(r.fun) else None
    except Exception: return None
def best_of(rs):
    rs = [r for r in rs if r is not None]; return min(rs, key=lambda r: r.fun) if rs else None

def one_name(args):
    pn, y, nsel = args; sp = int(len(y) * 0.6); out = {"pn": pn}; rng = np.random.default_rng(int(pn))
    for a in ALPHAS:
        for tag, tr, mi, xa, fa in (("pub", y[max(0, sp - 1200):sp], None, 1e-3, 1e-4), ("fix", y[:sp], 3000, 1e-5, 1e-7)):
            qa = float(np.quantile(tr, a)); ea = float(np.mean(tr[tr <= qa])) if (tr <= qa).any() else qa * 1.2; eb = min(ea, qa * 1.05)
            # ---- GAS
            if tag == "pub":
                def obj(th):
                    om_, be_, ga_ = th
                    if not (0.0 <= be_ <= 0.999): return 1e6
                    v, e = gas_filter(tr, qa, eb, om_, be_, ga_, 0.0, a); L = fz0(tr, v, e, a); return float(np.mean(L)) if np.isfinite(L).all() else 1e6
                b = best_of([nm(obj, x0, 120, xa, fa) for x0 in ([0.0, 0.95, 0.02], [0.0, 0.90, 0.05], [0.0, 0.80, 0.10])])
                v, e = gas_filter(y, qa, eb, *b.x, 0.0, a)
            else:
                d0 = math.log(max(eb / qa - 1.0, 1e-3))
                def obj(th):
                    la, dd, be_, ga_ = th
                    if not (0.0 <= be_ <= 0.999) or abs(la) > 5 or dd > 3: return 1e6
                    aa = -math.exp(la); bb = aa * (1.0 + math.exp(dd))
                    v, e = gas_filter(tr, aa, bb, 0.0, be_, ga_, 0.0, a); L = fz0(tr, v, e, a); return float(np.mean(L)) if np.isfinite(L).all() else 1e6
                la0 = math.log(-qa)
                starts = [[la0, d0, be_, ga_] for be_, ga_ in ((0.95, 0.02), (0.90, 0.05), (0.80, 0.10), (0.98, 0.01), (0.97, 0.03), (0.60, 0.15))]
                b = best_of([nm(obj, x0, mi, xa, fa) for x0 in starts])
                aa = -math.exp(b.x[0]); v, e = gas_filter(y, aa, aa * (1.0 + math.exp(b.x[1])), 0.0, b.x[2], b.x[3], 0.0, a)
                out["gas_conv_%s" % a] = bool(b.success); out["gas_ins_%s" % a] = float(b.fun)
            out["gas_%s_%s" % (tag, a)] = (v[-nsel:], e[-nsel:])
            # ---- Taylor
            mult0 = max(ea / qa, 1.02) if qa < 0 else 1.2; dl0 = math.log(max(mult0 - 1.0, 1e-3))
            def objt(th):
                b1, b2, b3, dl = th
                if not (0.0 <= b2 <= 0.999): return 1e6
                v, e = taylor_filter(tr, b1, b2, b3, dl, qa)
                if not np.isfinite(v).all(): return 1e6
                L = fz0(tr, v, e, a); return float(np.mean(L)) if np.isfinite(L).all() else 1e6
            st = [[qa * 0.1, 0.9, -0.05, dl0], [qa * 0.05, 0.8, -0.10, dl0], [qa * 0.2, 0.85, -0.02, dl0]]
            if tag == "fix": st += [[qa * 0.02, 0.95, -0.03, dl0], [qa * 0.3, 0.6, -0.20, dl0], [qa * 0.1, 0.9, -0.05, dl0 + 0.5]]
            b = best_of([nm(objt, x0, 200 if tag == "pub" else mi, xa, fa) for x0 in st])
            v, e = taylor_filter(y, *b.x, qa); out["taylor_%s_%s" % (tag, a)] = (v[-nsel:], e[-nsel:])
            if tag == "fix": out["taylor_conv_%s" % a] = bool(b.success)
    # ---- SAV-CAViaR, eleven levels
    rtrain = y[:sp]; n = len(y); absr = np.abs(y); ntr = sp
    for tag in ("pub", "fix"):
        Q = np.empty((nsel, len(TAUS)))
        for j, tau in enumerate(TAUS):
            q0 = float(np.quantile(rtrain[:300], tau))
            def path(b, m):
                f = np.empty(m); f[0] = q0
                for i in range(1, m): f[i] = b[0] + b[1] * f[i - 1] + b[2] * absr[i - 1]
                return f
            def obj(b):
                if not (0 <= b[1] <= 0.999): return 1e6
                d = rtrain - path(b, ntr); return float(np.mean(np.where(d >= 0, tau * d, (tau - 1) * d)))
            b0 = np.array([q0 * 0.1, 0.8, -0.1 if tau < 0.5 else 0.1])
            if tag == "pub":
                r = nm(obj, b0, 300, 1e-3, 1e-6); b = r.x if r is not None else b0
            else:
                st = [b0] + [np.array([q0 * rng.uniform(0.01, 0.4), rng.uniform(0.5, 0.98), (-1 if tau < 0.5 else 1) * rng.uniform(0.01, 0.4)]) for _ in range(7)]
                r = best_of([nm(obj, x0, 1500, 1e-5, 1e-9) for x0 in st]); b = r.x if r is not None else b0
            Q[:, j] = path(b, n)[-nsel:]
        out["caviar_%s" % tag] = Q
    return out

if __name__ == "__main__":
    D = np.load(DUMP); PN = D["permno"]; Y = D["y"]; dates = pd.to_datetime(D["date"]); rk = D["rk_mk"]
    rr = pd.read_csv(os.path.join(P, "crsp_panel_returns.csv"), dtype={'permno': 'int32'}); rr['date'] = pd.to_datetime(rr['date'])
    rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    jobs = []
    for pn in pd.unique(PN):
        y = rr[rr.permno == pn].sort_values('date')['ret'].values.astype(float); nsel = int((PN == pn).sum())
        assert np.allclose(y[-nsel:], Y[PN == pn]), pn          # the dump's rows are this name's last nsel days
        jobs.append((int(pn), y, nsel))
    lg("%d names" % len(jobs))
    with Pool(15) as pool: res = pool.map(one_name, jobs, chunksize=1)
    lg("fits done")
    def nw_dm(d, mask=None):
        s = pd.Series(d if mask is None else d[mask], index=dates if mask is None else dates[mask]).groupby(level=0).mean().values
        s = s[np.isfinite(s)]; dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(v / len(s))), 2)
    out = {"note": "pub = the paper's fitting re-run here; fix = full window, free GAS scales, more starts and iterations. "
                   "DM > 0: the accuracy layer has the lower loss.", "n_rows": int(len(Y)), "fz0": {}, "caviar": {}}
    for a in ALPHAS:
        Le = D["fz0__engine__%s" % a]; blk = {}
        for m in ("gas", "taylor"):
            for tag in ("pub", "fix"):
                V = np.empty(len(Y)); E = np.empty(len(Y))
                for r in res: msk = PN == r["pn"]; V[msk], E[msk] = r["%s_%s_%s" % (m, tag, a)]
                L = fz0(Y, V, E, a)
                blk["%s_%s" % (m, tag)] = {"meanFZ0": round(float(np.nanmean(L)), 5), "nonfinite": int((~np.isfinite(L)).sum()),
                                           "breach_pct": round(100 * float((Y <= V).mean()), 2), "engine_DM": nw_dm(L - Le),
                                           "engine_DM_top_decile": nw_dm(L - Le, rk == 10)}
            dumped = float(np.nanmean(D["fz0__%s__%s" % ("gas_pzc" if m == "gas" else "taylor", a)]))
            blk["%s_dumped_meanFZ0" % m] = round(dumped, 5)
            blk["%s_converged_names" % m] = int(sum(r["%s_conv_%s" % (m, a)] for r in res))
        blk["engine_meanFZ0"] = round(float(np.nanmean(Le)), 5); out["fz0"][str(a)] = blk
    Pe = D["pin__engine"]; Pg = D["pin__garch_t"]
    for tag in ("pub", "fix"):
        L = np.zeros(len(Y))
        for j, t in enumerate(TAUS):
            q = np.empty(len(Y))
            for r in res: q[PN == r["pn"]] = r["caviar_%s" % tag][:, j]
            L += pin(Y, q, t)
        L /= len(TAUS)
        out["caviar"][tag] = {"mean_pinball": round(float(L.mean()), 5),
                              "engine_minus_caviar_pct": {k: round(100 * float((Pe[m] - L[m]).mean() / L[m].mean()), 3) for k, m in (("overall", np.ones(len(Y), bool)), ("top_mk63_decile", rk == 10), ("deciles_1to9", (rk >= 1) & (rk <= 9)))},
                              "engine_DM": {"overall": nw_dm(L - Pe), "top_mk63_decile": nw_dm(L - Pe, rk == 10), "deciles_1to9": nw_dm(L - Pe, (rk >= 1) & (rk <= 9))},
                              "caviar_vs_garch_t_top_decile_pct": round(100 * float((Pg - L)[rk == 10].mean() / Pg[rk == 10].mean()), 3)}
    out["caviar"]["dumped_mean_pinball"] = round(float(D["pin__sav_caviar"].mean()), 5); out["caviar"]["engine_mean_pinball"] = round(float(Pe.mean()), 5)
    json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out, indent=1)); lg("wrote %s" % OUT)
