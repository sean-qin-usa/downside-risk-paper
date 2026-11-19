# r73_simulation.py -- DOES THE CLOSED-FORM TEST OF tools/r71_shock_response.py RECOVER THE TRUTH WHEN THE TRUTH IS KNOWN?
#
# r71 estimates lambda, the share of a large shock's GARCH variance impact that is real, and finds it near 0.1 on stocks.
# Two things could produce a low lambda-hat without the shocks being one-offs: selection (rows are chosen because the
# trailing window holds a large residual) and the estimation of GARCH parameters on fat-tailed data. This simulation
# runs the same code path on panels whose data-generating process is known.
#
# Processes (100 series of 2,768 days per panel, the canonical panel's shape; estimation on the first 60%):
#  garch      GARCH(1,1) with Student-t(4) innovations, alpha 0.14, beta 0.81: the canonical panel's median fitted
#             values. Every shock feeds variance exactly as the model says. True lambda = 1.
#  jump_0     a smooth GARCH(1,1) (alpha 0.06, beta 0.92, t(8) innovations) plus one-off jumps: on average one per 63
#             days, normal with standard deviation five times the unconditional one. Jumps do NOT enter the variance
#             recursion. True lambda = 0.
#  jump_0.25  the same, but a quarter of each squared jump enters the recursion.
#  jump_1     the same, but the whole squared jump enters the recursion, like any return.
# Per panel: GARCH-t per series on the estimation window; the score deciles, D and lambda-hat exactly as in r71; the
# corrected GARCH-t; Beta-t-EGARCH per series.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19):
#  M1. garch: lambda-hat (pinball 2.5%, top decile) averages within 0.15 of 1; GARCH-t's 2.5% breach rate is within
#      0.4 points of nominal in every shock-age bucket; neither the corrected GARCH-t (lambda = 0) nor Beta-t-EGARCH
#      beats GARCH-t in the top decile.
#  M2. jump_0: lambda-hat averages below 0.2; GARCH-t's breach rate at ages 1-10 is below 1%; the corrected GARCH-t and
#      Beta-t-EGARCH each beat GARCH-t in the top decile by more than 5%.
#  M3. lambda-hat rises with the share that feeds variance: jump_0 < jump_0.25 < jump_1, and jump_1 is above 0.7.
#  M4. On jump_0 the fitted GARCH-t has a higher alpha and a lower beta than the smooth process that generated the data
#      (0.06 and 0.92): one-off jumps make volatility look less persistent and more reactive than it is.
# Usage: python tools/r73_simulation.py [reps]
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("GBC_PROJ", ".")
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "code", "case_studies")); sys.path.insert(0, os.path.join(ROOT, "code", "paper"))
import nvda_engine_case as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6; NS, NT = 100, 2768; NW = int(os.environ.get("R73_WORKERS", "8")); t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 2); AGES = (("1-10", 1, 10), ("11-21", 11, 21), ("22-42", 22, 42), ("43-63", 43, 63))
DGPS = {"garch": dict(al=0.14, be=0.81, nu=4.0, p=0.0, js=0.0, kappa=0.0), "jump_0": dict(al=0.06, be=0.92, nu=8.0, p=1 / 63, js=5.0, kappa=0.0),
        "jump_0.25": dict(al=0.06, be=0.92, nu=8.0, p=1 / 63, js=5.0, kappa=0.25), "jump_1": dict(al=0.06, be=0.92, nu=8.0, p=1 / 63, js=5.0, kappa=1.0)}
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))

def simulate(d, seed):
    rng = np.random.default_rng(seed); nb = 500; n = NT + nb; z = rng.standard_t(d["nu"], n) / math.sqrt(d["nu"] / (d["nu"] - 2)); J = (rng.random(n) < d["p"]) * rng.normal(0, d["js"], n) if d["p"] > 0 else np.zeros(n)
    om = 1 - d["al"] - d["be"]; y = np.empty(n); s2 = 1.0
    for t in range(n):
        e = math.sqrt(s2) * z[t]; y[t] = e + J[t]; s2 = om + d["al"] * (e * e + d["kappa"] * J[t] ** 2) + d["be"] * s2
    return y[nb:]

def one(a):
    name, seed = a; d = DGPS[name]; y = simulate(d, seed); n = len(y); ne = int(n * 0.6)
    pr = arch_model(y[:ne], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params
    om, al, be, mu, nu = float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"]), float(pr["mu"]), float(pr["nu"]); e0 = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:ne])
    for k in range(1, n): s2[k] = max(om + al * e0[k - 1] ** 2 + be * s2[k - 1], 1e-8)
    sig = np.sqrt(s2); z = e0 / sig; win = sliding_window_view(z, 63)[:-1]; j = np.argmax(np.abs(win - win.mean(1, keepdims=True)), 1); s = np.arange(len(win)) + j
    age = np.zeros(n, int); D = np.full(n, np.nan); age[63:] = 63 - j; D[63:] = al * be ** (age[63:] - 1.0) * sig[s] ** 2 * (z[s] ** 2 - 1.0) / sig[63:] ** 2; siz = np.full(n, np.nan); siz[63:] = np.abs(z[s])
    mk = pd.Series(z).rolling(63, min_periods=30).kurt().shift(1).values
    bx = C.fit_bteg(y[:ne]); bmu, bom, phi, kap, bnu = C.bteg_unpack(bx); lam = C._bteg_filter(y, bmu, bom, phi, kap, bnu); bq = bmu + np.exp(lam) * stats.t.ppf(0.025, bnu)
    te = np.arange(ne, n); pre = np.arange(126, ne)
    return dict(t=te, y=y[te], sig=sig[te], mu=np.full(len(te), mu), nu=np.full(len(te), nu), D=np.clip(D[te], -0.5, 0.98), age=age[te], mk=mk[te], siz=siz[te], bq=bq[te], al=al, be=be, nuf=nu,
                pre=dict(y=y[pre], sig=sig[pre], mu=mu, nu=nu, D=np.clip(D[pre], -0.5, 0.98), mk=mk[pre]))

def panel(name, rep, pool):
    R = pool.map(one, [(name, 100000 * rep + 1000 * list(DGPS).index(name) + i) for i in range(NS)], chunksize=2)
    cat = lambda k: np.concatenate([r[k] for r in R]); T, Y, SIG, MU, NU, DD, AGE, MK, SIZ, BQ = (cat(k) for k in ("t", "y", "sig", "mu", "nu", "D", "age", "mk", "siz", "bq"))
    rk = pd.qcut(pd.Series(MK).rank(method="first"), 10, labels=False).values + 1; TOP = rk == 10
    cfq = lambda lam, y=None, mu=MU, sig=SIG, nu=NU, Dv=DD: mu + sig * np.sqrt(np.maximum(1 - (1 - lam) * Dv, 0.02)) * tq(0.025, nu)
    def lam_hat(mask, kind="pin"):
        if kind == "pin": L = [pin(Y[mask], cfq(l, mu=MU[mask], sig=SIG[mask], nu=NU[mask], Dv=DD[mask]), 0.025).sum() for l in GRID]
        else:
            L = []
            for l in GRID: s_ = SIG[mask] * np.sqrt(np.maximum(1 - (1 - l) * DD[mask], 0.02)) / np.sqrt(NU[mask] / (NU[mask] - 2)); L.append(-(stats.t.logpdf((Y[mask] - MU[mask]) / s_, NU[mask]) - np.log(s_)).sum())
        return float(GRID[int(np.argmin(L))])
    pm = np.concatenate([r["pre"]["mk"] for r in R]); pk = np.isfinite(pm) & np.isfinite(np.concatenate([r["pre"]["D"] for r in R])); thr = np.quantile(pm[pk], 0.9); pt = pk & (pm >= thr)
    pc = lambda k: np.concatenate([np.full(len(r["pre"]["y"]), r["pre"][k]) if np.isscalar(r["pre"][k]) else r["pre"][k] for r in R])[pt]
    py, psig, pmu, pnu, pD = pc("y"), pc("sig"), pc("mu"), pc("nu"), pc("D"); lam_pre = float(GRID[int(np.argmin([pin(py, cfq(l, mu=pmu, sig=psig, nu=pnu, Dv=pD), 0.025).sum() for l in GRID]))])
    Lg = pin(Y, cfq(1.0), 0.025); L0 = pin(Y, cfq(0.0), 0.025); Lp = pin(Y, cfq(lam_pre), 0.025); Lb = pin(Y, BQ, 0.025); hit = Y <= cfq(1.0)
    def edge(Lm, mask):
        s = pd.Series((Lg - Lm)[mask], index=T[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return 100 * float((Lg - Lm)[mask].mean() / Lg[mask].mean()), float(s.mean() / math.sqrt(max(v / len(s), 1e-18)))
    rd = pd.qcut(pd.Series(DD).rank(method="first"), 10, labels=False).values + 1; TD = rd == 10; ALLM = np.ones(len(Y), bool)
    pred = lambda mask: 100 * float(stats.t.cdf(stats.t.ppf(0.025, NU[mask]) / np.sqrt(1 - DD[mask]), NU[mask]).mean())
    o = {"fitted_alpha_median": float(np.median([r["al"] for r in R])), "fitted_beta_median": float(np.median([r["be"] for r in R])), "fitted_nu_median": float(np.median([r["nuf"] for r in R])),
         "top_decile_shock_size_median": float(np.median(SIZ[TOP])), "top_decile_mean_D": float(DD[TOP].mean()), "lambda_top_decile_pinball": lam_hat(TOP), "lambda_top_decile_likelihood": lam_hat(TOP, "lik"),
         "lambda_top_D_decile_pinball": lam_hat(TD), "lambda_all_rows_pinball": lam_hat(ALLM), "lambda_pre_test": lam_pre, "garch_t_breach_all": 100 * float(hit.mean()), "garch_t_breach_top_decile": 100 * float(hit[TOP].mean()),
         "bteg_breach_top_decile": 100 * float((Y <= BQ)[TOP].mean())}
    for lab, lo, hi in AGES: m = TOP & (AGE >= lo) & (AGE <= hi); o["breach_age_" + lab] = 100 * float(hit[m].mean()); o["predicted_lambda0_age_" + lab] = pred(m)
    for lab, Lm in (("cf0", L0), ("cf_pre", Lp), ("bteg", Lb)):
        for sl, m in (("top_decile", TOP), ("top_D_decile", TD), ("all_days", ALLM)): e = edge(Lm, m); o["edge_%s_%s" % (lab, sl)] = e[0]; o["dm_%s_%s" % (lab, sl)] = e[1]
    return o

if __name__ == "__main__":
    OUT = {"note": "Each entry: mean and standard deviation over %d simulated panels of %d series x %d days. edge > 0: the named model has lower 2.5%% pinball loss than GARCH-t." % (REPS, NS, NT),
           "predictions_written_before_run": "see the script header (M1-M4)", "processes": DGPS, "results": {}, "per_panel": {}}
    with Pool(NW) as pool:
        for name in DGPS:
            rows = []
            for rep in range(REPS): rows.append(panel(name, rep, pool)); lg("%s panel %d: lambda %.2f, top-decile breach %.2f%%, cf0 edge %+.1f%%, bteg edge %+.1f%%" % (name, rep, rows[-1]["lambda_top_decile_pinball"], rows[-1]["garch_t_breach_top_decile"], rows[-1]["edge_cf0_top_decile"], rows[-1]["edge_bteg_top_decile"]))
            df = pd.DataFrame(rows); OUT["results"][name] = {k: {"mean": round(float(df[k].mean()), 3), "sd": round(float(df[k].std()), 3)} for k in df.columns}; OUT["per_panel"][name] = rows
            json.dump(OUT, open(os.path.join(ROOT, "results", "paper", "diagnostics", "r73_simulation.json"), "w"), indent=1)
    print("\n######## simulation, mean (sd) over %d panels" % REPS)
    keys = ["fitted_alpha_median", "fitted_beta_median", "fitted_nu_median", "top_decile_shock_size_median", "top_decile_mean_D", "lambda_top_decile_pinball", "lambda_top_decile_likelihood", "lambda_top_D_decile_pinball", "lambda_all_rows_pinball", "lambda_pre_test",
            "garch_t_breach_all", "garch_t_breach_top_decile", "bteg_breach_top_decile"] + [x % lab for lab, _, _ in AGES for x in ("breach_age_%s", "predicted_lambda0_age_%s")] + ["%s_%s_%s" % (p_, lab, sl) for lab in ("cf0", "cf_pre", "bteg") for sl in ("top_decile", "top_D_decile", "all_days") for p_ in ("edge", "dm")]
    print("%-34s" % "" + "".join("%-18s" % n_ for n_ in DGPS))
    for k in keys: print("%-34s" % k + "".join("%-18s" % ("%.2f (%.2f)" % (OUT["results"][n_][k]["mean"], OUT["results"][n_][k]["sd"])) for n_ in DGPS))
    lg("done")
