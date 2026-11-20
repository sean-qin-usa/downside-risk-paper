# r83_information_value.py -- WHAT A PIECE OF INFORMATION IS WORTH TO A VaR MODEL: ONE EQUATION, TESTED TWICE.
#
# Both errors of GARCH-t found in r71-r82 are errors of scale on an identifiable set of days: too narrow on earnings
# reaction days, too wide after a one-off shock. The cost of a scale error has a closed form.
#
# Let the standardized return on a day of type d be tau_d times a Student-t with nu_d degrees of freedom. The expected
# pinball loss at level a of a forecast q (in units of the model's sigma) is
#     EL_d(q) = q * (F(q/tau_d) - a) + tau_d * f(q/tau_d) * (nu_d + (q/tau_d)^2) / (nu_d - 1),
# F and f the cdf and density of that Student-t. A model that does not know the day type forecasts its usual quantile
# q on every day. A model that knows it forecasts c_d * q, with c_d the minimizer of EL_d(c*q). With p_d the share of
# days of type d and m_d the mean sigma on those days relative to all days, the value of knowing the type is
#     V(a) = sum_d p_d m_d [EL_d(q) - EL_d(c_d q)]  /  sum_d p_d m_d EL_d(q).
# Everything on the right is estimated before the test period: (tau_d, nu_d) by maximum likelihood on the pre-test
# standardized residuals of each day type, m_d from the pre-test rows, p_d from the calendar. No test return is used.
#
# Part A. V against the gain that the calendar multipliers gave GARCH-t on the test rows (r82), on both stock panels.
# Part C. A prediction that has not been tested anywhere: portfolios. For an equal-weighted portfolio of N names the
#   earnings jumps are firm-specific, so the variance a reporting name adds is (kappa-1)*sigma_i^2/N^2, with kappa the
#   single-name variance ratio of the day (pre-test). The share of portfolio variance that comes from earnings days is
#       x / (1 + rho*(N-1) + x),      x = sum_d p_d (kappa_d - 1),   rho = mean pairwise correlation (pre-test),
#   and the informed portfolio VaR adds that variance on the days it is due. Nothing is fitted at the portfolio level.
#   For N = 1, 2, 5, 10, 20, 50 and the whole panel: the gain of the informed VaR over the portfolio's own GARCH-t,
#   the gain V predicts from the same scale ratios, and the variance share against the formula.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r82 (calendar multipliers gave GARCH-t 7.8% and
# 12.9% on the CRSP panel at 2.5% and 1%, 5.5% and 9.5% on US large caps; multipliers 3.7 and 2.6 on the reaction
# day), and a rough two-number version of V run in conversation with approximate inputs (8.3%, 13.8%; 7.3%, 12.7%).
#  A1. V is within 1.0 point of the realized gain at 2.5% and within 2.0 points at 1%, on both panels.
#  A2. The multiplier the equation implies for the reaction day is within 0.3 of the one chosen on pre-test rows.
#  C1. The realized gain falls with N at 2.5%: 5% to 9% for single names on the CRSP panel, 0.5% to 3% at N = 10,
#      below 0.6% at N = 50, between -0.2% and +0.3% for the whole panel.
#  C2. V is within 0.5 points, or within a factor of 1.5, of the realized gain at every N.
#  C3. The earnings share of portfolio variance in the test period is within 30% of the formula at every N.
#  C4. At N = 10 the portfolio GARCH-t's 2.5% VaR is breached on at least 5% of the days in the top 5% by earnings
#      share of variance, and the informed VaR on fewer than 4%.
# Usage: python tools/r83_information_value.py canon | us_largecap      (needs data/panels/<panel>_calcache.pkl: tools/calpanel.py)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r83"); os.makedirs(OUTD, exist_ok=True); tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)

G = {}
def port(job):   # one equal-weighted portfolio: its own GARCH-t, and the same VaR with the earnings variance added on the days it is due
    N, seed = job; R, S2, F0, F1, pre, k0, k1 = G["R"], G["S2"], G["F0"], G["F1"], G["pre"], G["k0"], G["k1"]
    cols = np.array([seed]) if N == 1 else (np.arange(R.shape[1]) if N == R.shape[1] else np.random.default_rng(1000 * N + seed).choice(R.shape[1], N, replace=False))
    rp = R[:, cols].mean(1); extra = ((k0 - 1) * (S2[:, cols] * F0[:, cols]).sum(1) + (k1 - 1) * (S2[:, cols] * F1[:, cols]).sum(1)) / N ** 2
    try:
        pr = arch_model(rp[pre], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params; sig = np.asarray(arch_model(rp, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
    except Exception: return None
    mu, nu = float(pr["mu"]), float(pr["nu"]); v = sig ** 2; vi = np.maximum(v + extra - extra[pre].mean() / v[pre].mean() * v, 0.25 * v); s = np.sqrt(vi / v); te = ~pre; out = {"N": N, "share_test": float(extra[te].mean() / np.mean((rp[te] - mu) ** 2))}
    top = te & (extra / v >= np.quantile((extra / v)[te], 0.95))
    for a in AL:
        q = tq(a, nu); Lu = pin(rp, mu + sig * q, a); Li = pin(rp, mu + sig * s * q, a); k = math.sqrt(nu / (nu - 2)); lu = sig * s * EL(q / s, 1 / k, nu, a); li = sig * s * EL(q, 1 / k, nu, a)
        out[str(a)] = {"gain": 100 * float((Lu - Li)[te].sum() / Lu[te].sum()), "V": 100 * float((lu - li)[te].sum() / lu[te].sum()), "breach_top5_plain": 100 * float((rp <= mu + sig * q)[top].mean()), "breach_top5_informed": 100 * float((rp <= mu + sig * s * q)[top].mean()),
                       "breach_plain": 100 * float((rp <= mu + sig * q)[te].mean()), "breach_informed": 100 * float((rp <= mu + sig * s * q)[te].mean())}
    return out

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "tools")); A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0)]; R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json")))
    typ = np.where(A.r0 == 1, "reaction day", np.where(A.r1 == 1, "day after", "other days")); pre = ~A.test.values; z = ((A.y - A.mu) / A.sig).values; TY = ("reaction day", "day after", "other days")
    # ---- Part A
    par = {}
    for d in TY:
        f = pre & (typ == d); nu_, _, tau_ = stats.t.fit(z[f], floc=0); par[d] = {"tau": float(tau_), "nu": float(nu_), "p": float((typ[~pre] == d).mean()), "m": float(A.sig.values[f].mean() / A.sig.values[pre].mean()), "variance_ratio": float(np.mean(z[f] ** 2)), "rows": int(f.sum())}
    NU = A.groupby("name")["nu"].first().values; OUT = {"universe": WHAT, "names": int(A["name"].nunique()), "predictions_written_before_run": "see the script header (A1-A2, C1-C4)", "day_types_pre_test": par, "A": {}}
    cst = {}
    for d in ("reaction day", "day after"):   # the multiplier is chosen at 2.5% and used at both levels, as in r82
        q = tq(0.025, NU); cst[d] = float(optimize.minimize_scalar(lambda c: EL(c * q, par[d]["tau"], par[d]["nu"], 0.025).mean(), bounds=(0.5, 8.0), method="bounded").x)
    for a in AL:
        q = tq(a, NU); num = sum(par[d]["p"] * par[d]["m"] * (EL(q, par[d]["tau"], par[d]["nu"], a) - EL(cst[d] * q, par[d]["tau"], par[d]["nu"], a)).mean() for d in cst); den = sum(par[d]["p"] * par[d]["m"] * EL(q, par[d]["tau"], par[d]["nu"], a).mean() for d in TY)
        k = np.sqrt(NU / (NU - 2)); e2 = lambda d, c: (np.sqrt(par[d]["variance_ratio"]) * EL(c * q / np.sqrt(par[d]["variance_ratio"]), 1 / k, NU, a)).mean()   # two-number version: same tail, variance ratio only
        num2 = sum(par[d]["p"] * par[d]["m"] * (e2(d, 1.0) - e2(d, np.sqrt(par[d]["variance_ratio"]))) for d in cst); den2 = sum(par[d]["p"] * par[d]["m"] * e2(d, 1.0) for d in TY)
        OUT["A"][str(a)] = {"V_pct": round(100 * num / den, 2), "V_two_number_version_pct": round(100 * num2 / den2, 2), "realized_garch_t_with_multipliers_pct": R82[str(a)]["vs_garch_t"]["GARCH-t, calendar multipliers"]["all days"]["edge_pct"],
                            "realized_other_models_gain_from_calendar_pct": {m.split(",")[0]: round(R82[str(a)]["vs_garch_t"][m]["all days"]["edge_pct"] - R82[str(a)]["vs_garch_t"][m.split(",")[0]]["all days"]["edge_pct"], 2) for m in R82[str(a)]["vs_garch_t"] if m.endswith("calendar multipliers") and not m.startswith("GARCH-t")}}
    OUT["A"]["implied_multipliers"] = {d: round(c, 2) for d, c in cst.items()}; OUT["A"]["multipliers_chosen_on_pre_test_rows_r82"] = R82["multipliers_c0_c1"]["GARCH-t, calendar multipliers"]
    lg("part A: %s" % json.dumps(OUT["A"]))
    # ---- Part C: portfolios
    first = A.groupby("name")["date"].min(); start = first.quantile(0.9); keep = first[first <= start].index; B = A[A["name"].isin(keep) & (A.date >= start)]
    R = B.pivot(index="date", columns="name", values="y"); ok = R.notna().all(1); R = R[ok]; piv = lambda c: B.pivot(index="date", columns="name", values=c).reindex(R.index).values
    S2 = piv("msig") ** 2; F0 = (piv("r0") == 1).astype(float); F1 = ((piv("r1") == 1) & (F0 == 0)).astype(float); tst = B.groupby("date")["test"].first().reindex(R.index).values.astype(bool); prem = ~tst
    um = (piv("y") - piv("mmu")) ** 2 / S2; k0 = float(um[prem][F0[prem] == 1].mean()); k1 = float(um[prem][F1[prem] == 1].mean()); M = R.shape[1]
    Z = (R.values - R.values[prem].mean(0)) / R.values[prem].std(0); ordn = prem & (F0.sum(1) + F1.sum(1) < 0.02 * M); cc = np.corrcoef(Z[ordn].T); rho = float((cc.sum() - M) / (M * (M - 1))); x = float(F0[tst].mean() * (k0 - 1) + F1[tst].mean() * (k1 - 1))
    G.update(R=R.values, S2=S2, F0=F0, F1=F1, pre=prem, k0=k0, k1=k1); SIZES = [n for n in (1, 2, 5, 10, 20, 50) if n < M] + [M]; K = int(os.environ.get("R83_DRAWS", "40"))
    jobs = [(N, s) for N in SIZES for s in (range(M) if N == 1 else ([0] if N == M else range(K)))]
    with Pool(int(os.environ.get("R83_WORKERS", "12"))) as pool: res = [r for r in pool.map(port, jobs, chunksize=4) if r is not None]
    OUT["C"] = {"names": M, "days": int(len(R)), "test_days": int(tst.sum()), "from": str(R.index[0].date()), "variance_ratio_pre_test": {"reaction day": round(k0, 2), "day after": round(k1, 2)}, "mean_pairwise_correlation_pre_test": round(rho, 3), "x": round(x, 4), "by_N": {}}
    for N in SIZES:
        rr = [r for r in res if r["N"] == N]; row = {"portfolios": len(rr), "variance_share_test": round(float(np.mean([r["share_test"] for r in rr])), 4), "variance_share_formula": round(x / (1 + rho * (N - 1) + x), 4)}
        for a in AL:
            g = np.array([r[str(a)]["gain"] for r in rr]); row[str(a)] = {"gain_pct": round(float(g.mean()), 2), "se_across_portfolios": round(float(g.std() / math.sqrt(len(g))), 2) if len(g) > 1 else None, "share_of_portfolios_with_gain": round(float((g > 0).mean()), 2),
                                                                          **{k_: round(float(np.mean([r[str(a)][k_] for r in rr])), 2) for k_ in ("V", "breach_top5_plain", "breach_top5_informed", "breach_plain", "breach_informed")}}
        OUT["C"]["by_N"][str(N)] = row; lg("N=%d %s" % (N, json.dumps(row)))
    json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
