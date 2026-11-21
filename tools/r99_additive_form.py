# r99_additive_form.py -- THE EQUATION WITH ANNOUNCEMENT VARIANCE AS ITS OWN COMPONENT: ONE ATTEMPT, PASS OR FAIL.
#
# r97 and r98: the equation of r83 does not return the gain of its own sample (0.6 and 1.4 points high for single
# names, a quarter to a half high for portfolios). Two assumptions failed on reaction days: the standardized return is
# not symmetric, and it is smaller when the day's volatility is high. The second is what r85 found from another
# side: the announcement adds a variance of its own, which does not scale with the day's ordinary variance. So the
# informed model is not "a multiple of GARCH-t on reaction days" but
#     variance_t = msig_t^2 + J_it on a reaction day,     msig_t^2 on other days,
# msig the masked filter's ordinary volatility and J_it the firm's announcement variance from its own last eight
# announcements (r85's second clock: J = (1 - w) theta sbar2_i + w H_it; theta and w as chosen there on pre-test
# reaction days; the panel prior alone where a firm has fewer than three earlier announcements). Against the
# benchmark GARCH-t with volatility sig_t, the scale ratio s_t = sqrt(variance_t) / sig_t is known the day before for
# every day, and the equation is the nested form of r96 with this s:
#     V = sum_t sig_t s_t [ EL_d(q_t / s_t) - EL_d(c_d q_t) ]  /  sum_t sig_t s_t EL_d(q_t / s_t),
# with (tau_d, nu_d) fitted to z_t / s_t for each day type on the loss side (z < 0), weighted by sig_t s_t (the weight
# each day has in the loss sum), and c_d the best multiplier for reaction days and days after (1 on other days),
# chosen on pre-test rows. The informed forecast is c_d s_t q_t.
#
# THE TEST, STATED BEFORE THE RUN (2026-11-20). An expression that is right returns the gain of its own sample.
# For the pre-test rows and the test rows of both panels: the tails fitted on the rows, the expression over those
# rows, against the gain the multipliers c_d s_t make on those rows.
#   PASS: the gap is within 0.3 points at 2.5% and within 0.5 points at 1% in all four samples.
# Known beforehand: r97 (partition form: +0.6, +0.2, +1.4, +1.2 at 2.5%; with loss-side sigma-weighted tails -0.3,
# -0.5, +0.1, -0.2), r85 (the second clock adds about half a percent over all days to the masked calendar GARCH-t,
# which gains 9.2% and 6.5% over GARCH-t), r98.
# PREDICTIONS.
#  D1. The multipliers c_d s_t gain 9% to 10.5% over GARCH-t on the CRSP test rows and 6.5% to 7.5% on US large caps.
#  D2. The test above: I expect a pass on US large caps and a borderline result on the CRSP panel.
#  D3. With the tails fitted on both sides and equal weights (the fitting of r83) the largest gap is under 0.7 points
#      at 2.5%, half the partition form's 1.4: the additive scale alone removes at least half of the overshoot.
#  D4. As a forecast (tails and constants from pre-test rows) V is within 1.0 point of the test rows' gain at 2.5%
#      on both panels, because J follows each firm's recent announcements.
#  D5. Portfolios of N names (the law is now: a reporting firm adds J_it / N^2), tails fitted to the portfolio's own
#      standardized return divided by its scale ratio: the realized gain is 0.85 to 1.15 of V for single names and
#      within 0.3 points of V for N of 5, 10 and 20, on pre-test days and on test days (r98: 0.64 to 0.78).
# Usage: python tools/r99_additive_form.py canon | us_largecap     (needs data/panels/<panel>_calcache.pkl and r85's result file)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from scipy.special import gammaln
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.025, 0.01); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r99"); os.makedirs(OUTD, exist_ok=True); t0 = time.time()
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def tfit(x, w):   # weighted maximum likelihood for a Student-t with location zero: (nu, tau)
    def nll(p): nu = 2.05 + math.exp(p[0]); tau = math.exp(p[1]); return -float((w * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(tau) - (nu + 1) / 2 * np.log1p((x / tau) ** 2 / nu))).sum()) / float(w.sum())
    r = optimize.minimize(nll, [math.log(3.0), math.log(0.75)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}); return 2.05 + math.exp(r.x[0]), math.exp(r.x[1])
G = {}
def port(job):   # one equal-weighted portfolio: its GARCH-t, the additive law, and the expression with the portfolio's own loss-side tail
    N, seed = job; R, J, S2, F0, F1, pre, k1 = G["R"], G["J"], G["S2"], G["F0"], G["F1"], G["pre"], G["k1"]
    cols = np.array([seed]) if N == 1 else np.random.default_rng(1000 * N + seed).choice(R.shape[1], N, replace=False); rp = R[:, cols].mean(1); extra = ((J[:, cols] * F0[:, cols]).sum(1) + (k1 - 1) * (S2[:, cols] * F1[:, cols]).sum(1)) / N ** 2
    try: pr = arch_model(rp[pre], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params; sig = np.asarray(arch_model(rp, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
    except Exception: return None
    mu, nu = float(pr["mu"]), float(pr["nu"]); v = sig ** 2; z = (rp - mu) / sig; out = {"N": N}
    for lab, m in (("pre-test days", pre), ("test days", ~pre)):
        s = np.sqrt(np.maximum(v + extra - extra[m].mean() / v[m].mean() * v, 0.25 * v) / v); k = m & (z < 0); nu_, tau_ = tfit(z[k] / s[k], sig[k] * s[k]); out[lab] = {}
        for a in AL: q = tq(a, nu); Lu = pin(rp, mu + sig * q, a); Li = pin(rp, mu + sig * s * q, a); lu = sig * s * EL(q / s, tau_, nu_, a); out[lab][str(a)] = {"gain": 100 * float((Lu - Li)[m].sum() / Lu[m].sum()), "V": 100 * float((lu - sig * s * EL(q, tau_, nu_, a))[m].sum() / lu[m].sum())}
    return out

if __name__ == "__main__":
    A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0) & (A.U | ~A.test)].sort_values(["name", "date"]).reset_index(drop=True); P85 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r85", WHAT + ".json")))["parameters_pre_test"]["M2"]; th, w = P85["theta"], P85["w"]
    # the firm's announcement variance, as r85
    ordv = A[(A.r0 != 1) & (A.r1 != 1) & ~A.test]; sbar2 = ordv.groupby("name")["msig"].apply(lambda s: float(np.mean(s ** 2))); E = A[A.r0 == 1].copy(); E["e2"] = (E.y - E.mmu) ** 2; E["s2"] = E.msig ** 2; g = E.groupby("name")
    H = (g["e2"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()) - g["s2"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())).clip(lower=0); prior = th * E["name"].map(sbar2); A["J"] = 0.0; A.loc[E.index, "J"] = np.where(np.isfinite(H), (1 - w) * prior + w * H, prior)
    y, mu, sg, nu = A.y.values, A.mu.values, A.sig.values, A.nu.values; z = (y - mu) / sg; te = A.test.values; typ = np.where(A.r0.values == 1, 0, np.where(A.r1.values == 1, 1, 2)); TY = ("reaction day", "day after", "other days"); s = np.sqrt(A.msig.values ** 2 + A.J.values) / sg; wt = sg * s
    def tails(rows, left, wsig): return [tfit(z[k] / s[k], wt[k] if wsig else np.ones(int(k.sum()))) for k in (rows & (typ == d) & ((z < 0) if left else True) for d in range(3))]
    def expr(par, rows, c):
        o = {}
        for a in AL:
            num = den = 0.0
            for d in range(3): k = rows & (typ == d); q = tq(a, nu[k]); lu = wt[k] * EL(q / s[k], par[d][1], par[d][0], a); den += lu.sum(); num += (lu - wt[k] * EL(c[d] * q, par[d][1], par[d][0], a)).sum()
            o[str(a)] = 100 * num / den
        return o
    OUT = {"universe": WHAT, "test_and_predictions": "see the script header (the pass rule, D1-D5); written before the run", "second_clock_parameters_r85": {"theta": th, "w": w}, "reaction_days_with_the_firm_history": round(float(np.isfinite(H).mean()), 3), "scale_ratio_s": {TY[d]: {"median": round(float(np.median(s[typ == d])), 3), "p10": round(float(np.quantile(s[typ == d], 0.1)), 3), "p90": round(float(np.quantile(s[typ == d], 0.9)), 3)} for d in range(3)}, "variants": {}}
    for lab, left, wsig in (("loss side, weighted (the test)", 1, 1), ("both sides, equal weights", 0, 0)):
        pp = tails(~te, left, wsig); c = np.ones(3)
        for d in (0, 1): k = ~te & (typ == d); q = tq(0.025, nu[k]); c[d] = float(optimize.minimize_scalar(lambda c_: float((wt[k] * EL(c_ * q, pp[d][1], pp[d][0], 0.025)).sum()), bounds=(0.3, 3.0), method="bounded").x)
        m = c[typ] * s; real = lambda rows, a: 100 * float((pin(y, mu + sg * tq(a, nu), a) - pin(y, mu + sg * m * tq(a, nu), a))[rows].sum() / pin(y, mu + sg * tq(a, nu), a)[rows].sum()); row = {"multipliers_c": {TY[d]: round(float(c[d]), 3) for d in range(3)}, "tails_pre_test": {TY[d]: {"tail_index": round(pp[d][0], 2), "tau": round(pp[d][1], 3)} for d in range(3)}}
        for per, rows in (("pre-test rows", ~te), ("test rows", te)):
            v = expr(tails(rows, left, wsig), rows, c); row[per] = {str(a): {"expression_pct": round(v[str(a)], 2), "gain_on_these_rows_pct": round(real(rows, a), 2), "gap_points": round(v[str(a)] - real(rows, a), 2)} for a in AL}
        v = expr(pp, te, c); row["forecast_of_test_rows_from_pre_test"] = {str(a): {"V_pct": round(v[str(a)], 2), "gain_on_test_rows_pct": round(real(te, a), 2), "gap_points": round(v[str(a)] - real(te, a), 2)} for a in AL}
        row["gain_on_reaction_days_test_pct"] = {str(a): round(100 * float((pin(y, mu + sg * tq(a, nu), a) - pin(y, mu + sg * m * tq(a, nu), a))[te & (typ == 0)].sum() / pin(y, mu + sg * tq(a, nu), a)[te & (typ == 0)].sum()), 2) for a in AL}
        if left: row["PASS"] = bool(all(abs(row[per]["0.025"]["gap_points"]) <= 0.3 and abs(row[per]["0.01"]["gap_points"]) <= 0.5 for per in ("pre-test rows", "test rows")))
        OUT["variants"][lab] = row; print("%4.0fs %-32s c %s | own rows: pre-test %5.2f vs %5.2f (gap %+.2f), test %5.2f vs %5.2f (gap %+.2f) | 1%%: gaps %+.2f %+.2f | forecast %5.2f vs %5.2f (gap %+.2f; 1%% %+.2f) %s" % (time.time() - t0, lab, [round(float(x), 2) for x in c], row["pre-test rows"]["0.025"]["expression_pct"], row["pre-test rows"]["0.025"]["gain_on_these_rows_pct"], row["pre-test rows"]["0.025"]["gap_points"],
              row["test rows"]["0.025"]["expression_pct"], row["test rows"]["0.025"]["gain_on_these_rows_pct"], row["test rows"]["0.025"]["gap_points"], row["pre-test rows"]["0.01"]["gap_points"], row["test rows"]["0.01"]["gap_points"], row["forecast_of_test_rows_from_pre_test"]["0.025"]["V_pct"], row["forecast_of_test_rows_from_pre_test"]["0.025"]["gain_on_test_rows_pct"], row["forecast_of_test_rows_from_pre_test"]["0.025"]["gap_points"], row["forecast_of_test_rows_from_pre_test"]["0.01"]["gap_points"], ("PASS" if row.get("PASS") else "FAIL") if left else ""), flush=True)
    # ---- portfolios
    first = A.groupby("name")["date"].min(); start = first.quantile(0.9); keep = first[first <= start].index; B = A[A["name"].isin(keep) & (A.date >= start)]; R = B.pivot(index="date", columns="name", values="y"); R = R[R.notna().all(1)]; piv = lambda c_: B.pivot(index="date", columns="name", values=c_).reindex(R.index).values
    S2 = piv("msig") ** 2; F0 = (piv("r0") == 1).astype(float); F1 = ((piv("r1") == 1) & (F0 == 0)).astype(float); tst = B.groupby("date")["test"].first().reindex(R.index).values.astype(bool); um = (piv("y") - piv("mmu")) ** 2 / S2; M = R.shape[1]
    G.update(R=R.values, J=np.nan_to_num(piv("J")), S2=S2, F0=F0, F1=F1, pre=~tst, k1=float(um[~tst][F1[~tst] == 1].mean())); SIZES = (1, 2, 5, 10, 20)
    with Pool(int(os.environ.get("R99_WORKERS", "12"))) as pool: res = [r for r in pool.map(port, [(N, sd) for N in SIZES for sd in (range(M) if N == 1 else range(40))], chunksize=4) if r is not None]
    OUT["portfolios"] = {"names": int(M), "by_N": {}}
    for N in SIZES:
        rr = [r for r in res if r["N"] == N]; OUT["portfolios"]["by_N"][str(N)] = {"portfolios": len(rr), **{per: {str(a): {"gain_pct": round(float(np.mean([r[per][str(a)]["gain"] for r in rr])), 2), "V_pct": round(float(np.mean([r[per][str(a)]["V"] for r in rr])), 2), "se_of_gain_minus_V": round(float(np.std([r[per][str(a)]["gain"] - r[per][str(a)]["V"] for r in rr]) / math.sqrt(len(rr))), 2)} for a in AL} for per in ("pre-test days", "test days")}}
        print("%4.0fs N=%-3d %s" % (time.time() - t0, N, {per: (v["0.025"]["gain_pct"], v["0.025"]["V_pct"], v["0.025"]["se_of_gain_minus_V"]) for per, v in OUT["portfolios"]["by_N"][str(N)].items() if per != "portfolios"}), flush=True)
    json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
