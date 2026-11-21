# r96_nested_value.py -- THE NESTED FORM: DAY TYPES WITH THEIR OWN TAILS, AND A CONTINUOUS SCALE WITHIN EACH TYPE.
#
# WRITTEN AFTER r94 AND r95 (2026-11-20). Seen: a single scale family with one tail (r94, r95) accounts for small
# information to within 0.3 points after the fact, but misses the calendar by 10% to 20% in either direction (CRSP
# 6.9% against 7.6% realized, US large caps 6.9% against 5.8%), and before the fact it is 1.2 to 2.0 points low on
# every information set that contains the calendar. The partition form of r83, which gives each day type its own
# tail, was within 0.5 points on the calendar. So reaction days are not ordinary days at a larger scale. The form
# that nests both:
#     z_t = s_t * eps_t,   eps_t ~ Student-t(tau_d, nu_d) for the day's type d,   s_t a continuous scale within type,
#     V = sum_t sigma_t s_t [ EL_d(q_t / s_t) - EL_d(c_d q_t) ]  /  sum_t sigma_t s_t EL_d(q_t / s_t),
# with c_d the best multiplier for type d (1 on ordinary days, as in r83); the informed model forecasts c_d s_t q_t.
# With s = 1 this is r83; with one type and c = 1 it is r94.
# Benchmark GARCH-t, all days; types: reaction day, day after, other days. Estimation before the fact, as r95 part B:
# a regression of log|z| on the calendar, days since the last report and the state of past returns, cross-fitted over
# five blocks of pre-test dates; the fitted value is centred within type; one slope b within type; (tau_d, nu_d) by
# maximum likelihood on z / s for each type; c_d minimizes EL_d(c q) at 2.5%. Band: block bootstrap over pre-test
# dates (63 trading days). After the fact: the same expression with the slope and the tails measured on test rows.
#
# PREDICTIONS (written knowing r83, r94 and r95; before this ran).
#  M1. With s = 1 the machinery returns the partition form: V within 0.4 points of r83's 7.2% (CRSP) and 5.3% (US).
#  M2. Nested form before the fact: V within 1.0 point of the realized gain at 2.5% and 2.0 points at 1% on both
#      panels, and within 1.645 combined standard errors.
#  M3. The nested multiplier realizes 8.5% to 10% on the CRSP panel and 6.5% to 8% on US large caps at 2.5%.
#  M4. After the fact the expression is within the larger of 0.5 points and 10% of the realized gain at 2.5%.
#  M5. The 90% band is under 1.5 points either side at 2.5%.
# ADDED AFTER THE FIRST RUN (2026-11-20): a consistency check. Seen: after the fact the partition form over-states
# the US gain (6.9% against 5.7%) though the tails are measured on the test rows themselves. The check: the gain of
# each form's multipliers on the pre-test rows they were fitted to, against V. An expression that is right about its
# own sample returns its own sample's gain. No prediction was written for it.
# Usage: python tools/r96_nested_value.py canon | us_largecap        (needs data/panels/<panel>_calcache.pkl)
import os; os.environ["OMP_NUM_THREADS"] = "4"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r96"); os.makedirs(OUTD, exist_ok=True); NB = int(os.environ.get("R96_BOOT", "200")); t0 = time.time()
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def tfit(x, w):
    def nll(p): nu = 2.05 + math.exp(p[0]); tau = math.exp(p[1]); return -float((w * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(tau) - (nu + 1) / 2 * np.log1p((x / tau) ** 2 / nu))).sum()) / float(w.sum())
    r = optimize.minimize(nll, [math.log(3.0), math.log(0.75)], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7}); return 2.05 + math.exp(r.x[0]), math.exp(r.x[1])
def dm(d, dates):
    s = pd.Series(d, index=dates).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return float(s.mean() / math.sqrt(max(v / len(s), 1e-18)))
HGB = lambda: HistGradientBoostingRegressor(max_iter=250, max_depth=3, learning_rate=0.06, random_state=0)

X5 = ["logsig", "zl1", "absz5", "zstd21", "fracdn5"]; A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A["Dc"] = np.clip(A.Dm.values, -0.5, 0.98); COLS = ["r0", "r1", "ds"] + X5 + ["Dc", "wm"]
A = A[np.isfinite(A.r0) & np.isfinite(A[COLS]).all(1) & (A.U | ~A.test)].reset_index(drop=True); y, mu, sg, nu = A.y.values, A.mu.values, A.sig.values, A.nu.values; z = (y - mu) / sg; lz = np.log(np.maximum(np.abs(z), 0.02)); te = A.test.values; pr = ~te; dates = pd.DatetimeIndex(A.date.values)
typ = np.where(A.r0.values == 1, 0, np.where(A.r1.values == 1, 1, 2)); TY = ("reaction day", "day after", "other days"); Xv = A[COLS].values; pd_ = np.sort(dates[pr].unique().values); fold = np.searchsorted(pd_[(np.arange(1, 5) * len(pd_)) // 5], dates.values, side="right"); go = np.full(len(A), np.nan)
for k in range(5): f = pr & (fold != k); h = pr & (fold == k); go[h] = HGB().fit(Xv[f], lz[f]).predict(Xv[h])
gf = HGB().fit(Xv[pr], lz[pr]).predict(Xv); ho = np.zeros(len(A)); hf = np.zeros(len(A)); ld = np.zeros(len(A))   # fitted value and log|z|, centred within type on pre-test rows
for d in range(3): m = typ == d; ho[m] = go[m] - go[m & pr].mean(); hf[m] = gf[m] - gf[m & pr].mean(); ld[m] = lz[m] - lz[m & pr].mean()
def fit(idx, w, h, use_s=True):   # slope within type, tails by type, multipliers by type
    ww = w / w.sum(); hh = h[idx]; mh = (ww * hh).sum(); b = max(float((ww * (hh - mh) * ld[idx]).sum() / (ww * (hh - mh) ** 2).sum()), 0.0) if use_s else 0.0; par = {"b": b}
    for d in range(3):
        k = typ[idx] == d; nu_, tau_ = tfit(z[idx][k] / np.exp(b * hh[k]), w[k]); q = tq(0.025, nu[idx][k]); c = 1.0 if d == 2 else float(optimize.minimize_scalar(lambda c_: float((w[k] * EL(c_ * q, tau_, nu_, 0.025)).sum()), bounds=(0.5, 8.0), method="bounded").x); par[d] = (nu_, tau_, c)
    return par
def value(par, idx, w, h, mult=None):   # the expression; mult = None means the informed model applies c_d * s
    s = np.exp(par["b"] * h[idx]); o = {}
    for a in AL:
        num = den = 0.0
        for d in range(3):
            k = typ[idx] == d; nu_, tau_, c = par[d]; q = tq(a, nu[idx][k]); lu = w[k] * sg[idx][k] * s[k] * EL(q / s[k], tau_, nu_, a); den += lu.sum(); num += (lu - w[k] * sg[idx][k] * s[k] * EL((c if mult is None else mult[k] / s[k]) * q, tau_, nu_, a)).sum()
        o[a] = 100 * float(num / den)
    return o
ipr = np.flatnonzero(pr); one = np.ones(len(ipr)); rng0 = np.random.default_rng(0); io = np.flatnonzero(pr & (typ == 2)); sub = np.concatenate([np.flatnonzero(pr & (typ != 2)), rng0.choice(io, min(80000, len(io)), replace=False)]); wsub = np.where(typ[sub] == 2, len(io) / min(80000, len(io)), 1.0); sdi = np.searchsorted(pd_, dates.values[sub]); nblk = int(math.ceil(len(pd_) / 63))
OUT = {"universe": WHAT, "bootstrap_draws": NB, "predictions": "see the script header (M1-M5); written after r94 and r95", "rows_pre_test": int(pr.sum()), "rows_test": int(te.sum()), "forms": {}}; ite = np.flatnonzero(te); onet = np.ones(len(ite))
for form, use_s in (("partition form (s = 1)", False), ("nested form", True)):
    par = fit(ipr, one, ho, use_s); Vp = value(par, ipr, one, ho); Vs = value(fit(sub, wsub, ho, use_s), sub, wsub, ho); rng = np.random.default_rng(1); bs = {a: [] for a in AL}
    for _ in range(NB):
        cnt = np.zeros(len(pd_))
        for s_ in rng.integers(0, len(pd_), nblk): cnt[np.arange(s_, s_ + 63) % len(pd_)] += 1
        w = wsub * cnt[sdi]; k_ = w > 0; vb = value(fit(sub[k_], w[k_], ho, use_s), sub[k_], w[k_], ho)
        for a in AL: bs[a].append(vb[a] - Vs[a])
    cm = np.array([par[d][2] for d in range(3)])[typ] * np.exp(par["b"] * hf); post = fit(ite, onet, hf, use_s); Vx = value(post, ite, onet, hf, mult=cm[ite])   # after the fact: slope and tails from test rows, the multiplier actually applied
    row = {"slope_within_type_b": round(par["b"], 3), "slope_within_type_on_test_rows": round(post["b"], 3), "by_type_pre_test": {TY[d]: {"tail_index": round(par[d][0], 2), "tau": round(par[d][1], 3), "multiplier": round(par[d][2], 2)} for d in range(3)}, "by_type_test": {TY[d]: {"tail_index": round(post[d][0], 2), "tau": round(post[d][1], 3), "best_multiplier": round(post[d][2], 2)} for d in range(3)}, "sd_of_log_scale_within_type_pre_test": round(float((par["b"] * ho[pr]).std()), 4)}
    for a in AL:
        Lb = pin(y, mu + sg * tq(a, nu), a); Li = pin(y, mu + sg * cm * tq(a, nu), a); g_ = 100 * float((Lb - Li)[te].sum() / Lb[te].sum()); t_ = dm((Lb - Li)[te], dates[te]); lo, hi = (Vp[a] + np.quantile(bs[a], q_) for q_ in (0.05, 0.95)); seV = (hi - lo) / 3.29; seR = abs(g_ / t_)
        cmo = np.array([par[d][2] for d in range(3)])[typ] * np.exp(par["b"] * ho); Lo = pin(y, mu + sg * cmo * tq(a, nu), a); gpre = 100 * float((Lb - Lo)[pr].sum() / Lb[pr].sum())
        row[str(a)] = {"V_pct": round(Vp[a], 2), "gain_on_the_pre_test_rows_themselves_pct": round(gpre, 2), "V_band_90": [round(float(lo), 2), round(float(hi), 2)], "realized_pct": round(g_, 2), "DM": round(t_, 2), "realized_se": round(seR, 2), "z_of_realized_minus_V": round((g_ - Vp[a]) / math.sqrt(seV ** 2 + seR ** 2), 2), "expression_after_the_fact_pct": round(Vx[a], 2),
                       "realized_on_reaction_days_pct": round(100 * float((Lb - Li)[te & (typ == 0)].sum() / Lb[te & (typ == 0)].sum()), 2), "realized_on_other_days_pct": round(100 * float((Lb - Li)[te & (typ == 2)].sum() / Lb[te & (typ == 2)].sum()), 2)}
    OUT["forms"][form] = row; print("%5.0fs %-24s b %.2f (test %.2f) mult %s | pre-test rows: gain %5.2f, 1%%: %5.2f | 2.5%%: V %5.2f [%5.2f,%5.2f] real %5.2f (DM %4.1f) z %5.2f, after the fact %5.2f | 1%%: V %5.2f [%5.2f,%5.2f] real %5.2f z %5.2f, after the fact %5.2f" % (time.time() - t0, form, par["b"], post["b"], [round(par[d][2], 2) for d in range(3)], row["0.025"]["gain_on_the_pre_test_rows_themselves_pct"], row["0.01"]["gain_on_the_pre_test_rows_themselves_pct"], row["0.025"]["V_pct"], *row["0.025"]["V_band_90"], row["0.025"]["realized_pct"], row["0.025"]["DM"], row["0.025"]["z_of_realized_minus_V"], row["0.025"]["expression_after_the_fact_pct"],
          row["0.01"]["V_pct"], *row["0.01"]["V_band_90"], row["0.01"]["realized_pct"], row["0.01"]["z_of_realized_minus_V"], row["0.01"]["expression_after_the_fact_pct"]), flush=True)
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
