# r93_sp500_weights.py -- THE PORTFOLIO LAW OF r83 WITH UNEQUAL WEIGHTS: THE S&P 500 AND BOOKS CUT FROM IT.
#
# r83 and r92 tested the calendar's value in equal-weighted portfolios of N names and found it gone by N = 10 to 50.
# An index is not equal-weighted. With weights w_i the variance a reporting name adds to the portfolio is
#     (kappa_d - 1) * w_i^2 * sigma_i^2            (d = reaction day or the day after),
# so what matters is not the count of names but the sum of squared weights H: with equal volatilities and pairwise
# correlation rho the earnings share of portfolio variance is x*H / (rho + (1 - rho)*H + x*H), which is r83's formula
# at H = 1/N. The law therefore says a weighted portfolio behaves like 1/H equal-weighted names ("effective names").
#
# Data: every firm that was in the S&P 500 between 2004 and 2024 (CRSP membership, returns, market values; share
# classes of one firm are merged), announcement times from I/B/E/S, the reaction-day rule of r79. Weights use the
# previous day's market value. sigma_i^2 is the firm's mean squared return over its last 63 ordinary days (no
# parameter per firm, so firms that enter late can be used). kappa is the pre-test ratio of squared return to that
# variance on reaction days (and days after) relative to ordinary days, for three size groups (the 20 largest
# members, the next 80, the rest). Every portfolio has its own GARCH-t fitted on its pre-test returns (2005-2015); the
# informed VaR adds the law's variance on the days it is due and removes its pre-test average, as in r83. Nothing is
# fitted at the portfolio level. Test period: 2016-01-04 to 2024-12-31.
#
# Portfolios: the index itself (all members, value-weighted); the K largest members, value-weighted and
# equal-weighted (K = 1, 2, 5, 10, 20, 50, 100); all members with weights proportional to market value to the power
# gamma (0 = equal weights, 0.5, 1.5, 2, 3). For each: effective names, the realized gain of the informed VaR, and V,
# the gain the equation of r83 predicts from the scale ratios alone.
# Part S, spillover. The law assumes the news is the firm's own. On days when one of the ten largest members has its
# reaction day: the index's squared standardized return against the law, and the same for the index with that day's
# reporting firms removed (which the law says should be an ordinary day).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r83 and r92 (equal-weighted US large caps:
# gain 5.3% for one name, 1.2% at N = 5, 0.4% at N = 10, 0.0% from N = 20; pairwise correlation 0.47; variance
# ratios 8.5 and 1.6), and the public fact that the index became more concentrated after 2016. No return of this
# panel had been scored when this was written; only the download counts were known.
#  I1. The rebuilt value-weighted return has a correlation above 0.999 with CRSP's own S&P 500 return.
#  I2. kappa on the reaction day rises as firms get smaller: 4 to 8 for the 20 largest, 7 to 12 for the rest.
#  I3. The index: 50 to 130 effective names on average in the test period, fewer in 2020-2024 than in 2016-2019; the
#      law's earnings share of its variance is under 0.6% and at least 1.5 times larger in 2020-2024 than in
#      2016-2019; V is under 0.1% at 2.5% and the realized gain is within 0.2 points of zero.
#  I4. Concentrated books: value-weighted top 5 gains 0.5% to 2.0% at 2.5%, top 10 gains 0.2% to 1.0%, top 50 under
#      0.2%; the single largest firm gains 3% to 6%; gamma = 3 gains 0.5% to 2.0%.
#  I5. Over all portfolios V and the realized gain have a correlation above 0.9 and a mean absolute gap under 0.5
#      points at 2.5%.
#  I6. Effective names is the right count: every weighted portfolio's gain is within 0.7 points of the gain of the
#      equal-weighted top-K portfolios interpolated (in log K) at its effective names.
#  I7. Spillover: on reaction days of the ten largest firms the index without that day's reporters has a mean squared
#      standardized return between 0.95 and 1.25 times its value on other days (my guess is about 1.1, too small to
#      tell from 1 here); the index itself is 1.05 to 1.30, above the law's 1.02 to 1.06.
# FOLLOW-UP, WRITTEN AFTER THE FIRST RUN (2026-11-20). Seen: every portfolio above is one time series, so no gain is
# more than two standard errors from zero or from V; the realized gain exceeded V in every book of five or more
# effective names; the index without the day's reporters was more volatile on megacap reaction days (ratio 1.14 in
# the test period, 1.28 before it). Two checks with more power:
# Part R, random books. 60 draws each of K firms with a full 2005-2024 history: equal-weighted (K = 1, 2, 5, 10, 20,
#   60), value-weighted and value-squared-weighted (K = 5, 20, 60). Same law, same V.
#  F1. The mean gain of every weighted cell is within 0.3 points of the equal-weighted cells interpolated at its
#      effective names, and V is within 0.4 points of the mean gain in every cell of five or more effective names
#      (within 1.0 point below five).
# Part S, regression. Squared standardized return of the index without the day's reporters, minus one, on the index
#   weight of that day's reporting firms, split into the ten largest firms and all others; all days 2005-2024.
#  F2. The coefficient on the ten largest firms' weight is positive with a t-statistic above 2; the coefficient on
#      the other reporters' weight is smaller.
# SECOND FOLLOW-UP, WRITTEN AFTER THE FIRST FOLLOW-UP (2026-11-20). Seen: F1 and F2 (the law holds in random books;
# the rest of the index is more volatile when one of the ten largest firms reports, coefficient 7.0, t = 2.9, and
# not when other firms report). Megacap reports fall in the same weeks as other news, so:
#  G1. Placebo: the same reporting weight moved five trading days earlier, and five later, has a coefficient under
#      half the main one with |t| below 2.
#  G2. With a dummy for scheduled Fed decision days (test period, where the dates are on file) the coefficient on
#      the ten largest firms stays above 4 with t above 2.
#  G3. An index VaR that scales variance by 1 + b * (reporting weight of the ten largest), b from a pre-test
#      regression (one number fitted at the portfolio level, unlike the law), gains 0.1% to 0.6% at 2.5% in the test
#      period, more than the firm-specific law's 0.21%, with a Diebold-Mariano statistic between 1.6 and 2.5.
#  G4. The index's variance ratio on megacap reaction days is larger in 2020-2024 than in 2016-2019.
# Usage: python tools/r93_sp500_weights.py        (needs data/panels/sp500/ and data/panels/earnings_ibes_sp500/)
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from arch import arch_model
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r93"); os.makedirs(OUTD, exist_ok=True)
START, TEST0 = pd.Timestamp("2005-01-03"), pd.Timestamp("2016-01-01")
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def EL(q, tau, nu, a): x = q / tau; return q * (stats.t.cdf(x, nu) - a) + tau * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
def nw(d, L=10):   # mean over its Newey-West standard error
    dd = d - d.mean(); v = np.mean(dd * dd)
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * np.mean(dd[k:] * dd[:-k])
    return float(d.mean() / math.sqrt(max(v / len(d), 1e-18)))
def cal_flags(a, dates):   # the rule of r79/r82: the announcement day if before 16:00 Eastern, else the next trading day
    n = len(dates); d0 = pd.DatetimeIndex(a.str[:10]); hr = a.str[11:13].astype(int).values + a.str[14:16].astype(int).values / 60.0
    p = np.searchsorted(dates.values, d0.values); inr = p < n; same = np.zeros(len(p), bool); same[inr] = dates.values[p[inr]] == d0.values[inr]; p = p + ((hr >= 16) & same); p = p[p < n]
    r0 = np.zeros(n); r0[p] = 1; r1 = np.zeros(n); r1[p[p + 1 < n] + 1] = 1; return r0, r1

# ---- the firm-by-day tables
D = pd.read_pickle(os.path.join(ROOT, "data", "panels", "sp500", "daily.pkl")); D["capl"] = D.groupby("permno")["cap"].shift(1); D = D[np.isfinite(D.ret) & np.isfinite(D.capl) & (D.capl > 0)]
D["rc"] = D.ret * D.capl; D["mc"] = np.where(D.member, D.capl, 0.0); Fm = D.groupby(["date", "permco"]).agg(rc=("rc", "sum"), capl=("capl", "sum"), mc=("mc", "sum")).reset_index(); Fm["ret"] = Fm.rc / Fm.capl
R = Fm.pivot(index="date", columns="permco", values="ret"); dates = R.index; firms = R.columns.values; CW = Fm.pivot(index="date", columns="permco", values="mc").reindex(index=dates, columns=firms).fillna(0.0).values; CAPL = Fm.pivot(index="date", columns="permco", values="capl").reindex(index=dates, columns=firms).values; R = R.values; T, M = R.shape
ED = os.path.join(ROOT, "data", "panels", "earnings_ibes_sp500"); pc = D.groupby("permco")["permno"].unique(); F0 = np.zeros((T, M)); F1 = np.zeros((T, M)); nocal = 0
for j, f in enumerate(firms):
    a = sorted(set(s for p in pc[f] if os.path.exists(os.path.join(ED, "%d.csv" % p)) for s in pd.read_csv(os.path.join(ED, "%d.csv" % p))["announcement"].astype(str)))
    if not a: nocal += 1; continue
    F0[:, j], F1[:, j] = cal_flags(pd.Series(a), dates)
F1 = F1 * (F0 == 0); S2 = np.full((T, M), np.nan)
for j in range(M):   # the firm's variance: mean squared return over its last 63 ordinary days, known the day before
    o = np.flatnonzero(np.isfinite(R[:, j]) & (F0[:, j] == 0) & (F1[:, j] == 0))
    if len(o) >= 40: S2[:, j] = pd.Series(R[o, j] ** 2, index=o).rolling(63, min_periods=40).mean().reindex(np.arange(T)).ffill().shift(1).values
mem = (CW > 0) & np.isfinite(R); rank = np.where(mem, (-np.where(mem, CW, -1.0)).argsort(1).argsort(1), M); grp = np.where(rank < 20, 0, np.where(rank < 100, 1, 2)); use = dates >= START; pre = use & (dates < TEST0); te = use & (dates >= TEST0)
U = R ** 2 / S2; okU = mem & np.isfinite(U) & (S2 > 0); KAP = {}
for g, lab in enumerate(("20 largest", "next 80", "the rest")):
    b = okU & (grp == g) & pre[:, None]; base = U[b & (F0 == 0) & (F1 == 0)].mean(); KAP[lab] = {"reaction day": float(U[b & (F0 == 1)].mean() / base), "day after": float(U[b & (F1 == 1)].mean() / base), "reaction days pre-test": int((b & (F0 == 1)).sum())}
k0 = np.array([KAP[l]["reaction day"] for l in KAP])[grp]; k1 = np.array([KAP[l]["day after"] for l in KAP])[grp]; E = np.where(okU, S2 * ((k0 - 1) * F0 + (k1 - 1) * F1), 0.0)   # variance a firm adds, before its squared weight

def weights(K=None, gamma=1.0, drop=None):
    b = mem & (rank < (K or M));
    if drop is not None: b = b & ~drop
    w = np.where(b, np.where(b, CW, 1.0) ** gamma, 0.0); return w / np.maximum(w.sum(1, keepdims=True), 1e-300)
def run(w): return core((w * np.nan_to_num(R)).sum(1)[use], ((w ** 2) * E).sum(1)[use], (w ** 2).sum(1)[use])
def core(rp, ex, H):
    p = pre[use]; t = te[use]
    pr = arch_model(rp[p], vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False).params; sig = np.asarray(arch_model(rp, vol="GARCH", p=1, q=1, dist="t", rescale=False).fix(pr.values).conditional_volatility)
    mu, nu = float(pr["mu"]), float(pr["nu"]); v = sig ** 2; c = ex[p].mean() / v[p].mean(); vi = np.maximum(v + ex - c * v, 0.25 * v); s = np.sqrt(vi / v); z2 = (rp - mu) ** 2 / v
    out = {"effective_names_test": round(float(1 / H[t].mean()), 1), "effective_names_2016_2019": round(float(1 / H[t & (dates[use] < "2020-01-01")].mean()), 1), "effective_names_2020_2024": round(float(1 / H[t & (dates[use] >= "2020-01-01")].mean()), 1),
           "law_share_of_variance_test_pct": round(100 * float(ex[t].sum() / v[t].sum()), 3), "law_share_2016_2019_pct": round(100 * float(ex[t & (dates[use] < "2020-01-01")].sum() / v[t & (dates[use] < "2020-01-01")].sum()), 3),
           "law_share_2020_2024_pct": round(100 * float(ex[t & (dates[use] >= "2020-01-01")].sum() / v[t & (dates[use] >= "2020-01-01")].sum()), 3), "law_share_pre_test_pct": round(100 * float(c), 3)}
    xx = (ex / v - c)[t]; yy = (z2 - 1)[t]; out["slope_of_excess_squared_return_on_law"] = round(float(np.polyfit(xx, yy, 1)[0]), 2) if xx.std() > 0 else None
    for a in AL:
        q = tq(a, nu); Lu = pin(rp, mu + sig * q, a); Li = pin(rp, mu + sig * s * q, a); k = math.sqrt(nu / (nu - 2)); lu = sig * s * EL(q / s, 1 / k, nu, a); li = sig * s * EL(q, 1 / k, nu, a)
        out[str(a)] = {"gain_pct": round(100 * float((Lu - Li)[t].sum() / Lu[t].sum()), 2), "DM": round(nw((Lu - Li)[t]), 2), "V_pct": round(100 * float((lu - li)[t].sum() / lu[t].sum()), 2),
                       "breach_plain_pct": round(100 * float((rp <= mu + sig * q)[t].mean()), 2), "breach_informed_pct": round(100 * float((rp <= mu + sig * s * q)[t].mean()), 2)}
    return out, dict(rp=rp, mu=mu, nu=nu, sig=sig, s=s, ex=ex, v=v, c=c, z2=z2, t=t, p=p)

PORT = {"index (all members, value-weighted)": weights()}
for K in (1, 2, 5, 10, 20, 50, 100): PORT["top %d, equal-weighted" % K] = weights(K, 0.0)
for K in (5, 10, 20, 50, 100): PORT["top %d, value-weighted" % K] = weights(K, 1.0)
for g in (0.0, 0.5, 1.5, 2.0, 3.0): PORT["all members, weight = value^%.1f" % g] = weights(None, g)
RES = {}; KEEP = {}
for name, w in PORT.items():
    RES[name], KEEP[name] = run(w); r = RES[name]; print("%-38s eff %6.1f  share %6.3f%%  2.5%%: gain %5.2f (DM %5.2f) V %5.2f | 1%%: gain %5.2f V %5.2f | slope %s" % (name, r["effective_names_test"], r["law_share_of_variance_test_pct"], r["0.025"]["gain_pct"], r["0.025"]["DM"], r["0.025"]["V_pct"], r["0.01"]["gain_pct"], r["0.01"]["V_pct"], r["slope_of_excess_squared_return_on_law"]), flush=True)
# ---- checks and summaries
I = pd.read_csv(os.path.join(ROOT, "data", "panels", "sp500", "index.csv"), parse_dates=["date"]).set_index("date").reindex(dates[use]); ix = KEEP["index (all members, value-weighted)"]; okI = np.isfinite(I.vwretd.values)
names = list(RES); gV = np.array([RES[n]["0.025"]["V_pct"] for n in names]); gR = np.array([RES[n]["0.025"]["gain_pct"] for n in names]); gV1 = np.array([RES[n]["0.01"]["V_pct"] for n in names]); gR1 = np.array([RES[n]["0.01"]["gain_pct"] for n in names])
ewK = [(RES[n]["effective_names_test"], RES[n]["0.025"]["gain_pct"]) for n in names if "equal-weighted" in n] + [(RES["all members, weight = value^0.0"]["effective_names_test"], RES["all members, weight = value^0.0"]["0.025"]["gain_pct"])]; ewK.sort()
curve = lambda n: float(np.interp(math.log(n), [math.log(a) for a, _ in ewK], [b for _, b in ewK])); I6 = {n: {"effective_names": RES[n]["effective_names_test"], "gain_pct": RES[n]["0.025"]["gain_pct"], "equal_weighted_curve_at_effective_names_pct": round(curve(RES[n]["effective_names_test"]), 2)} for n in names if "equal-weighted" not in n and "^0.0" not in n}
# ---- Part S: spillover
MD = ((rank < 10) & (F0 == 1)).any(1)[use]; rep = (F0 == 1) | (F1 == 1); _, xr = run(weights(drop=rep)); t = ix["t"]; S = {"megacap_reaction_days_test": int((MD & t).sum()), "megacap_reaction_days_pre_test": int((MD & ix["p"]).sum())}
for lab, k in (("index", ix), ("index without the day's reporting firms", xr)):
    for per, m in (("test", t), ("pre_test", ix["p"])):
        a_, b_ = k["z2"][m & MD], k["z2"][m & ~MD]; S["%s, %s" % (lab, per)] = {"mean_squared_standardized_return_megacap_days": round(float(a_.mean()), 3), "other_days": round(float(b_.mean()), 3), "ratio": round(float(a_.mean() / b_.mean()), 3),
                                              "t_of_difference": round(float((a_.mean() - b_.mean()) / math.sqrt(a_.var() / len(a_) + b_.var() / len(b_))), 2)}
S["law_for_the_index_on_megacap_days_test"] = round(float((1 + (ix["ex"] / ix["v"] - ix["c"]))[t & MD].mean() / (1 + (ix["ex"] / ix["v"] - ix["c"]))[t & ~MD].mean()), 3)
for a in AL: q = tq(a, ix["nu"]); S["index_breach_on_megacap_days_test_%s" % a] = {"plain_pct": round(100 * float((ix["rp"] <= ix["mu"] + ix["sig"] * q)[t & MD].mean()), 2), "informed_pct": round(100 * float((ix["rp"] <= ix["mu"] + ix["sig"] * ix["s"] * q)[t & MD].mean()), 2)}
import statsmodels.api as sm
w_ix = PORT["index (all members, value-weighted)"]; Wm = (w_ix * ((rank < 10) & (F0 == 1))).sum(1)[use]; Wo = (w_ix * ((rank >= 10) & (F0 == 1) & mem)).sum(1)[use]; X = sm.add_constant(np.column_stack([Wm, Wo])); S["regression_on_reporting_weight"] = {"mean_weight_of_ten_largest_reporting_on_their_days": round(float(Wm[Wm > 0].mean()), 4), "mean_weight_of_other_reporters": round(float(Wo.mean()), 4)}
for lab, k in (("index", ix), ("index without the day's reporting firms", xr)):
    for per, m in (("all days", np.ones(len(Wm), bool)), ("test", t), ("pre_test", ix["p"])):
        f = sm.OLS(k["z2"][m] - 1, X[m]).fit(cov_type="HAC", cov_kwds={"maxlags": 10}); S["regression_on_reporting_weight"]["%s, %s" % (lab, per)] = {"ten_largest": round(float(f.params[1]), 2), "t": round(float(f.tvalues[1]), 2), "others": round(float(f.params[2]), 2), "t_others": round(float(f.tvalues[2]), 2)}
# second follow-up: placebo, Fed days, the index VaR with a spillover term, sub-periods
sh = lambda x, k: np.concatenate([np.zeros(k), x[:-k]]) if k > 0 else np.concatenate([x[-k:], np.zeros(-k)]); du = pd.DatetimeIndex(dates[use]); y_ = xr["z2"] - 1; G2_ = {}
f = sm.OLS(y_, sm.add_constant(np.column_stack([Wm, sh(Wm, 5), sh(Wm, -5), Wo]))).fit(cov_type="HAC", cov_kwds={"maxlags": 10}); G2_["placebo_all_days_index_without_reporters"] = {"ten_largest": round(float(f.params[1]), 2), "t": round(float(f.tvalues[1]), 2), "five_days_later": round(float(f.params[2]), 2), "t_later": round(float(f.tvalues[2]), 2), "five_days_earlier": round(float(f.params[3]), 2), "t_earlier": round(float(f.tvalues[3]), 2)}
fd = pd.DatetimeIndex(json.load(open(os.path.join(ROOT, "data", "calendars", "fomc_scheduled_decision_dates.json")))); fed = np.isin(du.values, fd.values).astype(float); f = sm.OLS(y_[t], sm.add_constant(np.column_stack([Wm, Wo, fed]))[t]).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
G2_["with_fed_decision_days_test_index_without_reporters"] = {"ten_largest": round(float(f.params[1]), 2), "t": round(float(f.tvalues[1]), 2), "fed_day": round(float(f.params[3]), 2), "t_fed": round(float(f.tvalues[3]), 2), "fed_days": int(fed[t].sum()), "fed_days_that_are_megacap_days": int((fed[t] * MD[t]).sum())}
pp = ix["p"]; bs = float(np.polyfit(Wm[pp], ix["z2"][pp] - 1, 1)[0]); ss = np.sqrt(np.maximum(1 + bs * (Wm - Wm[pp].mean()), 0.25)); G2_["index_var_with_spillover_term"] = {"b_pre_test": round(bs, 2)}
for a in AL:
    q = tq(a, ix["nu"]); Lu = pin(ix["rp"], ix["mu"] + ix["sig"] * q, a); Li = pin(ix["rp"], ix["mu"] + ix["sig"] * ss * q, a); G2_["index_var_with_spillover_term"][str(a)] = {"gain_pct": round(100 * float((Lu - Li)[t].sum() / Lu[t].sum()), 2), "DM": round(nw((Lu - Li)[t]), 2),
        "breach_megacap_days_plain_pct": round(100 * float((ix["rp"] <= ix["mu"] + ix["sig"] * q)[t & MD].mean()), 2), "breach_megacap_days_informed_pct": round(100 * float((ix["rp"] <= ix["mu"] + ix["sig"] * ss * q)[t & MD].mean()), 2)}
for lab, m in (("2005-2015", pp), ("2016-2019", t & (du < "2020-01-01")), ("2020-2024", t & (du >= "2020-01-01"))):
    G2_["megacap_days_%s" % lab] = {"days": int((m & MD).sum()), "mean_reporting_weight_pct": round(100 * float(Wm[m & MD].mean()), 2), "index_variance_ratio": round(float(ix["z2"][m & MD].mean() / ix["z2"][m & ~MD].mean()), 3), "without_reporters_variance_ratio": round(float(xr["z2"][m & MD].mean() / xr["z2"][m & ~MD].mean()), 3)}
S["second_follow_up"] = G2_; print(json.dumps(G2_, indent=1), flush=True)
# ---- Part R: random books of firms with a full history (follow-up)
full = np.flatnonzero(np.isfinite(R[use]).all(0) & np.isfinite(CAPL[use]).all(0))
rall = np.array([np.searchsorted(np.sort(-CW[i][mem[i]]), -np.nan_to_num(CAPL[i]), side="left") for i in range(T)])   # for every firm, the number of members with a larger market value
ga = np.where(rall < 20, 0, np.where(rall < 100, 1, 2)); EA = np.where(np.isfinite(U) & (S2 > 0) & np.isfinite(CAPL), S2 * ((np.array([KAP[l]["reaction day"] for l in KAP])[ga] - 1) * F0 + (np.array([KAP[l]["day after"] for l in KAP])[ga] - 1) * F1), 0.0)
def book(cols, gamma):
    w = np.ones((T, len(cols))) if gamma == 0 else CAPL[:, cols] ** gamma; w = w / w.sum(1, keepdims=True); return core((w * R[:, cols]).sum(1)[use], ((w ** 2) * EA[:, cols]).sum(1)[use], (w ** 2).sum(1)[use])[0]
CELLS = [("equal-weighted", 0, K) for K in (1, 2, 5, 10, 20, 60)] + [("value-weighted", 1, K) for K in (5, 20, 60)] + [("value-squared-weighted", 2, K) for K in (5, 20, 60)]; RB = {}; DR = []
for lab, g, K in CELLS:
    rr = []
    for sd in range(int(os.environ.get("R93_DRAWS", "60"))):
        try: rr.append(book(np.random.default_rng(1000 * K + sd).choice(full, K, replace=False), g))
        except Exception: pass
    gn = np.array([r["0.025"]["gain_pct"] for r in rr]); RB["%s, %d firms" % (lab, K)] = {"draws": len(rr), "effective_names": round(float(np.mean([r["effective_names_test"] for r in rr])), 1), "gain_pct": round(float(gn.mean()), 2), "se_across_draws": round(float(gn.std() / math.sqrt(len(gn))), 2), "V_pct": round(float(np.mean([r["0.025"]["V_pct"] for r in rr])), 2),
        "gain_1pct": round(float(np.mean([r["0.01"]["gain_pct"] for r in rr])), 2), "V_1pct": round(float(np.mean([r["0.01"]["V_pct"] for r in rr])), 2), "slope_of_excess_squared_return_on_law": round(float(np.median([r["slope_of_excess_squared_return_on_law"] for r in rr])), 2)}
    DR += [(lab, r["effective_names_test"], r["0.025"]["gain_pct"], r["0.025"]["V_pct"], K) for r in rr]; print("%-32s %s" % ("%s, %d firms" % (lab, K), RB["%s, %d firms" % (lab, K)]), flush=True)
ewc = sorted((v["effective_names"], v["gain_pct"]) for k_, v in RB.items() if k_.startswith("equal")); cv = lambda n: float(np.interp(math.log(n), [math.log(a) for a, _ in ewc], [b for _, b in ewc]))
for lab, g, K in CELLS:   # the equal-weighted curve read at each draw's effective names, for the weighted cells
    if g: RB["%s, %d firms" % (lab, K)]["equal_weighted_curve_at_effective_names_pct"] = round(float(np.mean([cv(r_[1]) for r_ in DR if r_[0] == lab and r_[4] == K])), 2)
dg = np.array([r_[2] for r_ in DR]); dv = np.array([r_[3] for r_ in DR]); RBS = {"firms_with_a_full_history": int(len(full)), "draws_in_all": len(DR), "correlation_of_gain_and_V_across_draws": round(float(np.corrcoef(dg, dv)[0, 1]), 3), "slope_of_gain_on_V": round(float(np.polyfit(dv, dg, 1)[0]), 2),
       "correlation_across_cell_means": round(float(np.corrcoef([v["gain_pct"] for v in RB.values()], [v["V_pct"] for v in RB.values()])[0, 1]), 3)}
yr = pd.DatetimeIndex(dates[use]).year; H_ix = (PORT["index (all members, value-weighted)"] ** 2).sum(1)[use]; BYY = {int(y_): {"effective_names": round(float(1 / H_ix[yr == y_].mean()), 1), "law_share_of_variance_pct": round(100 * float(ix["ex"][yr == y_].sum() / ix["v"][yr == y_].sum()), 3)} for y_ in sorted(set(yr))}
OUT = {"universe": "S&P 500 members, CRSP", "firms": int(M), "firms_without_a_calendar": nocal, "days_used": int(use.sum()), "pre_test_days": int(pre.sum()), "test_days": int(te.sum()), "test_from": str(dates[te][0].date()), "test_to": str(dates[te][-1].date()),
       "predictions_written_before_run": "see the script header (I1-I7)", "correlation_with_crsp_sp500_value_weighted_return": round(float(np.corrcoef(ix["rp"][okI], I.vwretd.values[okI])[0, 1]), 5), "mean_absolute_daily_gap_pct": round(float(np.abs(ix["rp"][okI] - I.vwretd.values[okI]).mean()), 4),
       "variance_ratio_pre_test_by_size_group": KAP, "portfolios": RES, "V_against_realized": {"0.025": {"correlation": round(float(np.corrcoef(gV, gR)[0, 1]), 3), "mean_absolute_gap_points": round(float(np.abs(gV - gR).mean()), 2)}, "0.01": {"correlation": round(float(np.corrcoef(gV1, gR1)[0, 1]), 3), "mean_absolute_gap_points": round(float(np.abs(gV1 - gR1).mean()), 2)}},
       "weighted_against_equal_weighted_curve": I6, "index_by_year": BYY, "spillover": S, "random_books_follow_up": {"summary": RBS, "cells": RB}}
json.dump(OUT, open(os.path.join(OUTD, "sp500.json"), "w"), indent=1); print(json.dumps({k: OUT[k] for k in ("correlation_with_crsp_sp500_value_weighted_return", "mean_absolute_daily_gap_pct", "variance_ratio_pre_test_by_size_group", "V_against_realized", "weighted_against_equal_weighted_curve", "spillover", "random_books_follow_up")}, indent=1))
