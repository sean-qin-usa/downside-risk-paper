# r85_two_clock.py -- EARNINGS-DAY RISK THAT DIFFERS BY FIRM AND BY QUARTER: A SECOND CLOCK FOR ANNOUNCEMENT VARIANCE.
#
# r78-r82 give every name the same multiplier on its reaction day (3.9 on the CRSP panel, 2.7 on US large caps). Two
# questions about the structure of the announcement variance, and one benchmark.
#   (1) Multiple or addition? A multiplier says the announcement variance is a multiple of today's ordinary variance.
#       The alternative is that the announcement adds a variance of its own: Var(r) = s2_t + J2, with s2_t the ordinary
#       variance (the masked filter) and J2 the variance of the announcement return. Then the standardized residual has
#       variance 1 + J2/s2_t, which falls when ordinary volatility is high. That is testable without a model.
#   (2) Does J2 belong to the firm? A second recursion, in event time: J2 for firm i at announcement j is a weighted
#       average of a panel prior and the firm's own last eight announcements,
#           J2_ij = (1 - w) * theta * sbar2_i + w * H_ij,     H_ij = mean of max((r - mu)^2 - s2, 0) over the last eight,
#       sbar2_i the firm's pre-test mean ordinary variance. theta and w are two numbers per panel, pre-test.
#   (3) The benchmark (CRSP panel): the announcement variance implied by option prices the evening before
#       (code/panels/download_optionm_earnings.py), scaled by one pre-test number b.
# Reaction-day VaR models, all with the masked filter's mean and Student-t, parameters chosen on pre-test reaction days
# by 2.5% pinball loss:
#   M0  c0 * s_t                               the constant multiplier of r78-r82
#   M1  sqrt(s2_t + theta * sbar2_i)           addition, the same for every announcement of a firm
#   M2  sqrt(s2_t + J2_ij)                     addition with the firm's own history (the second clock)
#   M3  s_t * sqrt((1-w) c0^2 + w * U_ij)      multiple with the firm's own history (U_ij = mean of the last eight u^2)
#   M4  sqrt(s2_t + b * J2_opt)                option-implied (M1 where no option data)
#   M5  sqrt(s2_t + (1-v) J2_ij + v b J2_opt)  both
# Other days are the masked calendar GARCH-t in every case.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r78-r84; of the option data only its coverage
# (90% of reaction days) and the median implied announcement standard deviation (4.3%).
#  W1. The mean squared standardized residual on reaction days falls with ordinary volatility: in the lowest third of
#      s_t relative to the firm's usual level it is at least twice what it is in the highest third, pre-test and test,
#      on both panels.
#  W2. M1 beats M0 on reaction days at 2.5% by at least 1.5% (DM above 2) on both panels.
#  W3. M2 beats M1 on reaction days by at least 1.5% (DM above 2), with w at least 0.3, on both panels.
#  W4. M2 beats M3 on reaction days on both panels.
#  W5. On reaction days M2 is not worse than any flexible model with calendar inputs (DM above -2), and over all days
#      it gains 0.1% to 0.6% over the masked calendar GARCH-t.
#  W6. (CRSP) M4 beats M2 on reaction days by at least 3% (DM above 2), and M5 beats both.
#
# FOLLOW-UP ON THE BENCHMARK, WRITTEN AFTER W6 FAILED WITH THE STANDARDIZED OPTIONS AND BEFORE THIS WAS RUN (2026-11-20).
# The 30-day and 60-day standardized volatilities are interpolations and gave an announcement standard deviation about
# half of what the returns show. Two estimates from the option chain itself replace them: "one expiry" = the implied
# variance to the nearest expiry on or after the reaction day less the ordinary variance of the days to that expiry,
# max(IV1^2 T1 - n1 s2_t, 0); "two expiries" = the term-structure formula with the two nearest actual expiries.
# M4 and M5 use the one-expiry estimate; the two-expiry estimate is reported beside it.
#  W7. With the one-expiry estimate M4 beats the constant multiplier on reaction days by at least 5% (DM above 2) and is
#      within 5% of the second clock M2 either way; M5 beats M2 by at least 2% (DM above 2) with a weight v on the
#      options between 0.3 and 0.7.
#
# SIZE SPLIT, WRITTEN BEFORE IT WAS RUN (2026-11-20). W7 failed on the whole CRSP panel, where 126 of 186 names are small
# companies with thin options, and the result runs against Govindaraj, Jin, Livnat and Zhao, who find that an
# option-implied measure adds to the firm's announcement history. R85_COHORT=largecap or smallcap restricts the panel
# (cohorts of crsp_panel_chars.csv); every parameter is chosen again on that cohort's pre-test reaction days.
#  W8. Large companies: the option-implied model M4 is within 5% of the second clock M2 on reaction days either way,
#      and the combination M5 puts a weight of at least 0.3 on the options and beats M2 (DM above 2). Small companies:
#      M4 is worse than M2 by more than 10% and the weight on the options is zero.
# Usage: python tools/r85_two_clock.py canon | us_largecap
import os, sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r85"); os.makedirs(OUTD, exist_ok=True)
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0)].reset_index(drop=True)
COH = os.environ.get("R85_COHORT")
if COH: ch = pd.read_csv(os.path.join(os.environ["GBC_PROJ"], "crsp_panel_chars.csv")); A = A[A["name"].isin(set(ch.loc[ch.cohort == COH, "permno"].astype(str)))].reset_index(drop=True); R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json")))
ordv = A[(A.r0 != 1) & (A.r1 != 1) & ~A.test]; sbar2 = ordv.groupby("name")["msig"].apply(lambda s: float(np.mean(s ** 2))); smed = ordv.groupby("name")["msig"].median()
E = A[A.r0 == 1].copy(); E["e2"] = (E.y - E.mmu) ** 2; E["s2"] = E.msig ** 2; E["u2"] = E.e2 / E.s2; E["sbar2"] = E["name"].map(sbar2); E["rel"] = E.msig / E["name"].map(smed)
g = E.groupby("name"); E["H"] = g["e2"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()) - g["s2"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()); E["H"] = E.H.clip(lower=0); E["U"] = g["u2"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
if WHAT == "canon":
    O = pd.read_pickle(os.path.join(ROOT, "data", "panels", "optionm_earnings_canon.pkl")); O = O.assign(**{c_: pd.to_numeric(O[c_], errors="coerce").astype(float) for c_ in ("J2_opt", "J2_ts", "iv1", "T1")})
    E = E.merge(O[["name", "date", "J2_opt", "J2_ts", "iv1", "T1"]], on=["name", "date"], how="left"); E["J2_std"] = E["J2_opt"]; E["J2_opt"] = np.maximum(1e4 * E.iv1 ** 2 * E.T1 - np.round(E.T1 * 252) * E.s2, 0)   # one expiry
else: E["J2_opt"] = np.nan; E["J2_std"] = np.nan; E["J2_ts"] = np.nan
E = E[np.isfinite(E.H) & np.isfinite(E.U)].reset_index(drop=True); pre = ~E.test.values; te = E.test.values
y, mu, nu, s2, sb, H, U, Jo = (E[c].values for c in ("y", "mmu", "mnu", "s2", "sbar2", "H", "U", "J2_opt")); ho = np.isfinite(Jo); b25 = tq(0.025, nu)
loss = lambda sd, f: pin(y[f], mu[f] + sd[f] * b25[f], 0.025).sum()
def grid(fun, grids):   # exhaustive search on pre-test reaction days
    best = (1e30, None)
    for p in np.array(np.meshgrid(*grids)).T.reshape(-1, len(grids)):
        L = loss(fun(*p), pre)
        if L < best[0]: best = (L, tuple(float(x) for x in p))
    return best[1]
CG = np.round(np.arange(1.0, 7.001, 0.1), 2); TG = np.round(np.arange(0.5, 40.001, 0.5), 2); WG = np.round(np.arange(0.0, 1.0001, 0.1), 2); BG = np.round(np.arange(0.2, 4.001, 0.1), 2)
m0 = lambda c: c * np.sqrt(s2); m1 = lambda th: np.sqrt(s2 + th * sb); m2 = lambda th, w: np.sqrt(s2 + (1 - w) * th * sb + w * H); m3 = lambda c, w: np.sqrt(s2 * ((1 - w) * c * c + w * U))
(c0,) = grid(m0, [CG]); (th1,) = grid(m1, [TG]); th2, w2 = grid(m2, [TG, WG]); c3, w3 = grid(m3, [CG, WG]); SD = {"M0 constant multiple": m0(c0), "M1 addition, constant": m1(th1), "M2 addition, firm history": m2(th2, w2), "M3 multiple, firm history": m3(c3, w3)}
PAR = {"M0": {"c0": c0}, "M1": {"theta": th1}, "M2": {"theta": th2, "w": w2}, "M3": {"c0": c3, "w": w3}}
if ho.any():
    m4 = lambda b: np.where(ho, np.sqrt(s2 + b * np.nan_to_num(Jo)), m1(th1)); (b4,) = grid(m4, [BG]); m5 = lambda v: np.where(ho, np.sqrt(s2 + (1 - v) * ((1 - w2) * th2 * sb + w2 * H) + v * b4 * np.nan_to_num(Jo)), m2(th2, w2)); (v5,) = grid(m5, [WG])
    for lab_, col_ in (("M4s option-implied, standardized 30/60", "J2_std"), ("M4t option-implied, two expiries", "J2_ts")):
        jj = E[col_].values; hh = np.isfinite(jj); mm = lambda b: np.where(hh, np.sqrt(s2 + b * np.nan_to_num(jj)), m1(th1)); (bb,) = grid(mm, [BG]); SD[lab_] = mm(bb); PAR[lab_[:3]] = {"b": bb, "share_zero": round(float((jj[hh] == 0).mean()), 3), "median_sd": round(float(np.sqrt(np.median(jj[hh]))), 2)}
    PAR["M4 one expiry"] = {"share_zero": round(float((Jo[ho] == 0).mean()), 3), "median_sd": round(float(np.sqrt(np.median(Jo[ho]))), 2), "median_realized_abs_move": round(float(np.median(np.sqrt(E.e2.values[ho]))), 2)}
    SD["M4 addition, option-implied"] = m4(b4); SD["M5 addition, history and options"] = m5(v5); PAR["M4"] = {"b": b4}; PAR["M5"] = {"v": v5}
dates = pd.DatetimeIndex(E.date.values)
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
edge = lambda Lr, Lm, mask: {"edge_pct": round(100 * float((Lr - Lm)[mask].sum() / Lr[mask].sum()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
ter = lambda f: {lab: round(float(E.u2.values[f & m_].mean()), 2) for lab, m_ in zip(("lowest third", "middle third", "highest third"), (lambda r, q: (r <= q[0], (r > q[0]) & (r <= q[1]), r > q[1]))(E.rel.values, np.quantile(E.rel.values[f], [1 / 3, 2 / 3])))}
OUT = {"universe": WHAT, "reaction_days": {"pre-test": int(pre.sum()), "test": int(te.sum()), "test with option data": int((te & ho).sum())}, "predictions_written_before_run": "see the script header (W1-W6; W7 written after W6 failed with the standardized options)", "parameters_pre_test": PAR,
       "mean_squared_standardized_residual_by_relative_volatility": {"pre-test": ter(pre), "test": ter(te)}, "regression_test_log_u2_on_log_relative_volatility": None}
ok = te & (E.u2.values > 0); sl = stats.linregress(np.log(E.rel.values[ok]), np.log(E.u2.values[ok])); OUT["regression_test_log_u2_on_log_relative_volatility"] = {"slope": round(float(sl.slope), 3), "se": round(float(sl.stderr), 3), "note": "a pure multiple gives 0; a pure addition gives about -2 when the announcement variance dominates"}
# all-days effect: reaction days' share of the masked calendar model's total loss on the common test rows
T = A[A.test & A.U]; cm = R82["multipliers_c0_c1"]["masked calendar GARCH-t"]; R0 = (T.r0 == 1).values; R1 = ((T.r1 == 1) & (T.r0 != 1)).values
for j, a in enumerate(AL):
    ba = tq(a, nu); Lm = {m: pin(y, mu + sd * ba, a) for m, sd in SD.items()}; Lmask = pin(T.y.values, T.mmu.values + T.msig.values * np.where(R0, cm[0], np.where(R1, cm[1], 1.0)) * tq(a, T.mnu.values), a); tot = float(Lmask.sum()); base = float(Lm["M0 constant multiple"][te].sum())
    flex = {nm: pin(y, E["%s__%s" % (nm, a)].values, a) for nm in ("trees, calendar inputs", "trees on robust scale, calendar inputs", "trees on masked scale, calendar inputs", "neural network, calendar inputs")}; tf = te & np.all([np.isfinite(v) for v in flex.values()], axis=0)
    OUT[str(a)] = {"vs_M0_on_reaction_days": {m: edge(Lm["M0 constant multiple"], L_, te) for m, L_ in Lm.items() if not m.startswith("M0")}, "M2_vs": {m: edge(L_, Lm["M2 addition, firm history"], te) for m, L_ in Lm.items() if not m.startswith("M2")},
                   "M2_vs_flexible_on_reaction_days": {nm: edge(v, Lm["M2 addition, firm history"], tf) for nm, v in flex.items()}, "M0_vs_flexible_on_reaction_days": {nm: edge(v, Lm["M0 constant multiple"], tf) for nm, v in flex.items()},
                   "all_days_gain_over_masked_calendar_garch_t_pct": {m: round(100 * float((Lm["M0 constant multiple"] - L_)[te].sum()) / tot, 2) for m, L_ in Lm.items() if not m.startswith("M0")}, "reaction_days_share_of_total_loss_pct": round(100 * base / tot, 1),
                   "breach_pct_reaction_days": {m: round(100 * float((y <= mu + sd * ba)[te].mean()), 2) for m, sd in SD.items()}}
    if ho.any():
        f = te & ho; OUT[str(a)]["on_reaction_days_with_option_data"] = {"M4_vs_M2": edge(Lm["M2 addition, firm history"], Lm["M4 addition, option-implied"], f), "M5_vs_M4": edge(Lm["M4 addition, option-implied"], Lm["M5 addition, history and options"], f), "M5_vs_M2": edge(Lm["M2 addition, firm history"], Lm["M5 addition, history and options"], f), "M4_vs_M0": edge(Lm["M0 constant multiple"], Lm["M4 addition, option-implied"], f)}
OUT["cohort"] = COH; OUT["names"] = int(A["name"].nunique()); json.dump(OUT, open(os.path.join(OUTD, WHAT + ("_" + COH if COH else "") + ".json"), "w"), indent=1); f_ = lambda e: "%+.1f%% (%s)" % (e["edge_pct"], e["DM"])
print("######## %s: %s; parameters %s" % (WHAT, OUT["reaction_days"], PAR)); print("u2 by relative volatility:", OUT["mean_squared_standardized_residual_by_relative_volatility"], OUT["regression_test_log_u2_on_log_relative_volatility"])
for a in ("0.025", "0.01"):
    B = OUT[a]; print("--- %s reaction days, against M0: %s" % (a, {m[:3]: f_(v) for m, v in B["vs_M0_on_reaction_days"].items()})); print("    M2 against: %s" % {m[:3]: f_(v) for m, v in B["M2_vs"].items()})
    print("    M2 against flexible: %s | M0 against flexible: %s" % ({m: f_(v) for m, v in B["M2_vs_flexible_on_reaction_days"].items()}, {m: f_(v) for m, v in B["M0_vs_flexible_on_reaction_days"].items()}))
    print("    all-days gain over the masked calendar GARCH-t: %s; reaction days are %s%% of its loss; breach %s" % (B["all_days_gain_over_masked_calendar_garch_t_pct"], B["reaction_days_share_of_total_loss_pct"], B["breach_pct_reaction_days"]))
    if "on_reaction_days_with_option_data" in B: print("    with option data: %s" % {k: f_(v) for k, v in B["on_reaction_days_with_option_data"].items()})
