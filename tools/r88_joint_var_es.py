# r88_joint_var_es.py -- THE CALENDAR RESULT ON THE JOINT VaR AND EXPECTED SHORTFALL LOSS.
#
# r79-r87 score VaR by quantile loss. The regulatory pair is VaR and expected shortfall, scored jointly by the FZ0 loss
# of Fissler and Ziegel (Patton, Ziegel and Chen 2019). Here every model that has an ES forecast is scored on FZ0 at 1%
# and 2.5%, without the calendar and with it, on the common test rows of r81/r79:
#   stored forecasts (no calendar): GARCH-t, GJR-skew-t, Beta-t-EGARCH, FHS, pooled trees, trees on the robust scale,
#       both with the EVT tail (and the pooled GARCH-EVT on the CRSP panel)
#   the same with the calendar as two multipliers on VaR and ES alike (r82's pre-test multipliers; the trees take GARCH-t's)
#   the masked calendar GARCH-t and the two-fix GARCH-t, with the Student-t's own ES
# The neural network and the trees retrained with calendar inputs have no ES forecast and are not in this comparison.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r79-r87 (quantile loss only).
#  Z1. Every model has a lower mean FZ0 with the calendar than without it, at both levels on both panels (DM above 3).
#  Z2. No model without the calendar is in the 90% model confidence set on FZ0, at either level on either panel.
#  Z3. The two-fix GARCH-t has a lower FZ0 than the masked calendar GARCH-t on the CRSP panel (DM above 2) and not on
#      US large caps.
#  Z4. On the CRSP panel at 2.5% the lowest FZ0 belongs to a model on the robust scale with the calendar, and it beats
#      the masked calendar GARCH-t (DM above 2).
# Usage: python tools/r88_joint_var_es.py canon <bench_all_bteg_rows.npz> | us_largecap
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from arch.bootstrap import MCS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r88"); os.makedirs(OUTD, exist_ok=True)
def fz0(r, v, e, a): v = np.minimum(v, -1e-8); e = np.minimum(e, v); return -(1.0 / (a * e)) * (r <= v) * (v - r) + v / e + np.log(-e) - 1.0
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); tes = lambda a, nu: -stats.t.pdf(stats.t.ppf(a, nu), nu) * (nu + stats.t.ppf(a, nu) ** 2) / ((nu - 1) * a) / np.sqrt(nu / (nu - 2))
A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); T = A[A.test & A.U & np.isfinite(A.Dm)].reset_index(drop=True); R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json")))
if WHAT == "canon":
    Rw = np.load(sys.argv[2]); rname = Rw["permno"].astype(int).astype(str); NM = {"GARCH-t": "garch_t", "GJR-skew-t": "gjr_skewt", "Beta-t-EGARCH": "param_bteg", "FHS": "fhs_name", "trees": "body", "trees on robust scale": "body_bteg", "trees with EVT tail": "engine", "trees with EVT tail, robust scale": "engine_bteg", "pooled GARCH-EVT": "evt_pool"}
    VE = lambda m, a: (Rw["var__%s__%s" % (m, a)].astype(float), Rw["es__%s__%s" % (m, a)].astype(float))
else:
    Rw = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); rname = np.array([str(x) for x in Rw["names"]])[Rw["name"]]; TAUS = [round(float(t), 4) for t in Rw["taus"]]
    NM = {"GARCH-t": "garch_t", "GJR-skew-t": "gjr_skewt", "Beta-t-EGARCH": "param_bteg", "FHS": "fhs", "trees": "body", "trees on robust scale": "body_bteg", "trees with EVT tail": "engine", "trees with EVT tail, robust scale": "engine_bteg"}
    VE = lambda m, a: (Rw["q__" + m].astype(float)[:, TAUS.index(a)], Rw["es__%s__%s" % (m, a)].astype(float))
pos = pd.Series(np.arange(len(rname)), index=pd.MultiIndex.from_arrays([rname, Rw["date"].astype(str)])).reindex(pd.MultiIndex.from_arrays([T["name"].values, T.date.dt.strftime("%Y-%m-%d").values])).values; assert np.isfinite(pos).all(); pos = pos.astype(int)
y = T.y.values; assert np.allclose(Rw["y"][pos], y, atol=1e-3); dates = pd.DatetimeIndex(T.date.values); R0 = T.r0.values == 1; R1 = (T.r1.values == 1) & ~R0; MU = R82["multipliers_c0_c1"]; mult = lambda c: np.where(R0, c[0], np.where(R1, c[1], 1.0))
lf, lm = (R82["real_share_pre_test"]["masked filter"][k] for k in ("lambda_f", "lambda_m")); fx = T.msig.values * np.sqrt(np.maximum(1 - (1 - (lf + (lm - lf) * T.wm.values)) * np.clip(T.Dm.values, -0.5, 0.98), 0.02))
def dm(d):
    s = pd.Series(d, index=dates).groupby(level=0).mean().values; s = s[np.isfinite(s)]; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
OUT = {"universe": WHAT, "rows": int(len(T)), "predictions_written_before_run": "see the script header (Z1-Z4)", "models_without_calendar": list(NM)}
for a in AL:
    L = {}
    for nm, m in NM.items():
        try: v, e = VE(m, a)
        except KeyError: continue
        v, e = v[pos], e[pos]; c = MU.get(nm + ", calendar multipliers", MU["GARCH-t, calendar multipliers"]); L[nm] = fz0(y, v, e, a); L[nm + ", calendar multipliers"] = fz0(y, v * mult(c), e * mult(c), a)
    for nm, sg_, c in (("masked calendar GARCH-t", T.msig.values, MU["masked calendar GARCH-t"]), ("two-fix GARCH-t", fx, MU["two-fix GARCH-t"])): L[nm] = fz0(y, T.mmu.values + sg_ * mult(c) * tq(a, T.mnu.values), T.mmu.values + sg_ * mult(c) * tes(a, T.mnu.values), a)
    ok = np.all([np.isfinite(v) for v in L.values()], axis=0); M = pd.DataFrame({m: pd.Series(v[ok], index=dates[ok]).groupby(level=0).mean() for m, v in L.items()}).dropna(); mc = MCS(M, size=0.10, reps=1000, block_size=10, method="max", bootstrap="stationary", seed=0); mc.compute()
    OUT[str(a)] = {"rows": int(ok.sum()), "mean_fz0": {m: round(float(v[ok].mean()), 4) for m, v in L.items()}, "DM_against_garch_t (positive: better)": {m: dm((L["GARCH-t"] - v)[ok]) for m, v in L.items() if m != "GARCH-t"},
                   "DM_calendar_against_same_model_without": {m: dm((L[m] - L[m + ", calendar multipliers"])[ok]) for m in NM if m in L}, "DM_two_fix_against_masked": dm((L["masked calendar GARCH-t"] - L["two-fix GARCH-t"])[ok]),
                   "DM_lowest_against_masked": None, "lowest": min(L, key=lambda m: L[m][ok].mean()), "mcs_90pct": sorted(mc.included)}
    OUT[str(a)]["DM_lowest_against_masked"] = dm((L["masked calendar GARCH-t"] - L[OUT[str(a)]["lowest"]])[ok])
    print("--- %s %s: lowest %s (DM against masked %s); two-fix against masked %s; calendar against same model %s" % (WHAT, a, OUT[str(a)]["lowest"], OUT[str(a)]["DM_lowest_against_masked"], OUT[str(a)]["DM_two_fix_against_masked"], OUT[str(a)]["DM_calendar_against_same_model_without"]))
    print("    mean FZ0 %s" % OUT[str(a)]["mean_fz0"]); print("    MCS %s" % OUT[str(a)]["mcs_90pct"])
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
