# r84_last_percent.py -- WHAT THE FLEXIBLE MODELS STILL ADD ONCE EVERY MODEL HAS THE EARNINGS CALENDAR.
#
# r82: with the calendar in every model, the trees and the neural network stay 0.8% to 1.2% ahead of the best
# parametric model, on ordinary days, and neither the shock correction nor the tail shape explains it. The trees see
# five inputs besides the calendar. This script asks which of them carries the margin, and whether a one-parameter
# version of that input closes the gap.
#
# Ablation. Pooled trees on the masked calendar GARCH-t scale (same settings and training rows as r79/r81), at the 1%
# and 2.5% levels, with these inputs:
#   S0  the calendar only (r0, r1, ds): pooled quantiles by day type, no other information
#   S1  S0 + the volatility level (logsig)
#   S2  S0 + the recent scale of the residuals (zstd21, absz5)
#   S3  S0 + yesterday's residual and the share of down days (zl1, fracdn5)
#   S5  S0 + logsig + zstd21 + absz5
#   S4  S0 + all five (the model of r79/r81)
# Parametric version of S2 ("scale tracking"): the model's sigma is multiplied by (v_t / v_med)^gamma, with v_t the
# standard deviation of the last 21 standardized residuals that are not on calendar days (lagged one day), v_med its
# pre-test median, and gamma one number per panel chosen on pre-test ordinary days by 2.5% pinball loss. It is added
# to the masked calendar GARCH-t and to the two-fix GARCH-t of r82 (lambdas and multipliers taken from r82).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-20). Known beforehand: r79-r83. The trees on the masked scale with
# calendar inputs beat the masked calendar GARCH-t by 1.2% (CRSP) and 1.1% (US large caps) at 2.5% over all days;
# empirical quantiles in place of the Student-t changed the two-fix model by -0.2% and +0.4%.
#  L1. S0 is within 0.4% of the masked calendar GARCH-t over all days at 2.5%, either sign: pooling the tail adds little.
#  L2. S4 reproduces the stored trees on the masked scale with calendar inputs (all-days loss within 0.3%).
#  L3. Of S4's margin over the masked model on ordinary days at 2.5%, the recent scale of the residuals (S2) carries at
#      least half on both panels; the volatility level (S1) carries less than S2; S3 carries less than a quarter.
#  L4. Scale tracking added to the masked calendar GARCH-t: gamma between 0.1 and 0.8, and a gain of 0.3% to 1.5% over
#      all days at 2.5% (DM above 2) on both panels.
#  L5. The two-fix GARCH-t with scale tracking is within 0.6% of the best flexible model with the calendar on both
#      panels, and on the CRSP panel its DM against that model is above -2.
#
# FOLLOW-UP, WRITTEN AFTER THE CRSP RUN OF L1-L5 AND BEFORE ANY RUN OF WHAT FOLLOWS (2026-11-20). That run put the
# margin on the volatility level (S1 carried 93% of S4's ordinary-day margin, S2 53%, scale tracking chose gamma = 0).
# The level mixes two things, so it is split: S1a = S0 + the name's usual volatility (the pre-test mean of logsig),
# S1b = S0 + today's volatility against that usual level (logsig minus that mean). Parametric version of S1b ("level
# shrinkage"): sigma is multiplied by (sigma_t / usual sigma)^delta, delta one number per panel chosen on pre-test
# ordinary days; a negative delta pulls the forecast toward the name's usual level.
#  M1. On both panels S1b carries more of S4's ordinary-day margin at 2.5% than S1a, and at least half of it.
#  M2. delta is between -0.40 and -0.05; level shrinkage added to the masked calendar GARCH-t gains 0.4% to 1.2% over
#      all days at 2.5% (DM above 2) on both panels.
#  M3. The two-fix GARCH-t with level shrinkage is within 0.6% of the best flexible model with the calendar on both panels.
# Usage: python tools/r84_last_percent.py canon | us_largecap     (needs data/panels/<panel>_calcache.pkl and results/paper/diagnostics/r82/<panel>.json)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; AL = (0.01, 0.025); t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r84"); os.makedirs(OUTD, exist_ok=True); tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
HGB = dict(max_iter=250, max_depth=3, learning_rate=0.06); CAL = ["r0", "r1", "ds"]
SETS = {"S0 calendar only": [], "S1 + volatility level": ["logsig"], "S1a + the name's usual volatility": ["lsbar"], "S1b + volatility against the usual": ["lsdev"], "S2 + recent residual scale": ["zstd21", "absz5"], "S3 + last residual, down days": ["zl1", "fracdn5"], "S5 + level and recent scale": ["logsig", "zstd21", "absz5"], "S4 + all five": ["logsig", "zl1", "absz5", "zstd21", "fracdn5"]}
G = {}
def fit(job):
    s, a = job; X, z, Xte = G["Xtr"][CAL + SETS[s]].values, G["ztr"], G["Xte"][CAL + SETS[s]].values
    return s, a, HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=0, **HGB).fit(X, z).predict(Xte)

if __name__ == "__main__":
    A = pd.read_pickle(os.path.join(ROOT, "data", "panels", WHAT + "_calcache.pkl")); A = A[np.isfinite(A.r0)].reset_index(drop=True); R82 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r82", WHAT + ".json")))
    A["z"] = (A.y - A.mmu) / A.msig; calday = (A.r0 == 1) | (A.r1 == 1); zx = A.z.where(~calday); A["v"] = zx.groupby(A["name"]).transform(lambda s: s.rolling(21, min_periods=8).std().shift(1))
    A["lsbar"] = A["name"].map(A[~A.test].groupby("name")["logsig"].mean()); A["lsdev"] = A.logsig - A.lsbar
    feat = CAL + SETS["S4 + all five"] + ["lsbar", "lsdev"]; okf = np.isfinite(A[feat]).all(1).values; tr = A.train.values & okf; te = A.test.values & A.U.values & okf & np.isfinite(A.v.values); pre = ~A.test.values & np.isfinite(A.v.values) & np.isfinite(A.Dm.values)
    G.update(Xtr=A.loc[tr, feat], ztr=A.z.values[tr], Xte=A.loc[te, feat]); T = A[te].reset_index(drop=True); Y = T.y.values; dates = pd.DatetimeIndex(T.date.values); lg("%s: %d training rows, %d test rows" % (WHAT, tr.sum(), te.sum()))
    with Pool(12) as pool: fitted = pool.map(fit, [(s, a) for s in SETS for a in AL], chunksize=1)
    Q = {}
    for s in SETS:
        zq = np.column_stack([[v for s_, a_, v in fitted if s_ == s and a_ == a][0] for a in AL]); zq[:, 0] = np.minimum(zq[:, 0], zq[:, 1]); Q[s] = T.mmu.values[:, None] + T.msig.values[:, None] * zq
    lg("trees fitted")
    R0 = T.r0.values == 1; R1 = (T.r1.values == 1) & ~R0; mult = lambda c: np.where(R0, c[0], np.where(R1, c[1], 1.0)); cm = R82["multipliers_c0_c1"]["masked calendar GARCH-t"]; c2 = R82["multipliers_c0_c1"]["two-fix GARCH-t"]; lf, lm = (R82["real_share_pre_test"]["masked filter"][k] for k in ("lambda_f", "lambda_m"))
    fixs = lambda df: df.msig.values * np.sqrt(np.maximum(1 - (1 - (lf + (lm - lf) * df.wm.values)) * np.clip(df.Dm.values, -0.5, 0.98), 0.02))
    PR = A[pre & ~calday.values]; vmed = float(np.median(PR.v.values)); GR = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    def gfit(sig):
        b = tq(0.025, PR.mnu.values); L = [pin(PR.y.values, PR.mmu.values + sig * (PR.v.values / vmed) ** g * b, 0.025).sum() for g in GR]; return float(GR[int(np.argmin(L))])
    g1 = gfit(PR.msig.values); g2 = gfit(fixs(PR)); track = lambda g: (T.v.values / vmed) ** g
    DG = np.round(np.arange(-0.6, 0.3001, 0.05), 2)
    def dfit(sig):
        b = tq(0.025, PR.mnu.values); L = [pin(PR.y.values, PR.mmu.values + sig * np.exp(d * PR.lsdev.values) * b, 0.025).sum() for d in DG]; return float(DG[int(np.argmin(L))])
    d1 = dfit(PR.msig.values); d2 = dfit(fixs(PR)); shr = lambda d: np.exp(d * T.lsdev.values)
    tail = lambda sig, c: np.column_stack([T.mmu.values + sig * mult(c) * tq(a, T.mnu.values) for a in AL])
    MK, TF = "masked calendar GARCH-t", "two-fix GARCH-t"; Q[MK] = tail(T.msig.values, cm); Q[TF] = tail(fixs(T), c2); Q[MK + ", scale tracking"] = tail(T.msig.values * track(g1), cm); Q[TF + ", scale tracking"] = tail(fixs(T) * track(g2), c2); Q[MK + ", level shrinkage"] = tail(T.msig.values * shr(d1), cm); Q[TF + ", level shrinkage"] = tail(fixs(T) * shr(d2), c2)
    for nm in ("trees on masked scale, calendar inputs", "trees on robust scale, calendar inputs", "neural network, calendar inputs"): Q[nm] = np.column_stack([T["%s__%s" % (nm, a)].values for a in ("0.01", "0.025")])
    Q["GARCH-t"] = np.column_stack([T["GARCH-t__%s" % a].values for a in ("0.01", "0.025")]); rk = T.rk.values; RW = R0 | R1; ALLR = np.ones(len(T), bool)
    L = {m: {a: pin(Y, q[:, j], a) for j, a in enumerate(AL)} for m, q in Q.items()}
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    edge = lambda Lr, Lm, mask: {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask)}
    CELLS = (("all days", ALLR), ("other days", ~RW), ("top decile, other days", (rk == 10) & ~RW), ("deciles 1-8, other days", (rk <= 8) & (rk > 0) & ~RW))
    OUT = {"universe": WHAT, "rows": int(len(T)), "training_rows": int(tr.sum()), "predictions_written_before_run": "see the script header (L1-L5; M1-M3 written after the CRSP run of L1-L5)", "gamma": {MK: g1, TF: g2}, "delta": {MK: d1, TF: d2}, "median_recent_scale_pre_test": round(vmed, 3)}
    for j, a in enumerate(AL):
        g = lambda m, c="all days": 100 * float((L[MK][a] - L[m][a])[dict(CELLS)[c]].mean() / L[MK][a][dict(CELLS)[c]].mean()); best = min(("trees on robust scale, calendar inputs", "neural network, calendar inputs", "trees on masked scale, calendar inputs"), key=lambda m: L[m][a].mean())
        OUT[str(a)] = {"vs_masked_calendar_garch_t": {m: {lab: edge(L[MK][a], L[m][a], c) for lab, c in CELLS} for m in Q if m != MK}, "best_flexible_with_calendar": best,
                       "share_of_S4_margin_other_days": {s: round(g(s, "other days") / g("S4 + all five", "other days"), 3) for s in SETS},
                       "against_best_flexible": {m: {lab: edge(L[best][a], L[m][a], c) for lab, c in CELLS} for m in (MK, TF, MK + ", scale tracking", TF + ", scale tracking", MK + ", level shrinkage", TF + ", level shrinkage", "S4 + all five")},
                       "share_of_gap_to_best_closed": {m: round(g(m) / g(best), 3) for m in (TF, MK + ", scale tracking", TF + ", scale tracking", MK + ", level shrinkage", TF + ", level shrinkage")}, "S4_vs_stored_trees_on_masked_scale": edge(L["trees on masked scale, calendar inputs"][a], L["S4 + all five"][a], ALLR),
                       "breach_pct": {m: {lab: round(100 * float((Y <= Q[m][:, j])[c].mean()), 2) for lab, c in CELLS[:2]} for m in Q}}
    json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1); f = lambda e: "%+.1f%% (%s)" % (e["edge_pct"], e["DM"])
    print("######## %s: gamma %s delta %s" % (WHAT, OUT["gamma"], OUT["delta"]))
    for a in ("0.025", "0.01"):
        print("--- %s against the masked calendar GARCH-t" % a)
        for m, v in OUT[a]["vs_masked_calendar_garch_t"].items(): print("   %-42s %s" % (m, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s share of S4's margin on other days: %s" % (a, OUT[a]["share_of_S4_margin_other_days"]))
        print("--- %s against the best flexible model (%s): %s" % (a, OUT[a]["best_flexible_with_calendar"], {m: (f(v["all days"]), f(v["other days"])) for m, v in OUT[a]["against_best_flexible"].items()}))
        print("--- %s share of the gap to the best flexible model closed: %s; S4 against the stored trees: %s; breach all days %s" % (a, OUT[a]["share_of_gap_to_best_closed"], f(OUT[a]["S4_vs_stored_trees_on_masked_scale"]), {m: v["all days"] for m, v in OUT[a]["breach_pct"].items()}))
    lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
