# r72_spa_mcs.py -- DOES ANYTHING BEAT GARCH-t, AND WHEN? Hansen's (2005) test for superior predictive ability and the
# Model Confidence Set (Hansen, Lunde and Nason 2011), run separately on the days after a large shock and on the other
# days, from the per-row forecasts already saved. Nothing is re-estimated.
#
# The SPA null: no model in the comparison set has a lower expected loss than the benchmark. Hansen and Lunde (2005) ask
# it of GARCH(1,1) over a whole sample. Here it is asked on four sets of days: the top decile of the residual-kurtosis
# score, decile 9, deciles 1 to 8, and all days; with two benchmarks, GARCH-t and Beta-t-EGARCH with a Student-t tail;
# and against three comparison sets:
#   all          every other model with a forecast on the same rows
#   flexible     the flexible models on the GARCH scale or with no scale: pooled trees, SAV-CAViaR, neural network
#   robust+shape the learned tail on the Beta-t-EGARCH scale: pooled trees and accuracy layer on that scale
# Losses: pinball at 1%, 2.5% and 5%; FZ0 at 1% and 2.5% for the models that give an ES. Each loss is averaged over the
# names in the set on each date; the tests run on the date series (stationary bootstrap, mean block 10 days, 2,000
# replications, studentized). MCS at the 10% level with the T_max statistic, and the T_R statistic for comparison.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: the pairwise DM statistics on these sets.
#  S1. Stock panels (canonical, us_largecap, intl_largecap), top decile, pinball at 1%, 2.5% and 5%: SPA rejects GARCH-t
#      against the flexible set (consistent p < 0.01).
#  S2. Deciles 1 to 8, pinball 2.5%, flexible set: SPA does not reject GARCH-t on the canonical panel or us_largecap
#      (consistent p > 0.05).
#  S3. Beta-t-EGARCH as benchmark, top decile, pinball 2.5%, stock panels: the flexible set does not reject it
#      (p > 0.05); the robust+shape set does (p < 0.05).
#  S4. MCS (10%), top decile, pinball 2.5%, stock panels: GARCH-t, GJR-skew-t and FHS are excluded; the pooled trees on
#      the Beta-t-EGARCH scale are included.
#  S5. Other panels, top decile, pinball 2.5%, flexible set: SPA rejects GARCH-t on crypto and fx, and does not on
#      etf_equity or etf_other.
#  S6. All days, all models: SPA rejects GARCH-t on every stock panel at every level and on FZ0.
# Usage:  python tools/r72_spa_mcs.py canon <bench_all_bteg_rows.npz> [r70_preds.npz]   |   python tools/r72_spa_mcs.py <panel>
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from arch.bootstrap import SPA, MCS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r72"); os.makedirs(OUTD, exist_ok=True); REPS = int(os.environ.get("R72_REPS", "2000")); BLOCK = 10; SEED = 0
pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def fz0(ret, v, e, a):
    v = np.minimum(v, -1e-8); e = np.minimum(e, v); return -(1.0 / (a * e)) * (ret <= v) * (v - ret) + v / e + np.log(-e) - 1.0
if WHAT == "canon":
    D = np.load(sys.argv[2]); rk = D["rk_mk"]; MS = [k[3:] for k in D.files if k.startswith("q__")]; Q = {m: D["q__" + m].astype(float) for m in MS}
    if len(sys.argv) > 3:
        N = np.load(sys.argv[3]); kk = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); Q["neural"] = N["q"][kk.loc[list(zip(D["permno"], D["date"]))].values].astype(float); MS.append("neural")
    FZ = {a: {k.split("__")[1]: D[k].astype(float) for k in D.files if k.startswith("fz0__") and k.endswith("__%s" % a)} for a in (0.01, 0.025)}
    for a in FZ: FZ[a].pop("engine_overlay", None)
else:
    D = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); rk = D["rk"]; MS = [k[3:] for k in D.files if k.startswith("q__")]; Q = {m: D["q__" + m].astype(float) for m in MS}
Y = D["y"].astype(float); dates = pd.to_datetime(D["date"]); TAUS = [round(float(t), 4) for t in D["taus"]]
if WHAT != "canon":
    FZ = {a: {k.split("__")[1]: fz0(Y, Q[k.split("__")[1]][:, TAUS.index(a)], D[k].astype(float), a) for k in D.files if k.startswith("es__") and k.endswith("__%s" % a)} for a in (0.01, 0.025)}
# at 5% the accuracy layer equals the pooled trees (the pooled tail applies only at 2.5% and below), so it is left out there
LOSS = {"pinball_%s" % a: {m: pin(Y, Q[m][:, TAUS.index(a)], a) for m in MS if not (a > 0.025 and m.startswith("engine"))} for a in (0.01, 0.025, 0.05)}; LOSS.update({"fz0_%s" % a: FZ[a] for a in FZ})
SETS = {"top_decile": rk == 10, "decile9": rk == 9, "deciles1to8": (rk >= 1) & (rk <= 8), "all_days": np.ones(len(Y), bool)}
FLEXS = ("body", "sav_caviar", "neural"); ROBS = ("body_bteg", "engine_bteg")
def spa(bench, alts, M):
    if not alts: return None
    s = SPA(M[bench].values, M[list(alts)].values, block_size=BLOCK, reps=REPS, bootstrap="stationary", studentize=True, seed=SEED); s.compute(); p = s.pvalues
    best = min(alts, key=lambda m: M[m].mean()); return {"p_lower": round(float(p["lower"]), 4), "p_consistent": round(float(p["consistent"]), 4), "p_upper": round(float(p["upper"]), 4), "n_models": len(alts),
                                                           "best_model": best, "best_edge_pct": round(100 * float((M[bench].mean() - M[best].mean()) / M[bench].mean()), 2)}
OUT = {"universe": WHAT, "note": "SPA null: no model in the set has lower expected loss than the benchmark; small p rejects the benchmark. Date series of cross-sectional mean losses; stationary bootstrap, mean block %d, %d replications." % (BLOCK, REPS),
       "predictions_written_before_run": "see the script header (S1-S6)", "models": MS, "results": {}}
for sname, mask in SETS.items():
    OUT["results"][sname] = {}
    for lname, L in LOSS.items():
        M = pd.DataFrame({m: pd.Series(v[mask], index=dates[mask]).groupby(level=0).mean() for m, v in L.items()}).dropna(); ms = list(M.columns); r = {"dates": int(len(M)), "rows": int(mask.sum()), "mean_loss": {m: round(float(M[m].mean()), 5) for m in ms}}
        for bench in ("garch_t", "param_bteg"):
            if bench not in ms: continue
            r["spa_benchmark_" + bench] = {"all": spa(bench, [m for m in ms if m != bench], M), "flexible": spa(bench, [m for m in FLEXS if m in ms], M), "robust_scale_learned_shape": spa(bench, [m for m in ROBS if m in ms and m != bench], M)}
        for meth in ("max", "R"):
            try:
                mc = MCS(M, size=0.10, reps=REPS, block_size=BLOCK, method=meth, bootstrap="stationary", seed=SEED); mc.compute()
                r["mcs_10pct_" + meth] = {"included": list(mc.included), "excluded": list(mc.excluded), "pvalues": {str(k): round(float(v), 4) for k, v in mc.pvalues["Pvalue"].items()}}
            except Exception as e: r["mcs_10pct_" + meth] = {"included": [], "excluded": [], "error": str(e)[:80]}
        OUT["results"][sname][lname] = r
    lg("%s done" % sname)
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
print("\n######## %s  (SPA consistent p; best model and its edge)" % WHAT)
for sname in SETS:
    for lname in LOSS:
        r = OUT["results"][sname][lname]; g = r.get("spa_benchmark_garch_t", {}); b = r.get("spa_benchmark_param_bteg", {})
        fm = lambda x: "n/a" if not x else "p=%.3f (%s %+.1f%%)" % (x["p_consistent"], x["best_model"], x["best_edge_pct"])
        print("%-12s %-14s GARCH-t: all %s | flexible %s || BtEG: all %s | flexible %s | robust+shape %s || MCS: %s" % (sname, lname, fm(g.get("all")), fm(g.get("flexible")), fm(b.get("all")), fm(b.get("flexible")), fm(b.get("robust_scale_learned_shape")), ",".join(r["mcs_10pct_max"]["included"]) or r["mcs_10pct_max"].get("error")))
lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
