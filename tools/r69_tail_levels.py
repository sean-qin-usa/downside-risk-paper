# r69_tail_levels.py -- READ-ONLY re-scoring of the canonical comparison at the two regulatory levels, from the per-row
# forecasts saved by `code/paper/job_bench_all_bteg.py --dump-rows` (200 names, 221,600 test rows, every model on the
# same rows, frozen fits). Nothing is re-estimated except the GARCH-t filter, which is refitted exactly as in the jobs to
# recover each row's 63-day residual window (sign, size and age of its largest shock; alternative window lengths).
#
# Blocks
#  A  the frontier at one level at a time: pinball at 1% and at 2.5%, and FZ0 at each, by mk63 decile, for every model
#     against GARCH-t (the paper's Table 2 uses the eleven-level average)
#  B  the scale-versus-shape decomposition at those levels, top decile and elsewhere
#  C  the sign of the shock at those levels
#  D  size: GARCH-t's realized breach rate and the realized quantile of its standardized residuals by decile, and by the
#     age of the largest shock inside the top decile; the VaR each model sets relative to GARCH-t's
#  E  what the learner predicts: its standardized 1% and 2.5% quantile by decile, shock sign and shock age, beside the
#     realized quantile and the Student-t quantile GARCH-t uses
#  F  the score's window: top-decile edges with the kurtosis computed over 21, 42, 63, 126 and 250 days
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19):
#  P1. At 1% and at 2.5% taken alone, the pooled body's top-decile pinball edge over GARCH-t is positive and significant
#      (DM > 3) and at least 1.5 times the eleven-level +2.98%; deciles 1-8 are within noise.
#  P2. The accuracy layer's top-decile pinball edge at those levels is under half the body's (the pooled GPD tail
#      overrides the body there), and its FZ0 gain over GARCH-t is no larger in the top decile than elsewhere.
#  P3. At both levels Beta-t-EGARCH with a Student-t tail beats GARCH-t in the top decile by more than the body does.
#  P4. On the Beta-t-EGARCH scale the body beats that scale's own Student-t tail in the top decile at both levels
#      (DM > 2): the shape component is visible at the tail levels.
#  P5. GARCH-t's realized breach rate in the top decile is below nominal at both levels (under 0.8% at 1%, under 2.0% at
#      2.5%), and lowest when the shock is 6 to 30 days old.
#  P6. The top-decile edge is positive and significant for every window from 42 to 250 days and largest at 63 or 126.
# Usage: GBC_PROJ=<panel dir> python tools/r69_tail_levels.py <bench_all_bteg_rows.npz>
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from arch import arch_model
P = os.environ["GBC_PROJ"]; D = np.load(sys.argv[1]); PN = D["permno"]; dates = pd.to_datetime(D["date"]); Y = D["y"].astype(float)
TAUS = [round(float(t), 4) for t in D["taus"]]; rk = D["rk_mk"]; rkb = D["rk_bmk"]; SIG = D["sig"].astype(float); MU = D["mu"].astype(float); NU = D["nu"].astype(float)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "paper", "diagnostics", "r69_tail_levels.json")
MODELS = [k[3:] for k in D.files if k.startswith("q__")]
Q = {m: D["q__" + m].astype(float) for m in MODELS}; J = {0.01: TAUS.index(0.01), 0.025: TAUS.index(0.025)}
pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
PIN = {a: {m: pin(Y, Q[m][:, j], a) for m in MODELS} for a, j in J.items()}
PIN11 = {m: np.mean([pin(Y, Q[m][:, j], t) for j, t in enumerate(TAUS)], axis=0) for m in MODELS}
FZ = {a: {k.split("__")[1]: D[k].astype(float) for k in D.files if k.startswith("fz0__") and k.endswith("__%s" % a)} for a in J}
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
    if len(s) < 30: return None
    dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
def edge(Lref, Lm, mask):          # > 0: m has the lower loss than the reference
    return {"edge_pct": round(100 * float((Lref - Lm)[mask].mean() / Lref[mask].mean()), 2), "DM": dm(Lref - Lm, mask)}
def gain(Lref, Lm, mask): return {"gain": round(float((Lref - Lm)[mask].mean()), 4), "DM": dm(Lref - Lm, mask)}
ALL = np.ones(len(Y), bool); TOP = rk == 10; D9 = rk == 9; LOW = (rk >= 1) & (rk <= 8); REST = (rk >= 1) & (rk <= 9)
out = {"note": "Canonical rows re-scored one level at a time. edge_pct > 0 and DM > 0: the model has the lower loss than the reference. "
               "FZ0 here comes from job_bench_all_bteg.py, whose engine ES uses the 20-node rule; the canonical FZ0 of record is bench_all_results.json.",
       "predictions_written_before_run": "see the script header (P1-P6)", "n_rows": int(len(Y))}

# ---------------------------------------------------------------- A. the frontier one level at a time
A_ = {}
for a in J:
    blk = {}
    for m in MODELS:
        if m == "garch_t": continue
        blk[m] = {"top": edge(PIN[a]["garch_t"], PIN[a][m], TOP), "decile9": edge(PIN[a]["garch_t"], PIN[a][m], D9), "deciles1to8": edge(PIN[a]["garch_t"], PIN[a][m], LOW),
                  "overall": edge(PIN[a]["garch_t"], PIN[a][m], ALL), "profile": [edge(PIN[a]["garch_t"], PIN[a][m], rk == d)["edge_pct"] for d in range(1, 11)]}
        if m in FZ[a] and "garch_t" in FZ[a]:
            blk[m]["fz0"] = {"top": gain(FZ[a]["garch_t"], FZ[a][m], TOP), "deciles1to9": gain(FZ[a]["garch_t"], FZ[a][m], REST), "overall": gain(FZ[a]["garch_t"], FZ[a][m], ALL)}
    A_[str(a)] = blk
A_["eleven_level"] = {m: {"top": edge(PIN11["garch_t"], PIN11[m], TOP), "deciles1to8": edge(PIN11["garch_t"], PIN11[m], LOW), "overall": edge(PIN11["garch_t"], PIN11[m], ALL)} for m in MODELS if m != "garch_t"}
out["A_frontier_by_level"] = A_

# ---------------------------------------------------------------- B. scale versus shape at the tail levels
B_ = {}
for a in J:
    L = PIN[a]; TB = rkb == 10
    B_[str(a)] = {"flexible_over_t_standard_scale_top": edge(L["garch_t"], L["body"], TOP), "robust_over_standard_scale_t_tail_top": edge(L["garch_t"], L["param_bteg"], TOP),
                  "flexible_over_t_robust_scale_top_stdscore": edge(L["param_bteg"], L["body_bteg"], TOP), "flexible_over_t_robust_scale_top_robscore": edge(L["param_bteg"], L["body_bteg"], TB),
                  "robust_scale_and_flexible_over_garch_t_top": edge(L["garch_t"], L["body_bteg"], TOP), "body_vs_param_bteg_top": edge(L["param_bteg"], L["body"], TOP),
                  "caviar_vs_param_bteg_top": edge(L["param_bteg"], L["sav_caviar"], TOP),
                  "all_days": {"flexible_std": edge(L["garch_t"], L["body"], ALL), "robust_t": edge(L["garch_t"], L["param_bteg"], ALL), "robust_flexible": edge(L["garch_t"], L["body_bteg"], ALL),
                               "flexible_over_t_robust_scale": edge(L["param_bteg"], L["body_bteg"], ALL)}}
    if "param_bteg" in FZ[a]:
        F = FZ[a]; B_[str(a)]["fz0"] = {"robust_t_vs_garch_t_top": gain(F["garch_t"], F["param_bteg"], TOP), "body_bteg_vs_param_bteg_top": gain(F["param_bteg"], F["body_bteg"], TOP),
                                        "body_bteg_vs_param_bteg_all": gain(F["param_bteg"], F["body_bteg"], ALL), "engine_bteg_vs_param_bteg_all": gain(F["param_bteg"], F["engine_bteg"], ALL),
                                        "body_vs_garch_t_top": gain(F["garch_t"], F["body"], TOP), "engine_vs_garch_t_top": gain(F["garch_t"], F["engine"], TOP)}
out["B_scale_vs_shape"] = B_

# ---------------------------------------------------------------- residual windows (canonical filter), for C-F
rr = pd.read_csv(os.path.join(P, "crsp_panel_returns.csv"), dtype={'permno': 'int32'}); rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
WINS = [21, 42, 63, 126, 250]; sign = np.zeros(len(Y)); age = np.zeros(len(Y)); size = np.zeros(len(Y)); Z = np.zeros(len(Y)); MKW = {w: np.full(len(Y), np.nan) for w in WINS}
for pn in pd.unique(PN):
    y = rr[rr.permno == pn].sort_values('date')['ret'].values.astype(float); n = len(y); sp = int(n * 0.6); m = PN == pn; nsel = int(m.sum())
    assert np.allclose(y[-nsel:], Y[m], atol=1e-4), pn
    pp = arch_model(y[:sp], vol='Garch', p=1, q=1, dist='t', rescale=False).fit(disp='off', show_warning=False).params
    om, al, be, mu = float(pp['omega']), float(pp['alpha[1]']), float(pp['beta[1]']), float(pp.get('mu', 0)); e0 = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:sp])
    for k in range(1, n): s2[k] = max(om + al * e0[k - 1] ** 2 + be * s2[k - 1], 1e-8)
    z = (y - mu) / np.sqrt(s2); Z[m] = z[-nsel:]; zs = pd.Series(z)
    win = sliding_window_view(z, 63); tgt = np.arange(63, n + 1); keep = (tgt >= n - nsel) & (tgt < n); win = win[keep]
    d = win - win.mean(1, keepdims=True); j = np.argmax(np.abs(d), 1); r = np.arange(len(win)); sign[m] = np.sign(win[r, j]); size[m] = np.abs(win[r, j]); age[m] = 63 - j
    for w in WINS: MKW[w][m] = zs.rolling(w, min_periods=max(10, w // 2)).kurt().shift(1).values[-nsel:]

# ---------------------------------------------------------------- C. sign of the shock at the tail levels
C_ = {}
for a in J:
    C_[str(a)] = {lab: {m: edge(PIN[a]["garch_t"], PIN[a][m], msk) for m in ("body", "engine", "sav_caviar", "gjr_skewt", "param_bteg", "body_bteg")}
                  for lab, msk in (("top_positive", TOP & (sign > 0)), ("top_negative", TOP & (sign < 0)))}
out["C_sign"] = C_

# ---------------------------------------------------------------- D. size
tq = lambda a: stats.t.ppf(a, NU) / np.sqrt(NU / (NU - 2))
D_ = {"by_decile": {}, "top_decile_by_shock_age": {}, "top_decile_by_sign": {}}
def cell(msk):
    r = {"rows": int(msk.sum())}
    for a, j in J.items():
        r[str(a)] = {"garch_t_breach_pct": round(100 * float((Y <= Q["garch_t"][:, j])[msk].mean()), 2), "body_breach_pct": round(100 * float((Y <= Q["body"][:, j])[msk].mean()), 2),
                     "engine_breach_pct": round(100 * float((Y <= Q["engine"][:, j])[msk].mean()), 2), "bteg_breach_pct": round(100 * float((Y <= Q["param_bteg"][:, j])[msk].mean()), 2),
                     "body_bteg_breach_pct": round(100 * float((Y <= Q["body_bteg"][:, j])[msk].mean()), 2),
                     "realized_z_quantile": round(float(np.quantile(Z[msk], a)), 3), "student_t_z_quantile": round(float(tq(a)[msk].mean()), 3),
                     "body_z_quantile": round(float(((Q["body"][:, j] - MU) / SIG)[msk].mean()), 3), "engine_z_quantile": round(float(((Q["engine"][:, j] - MU) / SIG)[msk].mean()), 3),
                     "var_ratio_body_to_garch_t": round(float(((Q["body"][:, j] - MU) / (Q["garch_t"][:, j] - MU))[msk].mean()), 3),
                     "var_ratio_bteg_to_garch_t": round(float((Q["param_bteg"][:, j] / Q["garch_t"][:, j])[msk].mean()), 3)}
    r["garch_sigma_over_bteg_sigma"] = round(float((SIG / D["bsig"].astype(float))[msk].mean()), 3)
    return r
for d in range(1, 11): D_["by_decile"][str(d)] = cell(rk == d)
for lab, lo, hi in (("1-5", 1, 5), ("6-15", 6, 15), ("16-30", 16, 30), ("31-45", 31, 45), ("46-63", 46, 63)): D_["top_decile_by_shock_age"][lab] = cell(TOP & (age >= lo) & (age <= hi))
D_["top_decile_by_sign"] = {"positive": cell(TOP & (sign > 0)), "negative": cell(TOP & (sign < 0))}
out["D_size_and_what_the_learner_predicts"] = D_

# ---------------------------------------------------------------- F. the score's window
F_ = {}
for w in WINS:
    mk = MKW[w]; ok = np.isfinite(mk); r_ = np.full(len(Y), -1); r_[ok] = pd.qcut(pd.Series(mk[ok]).rank(method="first"), 10, labels=False).values + 1
    F_[str(w)] = {"eleven_level_body_top": edge(PIN11["garch_t"], PIN11["body"], r_ == 10), "eleven_level_body_deciles1to8": edge(PIN11["garch_t"], PIN11["body"], (r_ >= 1) & (r_ <= 8)),
                  **{"body_top_%s" % a: edge(PIN[a]["garch_t"], PIN[a]["body"], r_ == 10) for a in J}, "overlap_with_63day_top_decile": round(float(((r_ == 10) & TOP).sum() / max((r_ == 10).sum(), 1)), 3)}
out["F_window"] = F_
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
f = lambda e: "%+.2f%% (%s)" % (e["edge_pct"], e["DM"])
for a in ("0.01", "0.025"):
    print("\n=== level %s: pinball edge over GARCH-t  [top decile | decile 9 | deciles 1-8 | overall]" % a)
    for m, v in out["A_frontier_by_level"][a].items(): print("  %-12s %s | %s | %s | %s%s" % (m, f(v["top"]), f(v["decile9"]), f(v["deciles1to8"]), f(v["overall"]), ("   FZ0 top %+.4f (%s) rest %+.4f (%s)" % (v["fz0"]["top"]["gain"], v["fz0"]["top"]["DM"], v["fz0"]["deciles1to9"]["gain"], v["fz0"]["deciles1to9"]["DM"])) if "fz0" in v else ""))
    b = out["B_scale_vs_shape"][a]; print("  DECOMPOSITION top decile:", {k: f(v) for k, v in b.items() if k not in ("all_days", "fz0")}); print("  all days:", {k: f(v) for k, v in b["all_days"].items()}); print("  FZ0:", b.get("fz0"))
    print("  SIGN:", {lab: {m: f(e) for m, e in d.items()} for lab, d in out["C_sign"][a].items()})
print("\n=== SIZE by decile (2.5%): GARCH-t breach | realized z | t z | body z | VaR ratio body/GARCH-t | sigma ratio GARCH/BtEG")
for d, c in D_["by_decile"].items(): x = c["0.025"]; print("  %2s  %.2f%%  %.3f  %.3f  %.3f  %.3f  %.3f | 1%%: breach %.2f%% realized %.3f t %.3f body %.3f" % (d, x["garch_t_breach_pct"], x["realized_z_quantile"], x["student_t_z_quantile"], x["body_z_quantile"], x["var_ratio_body_to_garch_t"], c["garch_sigma_over_bteg_sigma"], c["0.01"]["garch_t_breach_pct"], c["0.01"]["realized_z_quantile"], c["0.01"]["student_t_z_quantile"], c["0.01"]["body_z_quantile"]))
print("top decile by shock age:", {k: (c["rows"], c["0.025"]["garch_t_breach_pct"], c["0.025"]["realized_z_quantile"], c["0.025"]["body_z_quantile"], c["0.01"]["garch_t_breach_pct"]) for k, c in D_["top_decile_by_shock_age"].items()})
print("top decile by sign:", {k: (c["rows"], c["0.025"]["garch_t_breach_pct"], c["0.025"]["realized_z_quantile"], c["0.025"]["body_z_quantile"], c["0.025"]["student_t_z_quantile"]) for k, c in D_["top_decile_by_sign"].items()})
print("WINDOW:", {w: (f(v["eleven_level_body_top"]), f(v["body_top_0.01"]), f(v["body_top_0.025"]), v["overlap_with_63day_top_decile"]) for w, v in F_.items()})
print("wrote", OUT)
