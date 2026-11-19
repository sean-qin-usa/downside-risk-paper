# r78_calendar_garch.py -- A GARCH-t THAT KNOWS THE EARNINGS CALENDAR, FITTED PROPERLY: WHAT IS LEFT FOR FLEXIBLE MODELS?
#
# tools/r76_origin_accounting.py found that on stocks GARCH-t's 2.5% VaR is breached on 13 to 14% of announcement
# dates and days after, and that widening it on those days and discarding earnings shocks afterwards beats every
# flexible model over all days. That model kept the ordinary GARCH-t parameters, which were estimated on data that
# include the announcement days, and on the other days the flexible models still beat it by 1 to 2%.
#
# This script fits the model the finding calls for. Announcement dates and the day after are "masked": they add nothing
# to the likelihood, and in the variance recursion their squared return is replaced by its conditional expectation,
#       s2[t+1] = omega + alpha * (s2[t] if masked else e[t]^2) + beta * s2[t].
# So (omega, alpha, beta, mu, nu) describe the ordinary days only, and an announcement return never enters the variance.
# On masked days the scale is multiplied by c0 (announcement date) or c1 (day after), chosen on pre-test rows by 2.5%
# pinball over a grid. A control is fitted by the same code on the same window with nothing masked.
# Estimation uses the pre-test days from the first known announcement date; names with fewer than 750 such days are
# left out. US stock universes only (earnings dates from code/panels/download_earnings.py).
#
# Then a like-for-like comparison: every model is given the calendar. For each model the multipliers (c0, c1) on its
# own VaR are chosen on the first half of the test dates and all models are compared on the second half.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: r76 on the canonical panel and us_largecap (the
# simple calendar model gains 0.9% and 0.8% over GARCH-t on other days; body_bteg beats it there by 2.0% and 1.8%).
#  R1. Masking raises the fitted nu and lowers the fitted alpha: the median nu is at least 1 above the control's and the
#      median alpha is lower, on both universes.
#  R2. On other days the masked model beats GARCH-t at 2.5% by at least 1.5% (DM above 4) on both universes.
#  R3. On other days body_bteg's 2.5% edge over the masked model is at most 1.0% on both universes: at least half of
#      what the flexible model had over the simple calendar model is gone.
#  R4. Over all days the masked model with the announcement multipliers beats Beta-t-EGARCH, the pooled trees, the
#      neural network and body_bteg at 2.5% and at 1% (DM above 3) on both universes.
#  R5. Second half of the test period, every model given calendar multipliers: body_bteg's 2.5% edge over the masked
#      model over all days is below 1.5%, and the masked model beats GARCH-t with multipliers (DM above 3).
# Usage:  python tools/r78_calendar_garch.py canon <bench_all_bteg_rows.npz> [r70_preds.npz]
#         python tools/r78_calendar_garch.py us_largecap
#         (needs data/panels/<universe>_dump.npz and _series.pkl, written by R71_DUMP=1 python tools/r71_shock_response.py ...)
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats, optimize
from scipy.special import gammaln
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r78"); os.makedirs(OUTD, exist_ok=True); AL = (0.01, 0.025); CG = np.round(np.arange(1.0, 4.001, 0.1), 2); NW = int(os.environ.get("R78_WORKERS", "14"))
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))

def filt(th, y, mask, s20):
    mu, om, al, be = th[0], th[1], th[2], th[3]; n = len(y); s2 = np.empty(n); s = s20
    for t in range(n):
        s2[t] = s; e = y[t] - mu; s = om + al * (s if mask[t] else e * e) + be * s
    return s2
def nll(th, y, mask, s20, keep):
    mu, om, al, be, nu = th
    if om <= 1e-10 or al < 0 or be < 0 or al + be >= 0.9999 or nu <= 2.05 or nu > 200: return 1e12
    s2 = filt(th, y, mask, s20)[keep]; e = (y - mu)[keep]
    return -float(np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(math.pi * (nu - 2) * s2) - (nu + 1) / 2 * np.log1p(e * e / ((nu - 2) * s2))))
def fit_one(a):
    key, y, f0, f1, ts, st = a; k0 = int(np.argmax(np.isfinite(f0)))
    if ts - k0 < 750: return key, None
    yy = y[k0:]; m_ = ((f0[k0:] == 1) | (f1[k0:] == 1)); ne = ts - k0; out = {}
    for lab, mask in (("masked", m_), ("control", np.zeros(len(yy), bool))):
        ye = yy[:ne]; me = mask[:ne]; keep = ~me; v = float(np.var(ye[keep])); best = None
        for x0 in (st, [float(np.mean(ye[keep])), v * 0.05, 0.08, 0.87, 6.0]):
            x0 = list(x0); x0[1] = max(v * (1 - x0[2] - x0[3]), v * 0.01)
            r = optimize.minimize(nll, x0, args=(ye, me, v, keep), method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-5, "fatol": 1e-6})
            if best is None or r.fun < best.fun: best = r
        th = best.x; out[lab] = dict(th=th, s2=filt(th, yy, mask, v), nll=float(best.fun))
    return key, dict(k0=k0, **out)

if __name__ == "__main__":
    if WHAT == "canon":
        R = np.load(sys.argv[2]); NAME = R["permno"]; rk = R["rk_mk"]; MODELS = ["garch_t", "param_bteg", "body", "body_bteg"]
    else:
        R = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); NAME = np.array([str(x) for x in R["names"]])[R["name"]]; rk = R["rk"]; MODELS = ["garch_t", "param_bteg", "neural", "body", "body_bteg"]
    TAUS = [round(float(t), 4) for t in R["taus"]]; J = {a: TAUS.index(a) for a in AL}; Y = R["y"].astype(float); dates = pd.to_datetime(R["date"]); n = len(Y); Q = {m: R["q__" + m].astype(float)[:, [J[a] for a in AL]] for m in MODELS}
    if WHAT == "canon" and len(sys.argv) > 3:
        N = np.load(sys.argv[3]); kk = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); Q["neural"] = N["q"][kk.loc[list(zip(NAME, R["date"]))].values].astype(float)[:, [J[a] for a in AL]]; MODELS.append("neural")
    S = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_dump.npz")); MU, SIG, NU, DD, EARN, E0, E1, ALR, BER = (S[k].astype(float) for k in ("mu", "sig", "nu", "D", "earn", "e0", "e1", "alpha", "beta"))
    SER = pickle.load(open(os.path.join(ROOT, "data", "panels", WHAT + "_series.pkl"), "rb")); jobs = []
    for k, s in SER.items(): i = s["rows"][0]; jobs.append((k, s["y"], s["f0"], s["f1"], s["ts"], [MU[i], 0.0, ALR[i], BER[i], NU[i]]))
    with Pool(NW) as pool: res = dict(pool.map(fit_one, jobs, chunksize=1))
    ok = [k for k, v in res.items() if v is not None]; lg("%s: %d names with a calendar, %d fitted" % (WHAT, len(SER), len(ok)))
    SM, SC, MUM, NUM, MUC, NUC = (np.full(n, np.nan) for _ in range(6)); pre = []
    for k in ok:
        s = SER[k]; v = res[k]; p = s["pos"] - v["k0"]; m = s["rows"]; SM[m] = np.sqrt(v["masked"]["s2"][p]); SC[m] = np.sqrt(v["control"]["s2"][p]); MUM[m] = v["masked"]["th"][0]; NUM[m] = v["masked"]["th"][4]; MUC[m] = v["control"]["th"][0]; NUC[m] = v["control"]["th"][4]
        i = np.arange(126, s["ts"] - v["k0"]); pre.append(pd.DataFrame({"y": s["y"][v["k0"]:][i], "sig": np.sqrt(v["masked"]["s2"][i]), "mu": v["masked"]["th"][0], "nu": v["masked"]["th"][4], "e0": s["f0"][v["k0"]:][i], "e1": s["f1"][v["k0"]:][i]}))
    pre = pd.concat(pre, ignore_index=True)
    def cfit(df): return float(CG[int(np.argmin([pin(df.y.values, df.mu.values + c * df.sig.values * tq(0.025, df.nu.values), 0.025).sum() for c in CG]))])
    c0 = cfit(pre[pre.e0 == 1]); c1 = cfit(pre[(pre.e1 == 1) & (pre.e0 != 1)]); mult = np.where(E0 == 1, c0, np.where(E1 == 1, c1, 1.0)); U = np.isfinite(SM) & np.isfinite(E0); AN = U & ((E0 == 1) | (E1 == 1)); NO = U & ~AN; TOP = rk == 10
    c76 = json.load(open(os.path.join(ROOT, "results", "paper", "diagnostics", "r76", WHAT + ".json")))["C_calendar"]; m76 = np.where(E0 == 1, c76["c0_announcement_date"], np.where(E1 == 1, c76["c1_day_after"], 1.0))
    QQ = {m: Q[m] for m in MODELS}
    QQ["masked"] = np.column_stack([MUM + SM * mult * tq(a, NUM) for a in AL]); QQ["masked, no multiplier"] = np.column_stack([MUM + SM * tq(a, NUM) for a in AL]); QQ["control"] = np.column_stack([MUC + SC * tq(a, NUC) for a in AL])
    QQ["simple calendar (r76)"] = np.column_stack([MU + SIG * np.sqrt(np.maximum(1 - np.where(EARN == 1, DD, 0.0), 0.02)) * m76 * tq(a, NU) for a in AL])
    L = {m: {a: pin(Y, q[:, j], a) for j, a in enumerate(AL)} for m, q in QQ.items()}
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
        if len(s) < 30: return None
        dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    edge = lambda Lr, Lm, mask: None if mask.sum() < 50 else {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
    CELLS = (("all days", U), ("announcement window", AN), ("other days", NO), ("top decile", U & TOP), ("top decile, other days", NO & TOP), ("deciles 1-8, other days", NO & (rk <= 8)))
    th = lambda lab, j: float(np.median([res[k][lab]["th"][j] for k in ok]))
    OUT = {"universe": WHAT, "names_fitted": len(ok), "rows": int(U.sum()), "predictions_written_before_run": "see the script header (R1-R5)", "c0_announcement_date": c0, "c1_day_after": c1,
           "median_parameters": {lab: {"alpha": round(th(lab, 2), 4), "beta": round(th(lab, 3), 4), "nu": round(th(lab, 4), 2), "omega": th(lab, 1)} for lab in ("masked", "control")},
           "median_parameters_stored_garch_t": {"alpha": round(float(np.median([ALR[SER[k]["rows"][0]] for k in ok])), 4), "beta": round(float(np.median([BER[SER[k]["rows"][0]] for k in ok])), 4), "nu": round(float(np.median([NU[SER[k]["rows"][0]] for k in ok])), 2)},
           "masked_vs": {}, "vs_garch_t": {}, "breach_pct": {}}
    for j, a in enumerate(AL):
        OUT["masked_vs"][str(a)] = {r_: {lab: edge(L[r_][a], L["masked"][a], m_) for lab, m_ in CELLS} for r_ in QQ if r_ != "masked"}
        OUT["vs_garch_t"][str(a)] = {m: {lab: edge(L["garch_t"][a], L[m][a], m_) for lab, m_ in CELLS} for m in QQ if m != "garch_t"}
        OUT["breach_pct"][str(a)] = {m: {lab: round(100 * float((Y <= QQ[m][:, j])[m_].mean()), 2) for lab, m_ in CELLS} for m in QQ}
    # like for like: every model gets calendar multipliers chosen on the first half of the test dates
    ud = np.sort(pd.unique(dates[U])); cut = ud[len(ud) // 2]; H1 = U & (dates < cut); H2 = U & (dates >= cut); LL = {}; CM = {}
    for m in [x for x in QQ if x not in ("masked", "simple calendar (r76)")]:
        cen = MUM if m.startswith("masked") else MU; q = QQ[m][:, 1]; cs = []
        for fl_ in (H1 & (E0 == 1), H1 & (E1 == 1) & (E0 != 1)): cs.append(float(CG[int(np.argmin([pin(Y[fl_], cen[fl_] + c * (q[fl_] - cen[fl_]), 0.025).sum() for c in CG]))]))
        mm = np.where(E0 == 1, cs[0], np.where(E1 == 1, cs[1], 1.0)); CM[m] = cs; LL[m] = {a: pin(Y, cen + mm * (QQ[m][:, j] - cen), a) for j, a in enumerate(AL)}
    C2 = (("all days", H2), ("announcement window", H2 & AN), ("other days", H2 & NO))
    OUT["second_half_all_models_with_calendar"] = {"cut_date": str(pd.Timestamp(cut).date()), "multipliers_c0_c1": CM,
        **{str(a): {"vs_garch_t_with_calendar": {m: {lab: edge(LL["garch_t"][a], LL[m][a], m_) for lab, m_ in C2} for m in LL if m != "garch_t"},
                    "masked_vs": {m: {lab: edge(LL[m][a], LL["masked, no multiplier"][a], m_) for lab, m_ in C2} for m in LL if m != "masked, no multiplier"},
                    "calendar_gain_of_each_model": {m: edge(L[m][a], LL[m][a], H2) for m in LL}} for a in AL}}
    json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
    f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"])
    print("\n######## %s: %d names, %d rows; c0 %.1f, c1 %.1f; median parameters %s | stored GARCH-t %s" % (WHAT, len(ok), U.sum(), c0, c1, OUT["median_parameters"], OUT["median_parameters_stored_garch_t"]))
    for a in ("0.025", "0.01"):
        print("--- %s: the masked calendar model against each model (positive: masked model better)" % a)
        for r_, v in OUT["masked_vs"][a].items(): print("   %-24s %s" % (r_, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s: each model against GARCH-t" % a)
        for m, v in OUT["vs_garch_t"][a].items(): print("   %-24s %s" % (m, "  ".join("%s %s" % (lab, f(e)) for lab, e in v.items())))
        print("--- %s: breach rates" % a)
        for m, v in OUT["breach_pct"][a].items(): print("   %-24s %s" % (m, v))
    Z = OUT["second_half_all_models_with_calendar"]; print("--- second half (from %s), every model with calendar multipliers %s" % (Z["cut_date"], Z["multipliers_c0_c1"]))
    for a in ("0.025", "0.01"):
        print("   %s vs GARCH-t with calendar:" % a, {m: {lab: f(e) for lab, e in v.items()} for m, v in Z[a]["vs_garch_t_with_calendar"].items()}); print("   %s masked model against:" % a, {m: {lab: f(e) for lab, e in v.items()} for m, v in Z[a]["masked_vs"].items()})
        print("   %s what the calendar adds to each model:" % a, {m: f(e) for m, e in Z[a]["calendar_gain_of_each_model"].items()})
    lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
