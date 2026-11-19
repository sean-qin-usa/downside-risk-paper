# r76_origin_accounting.py -- WHAT DECIDES lambda, HOW MUCH OF THE GAIN THE CLOSED FORM ACCOUNTS FOR, WHY GARCH-t IS
# ALSO BEATEN ON QUIET DAYS, AND WHAT THE LEARNED TAIL ADDS.
#
# tools/r71_shock_response.py measured lambda, the share of a large shock's GARCH variance impact that is real: about
# 0.1 on stocks, near 1 on ETFs. Four questions were left open, and this script takes them in turn.
#
#  H  ORIGIN. Is lambda low because the shock is firm-specific? For each row the largest shock of the last 63 days is
#     split by w = clip(b * m / r, 0, 1): r is the name's return on the shock date (less its mean), m the equal-weighted
#     return of the other names of the panel that day, b the name's pre-test slope on m. w is the share of the shock
#     return that the market move accounts for. Firm-specific: w < 0.1. Mixed: 0.1 to 0.5. Market-wide: w >= 0.5.
#     lambda is estimated in each class (top score decile, 2.5% pinball, stationary bootstrap over dates). On the two US
#     stock universes the classes are crossed with the earnings flag of r71.
#  K  ACCOUNTING. Under the one-parameter family (true variance = s2 * (1 - (1 - lambda) * D), Student-t) the expected
#     pinball loss of GARCH-t's quantile and of the correct quantile are closed forms of D, nu, sigma and lambda. With
#     lambda fixed at its pre-test value this gives, row by row and without using any test return, the gain the formula
#     predicts. It is compared with what each model gained: by decile of D, by name, and over all days.
#  Q  QUIET DAYS. GARCH-t's 2.5% VaR is breached too often when no large shock is in the window (r71, block E). One
#     candidate: those days include the days on which the next announcement arrives. Breach rates on announcement
#     dates and the day after, and on the other days. US stock universes only.
#  C  CALENDAR. A GARCH-t that knows the earnings calendar and nothing else: lambda = 0 after an earnings shock, and the
#     scale multiplied by c0 on an announcement date and c1 on the day after (c0, c1 chosen on pre-test rows by 2.5%
#     pinball over a grid). Compared with GARCH-t, Beta-t-EGARCH and the flexible models. US stock universes only.
#  S  SHAPE. Where the learned tail on the robust scale (body_bteg) beats Beta-t-EGARCH: by score decile, by shock age,
#     by level, on announcement days; and what it does to the quantile there (mean log ratio of the two quantiles).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: everything in docs/SHOCK_RESPONSE_RESULTS.md.
# Not known: any quantity split by w, any expected-loss prediction, any breach rate by announcement day, the calendar
# model, any split of body_bteg's edge over Beta-t-EGARCH.
#  H1. On the three stock panels (canon, us_largecap, intl_largecap_screened) lambda is at most 0.2 for firm-specific
#      shocks and at least 0.5 for market-wide shocks, and the market-wide estimate is the higher of the two.
#  H2. On the two US universes, among shocks that are not earnings shocks, firm-specific ones have lambda below 0.3:
#      the higher lambda of "other shocks" in r71 comes from the market-wide ones.
#  H3. Market-wide shocks are under 15% of top-decile rows on each stock panel and over half on the equity ETF panel.
#      Across the seven panels the market-wide share and the panel's lambda have a Spearman correlation above 0.5.
#  K1. On each stock panel the predicted and the realized edge of the pooled trees over GARCH-t, by decile of D, have a
#      correlation above 0.8, and in the top D decile the realized edge is between 0.5 and 1.2 times the predicted one.
#  K2. Across names the predicted all-day edge and the pooled trees' realized all-day edge have a Spearman correlation
#      above 0.3 on each stock panel.
#  Q1. On both US universes, in rows with D < 0.05, GARCH-t's 2.5% breach rate is above 5% on announcement dates and
#      the day after, and at most 2.8% on the other days.
#  C1. On both US universes the calendar GARCH-t beats GARCH-t over all days at 2.5% (DM above 4); its all-day edge is
#      at least Beta-t-EGARCH's and at least half of the pooled trees'.
#  C2. On announcement dates and the day after it beats the pooled trees and the neural network, which cannot see the
#      calendar.
#  S1. body_bteg's 2.5% edge over Beta-t-EGARCH in the top decile at shock ages 1 to 5 is at least twice its edge over
#      the other rows. The rest of block S is description, with no prediction.
# Usage:  python tools/r76_origin_accounting.py canon <bench_all_bteg_rows.npz> [r70_preds.npz]
#         python tools/r76_origin_accounting.py <panel>
#         (needs data/panels/<universe>_dump.npz, written by R71_DUMP=1 python tools/r71_shock_response.py ...)
import os, sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r76"); os.makedirs(OUTD, exist_ok=True)
AL = (0.01, 0.025); GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 2); NB = int(os.environ.get("R76_BOOT", "500")); CG = np.round(np.arange(1.0, 4.001, 0.1), 2)
if WHAT == "canon":
    R = np.load(sys.argv[2]); NAME = R["permno"]; rk = R["rk_mk"]; MODELS = ["garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "body", "engine", "body_bteg", "engine_bteg"]
else:
    R = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); NAME = np.array([str(x) for x in R["names"]])[R["name"]]; rk = R["rk"]; MODELS = ["garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "neural", "body", "engine", "body_bteg", "engine_bteg", "fhs"]
TAUS = [round(float(t), 4) for t in R["taus"]]; J = {a: TAUS.index(a) for a in AL + (0.05,)}; Y = R["y"].astype(float); dates = pd.to_datetime(R["date"]); n = len(Y); Q = {m: R["q__" + m].astype(float) for m in MODELS}
if WHAT == "canon" and len(sys.argv) > 3:
    N = np.load(sys.argv[3]); kk = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); Q["neural"] = N["q"][kk.loc[list(zip(NAME, R["date"]))].values].astype(float); MODELS.append("neural")
S = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_dump.npz")); SIG, MU, NU, DD, AGE, SGN, EARN, SRET, MS, BM, E0, E1 = (S[k].astype(float) for k in ("sig", "mu", "nu", "D", "age", "sign", "earn", "sret", "mkt_s", "bmk", "e0", "e1")); SDATE = S["sdate"]
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
assert np.nanquantile(np.abs(MU + SIG * tq(0.025, NU) - Q["garch_t"][:, J[0.025]]) / np.abs(Q["garch_t"][:, J[0.025]]), 0.999) < 0.02
cfq = lambda a, lam, Dv=None, sig=None, mu=None, nu=None: (MU if mu is None else mu) + (SIG if sig is None else sig) * np.sqrt(np.maximum(1 - (1 - lam) * (DD if Dv is None else Dv), 0.02)) * tq(a, NU if nu is None else nu)
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
    if len(s) < 30: return None
    dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
def edge(Lr, Lm, mask):
    if mask.sum() < 50: return None
    return {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
rng = np.random.default_rng(0)
def sb_index(T, B, p=0.1):
    idx = np.empty((B, T), int); idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T): new = rng.random(B) < p; idx[:, t] = np.where(new, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx
def lam_hat(mask, a=0.025):
    if mask.sum() < 300: return {"rows": int(mask.sum())}
    y, mu, sig, nu, Dv = Y[mask], MU[mask], SIG[mask], NU[mask], DD[mask]
    M = pd.DataFrame(np.column_stack([pin(y, cfq(a, l, Dv, sig, mu, nu), a) for l in GRID]), index=dates[mask]).groupby(level=0).sum().values; o = {"lambda": float(GRID[int(np.argmin(M.sum(0)))]), "rows": int(mask.sum()), "dates": int(len(M))}
    if len(M) >= 60: bs = GRID[np.argmin(M[sb_index(len(M), NB)].sum(1), axis=1)]; o["ci95"] = [round(float(np.quantile(bs, 0.025)), 2), round(float(np.quantile(bs, 0.975)), 2)]
    return o
def pred_breach(mask, a, lam): return round(100 * float(stats.t.cdf(stats.t.ppf(a, NU[mask]) / np.sqrt(np.maximum(1 - (1 - lam) * DD[mask], 0.02)), NU[mask]).mean()), 2)
hit = lambda m, a, mask: round(100 * float((Y <= Q[m][:, J[a]])[mask].mean()), 2) if mask.sum() >= 100 else None
def eloss(c, a, nu):
    """Expected pinball loss at level a of the quantile forecast c for a unit-variance Student-t(nu) variable."""
    k = np.sqrt(nu / (nu - 2)); return c * (stats.t.cdf(c * k, nu) - a) + stats.t.pdf(c * k, nu) * (nu + (c * k) ** 2) / ((nu - 1) * k)
LOSS = {a: {m: pin(Y, Q[m][:, J[a]], a) for m in MODELS} for a in AL}; TOP = rk == 10; ALLM = np.ones(n, bool)
# pre-test lambda, as in r71
PD_ = np.clip(S["pre_D"], -0.5, 0.98); pt = S["pre_mk"] >= np.quantile(S["pre_mk"], 0.9)
LAMP = float(GRID[int(np.argmin([pin(S["pre_y"][pt], cfq(0.025, l, PD_[pt], S["pre_sig"][pt], S["pre_mu"][pt], S["pre_nu"][pt]), 0.025).mean() for l in GRID]))]); lg("%s: %d rows; pre-test lambda %.2f" % (WHAT, n, LAMP))
for a in AL: LOSS[a]["cf0"] = pin(Y, cfq(a, 0.0), a); LOSS[a]["cf_pre"] = pin(Y, cfq(a, LAMP), a)
FLEX = [m for m in ("cf0", "cf_pre", "param_bteg", "body", "neural", "body_bteg") if m in LOSS[0.025]]
OUT = {"universe": WHAT, "rows": n, "lambda_pre_test": LAMP, "bootstrap_reps": NB, "predictions_written_before_run": "see the script header (H1-H3, K1-K2, Q1, C1-C2, S1)",
       "note": "edge_pct > 0 and DM > 0: the named model has the lower pinball loss than the reference (GARCH-t unless stated)."}

# ---------------------------------------------------------------- H. origin of the shock
w = np.clip(BM * MS / np.where(np.abs(SRET) > 1e-9, SRET, np.nan), 0, 1); CLS = np.where(~np.isfinite(w), "unclassified", np.where(w < 0.1, "firm-specific", np.where(w < 0.5, "mixed", "market-wide")))
def cls_cell(mask):
    o = {"rows": int(mask.sum())}
    if mask.sum() < 300: return o
    o.update({"shock_events": int(len(set(zip(NAME[mask].tolist(), SDATE[mask].tolist())))), "mean_D": round(float(DD[mask].mean()), 3), "mean_w": round(float(np.nanmean(w[mask])), 3), "lambda": lam_hat(mask),
              "garch_t_breach_0.025": hit("garch_t", 0.025, mask), "predicted_breach_lambda0": pred_breach(mask, 0.025, 0.0), "edges_over_garch_t_0.025": {m: edge(LOSS[0.025]["garch_t"], LOSS[0.025][m], mask) for m in FLEX}})
    return o
H = {"definition": "w = clip(b * m / r, 0, 1); firm-specific w < 0.1, mixed 0.1 to 0.5, market-wide w >= 0.5", "classified_share_of_top_decile": round(float((CLS[TOP] != "unclassified").mean()), 3),
     "share_of_top_decile_rows": {c: round(float((CLS[TOP] == c).sum() / max((CLS[TOP] != "unclassified").sum(), 1)), 3) for c in ("firm-specific", "mixed", "market-wide")},
     "share_of_all_rows": {c: round(float((CLS == c).sum() / max((CLS != "unclassified").sum(), 1)), 3) for c in ("firm-specific", "mixed", "market-wide")},
     "top_decile": {c: cls_cell(TOP & (CLS == c)) for c in ("firm-specific", "mixed", "market-wide")}, "rows_with_D_above_0.10": {c: cls_cell((DD > 0.10) & (CLS == c)) for c in ("firm-specific", "mixed", "market-wide")}}
mw = TOP & (CLS == "market-wide"); vc = pd.Series(SDATE[mw]).value_counts()
H["market_wide_shock_dates_most_rows"] = {str(d): int(c) for d, c in vc.head(12).items()}; H["market_wide_shock_dates_count"] = int(len(vc)); H["top5_dates_share_of_market_wide_rows"] = round(float(vc.head(5).sum() / max(mw.sum(), 1)), 3)
if np.isfinite(EARN).sum() > 2000:
    H["crossed_with_earnings_top_decile"] = {"earnings shock": cls_cell(TOP & (EARN == 1)), **{"not earnings, " + c: cls_cell(TOP & (EARN == 0) & (CLS == c)) for c in ("firm-specific", "mixed", "market-wide")}}
    H["crossed_with_earnings_D_above_0.10"] = {"earnings shock": cls_cell((DD > 0.10) & (EARN == 1)), **{"not earnings, " + c: cls_cell((DD > 0.10) & (EARN == 0) & (CLS == c)) for c in ("firm-specific", "mixed", "market-wide")}}
OUT["H_origin"] = H; lg("H done")

# ---------------------------------------------------------------- K. accounting
K = {}
for a in AL:
    st = np.sqrt(np.maximum(1 - (1 - LAMP) * DD, 0.02)); cg = tq(a, NU) / st; ct = tq(a, NU); Eg = SIG * st * eloss(cg, a, NU); Et = SIG * st * eloss(ct, a, NU); dpred = Eg - Et
    rd = pd.qcut(pd.Series(DD).rank(method="first"), 10, labels=False).values + 1; rows = []
    for d in range(1, 11):
        m_ = rd == d; r_ = {"D_decile": d, "mean_D": round(float(DD[m_].mean()), 3), "predicted_edge_pct": round(100 * float(dpred[m_].mean() / Eg[m_].mean()), 2), "predicted_garch_t_loss": round(float(Eg[m_].mean()), 4), "realized_garch_t_loss": round(float(LOSS[a]["garch_t"][m_].mean()), 4)}
        for m in FLEX: r_[m] = edge(LOSS[a]["garch_t"], LOSS[a][m], m_)
        rows.append(r_)
    pe = np.array([r_["predicted_edge_pct"] for r_ in rows]); o = {"by_D_decile": rows, "all_days_predicted_edge_pct": round(100 * float(dpred.mean() / Eg.mean()), 2), "all_days_realized": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], ALLM) for m in FLEX},
                                                                   "top_decile_predicted_edge_pct": round(100 * float(dpred[TOP].mean() / Eg[TOP].mean()), 2), "top_decile_realized": {m: edge(LOSS[a]["garch_t"], LOSS[a][m], TOP) for m in FLEX}}
    o["correlation_across_D_deciles"] = {m: round(float(np.corrcoef(pe, [r_[m]["edge_pct"] for r_ in rows])[0, 1]), 3) for m in FLEX}; o["top_D_decile_realized_over_predicted"] = {m: round(rows[-1][m]["edge_pct"] / rows[-1]["predicted_edge_pct"], 2) for m in FLEX}
    df = pd.DataFrame({"name": NAME, "dp": dpred, "eg": Eg, "lg": LOSS[a]["garch_t"], **{m: LOSS[a][m] for m in FLEX}}).groupby("name").mean(); pn = df.dp / df.eg
    o["across_names"] = {m: {"spearman": round(float(stats.spearmanr(pn, (df.lg - df[m]) / df.lg)[0]), 3), "pearson": round(float(np.corrcoef(pn, (df.lg - df[m]) / df.lg)[0, 1]), 3), "slope": round(float(np.polyfit(pn, (df.lg - df[m]) / df.lg, 1)[0]), 2)} for m in FLEX}
    o["across_names"]["names"] = int(len(df)); o["across_names"]["predicted_edge_pct_quartiles"] = [round(100 * float(x), 2) for x in np.quantile(pn, [0.25, 0.5, 0.75])]
    # the predicted loss difference as a regressor: realized difference = c0 + c1 * predicted, per-date means, date-clustered through the per-date series
    reg = {}
    for m in FLEX:
        g = pd.DataFrame({"d": LOSS[a]["garch_t"] - LOSS[a][m], "p": dpred}, index=dates).groupby(level=0).mean(); X = np.column_stack([np.ones(len(g)), g.p.values]); bh = np.linalg.lstsq(X, g.d.values, rcond=None)[0]; u = g.d.values - X @ bh; xu = X * u[:, None]; Sx = xu.T @ xu
        for k_ in range(1, 11): G_ = xu[k_:].T @ xu[:-k_]; Sx += (1 - k_ / 11) * (G_ + G_.T)
        XtXi = np.linalg.inv(X.T @ X); V = XtXi @ Sx @ XtXi; reg[m] = {"intercept": float(bh[0]), "slope": round(float(bh[1]), 3), "t_intercept": round(float(bh[0] / math.sqrt(V[0, 0])), 2), "t_slope": round(float(bh[1] / math.sqrt(V[1, 1])), 2),
                                                                         "share_of_mean_gain_from_slope_term": round(float(bh[1] * g.p.mean() / g.d.mean()), 2) if abs(g.d.mean()) > 1e-12 else None}
    o["date_level_regression_on_predicted_difference"] = reg; K[str(a)] = o
OUT["K_accounting"] = K; lg("K done")

# ---------------------------------------------------------------- Q, C. announcement days and the calendar model (US stock universes)
HAVE = np.isfinite(E0) & np.isfinite(E1)
if HAVE.sum() > 5000:
    AN = HAVE & ((E0 == 1) | (E1 == 1)); NO = HAVE & ~AN; Qd = {"rows_with_calendar": int(HAVE.sum()), "share_announcement_window_days": round(float(AN.sum() / HAVE.sum()), 4)}
    for lab, dm_ in (("D_below_0.02", DD < 0.02), ("D_below_0.05", DD < 0.05), ("all_D", ALLM)):
        Qd[lab] = {k_: {"rows": int((m_ & dm_).sum()), **{"breach_%s_%s" % (mm, a): hit(mm, a, m_ & dm_) for mm in ("garch_t", "param_bteg", "body") for a in AL}} for k_, m_ in (("all days", HAVE), ("announcement date", HAVE & (E0 == 1)), ("day after", HAVE & (E1 == 1) & (E0 != 1)), ("announcement window", AN), ("other days", NO))}
    zz = (Y - MU) / SIG; Qd["mean_squared_garch_residual"] = {"announcement date": round(float(np.mean(zz[HAVE & (E0 == 1)] ** 2)), 2), "day after": round(float(np.mean(zz[HAVE & (E1 == 1) & (E0 != 1)] ** 2)), 2), "other days": round(float(np.mean(zz[NO] ** 2)), 2)}
    OUT["Q_quiet_days"] = Qd
    # calendar model: c0, c1 on pre-test rows (2.5% pinball), scale already corrected for earnings shocks
    pe0, pe1, pearn = S["pre_e0"], S["pre_e1"], S["pre_earn"]; pk = np.isfinite(pe0) & np.isfinite(pe1); psig = S["pre_sig"] * np.sqrt(np.maximum(1 - np.where(pearn == 1, PD_, 0.0), 0.02)); ptq = tq(0.025, S["pre_nu"])
    def cfit(mask): return float(CG[int(np.argmin([pin(S["pre_y"][mask], S["pre_mu"][mask] + c * psig[mask] * ptq[mask], 0.025).sum() for c in CG]))]) if mask.sum() >= 200 else 1.0
    m0 = pk & (pe0 == 1); m1 = pk & (pe1 == 1) & (pe0 != 1); c0, c1 = cfit(m0), cfit(m1); mult = np.where(E0 == 1, c0, np.where(E1 == 1, c1, 1.0)); sc = SIG * np.sqrt(np.maximum(1 - np.where(EARN == 1, DD, 0.0), 0.02))
    Cc = {"c0_announcement_date": c0, "c1_day_after": c1, "pre_test_rows_announcement_date": int(m0.sum()), "pre_test_rows_day_after": int(m1.sum()), "pre_test_rows_with_calendar": int(pk.sum())}
    U = HAVE & np.isfinite(EARN)
    for a in AL:
        Lc = pin(Y, MU + sc * mult * tq(a, NU), a); Lw = pin(Y, MU + SIG * mult * tq(a, NU), a); Le = pin(Y, MU + sc * tq(a, NU), a); LOSS[a]["calendar"] = Lc; refs = [m for m in ("garch_t", "param_bteg", "body", "neural", "body_bteg") if m in LOSS[a]]
        Cc[str(a)] = {"calendar_vs": {r_: {"all_days": edge(LOSS[a][r_], Lc, U), "announcement_window": edge(LOSS[a][r_], Lc, U & AN), "other_days": edge(LOSS[a][r_], Lc, U & NO), "top_decile": edge(LOSS[a][r_], Lc, U & TOP)} for r_ in refs},
                      "parts_vs_garch_t_all_days": {"widen on announcement days only": edge(LOSS[a]["garch_t"], Lw, U), "discard earnings shocks only": edge(LOSS[a]["garch_t"], Le, U)},
                      "others_vs_garch_t_same_rows": {m: {"all_days": edge(LOSS[a]["garch_t"], LOSS[a][m], U), "announcement_window": edge(LOSS[a]["garch_t"], LOSS[a][m], U & AN), "other_days": edge(LOSS[a]["garch_t"], LOSS[a][m], U & NO)} for m in refs[1:]},
                      "calendar_breach_pct": {"all_days": round(100 * float((Y <= MU + sc * mult * tq(a, NU))[U].mean()), 2), "announcement_window": round(100 * float((Y <= MU + sc * mult * tq(a, NU))[U & AN].mean()), 2), "other_days": round(100 * float((Y <= MU + sc * mult * tq(a, NU))[U & NO].mean()), 2)}}
    OUT["C_calendar"] = Cc; lg("Q, C done: c0 %.1f, c1 %.1f" % (c0, c1))

# ---------------------------------------------------------------- S. what the learned tail on the robust scale adds
if "body_bteg" in Q:
    Sd = {}; cells = [("top decile, shock age 1-5", TOP & (AGE <= 5)), ("top decile, shock age 6-21", TOP & (AGE >= 6) & (AGE <= 21)), ("top decile, shock age 22-63", TOP & (AGE >= 22)), ("top decile", TOP), ("decile 9", rk == 9), ("deciles 1-8", rk <= 8),
                      ("all rows except top decile at ages 1-5", ~(TOP & (AGE <= 5))), ("all days", ALLM), ("positive shock, top decile", TOP & (SGN > 0)), ("negative shock, top decile", TOP & (SGN < 0))]
    if HAVE.sum() > 5000: cells += [("announcement window", AN), ("other days (calendar known)", NO)]
    for a in (0.01, 0.025, 0.05):
        Lb = pin(Y, Q["param_bteg"][:, J[a]], a); Lx = pin(Y, Q["body_bteg"][:, J[a]], a); lr = np.log(np.maximum(Q["body_bteg"][:, J[a]] / Q["param_bteg"][:, J[a]], 1e-3))
        Sd[str(a)] = {lab: {"edge_over_bteg": edge(Lb, Lx, m_), "mean_log_quantile_ratio": round(float(np.nanmean(lr[m_])), 4), "sd_log_quantile_ratio": round(float(np.nanstd(lr[m_])), 4), "bteg_breach_pct": round(100 * float((Y <= Q["param_bteg"][:, J[a]])[m_].mean()), 2),
                            "body_bteg_breach_pct": round(100 * float((Y <= Q["body_bteg"][:, J[a]])[m_].mean()), 2), "share_of_total_gain": round(float((Lb - Lx)[m_].sum() / (Lb - Lx).sum()), 3)} for lab, m_ in cells}
    # how much of what the learned tail does to the 2.5% quantile is a function of four shock variables
    lr = np.log(np.maximum(Q["body_bteg"][:, J[0.025]] / Q["param_bteg"][:, J[0.025]], 1e-3)); X = np.column_stack([np.ones(n), DD, (AGE <= 5) & TOP, TOP, np.log(SIG / pd.Series(SIG).groupby(NAME).transform("median").values)]); ok = np.isfinite(lr) & np.isfinite(X).all(1)
    bh = np.linalg.lstsq(X[ok], lr[ok], rcond=None)[0]; Sd["log_ratio_regression_0.025"] = {"coefficients": dict(zip(["const", "D", "top decile and age 1-5", "top decile", "log GARCH sigma over the name's median"], [round(float(x), 4) for x in bh])), "r2": round(float(1 - np.var(lr[ok] - X[ok] @ bh) / np.var(lr[ok])), 3)}
    OUT["S_shape"] = Sd; lg("S done")
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)

# ---------------------------------------------------------------- print
f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"]); fl = lambda b: "n/a" if (b is None or "lambda" not in b) else "%.2f %s" % (b["lambda"], b.get("ci95", ""))
print("\n######## %s: pre-test lambda %.2f" % (WHAT, LAMP)); print("H. origin. classified %.0f%% of top-decile rows; shares of top decile %s; of all rows %s" % (100 * H["classified_share_of_top_decile"], H["share_of_top_decile_rows"], H["share_of_all_rows"]))
for blk in ("top_decile", "rows_with_D_above_0.10", "crossed_with_earnings_top_decile", "crossed_with_earnings_D_above_0.10"):
    if blk not in H: continue
    print("  -- %s: rows, events | D | lambda | GARCH-t breach (predicted) | edges cf0, bteg, body" % blk)
    for c, v in H[blk].items():
        if "lambda" not in v: print("   %-28s %6d rows" % (c, v["rows"])); continue
        e = v["edges_over_garch_t_0.025"]; print("   %-28s %6d %5d | %.3f | %s | %s%% (%s%%) | %s %s %s" % (c, v["rows"], v["shock_events"], v["mean_D"], fl(v["lambda"]), v["garch_t_breach_0.025"], v["predicted_breach_lambda0"], f(e["cf0"]), f(e["param_bteg"]), f(e["body"])))
print("  market-wide shock dates (%d; top five hold %.0f%% of rows):" % (H["market_wide_shock_dates_count"], 100 * H["top5_dates_share_of_market_wide_rows"]), H["market_wide_shock_dates_most_rows"])
for a in ("0.025", "0.01"):
    k = K[a]; print("K. accounting at %s. all days: predicted %+.1f%%, realized %s | top decile: predicted %+.1f%%, realized %s" % (a, k["all_days_predicted_edge_pct"], {m: f(e) for m, e in k["all_days_realized"].items()}, k["top_decile_predicted_edge_pct"], {m: f(e) for m, e in k["top_decile_realized"].items()}))
    for r_ in k["by_D_decile"]: print("   D decile %2d (D %.3f): predicted %+5.1f%% | %s | GARCH-t loss predicted %.4f realized %.4f" % (r_["D_decile"], r_["mean_D"], r_["predicted_edge_pct"], "  ".join("%s %s" % (m, f(r_[m])) for m in FLEX), r_["predicted_garch_t_loss"], r_["realized_garch_t_loss"]))
    print("   correlation across D deciles:", k["correlation_across_D_deciles"], "| top D decile realized/predicted:", k["top_D_decile_realized_over_predicted"]); print("   across names:", k["across_names"]); print("   regression on predicted difference:", {m: (v["slope"], v["t_slope"], v["t_intercept"], v["share_of_mean_gain_from_slope_term"]) for m, v in k["date_level_regression_on_predicted_difference"].items()})
if "Q_quiet_days" in OUT:
    Qd = OUT["Q_quiet_days"]; print("Q. announcement days: %.1f%% of %d rows; mean squared GARCH residual %s" % (100 * Qd["share_announcement_window_days"], Qd["rows_with_calendar"], Qd["mean_squared_garch_residual"]))
    for lab in ("D_below_0.02", "D_below_0.05", "all_D"): print("   %s:" % lab, {k_: (v["rows"], v["breach_garch_t_0.025"], v["breach_param_bteg_0.025"], v["breach_body_0.025"], v["breach_garch_t_0.01"]) for k_, v in Qd[lab].items()})
    Cc = OUT["C_calendar"]; print("C. calendar model: c0 %.1f, c1 %.1f (pre-test rows %d, %d)" % (Cc["c0_announcement_date"], Cc["c1_day_after"], Cc["pre_test_rows_announcement_date"], Cc["pre_test_rows_day_after"]))
    for a in ("0.025", "0.01"):
        print("   %s calendar vs:" % a, {r_: {k_: f(e) for k_, e in v.items()} for r_, v in Cc[a]["calendar_vs"].items()}); print("      parts:", {k_: f(e) for k_, e in Cc[a]["parts_vs_garch_t_all_days"].items()}, "| others vs GARCH-t:", {m: {k_: f(e) for k_, e in v.items()} for m, v in Cc[a]["others_vs_garch_t_same_rows"].items()}, "| breach:", Cc[a]["calendar_breach_pct"])
if "S_shape" in OUT:
    for a in ("0.01", "0.025", "0.05"):
        print("S. learned tail on the robust scale against Beta-t-EGARCH at %s: cell | edge | mean log quantile ratio | breach bteg, body_bteg | share of total gain" % a)
        for lab, v in OUT["S_shape"][a].items(): print("   %-40s %-18s %+.4f | %.2f%% %.2f%% | %.3f" % (lab, f(v["edge_over_bteg"]), v["mean_log_quantile_ratio"], v["bteg_breach_pct"], v["body_bteg_breach_pct"], v["share_of_total_gain"]))
    print("   log-ratio regression:", OUT["S_shape"]["log_ratio_regression_0.025"])
lg("wrote %s" % os.path.join(OUTD, WHAT + ".json"))
