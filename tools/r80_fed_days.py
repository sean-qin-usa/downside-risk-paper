# r80_fed_days.py -- SCHEDULED FEDERAL RESERVE DECISION DAYS ON EVERY PANEL: IS THERE A CALENDAR EFFECT OUTSIDE SINGLE STOCKS?
#
# On single stocks the earnings calendar is worth more than the choice of model (tools/r76, r78, r79). The ETF, currency
# and crypto panels have no earnings. Their natural scheduled event is the Federal Reserve's rate decision: 87 scheduled
# decision dates from 2016 to 2026, taken from the Board's meeting calendar (second day of each scheduled meeting;
# unscheduled meetings and conference calls excluded; data/calendars/fomc_scheduled_decision_dates.json).
#
# Per panel: the mean squared GARCH-t residual and the breach rates on decision days, the day after, and other days;
# a calendar multiplier on the two days, chosen for each model on the first half of the test dates and scored on the
# second half; and lambda (tools/r71) for large shocks that fell on a decision day or the day after, against the rest.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19). Known beforehand: on stocks the mean squared GARCH-t residual is 6
# to 8 on earnings days and the 2.5% breach rate 13 to 14%; lambda is near 1 on both ETF panels.
#  G1. On equity ETFs, other ETFs and currencies the mean squared GARCH-t residual on the decision day is between 1.0
#      and 2.5, and GARCH-t's 2.5% breach rate there is below 6%.
#  G2. Calendar multipliers gain less than 0.5% over GARCH-t over all days of the second half on each of these panels.
#  G3. On the equity ETF panel large shocks that fell on a decision day or the day after are under 10% of top-decile
#      rows and their lambda is at least 0.5: a Fed-day shock is not a one-off in the way an earnings shock is.
# Usage: python tools/r80_fed_days.py <universe>     (needs data/panels/<universe>_rows.npz and _dump.npz; canon: pass the rows file)
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); WHAT = sys.argv[1]; OUTD = os.path.join(ROOT, "results", "paper", "diagnostics", "r80"); os.makedirs(OUTD, exist_ok=True)
AL = (0.01, 0.025); GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 2); CG = np.round(np.arange(0.7, 3.001, 0.05), 2); NB = 500
if WHAT == "canon": R = np.load(sys.argv[2]); rk = R["rk_mk"]; MODELS = ["garch_t", "param_bteg", "body", "body_bteg"]
else: R = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_rows.npz")); rk = R["rk"]; MODELS = ["garch_t", "param_bteg", "body", "neural", "body_bteg"]
TAUS = [round(float(t), 4) for t in R["taus"]]; J = {a: TAUS.index(a) for a in AL}; Y = R["y"].astype(float); dates = pd.to_datetime(R["date"]); n = len(Y); Q = {m: R["q__" + m].astype(float) for m in MODELS}
S = np.load(os.path.join(ROOT, "data", "panels", WHAT + "_dump.npz")); SIG, MU, NU, DD = (S[k].astype(float) for k in ("sig", "mu", "nu", "D")); SDATE = pd.to_datetime(S["sdate"])
FED = pd.DatetimeIndex(json.load(open(os.path.join(ROOT, "data", "calendars", "fomc_scheduled_decision_dates.json")))); ud = np.sort(pd.unique(dates)); pos = np.searchsorted(ud, FED.values); pos = pos[pos < len(ud)]
d0 = ud[pos]; F0 = np.isin(dates.values, d0); F1 = np.isin(dates.values, ud[np.minimum(pos + 1, len(ud) - 1)]) & ~F0; OT = ~(F0 | F1); TOP = rk == 10
sd0 = set(pd.DatetimeIndex(ud[pos]).normalize()); sd1 = set(pd.DatetimeIndex(ud[np.minimum(pos + 1, len(ud) - 1)]).normalize()); SF = np.array([(d in sd0) or (d in sd1) for d in SDATE.normalize()])
tq = lambda a, nu: stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2)); pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; s = s[np.isfinite(s)]
    if len(s) < 30: return None
    dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
edge = lambda Lr, Lm, mask: None if mask.sum() < 50 else {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask), "rows": int(mask.sum())}
rng = np.random.default_rng(0)
def lam_hat(mask):
    if mask.sum() < 300: return {"rows": int(mask.sum())}
    cols = [pin(Y[mask], MU[mask] + SIG[mask] * np.sqrt(np.maximum(1 - (1 - l) * DD[mask], 0.02)) * tq(0.025, NU[mask]), 0.025) for l in GRID]
    M = pd.DataFrame(np.column_stack(cols), index=dates[mask]).groupby(level=0).sum().values; o = {"lambda": float(GRID[int(np.argmin(M.sum(0)))]), "rows": int(mask.sum())}
    if len(M) >= 60:
        T = len(M); idx = np.empty((NB, T), int); idx[:, 0] = rng.integers(0, T, NB)
        for t in range(1, T): new = rng.random(NB) < 0.1; idx[:, t] = np.where(new, rng.integers(0, T, NB), (idx[:, t - 1] + 1) % T)
        bs = GRID[np.argmin(M[idx].sum(1), axis=1)]; o["ci95"] = [round(float(np.quantile(bs, 0.025)), 2), round(float(np.quantile(bs, 0.975)), 2)]
    return o
z2 = ((Y - MU) / SIG) ** 2; cells = (("decision day", F0), ("day after", F1), ("other days", OT), ("all days", np.ones(n, bool)))
OUT = {"universe": WHAT, "rows": n, "decision_days_in_test": int(len(d0)), "predictions_written_before_run": "see the script header (G1-G3)",
       "mean_squared_garch_residual": {k: round(float(z2[m].mean()), 2) for k, m in cells}, "per_date_mean_squared_residual_median": {k: round(float(pd.Series(z2[m], index=dates[m]).groupby(level=0).mean().median()), 2) for k, m in cells},
       "breach_pct": {m_: {k: {str(a): round(100 * float((Y <= Q[m_][:, J[a]])[m].mean()), 2) for a in AL} for k, m in cells} for m_ in MODELS}}
cut = ud[len(ud) // 2]; H1 = dates < cut; H2 = dates >= cut; CM = {}; LL = {}
for m_ in MODELS:
    q = Q[m_][:, J[0.025]]; cs = [float(CG[int(np.argmin([pin(Y[fl], MU[fl] + c * (q[fl] - MU[fl]), 0.025).sum() for c in CG]))]) for fl in (H1 & F0, H1 & F1)]
    mm = np.where(F0, cs[0], np.where(F1, cs[1], 1.0)); CM[m_] = cs; LL[m_] = {a: pin(Y, MU + mm * (Q[m_][:, J[a]] - MU), a) for a in AL}
OUT["calendar_multipliers_second_half"] = {"cut_date": str(pd.Timestamp(cut).date()), "multipliers_decision_day_and_day_after": CM,
    **{str(a): {m_: {"gain_from_calendar_all_days": edge(pin(Y, Q[m_][:, J[a]], a), LL[m_][a], H2), "gain_on_the_two_days": edge(pin(Y, Q[m_][:, J[a]], a), LL[m_][a], H2 & ~OT)} for m_ in MODELS} for a in AL}}
OUT["lambda"] = {"share_of_top_decile_rows_with_fed_day_shock": round(float(SF[TOP].mean()), 3), "top_decile_fed_day_shock": lam_hat(TOP & SF), "top_decile_other_shock": lam_hat(TOP & ~SF),
                 "D_above_0.10_fed_day_shock": lam_hat((DD > 0.10) & SF), "D_above_0.10_other_shock": lam_hat((DD > 0.10) & ~SF)}
json.dump(OUT, open(os.path.join(OUTD, WHAT + ".json"), "w"), indent=1)
f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"]); fl_ = lambda b: "n/a (%d rows)" % b["rows"] if "lambda" not in b else "%.2f %s" % (b["lambda"], b.get("ci95", ""))
print("######## %s: %d decision days | mean z2 %s | median per-date z2 %s" % (WHAT, len(d0), OUT["mean_squared_garch_residual"], OUT["per_date_mean_squared_residual_median"]))
print("   GARCH-t breach:", OUT["breach_pct"]["garch_t"]); print("   multipliers:", CM, "| gain all days 2.5%:", {m_: f(v["gain_from_calendar_all_days"]) for m_, v in OUT["calendar_multipliers_second_half"]["0.025"].items()})
print("   lambda: Fed-day shocks are %.1f%% of top-decile rows | top decile: Fed %s, other %s | D>0.10: Fed %s, other %s" % (100 * OUT["lambda"]["share_of_top_decile_rows_with_fed_day_shock"], fl_(OUT["lambda"]["top_decile_fed_day_shock"]), fl_(OUT["lambda"]["top_decile_other_shock"]), fl_(OUT["lambda"]["D_above_0.10_fed_day_shock"]), fl_(OUT["lambda"]["D_above_0.10_other_shock"])))
