# r66_group_breakdown.py -- READ-ONLY BREAKDOWN of the paper's canonical benchmark comparison by kind of stock.
# Input: the per-row losses saved by `code/paper/job_bench_all.py --dump-rows <file>` (the canonical 200-name, 221.6k-row
# comparison, every benchmark on the same rows). Nothing is re-estimated here. Groups, each fixed before any loss is read:
#   sic        SIC division (crsp_panel_chars.csv `sector` = SIC code // 1000). Divisions with fewer than 10 of the 200
#              names are merged into "other"; division 9 (non-classifiable, mostly missing SIC codes) is kept separate
#              and not interpreted.
#   vol        tercile of each name's return standard deviation over its estimation window (first 60% of its history,
#              the window the filters are fitted on), so the grouping uses no test data.
#   cohort     crsp_panel_chars.csv `cohort` (largecap / smallcap / recent_ipo).
# Per group: the accuracy layer ("engine") against each benchmark on FZ0 at 1% and 2.5% and on eleven-level pinball,
# as a per-date series averaged over the group's names, Newey-West(10) DM; the share of names on which the engine has the
# lower mean loss; and the pinball edge of the pooled body over GARCH-t in the top mk63 decile within the group.
# Multiplicity: Holm across the groups of each grouping, per comparison. Heterogeneity: a Wald test that the engine's
# FZ0 gain over GARCH-t is equal across the groups of a grouping (group per-date means stacked by date, Newey-West(10)
# covariance, chi-square with G-1 df).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-17):
#  P1. The engine's FZ0 gain over GARCH-t is positive (point estimate) in every SIC group at 1%.
#  P2. The heterogeneity test does not reject across SIC groups at 5% at either level, and does reject across
#      volatility terciles at 1%, with the smallest gain (possibly negative) in the highest-volatility tercile.
#  P3. The engine and GJR-skew-t are within noise (|DM| < 1.96) in every SIC group at both levels.
#  P4. In the top mk63 decile the pooled body's pinball edge over GARCH-t is positive in every SIC group, and largest
#      in the highest-volatility tercile.
# Usage: GBC_PROJ=<panel dir> python tools/r66_group_breakdown.py <dump.npz>
import os, sys, json, math
import numpy as np, pandas as pd
from scipy import stats
P = os.environ["GBC_PROJ"]; D = np.load(sys.argv[1], allow_pickle=False)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "paper", "diagnostics", "r66_group_breakdown.json")
SIC = {0: "agriculture", 1: "mining and construction", 2: "manufacturing (nondurable: food, chemicals, pharma)",
       3: "manufacturing (durable: machinery, electronics, computers)", 4: "transport, communications, utilities",
       5: "wholesale and retail trade", 6: "finance, insurance, real estate", 7: "business services (incl. software)",
       8: "health and other services", 9: "non-classifiable (mostly missing SIC)"}
PN = D["permno"]; dates = pd.to_datetime(D["date"]); rk = D["rk_mk"]
def nw_cov(X, l=10):
    X = np.asarray(X, float); X = X - X.mean(0); T = len(X); S = X.T @ X / T
    for k in range(1, l + 1): G = X[k:].T @ X[:-k] / T; S += (1 - k / (l + 1)) * (G + G.T)
    return S / T
def series(L, mask):
    s = pd.Series(L[mask], index=dates[mask]); return s.groupby(level=0).mean()
def dm(x): x = np.asarray(x, float); return float(x.mean() / math.sqrt(nw_cov(x[:, None])[0, 0]))
def holm(p):
    k = sorted(range(len(p)), key=lambda i: p[i]); adj = [0.0] * len(p); run = 0.0
    for r, i in enumerate(k): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj

# ---------------------------------------------------------------- groups (no test data used)
ch = pd.read_csv(os.path.join(P, "crsp_panel_chars.csv")).set_index("permno")
rr = pd.read_csv(os.path.join(P, "crsp_panel_returns.csv"), dtype={'permno': 'int32'})
rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce'); rr['date'] = pd.to_datetime(rr['date'])
names = np.unique(PN); sd_est = {}
for pn in names:
    y = rr[rr.permno == pn].sort_values('date')['ret'].values; sd_est[pn] = float(np.nanstd(y[:int(len(y) * 0.6)]))
q1, q2 = np.quantile(list(sd_est.values()), [1 / 3, 2 / 3])
sic = {pn: int(ch.loc[pn, "sector"]) for pn in names}; cnt = pd.Series(sic).value_counts()
GROUPS = {
    "sic": {pn: (SIC[s] if (cnt[s] >= 10 or s == 9) else "other (divisions with <10 names)") for pn, s in sic.items()},
    "vol": {pn: ("low" if v <= q1 else "mid" if v <= q2 else "high") + " volatility tercile" for pn, v in sd_est.items()},
    "cohort": {pn: str(ch.loc[pn, "cohort"]) for pn in names}}
BENCH = ["garch_t", "gjr_skewt", "fhs_pool", "fhs_name", "evt_pool", "evt_name", "ewma", "hs500", "gas_pzc", "taylor"]
out = {"note": "Canonical bench_all rows broken down by kind of stock; engine = accuracy layer. gain > 0 and DM > 0 mean the "
               "engine has the LOWER loss than the benchmark.",
       "predictions_written_before_run": "P1 engine FZ0 gain over GARCH-t positive in every SIC group at 1%; P2 heterogeneity "
               "not rejected across SIC groups at either level, rejected across volatility terciles at 1% with the smallest "
               "gain in the high tercile; P3 engine and GJR-skew-t within noise in every SIC group at both levels; P4 top-decile "
               "body pinball edge over GARCH-t positive in every SIC group and largest in the high-volatility tercile.",
       "volatility_tercile_cutoffs_pct_daily": [round(100 * q1, 3), round(100 * q2, 3)], "groupings": {}}
for gname, gmap in GROUPS.items():
    labels = sorted(set(gmap.values())); gpn = np.array([gmap[p] for p in PN]); res = {}
    for lab in labels:
        m = gpn == lab; r = {"n_names": int(len(set(PN[m]))), "n_rows": int(m.sum())}
        for a in ("0.01", "0.025"):
            Le = D["fz0__engine__" + a]; ra = {}
            for b in BENCH:
                k = "fz0__%s__%s" % (b, a)
                if k not in D.files: continue
                Lb = D[k]; ok = m & np.isfinite(Lb) & np.isfinite(Le); s = series(Lb - Le, ok)
                per = pd.DataFrame({"pn": PN[ok], "d": (Lb - Le)[ok]}).groupby("pn")["d"].mean()
                ra[b] = {"gain": round(float(s.mean()), 4), "DM": round(dm(s.values), 2), "names_engine_lower": int((per > 0).sum()), "names": int(len(per))}
            r["fz0_" + a] = ra
        s = series(D["pin__garch_t"] - D["pin__engine"], m)
        r["pinball_engine_vs_garch_t"] = {"edge_pct": round(100 * float(s.mean() / series(D["pin__garch_t"], m).mean()), 3), "DM": round(dm(s.values), 2)}
        mt = m & (rk == 10)
        if mt.sum() > 500:
            s = series(D["pin__garch_t"] - D["pin__body"], mt)
            r["top_decile_body_vs_garch_t_pinball"] = {"edge_pct": round(100 * float(s.mean() / series(D["pin__garch_t"], mt).mean()), 3), "DM": round(dm(s.values), 2), "rows": int(mt.sum())}
        res[lab] = r
    # Holm across groups for engine vs GARCH-t and vs GJR-skew-t (two-sided)
    for a in ("0.01", "0.025"):
        for b in ("garch_t", "gjr_skewt"):
            p = [2 * (1 - stats.norm.cdf(abs(res[l]["fz0_" + a][b]["DM"]))) for l in labels]
            for l, pa in zip(labels, holm(p)): res[l]["fz0_" + a][b]["p_holm_two_sided"] = round(pa, 4)
    # heterogeneity: equal engine-minus-GARCH-t gain across groups
    het = {}
    for a in ("0.01", "0.025"):
        Lg, Le = D["fz0__garch_t__" + a], D["fz0__engine__" + a]
        S = pd.concat([series(Lg - Le, gpn == l).rename(l) for l in labels], axis=1).dropna()
        mu = S.mean().values; V = nw_cov(S.values); G = len(labels); R = np.eye(G)[1:] - np.eye(G)[:-1]
        w = float((R @ mu) @ np.linalg.solve(R @ V @ R.T, R @ mu))
        het[a] = {"wald_chi2": round(w, 2), "df": G - 1, "p": round(float(1 - stats.chi2.cdf(w, G - 1)), 4), "dates_all_groups": int(len(S))}
    out["groupings"][gname] = {"groups": res, "heterogeneity_engine_vs_garch_t": het}
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
for gname, g in out["groupings"].items():
    print("\n=== %s   (FZ0 gain of the engine over the benchmark, DM; heterogeneity p: 1%% %.3f, 2.5%% %.3f)" % (
        gname, g["heterogeneity_engine_vs_garch_t"]["0.01"]["p"], g["heterogeneity_engine_vs_garch_t"]["0.025"]["p"]))
    for lab, r in g["groups"].items():
        f1, f2 = r["fz0_0.01"], r["fz0_0.025"]; td = r.get("top_decile_body_vs_garch_t_pinball", {})
        print("  %-58s n=%3d | vs GARCH-t 1%% %+.4f (%+.2f) 2.5%% %+.4f (%+.2f) | vs GJR-skew-t 1%% %+.2f 2.5%% %+.2f | names won 1%% %d/%d | top-dec body pin %s" % (
            lab[:58], r["n_names"], f1["garch_t"]["gain"], f1["garch_t"]["DM"], f2["garch_t"]["gain"], f2["garch_t"]["DM"],
            f1["gjr_skewt"]["DM"], f2["gjr_skewt"]["DM"], f1["garch_t"]["names_engine_lower"], f1["garch_t"]["names"],
            ("%+.2f%% (DM %.1f)" % (td["edge_pct"], td["DM"])) if td else "--"))
print("\nwrote", OUT)
