# make_r71_summary.py -- prints the Markdown tables of docs/SHOCK_RESPONSE_RESULTS.md from the result files of
# tools/r71_shock_response.py, r72_spa_mcs.py, r73_simulation.py and r75_conditional_tests.py, so no number in that
# document is typed by hand.   Usage: python tools/make_r71_summary.py > /tmp/tables.md
import os, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); DG = os.path.join(ROOT, "results", "paper", "diagnostics")
UNI = [("canon", "CRSP stocks (canonical, 200)"), ("us_largecap", "US large caps (201)"), ("intl_largecap_screened", "International large caps, screened"), ("intl_largecap", "International large caps, as registered"),
       ("crypto", "Crypto (27)"), ("fx", "Currencies (29)"), ("etf_equity", "Equity ETFs (113)"), ("etf_other", "Bond, commodity, currency ETFs (32)")]
def load(d, u):
    f = os.path.join(DG, d, u + ".json"); return json.load(open(f)) if os.path.exists(f) else None
R71 = {u: load("r71", u) for u, _ in UNI}; R72 = {u: load("r72", u) for u, _ in UNI}; R75 = {u: load("r75", u) for u, _ in UNI}
fl = lambda b: "n/a" if not b else ("%.2f" % b["lambda"] + (" [%.2f, %.2f]" % tuple(b["ci95"]) if "ci95" in b else ""))
fe = lambda e: "n/a" if not e else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"])
print("### Table 1. The share of a large shock's variance impact that is real (lambda), by universe\n")
print("| universe | names | rows | median beta (half-life, days) | largest shock, top decile (SD) | lambda, 2.5% pinball [95%] | lambda, 1% pinball | lambda, likelihood | positive shocks | negative shocks | lambda from pre-test rows |\n|---|---|---|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    a = r["all"]; b = a["B_lambda"]
    print("| %s | %d | %d | %.3f (%.1f) | %.1f | %s | %s | %s | %s | %s | %.2f |" % (lab, a["names"], a["rows"], a["median_beta"], a["median_half_life_days"], a["top_decile_median_shock_size"], fl(b["top_decile_pinball_0.025"]), fl(b["top_decile_pinball_0.01"]),
          fl(b["top_decile_likelihood"]), fl(b["positive_shock"]), fl(b["negative_shock"]), r["lambda_pre_test"]["0.025"]))
print("\n### Table 2. Top score decile, 2.5% pinball edge over GARCH-t (DM)\n")
print("| universe | corrected GARCH-t, lambda = 0 | corrected, lambda from pre-test rows | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t |\n|---|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    e = r["all"]["A_top_decile"]["0.025"]["edges_over_garch_t"]; print("| %s | %s |" % (lab, " | ".join(fe(e.get(m)) for m in ("cf0", "cf_pre", "param_bteg", "body", "sav_caviar", "neural", "body_bteg", "gjr_skewt"))))
print("\nAll days:\n\n| universe | corrected GARCH-t, lambda = 0 | corrected, lambda from pre-test rows | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t |\n|---|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    e = r["all"]["A_all_days"]["0.025"]; print("| %s | %s |" % (lab, " | ".join(fe(e.get(m)) for m in ("cf0", "cf_pre", "param_bteg", "body", "sav_caviar", "neural", "body_bteg", "gjr_skewt"))))
print("\n### Table 3. Top score decile by age of the shock: GARCH-t's 2.5% breach rate, realized / predicted by lambda = 0; then the 2.5% quantile of residuals on the corrected scale (Student-t value in the header row of each universe)\n")
ages = ["1-5", "6-10", "11-15", "16-21", "22-30", "31-45", "46-63"]; print("| universe | " + " | ".join(ages) + " |\n|---|" + "---|" * len(ages))
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    c = r["all"]["A_top_decile_by_shock_age"]
    print("| %s: breach | %s |" % (lab, " | ".join("%.2f / %.2f" % (c[a]["0.025"]["garch_t_breach_pct"], c[a]["0.025"]["predicted_breach_pct_lambda0"]) if "0.025" in c[a] else "n/a" for a in ages)))
    print("| corrected-scale quantile (t: %.2f) | %s |" % (c[ages[-1]]["0.025"]["student_t_quantile"] if "0.025" in c[ages[-1]] else float("nan"), " | ".join("%.2f" % c[a]["0.025"]["realized_quantile_corrected_residual"] if "0.025" in c[a] else "n/a" for a in ages)))
print("\n### Table 4. Duration and the persistence parameter: GARCH-t's 2.5% breach rate in the top decile by shock age, names split by beta tercile (realized / predicted by lambda = 0)\n")
print("| universe | tercile | median beta | half-life (days) | 1-10 | 11-21 | 22-42 | 43-63 |\n|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    for g, v in r["all"]["D_duration_by_beta_tercile"].items():
        print("| %s | %s | %.3f | %.1f | %s |" % (lab, g, v["median_beta"], v["half_life_days"], " | ".join("n/a" if x["garch_t_breach_0.025"] is None else "%.2f / %.2f" % (x["garch_t_breach_0.025"], x["predicted_lambda0"]) for x in v["by_age"].values())))
print("\n### Table 5. All test rows by the shock share D: GARCH-t's 2.5% breach rate, realized / predicted by lambda = 0\n")
bins = ["0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.35", "0.35-0.50", "0.50-0.70", "0.70-1.00"]; print("| universe | " + " | ".join(bins) + " |\n|---|" + "---|" * len(bins))
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    E = r["all"]["E_dose_response_all_rows"]; print("| %s | %s |" % (lab, " | ".join("%.2f / %.2f" % (E[b]["breach_0.025"]["garch_t"], E[b]["breach_0.025"]["predicted_lambda0"]) if b in E and "breach_0.025" in E[b] else "n/a" for b in bins)))
print("\n### Table 6. What each model discards: slope of (1 - squared VaR ratio to GARCH-t) on D, top decile (R-squared)\n")
ms = ["param_bteg", "body", "sav_caviar", "neural", "body_bteg", "gjr_skewt", "engine"]; print("| universe | Beta-t-EGARCH | pooled trees | SAV-CAViaR | neural network | trees on Beta-t-EGARCH | GJR-skew-t | accuracy layer |\n|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    C = r["all"]["C_what_each_model_discards"]; print("| %s | %s |" % (lab, " | ".join("%.2f (%.2f)" % (C[m]["slope_discard_share"], C[m]["r2"]) if m in C else "n/a" for m in ms)))
print("\n### Table 7. D against the kurtosis score as a locator: pooled trees' 2.5% edge over GARCH-t (DM)\n")
print("| universe | top D decile | top kurtosis decile | overlap | D only | kurtosis only |\n|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    F = r["all"]["F_D_as_score"]; x = F["0.025"]; print("| %s | %s | %s | %.2f | %s | %s |" % (lab, fe(x["top_D_decile"]["body"]), fe(x["top_kurtosis_decile"]["body"]), F["overlap_top_D_with_top_kurtosis"], fe(x["top_D_not_top_kurtosis"]["body"]), fe(x["top_kurtosis_not_top_D"]["body"])))
print("\n### Table 8. Earnings shocks (US stock universes)\n")
print("| universe | names with dates | days in an earnings window | top-decile rows whose largest shock is an earnings shock | lambda, earnings shocks | lambda, other shocks | corrected GARCH-t edge: earnings / other | pooled trees edge: earnings / other |\n|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r or "G_earnings" not in r["all"]: continue
    G = r["all"]["G_earnings"]; e1, e0 = G["earnings_shock"]["0.025"]["edges_over_garch_t"], G["other_shock"]["0.025"]["edges_over_garch_t"]
    print("| %s | %d | %.1f%% | %.1f%% | %s | %s | %s / %s | %s / %s |" % (lab, G["names_with_dates"], 100 * G["share_of_trading_days_in_an_earnings_window"], 100 * G["share_of_top_decile_rows_with_earnings_shock"], fl(G["lambda_earnings_shock"]), fl(G["lambda_other_shock"]), fe(e1["cf0"]), fe(e0["cf0"]), fe(e1["body"]), fe(e0["body"])))
print("\n### Table 9. Subgroups: lambda (2.5% pinball, top decile) and GARCH-t's 2.5% breach rate at shock ages 1-10 and 6-15\n")
print("| universe | grouping | group | names | lambda [95%] | breach, ages 1-10 | breach, ages 6-15 | pooled trees' top-decile edge |\n|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r: continue
    for gn, gs in r["groups"].items():
        for g, v in gs.items():
            if "skipped" in v: continue
            c = v["A_top_decile_by_shock_age"]["1-10"]; c6 = v.get("A_top_decile_ages_6_15", {})
            print("| %s | %s | %s | %d | %s | %s | %s | %s |" % (lab, gn, g, v["names"], fl(v["B_lambda"]["top_decile_pinball_0.025"]), "%.2f" % c["0.025"]["garch_t_breach_pct"] if "0.025" in c else "n/a", "%.2f" % c6["0.025"]["garch_t_breach_pct"] if "0.025" in c6 else "n/a",
                  fe(v["A_top_decile"]["0.025"]["edges_over_garch_t"]["body"]) if "0.025" in v["A_top_decile"] else "n/a"))
print("\n### Table 10. Stability over the test period: lambda by third of the test dates\n")
print("| universe | first third | second third | last third |\n|---|---|---|---|")
for u, lab in UNI:
    r = R71[u]
    if not r or "by_test_period" not in r["all"]["B_lambda"]: continue
    bp = r["all"]["B_lambda"]["by_test_period"]; print("| %s | %s |" % (lab, " | ".join("%s: %s" % (k, fl(v)) for k, v in bp.items())))
print("\n### Table 11. Superior predictive ability (consistent p-value), 2.5% pinball; best model in the set and its edge\n")
print("| universe | days | GARCH-t vs flexible set | GARCH-t vs all | Beta-t-EGARCH vs flexible set | Beta-t-EGARCH vs learned tail on it | model confidence set (10%, T-max) |\n|---|---|---|---|---|---|---|")
fs = lambda x: "n/a" if not x else "p = %.3f (%s, %+.1f%%)" % (x["p_consistent"], x["best_model"], x["best_edge_pct"])
for u, lab in UNI:
    r = R72[u]
    if not r: continue
    for sn in ("top_decile", "deciles1to8", "all_days"):
        x = r["results"][sn]["pinball_0.025"]; g = x.get("spa_benchmark_garch_t", {}); b = x.get("spa_benchmark_param_bteg", {})
        print("| %s | %s | %s | %s | %s | %s | %s |" % (lab, sn, fs(g.get("flexible")), fs(g.get("all")), fs(b.get("flexible")), fs(b.get("robust_scale_learned_shape")), ", ".join(x["mcs_10pct_max"]["included"])))
print("\n### Table 12. Conditional coverage given D (2.5%): slope of the hit rate on D (t), implied breach rate at D = 0 and D = 0.5\n")
ms = ["garch_t", "gjr_skewt", "param_bteg", "sav_caviar", "neural", "body", "body_bteg", "cf_pre"]; print("| universe | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms))
for u, lab in UNI:
    r = R75[u]
    if not r: continue
    A = r["A_conditional_coverage"]["0.025"]; print("| %s | %s |" % (lab, " | ".join("%+.3f (%.1f); %.2f%% to %.2f%%" % (A[m]["slope_on_D"], A[m]["t_slope"], A[m]["implied_breach_pct_at_D_0"], A[m]["implied_breach_pct_at_D_0.5"]) if m in A else "n/a" for m in ms)))
print("\n### Table 13. Conditional predictive ability given D (2.5% pinball): slope of the gain over GARCH-t on D, t-statistic; then t of D and of the kurtosis-decile dummy when both enter\n")
ms = ["param_bteg", "sav_caviar", "neural", "body", "body_bteg", "gjr_skewt"]; print("| universe | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms))
for u, lab in UNI:
    r = R75[u]
    if not r: continue
    B = r["B_conditional_predictive_ability"]["0.025"]; print("| %s | %s |" % (lab, " | ".join("t = %.1f; both: %.1f, %.1f" % (B[m]["on_D"]["t_slope"], B[m]["on_D_and_kurtosis_decile"]["t_D"], B[m]["on_D_and_kurtosis_decile"]["t_kurtosis_dummy"]) if m in B else "n/a" for m in ms)))
f = os.path.join(DG, "r73_simulation.json")
if os.path.exists(f):
    S = json.load(open(f))["results"]; print("\n### Table 14. Simulation: mean (standard deviation) over simulated panels\n"); names = list(S); print("| quantity | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    rows = [("true lambda", None), ("fitted alpha, median", "fitted_alpha_median"), ("fitted beta, median", "fitted_beta_median"), ("lambda-hat, top decile, pinball", "lambda_top_decile_pinball"), ("lambda-hat, top decile, likelihood", "lambda_top_decile_likelihood"),
            ("lambda-hat, pre-test rows", "lambda_pre_test"), ("GARCH-t breach, top decile (%)", "garch_t_breach_top_decile"), ("GARCH-t breach, shock age 1-10 (%)", "breach_age_1-10"), ("predicted by lambda = 0 (%)", "predicted_lambda0_age_1-10"),
            ("GARCH-t breach, shock age 43-63 (%)", "breach_age_43-63"), ("corrected GARCH-t (lambda = 0) edge, top decile (%)", "edge_cf0_top_decile"), ("corrected (pre-test lambda) edge, top decile (%)", "edge_cf_pre_top_decile"),
            ("Beta-t-EGARCH edge, top decile (%)", "edge_bteg_top_decile"), ("Beta-t-EGARCH edge, all days (%)", "edge_bteg_all_days"), ("corrected (pre-test lambda) edge, all days (%)", "edge_cf_pre_all_days")]
    TRUE = {"garch": "1", "jump_0": "0", "jump_0.25": "0.25", "jump_1": "1"}
    for lab, k in rows: print("| %s | %s |" % (lab, " | ".join(TRUE.get(n_, "") if k is None else "%.2f (%.2f)" % (S[n_][k]["mean"], S[n_][k]["sd"]) for n_ in names)))
