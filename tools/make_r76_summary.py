# make_r76_summary.py -- Markdown tables for docs/EARNINGS_CALENDAR_RESULTS.md from results/paper/diagnostics/r76 and r78.
# Usage: python tools/make_r76_summary.py >> docs/EARNINGS_CALENDAR_RESULTS.md   (after the hand-written part)
import os, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "results", "paper", "diagnostics")
UNI = [("canon", "CRSP stocks (canonical)"), ("us_largecap", "US large caps"), ("intl_largecap_screened", "International large caps, screened"), ("crypto", "Crypto"), ("fx", "Currencies"), ("etf_equity", "Equity ETFs"), ("etf_other", "Bond and commodity ETFs")]
R = {u: json.load(open(os.path.join(D, "r76", u + ".json"))) for u, _ in UNI}; C = {u: json.load(open(os.path.join(D, "r78", u + ".json"))) for u in ("canon", "us_largecap")}
f = lambda e: "n/a" if e is None else "%+.1f%% (%s)" % (e["edge_pct"], e["DM"]); fl = lambda b: "n/a" if (b is None or "lambda" not in b) else "%.2f [%.2f, %.2f]" % (b["lambda"] + 0.0, *b["ci95"]) if "ci95" in b else "%.2f" % b["lambda"]
CL = ("firm-specific", "mixed", "market-wide"); NM = {"cf_pre": "corrected GARCH-t", "param_bteg": "Beta-t-EGARCH", "body": "pooled trees", "neural": "neural network", "body_bteg": "trees on robust scale"}

print("\n### Table 1. Origin of the largest shock: share of top-decile rows, and lambda (95% interval) in the top decile and in all rows with D above 0.10\n")
print("| Universe | Class | Share of top decile | Rows (events) | lambda, top decile | GARCH-t 2.5% breach | lambda, D > 0.10 | GARCH-t 2.5% breach |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI:
    H = R[u]["H_origin"]
    for c in CL:
        t, d = H["top_decile"][c], H["rows_with_D_above_0.10"][c]
        print("| %s | %s | %.1f%% | %d (%s) | %s | %s | %s | %s |" % (lab, c, 100 * H["share_of_top_decile_rows"][c], t["rows"], t.get("shock_events", "n/a"), fl(t.get("lambda")), "%.2f%%" % t["garch_t_breach_0.025"] if "lambda" in t else "n/a", fl(d.get("lambda")), "%.2f%%" % d["garch_t_breach_0.025"] if "lambda" in d else "n/a"))
print("\nMarket-wide shock dates holding the most top-decile rows: " + "; ".join("%s: %s" % (lab, ", ".join(list(R[u]["H_origin"]["market_wide_shock_dates_most_rows"])[:5])) for u, lab in UNI) + ".")

print("\n### Table 2. Origin crossed with the earnings flag (US stock universes): lambda and the corrected GARCH-t's 2.5% edge over GARCH-t\n")
print("| Universe | Shock | Rows, top decile | lambda, top decile | Correction edge | Rows, D > 0.10 | lambda, D > 0.10 | Correction edge |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI[:2]:
    H = R[u]["H_origin"]
    for c in H["crossed_with_earnings_top_decile"]:
        t, d = H["crossed_with_earnings_top_decile"][c], H["crossed_with_earnings_D_above_0.10"][c]
        print("| %s | %s | %d | %s | %s | %d | %s | %s |" % (lab, c, t["rows"], fl(t.get("lambda")), f(t["edges_over_garch_t_0.025"]["cf0"]) if "lambda" in t else "n/a", d["rows"], fl(d.get("lambda")), f(d["edges_over_garch_t_0.025"]["cf0"]) if "lambda" in d else "n/a"))

for a in ("0.025", "0.01"):
    print("\n### Table 3%s. Accounting at the %s level: the edge over GARCH-t that the closed form predicts with the pre-test lambda, and what each model gained (DM)\n" % ("a" if a == "0.025" else "b", a))
    print("| Universe | Pre-test lambda | Days | Predicted | " + " | ".join(NM.values()) + " |"); print("|---|---|---|---|" + "---|" * len(NM))
    for u, lab in UNI:
        K = R[u]["K_accounting"][a]; t = K["by_D_decile"][-1]
        print("| %s | %.2f | all | %+.1f%% | %s |" % (lab, R[u]["lambda_pre_test"], K["all_days_predicted_edge_pct"], " | ".join(f(K["all_days_realized"].get(m)) for m in NM)))
        print("| | | top D decile (mean D %.2f) | %+.1f%% | %s |" % (t["mean_D"], t["predicted_edge_pct"], " | ".join(f(t.get(m)) for m in NM)))
        print("| | | correlation over the ten D deciles | | %s |" % " | ".join("%.2f" % K["correlation_across_D_deciles"][m] if m in K["correlation_across_D_deciles"] else "n/a" for m in NM))
        print("| | | Spearman across %d names | | %s |" % (K["across_names"]["names"], " | ".join("%.2f" % K["across_names"][m]["spearman"] if m in K["across_names"] else "n/a" for m in NM)))

print("\n### Table 4. Announcement days: breach rate of each model's VaR by type of day (all values of D), and the mean squared GARCH-t residual\n")
print("| Universe | Day | Rows | GARCH-t 2.5% | Beta-t-EGARCH 2.5% | Pooled trees 2.5% | GARCH-t 1% | Mean squared residual |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI[:2]:
    Qd = R[u]["Q_quiet_days"]
    for k in ("announcement date", "day after", "other days", "all days"):
        v = Qd["all_D"][k]; print("| %s | %s | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %s |" % (lab, k, v["rows"], v["breach_garch_t_0.025"], v["breach_param_bteg_0.025"], v["breach_body_0.025"], v["breach_garch_t_0.01"], Qd["mean_squared_garch_residual"].get(k, "")))
    for k in ("announcement window", "other days", "all days"):
        v = Qd["D_below_0.05"][k]; print("| %s | %s, D < 0.05 | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | |" % (lab, k, v["rows"], v["breach_garch_t_0.025"], v["breach_param_bteg_0.025"], v["breach_body_0.025"], v["breach_garch_t_0.01"]))

print("\n### Table 5. Fitted GARCH-t parameters (medians over names) with announcement days masked, and the control fitted by the same code with nothing masked\n")
print("| Universe | Names | Fit | alpha | beta | nu | c0 (announcement date) | c1 (day after) |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI[:2]:
    for k in ("masked", "control"): p = C[u]["median_parameters"][k]; print("| %s | %d | %s | %.3f | %.3f | %.2f | %s | %s |" % (lab, C[u]["names_fitted"], k, p["alpha"], p["beta"], p["nu"], C[u]["c0_announcement_date"] if k == "masked" else "", C[u]["c1_day_after"] if k == "masked" else ""))

CE = ("all days", "announcement window", "other days", "top decile", "top decile, other days", "deciles 1-8, other days"); RN = {"garch_t": "GARCH-t", "param_bteg": "Beta-t-EGARCH", "body": "pooled trees", "neural": "neural network", "body_bteg": "trees on robust scale", "simple calendar (r76)": "simple calendar model", "masked, no multiplier": "masked, no multiplier"}
for a in ("0.025", "0.01"):
    print("\n### Table 6%s. The masked calendar GARCH-t against each model at the %s level (positive: the calendar model has the lower loss; DM)\n" % ("a" if a == "0.025" else "b", a)); print("| Universe | Against | " + " | ".join(CE) + " |"); print("|---|---|" + "---|" * len(CE))
    for u, lab in UNI[:2]:
        for r_, nm in RN.items(): print("| %s | %s | %s |" % (lab, nm, " | ".join(f(C[u]["masked_vs"][a][r_][c]) for c in CE)))
    print("\nBreach rates (%s nominal), other days: " % a + "; ".join("%s: %s" % (lab, ", ".join("%s %.2f%%" % (RN.get(m, "masked calendar"), v["other days"]) for m, v in C[u]["breach_pct"][a].items() if m in RN or m == "masked")) for u, lab in UNI[:2]) + ".")

print("\n### Table 7. Like for like: every model given calendar multipliers chosen on the first half of the test dates; second half only (DM)\n")
print("| Universe | Level | Model | What the calendar adds to it | Edge over GARCH-t with calendar, all days | Other days | Masked GARCH-t against it, all days | Other days |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI[:2]:
    Z = C[u]["second_half_all_models_with_calendar"]
    for a in ("0.025", "0.01"):
        for m, nm in (("garch_t", "GARCH-t"), ("param_bteg", "Beta-t-EGARCH"), ("body", "pooled trees"), ("neural", "neural network"), ("body_bteg", "trees on robust scale"), ("masked, no multiplier", "masked GARCH-t")):
            g = Z[a]["vs_garch_t_with_calendar"].get(m); k = Z[a]["masked_vs"].get(m)
            print("| %s (from %s) | %s | %s | %s | %s | %s | %s | %s |" % (lab, Z["cut_date"], a, nm, f(Z[a]["calendar_gain_of_each_model"][m]), f(g["all days"]) if g else "", f(g["other days"]) if g else "", f(k["all days"]) if k else "", f(k["other days"]) if k else ""))

print("\n### Table 8. What the learned tail on the robust scale adds to Beta-t-EGARCH: edge (DM), mean log ratio of the two quantiles, breach rates, share of the total gain\n")
print("| Universe | Level | Rows | Edge over Beta-t-EGARCH | Mean log quantile ratio | Breach, Beta-t-EGARCH | Breach, with learned tail | Share of total gain |"); print("|---|---|---|---|---|---|---|---|")
for u, lab in UNI[:3]:
    for a in ("0.01", "0.025", "0.05"):
        for c, v in R[u]["S_shape"][a].items():
            if c.startswith(("positive", "negative", "all rows except")): continue
            print("| %s | %s | %s | %s | %+.3f | %.2f%% | %.2f%% | %.2f |" % (lab, a, c, f(v["edge_over_bteg"]), v["mean_log_quantile_ratio"], v["bteg_breach_pct"], v["body_bteg_breach_pct"], v["share_of_total_gain"]))
