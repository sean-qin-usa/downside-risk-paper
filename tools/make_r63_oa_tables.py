# make_r63_oa_tables.py -- table bodies for the two R63 online-appendix tables, from committed result files only.
#   tab:anatomy      results/paper/diagnostics/r63_mk63_anatomy_{canon200,holdout}.json (tools/r63_mk63_anatomy.py)
#   tab:attribution  results/paper/bench_all_bteg_results.json (code/paper/job_bench_all_bteg.py)
# Prints LaTeX tabular bodies; they are pasted into submission/paper_A_jfec_online_appendix.tex between the
# BEGIN/END markers that name this script. Run from the repo root: python tools/make_r63_oa_tables.py
import json
R = "results/paper/"

def anatomy():
    out = []
    for panel, title in (("canon200", "Panel A: 2014--2024 design panel (Table~2 rows, 221.6k asset-days)"),
                         ("holdout", "Panel B: 2000--2013 holdout")):
        d = json.load(open(R + "diagnostics/r63_mk63_anatomy_%s.json" % panel))
        out.append(r"\multicolumn{7}{l}{\emph{%s}} \\" % title)
        for k in map(str, range(1, 11)):
            v = d["by_decile"][k]
            out.append("%s & %.2f & %.2f & %.2f & %.1f & %.2f & %.2f \\\\" % (
                k, v["mk63_median"], v["share1_q25_med_q75"][1], v["frac_share1_gt_half"],
                v["zraw_q25_med_q75"][1], v["sd_rest_q25_med_q75"][1], v["mk_drop1_q25_med_q75"][1]))
        if panel == "canon200": out.append(r"\midrule")
    return "\n".join(out)

def attribution():
    pb = json.load(open(R + "bench_all_bteg_results.json"))["pinball"]
    V, A, B = pb["vs_garch_t"], pb["mk63_decile_profile"], pb["mk63_decile_profile_vs_param_bteg"]
    rows = [("Pooled body (GARCH-$t$ scale)", "body"), ("Accuracy layer (GARCH-$t$ scale)", "engine"),
            ("SAV-CAViaR", "sav_caviar"), ("GJR-GARCH-skew-$t$", "gjr_skewt"), ("GARCH-EVT (per name)", "evt_name"),
            ("GARCH-FHS (per name)", "fhs_name"), ("Beta-$t$-EGARCH, Student-$t$ tail", "param_bteg"),
            ("Pooled body (Beta-$t$-EGARCH scale)", "body_bteg"), ("Accuracy layer (Beta-$t$-EGARCH scale)", "engine_bteg")]
    f = lambda e, t: "$%+.2f$ (%.2f)" % (e, t)
    out = []
    for lab, m in rows:
        v = V[m]
        top_b = f(B[m]["10"]["edge_pct"], B[m]["10"]["DM"]) if m in B else "(reference)"
        out.append("%s & %s & %s & %s & %s \\\\" % (lab, f(v["top_mk63_decile"]["edge_pct"], v["top_mk63_decile"]["DM"]),
                   f(v["bulk_mk63_d1to9"]["edge_pct"], v["bulk_mk63_d1to9"]["DM"]),
                   f(v["overall"]["edge_pct"], v["overall"]["DM"]), top_b))
    return "\n".join(out)

if __name__ == "__main__":
    print("% ---- tab:anatomy body"); print(anatomy())
    print("% ---- tab:attribution body"); print(attribution())
