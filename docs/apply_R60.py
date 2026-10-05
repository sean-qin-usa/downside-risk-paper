# apply_R60.py -- R60: three scope statements on top of a04779a. No number changes.
#  M1  intro: Francq-Zakoian's theory is for the filtered empirical quantile
#  M2  Section 2.2: no asymptotic result covers a pooled tree learner; inference is on the forecasts
#  M3  Section 3 conventions: Giacomini-White covers any estimator under the fixed scheme of the headline
#      comparisons; the expanding-window refits are robustness checks
#  R1  README: which jobs rebuild their own engine, and that every FZ0 claim against benchmarks uses bench_all
# Exact-unique replacements; every anchor is checked before anything is written.
# Run from the repo root:  python docs/apply_R60.py        (add --dry to check without writing)
import sys, io

FILES = ["submission/paper_A_jfec.tex", "preprint/paper_A_ssrn.tex"]
EDITS = []
E = lambda n, b, a: EDITS.append((n, b, a))

E("M1 intro: scope of Francq-Zakoian",
r"""\citep{engle2004caviar}, and \citet{francqzakoian2015} supply the
two-step estimator's asymptotic theory.""",
r"""\citep{engle2004caviar}, and \citet{francqzakoian2015} supply
asymptotic theory for the filtered empirical quantile.""")

E("M2 Section 2.2: no theory for a pooled tree learner",
r"""This two-step design, a parametric filter followed by a residual quantile, is the one whose asymptotics \citet{francqzakoian2015} derive, and \citet{mcneilfrey2000} fit the same filter by Gaussian pseudo-likelihood while explicitly declining to believe the innovation law it assumes.""",
r"""This two-step design, a parametric filter followed by a residual quantile, is the one whose asymptotics \citet{francqzakoian2015} derive for an empirical residual quantile, and \citet{mcneilfrey2000} fit the same filter by Gaussian pseudo-likelihood while explicitly declining to believe the innovation law it assumes. No comparable result covers a pooled tree learner in the second stage, so the paper's inference is on the forecasts (Section~\ref{sec:conventions}).""")

E("M3 conventions: Giacomini-White and the estimation scheme",
r"""the use \citet{diebold2015} endorses.""",
r"""the use \citet{diebold2015} endorses. Estimation is part of each forecasting method, and the conditional predictive-ability framework of \citet{giacominiwhite2006} holds for any estimator, parametric or not, under the fixed estimation scheme of every headline comparison; the annual-refit runs use expanding windows, which that framework does not cover, and are reported as robustness checks.""")

README_ANCHOR = "| `code/paper/job_coherent.py` | `results/paper/coherent_results.json` | Monotonized curve audit; ES as the converged integral of the same curve |"
README_AFTER = README_ANCHOR + """

Several jobs rebuild the accuracy layer themselves rather than reading the canonical forecast of `bench_all_results.json`, among them `job_coherent.py`, `job_fz_strict_calibration.py`, `job_fz_fullpanel.py` and `job_pzc_taylor.py`. Each refits the pooled GPD and the body on its own row set, so their engine FZ0 can differ from the canonical 2.14653 / 1.84863 in the fourth decimal (`coherent_results.json`: 1.84806 at 2.5%). Every FZ0 comparison against a benchmark in the paper uses `bench_all_results.json`. The splice-level, closed-form-ES and strict-split statements are comparisons within a single job's own build, and are quoted from that job."""


def main():
    dry = "--dry" in sys.argv
    T, problems = {}, []
    for f in FILES:
        T[f] = io.open(f, encoding="utf-8").read()
        for n, b, _ in EDITS:
            c = T[f].count(b)
            if c != 1: problems.append(f"{f}: {n}: anchor found {c} times")
    T["README.md"] = io.open("README.md", encoding="utf-8").read()
    c = T["README.md"].count(README_ANCHOR)
    if c != 1: problems.append(f"README.md: R1: anchor found {c} times")
    if problems:
        print("NOT APPLIED:\n  " + "\n  ".join(problems)); sys.exit(1)
    if dry:
        print("dry run OK: %d edits x %d files + README" % (len(EDITS), len(FILES))); return
    for f in FILES:
        for _, b, a in EDITS: T[f] = T[f].replace(b, a, 1)
    T["README.md"] = T["README.md"].replace(README_ANCHOR, README_AFTER, 1)
    for f, t in T.items(): io.open(f, "w", encoding="utf-8", newline="\n").write(t)
    print("R60 applied: %d edits x %d files + README" % (len(EDITS), len(FILES)))

if __name__ == "__main__":
    main()
