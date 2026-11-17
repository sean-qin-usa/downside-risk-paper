# r63_mk63_anatomy.py -- READ-ONLY DIAGNOSTIC FOR R63. What does a top-decile mk63 window look like?
# No estimator, no forecast, nothing the paper's tables read. Panel, filter and split are the canonical ones
# (tools/r61_diag.py / code/paper/job_garch_evt.py): GARCH(1,1)-t per name on the first 60% of each name's
# history, standardized residuals z over the full history, mk63 = pandas rolling(63, min_periods=30).kurt()
# shifted one day (bias-corrected sample excess kurtosis, as eq. (6) of the paper), test rows idx >= 0.6 n.
#
# THE QUESTION. A referee reads the top mk63 decile (mean excess kurtosis ~25) as "one large residual in the
# window", i.e. a post-shock indicator rather than a shape measure. Arithmetic says one residual of about
# 10 sigma among 62 unit-variance draws gives sample kurtosis about 24.5. This script measures it directly.
#
# Per test row, over the 63-day window that defines its mk63:
#   share1   share of the window's centered fourth-moment sum sum((z-zbar)^4) contributed by its single
#            largest term
#   zmax     the largest |z - zbar| / s_z in the window (s_z the window SD, as in eq. (6))
#   mk_drop1 mk63 recomputed with that one residual removed (62 observations, same estimator)
#   age      trading days since that largest residual (1 = yesterday)
#   zraw     that residual's own size |z| in units of the GARCH conditional SD of its day (not window-normalized;
#            added after the first run to put the shock in the filter's units, no prediction attached)
#   sd_rest  SD of the other 62 residuals of the window, in GARCH units (added after the second run to explain
#            why the decile's kurtosis exceeds what one shock among unit-variance draws gives; no prediction)
# Reported by pooled test-row mk63 decile: medians and quartiles, and the share of rows with share1 > 0.5.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-08):
#  P1. Top decile: median share1 >= 0.60 (one residual carries most of the fourth moment).
#  P2. Top decile: median zmax >= 6.5.
#  P3. Top decile: median mk_drop1 < 3, i.e. removing one residual moves a typical top-decile window into the
#      range of deciles 1-8.
#  P4. Deciles 1-8: median share1 < 0.30.
#  P5. Top decile: the median age of the largest residual is spread over the window (between 15 and 45 days),
#      so the signal is "a large shock in the last three months", not "a shock yesterday".
# If P1 or P3 fails, the paper does not say "one residual"; it says what the table shows.
# Usage: GBC_PROJ=<data dir> python tools/r63_mk63_anatomy.py [--panel canon200|holdout]
# Output: results/paper/diagnostics/r63_mk63_anatomy_<panel>.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from arch import arch_model

PANEL = sys.argv[sys.argv.index("--panel") + 1] if "--panel" in sys.argv else "canon200"
assert PANEL in ("canon200", "holdout")
P = os.environ.get("GBC_PROJ", os.environ.get("GBC_PROJECT_DIR", "."))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
t0 = time.time(); lg = lambda s: print(s, flush=True)

rr = pd.read_csv(os.path.join(P, "holdout_panel_2000_2013.csv" if PANEL == "holdout" else "crsp_panel_returns.csv"),
                 dtype={'permno': 'int32'})
rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); names = cnt[cnt >= 1500].index.tolist()[:200]

W = 63
def g2_bias_corrected(x):
    """pandas' rolling kurt: bias-corrected sample excess kurtosis, rows of x are windows."""
    n = x.shape[1]; m = x.mean(1, keepdims=True); d = x - m
    m2 = (d ** 2).mean(1); m4 = (d ** 4).mean(1)
    g2 = m4 / m2 ** 2 - 3.0
    return ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))

recs = []
for pn in names:
    g = rr[rr.permno == pn].sort_values('date'); y = g['ret'].values.astype(float); n = len(y)
    if n < 1500 or not np.isfinite(y).all():
        y = y[np.isfinite(y)]; n = len(y)
        if n < 1500: continue
    sp = int(n * 0.6)
    try:
        r1 = arch_model(y[:sp], vol='Garch', p=1, q=1, dist='t', rescale=False).fit(disp='off', show_warning=False)
        pp = r1.params; om, al, be, mu = float(pp['omega']), float(pp['alpha[1]']), float(pp['beta[1]']), float(pp.get('mu', 0))
    except Exception:
        continue
    e0 = y - mu; s2 = np.empty(n); s2[0] = np.var(y[:sp])
    for k in range(1, n): s2[k] = max(om + al * e0[k - 1] ** 2 + be * s2[k - 1], 1e-8)
    z = (y - mu) / np.maximum(np.sqrt(s2), 1e-6)
    # window ending at t-1 for target t: windows[j] = z[j:j+W] is the window for target t = j+W
    win = sliding_window_view(z, W)                      # shape (n-W+1, W)
    tgt = np.arange(W, n + 1)                            # target index of each window (last one is n, dropped)
    keep = (tgt >= sp) & (tgt < n)
    win = win[keep]; tgt = tgt[keep]
    m = win.mean(1, keepdims=True); s = win.std(1, ddof=1, keepdims=True); d = win - m
    d4 = d ** 4; j = np.argmax(np.abs(d), 1); r = np.arange(len(win))
    share1 = d4[r, j] / d4.sum(1)
    zmax = np.abs(d[r, j]) / s[:, 0]
    mk = g2_bias_corrected(win)
    mask = np.ones_like(win, bool); mask[r, j] = False
    mk_drop1 = g2_bias_corrected(win[mask].reshape(len(win), W - 1))
    age = W - j                                          # j = W-1 is yesterday -> age 1
    zraw = np.abs(win[r, j])
    sd_rest = win[mask].reshape(len(win), W - 1).std(1, ddof=1)
    recs.append(pd.DataFrame({'permno': pn, 'mk63': mk, 'share1': share1, 'zmax': zmax,
                              'mk_drop1': mk_drop1, 'age': age, 'zraw': zraw, 'sd_rest': sd_rest}))
D = pd.concat(recs, ignore_index=True)
lg("panel %s: %d names, %d test rows, %.0fs" % (PANEL, D.permno.nunique(), len(D), time.time() - t0))

D['dec'] = pd.qcut(D['mk63'].rank(method='first'), 10, labels=False) + 1
q = lambda x, p: round(float(np.quantile(x, p)), 3)
out = {'panel': PANEL, 'n_names': int(D.permno.nunique()), 'n_rows': int(len(D)),
       'note': "Read-only. Per test row, over the 63-day window defining its mk63: share1 = share of the "
               "window's centered fourth-moment sum from its single largest term; zmax = largest |z-zbar|/s_z; "
               "mk_drop1 = mk63 with that residual removed; age = trading days since it (1 = yesterday). "
               "Deciles are pooled test-row ranks of mk63.",
       'predictions_written_before_run': "P1 top-decile median share1 >= 0.60; P2 top-decile median zmax >= 6.5; "
               "P3 top-decile median mk_drop1 < 3; P4 deciles 1-8 median share1 < 0.30; "
               "P5 top-decile median age between 15 and 45.",
       'by_decile': {}}
for k, g in D.groupby('dec'):
    out['by_decile'][str(int(k))] = {
        'n': int(len(g)), 'mk63_mean': round(float(g.mk63.mean()), 2), 'mk63_median': q(g.mk63, .5),
        'share1_q25_med_q75': [q(g.share1, .25), q(g.share1, .5), q(g.share1, .75)],
        'frac_share1_gt_half': round(float((g.share1 > 0.5).mean()), 3),
        'zmax_q25_med_q75': [q(g.zmax, .25), q(g.zmax, .5), q(g.zmax, .75)],
        'mk_drop1_q25_med_q75': [q(g.mk_drop1, .25), q(g.mk_drop1, .5), q(g.mk_drop1, .75)],
        'age_q25_med_q75': [q(g.age, .25), q(g.age, .5), q(g.age, .75)],
        'zraw_q25_med_q75': [q(g.zraw, .25), q(g.zraw, .5), q(g.zraw, .75)],
        'sd_rest_q25_med_q75': [q(g.sd_rest, .25), q(g.sd_rest, .5), q(g.sd_rest, .75)]}
b = D[D.dec <= 8]; t = D[D.dec == 10]
out['bulk_d1to8'] = {'share1_median': q(b.share1, .5), 'mk63_median': q(b.mk63, .5), 'mk63_q90': q(b.mk63, .9)}
out['top_d10'] = {'share1_median': q(t.share1, .5), 'zmax_median': q(t.zmax, .5), 'mk_drop1_median': q(t.mk_drop1, .5),
                  'age_median': q(t.age, .5), 'zraw_median': q(t.zraw, .5), 'frac_share1_gt_half': round(float((t.share1 > 0.5).mean()), 4), 'frac_mk_drop1_below_bulk_q90': round(float((t.mk_drop1 < out['bulk_d1to8']['mk63_q90']).mean()), 3)}
out['grades'] = {'P1': out['top_d10']['share1_median'] >= 0.60, 'P2': out['top_d10']['zmax_median'] >= 6.5,
                 'P3': out['top_d10']['mk_drop1_median'] < 3, 'P4': out['bulk_d1to8']['share1_median'] < 0.30,
                 'P5': 15 <= out['top_d10']['age_median'] <= 45}
od = os.path.join(ROOT, "results", "paper", "diagnostics"); os.makedirs(od, exist_ok=True)
fn = os.path.join(od, "r63_mk63_anatomy_%s.json" % PANEL); json.dump(out, open(fn, "w"), indent=2)
lg(json.dumps({'top_d10': out['top_d10'], 'bulk_d1to8': out['bulk_d1to8'], 'grades': out['grades']}, indent=1))
lg("wrote %s  %.0fs" % (fn, time.time() - t0))
