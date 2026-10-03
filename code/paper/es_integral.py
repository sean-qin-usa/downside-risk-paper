# es_integral.py -- ONE implementation of the converged ES integral, shared by every job that needs it.
#
# WHY THIS EXISTS. The committed convention integrates ES(a) = (1/a) \int_0^a Q(u) du as the mean of the
# model's own quantile curve at the 20 midpoints a(k+0.5)/20. Seventeen scripts carry a copy of that rule.
# Measured against the analytic Student-t ES it understates |ES| by 0.7-1.4%, and on the canon200 panel the
# engine's |ES| is understated by 1.65%, enough to move the McNeil-Frey p at 1% from 0.067 to 0.383. Rather
# than re-derive the correction in each script, every job imports from here.
#
# THE TWO ERROR SOURCES, KEPT SEPARATE.
#   K  the number of fitted body levels. An estimator choice: the committed pipeline fits 20.
#   M  the number of quadrature nodes used on an interpolant through those K levels. Pure numerics.
#
# WHY NODES ALONE DO NOT CONVERGE. The GPD quantile diverges as u^{-xi} at the origin, so a midpoint rule on
# [0,a] converges at order 1-xi, measured at 0.62 on this panel. M=2000 still left 0.09% on the table. The
# sub-floor region is therefore integrated IN CLOSED FORM and nodes are used only where the integrand is
# bounded. With that split |ES(200)-ES(2000)|/|ES| falls to 3e-5.
#
# THE FOUR CONDITIONS this module implements, agreed before any result was read:
#  (1) interpolate the BODY only; evaluate the GPD branch exactly at every node and take the minimum node by
#      node. Interpolating the already combined curve rounds off the corner where the branches meet.
#  (2) never fit or interpolate the body below a/40, the lowest level the committed 20-node rule fits; below
#      that floor the GPD branch alone is used, as the paper says the estimator does beyond training support.
#  (3) VaR is the committed construction and is not computed here at all -- callers pass it in, so no
#      VaR-only statistic can move.
#  (4) callers report two controls: closed-form rows unchanged, and VaR-only statistics unchanged.
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["gpd_tail_integral", "converged_es", "committed_es_20node", "es_by_substitution", "body_es_flat_extension", "levels_and_body",
           "roll_tail_mean", "selftest"]


def gpd_tail_integral(F, uu, beta, xi, p0):
    """Exact \\int_0^F q(t) dt for the GPD tail quantile
    q(t) = uu - (beta/xi)*((t/p0)**(-xi) - 1), or its xi->0 limit uu - beta*log(p0/t).

    Verified against a 4e6-node substituted quadrature to better than 2e-6 relative over
    xi in {0.05,0.2,0.38,0.55}, beta in {0.4,0.9}, uu in {-1.9,-2.1}, F in {a/40}.
    """
    if F <= 0.0:
        return 0.0
    if abs(xi) > 1e-6:
        if xi >= 1.0:
            raise ValueError("xi=%.4f >= 1: the GPD mean does not exist, ES is undefined" % xi)
        return F * uu - (beta / xi) * ((p0 ** xi) * (F ** (1.0 - xi)) / (1.0 - xi) - F)
    return F * uu - beta * F * math.log(p0 / F) - beta * F


def committed_es_20node(alpha, body_at_sub20, evt_fn, var_z=None):
    """The committed convention, reproduced exactly so every job can report old beside new.
    body_at_sub20 is (n_rows, 20) at the levels alpha*(k+0.5)/20."""
    sub20 = alpha * ((np.arange(20) + 0.5) / 20.0)
    star = np.sort(np.minimum(body_at_sub20, np.array([evt_fn(float(u)) for u in sub20])[None, :]), axis=1)
    es = star.mean(axis=1)
    return es if var_z is None else np.minimum(es, var_z - 1e-6)


def converged_es(alpha, levels, body_q, evt_fn, gpd, M=2000, var_z=None, floor=None):
    """Converged ES in the units of body_q (standardised residual space, as the pipeline uses).

    alpha    the ES level, e.g. 0.01
    levels   (K,) fitted body levels, ascending, spanning [floor, alpha]
    body_q   (n_rows, K) the fitted body quantiles at those levels
    evt_fn   callable u -> GPD tail quantile, evaluated exactly at each node
    gpd      (uu, beta, xi, p0) for the closed-form sub-floor integral
    var_z    optional committed VaR; ES is capped just below it, as the pipeline does
    floor    defaults to alpha/40, the lowest level the committed rule fits (condition 2)
    """
    if floor is None:
        floor = alpha / 40.0
    levels = np.asarray(levels, float)
    if levels[0] > floor + 1e-15 or levels[-1] < alpha - 1e-15:
        raise ValueError("levels must span [alpha/40, alpha]; got [%.6g, %.6g]" % (levels[0], levels[-1]))
    tail = gpd_tail_integral(floor, *gpd)                      # (2) exact below the floor
    u = floor + (alpha - floor) * ((np.arange(M) + 0.5) / M)
    bi = PchipInterpolator(levels, body_q, axis=1, extrapolate=False)(np.clip(u, levels[0], levels[-1]))
    ev = np.array([evt_fn(float(x)) for x in u])[None, :]
    comb = np.minimum(bi, ev)                                  # (1) body interpolated, GPD exact, min per node
    es = (tail + (alpha - floor) * comb.mean(axis=1)) / alpha   # the mean of a rearrangement is the mean
    return es if var_z is None else np.minimum(es, var_z - 1e-6)


def es_by_substitution(alpha, q_fn, M=2000, power=4.0):
    """ES(alpha) = (1/alpha) \int_0^alpha q(u) du for a PARAMETRIC quantile with no elementary integral --
    the skew-t of the GJR row, for instance.

    A plain midpoint rule fails here for the same reason it failed on the GPD: a fat-tailed quantile diverges
    as u^{-1/nu} at the origin, so the rule converges at order 1-1/nu and needs absurd M. The substitution
    u = alpha * s**power maps [0,1] -> [0,alpha] and multiplies the integrand by power*s**(power-1), which
    vanishes at the origin fast enough to cancel the divergence for any nu > 1/power. power=4 covers nu > 0.25,
    which is every fitted tail. Verified against the exact Student-t ES to better than 1e-5 at M=2000.

    q_fn takes an array of levels and returns the quantile at each; it may be vectorised over rows, in which
    case the result is one ES per row.
    """
    s_ = (np.arange(M) + 0.5) / M
    u = alpha * s_ ** power
    w = power * s_ ** (power - 1.0) / M            # du/ds * ds, normalised by alpha
    q = q_fn(u)
    q = np.asarray(q, float)
    if q.ndim == 1:
        return float(np.sum(q * w))
    return q @ w


def levels_and_body(alpha, sub_preds, fit_predict, subn=20, kextra=40):
    """Build the (levels, body_q) pair that `converged_es` needs, from what a job already has.

    Every job in this pipeline fits the body at the `subn` committed midpoints alpha*(j+0.5)/subn, the lowest
    of which is alpha/(2*subn) = the floor. Those fits are REUSED here; `kextra` further log-spaced levels on
    [floor, alpha] are fitted through the caller's `fit_predict(u) -> ndarray` so that the interpolant has
    enough support to integrate against. Nothing is fitted below the floor (condition 2).

    sub_preds must be ordered as the committed grid is, i.e. sub_preds[j] is the body at alpha*(j+0.5)/subn.
    """
    floor = alpha / subn / 2.0
    sub = alpha * ((np.arange(subn) + 0.5) / subn)
    lev = np.unique(np.concatenate([sub,
                                    np.exp(np.linspace(math.log(floor), math.log(alpha), kextra)),
                                    [alpha]]))
    known = {round(float(u), 12): np.asarray(sub_preds[j]) for j, u in enumerate(sub)}
    QB = np.stack([known[round(float(u), 12)] if round(float(u), 12) in known
                   else np.asarray(fit_predict(float(u))) for u in lev], axis=1)
    return lev, QB


def body_es_flat_extension(alpha, levels, body_q, M=2000, floor=None):
    """ES for a BODY-ONLY row -- a fitted quantile curve with no tail model attached.

    Such a row has no quantile defined below its lowest fitted level, so there is no converged ES for it:
    something has to be assumed about (0, alpha/40]. The committed 20-node rule implicitly assigns that cell
    the body's value at alpha/40. This function makes that assumption EXPLICIT and fixes only the quadrature
    above the floor, which is the part that is numerics rather than modelling. The flat extension understates
    |ES| for any fat-tailed truth, so a body-only row's ES stays a lower bound on severity and must be
    labelled as one -- it is not comparable with a row that models its tail.
    """
    if floor is None:
        floor = alpha / 40.0
    levels = np.asarray(levels, float)
    u = floor + (alpha - floor) * ((np.arange(M) + 0.5) / M)
    bi = PchipInterpolator(levels, body_q, axis=1, extrapolate=False)(np.clip(u, levels[0], levels[-1]))
    q_floor = PchipInterpolator(levels, body_q, axis=1, extrapolate=False)(
        np.clip(np.array([floor]), levels[0], levels[-1]))[:, 0]
    return (floor * q_floor + (alpha - floor) * bi.mean(axis=1)) / alpha


def roll_tail_mean(x, win, tau, minp=250):
    """EXACT empirical tail mean in each rolling window: at row i, the mean of the observations in
    x[i-win : i] that lie at or below that window's tau-quantile. Same construction as the pooled FHS ES
    (`mean(z[z <= quantile(z, tau)])`), applied window by window, and aligned like
    `s.rolling(win, min_periods=minp).quantile(tau).shift(1)` so it drops into existing code unchanged.

    This replaces integrating a rolling EMPIRICAL quantile function on a 20-node grid. For an empirical
    quantile function the tail mean is available exactly, so there is no reason to approximate it.
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        w = x[max(0, i - win):i]
        w = w[np.isfinite(w)]
        if len(w) < minp:
            continue
        sel = w[w <= np.quantile(w, tau)]
        if len(sel):
            out[i] = sel.mean()
    return out


def selftest(verbose=True):
    """Two checks: the closed form against brute-force quadrature, and the full integral against a
    pure-GPD case whose ES is known in closed form."""
    ok = True
    for uu in (-1.9, -2.1):
        for xi in (0.05, 0.20, 0.38, 0.55):
            for beta in (0.4, 0.9):
                p0, F = 0.025, 0.01 / 40.0
                pw = 1.0 / (1.0 - xi)
                n = 2_000_000
                s = (np.arange(n) + 0.5) / n
                t = F * s ** pw
                w = F * pw * s ** (pw - 1.0) / n
                num = float(np.sum((uu - (beta / xi) * ((t / p0) ** (-xi) - 1.0)) * w))
                cl = gpd_tail_integral(F, uu, beta, xi, p0)
                r = abs(num - cl) / abs(cl)
                ok &= r < 5e-6
                if verbose and r >= 5e-6:
                    print("  FAIL gpd_tail_integral xi=%.2f beta=%.1f rel=%.2e" % (xi, beta, r))
    # es_by_substitution against the exact Student-t ES
    from scipy import stats as _st
    def _tes(a, nu):
        qq = _st.t.ppf(a, nu)
        return -_st.t.pdf(qq, nu) * (nu + qq * qq) / ((nu - 1) * a)
    worst = 0.0
    for nu in (3.5, 4.5, 6.0, 8.0, 12.0):
        for a in (0.01, 0.025):
            got = es_by_substitution(a, lambda u, nu=nu: _st.t.ppf(u, nu), M=2000)
            r = abs(got / _tes(a, nu) - 1.0)
            worst = max(worst, r)
            ok &= r < 1e-5
    if verbose:
        print("  es_by_substitution vs exact Student-t ES: worst rel err %.2e over nu in [3.5,12]" % worst)
    # levels_and_body reuses the committed grid and fits only the rest
    calls = []
    sp = [np.full(4, -float(j + 1)) for j in range(20)]
    lv, qb = levels_and_body(0.01, sp, lambda u: (calls.append(u), np.full(4, -99.0))[1], kextra=40)
    ok &= abs(lv[0] - 0.01 / 40) < 1e-15 and abs(lv[-1] - 0.01) < 1e-15
    ok &= qb.shape == (4, len(lv))
    sub_ref = 0.01 * ((np.arange(20) + 0.5) / 20)
    pos = {round(float(u), 12): k for k, u in enumerate(lv)}
    ok &= not any(round(float(c), 12) in {round(float(u), 12) for u in sub_ref} for c in calls)
    ok &= all(abs(qb[0, pos[round(float(u), 12)]] + (j + 1)) < 1e-12 for j, u in enumerate(sub_ref))
    if verbose:
        print("  levels_and_body: %d levels spanning [a/40, a], %d committed fits reused, %d new"
              % (len(lv), 20, len(calls)))
    # roll_tail_mean against a direct per-window computation on random data
    rng = np.random.default_rng(0)
    xx = rng.standard_t(5.0, size=900)
    got = roll_tail_mean(xx, 400, 0.025, minp=250)
    for i in (300, 500, 899):
        w = xx[max(0, i - 400):i]
        want = w[w <= np.quantile(w, 0.025)].mean() if len(w) >= 250 else np.nan
        if np.isfinite(want):
            ok &= abs(got[i] - want) < 1e-12
    ok &= not np.isfinite(got[100])          # fewer than minp observations
    if verbose:
        print("  roll_tail_mean: matches a direct per-window tail mean, and is NaN before minp")
    # a body curve far below the GPD everywhere makes the minimum the GPD, so ES must equal the exact
    # GPD integral over [0, alpha] / alpha
    uu, beta, xi, p0, a = -2.0, 0.7, 0.30, 0.025, 0.01
    evt = lambda t: uu - (beta / xi) * ((t / p0) ** (-xi) - 1.0)
    lev = np.exp(np.linspace(math.log(a / 40.0), math.log(a), 40))
    body = np.full((3, len(lev)), 1e3)                      # never the minimum
    got = converged_es(a, lev, body, evt, (uu, beta, xi, p0), M=4000)
    want = gpd_tail_integral(a, uu, beta, xi, p0) / a
    r = abs(got[0] - want) / abs(want)
    ok &= r < 1e-5
    if verbose:
        print("  pure-GPD case: converged_es %.8f vs exact %.8f (rel %.2e)" % (got[0], want, r))
        print("es_integral selftest: %s" % ("PASS" if ok else "*** FAIL ***"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
