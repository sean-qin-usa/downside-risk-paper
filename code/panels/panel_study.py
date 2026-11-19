# panel_study.py -- the six further panels of docs/PANELS_PRECOMMIT.md. Protocol, measures, hypotheses and predictions
# are fixed there; nothing here may change them. One panel per call:
#   python code/panels/panel_study.py <panel>          (panel names: code/panels/panels.json)
# Each panel trains its own pooled models under a strict calendar split; every model is fitted once before test0.
# Output: results/panels/<panel>_results.json (committed) and data/panels/<panel>_rows.npz (per-row forecasts, not committed).
# PS_SCREEN=1 (a deviation recorded in docs/PANELS_PRECOMMIT.md, off by default): a data-quality screen on the price file
# alone. A name is dropped if it has a daily return above +100% or below -70% (unadjusted splits and spin-offs), or ten
# or more consecutive unchanged closes (a stale feed). Output files then carry the suffix "_screened".
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "code", "case_studies")); sys.path.insert(0, os.path.join(ROOT, "code", "paper"))
os.environ.setdefault("GBC_PROJ", ".")
import crossasset as X
C = X.C; converged_es = X.converged_es; A, TAUS, ZX, P0, SUBN, HGB, LEVS, lev_for = X.A, X.TAUS, X.ZX, X.P0, X.SUBN, X.HGB, X.LEVS, X.lev_for
PANEL = sys.argv[1]; CFG = json.load(open(os.path.join(HERE, "panels.json")))[PANEL]; END = "2026-11-13"
CAL0, TEST0 = pd.Timestamp(CFG["cal0"]), pd.Timestamp(CFG["test0"]); DDIR = os.path.join(ROOT, "data", "panels", PANEL)
NW = int(os.environ.get("PS_WORKERS", "15")); t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
JA = {a: TAUS.index(a) for a in A}; SCREEN = os.environ.get("PS_SCREEN", "0") == "1"; TAG = PANEL + ("_screened" if SCREEN else "")

def nfeat(y):       # returns-space inputs of the neural network (tools/r70_neural_quantile.py)
    s = pd.Series(y); f = {}
    for k in range(1, 6): f["lag%d" % k] = s.shift(k)
    for w in (5, 21, 63):
        f["mabs%d" % w] = s.abs().rolling(w, min_periods=max(3, w // 3)).mean().shift(1); f["sd%d" % w] = s.rolling(w, min_periods=max(3, w // 3)).std().shift(1)
    f["min21"] = s.rolling(21, min_periods=8).min().shift(1); f["max21"] = s.rolling(21, min_periods=8).max().shift(1); f["dn5"] = (s < 0).rolling(5, min_periods=3).mean().shift(1)
    return pd.DataFrame(f).values

def one_name(tk):
    try:
        s = pd.read_csv(os.path.join(DDIR, tk.replace("=", "_") + ".csv"), index_col=0, parse_dates=True)["Close"]
    except Exception: return tk, "no data"
    s = s[(s > 0) & (s.index <= END)]; nbad = 0
    if tk.endswith("=X"):
        r_ = s.pct_change(); bad = ((r_.abs() > 0.03) & (r_.shift(-1).abs() > 0.03) & (np.sign(r_) != np.sign(r_.shift(-1)))).fillna(False).values; nbad = int(bad.sum()); s = s[~bad]
    r = (s.pct_change() * 100).dropna(); r = r[np.isfinite(r)]; dates = r.index; y = r.values.astype(float); n = len(y)
    if SCREEN and n:
        zr = (y == 0); run = mx = 0
        for v in zr: run = run + 1 if v else 0; mx = max(mx, run)
        if y.max() > 100 or y.min() < -70: return tk, "screened: daily return of %+.0f%%" % (y.max() if y.max() > 100 else y.min())
        if mx >= 10: return tk, "screened: %d consecutive unchanged closes" % mx
    tr = dates < CAL0; te = dates >= TEST0; est = dates < TEST0; ne = int(est.sum())
    if tr.sum() < CFG["min_train"] or te.sum() < 250: return tk, "too short (%d train, %d test)" % (tr.sum(), te.sum())
    try:
        o = {"dates": np.array([str(d.date()) for d in dates[te]]), "y": y[te], "nbad": nbad}; yest = y[:ne]
        Q = {}; VE = {}
        # GARCH-t and GJR-skew-t
        for m, oo, dist in (("garch_t", 0, "t"), ("gjr_skewt", 1, "skewt")):
            pr = arch_model(yest, vol="GARCH", p=1, o=oo, q=1, dist=dist, rescale=False).fit(disp="off", show_warning=False).params
            sig = arch_model(y, vol="GARCH", p=1, o=oo, q=1, dist=dist, rescale=False).fix(pr.values).conditional_volatility; mu = float(pr["mu"])
            if dist == "t":
                nu = float(pr["nu"]); zt = stats.t.ppf(TAUS, nu) / math.sqrt(nu / (nu - 2)); ze = {a: C.t_std_q_es(a, nu)[1] for a in A}
                df = C.feats(y, sig, mu); zg = df["z"].values; o["sig"] = sig[te]; o["mu"] = mu; o["nu"] = nu; o["mk63"] = df["mk63"].values[te]
                fg = df[ZX].values; ok = np.isfinite(fg).all(1); o["Xg"] = fg[te]; o["trn_g"] = np.column_stack([fg[tr & ok], zg[tr & ok]])
                ztr = zg[:ne]; ztr = ztr[np.isfinite(ztr)]; Q["fhs"] = mu + sig[te][:, None] * np.quantile(ztr, TAUS)[None, :]
                VE["fhs"] = {a: (mu + sig[te] * float(np.quantile(ztr, a)), mu + sig[te] * float(ztr[ztr <= np.quantile(ztr, a)].mean())) for a in A}
                win = sliding_window_view(zg, 63); j = np.argmax(np.abs(win - win.mean(1, keepdims=True)), 1); rr_ = np.arange(len(win))
                sg = np.full(n, np.nan); sz = np.full(n, np.nan); sg[63:] = np.sign(win[rr_, j])[:-1]; sz[63:] = np.abs(win[rr_, j])[:-1]; o["shock_sign"] = sg[te]; o["shock_size"] = sz[te]
            else:
                ppf = lambda u, pr=pr: C.SK.ppf(np.clip(u, 1e-300, 1.0), np.array([float(pr["eta"]), float(pr["lambda"])])); zt = ppf(np.array(TAUS)); ze = {a: C.es_from_ppf(ppf, a) for a in A}
            Q[m] = mu + sig[te][:, None] * zt[None, :]; VE[m] = {a: (Q[m][:, JA[a]], mu + sig[te] * ze[a]) for a in A}
        # Beta-t-EGARCH
        bx = C.fit_bteg(yest); mu, om, phi, kap, nu = C.bteg_unpack(bx); lam = C._bteg_filter(y, mu, om, phi, kap, nu); sc = math.sqrt(nu / (nu - 2)); sd = np.exp(lam) * sc
        dfb = C.feats(y, sd, mu); fb = dfb[ZX].values; ok = np.isfinite(fb).all(1); o["Xb"] = fb[te]; o["trn_b"] = np.column_stack([fb[tr & ok], dfb["z"].values[tr & ok]])
        o["bsig"] = sd[te]; o["bmu"] = mu; Q["param_bteg"] = mu + sd[te][:, None] * (stats.t.ppf(TAUS, nu) / sc)[None, :]
        VE["param_bteg"] = {a: (Q["param_bteg"][:, JA[a]], mu + sd[te] * C.t_std_q_es(a, nu)[1]) for a in A}
        # SAV-CAViaR, one fit per level
        Q["sav_caviar"] = np.sort(np.column_stack([X.caviar_sav(yest, y, t)[te] for t in TAUS]), axis=1)
        # returns-space inputs for the neural network
        scn = float(np.std(y[tr])); F = nfeat(y / scn); okn = np.isfinite(F).all(1); ca = (dates >= CAL0) & (dates < TEST0)
        o["n_tr"] = (F[tr & okn], y[tr & okn] / scn); o["n_va"] = (F[ca & okn], y[ca & okn] / scn); o["n_te"] = np.nan_to_num(F[te]); o["n_sc"] = scn
        o["Q"] = Q; o["VE"] = VE
        return tk, o
    except Exception as e:
        return tk, "fit failed: %s" % str(e)[:80]

POOLS = {}
def fit_job(args):
    f, t = args; D = POOLS[f]
    return f, t, HistGradientBoostingRegressor(loss='quantile', quantile=t, random_state=0, **HGB).fit(D[:, :-1], D[:, -1])

def neural(tr, va, te):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tt = torch.tensor(TAUS, dtype=torch.float32, device=dev); T_ = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
    mu_x, sd_x = tr[0].mean(0), tr[0].std(0) + 1e-8; Xtr, Ytr, Xva, Yva, Xte = T_((tr[0] - mu_x) / sd_x), T_(tr[1]), T_((va[0] - mu_x) / sd_x), T_(va[1]), T_((te - mu_x) / sd_x)
    pb = lambda q, y: torch.maximum(tt * (y[:, None] - q), (tt - 1) * (y[:, None] - q)).mean(); preds = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed); net = nn.Sequential(nn.Linear(tr[0].shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(TAUS))).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); best = (1e9, None); bad = 0
        for ep in range(60):
            net.train(); perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), 4096):
                b = perm[i:i + 4096]; opt.zero_grad(); loss = pb(torch.sort(net(Xtr[b]), dim=1).values, Ytr[b]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): v = float(pb(torch.sort(net(Xva), dim=1).values, Yva))
            if v < best[0] - 1e-6: best = (v, {k: w.clone() for k, w in net.state_dict().items()}); bad = 0
            else:
                bad += 1
                if bad >= 5: break
        net.load_state_dict(best[1]); net.eval()
        with torch.no_grad(): preds.append(torch.cat([torch.sort(net(Xte[i:i + 200000]), dim=1).values for i in range(0, len(Xte), 200000)]).cpu().numpy())
    return np.mean(preds, axis=0)

if __name__ == "__main__":
    with Pool(NW) as pool: res = pool.map(one_name, CFG["tickers"], chunksize=1)
    R = {tk: o for tk, o in res if isinstance(o, dict)}; dropped = {tk: o for tk, o in res if not isinstance(o, dict)}; names = list(R)
    lg("%s: %d listed, %d in the panel, %d dropped" % (PANEL, len(CFG["tickers"]), len(names), len(dropped)))
    for f, k in (("garch_t", "trn_g"), ("bteg", "trn_b")): POOLS[f] = np.concatenate([R[t][k] for t in names])
    with Pool(NW) as pool: fitted = pool.map(fit_job, [(f, t) for f in POOLS for t in LEVS], chunksize=4)
    MODELS = {(f, round(t, 12)): m for f, t, m in fitted}; TAIL = {f: C.GPDTail(POOLS[f][:, -1], P0) for f in POOLS}
    lg("pooled trees fitted on %d training rows; GPD xi %.3f / %.3f" % (len(POOLS["garch_t"]), TAIL["garch_t"].xi, TAIL["bteg"].xi))
    cat = lambda k: np.concatenate([np.asarray(R[t][k]) for t in names]); rep = lambda k: np.concatenate([np.full(len(R[t]["y"]), R[t][k]) for t in names])
    Y = cat("y"); dates = pd.to_datetime(cat("dates")); NAME = np.concatenate([np.full(len(R[t]["y"]), i) for i, t in enumerate(names)]); mk = cat("mk63"); sgn = cat("shock_sign"); ssz = cat("shock_size")
    Q = {m: np.concatenate([R[t]["Q"][m] for t in names]) for m in ("garch_t", "gjr_skewt", "fhs", "param_bteg", "sav_caviar")}
    VE = {m: {a: (np.concatenate([R[t]["VE"][m][a][0] for t in names]), np.concatenate([R[t]["VE"][m][a][1] for t in names])) for a in A} for m in ("garch_t", "gjr_skewt", "fhs", "param_bteg")}
    for f, lab, kx, ks, km in (("garch_t", "", "Xg", "sig", "mu"), ("bteg", "_bteg", "Xb", "bsig", "bmu")):
        Xf = cat(kx); ok = np.isfinite(Xf).all(1); S = cat(ks); M_ = rep(km); tail = TAIL[f]; Qc = {}
        for t in LEVS:
            q = np.full(len(Xf), np.nan); q[ok] = MODELS[(f, round(t, 12))].predict(Xf[ok]); Qc[round(t, 12)] = q
        Qz = lambda t: Qc[round(float(t), 12)]
        eng = {t: (np.minimum(Qz(t), tail.q(t)) if t <= P0 else Qz(t)) for t in TAUS}
        Es = np.sort(np.stack([eng[t] for t in TAUS], 1), 1); bod = np.sort(np.stack([Qz(t) for t in TAUS], 1), 1)
        Q["body" + lab] = M_[:, None] + S[:, None] * bod; Q["engine" + lab] = M_[:, None] + S[:, None] * Es; VE["body" + lab] = {}; VE["engine" + lab] = {}
        for a in A:
            lev = lev_for(a); QB = np.stack([Qz(u) for u in lev], 1); zq = Es[:, JA[a]]; es = np.full(len(zq), np.nan)
            es[ok] = converged_es(a, lev, QB[ok], tail.q, (-tail.u, float(tail.beta), float(tail.xi), P0), M=2000, var_z=zq[ok])
            zqb = bod[:, JA[a]]; esb = np.minimum(np.stack([Qz(a * (j + 0.5) / SUBN) for j in range(SUBN)], 1).mean(1), zqb - 1e-6)
            VE["engine" + lab][a] = (M_ + S * zq, M_ + S * es); VE["body" + lab][a] = (M_ + S * zqb, M_ + S * esb)
    ntr = (np.concatenate([R[t]["n_tr"][0] for t in names]), np.concatenate([R[t]["n_tr"][1] for t in names])); nva = (np.concatenate([R[t]["n_va"][0] for t in names]), np.concatenate([R[t]["n_va"][1] for t in names]))
    Q["neural"] = neural(ntr, nva, cat("n_te")) * rep("n_sc")[:, None]; lg("neural network fitted (%d training rows)" % len(ntr[1]))
    MS = ["garch_t", "gjr_skewt", "fhs", "param_bteg", "sav_caviar", "neural", "body", "engine", "body_bteg", "engine_bteg"]
    keep = np.isfinite(mk) & np.isfinite(sgn) & np.all([np.isfinite(Q[m]).all(1) for m in MS], axis=0) & np.all([np.isfinite(VE[m][a][1]) for m in VE for a in A], axis=0)
    Y, dates, NAME, mk, sgn, ssz = Y[keep], dates[keep], NAME[keep], mk[keep], sgn[keep], ssz[keep]; Q = {m: Q[m][keep] for m in MS}; VE = {m: {a: (VE[m][a][0][keep], VE[m][a][1][keep]) for a in A} for m in VE}
    rk = pd.qcut(pd.Series(mk).rank(method="first"), 10, labels=False).values + 1
    np.savez_compressed(os.path.join(ROOT, "data", "panels", TAG + "_rows.npz"), y=Y.astype("float32"), date=np.array([str(d.date()) for d in dates]), name=NAME, names=np.array(names), mk63=mk, rk=rk,
                        shock_sign=sgn, shock_size=ssz, taus=np.array(TAUS), **{"q__" + m: Q[m].astype("float32") for m in MS}, **{"es__%s__%s" % (m, a): VE[m][a][1].astype("float32") for m in VE for a in A})
    # ---------------------------------------------------------------- scoring
    pin = lambda q, t: np.where(Y - q >= 0, t * (Y - q), (t - 1) * (Y - q))
    LOSS = {"pin_0.025": {m: pin(Q[m][:, JA[0.025]], 0.025) for m in MS}, "pin_0.01": {m: pin(Q[m][:, JA[0.01]], 0.01) for m in MS},
            "pin11": {m: np.mean([pin(Q[m][:, j], t) for j, t in enumerate(TAUS)], axis=0) for m in MS}}
    FZ = {a: {m: C.fz0(Y, VE[m][a][0], VE[m][a][1], a) for m in VE} for a in A}
    def dm(d, mask):
        s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values
        if len(s) < 30: return None
        dd = s - s.mean(); v = np.mean(dd * dd)
        for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
        return round(float(s.mean() / math.sqrt(max(v / len(s), 1e-18))), 2)
    edge = lambda Lr, Lm, mask: {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask)}
    gain = lambda Lr, Lm, mask: {"gain": round(float((Lr - Lm)[mask].mean()), 4), "DM": dm(Lr - Lm, mask)}
    ALL = np.ones(len(Y), bool); TOP = rk == 10; D9 = rk == 9; LOW = rk <= 8; REST = rk <= 9
    OUT = {"panel": TAG, "screened": SCREEN, "class": CFG["class"], "precommit": "docs/PANELS_PRECOMMIT.md", "note": "edge_pct > 0 and DM > 0: the first-named model has the lower loss than the reference.",
           "listed": len(CFG["tickers"]), "n_names": len(names), "dropped": dropped, "fx_ticks_removed": int(sum(R[t]["nbad"] for t in names)), "n_test_rows": int(len(Y)), "n_dates": int(len(pd.unique(dates))),
           "test": [str(dates.min().date()), str(dates.max().date())], "n_train_rows": int(len(POOLS["garch_t"])),
           "top_decile": {"mk63_median": round(float(np.median(mk[TOP])), 2), "largest_shock_median": round(float(np.median(ssz[TOP])), 2), "share_positive_shock": round(float((sgn[TOP] > 0).mean()), 3)}, "measures": {}, "fz0": {}}
    for meas, L in LOSS.items():
        b = {"vs_garch_t": {m: {"top": edge(L["garch_t"], L[m], TOP), "decile9": edge(L["garch_t"], L[m], D9), "deciles1to8": edge(L["garch_t"], L[m], LOW), "overall": edge(L["garch_t"], L[m], ALL),
                                "profile": [edge(L["garch_t"], L[m], rk == d)["edge_pct"] for d in range(1, 11)]} for m in MS if m != "garch_t"},
             "vs_bteg_t_top": {m: edge(L["param_bteg"], L[m], TOP) for m in MS if m != "param_bteg"}, "vs_bteg_t_overall": {m: edge(L["param_bteg"], L[m], ALL) for m in MS if m != "param_bteg"},
             "sign": {m: {"top_positive": edge(L["garch_t"], L[m], TOP & (sgn > 0)), "top_negative": edge(L["garch_t"], L[m], TOP & (sgn < 0))} for m in ("body", "sav_caviar", "neural", "gjr_skewt", "param_bteg")}}
        OUT["measures"][meas] = b
    for a in A:
        F = FZ[a]; OUT["fz0"][str(a)] = {"vs_garch_t": {m: {"top": gain(F["garch_t"], F[m], TOP), "deciles1to9": gain(F["garch_t"], F[m], REST), "overall": gain(F["garch_t"], F[m], ALL)} for m in F if m != "garch_t"},
                                         "vs_bteg_t": {m: {"top": gain(F["param_bteg"], F[m], TOP), "overall": gain(F["param_bteg"], F[m], ALL)} for m in F if m != "param_bteg"},
                                         "breach_pct": {m: round(100 * float((Y <= VE[m][a][0]).mean()), 2) for m in VE}, "breach_pct_top_decile": {m: round(100 * float((Y <= VE[m][a][0])[TOP].mean()), 2) for m in VE}}
    P_ = OUT["measures"]["pin_0.025"]; vg = P_["vs_garch_t"]; vb = P_["vs_bteg_t_top"]; sig = lambda e: e["DM"] is not None and e["DM"] > 1.96
    OUT["graded_on_pinball_2.5pct"] = {
        "W_where": bool(sig(vg["body"]["top"]) and vg["body"]["top"]["edge_pct"] > vg["body"]["deciles1to8"]["edge_pct"]),
        "Y_why": bool(sig(vg["param_bteg"]["top"]) and not sig(vb["body"])),
        "C_class": bool(sig(vg["sav_caviar"]["top"]) and sig(vg["neural"]["top"]) and not sig(vb["sav_caviar"]) and not sig(vb["neural"])),
        "S_shape": bool(vb["body_bteg"]["edge_pct"] > 0), "G_sign": bool(P_["sign"]["body"]["top_positive"]["edge_pct"] > P_["sign"]["body"]["top_negative"]["edge_pct"])}
    json.dump(OUT, open(os.path.join(ROOT, "results", "panels", TAG + "_results.json"), "w"), indent=1)
    f = lambda e: "%+.2f%% (%s)" % (e["edge_pct"], e["DM"])
    print("\n######## %s: %d names, %d test rows, %d dates; top decile: kurtosis %.1f, largest shock %.1f SD, %.0f%% positive" % (TAG, len(names), len(Y), OUT["n_dates"], OUT["top_decile"]["mk63_median"], OUT["top_decile"]["largest_shock_median"], 100 * OUT["top_decile"]["share_positive_shock"]))
    for meas in ("pin_0.025", "pin_0.01", "pin11"):
        b = OUT["measures"][meas]; print("--- %s vs GARCH-t  [top | decile 9 | deciles 1-8 | overall]   then top decile vs Beta-t-EGARCH" % meas)
        for m in MS[1:]: print("  %-12s %s | %s | %s | %s   || %s" % (m, f(b["vs_garch_t"][m]["top"]), f(b["vs_garch_t"][m]["decile9"]), f(b["vs_garch_t"][m]["deciles1to8"]), f(b["vs_garch_t"][m]["overall"]), f(b["vs_bteg_t_top"][m]) if m in b["vs_bteg_t_top"] else "ref"))
        print("  sign (top decile, positive | negative):", {m: (f(v["top_positive"]), f(v["top_negative"])) for m, v in b["sign"].items()})
    for a in ("0.01", "0.025"): print("--- FZ0 %s gain over GARCH-t [top | rest | overall]:" % a, {m: (v["top"]["gain"], v["top"]["DM"], v["deciles1to9"]["gain"], v["deciles1to9"]["DM"], v["overall"]["gain"], v["overall"]["DM"]) for m, v in OUT["fz0"][a]["vs_garch_t"].items()}, "| breach", OUT["fz0"][a]["breach_pct"])
    print("GRADED:", OUT["graded_on_pinball_2.5pct"]); lg("done")
