# r70_neural_quantile.py -- A THIRD FLEXIBLE MODEL for the attribution: a pooled neural quantile regression in returns
# space, with no GARCH filter anywhere in it. The paper's attribution (most of a flexible model's top-decile gain over
# GARCH-t is a scale correction that a bounded-influence filter delivers on its own) rests on two flexible models: the
# pooled tree model on GARCH residuals and SAV-CAViaR. This adds a model of the deep-quantile-regression kind
# (Chronopoulos, Raftapostolos and Kapetanios 2024): a feed-forward network mapping past returns to the eleven return
# quantiles, trained by pinball loss, pooled over the 200 names.
#
# Design, fixed before any test loss was computed:
#  rows      the canonical split: per name, estimation window = first 60%, training = first 45%, validation = the next 15%
#            (the calibration split of the paper), test = last 40%; same 221,600 test rows as every other model
#  scaling   each name's returns are divided by the standard deviation of its training returns, so names pool
#  inputs    the last five returns; mean absolute return over 5, 21 and 63 days; standard deviation over 5, 21 and 63
#            days; minimum and maximum return over 21 days; share of negative returns over 5 days. All end at t-1.
#  network   three hidden layers of 64 units, ReLU, eleven outputs sorted to be non-crossing; Adam, learning rate 1e-3,
#            batch 4096, early stopping on validation pinball (patience 5, at most 60 epochs); three seeds averaged
#  score     eleven-level pinball and pinball at 1% and 2.5%, per-date Newey-West(10) DM, by decile of the GARCH-t
#            residual-kurtosis score (the same decile labels as every other model)
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-19):
#  P1. The network carries the same frontier: its eleven-level top-decile edge over GARCH-t is positive with DM > 3, and
#      its edge in deciles 1 to 8 is below the top-decile edge by at least 1.5 percentage points.
#  P2. In the top decile it does not beat Beta-t-EGARCH with a Student-t tail (its DM against that model is below 1.96).
#  P3. Over all asset-days it does not beat GARCH-t significantly (|DM| < 1.96 or negative): a pooled returns-space
#      network without a filter is not a better VaR model on average.
# Usage: GBC_PROJ=<panel dir> python tools/r70_neural_quantile.py <bench_all_bteg_rows.npz> [preds.npz]
import os, sys, json, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn
P = os.environ["GBC_PROJ"]; DUMP = sys.argv[1]; PRED = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(DUMP), "r70_preds.npz")
TAUS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]; t0 = time.time(); lg = lambda s: print("%5.0fs %s" % (time.time() - t0, s), flush=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "paper", "diagnostics", "r70_neural_quantile.json")

def features(y):
    s = pd.Series(y); f = {}
    for k in range(1, 6): f["lag%d" % k] = s.shift(k)
    for w in (5, 21, 63):
        f["mabs%d" % w] = s.abs().rolling(w, min_periods=max(3, w // 3)).mean().shift(1); f["sd%d" % w] = s.rolling(w, min_periods=max(3, w // 3)).std().shift(1)
    f["min21"] = s.rolling(21, min_periods=8).min().shift(1); f["max21"] = s.rolling(21, min_periods=8).max().shift(1); f["dn5"] = (s < 0).rolling(5, min_periods=3).mean().shift(1)
    return pd.DataFrame(f)

if not os.path.exists(PRED):
    rr = pd.read_csv(os.path.join(P, "crsp_panel_returns.csv"), dtype={'permno': 'int32'}); rr['date'] = pd.to_datetime(rr['date']); rr['ret'] = pd.to_numeric(rr['ret'], errors='coerce') * 100.0
    cnt = rr.groupby('permno')['ret'].count().sort_values(ascending=False); names = cnt[cnt >= 1500].index.tolist()[:200]
    TR, VA, TE = [], [], []
    for pn in names:
        g = rr[rr.permno == pn].sort_values('date'); y = g['ret'].values.astype(float); n = len(y); sp = int(n * 0.6); cp = int(sp * 0.75); sc = float(np.std(y[:cp]))
        F = features(y / sc); F["y"] = y / sc; F["idx"] = np.arange(n); F["sc"] = sc; F["permno"] = pn; F["date"] = g['date'].values; F["ret"] = y
        ok = F.drop(columns=["y", "idx", "sc", "permno", "date", "ret"]).notna().all(axis=1)
        TR.append(F[ok & (F.idx < cp)]); VA.append(F[ok & (F.idx >= cp) & (F.idx < sp)]); TE.append(F[F.idx >= sp].fillna(0.0))
    TR, VA, TE = pd.concat(TR), pd.concat(VA), pd.concat(TE); XC = [c for c in TR.columns if c not in ("y", "idx", "sc", "permno", "date", "ret")]
    mu_x, sd_x = TR[XC].mean().values, TR[XC].std().values + 1e-8
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tt = torch.tensor(TAUS, dtype=torch.float32, device=dev)
    T_ = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
    Xtr, Ytr = T_((TR[XC].values - mu_x) / sd_x), T_(TR["y"].values); Xva, Yva = T_((VA[XC].values - mu_x) / sd_x), T_(VA["y"].values); Xte = T_((TE[XC].values - mu_x) / sd_x)
    def pinball(q, y): d = y[:, None] - q; return torch.maximum(tt * d, (tt - 1) * d).mean()
    preds = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed); np.random.seed(seed)
        net = nn.Sequential(nn.Linear(len(XC), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(TAUS))).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); best = (1e9, None); bad = 0
        for ep in range(60):
            net.train(); perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), 4096):
                b = perm[i:i + 4096]; opt.zero_grad(); loss = pinball(torch.sort(net(Xtr[b]), dim=1).values, Ytr[b]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): v = float(pinball(torch.sort(net(Xva), dim=1).values, Yva))
            if v < best[0] - 1e-6: best = (v, {k: w.clone() for k, w in net.state_dict().items()}); bad = 0
            else:
                bad += 1
                if bad >= 5: break
        net.load_state_dict(best[1]); net.eval()
        with torch.no_grad(): preds.append(torch.sort(net(Xte), dim=1).values.cpu().numpy())
        lg("seed %d: %d epochs, validation pinball %.5f" % (seed, ep + 1, best[0]))
    q = np.mean(preds, axis=0) * TE["sc"].values[:, None]
    np.savez_compressed(PRED, q=q.astype("float32"), permno=TE["permno"].values, date=np.array([str(pd.Timestamp(d).date()) for d in TE["date"].values]), y=TE["ret"].values,
                        n_train=len(TR), n_valid=len(VA), features=np.array(XC))
    lg("predictions written to %s (%d training rows, %d validation rows, %d test rows)" % (PRED, len(TR), len(VA), len(TE)))

if not os.path.exists(DUMP): lg("per-row dump not there yet; rerun to score"); sys.exit(0)
D = np.load(DUMP); N = np.load(PRED); PN = D["permno"]; dates = pd.to_datetime(D["date"]); Y = D["y"].astype(float); rk = D["rk_mk"]
key = pd.Series(np.arange(len(N["permno"])), index=pd.MultiIndex.from_arrays([N["permno"], N["date"]])); idx = key.loc[list(zip(PN, D["date"]))].values
QN = N["q"][idx].astype(float); assert np.allclose(N["y"][idx], Y, atol=1e-4)
pin = lambda y, q, t: np.where(y - q >= 0, t * (y - q), (t - 1) * (y - q))
def L11(Q): return np.mean([pin(Y, Q[:, j], t) for j, t in enumerate(TAUS)], axis=0)
def Lat(Q, a): return pin(Y, Q[:, TAUS.index(a)], a)
def dm(d, mask):
    s = pd.Series(d[mask], index=dates[mask]).groupby(level=0).mean().values; dd = s - s.mean(); v = np.mean(dd * dd)
    for k in range(1, 11): v += 2 * (1 - k / 11) * np.mean(dd[k:] * dd[:-k])
    return round(float(s.mean() / math.sqrt(v / len(s))), 2)
edge = lambda Lr, Lm, mask: {"edge_pct": round(100 * float((Lr - Lm)[mask].mean() / Lr[mask].mean()), 2), "DM": dm(Lr - Lm, mask)}
REG = {"top_decile": rk == 10, "decile9": rk == 9, "deciles1to8": (rk >= 1) & (rk <= 8), "overall": np.ones(len(Y), bool)}
Qm = {m: D["q__" + m].astype(float) for m in ("garch_t", "param_bteg", "body", "sav_caviar", "gjr_skewt", "body_bteg")}
out = {"note": "edge_pct > 0 and DM > 0: the first-named model has the lower loss. Neural model: pooled returns-space quantile network, no GARCH filter.",
       "predictions_written_before_run": "P1 same frontier (top-decile eleven-level edge over GARCH-t DM > 3; deciles 1-8 at least 1.5 points lower); P2 does not beat Beta-t-EGARCH in the top decile (DM < 1.96); P3 does not beat GARCH-t significantly overall.",
       "n_train": int(N["n_train"]), "n_valid": int(N["n_valid"]), "features": [str(x) for x in N["features"]], "eleven_level": {}, "level_0.01": {}, "level_0.025": {}}
for lab, fn in (("eleven_level", L11), ("level_0.01", lambda Q: Lat(Q, 0.01)), ("level_0.025", lambda Q: Lat(Q, 0.025))):
    Ln = fn(QN)
    out[lab] = {"neural_vs_garch_t": {r: edge(fn(Qm["garch_t"]), Ln, m) for r, m in REG.items()}, "neural_vs_bteg_t": {r: edge(fn(Qm["param_bteg"]), Ln, m) for r, m in REG.items()},
                "neural_vs_pooled_trees": {r: edge(fn(Qm["body"]), Ln, m) for r, m in REG.items()}, "neural_vs_sav_caviar": {r: edge(fn(Qm["sav_caviar"]), Ln, m) for r, m in REG.items()},
                "decile_profile_vs_garch_t": [edge(fn(Qm["garch_t"]), Ln, rk == d)["edge_pct"] for d in range(1, 11)],
                "for_comparison_top_decile_vs_garch_t": {m: edge(fn(Qm["garch_t"]), fn(Qm[m]), rk == 10) for m in ("body", "sav_caviar", "param_bteg", "body_bteg")},
                "breach_pct": {a: round(100 * float((Y <= QN[:, TAUS.index(a)]).mean()), 2) for a in (0.01, 0.025)}}
json.dump(out, open(OUT, "w"), indent=1)
for lab in ("eleven_level", "level_0.01", "level_0.025"):
    o = out[lab]; print("\n%s" % lab); [print("  %-22s" % k, {r: (v["edge_pct"], v["DM"]) for r, v in o[k].items()}) for k in ("neural_vs_garch_t", "neural_vs_bteg_t", "neural_vs_pooled_trees", "neural_vs_sav_caviar")]
    print("  profile:", o["decile_profile_vs_garch_t"], "| others top decile vs GARCH-t:", {m: (v["edge_pct"], v["DM"]) for m, v in o["for_comparison_top_decile_vs_garch_t"].items()}, "| breach", o["breach_pct"])
lg("wrote %s" % OUT)
