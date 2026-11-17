# r68_shock_sign.py -- READ-ONLY TEST of an interpretation, on the canonical 221,600 rows.
# Established facts the interpretation rests on: jumps are less persistent than continuous variation (Andersen,
# Bollerslev and Diebold 2007); in individual stocks negative jumps raise future volatility and positive jumps lower it
# (Patton and Sheppard 2015); a GARCH(1,1) responds to the squared shock whatever its sign. If the top-decile edge of the
# flexible models over GARCH-t is the filter's post-shock overstatement, it should be larger when the window's largest
# shock was positive (volatility falls afterwards, GARCH-t raised it) than when it was negative.
# Input: per-row losses from `job_bench_all.py --dump-rows`. The sign and size of the largest residual in each row's
# 63-day window are recomputed with the canonical filter (as tools/r63_mk63_anatomy.py).
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-11-17):
#  P1. In the top mk63 decile the pooled body's pinball edge over GARCH-t is larger after a positive largest shock
#      than after a negative one, by at least a factor of 1.5.
#  P2. The edge after a negative largest shock is still positive (DM > 2): asymmetry does not explain all of it.
#  P3. GJR-GARCH-skew-t, whose variance responds less to positive shocks, has a positive top-decile edge over GARCH-t
#      after positive shocks and none after negative shocks.
# Usage: GBC_PROJ=<panel dir> python tools/r68_shock_sign.py <bench_all_rows.npz>
import os, sys, json, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from arch import arch_model
P=os.environ["GBC_PROJ"]; D=np.load(sys.argv[1]); PN=D["permno"]; dates=pd.to_datetime(D["date"]); rk=D["rk_mk"]
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'}); rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
sign=np.zeros(len(PN)); size=np.zeros(len(PN)); W=63
for pn in pd.unique(PN):
    y=rr[rr.permno==pn].sort_values('date')['ret'].values.astype(float); n=len(y); sp=int(n*0.6); m=PN==pn; nsel=int(m.sum())
    pp=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False).params
    om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0))
    e0=y-mu; s2=np.empty(n); s2[0]=np.var(y[:sp])
    for k in range(1,n): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
    z=(y-mu)/np.sqrt(s2); win=sliding_window_view(z,W); tgt=np.arange(W,n+1); keep=(tgt>=n-nsel)&(tgt<n); win=win[keep]
    d=win-win.mean(1,keepdims=True); j=np.argmax(np.abs(d),1); r=np.arange(len(win))
    sign[m]=np.sign(win[r,j]); size[m]=np.abs(win[r,j])
def dm(L,mask):
    s=pd.Series(L[mask],index=dates[mask]).groupby(level=0).mean().values; dd=s-s.mean(); v=np.mean(dd*dd)
    for k in range(1,11): v+=2*(1-k/11)*np.mean(dd[k:]*dd[:-k])
    return round(float(s.mean()/math.sqrt(v/len(s))),2)
Pg=D["pin__garch_t"]; out={"predictions_written_before_run":"P1 top-decile body edge after a positive largest shock at least 1.5x the edge after a negative one; P2 edge after negative shocks positive with DM>2; P3 GJR-skew-t edge positive after positive shocks, none after negative.","regions":{}}
for lab,mask in (("top decile, largest shock POSITIVE",(rk==10)&(sign>0)),("top decile, largest shock NEGATIVE",(rk==10)&(sign<0)),
                 ("deciles 1-9, largest shock positive",(rk<=9)&(rk>=1)&(sign>0)),("deciles 1-9, largest shock negative",(rk<=9)&(rk>=1)&(sign<0))):
    r={"rows":int(mask.sum()),"median_shock_size":round(float(np.median(size[mask])),2)}
    for m in ("body","engine","sav_caviar","gjr_skewt","fhs_name","evt_name"):
        L=Pg-D["pin__"+m]; r[m]={"edge_pct":round(100*float(L[mask].mean()/Pg[mask].mean()),2),"DM":dm(L,mask)}
    for a in ("0.01","0.025"):
        L=D["fz0__garch_t__"+a]-D["fz0__engine__"+a]; r["engine_fz0_gain_"+a]={"gain":round(float(L[mask].mean()),4),"DM":dm(L,mask)}
    out["regions"][lab]=r
o=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","paper","diagnostics","r68_shock_sign.json"); json.dump(out,open(o,"w"),indent=1)
for lab,r in out["regions"].items():
    print("%-38s rows %6d shock %.1f | "%(lab,r["rows"],r["median_shock_size"])+"  ".join("%s %+.2f%% (%.1f)"%(m,r[m]["edge_pct"],r[m]["DM"]) for m in ("body","engine","sav_caviar","gjr_skewt"))+"  | FZ0 1%% %+.4f (%.1f) 2.5%% %+.4f (%.1f)"%(r["engine_fz0_gain_0.01"]["gain"],r["engine_fz0_gain_0.01"]["DM"],r["engine_fz0_gain_0.025"]["gain"],r["engine_fz0_gain_0.025"]["DM"]))
