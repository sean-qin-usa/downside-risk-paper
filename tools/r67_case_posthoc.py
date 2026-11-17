# r67_case_posthoc.py -- post-hoc analyses of the cross-asset case study on its saved daily losses (frozen protocol,
# all 32 instruments). Recorded in docs/CASE_STUDY_PRECOMMIT.md as NOT confirmatory. No model is refitted.
# Flagged days: the days the score-gated row switches scale (mk63 above the instrument's pre-2020 90th percentile).
# Writes results/case_studies/crossasset_posthoc.json. Run from the repo root.
import json, math, numpy as np, pandas as pd
from scipy import stats
D={}
for s in ('confirmatory','exploratory'):
    z=np.load('data/case_studies/crossasset_frozen_daily_%s.npz'%s); D.update({k:z[k] for k in z.files})
TK=sorted({k.split('|')[0] for k in D}); STK=["MSFT","PFE","WMT","CAT","BA","NFLX","KO","JNJ","XOM","JPM","AAPL","AMD","TSLA","NVDA"]
CLASS={"stocks":STK,"equity ETFs":["IWM","XLU","SPY","EEM"],"credit and rates":["HYG","LQD","TLT"],"commodities":["SLV","DBA","GLD","USO","UNG"],"FX":["GBPUSD=X","AUDUSD=X","EURUSD=X","JPY=X"],"crypto":["ETH-USD","BTC-USD"]}
nyse=set(pd.read_csv('data/case_studies/SPY.csv',index_col=0,parse_dates=True).index)
def nwse(x,l=10):
    x=np.asarray(x,float); m=len(x); d=x-x.mean(); v=np.mean(d*d)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(d[k:]*d[:-k])
    return math.sqrt(v/m)
def holm(p):
    k=sorted(range(len(p)),key=lambda i:p[i]); adj=[0]*len(p); run=0
    for r,i in enumerate(k): run=max(run,min(1,(len(p)-r)*p[i])); adj[i]=run
    return adj
flag={t:(D[t+'|fz0|gate|0.01']!=D[t+'|fz0|garch_t|0.01'])|(D[t+'|pin|gate|0.025']!=D[t+'|pin|garch_t|0.025']) for t in TK}
def pooled(tks,kind,a,m,ref,mask=None):
    parts=[]
    for t in tks:
        v=D['%s|%s|%s|%s'%(t,kind,ref,a)]-D['%s|%s|%s|%s'%(t,kind,m,a)]
        if kind=='pin': v=v/D['%s|pin|garch_t|%s'%(t,a)].mean()
        msk=np.ones(len(v),bool) if mask is None else (flag[t] if mask=='flag' else ~flag[t])
        parts.append(pd.Series(v[msk],index=pd.to_datetime(D[t+'|dates'])[msk]))
    s=pd.concat(parts,axis=1); s=s[s.index.isin(nyse)].mean(axis=1).dropna().values
    return s
out={"note":"gain > 0 and DM > 0: the model has the lower loss than the reference. Frozen protocol (every model fitted once on pre-2020 data). Post hoc, not confirmatory.",
     "flagged_share_pct":round(100*float(np.mean(np.concatenate([flag[t] for t in TK]))),1),"flagged_days_vs_garch_t":{},"noninferiority_all32":{},"gate":{},"by_class_vs_benchmarks":{}}
for grp,tks in (("all 32",TK),("14 stocks",STK),("18 non-stocks",[t for t in TK if t not in STK])):
    out["flagged_days_vs_garch_t"][grp]={}
    for which in ("flag","rest"):
        r={}
        for m in ("bteg","gjr_skewt","engine","body_bteg","engine_bteg","fhs"):
            r[m]={}
            for a in ("0.01","0.025"):
                f=pooled(tks,'fz0',a,m,'garch_t',which); p=pooled(tks,'pin',a,m,'garch_t',which)
                r[m][a]={"fz0_gain":round(float(f.mean()),4),"fz0_DM":round(float(f.mean()/nwse(f)),2),"pinball_gain_pct":round(100*float(p.mean()),2),"pinball_DM":round(float(p.mean()/nwse(p)),2)}
        out["flagged_days_vs_garch_t"][grp]["flagged" if which=="flag" else "other"]=r
B=['hs500','ewma','garch_t','gjr_skewt','fhs','garch_evt','bteg']
for a in ("0.01","0.025"):
    rows=[]
    for b in B:
        s=-pooled(TK,"fz0",a,"engine",b)   # engine minus benchmark
        se=nwse(s); rows.append((b,float(s.mean()),float(s.mean()+1.645*se),float(1-stats.norm.cdf((0.01-s.mean())/se)),float(s.mean()/se)))
    adj=holm([r[3] for r in rows])
    out["noninferiority_all32"][a]={r[0]:{"engine_minus_benchmark":round(r[1],4),"upper95":round(r[2],4),"two_sided_DM":round(r[4],2),"p_holm":round(p,4),"noninferior":bool(p<0.05)} for r,p in zip(rows,adj)}
    out["gate"][a]={ref:{"gain":round(float(s.mean()),4),"DM":round(float(s.mean()/nwse(s)),2)} for ref in ("garch_t","bteg") for s in [pooled(TK,'fz0',a,'gate',ref)]}
    out["by_class_vs_benchmarks"][a]={c:{b:{"gain":round(float(s.mean()),4),"DM":round(float(s.mean()/nwse(s)),2)} for b in B for s in [pooled(tks,'fz0',a,'engine',b)]} for c,tks in list(CLASS.items())+[("all 32",TK)]}
json.dump(out,open('results/case_studies/crossasset_posthoc.json','w'),indent=1)
print(json.dumps(out["noninferiority_all32"],indent=0)[:900]); print(out["gate"]); print(out["flagged_days_vs_garch_t"]["14 stocks"]["flagged"]["bteg"], out["flagged_days_vs_garch_t"]["18 non-stocks"]["flagged"]["bteg"])
