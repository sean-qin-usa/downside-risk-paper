# r63_coldstart_vs_peer.py -- READ-ONLY POST-HOC DIAGNOSTIC FOR R63. No prediction was written before this ran:
# it was written to answer an outside reader's question (does the cold-start claim survive a peer proxy?) after
# job_coldstart_peers.py's own pre-registered predictions P1-P4 had failed, and is reported as a check, not a test.
# Rows, split and the reference forecast are job_coldstart_peers.py's, rebuilt verbatim (peer_young_emp: leave-one-out
# median EWMA(0.94) log-vol of other test-cohort names times pooled TRAIN empirical residual quantiles; per-bucket mean
# pinball matches coldstart_peers_results.json to five decimals). Rivals: the paper's amortized model with the
# point-in-time feature set XC_PIT of job_amort_pit.py, trained on the TRAIN names before the 2018 cutoff ("oot") or on
# all their dates ("intime"), the look-ahead feature set XC_OLD for reference, and the paper's Gaussian expanding-window
# own-history benchmark. Losses at the paper's seven levels (T7) and the peers job's eleven (T11); per-date NW(10) DM.
# AGE here is true days since listing for the 2018+ test cohort, unlike job_amort_pit.py, whose "age" is days since
# the name enters the 2014 panel. Usage: GBC_PROJ=<data dir> python tools/r63_coldstart_vs_peer.py
import os, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
P=os.environ["GBC_PROJ"]; t0=time.time(); lg=lambda s:print("%5.0fs %s"%(time.time()-t0,s),flush=True)
T11=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; T7=[0.05,0.10,0.25,0.50,0.75,0.90,0.95]
LAM=0.94; MINEW=60; CUT=pd.Timestamp("2018-01-01"); ZX=['logsig','peer_disp','peer_absz5']
HGB_A=dict(max_iter=250,max_depth=4,learning_rate=0.06,max_bins=128,random_state=0)
def pin(y,q,t): d=y-q; return np.where(d>=0,t*d,(t-1)*d)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    dm=x-x.mean(); v=np.mean(dm*dm)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(dm[k:]*dm[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
# ---------------- peers job panel, verbatim logic
ch=pd.read_csv(os.path.join(P,"crsp_panel_chars.csv"))
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
lg("NaN rets in panel: %d"%rr['ret'].isna().sum())
R=rr.pivot_table(index='date',columns='permno',values='ret').sort_index()
first=R.notna().idxmax(); first[R.notna().sum()==0]=pd.NaT
SEC=ch.set_index('permno')['sector'].to_dict(); COH=ch.set_index('permno')['cohort'].to_dict()
MC=ch.set_index('permno')['mcap_mm'].to_dict(); BET=ch.set_index('permno')['beta'].to_dict(); AV=ch.set_index('permno')['annvol'].to_dict()
cols=list(R.columns)
TRN=[p for p in cols if pd.notna(first[p]) and first[p]<CUT and COH.get(p) in ('largecap','smallcap')]
TST=[p for p in cols if pd.notna(first[p]) and first[p]>=CUT]
S=np.sqrt(R.pow(2).ewm(alpha=1-LAM,min_periods=MINEW,adjust=True).mean()).shift(1)
AGE=R.notna().cumsum().where(R.notna()); LV=np.log(S.clip(lower=1e-6))
def pool_median(LVblock,loo):
    A=LVblock.values; nd,nc=A.shape; out=np.full((nd,nc),np.nan)
    for d in range(nd):
        row=A[d]; m=np.isfinite(row); n=int(m.sum())
        if n<3: continue
        v=row[m]; order=np.argsort(v,kind='mergesort'); s=v[order]
        pos=np.empty(n,dtype=int); pos[order]=np.arange(n); k=n//2
        med=s[k] if n%2 else 0.5*(s[k-1]+s[k])
        if not loo: out[d,:]=med; continue
        if n%2: out[d,m]=np.where(pos<k,0.5*(s[k]+s[k+1]),np.where(pos==k,0.5*(s[k-1]+s[k+1]),0.5*(s[k-1]+s[k])))
        else:   out[d,m]=np.where(pos<=k-1,s[k],s[k-1])
        out[d,~m]=med
    return pd.DataFrame(out,index=LVblock.index,columns=LVblock.columns)
def pool_disp(B):
    q=B.quantile([0.25,0.75],axis=1); return (q.loc[0.75]-q.loc[0.25]).reindex(B.index)
PEER={}
sec_pool=pd.DataFrame(np.nan,index=R.index,columns=cols); glob_tr=pool_median(LV[TRN],loo=False)
for s in sorted(set(SEC.get(p,-1) for p in cols)):
    mem=[p for p in TRN if SEC.get(p)==s]
    blk=pool_median(LV[mem],loo=False) if len(mem)>=5 else None
    for p in [q for q in cols if SEC.get(q)==s]: sec_pool[p]=(blk.iloc[:,0] if blk is not None else glob_tr.iloc[:,0]).values
PEER['peer_sector']=sec_pool
PEER['peer_young']=pool_median(LV[TST],loo=True).reindex(columns=cols)
PEER['global']=pool_median(LV[cols],loo=True)
DISP=pool_disp(LV[TRN])
def frame(pset,names):
    LS=PEER[pset][names]
    mkt=(R[names].abs()/np.exp(LS)).mean(axis=1).rolling(5,min_periods=3).mean().shift(1)
    d=pd.DataFrame({'y':R[names].stack(dropna=True)}).reset_index().rename(columns={'level_1':'permno'})
    ix=pd.MultiIndex.from_arrays([d['date'],d['permno']])
    d['logsig']=LS.stack(dropna=True).reindex(ix).values
    d['age']=AGE[names].stack(dropna=True).reindex(ix).values
    d['peer_disp']=d['date'].map(DISP); d['peer_absz5']=d['date'].map(mkt)
    return d.dropna(subset=['logsig','y'])
tr=frame('peer_sector',TRN); tr=tr[tr['date']<CUT].dropna(subset=ZX)
z=(tr['y']/np.exp(tr['logsig'])).values; z=z[np.isfinite(z)]
POOLQ={t:float(np.quantile(z,t)) for t in T11}
TEb={ps:frame(ps,TST).dropna(subset=ZX) for ps in PEER}
base=TEb['peer_sector'][['permno','date','y','age']].copy()
for ps in PEER: base=base.merge(TEb[ps][['permno','date','logsig']].rename(columns={'logsig':'ls_'+ps}),on=['permno','date'],how='inner')
TE=base.sort_values(['permno','date']).reset_index(drop=True)
lg("peers rows: train %d test %d (JSON: 360950 / 203067)"%(len(tr),len(TE)))
# ---------------- paper amortized features, verbatim from job_amort_pit.py
r=rr[['permno','date','ret']].copy().sort_values(['permno','date']).reset_index(drop=True)
g=r.groupby('permno',sort=False); r['age_p']=g.cumcount()
r['lag1']=g['ret'].shift(1); r['abs1']=r['lag1'].abs()
r['rv5']=g['ret'].transform(lambda x:x.rolling(5,min_periods=3).std().shift(1))
r['rv21']=g['ret'].transform(lambda x:x.rolling(21,min_periods=8).std().shift(1))
r['mean21']=g['ret'].transform(lambda x:x.rolling(21,min_periods=8).mean().shift(1))
r['dn']=(r['lag1']<0).astype(float); r['ret2']=r['ret']**2
cs=g['ret'].cumsum()-r['ret']; cs2=g['ret2'].cumsum()-r['ret2']; cnt=r['age_p'].values.astype(float)
with np.errstate(invalid='ignore',divide='ignore'):
    om=np.where(cnt>0,cs/np.maximum(cnt,1),0.0); ov=np.where(cnt>1,cs2/np.maximum(cnt,1)-om**2,np.nan)
r['own_mean']=om; r['own_std']=np.sqrt(np.clip(ov,1e-6,None))
r['annvol_pit']=g['ret'].transform(lambda x:x.rolling(250,min_periods=60).std().shift(1))*math.sqrt(252)/100.0
r['_m']=r.groupby('date')['ret'].transform('mean')
cov=g.apply(lambda d: d['ret'].rolling(250,min_periods=60).cov(d['_m']).shift(1)).reset_index(level=0,drop=True)
var=g['_m'].transform(lambda x:x.rolling(250,min_periods=60).var().shift(1))
r['beta_pit']=(cov/var.replace(0,np.nan)).clip(-3,5)
r['sector']=r['permno'].map(SEC).astype(float).fillna(-1)
r['logmcap']=np.log(np.maximum(pd.to_numeric(r['permno'].map(MC),errors='coerce').fillna(300.0),1.0))
r['beta']=r['permno'].map(BET).astype(float).fillna(1.0); r['annvol']=r['permno'].map(AV).astype(float).fillna(0.3)
r['beta_pit']=r['beta_pit'].fillna(1.0); r['annvol_pit']=r['annvol_pit'].fillna(0.3)
r['rv5']=r['rv5'].fillna(r['rv21']).fillna(2.0); r['rv21']=r['rv21'].fillna(2.0); r['mean21']=r['mean21'].fillna(0.0)
r=r.dropna(subset=['lag1']); r['abs1']=r['abs1'].fillna(r['abs1'].median())
XC_PIT=['lag1','abs1','rv5','rv21','mean21','dn','age_p','sector','beta_pit','annvol_pit']
XC_OLD=['lag1','abs1','rv5','rv21','mean21','dn','age_p','logmcap','sector','beta','annvol']
lg("paper features built")
FE=r[['permno','date','ret']+sorted(set(XC_PIT+XC_OLD))+['own_mean','own_std']]
TE=TE.merge(FE,on=['permno','date'],how='left')
lg("test rows with paper features: %d of %d; max |y-ret| %.2e"%(TE['lag1'].notna().sum(),len(TE),np.nanmax(np.abs(TE['y']-TE['ret']))))
TRF=r[r.permno.isin(TRN)]
Y=TE['y'].values; RQ={}
RQ['peer_young_emp']={t:np.exp(TE['ls_peer_young'].values)*POOLQ[t] for t in T11}
for nm,XC,trm in [('amort_full_pit_oot',XC_PIT,TRF[TRF['date']<CUT]),('amort_full_pit_intime',XC_PIT,TRF),
                  ('amort_full_lookahead_oot',XC_OLD,TRF[TRF['date']<CUT])]:
    ok=TE['lag1'].notna().values; RQ[nm]={}
    for t in T11:
        q=np.full(len(TE),np.nan)
        q[ok]=HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB_A).fit(trm[XC].values.astype('float32'),trm['ret'].values).predict(TE.loc[ok,XC].values.astype('float32'))
        RQ[nm][t]=q
    lg("%s trained on %d rows"%(nm,len(trm)))
from scipy import stats
ownm=TE['own_mean'].values; owns=TE['own_std'].values
RQ['own_gauss']={t:ownm+np.clip(owns,1e-3,None)*stats.norm.ppf(t) for t in T11}
di,_=pd.factorize(pd.to_datetime(TE['date'].values),sort=True); di=np.asarray(di)
BK=[(1,5),(6,14),(15,60),(61,125),(126,250),(251,10**9),(1,10**9)]
age=TE['age'].values
OUT={}
for tl,TS in (('T7',T7),('T11',T11)):
    PL={m:np.where(np.isfinite(np.sum(np.stack([RQ[m][t] for t in TS]),0)),sum(pin(Y,RQ[m][t],t) for t in TS)/len(TS),np.nan) for m in RQ}
    OUT[tl]={}
    for lo,hi in BK:
        b="%d-%s"%(lo,'inf' if hi>1e8 else hi); msk=(age>=lo)&(age<=hi); row={'n_rows':int(msk.sum())}
        for ref in ('peer_young_emp','own_gauss'):
            for m in RQ:
                if m==ref: continue
                mm=msk&np.isfinite(PL[ref])&np.isfinite(PL[m])
                if mm.sum()<30: row['%s_vs_%s'%(m,ref)]=None; continue
                d=(PL[ref]-PL[m])[mm]; gg=pd.DataFrame({'d':d,'dt':di[mm]}).groupby('dt')['d'].mean()
                row['%s_vs_%s'%(m,ref)]={'edge_pct':round(100*d.mean()/PL[ref][mm].mean(),2),'DM':nw_t(gg.values),'n':int(mm.sum()),
                    'ref_loss':round(float(PL[ref][mm].mean()),4),'model_loss':round(float(PL[m][mm].mean()),4),
                    'date_win':round(float((gg>0).mean()),3)}
        row['peer_young_emp_mean_all_rows']=round(float(np.nanmean(PL['peer_young_emp'][msk])),5)
        OUT[tl][b]=row
od=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","paper","diagnostics"); os.makedirs(od,exist_ok=True)
json.dump(OUT,open(os.path.join(od,"r63_coldstart_vs_peer.json"),"w"),indent=1)
for tl in OUT:
    print("=====",tl)
    for b,row in OUT[tl].items():
        print(b,row['n_rows'],'peerY(all rows)',row['peer_young_emp_mean_all_rows'])
        for k,v in row.items():
            if isinstance(v,dict): print("   %-50s edge %6.2f DM %6s n %6d ref %.4f mdl %.4f dwin %.3f"%(k,v['edge_pct'],v['DM'],v['n'],v['ref_loss'],v['model_loss'],v['date_win']))
lg("DONE")
