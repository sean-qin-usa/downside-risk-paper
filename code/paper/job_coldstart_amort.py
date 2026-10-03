# job_coldstart_amort.py -- THE PAPER'S CHARACTERISTICS-ONLY AMORTIZED MODEL AND THE RESIDUAL-HYBRID,
# AGAINST own_short_pool AND peer_young, BY LISTING AGE, ON THE ROWS OF job_coldstart_peers.py.
#
# PRE-REGISTERED PREDICTION, WRITTEN BEFORE THE RUN (2026-10-01), AS GIVEN:
#   P1. The amortized model beats own_short_pool ONLY at ages 1-5.
# Read strictly: amort_chars has the lower mean pinball in the 1-5 bucket and the higher mean pinball in
# every one of 6-14, 15-60 and 61-125. Consequences fixed in advance:
#   - If amort_chars also wins any bucket from 6 on, the cold-start claim is broader than days 1-5 and the
#     handover sentence in Section 6 must be rewritten upward.
#   - If amort_chars loses at 1-5 as well, the characteristics-only variant has no age region of its own and
#     the day-one claim rests on availability alone, not on accuracy.
# My own sub-predictions, same discipline:
#   P2. own_short_pool is unavailable at ages 1-5 by construction (20-day window, min 5 observations, lagged),
#       so the 1-5 head-to-head is vacuous unless a shorter window is admitted. The pre-registered remedy is
#       own_short2_pool, the same row with min 2 observations, which is available from age 3; P1 is graded on
#       own_short_pool where it exists and on own_short2_pool at ages 3-5.
#   P3. The residual-hybrid is unavailable in ALL FOUR buckets, because its scale needs 250 own observations.
#       That is the handover the paper describes, and the table should show it as coverage, not as a loss.
#   P4. amort_chars beats amort_chars_pit at every age, because three of its five characteristics are
#       forward-looking (see the LOOK-AHEAD note). The gap is an upper bound on how much of the published
#       cold-start edge is look-ahead rather than transfer.
#
# PART (b): AN AGE-BUCKET CONFORMAL SHIFT, pre-registered separately.
#   P5. A split-conformal shift estimated per age bucket and per level on names that ENTER the panel between
#       2014 and 2017, using only their pre-2018 rows, and applied to the post-2018 listings, brings breach
#       rates at ages 1-5 to within 0.5 points of nominal at BOTH levels, at a pinball cost under 2%.
#   PRECISION CAVEAT, registered in advance rather than discovered later: only 72 names enter the panel
#   genuinely (after the first month and before 2018), giving about 360 rows at ages 1-5. The 1% shift there
#   is therefore an order statistic of roughly four observations and the 2.5% shift of about nine. The ages
#   1-5 shift at 1% should be read as indicative. 70 of the 72 are smallcap where the test names are recent
#   IPOs, so the calibration set is also a different risk class; both are reported, neither is repaired.
#
# LOOK-AHEAD NOTE, the reason this job reports two characteristics-only variants.
# The paper's amortized feature set is XC=[lag1,abs1,rv5,rv21,mean21,dn,age,logmcap,sector,beta,annvol]
# (code/amortization/amort_full.py); its characteristics-only form is the last five. Of those, logmcap comes
# from a market-cap pull dated 2024-12-31 -- the END of the panel -- and beta and annvol are full-sample
# quantities in crsp_panel_chars.csv. For a name that lists in 2018 all three are computed from its own
# future. annvol is the most serious: it is essentially the scale the model is asked to predict.
#   amort_chars      the paper's five characteristics, as published, so the number is comparable
#   amort_chars_pit  age and sector plus a POINT-IN-TIME sector-peer log-volatility (median EWMA(0.94) of
#                    mature same-sector names through t-1), which a desk genuinely has on day one
# A further caveat that cannot be fixed here: 'age' in this panel is days since the panel start, so for the
# mature TRAINING names it is not listing age at all. The age curve is therefore identified off mature names'
# panel entry, not off real listings, and no pre-2018 listing exists in the panel to identify it properly.
#
# Panel, split and peer construction are job_coldstart_peers.py's, unchanged, so the rows are the same.
# Output: coldstart_amort_results.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
LAM=0.94; MINEW=60; CUT=pd.Timestamp("2018-01-01")
HGB=dict(max_iter=250,max_depth=4,learning_rate=0.06,max_bins=128,random_state=0)   # amort_full.py settings
ZXH=['logsig','zl1','absz5','zstd21','fracdn5']                                     # residual-hybrid features
AGEB=[(1,5),(6,14),(15,60),(61,125)]
def ab(a):
    for lo,hi in AGEB:
        if lo<=a<=hi: return "%d-%d"%(lo,hi)
    return None
def pin(y,q,t): d=y-q; return np.where(d>=0,t*d,(t-1)*d)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    d=x-x.mean(); v=np.mean(d*d)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(d[k:]*d[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)

ch=pd.read_csv(os.path.join(P,"crsp_panel_chars.csv"))
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
R=rr.pivot_table(index='date',columns='permno',values='ret').sort_index()
first=R.notna().idxmax(); first[R.notna().sum()==0]=pd.NaT
SEC=ch.set_index('permno')['sector'].to_dict(); COH=ch.set_index('permno')['cohort'].to_dict()
MC=ch.set_index('permno')['mcap_mm'].to_dict(); BET=ch.set_index('permno')['beta'].to_dict(); AV=ch.set_index('permno')['annvol'].to_dict()
cols=list(R.columns)
TRN=[p for p in cols if pd.notna(first[p]) and first[p]<CUT and COH.get(p) in ('largecap','smallcap')]
TST=[p for p in cols if pd.notna(first[p]) and first[p]>=CUT]
lg("panel %d dates %d names | train %d | test %d"%(R.shape[0],R.shape[1],len(TRN),len(TST)))
S=np.sqrt(R.pow(2).ewm(alpha=1-LAM,min_periods=MINEW,adjust=True).mean()).shift(1)
SH =R.rolling(20,min_periods=5).std().shift(1)    # own_short_pool scale, as job_coldstart_peers.py
SH2=R.rolling(20,min_periods=2).std().shift(1)    # pre-registered shorter-window sensitivity
AGE=R.notna().cumsum().where(R.notna()); LV=np.log(S.clip(lower=1e-6))
def pool_median(B,loo):
    A=B.values; nd,nc=A.shape; out=np.full((nd,nc),np.nan)
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
    return pd.DataFrame(out,index=B.index,columns=B.columns)
glob_tr=pool_median(LV[TRN],loo=False)
sec_pool=pd.DataFrame(np.nan,index=R.index,columns=cols)
for s_ in sorted(set(SEC.get(p,-1) for p in cols)):
    mem=[p for p in TRN if SEC.get(p)==s_]
    blk=pool_median(LV[mem],loo=False) if len(mem)>=5 else None
    for p in [q for q in cols if SEC.get(q)==s_]:
        sec_pool[p]=(blk.iloc[:,0] if blk is not None else glob_tr.iloc[:,0]).values
yl=pool_median(LV[TST],loo=True).reindex(columns=cols)
lg("peer pools built %.0fs"%(time.time()-t0))

def frame(names,lsrc):
    LS=lsrc[names]
    d=pd.DataFrame({'y':R[names].stack(dropna=True)}).reset_index().rename(columns={'level_1':'permno'})
    ix=pd.MultiIndex.from_arrays([d['date'],d['permno']])
    d['logsig_peer']=LS.stack(dropna=True).reindex(ix).values
    d['age']=AGE[names].stack(dropna=True).reindex(ix).values
    d['own_sh']=SH[names].stack(dropna=True).reindex(ix).values
    d['own_sh2']=SH2[names].stack(dropna=True).reindex(ix).values
    d['logmcap']=np.log(np.maximum(d['permno'].map(MC).astype(float),1.0))
    d['sector']=d['permno'].map(SEC).astype(float)
    d['beta']=d['permno'].map(BET).astype(float); d['annvol']=d['permno'].map(AV).astype(float)
    return d.dropna(subset=['y','logsig_peer','age'])
TR=frame(TRN,sec_pool); TR=TR[TR['date']<CUT]
TE=frame(TST,sec_pool); TEy=frame(TST,yl)[['permno','date','logsig_peer']].rename(columns={'logsig_peer':'logsig_young'})
TE=TE.merge(TEy,on=['permno','date'],how='inner').sort_values(['permno','date']).reset_index(drop=True)
# pooled TRAIN residual shape, exactly job_coldstart_peers.py's
ztr=(TR['y']/np.exp(TR['logsig_peer'])).values; ztr=ztr[np.isfinite(ztr)]
POOLQ={t:float(np.quantile(ztr,t)) for t in TAUS}
lg("train rows %d | test rows %d | pooled q01 %.3f %.0fs"%(len(TR),len(TE),POOLQ[0.01],time.time()-t0))

# ---- the two characteristics-only amortized models, trained on TRAIN names and pre-CUT dates only
CH_PAPER=['age','logmcap','sector','beta','annvol']
CH_PIT  =['age','sector','logsig_peer']
AM={}; AMOD={}
for nm,XC in [('amort_chars',CH_PAPER),('amort_chars_pit',CH_PIT)]:
    AM[nm]={}
    for t in TAUS:
        mdl=HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TR[XC].values,TR['y'].values)
        AM[nm][t]=mdl.predict(TE[XC].values); AMOD[(nm,t)]=(mdl,XC)
    lg("  %s trained on %s %.0fs"%(nm,XC,time.time()-t0))

# ---- the residual-hybrid: own GARCH-t scale (expanding refits 250/500/1000) x pooled residual shape
gs=np.full(len(TE),np.nan); gmu=np.full(len(TE),np.nan)
for p in TE['permno'].unique():
    y=R[p].dropna(); yv=y.values; n=len(yv)
    sg=np.full(n,np.nan); mu_=np.full(n,np.nan)
    for cutk in (250,500,1000):
        if n<=cutk: break
        try:
            r1=arch_model(yv[:cutk],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
            pp=r1.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0))
            e=yv-mu; s2=np.empty(n); s2[0]=np.var(yv[:cutk])
            for k in range(1,n): s2[k]=max(om+al*e[k-1]**2+be*s2[k-1],1e-8)
            hi=n if cutk==1000 or n<=2*cutk else 2*cutk
            sg[cutk:hi]=np.sqrt(s2[cutk:hi]); mu_[cutk:hi]=mu
        except Exception: pass
    ser=pd.DataFrame({'date':y.index,'g':sg,'m':mu_}); ser['permno']=p
    m=TE['permno']==p
    j=TE.loc[m,['date']].merge(ser,on='date',how='left')
    gs[m.values]=j['g'].values; gmu[m.values]=j['m'].values
lg("residual-hybrid scale built %.0fs"%(time.time()-t0))

# ---- part (b): calibration names that genuinely ENTER the panel 2014-2017, pre-2018 rows only
P0DATE=R.index.min()+pd.Timedelta(days=20)
CALN=[p for p in TRN if pd.notna(first[p]) and P0DATE<first[p]<CUT]
CAL=frame(CALN,sec_pool); CAL=CAL[CAL['date']<CUT].copy()
CAL['bucket']=[ab(int(a)) for a in CAL['age'].values]
lg("age-conformal calibration: %d names, %d pre-2018 rows, by bucket %s"%(
    len(CALN),len(CAL),CAL['bucket'].value_counts().to_dict()))

TE['bucket']=[ab(int(a)) for a in TE['age'].values]
Y=TE['y'].values; di,_=pd.factorize(pd.to_datetime(TE['date'].values),sort=True); di=np.asarray(di)
RQ={}
RQ['amort_chars']={t:AM['amort_chars'][t] for t in TAUS}
RQ['amort_chars_pit']={t:AM['amort_chars_pit'][t] for t in TAUS}
RQ['resid_hybrid']={t:gmu+gs*POOLQ[t] for t in TAUS}
RQ['own_short_pool']={t:TE['own_sh'].values*POOLQ[t] for t in TAUS}
RQ['own_short2_pool']={t:TE['own_sh2'].values*POOLQ[t] for t in TAUS}
RQ['peer_young']={t:np.exp(TE['logsig_young'].values)*POOLQ[t] for t in TAUS}
MODELS=list(RQ)
# the same forecasts on the calibration rows, so a per-bucket shift can be estimated out of sample
CQ={}
for nm in ('amort_chars','amort_chars_pit'):
    CQ[nm]={t:AMOD[(nm,t)][0].predict(CAL[AMOD[(nm,t)][1]].values) for t in TAUS}
CQ['peer_young']={t:np.exp(CAL['logsig_peer'].values)*POOLQ[t] for t in TAUS}   # sector pool stands in pre-2018
SHIFT={}; CORR=['amort_chars','amort_chars_pit','peer_young']
for m in CORR:
    SHIFT[m]={}
    for b in [ab(lo) for lo,_ in AGEB]:
        cm=(CAL['bucket']==b).values
        if cm.sum()<30: continue
        SHIFT[m][b]={str(t):float(np.quantile((CAL['y'].values-CQ[m][t])[cm],t)) for t in TAUS}
    RQ[m+'_ageconf']={}
    for t in TAUS:
        q=RQ[m][t].copy()
        for b,sh in SHIFT[m].items():
            bm=(TE['bucket']==b).values
            q[bm]=q[bm]+sh[str(t)]
        RQ[m+'_ageconf'][t]=q
MODELS=list(RQ)
PL={}
for m in MODELS:
    L=sum(pin(Y,RQ[m][t],t) for t in TAUS)/len(TAUS)
    fin=np.isfinite(np.sum(np.stack([RQ[m][t] for t in TAUS]),axis=0))
    PL[m]=np.where(fin,L,np.nan)
def head(Lref,Lm,msk):
    msk=msk&np.isfinite(Lref)&np.isfinite(Lm)
    if msk.sum()<30: return None
    d=(Lref-Lm)[msk]; g=pd.DataFrame({'d':d,'dt':di[msk]}).groupby('dt')['d'].mean()
    return {'edge_pct':round(100*float(d.mean())/float(Lref[msk].mean()),3),'DM':nw_t(g.values),'n':int(msk.sum())}
OUT={'note':'The paper characteristics-only amortized model (amort_chars, features age/logmcap/sector/beta/'
     'annvol as in amort_full.py) and a look-ahead-free variant (amort_chars_pit, age/sector plus a '
     'point-in-time sector-peer log-vol), the residual-hybrid (own GARCH-t scale x pooled TRAIN residual '
     'shape), and job_coldstart_peers.py own_short_pool and peer_young, on that job rows and split. '
     'SIGN: edge_pct>0 and DM>0 both mean the RIVAL beats the reference, since edge=(ref loss - rival loss)/ref.',
     'prediction_written_before_run':'P1 (as given): the amortized model beats own_short_pool ONLY at ages 1-5. '
     'P2 own_short_pool is unavailable at 1-5; graded on own_short2_pool there. P3 the residual-hybrid is '
     'unavailable in all four buckets. P4 amort_chars beats amort_chars_pit everywhere, the gap bounding look-ahead.',
     'lookahead_note':'logmcap is a 2024-12-31 market-cap pull and beta/annvol are full-sample, so three of the '
     'five published characteristics are computed from a test name future; age is days since panel start, so '
     'for mature training names it is not listing age.',
     'train_names':len(TRN),'train_rows':int(len(TR)),'test_names':int(TE.permno.nunique()),'test_rows':int(len(TE)),
     'features':{'amort_chars':CH_PAPER,'amort_chars_pit':CH_PIT},'age_buckets':[ab(lo) for lo,_ in AGEB],'by_age':{}}
for b in [ab(lo) for lo,_ in AGEB]:
    msk=(TE['bucket']==b).values
    if msk.sum()<30: continue
    row={'n_rows':int(msk.sum()),'n_names':int(TE.loc[msk,'permno'].nunique()),'availability':{},'mean_pinball':{},'breach':{},
         'vs_own_short_pool':{},'vs_own_short2_pool':{}}
    for m in MODELS:
        av=float(np.mean(np.isfinite(PL[m][msk]))); row['availability'][m]=round(av,4)
        row['mean_pinball'][m]=round(float(np.nanmean(PL[m][msk])),5) if av>0 else None
        row['breach'][m]={}
        for a in ALPHAS:
            q=RQ[m][a]; ok=msk&np.isfinite(q)
            row['breach'][m]['alpha_%g'%a]=round(float(np.mean((Y<=q)[ok])),4) if ok.sum()>=30 else None
    for ref in ('own_short_pool','own_short2_pool'):
        for m in MODELS:
            if m==ref: continue
            row['vs_'+ref][m]=head(PL[ref],PL[m],msk)
    OUT['by_age'][b]=row
    lg("[age %s] n=%d pinball: "%(b,msk.sum())+json.dumps({m:row['mean_pinball'][m] for m in MODELS}))
json.dump(OUT,open(os.path.join(P,"coldstart_amort_results.json"),"w"),indent=2)
lg("COLDSTARTAMORTDONE %.0fs"%(time.time()-t0))
