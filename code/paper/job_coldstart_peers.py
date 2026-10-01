# job_coldstart_peers.py -- COLD START WITH POOLED PEER-GROUP BENCHMARKS. The amortization claim is that a
# pooled residual model prices a new listing's tail from its first day, where a per-asset model cannot run at
# all. The benchmark reported so far is the name's OWN short history, and the obvious objection is that it is a
# strawman: a desk onboarding an IPO would not use ten days of the name's own returns, it would use its PEER
# GROUP. This job makes the peer group the benchmark and asks whether anything is left of the cold-start edge.
#
# PANEL. crsp_panel_returns.csv, 720 names, 2014-2024, with crsp_panel_chars.csv giving sector and cohort.
#   TRAIN  417 mature names (largecap + smallcap) whose history starts before 2018-01-01, on dates < 2018-01-01.
#   TEST   303 names whose first panel date is >= 2018-01-01 (300 of them the recent_ipo cohort), full lives,
#          scored by listing AGE in trading days. No test name and no test date enters any fitted stage, so the
#          pooled shape and the learner are out-of-name and out-of-time; peer scales are point-in-time and
#          leave-one-out, built only from OTHER names' returns through t-1.
#
# THE THREE PEER POOLS (each gives a day-one conditional scale; median over the pool's own EWMA(0.94) vols):
#   peer_sector  mature TRAIN names in the same sector      -- the realistic desk choice; disjoint from the test
#                                                              names, so self-information is structurally zero
#   peer_young   the other TEST-cohort names, leave-one-out  -- matched risk class, exact LOO median by order
#                                                              statistic; the pool a desk would use if it knew
#                                                              the name was a young listing
#   global       all other names, leave-one-out              -- the no-peer-information control
# Shapes on top of a pooled scale: pooled empirical quantiles of the TRAIN residuals, a Student-t with nu by
# MLE on those residuals, and the flexible HistGBM learner on day-one-available features only
# (log peer scale, cross-sectional peer dispersion, peer mean |z| over five days). AGE IS NOT A FEATURE:
# the training names are all mature, so an age feature would extrapolate; age is the evaluation axis instead.
# Own-history rivals carry their realized availability, which is the point of the exercise: own 20-day vol with the
# pooled shape (own_short_pool) needs 5 own days and isolates whether pooling is repairing the scale or the shape;
# own EWMA(0.94) vol with own expanding empirical quantiles (own_ewma_emp) needs 60 own days for the scale and 60
# standardized observations on top of it, so in practice about 120; own GARCH-t needs 250, with expanding refits at
# 250/500/1000. Where a rival is unavailable the row is N/A and every head-to-head is restricted to rows where both
# models exist, so no comparison is flattered by the rival's absence.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01):
#  P1. The peer-group benchmark is much stronger than own short history: at ages 15-60 the own-history edge of
#      about +5% that the amortization study reports against own history falls below +2% against peer_young.
#  P2. peer_young beats peer_sector at every age below 250 days, by more than 3% of pinball, because a young
#      listing's volatility level belongs to its cohort and not to its sector's mature names, which under-scale it.
#  P3. The flexible shape adds little over pooled empirical quantiles on the same peer scale at ages under 15
#      (under +1%), since day-one features carry almost no name-specific state; its gain grows with age.
#  P4. Coverage, not accuracy, is where the pooled models earn their place: breach rates at 1% and 2.5% stay
#      within half a point of nominal from age 1, while own-history rivals are unavailable or badly off there.
#  If P2 fails and the sector pool wins, the cohort framing is wrong and the text should say peers mean sector.
# Output: coldstart_peers_results.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
LAM=0.94; MINEW=60; CUT=pd.Timestamp("2018-01-01"); ZX=['logsig','peer_disp','peer_absz5']
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06)
AGEB=[(1,5),(6,14),(15,60),(61,125),(126,250),(251,500),(501,10**9)]
def ab(a):
    for lo,hi in AGEB:
        if lo<=a<=hi: return "%d-%s"%(lo,"inf" if hi>10**8 else hi)
    return None
def pin(y,q,t): dd=y-q; return np.where(dd>=0,t*dd,(t-1)*dd)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    dm=x-x.mean(); v=np.mean(dm*dm)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(dm[k:]*dm[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def fz0(r,v,e,a):
    v=np.minimum(v,-1e-8); e=np.minimum(e,v); hit=(r<=v).astype(float)
    return -(1.0/(a*e))*hit*(v-r)+v/e+np.log(-e)-1.0
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)

# ---------------------------------------------------------------- panel
ch=pd.read_csv(os.path.join(P,"crsp_panel_chars.csv"))
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
R=rr.pivot_table(index='date',columns='permno',values='ret').sort_index()
first=R.notna().idxmax(); first[R.notna().sum()==0]=pd.NaT
SEC=ch.set_index('permno')['sector'].to_dict(); COH=ch.set_index('permno')['cohort'].to_dict()
cols=list(R.columns)
TRN=[p for p in cols if pd.notna(first[p]) and first[p]<CUT and COH.get(p) in ('largecap','smallcap')]
TST=[p for p in cols if pd.notna(first[p]) and first[p]>=CUT]
lg("panel %d dates %d names | train %d | test %d | %.0fs"%(R.shape[0],R.shape[1],len(TRN),len(TST),time.time()-t0))

# EWMA(0.94) vol per name, using data through t-1 only
S=np.sqrt(R.pow(2).ewm(alpha=1-LAM,min_periods=MINEW,adjust=True).mean()).shift(1)
SH=R.rolling(20,min_periods=5).std().shift(1)                # own short-window vol: earliest own-history scale
AGE=R.notna().cumsum().where(R.notna())                      # own trading-day age at t
LV=np.log(S.clip(lower=1e-6))                                # log vol per name, point-in-time

def pool_median(LVblock,loo):
    """Median over a pool of log-vols per date. loo=True -> exact leave-one-out median per column, by sorted
    rank (three cases when the pool is odd-sized, since removing the median element itself is distinct), so a
    name never sees its own volatility. Verified against brute force including tied pools."""
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
def pool_disp(LVblock):
    q=LVblock.quantile([0.25,0.75],axis=1)
    return (q.loc[0.75]-q.loc[0.25]).reindex(LVblock.index)

# three peer pools -> a log-scale per (date, name)
PEER={}
secs=sorted(set(SEC.get(p,-1) for p in cols))
sec_pool=pd.DataFrame(np.nan,index=R.index,columns=cols); glob_tr=pool_median(LV[TRN],loo=False)
for s in secs:
    mem=[p for p in TRN if SEC.get(p)==s]
    blk=pool_median(LV[mem],loo=False) if len(mem)>=5 else None
    tgt=[p for p in cols if SEC.get(p)==s]
    for p in tgt: sec_pool[p]=(blk.iloc[:,0] if blk is not None else glob_tr.iloc[:,0]).values
PEER['peer_sector']=sec_pool                                  # pool = mature TRAIN names only, disjoint from TST
yl=pool_median(LV[TST],loo=True); PEER['peer_young']=yl.reindex(columns=cols)
gl=pool_median(LV[cols],loo=True);  PEER['global']=gl
DISP=pool_disp(LV[TRN]); Z_TR_ABS=None
lg("peer pools built %.0fs"%(time.time()-t0))

def frame(pset,names):
    """long frame of (permno,date,y,age,logsig,peer_disp,peer_absz5) for one peer pool"""
    LS=PEER[pset][names]
    mkt=(R[names].abs()/np.exp(LS)).mean(axis=1).rolling(5,min_periods=3).mean().shift(1)
    d=pd.DataFrame({'y':R[names].stack(dropna=True)}).reset_index().rename(columns={'level_1':'permno'})
    d['logsig']=LS.stack(dropna=True).reindex(pd.MultiIndex.from_arrays([d['date'],d['permno']])).values
    d['age']=AGE[names].stack(dropna=True).reindex(pd.MultiIndex.from_arrays([d['date'],d['permno']])).values
    d['peer_disp']=d['date'].map(DISP)
    d['peer_absz5']=d['date'].map(mkt)
    return d.dropna(subset=['logsig','y'])

# ---------------------------------------------------------------- fitted stages, TRAIN names, dates < CUT
tr=frame('peer_sector',TRN); tr=tr[tr['date']<CUT].dropna(subset=ZX)
tr['z']=tr['y']/np.exp(tr['logsig'])
ztr=tr['z'].values; ztr=ztr[np.isfinite(ztr)]
POOLQ={t:float(np.quantile(ztr,t)) for t in TAUS}
POOLE={a:float(np.mean(ztr[ztr<=np.quantile(ztr,a)])) for a in ALPHAS}
nu_hat=float(optimize.minimize_scalar(lambda v:-np.sum(stats.t.logpdf(ztr*math.sqrt(v/(v-2)),v)+0.5*math.log(v/(v-2))),
             bounds=(2.1,30),method='bounded').x)
tsc=math.sqrt(nu_hat/(nu_hat-2))
GB={t:HistGradientBoostingRegressor(loss='quantile',quantile=t,random_state=0,**HGB).fit(tr[ZX].values,tr['z'].values) for t in TAUS}
SUB={a:[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/20,random_state=0,**HGB).fit(tr[ZX].values,tr['z'].values) for j in range(20)] for a in ALPHAS}
lg("fitted on %d train rows: pooled nu=%.2f, q01=%.3f, learner ready %.0fs"%(len(tr),nu_hat,POOLQ[0.01],time.time()-t0))

# ---------------------------------------------------------------- own-history rivals on the TEST names
own={}
for p in TST:
    y=R[p].dropna(); n=len(y)
    if n<20: continue
    ew=S[p].reindex(y.index).values                                        # own EWMA vol (min 60 own days)
    sh=SH[p].reindex(y.index).values                                       # own 20-day vol (min 5 own days)
    oq={t:np.full(n,np.nan) for t in TAUS}; oe={a:np.full(n,np.nan) for a in ALPHAS}
    yv=y.values
    for k in range(60,n):                                                   # own empirical quantiles, expanding
        h=yv[:k]/np.maximum(ew[:k],1e-6); h=h[np.isfinite(h)]
        if len(h)<60: continue
        for t in TAUS: oq[t][k]=np.quantile(h,t)
        for a in ALPHAS:
            qa=np.quantile(h,a); oe[a][k]=np.mean(h[h<=qa]) if (h<=qa).any() else qa*1.2
    gs=np.full(n,np.nan); gnu=np.full(n,np.nan); gmu=np.full(n,np.nan)
    for cutk in (250,500,1000):                                             # own GARCH-t, expanding refits
        if n<=cutk: break
        try:
            r1=arch_model(yv[:cutk],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
            pp=r1.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
            e0=yv-mu; s2=np.empty(n); s2[0]=np.var(yv[:cutk])
            for k in range(1,n): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
            hi=n if cutk==1000 or n<=2*cutk else 2*cutk
            gs[cutk:hi]=np.sqrt(s2[cutk:hi]); gnu[cutk:hi]=nu; gmu[cutk:hi]=mu
        except Exception: pass
    own[p]=pd.DataFrame({'date':y.index,'own_ew':ew,'own_sh':sh,'g_sig':gs,'g_nu':gnu,'g_mu':gmu,
                         **{'oq_%g'%t:oq[t] for t in TAUS},**{'oe_%g'%a:oe[a] for a in ALPHAS}})
lg("own-history rivals built for %d names %.0fs"%(len(own),time.time()-t0))

# ---------------------------------------------------------------- score every model on the test rows
TEb={ps:frame(ps,TST).dropna(subset=ZX) for ps in PEER}
base=TEb['peer_sector'][['permno','date','y','age']].copy()
for ps in PEER: base=base.merge(TEb[ps][['permno','date','logsig','peer_disp','peer_absz5']].rename(
        columns={'logsig':'ls_'+ps,'peer_disp':'pd_'+ps,'peer_absz5':'pa_'+ps}),on=['permno','date'],how='inner')
ow=pd.concat([d.assign(permno=p) for p,d in own.items()],ignore_index=True)
TE=base.merge(ow,on=['permno','date'],how='left').sort_values(['permno','date']).reset_index(drop=True)
Y=TE['y'].values; di,_=pd.factorize(pd.to_datetime(TE['date'].values),sort=True); di=np.asarray(di)
lg("test rows %d over %d names, %d dates %.0fs"%(len(TE),TE.permno.nunique(),len(set(di)),time.time()-t0))

RQ={}; VE={a:{} for a in ALPHAS}
for ps in PEER:
    sg=np.exp(TE['ls_'+ps].values)
    RQ[ps+'_emp']={t:sg*POOLQ[t] for t in TAUS}
    for a in ALPHAS: VE[a][ps+'_emp']=(sg*POOLQ[a],sg*POOLE[a])
    RQ[ps+'_t']={t:sg*stats.t.ppf(t,nu_hat)/tsc for t in TAUS}
    for a in ALPHAS: VE[a][ps+'_t']=(sg*stats.t.ppf(a,nu_hat)/tsc,sg*t_es(a,nu_hat)/tsc)
X=lambda ps: np.column_stack([TE['ls_'+ps].values,TE['pd_'+ps].values,TE['pa_'+ps].values])
for ps in ('peer_young','peer_sector'):
    sg=np.exp(TE['ls_'+ps].values); Xp=X(ps)
    ZQ={t:GB[t].predict(Xp) for t in TAUS}
    M=np.sort(np.stack([ZQ[t] for t in TAUS],axis=1),axis=1); ZQ={t:M[:,j] for j,t in enumerate(TAUS)}
    RQ[ps+'_flex']={t:sg*ZQ[t] for t in TAUS}
    for a in ALPHAS:
        st=np.sort(np.stack([SUB[a][j].predict(Xp) for j in range(20)],axis=1),axis=1)
        zq=np.maximum(ZQ[a],st[:,-1]); es=np.minimum(st.mean(axis=1),zq-1e-6)
        VE[a][ps+'_flex']=(sg*zq,sg*es)
osh=TE['own_sh'].values                                   # own scale, pooled shape: isolates WHICH stage pooling repairs
RQ['own_short_pool']={t:osh*POOLQ[t] for t in TAUS}
for a in ALPHAS: VE[a]['own_short_pool']=(osh*POOLQ[a],osh*POOLE[a])
oew=TE['own_ew'].values
RQ['own_ewma_emp']={t:oew*TE['oq_%g'%t].values for t in TAUS}
for a in ALPHAS: VE[a]['own_ewma_emp']=(oew*TE['oq_%g'%a].values,oew*TE['oe_%g'%a].values)
gs,gnu,gmu=TE['g_sig'].values,TE['g_nu'].values,TE['g_mu'].values; gt=np.sqrt(gnu/(gnu-2))
RQ['own_garch_t']={t:gmu+gs*stats.t.ppf(t,gnu)/gt for t in TAUS}
for a in ALPHAS: VE[a]['own_garch_t']=(gmu+gs*stats.t.ppf(a,gnu)/gt,gmu+gs*t_es(a,gnu)/gt)
MODELS=list(RQ)
PL={m:sum(pin(Y,RQ[m][t],t) for t in TAUS)/len(TAUS) for m in MODELS}
for m in MODELS: PL[m]=np.where(np.isfinite(np.sum(np.stack([RQ[m][t] for t in TAUS]),axis=0)),PL[m],np.nan)

# ---------------------------------------------------------------- metrics by age bucket
TE['bucket']=[ab(int(a)) for a in TE['age'].values]
REF='peer_young_emp'
def head(Lref,Lm,mask):
    mask=mask&np.isfinite(Lref)&np.isfinite(Lm)
    if mask.sum()<30: return None
    d=(Lref-Lm)[mask]; gg=pd.DataFrame({'d':d,'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'edge_pct':round(100*float(d.mean())/float(Lref[mask].mean()),3),'DM':nw_t(gg.values),'n':int(mask.sum())}
BUCK={}
for b in [ab(lo) for lo,_ in AGEB]:
    msk=(TE['bucket']==b).values; nb=int(msk.sum())
    if nb<50: continue
    row={'n_rows':nb,'n_names':int(TE.loc[msk,'permno'].nunique()),'availability':{},'mean_pinball':{},
         'edge_over_%s'%REF:{},'fz0':{}}
    for m in MODELS:
        av=float(np.mean(np.isfinite(PL[m][msk]))); row['availability'][m]=round(av,4)
        row['mean_pinball'][m]=round(float(np.nanmean(PL[m][msk])),5) if av>0 else None
        if m!=REF and av>0: row['edge_over_%s'%REF][m]=head(PL[REF],PL[m],msk)
    for a in ALPHAS:
        fa={}
        for m in MODELS:
            v,e=VE[a][m]; L=fz0(Y,v,e,a); ok=msk&np.isfinite(L)
            if ok.sum()<30: continue
            fa[m]={'meanFZ0':round(float(np.mean(L[ok])),5),'breach':round(float(np.mean((Y<=v)[ok])),4),
                   'avail':round(float(np.mean(np.isfinite(L[msk]))),4)}
        for m in list(fa):
            if m==REF: continue
            v0,e0_=VE[a][REF]; L0=fz0(Y,v0,e0_,a); v1,e1=VE[a][m]; L1=fz0(Y,v1,e1,a)
            mm=msk&np.isfinite(L0)&np.isfinite(L1)
            if mm.sum()>=30:
                gg=pd.DataFrame({'d':(L1-L0)[mm],'dt':di[mm]}).groupby('dt')['d'].mean()
                fa[m]['DM_vs_%s'%REF]={'mean_diff':round(float(np.mean((L1-L0)[mm])),5),'DM_t':nw_t(gg.values),'n':int(mm.sum())}
        row['fz0']['alpha_%g'%a]=fa
    BUCK[b]=row
    lg("[age %s] n=%d pinball: "%(b,nb)+json.dumps({m:row['mean_pinball'][m] for m in MODELS if row['mean_pinball'][m] is not None}))
ALLm=np.ones(len(Y),bool)
OUT={'note':'Cold start with pooled peer-group benchmarks. Three peer pools (mature same-sector TRAIN names; '
     'other TEST-cohort names, exact leave-one-out; all other names) each supply a point-in-time day-one scale '
     'from the pool median of EWMA(0.94) log-vols through t-1; shapes are the pooled TRAIN empirical quantiles, '
     'a Student-t with nu by MLE, and the HistGBM learner on day-one features only. Own-history rivals carry '
     'their availability and every head-to-head is on rows where both models exist. Reference row is '
     +REF+'. SIGN CONVENTION: edge_pct>0 and DM_t>0 both mean the RIVAL beats the reference, since the edge is '
     '(reference loss - rival loss)/reference loss.',
     'predictions_written_before_run':'P1 own-history edge at ages 15-60 falls below +2% against peer_young; '
     'P2 peer_young beats peer_sector by >3% of pinball at every age under 250 days; P3 the learner adds <+1% '
     'over pooled empirical quantiles below age 15; P4 pooled breach rates stay within 0.5pp of nominal from age 1.',
     'train_names':len(TRN),'train_rows':int(len(tr)),'test_names':int(TE.permno.nunique()),'test_rows':int(len(TE)),
     'cutoff':str(CUT.date()),'ewma_lambda':LAM,'pooled_nu':round(nu_hat,3),'pooled_q':{str(t):round(POOLQ[t],4) for t in TAUS},
     'learner_features':ZX,'age_buckets':[ab(lo) for lo,_ in AGEB],'by_age':BUCK,
     'all_ages':{'mean_pinball':{m:round(float(np.nanmean(PL[m])),5) for m in MODELS},
                 'availability':{m:round(float(np.mean(np.isfinite(PL[m]))),4) for m in MODELS},
                 'edge_over_%s'%REF:{m:head(PL[REF],PL[m],ALLm) for m in MODELS if m!=REF}}}
json.dump(OUT,open(os.path.join(P,"coldstart_peers_results.json"),"w"),indent=2)
lg("COLDSTARTDONE %.0fs -> coldstart_peers_results.json"%(time.time()-t0))
