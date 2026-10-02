# job_walkforward_bteg.py -- THE ANNUAL-REFIT WALK-FORWARD WITH BOTH STAGE-1 SCALES.
# The Stage-1 swap to Beta-t-EGARCH was decided on a frozen 60/40 fit. The paper recommends ANNUAL refits,
# and under that schedule the GARCH-t engine's AVERAGE edge disappears (committed: overall -0.22%, DM -0.59,
# top decile +2.12%, DM 4.41, from job_walkforward_hybrid.py). A referee will ask whether the new Stage 1
# survives the recommended process. This job runs the identical schedule for both scales side by side.
#
# SCHEDULE, unchanged from job_walkforward_hybrid.py: at each Jan-1 cutoff 2020-2024, every stage is
# re-estimated on expanding pre-cutoff data only -- the per-name GARCH-t AND the per-name Beta-t-EGARCH MLE,
# the residuals and mk63 under those refit parameters, the pooled GBM body (11 levels plus the 20-node
# sub-alpha grid at each regulatory level), and the pooled GPD tail -- and the next calendar year is
# predicted. No stage carries a stale estimate into test, for either scale.
# Reported per scale: the body-only pinball arm (comparable to the committed file) and the full engine
# (body/EVT minimum, rearranged) on pinball and on FZ0 at both levels, overall and by mk63 decile, with
# per-date Newey-West(10) DM; plus the engine_bteg versus engine_garch_t head-to-head on the same rows.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01):
#  P1. THE DECISION. Under annual refits the Beta-t-EGARCH engine keeps its FZ0 advantage over the GARCH-t
#      engine at both levels, DM > 2 in its favour, because the advantage comes from the filter's bounded
#      response to a shock and not from parameter staleness, which refitting is what removes. If P1 fails,
#      the swap does not survive the recommended schedule and must be reported as a frozen-fit result only.
#  P2. The flexible shape's top-decile edge over its OWN-scale parametric benchmark reproduces the frozen
#      pattern: around +2% under GARCH-t (committed frozen +2.12% under refits) and under +0.5% under
#      Beta-t-EGARCH, since the robust scale already absorbs most of the frontier.
#  P3. The overall (not top-decile) edge stays undetectable under refits for both scales, |DM| < 2, matching
#      the committed -0.22% at DM -0.59 for GARCH-t.
# EXTENDED 2026-10-01 for the R57 work order. Added under the same schedule: the bounded-news (bip_t) scale,
# a pooled-unconditional-residual-quantile row for every scale (<scale>_uncond; garch_t_uncond is FHS on this
# panel), and the Acerbi-Szekely and McNeil-Frey ES tests for the engines, bootstrapped over refit-period dates.
# The garch_t and bteg rows are unchanged by construction -- bip_t reuses the GARCH-t fit and nothing is added
# to the dropna set -- so this rerun is also a determinism check against walkforward_bteg_results.json.
# ADDED PREDICTIONS (2026-10-01, before the extended run):
# THE RUN THIS DECIDES (2026-10-01). Stage 1 stays GARCH-t per P1. On the FROZEN protocol the GARCH-t engine
# IS significantly beaten on FZ0 by the robust-filter rows -- bip_t (2.5%, DM 3.35), bteg (2.5%, 2.53),
# bteg_uncond (1% 1.75, 2.5% 2.32), bip_t_uncond (1% 2.02, 2.5% 1.85) per robust_engine_results.json -- so the
# paper owes a disclosure sentence. Its strength depends on whether that gap survives the annual refits the
# paper recommends, which is what these rows measure.
#  P4. THE DISCLOSURE. Under refits engine_garch_t is NOT significantly beaten OVERALL on FZ0 by bip_t,
#      bip_t_uncond, bteg or bteg_uncond at either level (all |DM| < 1.96), because refitting removes the
#      parameter staleness those filters were repairing -- the same mechanism that made P1 fail. The
#      disclosure can then read "beaten under frozen fits, tied under the annual refits we recommend".
#      If any of the four is still significantly ahead overall, the stronger disclosure is required.
#  P5. In the TOP mk63 DECILE the robust rows remain ahead of engine_garch_t even under refits, because that
#      decile is where post-shock overshoot concentrates and refitting removes staleness, not overshoot. A
#      split verdict -- tied overall, behind in the top decile -- is the expected outcome and is reportable.
#  P6. engine_bteg's ES tests under refits do not reject at either level (Acerbi-Szekely and McNeil-Frey
#      p > 0.05), matching its frozen-fit behaviour.
# Output: walkforward_bteg_results.json. Self-test: --synthetic.
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
SYN="--synthetic" in sys.argv; t0=time.time(); lg=lambda s:print(s,flush=True)
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']; SUBN=20; P0=0.025
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06,random_state=0)   # random_state set: the committed
                                                                       # walk-forward scripts leave it unset
SCALES=['garch_t','bip_t','bteg']
KBIP=9.0   # bounded-news cap, news=min(e^2, K sig^2); reuses the GARCH-t fit, no extra estimation
def pin(y,q,t): d=y-q; return np.where(d>=0,t*d,(t-1)*d)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    d=x-x.mean(); v=np.mean(d*d)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(d[k:]*d[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def fz0(r,v,e,a):
    v=np.minimum(v,-1e-8); e=np.minimum(e,v); hit=(r<=v).astype(float)
    return -(1.0/(a*e))*hit*(v-r)+v/e+np.log(-e)-1.0
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)
# ---- Beta-t-EGARCH, as in job_robust_engine.py
def _bteg_filter(y,mu,om,phi,kap,nu):
    n=len(y); lam=np.empty(n); lam[0]=om
    for k in range(n-1):
        e=(y[k]-mu)*math.exp(-lam[k]); e2=e*e
        lam[k+1]=om+phi*(lam[k]-om)+kap*((nu+1.0)*e2/(nu+e2)-1.0)
        if lam[k+1]>12.0: lam[k+1]=12.0
        elif lam[k+1]<-12.0: lam[k+1]=-12.0
    return lam
def _bteg_nll(th,y):
    mu,om=th[0],th[1]; phi=1.0/(1.0+math.exp(-th[2])); kap=math.exp(th[3]); nu=2.05+math.exp(th[4])
    if kap>2.0 or nu>200.0: return 1e12
    lam=_bteg_filter(y,mu,om,phi,kap,nu); e=(y-mu)*np.exp(-lam)
    c=math.lgamma(0.5*(nu+1.0))-math.lgamma(0.5*nu)-0.5*math.log(math.pi*nu)
    s=float(np.sum(c-lam-0.5*(nu+1.0)*np.log1p(e*e/nu)))
    return -s if np.isfinite(s) else 1e12
def fit_bteg(y,sp):
    ytr=np.asarray(y[:sp],float); mu0=float(np.mean(ytr)); sd0=max(float(np.std(ytr)),1e-6); best=None
    for nu0,kap0,phi0 in ((6.0,0.08,0.98),(4.0,0.15,0.95)):
        om0=math.log(sd0)-0.5*math.log(nu0/(nu0-2.0))
        x0=np.array([mu0,om0,math.log(phi0/(1.0-phi0)),math.log(kap0),math.log(nu0-2.05)])
        try: r=optimize.minimize(_bteg_nll,x0,args=(ytr,),method='Nelder-Mead',
                                 options={'maxiter':3000,'maxfev':4500,'xatol':1e-5,'fatol':1e-5})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    if best is None: raise RuntimeError("bteg MLE failed")
    mu,om=float(best.x[0]),float(best.x[1]); phi=1.0/(1.0+math.exp(-best.x[2]))
    kap=math.exp(best.x[3]); nu=2.05+math.exp(best.x[4])
    lam=_bteg_filter(np.asarray(y,float),mu,om,phi,kap,nu)
    tsc=math.sqrt(nu/(nu-2.0)) if nu>2.0 else 1.0
    return np.exp(lam)*tsc, mu, nu

if SYN:
    rng=np.random.default_rng(3); recs=[]; dates=pd.bdate_range("2014-01-02",periods=2770)
    for pn in range(12):
        om,al,be,nu=0.04,0.08,0.90,5.0+rng.uniform(0,3); s2=np.empty(2770); e=np.empty(2770); s2[0]=om/(1-al-be)
        for k in range(2770):
            if k: s2[k]=om+al*e[k-1]**2+be*s2[k-1]
            zz=stats.t.rvs(nu,random_state=rng)/math.sqrt(nu/(nu-2))
            if rng.uniform()<0.02: zz-=rng.exponential(2.5)
            e[k]=math.sqrt(s2[k])*zz
        recs.append(pd.DataFrame({'permno':pn,'date':dates,'ret':(0.02+e)/100.0}))
    rr=pd.concat(recs); NMAX=12
else:
    rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'}); NMAX=200
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:NMAX]
series={pn:(g['ret'].values.astype(float),pd.to_datetime(g['date'].values)) for pn,g in
        ((pn,rr[rr.permno==pn].sort_values('date')) for pn in names)}
TESTYEARS=[2020,2021,2022,2023,2024]
def feats(y,sig,mu,dts):
    z=(y-mu)/np.maximum(sig,1e-6); df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1)
    df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1)
    df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)
    return df

PIECES=[]; NFAIL=0
for ty in TESTYEARS:
    cut=pd.Timestamp("%d-01-01"%ty); nxt=pd.Timestamp("%d-01-01"%(ty+1))
    TR={f:[] for f in SCALES}; TEyr=[]
    for pn,(y,dts) in series.items():
        sp=int(np.searchsorted(dts,cut)); te_end=int(np.searchsorted(dts,nxt))
        if sp<(300 if SYN else 750) or te_end-sp<30: continue
        n=te_end; D={}
        try:
            res=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
            pp=res.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
            e0=y[:n]-mu; s2=np.empty(n); s2[0]=np.var(y[:sp])
            for k in range(1,n): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
            d1=feats(y[:n],np.sqrt(s2),mu,dts[:n]); d1['mu']=mu; d1['nu']=nu
            d1['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0; D['garch_t']=d1
            s2b=np.empty(n); s2b[0]=np.var(y[:sp])
            for k in range(1,n): s2b[k]=max(om+al*min(e0[k-1]**2,KBIP*s2b[k-1])+be*s2b[k-1],1e-8)
            db=feats(y[:n],np.sqrt(s2b),mu,dts[:n]); db['mu']=mu; db['nu']=nu
            db['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0; D['bip_t']=db
            sd,mb,nb=fit_bteg(y[:n],sp)               # MLE on pre-cutoff rows only; filtered forward
            d2=feats(y[:n],sd,mb,dts[:n]); d2['mu']=mb; d2['nu']=nb
            d2['tsc']=math.sqrt(nb/(nb-2)) if nb>2 else 1.0; D['bteg']=d2
        except Exception as ex:
            NFAIL+=1; continue
        base=D['garch_t'].copy(); base['idx']=np.arange(n)
        for f in ('bteg','bip_t'):
            for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__'+f]=D[f][c].values
        # dropna set deliberately unchanged (garch_t + bteg): bip_t shares the same rolling windows,
        # so the row set is identical to the first run and the existing rows must reproduce exactly.
        ok=base.dropna(subset=ZX+['mk63']+[c+'__bteg' for c in ZX+['mk63']])
        trn=ok[ok['idx']<sp]; tst=ok[ok['idx']>=sp]
        if len(tst)<10 or len(trn)<200: continue
        for f in SCALES:
            sfx='' if f=='garch_t' else '__'+f
            TR[f].append(trn[[c+sfx for c in ZX+['z']]].rename(columns=lambda c:c.split('__')[0]))
        t2=tst.copy(); t2['permno']=pn; TEyr.append(t2)
    TEc=pd.concat(TEyr).reset_index(drop=True); TRc={f:pd.concat(TR[f]) for f in SCALES}
    lg("cut %s: %d names, %dk train rows, %d test rows %.0fs"%(cut.date(),TEc.permno.nunique(),len(TRc['garch_t'])//1000,len(TEc),time.time()-t0))
    Y=TEc['y'].values; rec={'y':Y,'date':TEc['date'].values,'permno':TEc['permno'].values}
    for f in SCALES:
        sfx='' if f=='garch_t' else '__'+f
        X=TEc[[c+sfx for c in ZX]].values; ztr=TRc[f]['z'].values
        uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]; xi,_,bt=stats.genpareto.fit(exc,floc=0.0)
        evt=lambda tau: (uu-(bt/xi)*((tau/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-bt*math.log(P0/tau))
        ZQ={t:HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TRc[f][ZX].values,ztr).predict(X) for t in TAUS}
        MU=TEc['mu'+sfx].values; SIG=TEc['sig'+sfx].values; NU=TEc['nu'+sfx].values; TSC=TEc['tsc'+sfx].values
        eng={t:(np.minimum(ZQ[t],evt(t)) if t<=P0 else ZQ[t]) for t in TAUS}
        E=np.sort(np.stack([eng[t] for t in TAUS],axis=1),axis=1)
        plp=np.mean([pin(Y,MU+SIG*stats.t.ppf(t,NU)/TSC,t) for t in TAUS],axis=0)
        plb=np.mean([pin(Y,MU+SIG*ZQ[t],t) for t in TAUS],axis=0)
        ple=np.mean([pin(Y,MU+SIG*E[:,j],t) for j,t in enumerate(TAUS)],axis=0)
        qu={t:float(np.quantile(ztr,t)) for t in TAUS}
        plu=np.mean([pin(Y,MU+SIG*qu[t],t) for t in TAUS],axis=0)
        rec['pl_param_'+f]=plp; rec['pl_body_'+f]=plb; rec['pl_engine_'+f]=ple
        rec['pl_uncond_'+f]=plu; rec['mk_'+f]=TEc['mk63'+sfx].values
        rec['z_'+f]=(Y-MU)/np.maximum(SIG,1e-12)
        for a in ALPHAS:
            SUB=[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,**HGB)
                 .fit(TRc[f][ZX].values,ztr).predict(X) for j in range(SUBN)]
            st=np.sort(np.stack([np.minimum(SUB[j],evt(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
            zq=np.maximum(np.minimum(ZQ[a],evt(a)),st[:,-1]); es=np.minimum(st.mean(axis=1),zq-1e-6)
            rec['fz_engine_%s_%g'%(f,a)]=fz0(Y,MU+SIG*zq,MU+SIG*es,a)
            rec['fz_param_%s_%g'%(f,a)]=fz0(Y,MU+SIG*stats.t.ppf(a,NU)/TSC,MU+SIG*t_es(a,NU)/TSC,a)
            qa=float(np.quantile(ztr,a)); ea=float(np.mean(ztr[ztr<=qa]))
            rec['fz_uncond_%s_%g'%(f,a)]=fz0(Y,MU+SIG*qa,MU+SIG*ea,a)
            rec['br_engine_%s_%g'%(f,a)]=(Y<=MU+SIG*zq).astype(float)
            rec['zq_engine_%s_%g'%(f,a)]=zq; rec['zes_engine_%s_%g'%(f,a)]=es
        lg("   %s done %.0fs"%(f,time.time()-t0))
    PIECES.append(pd.DataFrame(rec))
ALL=pd.concat(PIECES).reset_index(drop=True)
di,_=pd.factorize(pd.to_datetime(ALL['date'].values),sort=True); di=np.asarray(di)
lg("walk-forward panel: %d rows, %d dates, %d fits dropped %.0fs"%(len(ALL),len(set(di)),NFAIL,time.time()-t0))
def edge(ref,alt,mask):
    mask=mask&np.isfinite(ref)&np.isfinite(alt)
    if mask.sum()<100: return None
    d=(ref-alt)[mask]; gg=pd.DataFrame({'d':d,'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'edge_pct':round(100*float(d.mean())/float(ref[mask].mean()),3),'DM':nw_t(gg.values),'n':int(mask.sum())}
def dm(a1,a2,mask):
    mask=mask&np.isfinite(a1)&np.isfinite(a2)
    if mask.sum()<100: return None
    gg=pd.DataFrame({'d':(a1-a2)[mask],'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'mean_diff':round(float(np.mean((a1-a2)[mask])),5),'DM_t':nw_t(gg.values),'n':int(mask.sum())}
ALLM=np.ones(len(ALL),bool)
def dec10(x):
    r=np.full(len(x),-1); m=np.isfinite(x); r[m]=pd.qcut(pd.Series(x[m]),10,labels=False,duplicates='drop').values+1; return r
OUT={'note':'Annual-refit walk-forward with both Stage-1 scales on the same rows. At each Jan-1 cutoff '
     '2020-2024 the per-name GARCH-t and Beta-t-EGARCH are both re-estimated on expanding pre-cutoff data, '
     'residuals, features, mk63, the pooled GBM body, the 20-node sub-alpha grid and the pooled GPD tail are '
     'all rebuilt, and the next calendar year is predicted. Comparable to walkforward_hybrid_results.json, '
     'whose learner is the body-only arm under garch_t.',
     'predictions_written_before_run':'P1 (decision) engine_bteg keeps its FZ0 edge over engine_garch_t at '
     'both levels under refits, DM>2; P2 top-decile flexible edge over own-scale parametric near +2% under '
     'garch_t and under +0.5% under bteg; P3 overall edge undetectable for both, |DM|<2.',
     'synthetic':SYN,'test_years':TESTYEARS,'n_test':int(len(ALL)),'n_dates':int(len(set(di))),'n_dropped':NFAIL,
     'per_scale':{},'engine_head_to_head':{},'fz0':{}}
for f in SCALES:
    rk=dec10(ALL['mk_'+f].values); reg={'overall':ALLM,'top_decile':rk==10,'deciles_1to9':(rk>=1)&(rk<=9)}
    OUT['per_scale'][f]={
      'mean_pinball':{k:round(float(np.nanmean(ALL['pl_%s_%s'%(k,f)].values)),5) for k in ('param','body','engine')},
      'body_over_own_parametric':{r:edge(ALL['pl_param_'+f].values,ALL['pl_body_'+f].values,m) for r,m in reg.items()},
      'engine_over_own_parametric':{r:edge(ALL['pl_param_'+f].values,ALL['pl_engine_'+f].values,m) for r,m in reg.items()},
      'decile_profile_engine_over_own_parametric':{int(d0):edge(ALL['pl_param_'+f].values,ALL['pl_engine_'+f].values,rk==d0) for d0 in range(1,11)}}
    lg("[%s] engine over own parametric: "%f+json.dumps(OUT['per_scale'][f]['engine_over_own_parametric']))
rk=dec10(ALL['mk_garch_t'].values); reg={'overall':ALLM,'top_decile':rk==10,'deciles_1to9':(rk>=1)&(rk<=9)}
OUT['engine_head_to_head']={'pinball_bteg_over_garch_t':{r:edge(ALL['pl_engine_garch_t'].values,ALL['pl_engine_bteg'].values,m) for r,m in reg.items()}}
# stationary bootstrap over the refit-period dates, for the ES tests
BB=4000; MBLK=10.0; rngb=np.random.default_rng(20260906)
Dn=len(set(di)); IDX=np.empty((BB,Dn),dtype=np.int64)
_st=rngb.integers(0,Dn,size=(BB,Dn)); _nb=rngb.random((BB,Dn))<(1.0/MBLK); IDX[:,0]=_st[:,0]
for t in range(1,Dn): IDX[:,t]=np.where(_nb[:,t],_st[:,t],(IDX[:,t-1]+1)%Dn)
def dagg(v): out=np.zeros(Dn); np.add.at(out,di,np.asarray(v,float)); return out
def bratio(S,C):
    cc=C[IDX].sum(axis=1); ss=S[IDX].sum(axis=1)
    return np.where(cc>1e-12,ss/np.maximum(cc,1e-12),np.nan)
def z2_test(z,zq,zes,a):
    t=(z<=zq).astype(float)*z/(a*zes); obs=1.0-t.sum()/len(t)
    se=np.nanstd(1.0-bratio(dagg(t),dagg(np.ones(len(t)))))
    return float(obs),(float(2*(1-stats.norm.cdf(abs(obs)/se))) if se>0 else float('nan'))
def mf_test(z,zq,zes):
    m=z<=zq; erm=np.where(m,z-zes,0.0); nb=int(m.sum())
    if nb<10: return float('nan'),float('nan'),nb
    obs=erm.sum()/nb; se=np.nanstd(bratio(dagg(erm),dagg(m.astype(float))))
    return float(obs),(float(2*(1-stats.norm.cdf(abs(obs)/se))) if se>0 else float('nan')),nb
ROWKEY=[('engine_%s','fz_engine_%s_%g'),('body_%s','fz_body_%s_%g') ,('param_%s','fz_param_%s_%g'),('uncond_%s','fz_uncond_%s_%g')]
for a in ALPHAS:
    d={}; L={}
    for f in SCALES:
        for nm,ck in ROWKEY:
            c=ck%(f,a) if '%g' in ck else ck%f
            if c not in ALL.columns: continue
            L[nm%f]=ALL[c].values; d[nm%f]={'meanFZ0':round(float(np.nanmean(ALL[c].values)),5)}
        d['engine_'+f]['breach']=round(float(np.nanmean(ALL['br_engine_%s_%g'%(f,a)].values)),4)
    # FZ0 DM of every row against each engine (DM_t>0 => that row is worse than the named engine)
    rkg=dec10(ALL['mk_garch_t'].values); REG={'overall':ALLM,'top_mk63_decile':rkg==10}
    for ref in ('engine_garch_t','engine_bteg'):
        for m in L:
            if m==ref: continue
            d[m]['vs_'+ref]={r:dm(L[m],L[ref],msk) for r,msk in REG.items()}
    # the two head-to-heads the work order names explicitly
    d['DM_engine_garch_t_vs_engine_bteg']={r:dm(L['engine_garch_t'],L['engine_bteg'],msk) for r,msk in REG.items()}
    # exception + ES tests for each engine under refits
    for f in SCALES:
        bl=ALL['br_engine_%s_%g'%(f,a)].values
        fdf=pd.DataFrame({'b':bl,'dt':di}).groupby('dt')['b'].mean()
        t_=nw_t(fdf.values-a)
        d['engine_'+f]['uc_clustered_t']=t_
        d['engine_'+f]['uc_clustered_p']=round(float(2*(1-stats.norm.cdf(abs(t_)))),4) if t_ is not None else None
        z=ALL['z_'+f].values; zq=ALL['zq_engine_%s_%g'%(f,a)].values; zes=ALL['zes_engine_%s_%g'%(f,a)].values
        ok=np.isfinite(z)&np.isfinite(zq)&np.isfinite(zes)&(zes<0)
        if ok.all():
            z2,z2p=z2_test(z,zq,zes,a); mf,mfp,nb=mf_test(z,zq,zes)
            d['engine_'+f].update({'AS_Z2':round(z2,4),'AS_Z2_p':round(z2p,4),'MF_exres':round(mf,4),'MF_p':round(mfp,4),'n_breach':int(nb)})
    OUT['fz0'][str(a)]=d
    lg("FZ0 a=%s: "%a+json.dumps({k:(v['meanFZ0'] if isinstance(v,dict) and 'meanFZ0' in v else v) for k,v in d.items()}))
fn="walkforward_bteg_results%s.json"%("_synthetic" if SYN else "")
json.dump(OUT,open(os.path.join(P,fn),"w"),indent=2)
lg("WALKFORWARDBTEGDONE %.0fs -> %s"%(time.time()-t0,fn))
