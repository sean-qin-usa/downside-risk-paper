# job_scale_gate.py -- A SCORE-GATED STAGE-1 SCALE, UNDER ANNUAL REFITS.
# The robust filter and the GARCH-t filter are indistinguishable on average under refits but not in the top
# misspecification decile, where the robust one is well ahead (DM 5.37 and 7.25, walkforward_bteg_results.json).
# That is an argument for using the score to CHOOSE the scale rather than fixing it: run the paper's GARCH-t
# engine in ordinary states and the Beta-t-EGARCH engine in the states the score flags.
# GATE: use engine_garch_t when the previous day's mk63 is below a threshold, engine_bteg at or above it. The
# threshold is the 90th percentile of mk63 over all rows BEFORE the first test cutoff (2020-01-01), computed
# once and held fixed for all five test years, so nothing about it is fitted in the test era. mk63 is already
# lagged one day in the feature construction, so the gate is causal.
# Compared against always-GARCH-t engine, always-engine_bteg and always-param_bteg on FZ0 at both levels, with
# breach, date-clustered unconditional coverage and the per-name exception tests.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01), decisions included:
#  P1. The gate is at least as good as EVERY fixed choice overall on FZ0 at both levels (no fixed rule beats it
#      with DM > 1.96), and beats always-GARCH-t in the top mk63 decile with DM > 2.
#  P2. The gate's overall FZ0 is within noise of always-engine_bteg (|DM| < 2), because the two differ only on
#      the 10% of rows above the threshold -- the gate buys the top-decile gain without the rest of the switch.
#  P3. The gate's breach rate is between the two fixed engines' at both levels and its date-clustered test
#      passes, since it is a row-wise mixture of two rows that each pass.
#  DECISION: the gate is worth a paragraph only if P1 holds. If a fixed choice beats it overall, the score
#      does not carry enough information to select the scale and the result is reported as negative.
# Writes the per-row forecast panel to scale_gate_panel.parquet so later jobs reuse it instead of refitting.
# Output: scale_gate_results.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from es_integral import converged_es, levels_and_body
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']; SUBN=20; P0=0.025
ES_LEGACY={}   # superseded 20-node ES, kept so the paper can print old beside new
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06,random_state=0); TESTYEARS=[2020,2021,2022,2023,2024]
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
def _bf(y,mu,om,phi,kap,nu):
    n=len(y); lam=np.empty(n); lam[0]=om
    for k in range(n-1):
        e=(y[k]-mu)*math.exp(-lam[k]); e2=e*e
        lam[k+1]=min(max(om+phi*(lam[k]-om)+kap*((nu+1.0)*e2/(nu+e2)-1.0),-12.0),12.0)
    return lam
def _bn(th,y):
    mu,om=th[0],th[1]; phi=1.0/(1.0+math.exp(-th[2])); kap=math.exp(th[3]); nu=2.05+math.exp(th[4])
    if kap>2.0 or nu>200.0: return 1e12
    lam=_bf(y,mu,om,phi,kap,nu); e=(y-mu)*np.exp(-lam)
    c=math.lgamma(0.5*(nu+1.0))-math.lgamma(0.5*nu)-0.5*math.log(math.pi*nu)
    v=float(np.sum(c-lam-0.5*(nu+1.0)*np.log1p(e*e/nu)))
    return -v if np.isfinite(v) else 1e12
def fit_bteg(y,sp):
    ytr=np.asarray(y[:sp],float); mu0=float(np.mean(ytr)); sd0=max(float(np.std(ytr)),1e-6); best=None
    for nu0,kap0,phi0 in ((6.0,0.08,0.98),(4.0,0.15,0.95)):
        om0=math.log(sd0)-0.5*math.log(nu0/(nu0-2.0))
        x0=np.array([mu0,om0,math.log(phi0/(1.0-phi0)),math.log(kap0),math.log(nu0-2.05)])
        try: r=optimize.minimize(_bn,x0,args=(ytr,),method='Nelder-Mead',options={'maxiter':3000,'maxfev':4500,'xatol':1e-5,'fatol':1e-5})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    if best is None: raise RuntimeError("bteg MLE failed")
    mu,om=float(best.x[0]),float(best.x[1]); phi=1.0/(1.0+math.exp(-best.x[2]))
    kap=math.exp(best.x[3]); nu=2.05+math.exp(best.x[4])
    lam=_bf(np.asarray(y,float),mu,om,phi,kap,nu); tsc=math.sqrt(nu/(nu-2.0)) if nu>2.0 else 1.0
    return np.exp(lam)*tsc,mu,nu,tsc
def feats(y,sig,mu,dts):
    z=(y-mu)/np.maximum(sig,1e-6); df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1)
    df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1)
    df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)
    return df
def stage23(TRz,X,tag='?'):
    ztr=TRz['z'].values; uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]
    xi,_,bt=stats.genpareto.fit(exc,floc=0.0)
    evt=lambda t:(uu-(bt/xi)*((t/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-bt*math.log(P0/t))
    ZQ={t:HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TRz[ZX].values,ztr).predict(X) for t in TAUS}
    VE={}
    for a in ALPHAS:
        SUB=[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,**HGB).fit(TRz[ZX].values,ztr).predict(X) for j in range(SUBN)]
        st=np.sort(np.stack([np.minimum(SUB[j],evt(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
        zq=np.maximum(np.minimum(ZQ[a],evt(a)),st[:,-1])
        # CONVERGED ES; VaR untouched, so the gate's routing and every breach statistic are unchanged.
        _lev,_QB=levels_and_body(a,SUB,lambda u:HistGradientBoostingRegressor(loss='quantile',quantile=u,**HGB).fit(TRz[ZX].values,ztr).predict(X),subn=SUBN)
        _es20=np.minimum(st.mean(axis=1),zq-1e-6)
        _esC=converged_es(a,_lev,_QB,evt,(uu,float(bt),float(xi),P0),M=2000,var_z=zq)
        ES_LEGACY.setdefault(str(a),{})[tag]={'es_20node':round(float(np.nanmean(_es20)),5),'es_converged':round(float(np.nanmean(_esC)),5),'ratio':round(float(np.nanmean(_esC/_es20)),5)}
        VE[a]=(zq,_esC)
    return VE
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:200]
SER={pn:(g['ret'].values.astype(float),pd.to_datetime(g['date'].values)) for pn,g in ((p,rr[rr.permno==p].sort_values('date')) for p in names)}
PIECES=[]; PRE=[]
for ty in TESTYEARS:
    cut=pd.Timestamp("%d-01-01"%ty); nxt=pd.Timestamp("%d-01-01"%(ty+1)); rows=[]; TRg=[]; TRb=[]
    for pn in names:
        y,dts=SER[pn]; sp=int(np.searchsorted(dts,cut)); te=int(np.searchsorted(dts,nxt))
        if sp<750 or te-sp<30: continue
        try:
            r1=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
            pp=r1.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
            e0=y[:te]-mu; s2=np.empty(te); s2[0]=np.var(y[:sp])
            for k in range(1,te): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
            d1=feats(y[:te],np.sqrt(s2),mu,dts[:te]); d1['mu']=mu; d1['nu']=nu; d1['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0
            sd,mb,nb,tb=fit_bteg(y[:te],sp)
            d2=feats(y[:te],sd,mb,dts[:te]); d2['mu']=mb; d2['nu']=nb; d2['tsc']=tb
        except Exception: continue
        base=d1.copy(); base['idx']=np.arange(te)
        for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__b']=d2[c].values
        ok=base.dropna(subset=ZX+['mk63']+[c+'__b' for c in ZX+['mk63']])
        trn=ok[ok['idx']<sp]; tst=ok[ok['idx']>=sp]
        if len(tst)<10 or len(trn)<200: continue
        TRg.append(trn[ZX+['z']]); TRb.append(trn[[c+'__b' for c in ZX+['z']]].rename(columns=lambda c:c[:-3]))
        t2=tst.copy(); t2['permno']=pn; rows.append(t2)
        if ty==TESTYEARS[0]: PRE.append(trn['mk63'].values)   # pre-2020 rows only, for the frozen threshold
    TEc=pd.concat(rows).reset_index(drop=True); TRgc=pd.concat(TRg); TRbc=pd.concat(TRb)
    lg("cut %s: %d names %d rows %.0fs"%(cut.date(),TEc.permno.nunique(),len(TEc),time.time()-t0))
    rec={'y':TEc['y'].values,'date':TEc['date'].values,'permno':TEc['permno'].values,'mk':TEc['mk63'].values}
    for f,TRz,sfx in (('garch_t',TRgc,''),('bteg',TRbc,'__b')):
        VE=stage23(TRz,TEc[[c+sfx for c in ZX]].values,tag=f)
        MU=TEc['mu'+sfx].values; SIG=TEc['sig'+sfx].values; NU=TEc['nu'+sfx].values; TSC=TEc['tsc'+sfx].values
        for a in ALPHAS:
            zq,es=VE[a]
            rec['v_engine_%s_%g'%(f,a)]=MU+SIG*zq; rec['e_engine_%s_%g'%(f,a)]=MU+SIG*es
            rec['v_param_%s_%g'%(f,a)]=MU+SIG*stats.t.ppf(a,NU)/TSC; rec['e_param_%s_%g'%(f,a)]=MU+SIG*t_es(a,NU)/TSC
        rec['mu_'+f]=MU; rec['sig_'+f]=SIG
        lg("   %s %.0fs"%(f,time.time()-t0))
    PIECES.append(pd.DataFrame(rec))
ALL=pd.concat(PIECES).reset_index(drop=True)
THR=float(np.nanquantile(np.concatenate(PRE),0.90))     # frozen before the test era
Y=ALL['y'].values; di,_=pd.factorize(pd.to_datetime(ALL['date'].values),sort=True); di=np.asarray(di)
gate=(ALL['mk'].values>=THR)
lg("panel %d rows %d dates | frozen mk63 90th pct = %.3f | gate fires on %.1f%% of rows %.0fs"%(
   len(Y),len(set(di)),THR,100*np.mean(gate),time.time()-t0))
# saved as gzipped CSV, not parquet: pyarrow/fastparquet are not installed on the compute host and the
# parquet write failed silently in the first run, which blocked the independent recomputation.
ALL.to_csv(os.path.join(P,"scale_gate_panel.csv.gz"),index=False,compression="gzip")
lg("  per-row forecast panel written: scale_gate_panel.csv.gz (%d rows)"%len(ALL))
rk=np.full(len(Y),-1); fm=np.isfinite(ALL['mk'].values)
rk[fm]=pd.qcut(pd.Series(ALL['mk'].values[fm]),10,labels=False,duplicates='drop').values+1
REG={'overall':np.ones(len(Y),bool),'top_mk63_decile':rk==10,'deciles_1to9':(rk>=1)&(rk<=9)}
def dmv(a1,a2,msk):
    msk=msk&np.isfinite(a1)&np.isfinite(a2)
    if msk.sum()<100: return None
    g=pd.DataFrame({'d':(a1-a2)[msk],'dt':di[msk]}).groupby('dt')['d'].mean()
    return {'mean_diff':round(float(np.mean((a1-a2)[msk])),5),'DM_t':nw_t(g.values),'n':int(msk.sum())}
OUT={'es_convention':{'engine_rows':'converged integral (es_integral.converged_es, M=2000, 60 fitted body levels): body interpolated on [a/40,a], pooled GPD exact at every node, sub-floor region in closed form. VaR is the committed construction and is unchanged, so no VaR-only statistic can move.','parametric_rows':'closed form, unaffected','superseded_20node_values':ES_LEGACY},
     'note':'Score-gated Stage-1 scale under annual refits: the GARCH-t engine below a frozen mk63 threshold, '
     'the Beta-t-EGARCH engine at or above it. Threshold = the 90th percentile of mk63 over pre-2020 rows, '
     'fixed before the test era. DM_t>0 means the row is WORSE than the named reference.',
     'predictions_written_before_run':'P1 the gate is not beaten overall by any fixed choice (no DM>1.96 '
     'against it) and beats always-GARCH-t in the top decile with DM>2; P2 the gate is within noise of '
     'always-engine_bteg overall; P3 its breach sits between the two engines and the date-clustered test passes.',
     'mk63_threshold_frozen':round(THR,4),'gate_fire_rate':round(float(np.mean(gate)),4),
     'n_test':int(len(Y)),'n_dates':int(len(set(di))),'per_alpha':{}}
for a in ALPHAS:
    V={}
    for f in ('garch_t','bteg'):
        V['engine_'+f]=(ALL['v_engine_%s_%g'%(f,a)].values,ALL['e_engine_%s_%g'%(f,a)].values)
        V['param_'+f]=(ALL['v_param_%s_%g'%(f,a)].values,ALL['e_param_%s_%g'%(f,a)].values)
    V['gate']=(np.where(gate,V['engine_bteg'][0],V['engine_garch_t'][0]),
               np.where(gate,V['engine_bteg'][1],V['engine_garch_t'][1]))
    L={m:fz0(Y,*V[m],a) for m in V}; A={}
    for m in V:
        v,_=V[m]; b=(Y<=v).astype(int)
        fdf=pd.DataFrame({'b':b,'dt':di}).groupby('dt')['b'].mean(); dct=nw_t(fdf.values-a)
        A[m]={'meanFZ0':round(float(np.nanmean(L[m])),5),'breach':round(float(b.mean()),4),
              'dateclustered_NW_t':dct,'dateclustered_PASS':(abs(dct)<1.96) if dct is not None else None,
              'uc_clustered_p':round(float(2*(1-stats.norm.cdf(abs(dct)))),4) if dct is not None else None}
    for m in V:
        if m!='gate': A[m]['vs_gate']={r:dmv(L[m],L['gate'],msk) for r,msk in REG.items()}
    OUT['per_alpha'][str(a)]=A
    lg("a=%g: "%a+json.dumps({m:A[m]['meanFZ0'] for m in A}))
json.dump(OUT,open(os.path.join(P,"scale_gate_results.json"),"w"),indent=2)
lg("SCALEGATEDONE %.0fs"%(time.time()-t0))
