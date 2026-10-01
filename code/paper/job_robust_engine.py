# job_robust_engine.py -- A ROBUST STAGE-1 SCALE FOR THE ENGINE. Beta-t-EGARCH (Harvey-Chakravarty
# score-driven volatility, the t-GAS scale) placed beside the paper's GARCH(1,1)-t and the bounded-news
# (BIP) GARCH of job_frontier_robust.py, in BOTH roles: as a standalone parametric benchmark, and as the
# Stage-1 scale under the flexible shape. Every downstream stage is re-estimated on each filter's own
# residuals (features, mk63 score, pooled HistGBM body, pooled GPD tail at p0=0.025, conformal shift),
# and all rows are scored on one common set of test rows.
#
# WHY THIS FILTER. Section 4 attributes most of the top-decile edge to the standard filter's post-outlier
# variance overstatement: a bounded-news GARCH cuts +2.98% to +0.3-0.5% (DM 2.5-4.7). Beta-t-EGARCH is the
# principled version of that bound. Its scale is driven by the conditional score of the Student-t
# likelihood, u_t = (nu+1)e_t^2/(nu+e_t^2) - 1, which is a martingale difference bounded in [-1, nu]: one
# shock can move log-scale by at most kappa*nu however large it is, where GARCH's e^2 news term is
# unbounded. It is robust by construction rather than by a hand-set cap, and it is a maximum-likelihood
# model rather than a filter with borrowed parameters, so it answers the cap-tuning objection to the BIP row.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01):
#  P1. As a standalone benchmark on all rows, Beta-t-EGARCH beats GARCH-t on mean FZ0 at both levels, by
#      0.005-0.03 (DM 1.5-5 in its favour), because the bounded score avoids the post-shock overstatement.
#  P2. Under the flexible shape the top-mk63-decile edge over the SAME-scale parametric benchmark collapses
#      from the paper's +2.98% to the bounded-news range, +0.2% to +0.9% with DM 2-5 -- i.e. Beta-t-EGARCH
#      reproduces the BIP decomposition from likelihood rather than from a cap. If instead the edge stays
#      above +1.5%, the Section 4 attribution is specific to the cap and must be restated.
#  P3. engine_bteg has a lower mean FZ0 than the paper's engine at both levels (the robust scale helps the
#      flexible shape too), but by less than P1's standalone gap, since the learner already repairs part of
#      the scale error through its logsig and dispersion features.
#  P4. The top-decile row sets overlap less than 0.75 across the three sorts: a robust scale relabels which
#      days look misspecified, which is the mechanism, not a nuisance.
#
# Scale rows (each an SD; parametric quantile is always mu + sigma_t * t_q(nu)/tsc(nu)):
#   garch_t  GARCH(1,1)-t                            engine / body   the paper's Stage 1
#   bip_t    bounded-news GARCH, news=min(e^2,K*s^2) engine_bip      K=9 (3-sigma cap), params from the GARCH-t fit
#   bteg     Beta-t-EGARCH, own MLE per name         engine_bteg     THE NEW SCALE
#
# Panels:  --panel canon200 (default)  the 200-name CRSP rows of Table 1, full metric battery
#          --panel holdout             the frozen pre-committed 2000-2013 era (holdout_panel_2000_2013.csv).
#                                      Nothing is tuned here: the Beta-t-EGARCH parameters are fitted per name by
#                                      MLE inside each name's own training window within the era, and every
#                                      hyperparameter is the one frozen on the design era.
#          --panel taq30               the 30 large-cap TAQ rows of the scale-shape table, with the proper
#                                      HHS Realized GARCH scale and the 20-node ES integral, so the
#                                      Beta-t-EGARCH scale enters that table as a third scale on same rows.
# Self-test: --synthetic (24 simulated names, same code path).
# Output: robust_engine_results.json (or _taq30 / _synthetic suffix).
import os, sys, json, time, math, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
SYN="--synthetic" in sys.argv
PANEL=sys.argv[sys.argv.index("--panel")+1] if "--panel" in sys.argv else "canon200"
assert PANEL in ("canon200","taq30","holdout"), "unknown --panel %s"%PANEL
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']; P0_ENGINE=0.025; SUBN=20; KBIP=9.0
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06)

# ============================================================ Beta-t-EGARCH (t-GAS) scale, per-name MLE
# y_t = mu + exp(lam_t) eps_t,  eps_t ~ t_nu (unit SCALE, not unit variance)
# lam_{t+1} = om + phi (lam_t - om) + kap u_t,  u_t = (nu+1) e_t^2/(nu+e_t^2) - 1,  e_t=(y_t-mu)/exp(lam_t)
# u_t is the score of the t log-likelihood wrt lam: mean zero, bounded in [-1, nu]  ->  robust by design.
def _bteg_filter(y,mu,om,phi,kap,nu,n=None):
    n=len(y) if n is None else n; lam=np.empty(n); lam[0]=om
    for k in range(n-1):
        e=(y[k]-mu)*math.exp(-lam[k]); e2=e*e
        u=(nu+1.0)*e2/(nu+e2)-1.0
        lam[k+1]=om+phi*(lam[k]-om)+kap*u
        if lam[k+1]>12.0: lam[k+1]=12.0
        elif lam[k+1]<-12.0: lam[k+1]=-12.0
    return lam
def _bteg_nll(th,y):
    mu,om=th[0],th[1]; phi=1.0/(1.0+math.exp(-th[2])); kap=math.exp(th[3]); nu=2.05+math.exp(th[4])
    if kap>2.0 or nu>200.0: return 1e12
    lam=_bteg_filter(y,mu,om,phi,kap,nu)
    e=(y-mu)*np.exp(-lam)
    c=math.lgamma(0.5*(nu+1.0))-math.lgamma(0.5*nu)-0.5*math.log(math.pi*nu)
    ll=c-lam-0.5*(nu+1.0)*np.log1p(e*e/nu)
    s=float(np.sum(ll))
    return -s if np.isfinite(s) else 1e12
def fit_bteg(y,sp):
    """MLE on y[:sp]; returns (sd_t over the whole series, mu, nu, diagnostics). sd = exp(lam)*sqrt(nu/(nu-2))."""
    ytr=np.asarray(y[:sp],float); mu0=float(np.mean(ytr)); sd0=max(float(np.std(ytr)),1e-6)
    best=None
    for nu0,kap0,phi0 in ((6.0,0.08,0.98),(4.0,0.15,0.95),(10.0,0.04,0.99)):
        om0=math.log(sd0)-0.5*math.log(nu0/(nu0-2.0))
        x0=np.array([mu0,om0,math.log(phi0/(1.0-phi0)),math.log(kap0),math.log(nu0-2.05)])
        try:
            r=optimize.minimize(_bteg_nll,x0,args=(ytr,),method='Nelder-Mead',
                                options={'maxiter':4000,'maxfev':6000,'xatol':1e-5,'fatol':1e-5})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    if best is None: raise RuntimeError("bteg MLE failed")
    mu,om=float(best.x[0]),float(best.x[1]); phi=1.0/(1.0+math.exp(-best.x[2]))
    kap=math.exp(best.x[3]); nu=2.05+math.exp(best.x[4])
    lam=_bteg_filter(np.asarray(y,float),mu,om,phi,kap,nu)   # filtered forward with the training-window params
    tsc=math.sqrt(nu/(nu-2.0)) if nu>2.0 else 1.0
    return np.exp(lam)*tsc, mu, nu, dict(om=round(om,4),phi=round(phi,4),kap=round(kap,4),nu=round(nu,3),
                                         nll=round(float(best.fun),2),score_bound=round(kap*nu,4))
# ============================================================ end Beta-t-EGARCH block

def pin(y,q,t): dd=y-q; return np.where(dd>=0,t*dd,(t-1)*dd)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    dm=x-x.mean(); v=np.mean(dm*dm)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(dm[k:]*dm[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def conf_ostat(sc,tau):
    n=len(sc); k=min(max(int(math.ceil((n+1)*tau)),1),n); return float(np.sort(np.asarray(sc,float))[k-1])
def fz0(r,v,e,a):
    v=np.minimum(v,-1e-8); e=np.minimum(e,v); hit=(r<=v).astype(float)
    return -(1.0/(a*e))*hit*(v-r)+v/e+np.log(-e)-1.0
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)
class GPDTail:
    def __init__(self,z,p0):
        z=np.asarray(z,float); z=z[np.isfinite(z)]; zz=-z
        self.u=float(np.quantile(zz,1-p0)); exc=zz[zz>self.u]-self.u; self.p0=p0; self.n_exc=int(len(exc))
        self.xi,_,self.beta=stats.genpareto.fit(exc,floc=0.0)
    def q(self,tau):
        xi,b,u,p0=self.xi,self.beta,self.u,self.p0
        return -(u+(b/xi)*((tau/p0)**(-xi)-1.0) if abs(xi)>1e-6 else u+b*math.log(p0/tau))
def feats(y,sig,mu,dts):
    z=(y-mu)/np.maximum(sig,1e-6); df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1); df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1); df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)
    return df
def fit_realgarch(r,x,sp):
    """Proper HHS log-linear Realized GARCH with measurement equation, Gaussian QML (as job_scaleshape_canonical.py)."""
    r=np.asarray(r,float); lx=np.log(np.maximum(np.asarray(x,float),1e-10)); n=len(r); v0=max(np.var(r[:sp]),1e-6)
    def negll(th):
        om,be,ga,xi,phi,t1,t2,lsu,mu=th
        if not(0.0<=be<0.999) or lsu<-6 or lsu>4: return 1e12
        su2=math.exp(2*lsu); lh=math.log(v0); ll=0.0
        for k in range(sp):
            if k>0: lh=om+be*lh+ga*lx[k-1]
            if lh>25 or lh<-25: return 1e12
            h=math.exp(lh); z=(r[k]-mu)/math.sqrt(h); ll+=0.5*(lh+z*z)
            mx=xi+phi*lh+t1*z+t2*(z*z-1.0); u=lx[k]-mx; ll+=0.5*(2*lsu+u*u/su2)
        return ll
    best=None
    for th0 in ([-0.1,0.6,0.35,0.0,1.0,-0.05,0.03,math.log(0.4),float(np.mean(r[:sp]))],
                [0.0,0.4,0.5,-0.2,0.8,-0.1,0.05,math.log(0.5),0.0]):
        try: res=optimize.minimize(negll,np.array(th0),method="Nelder-Mead",options={"maxiter":6000,"xatol":1e-4,"fatol":1e-4})
        except Exception: continue
        if best is None or res.fun<best.fun: best=res
    om,be,ga,xi,phi,t1,t2,lsu,mu=best.x
    lh=np.empty(n); lh[0]=math.log(v0)
    for k in range(1,n): lh[k]=om+be*lh[k-1]+ga*lx[k-1]
    return np.sqrt(np.exp(np.clip(lh,-25,25))), float(mu)

# ---------------------------------------------------------------- returns panel
if SYN:
    rng=np.random.default_rng(11); recs=[]; dates=pd.bdate_range("2004-01-01",periods=2200)
    for pn in range(24):
        om,al,be,nu=0.05,0.07,0.90,4.5+rng.uniform(-0.5,3.0); s2=np.empty(2200); e=np.empty(2200); s2[0]=om/(1-al-be)
        for k in range(2200):
            if k: s2[k]=om+al*e[k-1]**2+be*s2[k-1]
            z=stats.t.rvs(nu,random_state=rng)/math.sqrt(nu/(nu-2))
            if rng.uniform()<0.015: z-=rng.exponential(3.0)
            e[k]=math.sqrt(s2[k])*z
        recs.append(pd.DataFrame({'permno':pn,'date':dates,'ret':(0.02+e)/100.0}))
    rr=pd.concat(recs); names=list(range(24)); RV=None
elif PANEL=="taq30":
    rm=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(P,"panel_rm_*.csv")))],ignore_index=True)
    rm['date']=pd.to_datetime(rm['date'])
    r30=pd.read_csv(os.path.join(P,"crsp_returns_30.csv")); r30['date']=pd.to_datetime(r30['date'])
    r30['ret']=pd.to_numeric(r30['ret'],errors='coerce'); rr=r30.rename(columns={'ticker':'permno'})
    names=sorted(rr['permno'].dropna().unique())[:30]; RV={tk:rm[rm.ticker==tk][['date','rv']].dropna() for tk in names}
else:
    src="holdout_panel_2000_2013.csv" if PANEL=="holdout" else "crsp_panel_returns.csv"
    rr=pd.read_csv(os.path.join(P,src),dtype={'permno':'int32'})
    cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:200]; RV=None
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
MINN=1200 if PANEL=="taq30" else 1500

# ---------------------------------------------------------------- per name: every scale on one row set
SCALES=['garch_t','bip_t','bteg']+(['rgarch'] if PANEL=="taq30" else [])
TR={f:[] for f in SCALES}; CAL={f:[] for f in SCALES}; rows=[]; BTDIAG={}; nfail=0
for pn in names:
    g=rr[rr.permno==pn].sort_values('date')
    if PANEL=="taq30":
        g=pd.merge(g[['date','ret']],RV[pn],on='date',how='inner').sort_values('date')
    y=g['ret'].values.astype(float); dts=g['date'].values; n=len(y)
    if n<MINN or not np.isfinite(y).all(): continue
    sp=int(n*0.6); cp=int(sp*0.75); D={}
    try:
        r1=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False); pp=r1.params
        om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
        e0=y-mu; v0=np.var(y[:sp])
        for f,rob in (('garch_t',False),('bip_t',True)):
            s2=np.empty(n); s2[0]=v0
            for k in range(1,n):
                news=e0[k-1]**2
                if rob: news=min(news,KBIP*s2[k-1])
                s2[k]=max(om+al*news+be*s2[k-1],1e-8)
            df=feats(y,np.sqrt(s2),mu,dts); df['mu']=mu; df['nu']=nu
            df['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0; D[f]=df
        sd_bt,mu_bt,nu_bt,dgb=fit_bteg(y,sp)
        df=feats(y,sd_bt,mu_bt,dts); df['mu']=mu_bt; df['nu']=nu_bt
        df['tsc']=math.sqrt(nu_bt/(nu_bt-2)) if nu_bt>2 else 1.0; D['bteg']=df; BTDIAG[str(pn)]=dgb
        if PANEL=="taq30":
            sig_r,mu_r=fit_realgarch(y,g['rv'].values.astype(float)*1e4,sp)
            df=feats(y,sig_r,mu_r,dts); df['mu']=mu_r; df['nu']=nu; df['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0; D['rgarch']=df
    except Exception as ex:
        nfail+=1; lg("  fail %s %s"%(pn,str(ex)[:60])); continue
    base=D[SCALES[0]].copy(); base['idx']=np.arange(n)
    for f in SCALES[1:]:
        for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__'+f]=D[f][c].values
    need=ZX+['mk63']+[c+'__'+f for f in SCALES[1:] for c in ZX+['mk63']]
    ok=base.dropna(subset=need)
    trn=ok[ok['idx']<cp]; cal=ok[(ok['idx']>=cp)&(ok['idx']<sp)]; tst=ok[ok['idx']>=sp]
    if len(tst)<60 or len(cal)<60 or len(trn)<200: continue
    for f in SCALES:
        sfx='' if f==SCALES[0] else '__'+f
        TR[f].append(trn[[c+sfx for c in ZX+['z']]].rename(columns=lambda c:c.split('__')[0]))
        CAL[f].append(cal[[c+sfx for c in ZX+['z']]].rename(columns=lambda c:c.split('__')[0]))
    t2=tst.copy(); t2['permno']=pn; rows.append(t2)
    lg("  fit %s n=%d bteg(phi=%.3f kap=%.3f nu=%.2f) %.0fs"%(pn,n,dgb['phi'],dgb['kap'],dgb['nu'],time.time()-t0))
TE=pd.concat(rows).reset_index(drop=True); TRc={f:pd.concat(TR[f]) for f in SCALES}; CALc={f:pd.concat(CAL[f]) for f in SCALES}
lg("panel[%s] %d names %d test rows (%d fails) %.0fs"%(PANEL,TE.permno.nunique(),len(TE),nfail,time.time()-t0))
Y=TE['y'].values; di,_=pd.factorize(pd.to_datetime(TE['date'].values),sort=True); di=np.asarray(di); ALL=np.ones(len(Y),bool)
def G(c,f): return TE[c if f==SCALES[0] else c+'__'+f].values

# ---------------------------------------------------------------- one engine per scale
def build_engine(f):
    sfx='' if f==SCALES[0] else '__'+f
    X=TE[[c+sfx for c in ZX]].values; ztr=TRc[f]['z'].values; tail=GPDTail(ztr,P0_ENGINE)
    ZQ={}; ZQc={}
    for t in TAUS:
        m=HistGradientBoostingRegressor(loss='quantile',quantile=t,random_state=0,**HGB).fit(TRc[f][ZX].values,ztr)
        ZQ[t]=m.predict(X); ZQc[t]=m.predict(CALc[f][ZX].values)
    SUB={a:[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,random_state=0,**HGB)
            .fit(TRc[f][ZX].values,ztr).predict(X) for j in range(SUBN)] for a in ALPHAS}
    c975=conf_ostat(CALc[f]['z'].values-ZQc[0.025],0.025)
    MU=G('mu',f); SIG=G('sig',f)
    eng={t:(np.minimum(ZQ[t],tail.q(t)) if t<=P0_ENGINE else ZQ[t]) for t in TAUS}
    E=np.sort(np.stack([eng[t] for t in TAUS],axis=1),axis=1); eng={t:E[:,j] for j,t in enumerate(TAUS)}
    RQ_e={t:MU+SIG*eng[t] for t in TAUS}; RQ_b={t:MU+SIG*ZQ[t] for t in TAUS}; VE={}
    for a in ALPHAS:
        st=np.sort(np.stack([np.minimum(SUB[a][j],tail.q(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
        zq=np.maximum(np.minimum(ZQ[a],tail.q(a)),st[:,-1]); es=np.minimum(st.mean(axis=1),zq-1e-6)
        stb=np.sort(np.stack(SUB[a],axis=1),axis=1); zqb=np.maximum(ZQ[a],stb[:,-1]); esb=np.minimum(stb.mean(axis=1),zqb-1e-6)
        VE[a]={'engine':(MU+SIG*zq,MU+SIG*es),'body':(MU+SIG*zqb,MU+SIG*esb)}
    lg("engine[%s]: GPD xi=%.3f beta=%.3f n_exc=%d conf975=%+.4f %.0fs"%(f,tail.xi,tail.beta,tail.n_exc,c975,time.time()-t0))
    return RQ_e,RQ_b,VE,{'xi':round(tail.xi,4),'beta':round(tail.beta,4),'n_exc':tail.n_exc,'conf975':round(c975,4)}
LAB={'garch_t':'engine','bip_t':'engine_bip','bteg':'engine_bteg','rgarch':'engine_rg'}
RQ={}; VEa={a:{} for a in ALPHAS}; DIAG={}
for f in SCALES:
    RQ_e,RQ_b,VE,d_=build_engine(f); RQ[LAB[f]]=RQ_e; RQ[LAB[f].replace('engine','body')]=RQ_b; DIAG[LAB[f]]=d_
    for a in ALPHAS: VEa[a][LAB[f]]=VE[a]['engine']; VEa[a][LAB[f].replace('engine','body')]=VE[a]['body']

# ---------------------------------------------------------------- the three (four) parametric scale rows
for f in SCALES:
    MU,SIG,NU,TSC=G('mu',f),G('sig',f),G('nu',f),G('tsc',f)
    RQ[f]={t:MU+SIG*stats.t.ppf(t,NU)/TSC for t in TAUS}
    for a in ALPHAS: VEa[a][f]=(MU+SIG*stats.t.ppf(a,NU)/TSC,MU+SIG*t_es(a,NU)/TSC)
for f in SCALES:   # scale + POOLED UNCONDITIONAL residual quantile: the reference form of the scale-shape table
    zt=TRc[f]['z'].values; MU,SIG=G('mu',f),G('sig',f)
    RQ[f+'_uncond']={t:MU+SIG*float(np.quantile(zt,t)) for t in TAUS}
    for a in ALPHAS:
        qa=float(np.quantile(zt,a))
        VEa[a][f+'_uncond']=(MU+SIG*qa,MU+SIG*float(np.mean(zt[zt<=qa])))
RQ['fhs_pool']=RQ['garch_t_uncond']
for a in ALPHAS: VEa[a]['fhs_pool']=VEa[a]['garch_t_uncond']

# ---------------------------------------------------------------- pinball, one sort per scale's own score
PL={m:sum(pin(Y,RQ[m][t],t) for t in TAUS)/len(TAUS) for m in RQ}
def dec(x):
    r=np.full(len(x),-1); m=np.isfinite(x); r[m]=pd.qcut(pd.Series(x[m]),10,labels=False,duplicates='drop').values+1; return r
def edge(Lref,Lm,mask):
    mask=mask&np.isfinite(Lref)&np.isfinite(Lm)
    if mask.sum()<30: return None
    d=(Lref-Lm)[mask]; gg=pd.DataFrame({'d':d,'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'edge_pct':round(100*float(d.mean())/float(Lref[mask].mean()),3),'DM':nw_t(gg.values),'n':int(mask.sum())}
rk={f:dec(G('mk63',f)) for f in SCALES}
pinball={}
for f in SCALES:
    r=rk[f]; regions={'top_decile':r==10,'deciles_1to9':(r>=1)&(r<=9),'overall':ALL}
    pinball['sort_mk63_'+f]={
      'flexible_over_same_scale_parametric':{rg:edge(PL[f],PL[LAB[f]],msk) for rg,msk in regions.items()},
      'body_over_same_scale_parametric':{rg:edge(PL[f],PL[LAB[f].replace('engine','body')],msk) for rg,msk in regions.items()},
      'vs_garch_t':{m:{rg:edge(PL['garch_t'],PL[m],msk) for rg,msk in regions.items()} for m in PL if m!='garch_t'},
      'decile_profile_vs_same_scale':{int(d0):edge(PL[f],PL[LAB[f]],r==d0) for d0 in range(1,11)}}
    lg("[sort %s] flexible over same-scale parametric: "%f+json.dumps(pinball['sort_mk63_'+f]['flexible_over_same_scale_parametric']))
overlap={'%s_vs_%s'%(a,b):round(float(np.mean((rk[a]==10)==(rk[b]==10))),4)
         for i,a in enumerate(SCALES) for b in SCALES[i+1:]}

# ---------------------------------------------------------------- FZ0, breach, DM, MCS
def dm_rows(Lm,Le,mask):
    mask=mask&np.isfinite(Lm)&np.isfinite(Le); d=(Lm-Le)[mask]
    if mask.sum()<30: return None
    gg=pd.DataFrame({'d':d,'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'mean_diff':round(float(np.mean(d)),5),'DM_t':nw_t(gg.values),'n':int(mask.sum())}
def mcs(Lmat,names_,B=1000,alpha=0.10,block=10,seed=0):
    T,M=Lmat.shape; rng_=np.random.default_rng(seed); idx=np.empty((B,T),dtype=int)
    for b in range(B):
        t_=0; pos=rng_.integers(T)
        while t_<T:
            if rng_.uniform()<1.0/block: pos=rng_.integers(T)
            idx[b,t_]=pos; pos=(pos+1)%T; t_+=1
    alive=list(range(M)); pv={}; pmax=0.0
    while len(alive)>1:
        L=Lmat[:,alive]; dbar=L.mean(axis=0)-L.mean(); Lb=L[idx].mean(axis=1); dbarb=Lb-Lb.mean(axis=1,keepdims=True)
        var=((dbarb-dbar)**2).mean(axis=0); tstat=dbar/np.sqrt(np.maximum(var,1e-30)); Tmax=float(tstat.max())
        Tb=((dbarb-dbar)/np.sqrt(np.maximum(var,1e-30))).max(axis=1); p=float(np.mean(Tb>=Tmax)); pmax=max(pmax,p)
        worst=alive[int(np.argmax(tstat))]; pv[names_[worst]]=round(pmax,4)
        if pmax>=alpha: break
        alive.remove(worst)
    for i in alive: pv.setdefault(names_[i],round(pmax,4))
    return {'in_90pct_MCS':[names_[i] for i in alive],'mcs_pvalues':pv}
FZ={}; MCS={}
for a in ALPHAS:
    L={}; out={}
    for m,(v,e) in VEa[a].items():
        Lm=fz0(Y,v,e,a); okm=np.isfinite(Lm)
        if okm.sum()<500: continue
        L[m]=Lm; out[m]={'meanFZ0':round(float(np.nanmean(Lm[okm])),5),'breach':round(float(np.mean((Y<=v)[okm])),4)}
    for m in out:
        out[m]['vs_garch_t']=dm_rows(L[m],L['garch_t'],ALL) if m!='garch_t' else None
        out[m]['vs_engine']=dm_rows(L[m],L['engine'],ALL) if m!='engine' else None
        out[m]['vs_engine_bteg']=dm_rows(L[m],L['engine_bteg'],ALL) if m!='engine_bteg' else None
        out[m]['vs_bteg_top_decile']=dm_rows(L[m],L['bteg'],rk['bteg']==10) if m!='bteg' else None
        # tab:rgarch convention: DM against the realized scale with its own parametric tail (taq30 panel only)
        REF_RG='rgarch_uncond'   # tab:rgarch reference: realized scale + unconditional residual quantile
        if REF_RG in L: out[m]['vs_rgarch_uncond']=dm_rows(L[m],L[REF_RG],ALL) if m!=REF_RG else None
    names_=[m for m in L if np.isfinite(L[m]).all()]
    dfm=pd.DataFrame({m:L[m] for m in names_}); dfm['dt']=di
    MCS['fz0_%g'%a]=mcs(dfm.groupby('dt')[names_].mean().values,names_); FZ[str(a)]=out
    lg("FZ0 a=%s: "%a+json.dumps({m:out[m]['meanFZ0'] for m in out})+" MCS "+json.dumps(MCS['fz0_%g'%a]['in_90pct_MCS']))
names_=[m for m in PL if np.isfinite(PL[m]).all()]; dfm=pd.DataFrame({m:PL[m] for m in names_}); dfm['dt']=di
MCS['pinball_11tau']=mcs(dfm.groupby('dt')[names_].mean().values,names_)

btd=pd.DataFrame(BTDIAG).T
OUT={'note':'Beta-t-EGARCH (t-GAS) Stage-1 scale beside GARCH(1,1)-t and bounded-news (BIP) GARCH'
     +(' and the HHS Realized GARCH' if PANEL=="taq30" else '')+', each as a standalone parametric benchmark '
     '(rows garch_t / bip_t / bteg'+(' / rgarch' if PANEL=="taq30" else '')+') and as the scale under the flexible '
     'shape (engine* / body*), one common set of test rows. DM_t>0 means the row model is worse than the named '
     'reference. The paper-comparable number is flexible_over_same_scale_parametric.top_decile under each sort.',
     'predictions_written_before_run':'P1 bteg beats garch_t on FZ0 at both levels by 0.005-0.03; P2 top-decile '
     'flexible edge over the same-scale parametric falls to +0.2..+0.9% (DM 2-5) under bteg; P3 engine_bteg FZ0 '
     'below engine at both levels but by less than P1; P4 top-decile sort overlap below 0.75.',
     'panel':PANEL,'synthetic':SYN,'K_bip':KBIP,'n_names':int(TE.permno.nunique()),'n_test':int(len(Y)),'n_fail':nfail,
     'taus':TAUS,'bteg_param_medians':{k:round(float(btd[k].median()),4) for k in ['phi','kap','nu','score_bound']},
     'bteg_per_name':BTDIAG,'engine_diagnostics':DIAG,'top_decile_overlap':overlap,
     'mean_pinball':{m:round(float(np.nanmean(PL[m])),5) for m in PL},'pinball':pinball,'fz0':FZ,'mcs':MCS}
fn="robust_engine_results%s.json"%("_synthetic" if SYN else ("" if PANEL=="canon200" else "_"+PANEL))
json.dump(OUT,open(os.path.join(P,fn),"w"),indent=2)
lg("ROBUSTENGINEDONE[%s] %.0fs -> %s"%(PANEL,time.time()-t0,fn))
