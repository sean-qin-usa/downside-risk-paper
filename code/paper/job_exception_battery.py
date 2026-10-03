# job_exception_battery.py -- THE CALIBRATION AND EXCEPTION BATTERY ON EVERY STAGE-1 SCALE, SAME ROWS.
# job_robust_engine.py showed that a Beta-t-EGARCH Stage 1 beats the paper's GARCH(1,1)-t Stage 1 on the
# joint FZ0 loss at both levels (DM 3.31 and 4.55) and sits inside the 90% model confidence set where the
# paper's engine does not. That is an ACCURACY result only. The paper's deployment choice was settled by
# CALIBRATION -- the date-clustered exception test is what disqualified the pooled body (NW t 4.23, 5.13)
# and the conformal overlay (-10.73, -9.79) -- and no calibration test has ever been run on a robust scale.
# This job runs the full battery on all three scales so the Stage-1 question can be decided on the same
# evidence the original choice was made on.
#
# TESTS, copied verbatim from the two committed scripts so the numbers are comparable by construction:
#   from job_perasset_v2.py   per-name Kupiec LR pass rate; per-name Christoffersen conditional-coverage
#                             LR pass rate; the date-clustered exception statistic, NW(10) t on the
#                             per-date breach frequency minus nominal
#   from job_engine_esbt.py   pooled Kupiec UC; Acerbi-Szekely (2014) Z2; McNeil-Frey (2000) exceedance
#                             residual; mean FZ0 and the date-clustered DM
# TWO DEPARTURES, both deliberate and both stated in the output file:
#   1. The battery runs on the canonical 200-name rows for every scale. The committed per-name exception
#      file covers 140 names, so its GARCH-t numbers are not reproduced here exactly; the GARCH-t arm of
#      THIS run is the like-for-like anchor for the other two scales, and the ES tests, which the committed
#      200-name file does cover, are reproduced exactly as the validation.
#   2. One stationary-bootstrap index matrix (B=4000, mean block 10 dates) is drawn once and reused for
#      every variant, where the committed script redrew it per call. Common random numbers tighten the
#      cross-variant comparison; the test STATISTICS are deterministic and unaffected, and only the
#      bootstrap p-values can move, in the third decimal.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01):
#  P1. engine_bteg passes the date-clustered exception test at both levels (|NW t| < 1.96), as the GARCH-t
#      engine does, because its breach rates (0.92% and 2.46%) sit closer to nominal than the GARCH-t
#      engine's (1.04% and 2.78%). THIS IS THE DECISION: if it fails, Stage 1 stays GARCH-t and the FZ0
#      gain is recorded as an accuracy-only result in the limitations section.
#  P2. engine_bteg's pooled Kupiec at 2.5% does not reject (p > 0.05) where the GARCH-t engine's rejects at
#      p < 0.001, since a 2.46% rate misses nominal by 0.04pp against the GARCH-t engine's 0.28pp.
#  P3. engine_bteg's Acerbi-Szekely Z2 is closer to zero than the GARCH-t engine's at both levels and does
#      not reject at 5% at either level; the GARCH-t engine's marginally rejects at 2.5% (p = 0.039).
#  P4. body_bteg, despite having the lowest FZ0 of any row, still fails the date-clustered test at both
#      levels as the GARCH-t body does, because it over-breaches (1.19% and 2.97%). The EVT branch, not the
#      scale, is what buys exception-test compliance.
# ADDED AFTER THE FIRST RUN (2026-10-01, stated as a post-hoc addition): the committed Stage-4 conformal
# shift is estimated against the BODY's calibration errors and then applied to the ENGINE, which already
# carries the EVT branch. The tail correction therefore enters twice, which is why every overlay row
# over-covers (NW t -8 to -11 at 2.5%). The conf2_* rows re-estimate the shift against the engine's own
# calibration-block quantile, at each level separately -- split conformal on the object it is applied to.
# PREDICTION FOR THE ADDED ROWS, written before they were run: conf2_bteg lands within 0.05pp of nominal at
# both levels and passes the date-clustered test at both, because the engine over-covers by only 0.10pp at
# 1% and 0.085pp at 2.5% and a correctly targeted shift removes exactly that bias. If conf2 overshoots to
# the under-covering side at either level, the mis-targeting was not the whole story.
# Output: exception_battery_results.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from es_integral import converged_es, committed_es_20node
from scipy import stats
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True); rng=np.random.default_rng(20260906)
SYN="--synthetic" in sys.argv
PANEL=sys.argv[sys.argv.index("--panel")+1] if "--panel" in sys.argv else "canon200"
assert PANEL in ("canon200","holdout"), "unknown --panel %s"%PANEL
KEXTRA=40        # extra log-spaced body levels in [a/40, a] for the converged ES; 20 committed + 40 = 60, matching job_es_converged.py
ALPHAS=[0.01,0.025]; ZX=['logsig','zl1','absz5','zstd21','fracdn5']; SUBN=20; KBIP=9.0; P0=0.025
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06); BBOOT=4000; MBLOCK=10.0
SCALES=['garch_t','bip_t','bteg']

# ---------------------------------------------------------------- Beta-t-EGARCH (as in job_robust_engine.py)
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
    lam=_bteg_filter(y,mu,om,phi,kap,nu); e=(y-mu)*np.exp(-lam)
    c=math.lgamma(0.5*(nu+1.0))-math.lgamma(0.5*nu)-0.5*math.log(math.pi*nu)
    s=float(np.sum(c-lam-0.5*(nu+1.0)*np.log1p(e*e/nu)))
    return -s if np.isfinite(s) else 1e12
def fit_bteg(y,sp):
    from scipy import optimize
    ytr=np.asarray(y[:sp],float); mu0=float(np.mean(ytr)); sd0=max(float(np.std(ytr)),1e-6); best=None
    for nu0,kap0,phi0 in ((6.0,0.08,0.98),(4.0,0.15,0.95),(10.0,0.04,0.99)):
        om0=math.log(sd0)-0.5*math.log(nu0/(nu0-2.0))
        x0=np.array([mu0,om0,math.log(phi0/(1.0-phi0)),math.log(kap0),math.log(nu0-2.05)])
        try: r=optimize.minimize(_bteg_nll,x0,args=(ytr,),method='Nelder-Mead',
                                 options={'maxiter':4000,'maxfev':6000,'xatol':1e-5,'fatol':1e-5})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    if best is None: raise RuntimeError("bteg MLE failed")
    mu,om=float(best.x[0]),float(best.x[1]); phi=1.0/(1.0+math.exp(-best.x[2]))
    kap=math.exp(best.x[3]); nu=2.05+math.exp(best.x[4])
    lam=_bteg_filter(np.asarray(y,float),mu,om,phi,kap,nu)
    tsc=math.sqrt(nu/(nu-2.0)) if nu>2.0 else 1.0
    return np.exp(lam)*tsc, mu, nu, dict(phi=round(phi,4),kap=round(kap,4),nu=round(nu,3))

# ---------------------------------------------------------------- tests, verbatim from the committed scripts
def _llb(pp,k0,k1):                                    # job_perasset_v2.py
    if k0+k1==0: return 0.0
    if pp<=0: return 0.0 if k1==0 else -1e300
    if pp>=1: return 0.0 if k0==0 else -1e300
    return k0*math.log(1-pp)+k1*math.log(pp)
def kupiec_pername(x,T,p):                             # job_perasset_v2.py
    if T==0 or x==T: return None
    if x==0: return round(float(1-stats.chi2.cdf(-2*_llb(p,T,0),1)),4)
    pi=x/T; lr=-2*(_llb(p,T-x,x)-_llb(pi,T-x,x)); return round(float(1-stats.chi2.cdf(max(lr,0),1)),4)
def christoffersen(b,p):                               # job_perasset_v2.py
    b=b.astype(int); T=len(b); x=int(b.sum())
    if x==0: return None
    n00=n01=n10=n11=0
    for i in range(1,T):
        a,c=b[i-1],b[i]
        if a==0 and c==0:n00+=1
        elif a==0 and c==1:n01+=1
        elif a==1 and c==0:n10+=1
        else:n11+=1
    pi=x/T; pi0=n01/max(n00+n01,1); pi1=n11/max(n10+n11,1)
    lr_uc=-2*(_llb(p,T-x,x)-_llb(pi,T-x,x))
    lr_ind=-2*(_llb(pi,n00+n10,n01+n11)-(_llb(pi0,n00,n01)+_llb(pi1,n10,n11)))
    return round(float(1-stats.chi2.cdf(max(lr_uc+lr_ind,0),2)),4)
def nw_t(x,l=10):                                      # job_perasset_v2.py
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    d=x-x.mean(); v=np.mean(d*d)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(d[k:]*d[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def kupiec_pooled(brind,a):                            # job_engine_esbt.py
    N=len(brind); nb=int(brind.sum()); ph=nb/N if N else 0.0
    if nb==0 or nb==N: return ph,float('nan')
    LR=-2*((nb*math.log(a)+(N-nb)*math.log(1-a))-(nb*math.log(ph)+(N-nb)*math.log(1-ph)))
    return ph,float(1-stats.chi2.cdf(LR,1))
def fz0(r,v,e,a):                                      # job_engine_esbt.py
    v=np.minimum(v,-1e-8); e=np.minimum(e,v); hit=(r<=v).astype(float)
    return -(1.0/(a*e))*hit*(v-r)+v/e+np.log(-e)-1.0
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)
def conf_ostat(sc,tau):
    n=len(sc); k=min(max(int(math.ceil((n+1)*tau)),1),n); return float(np.sort(np.asarray(sc,float))[k-1])

# ---------------------------------------------------------------- panel: three scales on one row set
if SYN:
    rngs=np.random.default_rng(5); recs=[]; dates=pd.bdate_range("2004-01-01",periods=2200)
    for pn in range(24):
        om,al,be,nu=0.05,0.07,0.90,4.5+rngs.uniform(-0.5,3.0); s2=np.empty(2200); e=np.empty(2200); s2[0]=om/(1-al-be)
        for k in range(2200):
            if k: s2[k]=om+al*e[k-1]**2+be*s2[k-1]
            z=stats.t.rvs(nu,random_state=rngs)/math.sqrt(nu/(nu-2))
            if rngs.uniform()<0.015: z-=rngs.exponential(3.0)
            e[k]=math.sqrt(s2[k])*z
        recs.append(pd.DataFrame({'permno':pn,'date':dates,'ret':(0.02+e)/100.0}))
    rr=pd.concat(recs); NMAX=24
else:
    src="holdout_panel_2000_2013.csv" if PANEL=="holdout" else "crsp_panel_returns.csv"
    rr=pd.read_csv(os.path.join(P,src),dtype={'permno':'int32'}); NMAX=200
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:NMAX]
def feats(y,sig,mu,dts):
    z=(y-mu)/np.maximum(sig,1e-6); df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1); df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1)
    df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)   # not used as a metric here, but it must
    return df                                                        # enter the dropna so the row set matches
                                                                     # every other job in the pipeline
TR={f:[] for f in SCALES}; CAL={f:[] for f in SCALES}; rows=[]; BTD={}; nfail=0
for pn in names:
    g=rr[rr.permno==pn].sort_values('date'); y=g['ret'].values.astype(float); dts=g['date'].values; n=len(y)
    if n<1500: continue
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
        df['tsc']=math.sqrt(nu_bt/(nu_bt-2)) if nu_bt>2 else 1.0; D['bteg']=df; BTD[str(pn)]=dgb
    except Exception as ex:
        nfail+=1; lg("  fail %s %s"%(pn,str(ex)[:50])); continue
    base=D['garch_t'].copy(); base['idx']=np.arange(n)
    for f in SCALES[1:]:
        for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__'+f]=D[f][c].values
    # mk63 must be in the dropna set: every other job in the pipeline (job_composite, job_bench_all,
    # job_robust_engine) drops on it, and omitting it here trained the body and fitted the GPD on ~1.8%
    # more rows -- the early rows where mk63 is still NaN -- which shifted the GPD threshold and flipped the
    # engine's date-clustered verdict at 2.5%. Fixed 2026-10-02.
    need=ZX+['mk63']+[c+'__'+f for f in SCALES[1:] for c in ZX+['mk63']]
    ok=base.dropna(subset=need)
    trn=ok[ok['idx']<cp]; cal=ok[(ok['idx']>=cp)&(ok['idx']<sp)]; tst=ok[ok['idx']>=sp]
    if len(tst)<60 or len(cal)<60 or len(trn)<200: continue
    for f in SCALES:
        sfx='' if f=='garch_t' else '__'+f
        TR[f].append(trn[[c+sfx for c in ZX+['z']]].rename(columns=lambda c:c.split('__')[0]))
        CAL[f].append(cal[[c+sfx for c in ZX+['z']]].rename(columns=lambda c:c.split('__')[0]))
    t2=tst.copy(); t2['permno']=pn; rows.append(t2)
    lg("  fit %s n=%d %.0fs"%(pn,n,time.time()-t0))
TE=pd.concat(rows).reset_index(drop=True); TRc={f:pd.concat(TR[f]) for f in SCALES}; CALc={f:pd.concat(CAL[f]) for f in SCALES}
Y=TE['y'].values; PERM=TE['permno'].values; DTS=TE['date'].values
lg("panel %d names %d rows (%d fails) %.0fs"%(TE.permno.nunique(),len(TE),nfail,time.time()-t0))
G=lambda c,f: TE[c if f=='garch_t' else c+'__'+f].values

# ---------------------------------------------------------------- (VaR,ES) per scale, engine_esbt construction
VE={a:{} for a in ALPHAS}; DIAG={}; ESDIAG={}
for f in SCALES:
    sfx='' if f=='garch_t' else '__'+f
    X=TE[[c+sfx for c in ZX]].values; Xc=CALc[f][ZX].values; ztr=TRc[f]['z'].values
    uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]
    xi,_,beta=stats.genpareto.fit(exc,floc=0.0)
    def evt_q(tau,uu=uu,xi=xi,beta=beta):
        return uu-(beta/xi)*((tau/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-beta*math.log(P0/tau)
    ZQ={}; ZQc={}
    for a in ALPHAS:
        m=HistGradientBoostingRegressor(loss='quantile',quantile=a,random_state=0,**HGB).fit(TRc[f][ZX].values,ztr)
        ZQ[a]=m.predict(X); ZQc[a]=m.predict(Xc)
    SUB={a:[] for a in ALPHAS}; SUBc={a:[] for a in ALPHAS}
    for a in ALPHAS:
        for j in range(SUBN):
            mz=HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,random_state=0,**HGB).fit(TRc[f][ZX].values,ztr)
            SUB[a].append(mz.predict(X)); SUBc[a].append(mz.predict(Xc))
    zcal=CALc[f]['z'].values
    # Stage 4, as committed: the shift is the tau-order statistic of the BODY's calibration errors, and the
    # committed code applies it only at 2.5%. It is then added to the ENGINE, which already carries the EVT
    # branch, so the tail correction enters twice -- the reason every overlay row over-covers.
    c975=conf_ostat(zcal-ZQc[0.025],0.025)
    # Stage 4, retargeted: the shift is estimated against the object it is applied to, the engine's own
    # calibration-block quantile, and at each level separately. This is split conformal on the engine.
    CE={}
    for a in ALPHAS:
        starc=np.sort(np.stack([np.minimum(SUBc[a][j],evt_q(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
        zqc=np.maximum(np.minimum(ZQc[a],evt_q(a)),starc[:,-1])
        CE[a]=conf_ostat(zcal-zqc,a)
    MU=G('mu',f); SIG=G('sig',f); NU=G('nu',f); TSC=G('tsc',f)
    for a in ALPHAS:
        star=np.sort(np.stack([np.minimum(SUB[a][j],evt_q(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
        zq=np.maximum(np.minimum(ZQ[a],evt_q(a)),star[:,-1]); es=np.minimum(star.mean(axis=1),zq-1e-6)
        stb=np.sort(np.stack(SUB[a],axis=1),axis=1); zqb=np.maximum(ZQ[a],stb[:,-1]); esb=np.minimum(stb.mean(axis=1),zqb-1e-6)
        # ---- CONVERGED ES, via the shared es_integral module. VaR (zq, zqb) is untouched, so no VaR-only
        # statistic can move; the body is interpolated only on [a/40, a] and the sub-floor GPD region is
        # integrated in closed form. The 20 committed sub-levels are reused, KEXTRA log-spaced ones added.
        floor=a/SUBN/2.0
        lev=np.unique(np.concatenate([a*((np.arange(SUBN)+0.5)/SUBN),
                                      np.exp(np.linspace(math.log(floor),math.log(a),KEXTRA)),[a]]))
        known={round(float(a*(j+0.5)/SUBN),12):SUB[a][j] for j in range(SUBN)}
        QB=np.stack([known[round(float(u),12)] if round(float(u),12) in known
                     else HistGradientBoostingRegressor(loss='quantile',quantile=float(u),random_state=0,**HGB
                          ).fit(TRc[f][ZX].values,ztr).predict(X) for u in lev],axis=1)
        esC=converged_es(a,lev,QB,evt_q,(uu,float(beta),float(xi),P0),M=2000,var_z=zq)
        # the body row has no tail below its lowest fitted level; the committed rule extends it flat, so the
        # converged body ES does the same, with a very large evt_fn standing in for 'the body is the minimum'.
        # NOTE on the body row: it has no tail below its lowest fitted level, so its ES there is not defined
        # by the estimator -- the committed 20-node rule extends it flat. There is no converged counterpart
        # without inventing a tail, so body_<f> keeps the committed convention and is flagged as such.
        sh=c975 if a==0.025 else 0.0
        VE[a]['engine_'+f]  =(MU+SIG*zq,       MU+SIG*es)
        VE[a]['body_'+f]    =(MU+SIG*zqb,      MU+SIG*esb)
        VE[a]['overlay_'+f] =(MU+SIG*(zq+sh),  MU+SIG*(es+sh))
        VE[a]['conf2_'+f]   =(MU+SIG*(zq+CE[a]),MU+SIG*(es+CE[a]))
        # converged counterparts, carried as separate rows so every committed number stays auditable
        VE[a]['engineC_'+f] =(MU+SIG*zq,        MU+SIG*esC)
        VE[a]['overlayC_'+f]=(MU+SIG*(zq+sh),   MU+SIG*(esC+sh))
        VE[a]['conf2C_'+f]  =(MU+SIG*(zq+CE[a]),MU+SIG*(esC+CE[a]))
        ESDIAG.setdefault(f,{})[str(a)]={'es_20node_mean':round(float(np.nanmean(es)),5),
            'es_converged_mean':round(float(np.nanmean(esC)),5),
            'ratio_conv_over_20node':round(float(np.nanmean(esC/es)),5),
            'n_body_levels':int(len(lev)),'floor':round(floor,6)}
        VE[a]['param_'+f]   =(MU+SIG*stats.t.ppf(a,NU)/TSC, MU+SIG*t_es(a,NU)/TSC)
        # pooled filtered historical simulation on this scale: the training residual quantile and the mean
        # below it. On the GARCH-t scale this row IS the paper's pooled FHS benchmark.
        qa=float(np.quantile(ztr,a)); ea=float(np.mean(ztr[ztr<=qa]))
        VE[a]['fhs_'+f]     =(MU+SIG*qa, MU+SIG*ea)
    DIAG['engine_'+f]={'gpd_u':round(uu,4),'gpd_xi':round(float(xi),4),'gpd_beta':round(float(beta),4),
                       'conf975_body_targeted':round(c975,4),
                       'conf_engine_targeted':{str(a):round(CE[a],4) for a in ALPHAS}}
    lg("scale[%s]: GPD u=%.3f xi=%.3f beta=%.3f | conf(body)=%+.4f conf(engine)=%s %.0fs"%(
        f,uu,xi,beta,c975,{str(a):round(CE[a],4) for a in ALPHAS},time.time()-t0))

# ---------------------------------------------------------------- one shared stationary-bootstrap index
class Ctx:
    """Date mapping and one stationary-bootstrap index for a given set of rows. Built per era so that a
    crisis sub-window is resampled over its own dates rather than the whole sample's."""
    def __init__(s,rowmask,tag):
        s.tag=tag; s.rows=np.where(rowmask)[0]
        d=DTS[s.rows]; s.udates=np.unique(d); s.Dn=len(s.udates)
        s.rdp=pd.Series(np.arange(s.Dn),index=s.udates).reindex(d).values
        s.IDX=np.empty((BBOOT,s.Dn),dtype=np.int64)
        st=rng.integers(0,s.Dn,size=(BBOOT,s.Dn)); nb=rng.random((BBOOT,s.Dn))<(1.0/MBLOCK)
        s.IDX[:,0]=st[:,0]
        for t in range(1,s.Dn): s.IDX[:,t]=np.where(nb[:,t],st[:,t],(s.IDX[:,t-1]+1)%s.Dn)
        s.rdp_full=np.full(len(DTS),-1,dtype=np.int64); s.rdp_full[s.rows]=s.rdp
        s.mask=rowmask
        lg("  ctx[%s]: %d rows, %d dates, bootstrap %dx%d %.0fs"%(tag,len(s.rows),s.Dn,BBOOT,s.Dn,time.time()-t0))
    def agg(s,v,dp): out=np.zeros(s.Dn); np.add.at(out,dp,np.asarray(v,float)); return out
    def boot(s,S,C):
        cc=C[s.IDX].sum(axis=1); ss=S[s.IDX].sum(axis=1)
        return np.where(cc>1e-12,ss/np.maximum(cc,1e-12),np.nan)
ERAS=[('all',np.ones(len(Y),bool))]
if PANEL=="holdout":
    yr=pd.to_datetime(DTS).year.values
    ERAS.append(('crisis_2008_2009',(yr>=2008)&(yr<=2009)))
CTX={tag:Ctx(m,tag) for tag,m in ERAS}
def z2_test(C,z,zq,zes,a,dp):                          # job_engine_esbt.py
    t=(z<=zq).astype(float)*z/(a*zes); obs=1.0-t.sum()/len(t)
    boot=1.0-C.boot(C.agg(t,dp),C.agg(np.ones(len(t)),dp)); se=np.nanstd(boot)
    return float(obs), (float(2*(1-stats.norm.cdf(abs(obs)/se))) if se>0 else float('nan'))
def mf_test(C,z,zq,zes,dp):                            # job_engine_esbt.py
    m=z<=zq; erm=np.where(m,z-zes,0.0); nb=int(m.sum())
    if nb<10: return float('nan'),float('nan'),nb
    obs=erm.sum()/nb
    boot=C.boot(C.agg(erm,dp),C.agg(m.astype(float),dp)); se=np.nanstd(boot)
    return float(obs), (float(2*(1-stats.norm.cdf(abs(obs)/se))) if se>0 else float('nan')), nb
def dclust_dm(C,Lm,Le,dp):                             # job_engine_esbt.py
    d=Lm-Le; sd=C.agg(d,dp); cd=C.agg(np.ones(len(d)),dp)
    dd=(sd/np.maximum(cd,1e-12))[cd>0]; nD=len(dd); mbar=dd.mean(); v=dd.var()
    for k in range(1,11): v+=2*(1-k/11)*np.mean((dd[k:]-mbar)*(dd[:-k]-mbar))
    se=math.sqrt(max(v,1e-16)/nD); return float(mbar),float(mbar/se if se>0 else 0.0),nD

# ---------------------------------------------------------------- the battery
MODELS=[p+'_'+f for f in SCALES for p in ('engine','body','overlay','conf2','param','fhs',
                                          'engineC','overlayC','conf2C')]
OUT={'note':'Full calibration and exception battery on three Stage-1 scales, same canonical rows. Tests are '
     'verbatim from job_perasset_v2.py (per-name Kupiec and Christoffersen pass rates, date-clustered '
     'exception NW(10) t) and job_engine_esbt.py (pooled Kupiec, Acerbi-Szekely Z2, McNeil-Frey, FZ0, '
     'date-clustered DM). DEPARTURES: the per-name tests run on 200 names here against 140 in the committed '
     'file, so this run\'s garch_t arm, not that file, is the like-for-like anchor; and one stationary '
     'bootstrap index (B=4000, mean block 10) is shared across variants, which leaves every test statistic '
     'unchanged and can move bootstrap p-values in the third decimal. A variant PASSES the date-clustered '
     'exception test when |NW t| < 1.96.',
     'predictions_written_before_run':'P1 engine_bteg passes the date-clustered test at both levels (the '
     'decision); P2 its pooled Kupiec at 2.5% does not reject; P3 its AS Z2 is closer to zero than the '
     'GARCH-t engine\'s at both levels and rejects at neither; P4 body_bteg still fails the date-clustered '
     'test at both levels despite the lowest FZ0.',
     'synthetic':SYN,'panel':PANEL,'n_names':int(TE.permno.nunique()),'n_test':int(len(Y)),'n_dates':int(CTX['all'].Dn),'n_fail':nfail,
     'bootstrap':{'B':BBOOT,'mean_block':MBLOCK,'shared_index':True},'diagnostics':DIAG,
     'es_convention':{'committed':'20-node midpoint mean of the model quantile curve, as shipped',
       'converged':'rows suffixed C: body interpolated on [a/40,a] with the sub-floor GPD region in closed form '
                   '(es_integral.converged_es, M=2000, 60 fitted body levels). VaR identical to the committed row, '
                   'so every VaR-only statistic must match between <row> and <row>C -- that is the built-in control.',
       'body_rows':'no converged counterpart: the body has no tail below its lowest fitted level',
       'per_scale':ESDIAG},
     'bteg_param_medians':{k:round(float(pd.DataFrame(BTD).T[k].median()),4) for k in ['phi','kap','nu']} if BTD else {},
     'per_alpha':{}}
def battery(a,C):
    """Every test for every variant, on the rows of one era context."""
    A={}; MUg={f:G('mu',f) for f in SCALES}; SIGg={f:G('sig',f) for f in SCALES}
    LE={f:fz0(Y,*VE[a]['engine_'+f],a) for f in SCALES}
    NE=len(C.rows)
    for m in MODELS:
        f=m.split('_',1)[1]; v,e=VE[a][m]; v=np.asarray(v,float); e=np.asarray(e,float)
        mu,sg=MUg[f],SIGg[f]; z=(Y-mu)/sg; zq=(v-mu)/sg; zes=(e-mu)/sg
        sel=C.mask&np.isfinite(zq)&np.isfinite(zes)&np.isfinite(z)&(zes<0)
        if sel.sum()<200: continue
        dp=C.rdp_full[sel]
        br,kp=kupiec_pooled((Y[sel]<=v[sel]).astype(float),a)
        z2,z2p=z2_test(C,z[sel],zq[sel],zes[sel],a,dp); mf,mfp,nb=mf_test(C,z[sel],zq[sel],zes[sel],dp)
        b=(Y<v).astype(int)
        kl=[];cl=[]
        for pn in np.unique(PERM[C.rows]):
            msk=C.mask&(PERM==pn)
            if msk.sum()<60: continue
            kv=kupiec_pername(int(b[msk].sum()),int(msk.sum()),a)
            if kv is not None: kl.append(kv)
            cv=christoffersen(b[msk],a)
            if cv is not None: cl.append(cv)
        fdf=pd.DataFrame({'b':b[C.rows],'date':DTS[C.rows]}).groupby('date')['b'].mean()
        dct=nw_t(fdf.values-a)
        ucp=float(2*(1-stats.norm.cdf(abs(dct)))) if dct is not None else None
        nsel=int(sel.sum()); zi=(br-a)/math.sqrt(a*(1-a)/nsel) if nsel else None
        de=(zi/dct)**2 if (dct not in (None,0.0) and zi is not None) else None
        nbar=nsel/C.Dn; rho=((de-1)/(nbar-1)) if de is not None and nbar>1 else None
        Lm=fz0(Y,v,e,a); _,dmt,nD=dclust_dm(C,Lm[C.rows],LE[f][C.rows],C.rdp)
        A[m]={'breach':round(br,4),'kupiec_pooled_p':round(kp,4),
              'kupiec_pername_passrate':round(float(np.mean([x>0.05 for x in kl])),3) if kl else None,
              'christoffersen_passrate':round(float(np.mean([x>0.05 for x in cl])),3) if cl else None,
              'dateclustered_NW_t':dct,'dateclustered_PASS':(abs(dct)<1.96) if dct is not None else None,
              'uc_clustered_p':round(ucp,4) if ucp is not None else None,
              'design_effect_iid_over_clustered':round(de,2) if de is not None else None,
              'rho_intradate_implied':round(rho,4) if rho is not None else None,
              'AS_Z2':round(z2,4),'AS_Z2_p':round(z2p,4),'MF_exres':round(mf,4),'MF_p':round(mfp,4),
              'n_breach':int(nb),'n_rows':int(sel.sum()),'meanFZ0':round(float(np.nanmean(Lm[C.rows])),5),
              'DM_vs_own_scale_engine':round(dmt,2),'n_dates':int(nD)}
        lg("  [%s] a=%g %-16s breach %.4f kupiec_pooled %.4f UC_clustered %.3f dclust_t %6s PASS %-5s AS_Z2 %+.4f (p %.3f) MF_p %.3f FZ0 %.5f"%(
            C.tag,a,m,br,kp,ucp if ucp is not None else float("nan"),dct,str(A[m]['dateclustered_PASS']),z2,z2p,mfp,A[m]['meanFZ0']))
    return A
for a in ALPHAS: OUT['per_alpha'][str(a)]=battery(a,CTX['all'])
if len(ERAS)>1:
    OUT['per_era']={}
    for tag,_ in ERAS[1:]:
        OUT['per_era'][tag]={str(a):battery(a,CTX[tag]) for a in ALPHAS}
OUT['eras']=[t for t,_ in ERAS]
fn="exception_battery_results%s.json"%("_synthetic" if SYN else ("" if PANEL=="canon200" else "_"+PANEL))
json.dump(OUT,open(os.path.join(P,fn),"w"),indent=2)
lg("EXCEPTIONBATTERYDONE[%s] %.0fs -> %s"%(PANEL,time.time()-t0,fn))
