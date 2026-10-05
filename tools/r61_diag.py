# r61_diag.py -- READ-ONLY DIAGNOSTICS FOR R61. No manuscript edits, no new estimator, nothing committed
# to results/. Derived from code/paper/job_garch_evt.py so that the engine, body and GARCH-t rows are built by
# the same code that reproduces the canonical accuracy layer bit-for-bit (FZ0 2.14653 / 1.84863 on canon200).
# Everything below the VE block is replaced by the four diagnostics R61 asks for.
#
# THE QUESTION (item 1). On the joint (VaR, ES) loss the engine LOSES to its own pooled body in the top mk63
# decile and ties it elsewhere. The hypothesis is a construction effect in equation (5): both models share the
# GARCH-t scale, which is too wide after a shock; the flexible body learns to pull its standardised tail
# quantile IN in exactly those states; the min-envelope then takes the minimum of that body and an
# UNCONDITIONAL pooled GPD, which can tighten but never loosen; so in the states where the body had corrected
# the scale error, the envelope puts the unconditional tail back.
#
# The decisive cells are rows where BOTH (i) the body is less extreme than the parametric Student-t quantile
# on the same scale -- it shrank the tail -- and (ii) the EVT branch is below the body, so the minimum
# overrides it. If the hypothesis holds, that share rises steeply with the mk63 decile and tracks the FZ0 loss.
# Usage: python tools/r61_diag.py [--panel canon200|holdout]   Set GBC_PROJ for the data directory.
# r61_diag.py (derived from job_garch_evt.py) -- STANDALONE GARCH-EVT (McNeil-Frey) HEAD-TO-HEAD + FOUR-WAY COMPONENT COMPARISON.
# Fills the missing benchmark flagged by the advisor and the ChatGPT referee rounds: the manuscript
# compares the engine to GARCH-t, FHS, GAS/PZC and CAViaR but never to a conventional GARCH-EVT
# tail on the same rows. This job rebuilds the exact panel, filter and splits of job_composite.py
# (11-level pinball frontier by mk63 decile and composite decile) and job_fz_fullpanel.py (FZ0 joint
# (VaR,ES) at 1% and 2.5%) and adds, on the SAME test rows:
#   garch_t        GARCH(1,1)-t per name (the paper's reference model)
#   evt_name       McNeil-Frey per name: GARCH filter + two-sided GPD tails on that name's own
#                  training residuals, threshold = empirical p0 point of the training residuals,
#                  empirical residual quantiles between the thresholds (FHS body)
#   evt_pool       same construction with ONE pooled threshold and ONE (xi,beta) per tail fitted
#                  on the pooled training residuals of all names (the conventional tail, pooled)
#   body           GARCH + pooled gradient-boosted residual body, no EVT, no rearrangement
#                  (this is the object scored in job_composite.py, i.e. the frontier headline)
#   engine         body/EVT minimum envelope + monotone rearrangement on the tau grid
#                  (Stage 3 of the paper; the FZ0 block scores this unshifted accuracy layer as the reference
#                  at both levels and reports the 97.5% conformal overlay as its own row; the pinball block
#                  carries no shift, matching job_composite.py)
# Thresholds, GPD parameters and body fits use training rows only (idx < cp = 0.45 n); the
# conformal shift uses the calibration split [cp, sp); everything is scored on idx >= sp.
# Reports: 11-tau pinball edge vs garch_t for every method, overall and by mk63 decile and by
# composite decile; head-to-head engine-vs-EVT edges with per-date NW(10) DM (top decile, bulk
# deciles 1-9, overall); FZ0 at 1% and 2.5% with breach rates and DM vs engine; GPD threshold
# diagnostics (exceedance counts, xi, beta over a p0 grid, pooled and per-name spread); how often
# the EVT branch binds inside the engine's minimum at 1% and 2.5%.
# Pre-set write-up rule (TODO C.3): engine beats evt_pool in the top decile at DM > 2 and is within
# noise in the bulk -> the frontier claim stands; top-decile edge within noise -> the estimator
# claim narrows to "the pooled shape learner matches a conventional EVT tail" and the score keeps
# its role as a monitor.
# Self-test: `python job_garch_evt.py --synthetic` builds a 24-name GARCH-t-with-jumps panel in
# memory, runs the whole pipeline and writes garch_evt_results_synthetic.json. Set GBC_PROJ to
# override the project path.
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","code","paper"))
from es_integral import converged_es, levels_and_body
from scipy import stats, optimize
try:
    from arch import arch_model; GARCH_BACKEND="arch"
except ImportError:                       # sandbox self-test only; the Windows run uses arch
    GARCH_BACKEND="builtin_qml"
    class _Res:
        def __init__(s,p): s.params=p
    def _garch_t_fit(y):
        y=np.asarray(y,float); n=len(y); v0=np.var(y)
        def nll(th):
            mu,lom,la,lb,lnu=th; om=math.exp(lom); a=1/(1+math.exp(-la)); b=(1-a)/(1+math.exp(-lb)); nu=2.05+math.exp(lnu)
            e=y-mu; s2=np.empty(n); s2[0]=v0
            for k in range(1,n): s2[k]=om+a*e[k-1]**2+b*s2[k-1]
            sc=math.sqrt(nu/(nu-2))                     # standardized-t: unit variance
            zt=e/np.sqrt(s2)*sc
            ll=stats.t.logpdf(zt,nu)+math.log(sc)-0.5*np.log(s2)
            return -float(np.sum(ll)) if np.isfinite(ll).all() else 1e12
        x0=[float(np.mean(y)),math.log(0.05*v0),math.log(0.1/0.9),math.log(0.9/0.1),math.log(6-2.05)]
        r=optimize.minimize(nll,x0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-6,'fatol':1e-6})
        mu,lom,la,lb,lnu=r.x; a=1/(1+math.exp(-la)); b=(1-a)/(1+math.exp(-lb))
        return _Res({'mu':mu,'omega':math.exp(lom),'alpha[1]':a,'beta[1]':b,'nu':2.05+math.exp(lnu)})
    class _AM:
        def __init__(s,y,**k): s.y=y
        def fit(s,**k): return _garch_t_fit(s.y)
    def arch_model(y,**k): return _AM(y)
PANEL=sys.argv[sys.argv.index("--panel")+1] if "--panel" in sys.argv else "canon200"
assert PANEL in ("canon200","holdout")
P=os.environ.get("GBC_PROJ",r"C:\Users\OWNER\Claude\Projects\GBC Project"); t0=time.time(); lg=lambda s:print(s,flush=True)
SYN="--synthetic" in sys.argv
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]
ALPHAS=[0.01,0.025]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']
P0_ENGINE=0.025          # engine EVT threshold (job_fz_fullpanel.py)
P0_MF=0.10               # McNeil-Frey threshold: 10% of the training residuals per tail
P0_GRID=[0.025,0.05,0.10,0.15]
SUBN=20                  # sub-alpha midpoint nodes for the integrated ES (job_fz_fullpanel.py)
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06)

# ---------------------------------------------------------------- data
if SYN:
    rng=np.random.default_rng(7); recs=[]
    dates=pd.bdate_range("2004-01-01",periods=2600)
    for pn in range(24):
        om,al,be,nu=0.04,0.08,0.90,5.0+rng.uniform(-1,3)
        s2=np.empty(2600); e=np.empty(2600); s2[0]=om/(1-al-be)
        for k in range(2600):
            if k: s2[k]=om+al*e[k-1]**2+be*s2[k-1]
            z=stats.t.rvs(nu,random_state=rng)/math.sqrt(nu/(nu-2))
            if rng.uniform()<0.02: z-=rng.exponential(2.5)   # left jumps: real skew/kurtosis stress
            e[k]=math.sqrt(s2[k])*z
        recs.append(pd.DataFrame({'permno':pn,'date':dates,'ret':(0.02+e)/100.0}))
    rr=pd.concat(recs); NMAX=24
else:
    rr=pd.read_csv(os.path.join(P,"holdout_panel_2000_2013.csv" if PANEL=="holdout" else "crsp_panel_returns.csv"),dtype={'permno':'int32'}); NMAX=200
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:NMAX]

# ---------------------------------------------------------------- helpers
def pin(y,q,t): dd=y-q; return np.where(dd>=0,t*dd,(t-1)*dd)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    dm=x-x.mean(); v=np.mean(dm*dm)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(dm[k:]*dm[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def conf_ostat(sc,tau):
    n=len(sc); k=int(math.ceil((n+1)*tau)); k=min(max(k,1),n)
    return float(np.sort(np.asarray(sc,float))[k-1])
def fz0(r,v,e,a):
    v=np.minimum(v,-1e-8); e=np.minimum(e,v)
    hit=(r<=v).astype(float)
    return -(1.0/(a*e))*hit*(v-r)+v/e+np.log(-e)-1.0
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)
class GPDTail:
    """Lower-tail GPD on standardized residuals z: threshold u = empirical p0 quantile of z (training),
    excesses u - z for z < u, ML fit with location 0 (McNeil-Frey 2000)."""
    def __init__(self,z,p0,sign=-1):
        z=np.asarray(z,float); z=z[np.isfinite(z)]
        self.sign=sign; zz=sign*z                       # sign=-1: lower tail as losses; +1: upper tail
        self.u=float(np.quantile(zz,1-p0)); exc=zz[zz>self.u]-self.u
        self.p0=p0; self.n_exc=int(len(exc)); self.ok=self.n_exc>=20
        if self.ok:
            self.xi,_,self.beta=stats.genpareto.fit(exc,floc=0.0)
        else: self.xi,self.beta=np.nan,np.nan
    def q(self,tau):
        """tau is the residual-space level of the quantile in z-space (tau<=p0 for lower tail, tau>=1-p0 upper)."""
        p=tau if self.sign<0 else 1-tau                 # tail probability beyond u
        xi,b,u,p0=self.xi,self.beta,self.u,self.p0
        loss=u+(b/xi)*((p/p0)**(-xi)-1.0) if abs(xi)>1e-6 else u+b*math.log(p0/p)
        return self.sign*loss
    def es(self,tau):
        """McNeil-Frey closed-form ES beyond the tau-quantile (lower tail), z-space; needs xi<1."""
        ql=-self.q(tau); xi,b,u=self.xi,self.beta,self.u
        if not (xi<1): return -np.inf
        return -(ql+(b+xi*(ql-u))/(1.0-xi))

# ---------------------------------------------------------------- panel (identical construction to the canonical jobs)
TRz=[]; CALz=[]; rows=[]; NAME_TAILS={}; NAME_DIAG={}
for pn in names:
    g=rr[rr.permno==pn].sort_values('date'); y=g['ret'].values.astype(float); dts=g['date'].values; n=len(y)
    if n<1500: continue
    sp=int(n*0.6); cp=int(sp*0.75)
    try:
        r1=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
        pp=r1.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
    except Exception: continue
    e0=y-mu; s2=np.empty(n); s2[0]=np.var(y[:sp])
    for k in range(1,n): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
    sig=np.sqrt(s2); z=(y-mu)/np.maximum(sig,1e-6); tsc=math.sqrt(nu/(nu-2)) if nu>2 else 1.0
    df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1)
    df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1)
    df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)
    df['skew63']=df['z'].rolling(63,min_periods=30).skew().abs().shift(1)
    df['jump5']=df['z'].abs().rolling(5,min_periods=3).max().shift(1)
    df['idx']=np.arange(n); df['mu']=mu; df['nu']=nu; df['tsc']=tsc
    dd=df.dropna(subset=ZX+['mk63','skew63','jump5'])
    trn=dd[dd['idx']<cp]; cal=dd[(dd['idx']>=cp)&(dd['idx']<sp)]; tst=dd[dd['idx']>=sp]
    if len(tst)<60 or len(cal)<60: continue
    # per-name McNeil-Frey tails on this name's TRAINING residuals only (idx < cp)
    ztr=z[:cp]; lo=GPDTail(ztr,P0_MF,-1); hi=GPDTail(ztr,P0_MF,+1)
    NAME_TAILS[pn]=(lo,hi,ztr)
    NAME_DIAG[pn]={str(p0):GPDTail(ztr,p0,-1) for p0 in P0_GRID}
    TRz.append(trn[ZX+['z']]); CALz.append(cal[ZX+['z']])
    t2=tst.copy(); t2['permno']=pn; rows.append(t2)
TE=pd.concat(rows).reset_index(drop=True); TRzc=pd.concat(TRz); CALzc=pd.concat(CALz)
lg("panel %d names %d test rows %.0fs"%(TE.permno.nunique(),len(TE),time.time()-t0))
Y=TE['y'].values; SIG=TE['sig'].values; MU=TE['mu'].values; NU=TE['nu'].values; TSC=TE['tsc'].values
PN=TE['permno'].values; dates=TE['date'].values
di,udates=pd.factorize(pd.to_datetime(dates),sort=True); di=np.asarray(di)

# ---------------------------------------------------------------- pooled EVT (engine threshold, as job_fz_fullpanel.py) and pooled McNeil-Frey
ztr_all=TRzc['z'].values
ENG_TAIL=GPDTail(ztr_all,P0_ENGINE,-1)
POOL_LO=GPDTail(ztr_all,P0_MF,-1); POOL_HI=GPDTail(ztr_all,P0_MF,+1)
POOL_DIAG={str(p0):GPDTail(ztr_all,p0,-1) for p0 in P0_GRID}
lg("engine GPD u=%.3f xi=%.3f beta=%.3f n_exc=%d | MF pooled lower u=%.3f xi=%.3f beta=%.3f n_exc=%d"%(
   -ENG_TAIL.u,ENG_TAIL.xi,ENG_TAIL.beta,ENG_TAIL.n_exc,-POOL_LO.u,POOL_LO.xi,POOL_LO.beta,POOL_LO.n_exc))

def mf_quantile(tail_lo,tail_hi,ztr,tau):
    if tau<=P0_MF and tail_lo.ok: return tail_lo.q(tau)
    if tau>=1-P0_MF and tail_hi.ok: return tail_hi.q(tau)
    return float(np.quantile(ztr,tau))
def mf_es(tail_lo,ztr,a):
    if tail_lo.ok and tail_lo.xi<1: return tail_lo.es(a)
    q=np.quantile(ztr,a); return float(np.mean(ztr[ztr<=q]))
# per-name z-quantiles (row-wise via permno map)
def per_name_map(fn):
    out=np.empty(len(Y)); cache={}
    for pn in np.unique(PN):
        cache[pn]=fn(pn)
    for pn,v in cache.items(): out[PN==pn]=v
    return out

# ---------------------------------------------------------------- pooled body (engine Stage 2), 11 levels + sub-alpha nodes
ZQ={}; ZQcal={}
for t in TAUS:
    m=HistGradientBoostingRegressor(loss='quantile',quantile=t,random_state=0,**HGB).fit(TRzc[ZX].values,TRzc['z'].values)
    ZQ[t]=m.predict(TE[ZX].values); ZQcal[t]=m.predict(CALzc[ZX].values)
    lg("  body tau %.3f %.0fs"%(t,time.time()-t0))
ES_LEGACY={}   # superseded 20-node ES, for the old-vs-new record
ZQSUB={a:{} for a in ALPHAS}
for a in ALPHAS:
    for j in range(SUBN):
        uu=a*(j+0.5)/SUBN
        ZQSUB[a][j]=HistGradientBoostingRegressor(loss='quantile',quantile=uu,random_state=0,**HGB).fit(TRzc[ZX].values,TRzc['z'].values).predict(TE[ZX].values)
    lg("  sub-alpha grid a=%.3f %.0fs"%(a,time.time()-t0))
CONF975=conf_ostat(CALzc['z'].values-ZQcal[0.025],0.025)
lg("conf975 %+.4f"%CONF975)

# ---------------------------------------------------------------- 11-tau pinball block (job_composite.py convention: engine = body, no shift)
Q={'garch_t':{},'evt_name':{},'evt_pool':{},'body':{},'engine':{}}
for t in TAUS:
    Q['garch_t'][t]=stats.t.ppf(t,NU)/TSC
    Q['evt_name'][t]=per_name_map(lambda pn: mf_quantile(*NAME_TAILS[pn],t))
    Q['evt_pool'][t]=np.full(len(Y),mf_quantile(POOL_LO,POOL_HI,ztr_all,t))
    Q['body'][t]=ZQ[t]
    Q['engine'][t]=np.minimum(ZQ[t],ENG_TAIL.q(t)) if t<=P0_ENGINE else ZQ[t]
# monotone rearrangement of the engine curve across the 11-level grid
Emat=np.sort(np.stack([Q['engine'][t] for t in TAUS],axis=1),axis=1)
for j,t in enumerate(TAUS): Q['engine'][t]=Emat[:,j]
PL={}
for m in Q:
    L=np.zeros(len(Y))
    for t in TAUS: L+=pin(Y,MU+SIG*Q[m][t],t)
    PL[m]=L/len(TAUS)
binds={str(a):round(float(np.mean(ENG_TAIL.q(a)<ZQ[a])),4) for a in ALPHAS}   # EVT branch binds inside the minimum

def edge(Lref,Lm,mask):
    if mask.sum()<30: return None
    d=(Lref-Lm)[mask]
    g=pd.DataFrame({'d':d,'dt':di[mask]}).groupby('dt')['d'].mean()
    return {'edge_pct':round(100*float(d.mean())/float(Lref[mask].mean()),3),'DM':nw_t(g.values),
            'n':int(mask.sum()),'occupancy_pct':round(100*float(mask.mean()),2)}
def dec(x):
    r=np.full(len(x),-1); m=np.isfinite(x)
    r[m]=pd.qcut(pd.Series(x[m]),10,labels=False,duplicates='drop').values+1
    return r
def pct(x):
    r=np.full(len(x),np.nan); m=np.isfinite(x); r[m]=pd.Series(x[m]).rank(pct=True).values
    return r
mk=TE['mk63'].values; sk=TE['skew63'].values; jp=TE['jump5'].values
ok=np.isfinite(mk)&np.isfinite(sk)&np.isfinite(jp)
rk_mk=dec(mk)
comp_pct=np.where(ok,np.fmax.reduce([pct(mk),pct(sk),pct(jp)]),np.nan); cdec=dec(comp_pct)
ALL=np.ones(len(Y),bool)
regions={'overall':ALL,'top_mk63_decile':rk_mk==10,'bulk_mk63_d1to9':(rk_mk>=1)&(rk_mk<=9),
         'top_composite_decile':cdec==10,'bulk_composite_d1to9':(cdec>=1)&(cdec<=9)}
pinball={'vs_garch_t':{m:{r:edge(PL['garch_t'],PL[m],msk) for r,msk in regions.items()} for m in Q if m!='garch_t'},
         'engine_vs_evt_pool':{r:edge(PL['evt_pool'],PL['engine'],msk) for r,msk in regions.items()},
         'engine_vs_evt_name':{r:edge(PL['evt_name'],PL['engine'],msk) for r,msk in regions.items()},
         'body_vs_evt_pool':{r:edge(PL['evt_pool'],PL['body'],msk) for r,msk in regions.items()},
         'engine_vs_body':{r:edge(PL['body'],PL['engine'],msk) for r,msk in regions.items()},
         'mk63_decile_profile':{m:{int(d0):edge(PL['garch_t'],PL[m],rk_mk==d0) for d0 in range(1,11)} for m in ['evt_pool','evt_name','body','engine']},
         'composite_decile_profile':{m:{int(d0):edge(PL['garch_t'],PL[m],cdec==d0) for d0 in range(1,11)} for m in ['evt_pool','evt_name','body','engine']},
         'mean_pinball':{m:round(float(PL[m].mean()),5) for m in Q}}
lg("pinball top mk63 decile: "+json.dumps({m:pinball['vs_garch_t'][m]['top_mk63_decile'] for m in pinball['vs_garch_t']}))
lg("engine vs evt_pool: "+json.dumps(pinball['engine_vs_evt_pool']))

# ---------------------------------------------------------------- FZ0 block (job_fz_fullpanel.py convention: coherent Q*, conformal at 97.5%)
def star(a,use_evt):
    cols=[np.minimum(ZQSUB[a][j],ENG_TAIL.q(a*(j+0.5)/SUBN)) if use_evt else ZQSUB[a][j] for j in range(SUBN)]
    return np.sort(np.stack(cols,axis=1),axis=1)
VE={}
for a in ALPHAS:
    st=star(a,True); zq=np.maximum(np.minimum(ZQ[a],ENG_TAIL.q(a)),st[:,-1])
    # CONVERGED ES for the engine row. VaR (zq) is the committed construction and is untouched, so every
    # VaR-only statistic and every pinball number is unchanged. GARCH-EVT's own ES is the McNeil-Frey closed
    # form and carries no quadrature error, so the DMs in this table move only through the engine side.
    for _t in (a/40.0,0.001,a):        # guard on the GPD parameterisation the closed form assumes
        assert abs(((-ENG_TAIL.u)-(ENG_TAIL.beta/ENG_TAIL.xi)*((_t/P0_ENGINE)**(-ENG_TAIL.xi)-1.0))-ENG_TAIL.q(_t))<1e-9,'GPD mismatch'
    _lev,_QB=levels_and_body(a,[ZQSUB[a][j] for j in range(SUBN)],
        lambda t:HistGradientBoostingRegressor(loss='quantile',quantile=t,random_state=0,**HGB
                 ).fit(TRzc[ZX].values,TRzc['z'].values).predict(TE[ZX].values),subn=SUBN)
    _es20=np.minimum(st.mean(axis=1),zq-1e-6)
    es=converged_es(a,_lev,_QB,ENG_TAIL.q,(-ENG_TAIL.u,float(ENG_TAIL.beta),float(ENG_TAIL.xi),P0_ENGINE),M=2000,var_z=zq)
    ES_LEGACY[str(a)]={'es_20node':round(float(np.nanmean(_es20)),5),'es_converged':round(float(np.nanmean(es)),5),
                       'ratio':round(float(np.nanmean(es/_es20)),5)}
    lg('  a=%g engine ES 20-node %.5f -> converged %.5f (ratio %.5f)'%(a,np.nanmean(_es20),np.nanmean(es),ES_LEGACY[str(a)]['ratio']))
    stb=star(a,False); zqb=np.maximum(ZQ[a],stb[:,-1]); esb=np.minimum(stb.mean(axis=1),zqb-1e-6)
    sh=CONF975 if a==0.025 else 0.0
    # accuracy layer (no shift) is the reference at BOTH levels, as in the paper; the overlay is its own row
    VE[a]={'engine':(MU+SIG*zq,MU+SIG*es),
           'engine_overlay':(MU+SIG*(zq+sh),MU+SIG*(es+sh)),
           'body':(MU+SIG*zqb,MU+SIG*esb),
           'garch_t':(MU+SIG*stats.t.ppf(a,NU)/TSC,MU+SIG*t_es(a,NU)/TSC),
           # pooled filtered historical simulation, the same construction as fhs_<scale> in the battery:
           # the training residual quantile and the mean of the residuals at or below it
           'fhs_pool':(MU+SIG*float(np.quantile(ztr_all,a)),
                       MU+SIG*float(np.mean(ztr_all[ztr_all<=np.quantile(ztr_all,a)]))),
           }

# ================================================================= R61 DIAGNOSTICS (read-only)
OUT={'panel':PANEL,'n_names':int(TE.permno.nunique()),'n_test':int(len(Y)),'n_dates':int(len(np.unique(di))),
     'note':'Read-only R61 diagnostics. engine/body/garch_t built by the job that reproduces the canonical '
            'accuracy layer. mk63 deciles are formed on the test rows, decile 10 = highest kurtosis score.',
     'gpd':{'u_z':round(float(-ENG_TAIL.u),4),'xi':round(float(ENG_TAIL.xi),4),'beta':round(float(ENG_TAIL.beta),4)},
     'per_alpha':{}}
dser=pd.Series(di)
def nwt(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<30: return None
    m=v.mean(); d=v-m; n=len(v); s=np.dot(d,d)/n
    for k in range(1,11): s+=2.0*(1.0-k/11.0)*np.dot(d[k:],d[:-k])/n
    return None if s<=0 else round(float(m/math.sqrt(s/n)),2)
for a in ALPHAS:
    ve,ee=VE[a]['engine']; vb,eb=VE[a]['body']; vg,eg=VE[a]['garch_t']
    Le=fz0(Y,ve,ee,a); Lb=fz0(Y,vb,eb,a); Lg=fz0(Y,vg,eg,a)
    be=(Y<=ve); bb=(Y<=vb); bg=(Y<=vg)
    # z-space quantiles at the alpha node: body (conditional), EVT (pooled, unconditional), parametric t
    zb=ZQ[a]; zev=np.full(len(Y),float(ENG_TAIL.q(a))); zt=stats.t.ppf(a,NU)/TSC
    shrank = zb > zt              # body less extreme than the parametric shape on the same scale
    binds  = zev < zb             # EVT branch is the minimum at the alpha node
    both   = shrank & binds       # the hypothesised cells
    # share of the 20 sub-grid nodes at which EVT binds (the ES integrand, not just the VaR node)
    sub=np.stack([ZQSUB[a][j] for j in range(SUBN)],axis=1)
    evsub=np.array([float(ENG_TAIL.q(a*(j+0.5)/SUBN)) for j in range(SUBN)])[None,:]
    bind_frac_nodes=(evsub<sub).mean(axis=1)
    rows=[]
    for d10 in range(1,11):
        m=(rk_mk==d10)
        if m.sum()<30: continue
        gd=dser[m].values
        def dm(x,y):   # DM of x-y, date-clustered; >0 means x worse
            s=pd.DataFrame({'d':(x-y)[m],'dt':gd}).groupby('dt')['d'].mean().values
            return nwt(s)
        rows.append({'decile':d10,'n':int(m.sum()),
            'breach_engine':round(float(be[m].mean()),4),'breach_body':round(float(bb[m].mean()),4),
            'breach_garch_t':round(float(bg[m].mean()),4),
            'FZ0_engine':round(float(np.nanmean(Le[m])),5),'FZ0_body':round(float(np.nanmean(Lb[m])),5),
            'FZ0_garch_t':round(float(np.nanmean(Lg[m])),5),
            'body_minus_engine':round(float(np.nanmean(Lb[m]-Le[m])),5),'DM_body_minus_engine':dm(Lb,Le),
            'garch_minus_engine':round(float(np.nanmean(Lg[m]-Le[m])),5),'DM_garch_minus_engine':dm(Lg,Le),
            'evt_binds_at_alpha':round(float(binds[m].mean()),4),
            'evt_binds_share_of_20_nodes':round(float(bind_frac_nodes[m].mean()),4),
            'body_shrank_vs_parametric':round(float(shrank[m].mean()),4),
            'shrank_AND_evt_binds':round(float(both[m].mean()),4),
            'z_body_at_alpha':round(float(np.nanmean(zb[m])),4),
            'z_evt_at_alpha':round(float(zev[0]),4),
            'z_parametric_at_alpha':round(float(np.nanmean(zt[m])),4),
            'z_evt_minus_body':round(float(np.nanmean(zev[m]-zb[m])),4),
            'z_override_depth_when_binding':(round(float(np.nanmean((zb-zev)[m&binds])),4) if (m&binds).sum() else None)})
    # counterfactual: what the engine's FZ0 would be WITHOUT the override, i.e. the body curve itself.
    # This is a decomposition of an existing forecast, not a new estimator: it is exactly VE[a]['body'].
    top=(rk_mk==10); bulk=(rk_mk>=1)&(rk_mk<=9)
    A={'deciles':rows,
       'top_vs_bulk':{
         'top_decile':{'FZ0_engine':round(float(np.nanmean(Le[top])),5),'FZ0_body':round(float(np.nanmean(Lb[top])),5),
                       'body_minus_engine':round(float(np.nanmean((Lb-Le)[top])),5),
                       'DM':nwt(pd.DataFrame({'d':(Lb-Le)[top],'dt':dser[top].values}).groupby('dt')['d'].mean().values),
                       'evt_binds_at_alpha':round(float(binds[top].mean()),4),
                       'shrank_AND_evt_binds':round(float(both[top].mean()),4)},
         'deciles_1to9':{'FZ0_engine':round(float(np.nanmean(Le[bulk])),5),'FZ0_body':round(float(np.nanmean(Lb[bulk])),5),
                       'body_minus_engine':round(float(np.nanmean((Lb-Le)[bulk])),5),
                       'DM':nwt(pd.DataFrame({'d':(Lb-Le)[bulk],'dt':dser[bulk].values}).groupby('dt')['d'].mean().values),
                       'evt_binds_at_alpha':round(float(binds[bulk].mean()),4),
                       'shrank_AND_evt_binds':round(float(both[bulk].mean()),4)}},
       'correlation_across_deciles':{
         'bind_share_vs_body_minus_engine':round(float(np.corrcoef([r['evt_binds_at_alpha'] for r in rows],
                                                                   [r['body_minus_engine'] for r in rows])[0,1]),4),
         'shrank_and_binds_vs_body_minus_engine':round(float(np.corrcoef([r['shrank_AND_evt_binds'] for r in rows],
                                                                   [r['body_minus_engine'] for r in rows])[0,1]),4)}}
    # ---- item 2: per-date breach structure and the power of the clustered test
    pdate={}
    for nm,bx in (('engine',be),('body',bb),('garch_t',bg)):
        per=pd.DataFrame({'b':bx.astype(float),'dt':di}).groupby('dt')['b'].mean()
        nper=pd.DataFrame({'b':bx.astype(float),'dt':di}).groupby('dt')['b'].size()
        nbar=float(nper.mean()); var_d=float(per.var(ddof=1)); mb=float(per.mean())
        se_cl=math.sqrt(var_d/len(per))                       # clustered SE of the mean breach
        se_iid=math.sqrt(mb*(1-mb)/len(bx))
        de=(se_cl/se_iid)**2 if se_iid>0 else None
        rho=(de-1.0)/(nbar-1.0) if de and nbar>1 else None
        pdate[nm]={'mean_breach':round(mb,5),'n_dates':int(len(per)),'mean_names_per_date':round(nbar,1),
          'clustered_SE_pp':round(100*se_cl,3),'iid_SE_pp':round(100*se_iid,3),
          'design_effect':round(de,1) if de else None,
          'implied_exchangeable_rho':round(rho,3) if rho is not None else None,
          'rho_is_a_valid_correlation':bool(rho is not None and 0<=rho<=1),
          'min_detectable_breach_two_sided_5pct':[round(a-1.96*se_cl,5),round(a+1.96*se_cl,5)],
          'date_breach_p99':round(float(per.quantile(0.99)),4),'date_breach_max':round(float(per.max()),4),
          'share_of_clustered_variance_from_top5_dates':round(float(
              ((per-mb)**2).sort_values(ascending=False).head(5).sum()/((per-mb)**2).sum()),4),
          'n_dates_with_breach_above_10pct':int((per>0.10).sum())}
    A['per_date_breach_structure']=pdate
    # ---- item 3: McNeil-Frey exceedance residual by decile
    mf={}
    for nm,(vv,eeh) in (('engine',VE[a]['engine']),('garch_t',VE[a]['garch_t']),('fhs_pool',VE[a]['fhs_pool'])):
        zres=(Y-eeh)/SIG                     # standardised shortfall residual, zero-mean under correct ES
        br=(Y<=vv)
        per=[]
        for d10 in range(1,11):
            m=(rk_mk==d10)&br
            per.append({'decile':d10,'n_breach':int(m.sum()),
                        'mean_exceedance_residual':round(float(np.nanmean(zres[m])),4) if m.sum()>=20 else None})
        allm=br
        mf[nm]={'overall_mean_exceedance_residual':round(float(np.nanmean(zres[allm])),4),
                'n_breach':int(allm.sum()),'by_decile':per}
    A['mcneil_frey_by_decile']=mf
    OUT['per_alpha'][str(a)]=A
    lg("  a=%g  top-decile body-minus-engine %+0.5f (DM %s) | EVT binds %.1f%% top vs %.1f%% bulk | shrank&binds %.1f%% vs %.1f%%"%(
       a,A['top_vs_bulk']['top_decile']['body_minus_engine'],A['top_vs_bulk']['top_decile']['DM'],
       100*A['top_vs_bulk']['top_decile']['evt_binds_at_alpha'],100*A['top_vs_bulk']['deciles_1to9']['evt_binds_at_alpha'],
       100*A['top_vs_bulk']['top_decile']['shrank_AND_evt_binds'],100*A['top_vs_bulk']['deciles_1to9']['shrank_AND_evt_binds']))
fn=os.path.join(os.path.dirname(os.path.abspath(__file__)),"r61_diag_%s.json"%PANEL)
json.dump(OUT,open(fn,"w"),indent=2); lg("R61DIAGDONE[%s] %.0fs -> %s"%(PANEL,time.time()-t0,fn))
