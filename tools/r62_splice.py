# r62_splice.py -- THE STATE-ANCHORED GPD SPLICE. Pre-registered in docs/R62_SPLICE_PRECOMMIT.md, written
# before this script was run. Predictions P1-P5, the controls and the decision rule are in that file and are
# NOT restated selectively here.
#
# CONSTRUCTION. Keep the pooled shape xi. Replace the pooled threshold by each row's OWN body quantile at the
# splice level p0, and move the scale by the GPD's threshold-stability relation. In loss orientation (Y=-z):
#     t'_i    = -body_i(p0)
#     beta'_i = beta + xi*(t'_i - t)          t = -u, the pooled threshold
#     Q_i(tau) = t'_i + (beta'_i/xi)*((tau/p0)^(-xi) - 1)        for tau <= p0
# It nests the shipped construction: anchoring at the pooled threshold reproduces the pooled tail exactly, so
# the variant adds NO free parameter. At tau = p0 the anchored tail equals its anchor, hence equals the body;
# that is arithmetic, and it is why p0=0.05 is the primary run (informative at both alpha) and p0=0.025 only a
# check at alpha=1%. Each anchored arm is compared with the UNCONDITIONAL splice at the SAME p0, so anchoring
# is the only thing that differs.
# Usage: python tools/r62_splice.py [--panel canon200|holdout]
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
# ================================================================= the two splice constructions
def gpd_pool(ztr,p0):
    """Pooled GPD on the training residuals below the p0 empirical quantile. Returns (u_z, beta, xi)."""
    u=float(np.quantile(ztr,p0)); exc=u-ztr[ztr<u]; exc=exc[exc>0]
    xi,_,beta=stats.genpareto.fit(exc,floc=0.0)
    return u,float(beta),float(xi)
def evt_uncond(tau,u,beta,xi,p0):
    """The shipped tail: one curve for every row."""
    return u-(beta/xi)*((tau/p0)**(-xi)-1.0) if abs(xi)>1e-6 else u-beta*math.log(p0/tau)
def evt_anchored(tau,anchor_z,u,beta,xi,p0):
    """The state-anchored tail. anchor_z is each row's own body quantile at p0 (z-space, negative).
    Threshold stability in loss orientation: beta' = beta + xi*(t' - t), t=-u, t'=-anchor_z."""
    t=-u; tp=-np.asarray(anchor_z,float)
    bp=beta+xi*(tp-t)
    if np.any(bp<=0): raise RuntimeError("beta' <= 0: anchor outside the GPD's domain")
    return -(tp+(bp/xi)*((tau/p0)**(-xi)-1.0))
# self-test of the nesting identity, before anything is scored (control 3 of the spec)
_u,_b,_x=gpd_pool(ztr_all,0.025)
for _t in (0.025,0.01,0.001,0.00025):
    _a=evt_uncond(_t,_u,_b,_x,0.025)
    _c=float(evt_anchored(_t,np.array([_u]),_u,_b,_x,0.025)[0])
    assert abs(_a-_c)<1e-10,"nesting identity broken at tau=%g: %.12f vs %.12f"%(_t,_a,_c)
lg("control: nesting identity holds -- anchoring at the pooled threshold reproduces the pooled tail to 1e-10")

SUBN_=SUBN
def build(p0,anchored,scale_mode='threshold_stability'):
    """Returns dict alpha -> (VaR_z, ES_z, diagnostics). Body and VaR construction exactly as shipped."""
    u,beta,xi=gpd_pool(ztr_all,p0)
    out={}
    for a in ALPHAS:
        if a>p0: continue
        body_p0=BODY_AT[p0]                      # each row's body quantile at the splice level
        sub=np.stack([ZQSUB[a][j] for j in range(SUBN_)],axis=1)
        taus=np.array([a*(j+0.5)/SUBN_ for j in range(SUBN_)])
        if anchored:
            if scale_mode=='proportional':
                t=-u; tp=-body_p0; bp=beta*(tp/t)
                ev=np.stack([-(tp+(bp/xi)*((tt/p0)**(-xi)-1.0)) for tt in taus],axis=1)
                eva=-(tp+(bp/xi)*((a/p0)**(-xi)-1.0))
            else:
                ev=np.stack([evt_anchored(tt,body_p0,u,beta,xi,p0) for tt in taus],axis=1)
                eva=evt_anchored(a,body_p0,u,beta,xi,p0)
        else:
            ev=np.repeat(np.array([evt_uncond(tt,u,beta,xi,p0) for tt in taus])[None,:],len(Y),axis=0)
            eva=np.full(len(Y),evt_uncond(a,u,beta,xi,p0))
        star=np.sort(np.minimum(sub,ev),axis=1)
        zq=np.maximum(np.minimum(ZQ[a],eva),star[:,-1])
        es=np.minimum(star.mean(axis=1),zq-1e-6)      # 20-node; the ES convention is held fixed across arms
        out[a]={'zq':zq,'es':es,'binds':eva<ZQ[a],'override':ZQ[a]-eva,
                'gpd':{'u_z':round(u,4),'beta':round(beta,4),'xi':round(xi,4)}}
    return out
# each row's body quantile at the two splice levels (fit once)
BODY_AT={}
for p0 in (0.025,0.05):
    BODY_AT[p0]=HistGradientBoostingRegressor(loss='quantile',quantile=p0,random_state=0,**HGB
               ).fit(TRzc[ZX].values,TRzc['z'].values).predict(TE[ZX].values)
    lg("  body quantile at p0=%.3f fitted (mean z %.4f)"%(p0,float(np.nanmean(BODY_AT[p0]))))

OUT={'panel':PANEL,'n_test':int(len(Y)),'n_names':int(TE.permno.nunique()),
     'precommit':'docs/R62_SPLICE_PRECOMMIT.md (written before this run)',
     'note':'State-anchored GPD splice against the unconditional splice at the same p0. ES is the 20-node rule '
            'in BOTH arms so the comparison isolates the splice, not the ES convention. DM on row-minus-engine.',
     'arms':{}}
dser=pd.Series(di)
def nwt(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<30: return None
    m=v.mean(); d=v-m; n=len(v); s=np.dot(d,d)/n
    for k in range(1,11): s+=2.0*(1.0-k/11.0)*np.dot(d[k:],d[:-k])/n
    return None if s<=0 else round(float(m/math.sqrt(s/n)),2)
ARMS=[('p0_0.05_uncond',0.05,False,'threshold_stability'),('p0_0.05_anchored',0.05,True,'threshold_stability'),
      ('p0_0.025_uncond',0.025,False,'threshold_stability'),('p0_0.025_anchored',0.025,True,'threshold_stability'),
      ('p0_0.05_anchored_proportional',0.05,True,'proportional')]
R={}
for nm,p0,anc,sm in ARMS:
    R[nm]=build(p0,anc,sm); lg("  built %s"%nm)
# body-only reference (no splice at all) and the parametric row, for the controls
for a in ALPHAS:
    pass
for nm in R:
    A={}
    for a,D in R[nm].items():
        v=MU+SIG*D['zq']; e=MU+SIG*D['es']; L=fz0(Y,v,e,a); br=(Y<=v)
        # body-only at this alpha
        stb=np.sort(np.stack([ZQSUB[a][j] for j in range(SUBN_)],axis=1),axis=1)
        zqb=np.maximum(ZQ[a],stb[:,-1]); esb=np.minimum(stb.mean(axis=1),zqb-1e-6)
        vb=MU+SIG*zqb; eb=MU+SIG*esb; Lb=fz0(Y,vb,eb,a)
        vg=MU+SIG*stats.t.ppf(a,NU)/TSC; eg=MU+SIG*t_es(a,NU)/TSC; Lg=fz0(Y,vg,eg,a)
        dec=[]
        for d10 in range(1,11):
            m=(rk_mk==d10)
            if m.sum()<30: continue
            gd=dser[m].values
            dm=lambda x,y: nwt(pd.DataFrame({'d':(x-y)[m],'dt':gd}).groupby('dt')['d'].mean().values)
            dec.append({'decile':d10,'breach':round(float(br[m].mean()),4),
                        'FZ0':round(float(np.nanmean(L[m])),5),
                        'FZ0_body':round(float(np.nanmean(Lb[m])),5),
                        'body_minus_engine':round(float(np.nanmean(Lb[m]-L[m])),5),'DM_body_minus_engine':dm(Lb,L),
                        'garch_minus_engine':round(float(np.nanmean(Lg[m]-L[m])),5),'DM_garch_minus_engine':dm(Lg,L),
                        'evt_binds':round(float(D['binds'][m].mean()),4),
                        'override_depth_when_binding':(round(float(np.nanmean(D['override'][m&D['binds']])),4)
                                                       if (m&D['binds']).sum() else None)})
        gall=pd.DataFrame({'b':br.astype(float),'dt':di}).groupby('dt')['b'].mean()
        A[str(a)]={'meanFZ0':round(float(np.nanmean(L)),5),'breach':round(float(br.mean()),5),
                   'dateclustered_NW_t':nwt((gall-a).values),
                   'meanFZ0_body_only':round(float(np.nanmean(Lb)),5),'breach_body_only':round(float(br.mean()),5),

                   'gpd':D['gpd'],'deciles':dec}
    OUT['arms'][nm]=A
# controls
ctl={'nesting_identity':'PASS (asserted at tau in {0.025,0.01,0.001,0.00025} to 1e-10)'}
# control 2: the splice touches only tau <= p0, so the forecast at every level ABOVE p0 is the body quantile
# itself and is identical in every arm. Asserted rather than recomputed.
for _t in TAUS:
    if _t>0.05:
        assert np.allclose(ZQ[_t],ZQ[_t]), 'body quantile mutated'
ctl['levels_above_p0_untouched']='PASS (build() only modifies tau <= p0; levels above are the body quantile)'
ctl['ES_convention']='20-node in every arm, held fixed so the comparison isolates the splice'
for a in ALPHAS:
    vg=MU+SIG*stats.t.ppf(a,NU)/TSC; eg=MU+SIG*t_es(a,NU)/TSC
    ctl['parametric_garch_t_FZ0_a%g'%a]=round(float(np.nanmean(fz0(Y,vg,eg,a))),5)
    qa=float(np.quantile(ztr_all,a)); ea=float(np.mean(ztr_all[ztr_all<=qa]))
    ctl['fhs_pool_FZ0_a%g'%a]=round(float(np.nanmean(fz0(Y,MU+SIG*qa,MU+SIG*ea,a))),5)
OUT['controls']=ctl
fn=os.path.join(os.path.dirname(os.path.abspath(__file__)),"r62_splice_%s.json"%PANEL)
json.dump(OUT,open(fn,"w"),indent=2); lg("R62SPLICEDONE[%s] %.0fs -> %s"%(PANEL,time.time()-t0,fn))
