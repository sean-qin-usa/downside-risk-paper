# job_synthetic_truth.py -- SYNTHETIC-TRUTH AND PLACEBO TESTS FOR THE FRONTIER CLAIM (Table OA.16).
# Two data-generating processes with a known answer, plus three shift placebos on the real panel:
#   garch       data from a per-name GARCH(1,1)-t; the TRUE model is param_garch_t. A frontier that were an
#               artifact of the flexible shape would show the engine beating the truth, so it must not.
#   bteg        the mirror image, generated from Beta-t-EGARCH; the true model is param_bteg.
#   placebo{,5,20}  real panel, every forecast scored against the return k days later. Diagnostic value is
#               LIMITED and the result file says so: both models share the same GARCH scale, so a shape
#               advantage survives any shift that preserves the scale. These rows bound how much of the
#               frontier a pure timing artifact could explain; they do not test for one.
# Engine ES is the CONVERGED integral (es_integral: body interpolated on [alpha/40, alpha], pooled GPD exact
# at every node, sub-floor region in closed form); parametric ES is closed form. The 20-node value is kept
# beside it because the engine-versus-truth FZ0 gaps are smaller than the 20-node error they carried, so
# their sign was not identified under the committed convention.
# Usage:  python job_synthetic_truth.py {garch|bteg|placebo|placebo5|placebo20}
#         then python make_synthetic_truth_results.py to consolidate the runs into
#         results/paper/synthetic_truth_results.json, the file Table OA.16 cites.
# Set GBC_PROJ to override the project path.
import os,sys,math,json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.interpolate import PchipInterpolator
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
MODE=sys.argv[1]
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ZX=['logsig','zl1','absz5','zstd21','fracdn5']
P0=0.025; HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06,random_state=0); NN=200; T=2768
def pin(y,q,t): d=y-q; return np.where(d>=0,t*d,(t-1)*d)
def nw_t(x,l=10):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<30: return None
    d=x-x.mean(); v=np.mean(d*d)
    for k in range(1,l+1): v+=2*(1-k/(l+1))*np.mean(d[k:]*d[:-k])
    return round(float(x.mean()/math.sqrt(max(v/n,1e-16))),2)
def _bf(y,mu,om,phi,kap,nu):
    n=len(y); lam=np.empty(n); lam[0]=om
    for k in range(n-1):
        e=(y[k]-mu)*math.exp(-lam[k]); e2=e*e
        lam[k+1]=min(max(om+phi*(lam[k]-om)+kap*((nu+1.0)*e2/(nu+e2)-1.0),-12.0),12.0)
    return lam
def _bn(th,y):
    mu,om=th[0],th[1]; phi=1/(1+math.exp(-th[2])); kap=math.exp(th[3]); nu=2.05+math.exp(th[4])
    if kap>2 or nu>200: return 1e12
    lam=_bf(y,mu,om,phi,kap,nu); e=(y-mu)*np.exp(-lam)
    c=math.lgamma(.5*(nu+1))-math.lgamma(.5*nu)-.5*math.log(math.pi*nu)
    v=float(np.sum(c-lam-.5*(nu+1)*np.log1p(e*e/nu))); return -v if np.isfinite(v) else 1e12
def fit_bteg(y,sp):
    ytr=np.asarray(y[:sp],float); best=None
    for nu0,kap0,phi0 in ((6.,.08,.98),(4.,.15,.95)):
        x0=np.array([float(np.mean(ytr)),math.log(max(np.std(ytr),1e-6))-.5*math.log(nu0/(nu0-2)),
                     math.log(phi0/(1-phi0)),math.log(kap0),math.log(nu0-2.05)])
        try: r=optimize.minimize(_bn,x0,args=(ytr,),method='Nelder-Mead',options={'maxiter':3000,'maxfev':4500,'xatol':1e-5,'fatol':1e-5})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    mu,om=float(best.x[0]),float(best.x[1]); phi=1/(1+math.exp(-best.x[2])); kap=math.exp(best.x[3]); nu=2.05+math.exp(best.x[4])
    lam=_bf(np.asarray(y,float),mu,om,phi,kap,nu); tsc=math.sqrt(nu/(nu-2)) if nu>2 else 1
    return np.exp(lam)*tsc,mu,nu,tsc
# ---- data
rs=np.random.default_rng(11); SER={}
if MODE=='garch':
    for i in range(NN):
        om,al,be,nu=0.05,0.07,0.90,4.5+rs.uniform(-.5,3); s2=np.empty(T); e=np.empty(T); s2[0]=om/(1-al-be)
        for k in range(T):
            if k: s2[k]=om+al*e[k-1]**2+be*s2[k-1]
            e[k]=math.sqrt(s2[k])*stats.t.rvs(nu,random_state=rs)/math.sqrt(nu/(nu-2))
        SER[i]=0.02+e
elif MODE=='bteg':
    for i in range(NN):
        om,phi,kap,nu=math.log(1.3),0.96,0.08,4.0+rs.uniform(0,3); lam=om; y=np.empty(T)
        for k in range(T):
            eps=stats.t.rvs(nu,random_state=rs); y[k]=0.02+math.exp(lam)*eps
            lam=om+phi*(lam-om)+kap*((nu+1)*eps**2/(nu+eps**2)-1)
        SER[i]=y
else:
    rr=pd.read_csv(P+"/crsp_panel_returns.csv",dtype={'permno':'int32'}); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100
    cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False)
    for pn in cnt[cnt>=1500].index.tolist()[:NN]: SER[pn]=rr[rr.permno==pn].sort_values('date')['ret'].values.astype(float)
def feats(y,sig,mu):
    z=(y-mu)/np.maximum(sig,1e-6); d=pd.DataFrame({'y':y,'sig':sig,'z':z})
    d['logsig']=np.log(np.maximum(sig,1e-6)); d['zl1']=d['z'].shift(1)
    d['absz5']=d['z'].abs().rolling(5,min_periods=3).mean().shift(1)
    d['zstd21']=d['z'].rolling(21,min_periods=8).std().shift(1)
    d['fracdn5']=(d['y']<0).rolling(5,min_periods=3).mean().shift(1)
    d['mk63']=d['z'].rolling(63,min_periods=30).kurt().shift(1); return d
TRg=[];TRb=[];rows=[]
for pn,y in SER.items():
    n=len(y); sp=int(n*.6); cp=int(sp*.75)
    try:
        r1=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False); pp=r1.params
        om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
        e=y-mu; s2=np.empty(n); s2[0]=np.var(y[:sp])
        for k in range(1,n): s2[k]=max(om+al*e[k-1]**2+be*s2[k-1],1e-8)
        d1=feats(y,np.sqrt(s2),mu); d1['mu']=mu;d1['nu']=nu;d1['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1
        sd,mb,nb,tb=fit_bteg(y,sp); d2=feats(y,sd,mb); d2['mu']=mb;d2['nu']=nb;d2['tsc']=tb
    except Exception: continue
    base=d1.copy(); base['idx']=np.arange(n)
    for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__b']=d2[c].values
    ok=base.dropna(subset=ZX+['mk63']+[c+'__b' for c in ZX+['mk63']])
    trn=ok[ok['idx']<cp]; tst=ok[ok['idx']>=sp]
    if len(tst)<60 or len(trn)<200: continue
    TRg.append(trn[ZX+['z']]); TRb.append(trn[[c+'__b' for c in ZX+['z']]].rename(columns=lambda c:c[:-3]))
    t2=tst.copy(); t2['pn']=pn; t2['dt']=np.arange(len(tst)); rows.append(t2)
TE=pd.concat(rows).reset_index(drop=True); TRgc=pd.concat(TRg); TRbc=pd.concat(TRb)
if MODE.startswith('placebo'):           # score each forecast against a return K days later
    K=int(MODE[7:]) if len(MODE)>7 else 1
    TE=TE.sort_values(['pn','idx']); TE['y']=TE.groupby('pn')['y'].shift(-K); TE=TE.dropna(subset=['y']).reset_index(drop=True)
    print("   PLACEBO shift = %d trading day(s)"%K,flush=True)
Y=TE['y'].values; di=TE['dt'].values
print("[%s] %d names %d test rows"%(MODE,TE['pn'].nunique(),len(TE)),flush=True)
Q={}
for f,TRz,sfx in (('garch_t',TRgc,''),('bteg',TRbc,'__b')):
    ztr=TRz['z'].values; uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]
    xi,_,bt=stats.genpareto.fit(exc,floc=0.0)
    evt=lambda t:(uu-(bt/xi)*((t/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-bt*math.log(P0/t))
    X=TE[[c+sfx for c in ZX]].values; MU=TE['mu'+sfx].values; SIG=TE['sig'+sfx].values
    NU=TE['nu'+sfx].values; TSC=TE['tsc'+sfx].values
    ZQ={t:HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TRz[ZX].values,ztr).predict(X) for t in TAUS}
    eng={t:(np.minimum(ZQ[t],evt(t)) if t<=P0 else ZQ[t]) for t in TAUS}
    E=np.sort(np.stack([eng[t] for t in TAUS],axis=1),axis=1)
    Q['engine_'+f]={t:MU+SIG*E[:,j] for j,t in enumerate(TAUS)}
    Q['param_'+f]={t:MU+SIG*stats.t.ppf(t,NU)/TSC for t in TAUS}
PL={m:np.mean([pin(Y,Q[m][t],t) for t in TAUS],axis=0) for m in Q}
print("  mean 11-level pinball:")
for m in sorted(PL,key=lambda k:np.nanmean(PL[k])): print("     %-16s %.6f"%(m,np.nanmean(PL[m])),flush=True)
rk=np.full(len(Y),-1); fm=np.isfinite(TE['mk63'].values)
rk[fm]=pd.qcut(pd.Series(TE['mk63'].values[fm]),10,labels=False,duplicates='drop').values+1
def dm(a1,a2,msk):
    msk=msk&np.isfinite(a1)&np.isfinite(a2)
    g=pd.DataFrame({'d':(a1-a2)[msk],'dt':di[msk]}).groupby('dt')['d'].mean(); return nw_t(g.values)
ALLM=np.ones(len(Y),bool)
print("  frontier (engine over its own-scale parametric), DM>0 = engine better:")
for f in ('garch_t','bteg'):
    print("     %-8s overall DM %6s | top mk63 decile DM %6s"%(f,dm(PL['param_'+f],PL['engine_'+f],ALLM),dm(PL['param_'+f],PL['engine_'+f],rk==10)),flush=True)
print("  engine_garch_t vs param_bteg, DM>0 = engine_garch_t better: overall %s"%dm(PL['param_bteg'],PL['engine_garch_t'],ALLM))
# ---- FZ0 at both regulatory levels, and a JSON record for the Online Appendix table
def t_es(a,nu):
    q=stats.t.ppf(a,nu); return -stats.t.pdf(q,nu)*(nu+q*q)/((nu-1)*a)
def fz0(y,v,e,a):
    v=np.minimum(v,-1e-8); e=np.minimum(e,v); return -(1.0/(a*e))*((y<=v).astype(float))*(v-y)+v/e+np.log(-e)-1.0
FZ={}
for a in (0.01,0.025):
    FZ[str(a)]={}
    for f in ('garch_t','bteg'):
        sfx='' if f=='garch_t' else '__b'
        MU=TE['mu'+sfx].values; SIG=TE['sig'+sfx].values; NU=TE['nu'+sfx].values; TSC=TE['tsc'+sfx].values
        FZ[str(a)]['param_'+f]=float(np.nanmean(fz0(Y,MU+SIG*stats.t.ppf(a,NU)/TSC,MU+SIG*t_es(a,NU)/TSC,a)))
        TRz=TRgc if f=='garch_t' else TRbc; Xf=TE[[c+sfx for c in ZX]].values
        ztr2=TRz['z'].values; uu2=float(np.quantile(ztr2,P0)); ex2=uu2-ztr2[ztr2<uu2]
        xi2,_,bt2=stats.genpareto.fit(ex2,floc=0.0)
        ev2=lambda t:(uu2-(bt2/xi2)*((t/P0)**(-xi2)-1.0) if abs(xi2)>1e-6 else uu2-bt2*math.log(P0/t))
        sb=[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/20,**HGB).fit(TRz[ZX].values,ztr2).predict(Xf) for j in range(20)]
        st=np.sort(np.stack([np.minimum(sb[j],ev2(a*(j+0.5)/20)) for j in range(20)],axis=1),axis=1)
        zq2=np.maximum(np.minimum(sb[-1],ev2(a)),st[:,-1]); es2=np.minimum(st.mean(axis=1),zq2-1e-6)
        FZ[str(a)]['engine_'+f]=float(np.nanmean(fz0(Y,MU+SIG*zq2,MU+SIG*es2,a)))   # 20-node midpoint integral
        # CONVERGED ES, same four conditions as job_es_converged.py: body interpolated only, GPD exact,
        # no body fit below a/40, VaR (zq2) exactly as committed. The sub-floor GPD integral is closed form.
        floor=a/40.0
        def evt_int(F,uu=uu2,bt=bt2,xi=xi2,P0=P0):
            if F<=0: return 0.0
            if abs(xi)>1e-6: return F*uu-(bt/xi)*((P0**xi)*(F**(1.0-xi))/(1.0-xi)-F)
            return F*uu-bt*F*math.log(P0/F)-bt*F
        lev=np.unique(np.concatenate([a*((np.arange(20)+0.5)/20),
                                      np.exp(np.linspace(math.log(floor),math.log(a),40)),[a]]))
        ib={round(float(u),12):k for k,u in enumerate(lev)}
        known={round(float(a*(j+0.5)/20),12):sb[j] for j in range(20)}
        QB=np.stack([known[round(float(u),12)] if round(float(u),12) in known
                     else HistGradientBoostingRegressor(loss='quantile',quantile=float(u),**HGB).fit(TRz[ZX].values,ztr2).predict(Xf)
                     for u in lev],axis=1)
        TAIL=evt_int(floor)
        def es_conv(M,levels=lev,Q=QB):
            u=floor+(a-floor)*((np.arange(M)+0.5)/M)
            bi=PchipInterpolator(levels,Q,axis=1,extrapolate=False)(np.clip(u,levels[0],levels[-1]))
            ev=np.array([ev2(float(x)) for x in u])[None,:]
            return (TAIL+(a-floor)*np.minimum(bi,ev).mean(axis=1))/a
        esC=np.minimum(es_conv(2000),zq2-1e-6)
        hi=np.arange(0,len(lev),2); esH=np.minimum(es_conv(2000,lev[hi],QB[:,hi]),zq2-1e-6)
        FZ[str(a)]['engine_%s__convergedES'%f]=float(np.nanmean(fz0(Y,MU+SIG*zq2,MU+SIG*esC,a)))
        FZ[str(a)]['engine_%s__ES_ratio_conv_over_20node'%f]=round(float(np.nanmean(esC/es2)),5)
        FZ[str(a)]['engine_%s__relK_halfK_vs_K'%f]=round(float(np.nanmean(np.abs(esH-esC)/np.abs(esC))),6)
        FZ[str(a)]['engine_%s__breach_control'%f]=round(float(np.mean(Y<=MU+SIG*zq2)),6)   # VaR-only, must not move
rec={'dgp':MODE,'n_names':int(TE['pn'].nunique()),'n_test':int(len(Y)),'test_length_per_name':int(T),
     'mean_pinball_11tau':{m:round(float(np.nanmean(PL[m])),6) for m in PL},
     'fz0_note':'parametric ES in closed form. engine_<scale> is the committed 20-node midpoint integral; engine_<scale>__convergedES integrates the body on an interpolant with the sub-floor GPD region in closed form (2000 nodes on [a/40,a]). VaR is identical in both, so engine_<scale>__breach_control must not differ between them.',
     'fz0':{k:{m:round(v,5) for m,v in d.items()} for k,d in FZ.items()},
     'frontier_DM_engine_over_own_parametric':{f:{'overall':dm(PL['param_'+f],PL['engine_'+f],ALLM),
                                                  'top_mk63_decile':dm(PL['param_'+f],PL['engine_'+f],rk==10)} for f in ('garch_t','bteg')},
     'cross_DM_engine_garch_t_over_param_bteg':dm(PL['param_bteg'],PL['engine_garch_t'],ALLM)}
import os as _os
fn=_os.path.join(P,"synthetic_truth_%s.json"%MODE)
json.dump(rec,open(fn,"w"),indent=2); print("  wrote %s"%fn)
