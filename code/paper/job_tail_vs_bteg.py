# job_tail_vs_bteg.py -- WHERE DOES THE FLEXIBLE TAIL ADD, ON A ROBUST SCALE?
# A plain Beta-t-EGARCH beats the paper's GARCH-t engine on eleven-level pinball. Eleven-level pinball
# averages the middle of the distribution, where a flexible tail cannot add anything; the joint (VaR, ES)
# loss and the two regulatory levels score the tail, which is what capital depends on. This job separates
# the two by reporting pinball AT EACH LEVEL rather than averaged, under both protocols.
# Stage 1 of the paper's engine stays GARCH-t; the robust-scale configuration is an additional result.
#
# Rows: the canonical 200-name panel. --protocol frozen = the 60/40 split of job_robust_engine.py;
# --protocol refit = the Jan-1 2020-2024 annual refits of job_walkforward_bteg.py, every stage re-estimated
# on expanding pre-cutoff data. Models: engine (GARCH-t scale, body/EVT minimum, rearranged), param_garch_t,
# engine_bteg, param_bteg. Reference for every comparison is param_bteg.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01), decisions included:
#  P1. UNDER REFITS, engine_bteg beats param_bteg at the 0.01 and 0.025 levels individually and on FZ0
#      overall at both levels, DM > 2 in every one of those four comparisons. If any falls below 2, the
#      claim "once the scale is fixed the flexible tail still adds where capital is measured" does not
#      survive the recommended schedule and must be stated for frozen fits only.
#  P2. At CENTRAL levels (0.25, 0.50, 0.75) engine_bteg ties or loses to param_bteg, |DM| < 2 or negative.
#      This is the mechanism behind the pinball result: averaging eleven levels dilutes a tail-only gain
#      with central levels where the flexible shape cannot help.
#  P3. Consequently the tail-only average (0.01 and 0.025 only) favours engine_bteg while the eleven-level
#      average favours param_bteg, under both protocols. If the eleven-level average also favours
#      engine_bteg the dilution story is wrong and the pinball concession can be narrowed.
#  P4. The GARCH-t engine loses to param_bteg at the tail levels under frozen fits, since param_bteg beat it
#      on FZ0 at 2.5% (DM 2.53, robust_engine_results.json); under refits the two draw level.
# Output: tail_vs_bteg_results_{frozen,refit}.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from es_integral import converged_es, levels_and_body
from scipy import stats, optimize
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
PROT=sys.argv[sys.argv.index("--protocol")+1] if "--protocol" in sys.argv else "frozen"
assert PROT in ("frozen","refit")
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
TAILT=[0.01,0.025]; CENT=[0.25,0.50,0.75]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']; SUBN=20; P0=0.025
ES_LEGACY={}   # superseded 20-node ES, kept so the paper can print old beside new
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06,random_state=0)
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
rr=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:200]
SER={pn:(g['ret'].values.astype(float),pd.to_datetime(g['date'].values)) for pn,g in ((p,rr[rr.permno==p].sort_values('date')) for p in names)}

def build(y,dts,sp,n):
    """both scales filtered over y[:n], estimated on y[:sp]"""
    D={}
    r1=arch_model(y[:sp],vol='Garch',p=1,q=1,dist='t',rescale=False).fit(disp='off',show_warning=False)
    pp=r1.params; om,al,be,mu=float(pp['omega']),float(pp['alpha[1]']),float(pp['beta[1]']),float(pp.get('mu',0)); nu=float(pp.get('nu',8))
    e0=y[:n]-mu; s2=np.empty(n); s2[0]=np.var(y[:sp])
    for k in range(1,n): s2[k]=max(om+al*e0[k-1]**2+be*s2[k-1],1e-8)
    d=feats(y[:n],np.sqrt(s2),mu,dts[:n]); d['mu']=mu; d['nu']=nu; d['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0; D['garch_t']=d
    sd,mb,nb,tb=fit_bteg(y[:n],sp)
    d2=feats(y[:n],sd,mb,dts[:n]); d2['mu']=mb; d2['nu']=nb; d2['tsc']=tb; D['bteg']=d2
    return D
def stage23(TRz,X,Xq=None,tag='?'):
    """pooled body at 11 levels + GPD tail + the 20-node sub-alpha grids; returns per-level engine curve"""
    ztr=TRz['z'].values; uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]
    xi,_,bt=stats.genpareto.fit(exc,floc=0.0)
    evt=lambda t:(uu-(bt/xi)*((t/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-bt*math.log(P0/t))
    ZQ={t:HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TRz[ZX].values,ztr).predict(X) for t in TAUS}
    eng={t:(np.minimum(ZQ[t],evt(t)) if t<=P0 else ZQ[t]) for t in TAUS}
    E=np.sort(np.stack([eng[t] for t in TAUS],axis=1),axis=1); eng={t:E[:,j] for j,t in enumerate(TAUS)}
    VE={}
    for a in ALPHAS:
        SUB=[HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,**HGB).fit(TRz[ZX].values,ztr).predict(X) for j in range(SUBN)]
        st=np.sort(np.stack([np.minimum(SUB[j],evt(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
        zq=np.maximum(np.minimum(ZQ[a],evt(a)),st[:,-1])
        # CONVERGED ES. VaR (zq) is the committed construction and is untouched, so no VaR-only statistic
        # can move; the body is interpolated on [a/40,a] and the sub-floor GPD region is closed form.
        _lev,_QB=levels_and_body(a,SUB,lambda u:HistGradientBoostingRegressor(loss='quantile',quantile=u,**HGB).fit(TRz[ZX].values,ztr).predict(X),subn=SUBN)
        _es20=np.minimum(st.mean(axis=1),zq-1e-6)
        _esC=converged_es(a,_lev,_QB,evt,(uu,float(bt),float(xi),P0),M=2000,var_z=zq)
        ES_LEGACY.setdefault(str(a),{})[tag]={'es_20node':round(float(np.nanmean(_es20)),5),'es_converged':round(float(np.nanmean(_esC)),5),'ratio':round(float(np.nanmean(_esC/_es20)),5)}
        VE[a]=(zq,_esC)
    return eng,VE

PIECES=[]
if PROT=="frozen":
    for pn in names:
        y,dts=SER[pn]; n=len(y)
        if n<1500: continue
        sp=int(n*0.6); cp=int(sp*0.75)
        # filters estimated on the first 60% (sp) and the learner on the first 45% (cp), as job_robust_engine.py
        try: D=build(y,dts,sp,n)
        except Exception as ex: lg("  fail %s %s"%(pn,str(ex)[:40])); continue
        base=D['garch_t'].copy(); base['idx']=np.arange(n)
        for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__b']=D['bteg'][c].values
        ok=base.dropna(subset=ZX+['mk63']+[c+'__b' for c in ZX+['mk63']])
        trn=ok[ok['idx']<cp]; tst=ok[ok['idx']>=sp]
        if len(tst)<60 or len(trn)<200: continue
        t2=tst.copy(); t2['permno']=pn; PIECES.append((t2,trn))
        lg("  fit %s %.0fs"%(pn,time.time()-t0))
    TE=pd.concat([a for a,_ in PIECES]).reset_index(drop=True)
    TRg=pd.concat([b[ZX+['z']] for _,b in PIECES]); TRb=pd.concat([b[[c+'__b' for c in ZX+['z']]].rename(columns=lambda c:c[:-3]) for _,b in PIECES])
    REC={'y':TE['y'].values,'date':TE['date'].values,'mk':TE['mk63'].values}
    for f,TRz,sfx in (('garch_t',TRg,''),('bteg',TRb,'__b')):
        eng,VE=stage23(TRz,TE[[c+sfx for c in ZX]].values,tag='%s_frozen'%f)
        MU=TE['mu'+sfx].values; SIG=TE['sig'+sfx].values; NU=TE['nu'+sfx].values; TSC=TE['tsc'+sfx].values
        for t in TAUS:
            REC['q_engine_%s_%g'%(f,t)]=MU+SIG*eng[t]; REC['q_param_%s_%g'%(f,t)]=MU+SIG*stats.t.ppf(t,NU)/TSC
        for a in ALPHAS:
            zq,es=VE[a]
            REC['v_engine_%s_%g'%(f,a)]=MU+SIG*zq; REC['e_engine_%s_%g'%(f,a)]=MU+SIG*es
            REC['v_param_%s_%g'%(f,a)]=MU+SIG*stats.t.ppf(a,NU)/TSC; REC['e_param_%s_%g'%(f,a)]=MU+SIG*t_es(a,NU)/TSC
            REC['sig_'+f]=SIG; REC['mu_'+f]=MU
        lg("  stage23[%s] %.0fs"%(f,time.time()-t0))
    ALL=pd.DataFrame(REC)
else:
    for ty in [2020,2021,2022,2023,2024]:
        cut=pd.Timestamp("%d-01-01"%ty); nxt=pd.Timestamp("%d-01-01"%(ty+1)); rows=[]; TRg=[]; TRb=[]
        for pn in names:
            y,dts=SER[pn]; sp=int(np.searchsorted(dts,cut)); te=int(np.searchsorted(dts,nxt))
            if sp<750 or te-sp<30: continue
            try: D=build(y,dts,sp,te)
            except Exception: continue
            base=D['garch_t'].copy(); base['idx']=np.arange(te)
            for c in ['sig','z','mu','nu','tsc']+ZX+['mk63']: base[c+'__b']=D['bteg'][c].values
            ok=base.dropna(subset=ZX+['mk63']+[c+'__b' for c in ZX+['mk63']])
            trn=ok[ok['idx']<sp]; tst=ok[ok['idx']>=sp]
            if len(tst)<10 or len(trn)<200: continue
            TRg.append(trn[ZX+['z']]); TRb.append(trn[[c+'__b' for c in ZX+['z']]].rename(columns=lambda c:c[:-3]))
            t2=tst.copy(); t2['permno']=pn; rows.append(t2)
        TEc=pd.concat(rows).reset_index(drop=True); TRgc=pd.concat(TRg); TRbc=pd.concat(TRb)
        lg("cut %s: %d names %d rows %.0fs"%(cut.date(),TEc.permno.nunique(),len(TEc),time.time()-t0))
        rec={'y':TEc['y'].values,'date':TEc['date'].values,'mk':TEc['mk63'].values}
        for f,TRz,sfx in (('garch_t',TRgc,''),('bteg',TRbc,'__b')):
            eng,VE=stage23(TRz,TEc[[c+sfx for c in ZX]].values,tag='%s_refit%d'%(f,ty))
            MU=TEc['mu'+sfx].values; SIG=TEc['sig'+sfx].values; NU=TEc['nu'+sfx].values; TSC=TEc['tsc'+sfx].values
            for t in TAUS:
                rec['q_engine_%s_%g'%(f,t)]=MU+SIG*eng[t]; rec['q_param_%s_%g'%(f,t)]=MU+SIG*stats.t.ppf(t,NU)/TSC
            for a in ALPHAS:
                zq,es=VE[a]
                rec['v_engine_%s_%g'%(f,a)]=MU+SIG*zq; rec['e_engine_%s_%g'%(f,a)]=MU+SIG*es
                rec['v_param_%s_%g'%(f,a)]=MU+SIG*stats.t.ppf(a,NU)/TSC; rec['e_param_%s_%g'%(f,a)]=MU+SIG*t_es(a,NU)/TSC
            rec['sig_'+f]=SIG; rec['mu_'+f]=MU
            lg("   %s done %.0fs"%(f,time.time()-t0))
        PIECES.append(pd.DataFrame(rec))
    ALL=pd.concat(PIECES).reset_index(drop=True)

Y=ALL['y'].values; di,_=pd.factorize(pd.to_datetime(ALL['date'].values),sort=True); di=np.asarray(di)
rk=np.full(len(Y),-1); m=np.isfinite(ALL['mk'].values)
rk[m]=pd.qcut(pd.Series(ALL['mk'].values[m]),10,labels=False,duplicates='drop').values+1
REG={'overall':np.ones(len(Y),bool),'top_mk63_decile':rk==10,'deciles_1to9':(rk>=1)&(rk<=9)}
MODELS=[('engine_bteg','q_engine_bteg_%g','v_engine_bteg_%g','e_engine_bteg_%g','bteg'),
        ('engine','q_engine_garch_t_%g','v_engine_garch_t_%g','e_engine_garch_t_%g','garch_t'),
        ('param_garch_t','q_param_garch_t_%g','v_param_garch_t_%g','e_param_garch_t_%g','garch_t'),
        ('param_bteg','q_param_bteg_%g','v_param_bteg_%g','e_param_bteg_%g','bteg')]
REF='param_bteg'
def dmv(a1,a2,msk):
    msk=msk&np.isfinite(a1)&np.isfinite(a2)
    if msk.sum()<100: return None
    g=pd.DataFrame({'d':(a1-a2)[msk],'dt':di[msk]}).groupby('dt')['d'].mean()
    return {'mean_diff':round(float(np.mean((a1-a2)[msk])),6),'DM_t':nw_t(g.values),'n':int(msk.sum())}
PLL={nm:{t:pin(Y,ALL[qk%t].values,t) for t in TAUS} for nm,qk,_,_,_ in MODELS}
OUT={'es_convention':{'engine_rows':'converged integral (es_integral.converged_es, M=2000, 60 fitted body levels): body interpolated on [a/40,a], pooled GPD exact at every node, sub-floor region in closed form. VaR is the committed construction and is unchanged, so no VaR-only statistic can move.','parametric_rows':'closed form, unaffected','superseded_20node_values':ES_LEGACY},
     'note':'Where the flexible tail adds on a robust scale. Pinball reported AT EACH LEVEL, not averaged, '
     'plus a tail-only average over 0.01 and 0.025 and the eleven-level average. Reference for every DM is '
     'param_bteg; DM_t>0 means the row is WORSE than param_bteg. Protocol: '+PROT+'.',
     'predictions_written_before_run':'P1 engine_bteg beats param_bteg at 0.01 and 0.025 and on FZ0 overall '
     'at both levels, DM>2, under refits; P2 ties or loses at 0.25/0.50/0.75; P3 tail-only average favours '
     'engine_bteg while the eleven-level average favours param_bteg; P4 the GARCH-t engine loses to '
     'param_bteg at tail levels under frozen fits and draws level under refits.',
     'protocol':PROT,'n_test':int(len(Y)),'n_dates':int(len(set(di))),
     'pinball_by_level':{},'pinball_tail_only':{},'pinball_eleven_level':{},'fz0':{}}
for nm,_,_,_,_ in MODELS:
    OUT['pinball_by_level'][nm]={}
    for t in TAUS:
        OUT['pinball_by_level'][nm][str(t)]={'mean':round(float(np.nanmean(PLL[nm][t])),6)}
        if nm!=REF:
            OUT['pinball_by_level'][nm][str(t)]['vs_param_bteg']={r:dmv(PLL[nm][t],PLL[REF][t],msk) for r,msk in REG.items()}
    tail=np.mean([PLL[nm][t] for t in TAILT],axis=0); ele=np.mean([PLL[nm][t] for t in TAUS],axis=0)
    OUT['pinball_tail_only'][nm]={'mean':round(float(np.nanmean(tail)),6)}
    OUT['pinball_eleven_level'][nm]={'mean':round(float(np.nanmean(ele)),6)}
    if nm!=REF:
        tr=np.mean([PLL[REF][t] for t in TAILT],axis=0); er=np.mean([PLL[REF][t] for t in TAUS],axis=0)
        OUT['pinball_tail_only'][nm]['vs_param_bteg']={r:dmv(tail,tr,msk) for r,msk in REG.items()}
        OUT['pinball_eleven_level'][nm]['vs_param_bteg']={r:dmv(ele,er,msk) for r,msk in REG.items()}
rngb=np.random.default_rng(20260906); Dn=len(set(di)); BB=4000
IDX=np.empty((BB,Dn),dtype=np.int64); _s=rngb.integers(0,Dn,size=(BB,Dn)); _n=rngb.random((BB,Dn))<0.1
IDX[:,0]=_s[:,0]
for t_ in range(1,Dn): IDX[:,t_]=np.where(_n[:,t_],_s[:,t_],(IDX[:,t_-1]+1)%Dn)
def dagg(v): o=np.zeros(Dn); np.add.at(o,di,np.asarray(v,float)); return o
def br(S,C):
    cc=C[IDX].sum(axis=1); return np.where(cc>1e-12,S[IDX].sum(axis=1)/np.maximum(cc,1e-12),np.nan)
for a in ALPHAS:
    d={}; L={}
    for nm,_,vk,ek,sc in MODELS:
        v=ALL[vk%a].values; e=ALL[ek%a].values; L[nm]=fz0(Y,v,e,a)
        d[nm]={'meanFZ0':round(float(np.nanmean(L[nm])),5),'breach':round(float(np.mean(Y<=v)),4)}
        z=(Y-ALL['mu_'+sc].values)/ALL['sig_'+sc].values; zq=(v-ALL['mu_'+sc].values)/ALL['sig_'+sc].values
        zes=(e-ALL['mu_'+sc].values)/ALL['sig_'+sc].values
        tt=(z<=zq).astype(float)*z/(a*zes); obs=1.0-tt.sum()/len(tt)
        se=np.nanstd(1.0-br(dagg(tt),dagg(np.ones(len(tt)))))
        d[nm]['AS_Z2']=round(float(obs),4); d[nm]['AS_Z2_p']=round(float(2*(1-stats.norm.cdf(abs(obs)/se))),4) if se>0 else None
        mm=z<=zq; erm=np.where(mm,z-zes,0.0); nb=int(mm.sum())
        if nb>=10:
            o2=erm.sum()/nb; se2=np.nanstd(br(dagg(erm),dagg(mm.astype(float))))
            d[nm]['MF_exres']=round(float(o2),4); d[nm]['MF_p']=round(float(2*(1-stats.norm.cdf(abs(o2)/se2))),4) if se2>0 else None
    for nm in L:
        if nm!=REF: d[nm]['vs_param_bteg']={r:dmv(L[nm],L[REF],msk) for r,msk in REG.items()}
    OUT['fz0'][str(a)]=d
    lg("FZ0 a=%s: "%a+json.dumps({k:v['meanFZ0'] for k,v in d.items()}))
for nm,_,_,_,_ in MODELS:
    lg("  %-14s tail-only %.6f | eleven-level %.6f"%(nm,OUT['pinball_tail_only'][nm]['mean'],OUT['pinball_eleven_level'][nm]['mean']))
fn="tail_vs_bteg_results_%s.json"%PROT
json.dump(OUT,open(os.path.join(P,fn),"w"),indent=2); lg("TAILVSBTEGDONE[%s] %.0fs -> %s"%(PROT,time.time()-t0,fn))
