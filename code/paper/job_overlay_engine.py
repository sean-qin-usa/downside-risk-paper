# job_overlay_engine.py -- STAGE 4 DONE PROPERLY: THE CONFORMAL SHIFT ESTIMATED ON THE ENGINE ITSELF,
# STATIC AND ADAPTIVE. The committed overlay estimates its shift from the BODY's calibration errors and then
# applies it to the ENGINE, which already carries the EVT branch, so the tail correction enters twice and the
# row over-covers badly (date-clustered t of -8.9 to -10.8 at 2.5%). Re-estimating the shift against the
# engine's own min-envelope-after-rearrangement calibration errors removes the double count, but a single
# additive shift fitted on one window still transfers badly: the calibration block and the test era disagree
# on the sign of their miscoverage. An ADAPTIVE shift (Gibbs-Candes) is the construction that does not have
# to transfer, and job_fz_aci.py already fixes the update and the two rates used here:
#     c_{t+1} = c_t - gamma * (b_t - alpha),   b_t = the panel breach frequency on date t,  gamma in {0.02,0.05}
# The update is causal: the shift applied on date t+1 uses only breaches observed through date t.
#
# Variants, both regulatory levels, 200-name canonical rows and the frozen 2000-2013 holdout (--panel holdout):
#   engine                 the accuracy layer, no shift (reference)
#   overlay_static_body    the committed construction, kept so the double count is visible
#   overlay_static_engine  the same estimator re-targeted at the engine
#   overlay_aci_0.02 / _0.05   adaptive, the two existing rates
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-01), decisions included:
#  P1. The adaptive overlay lands within 0.3 points of nominal at 2.5% on BOTH panels, where both static
#      forms miss by more than a point. An adaptive shift does not have to transfer across eras, which is the
#      defect that sinks the static ones.
#  P2. The adaptive overlay's FZ0 is within noise of the accuracy layer overall (|DM| < 2) at both levels.
#      If instead it costs significantly, coverage is being bought with accuracy and Stage 4 should stay off.
#  P3. overlay_static_engine stays MIS-SIGNED out of era: on the 2000-2013 holdout its shift still pushes the
#      quantile the wrong way, so its breach rate sits on the opposite side of nominal from the design era.
#  P4. overlay_static_body remains the worst row on the date-clustered test at 2.5% on both panels, because
#      the double count is a property of the construction and not of the era.
#  DECISION: Stage 4 is switched on only if the adaptive overlay satisfies P1 AND P2 on BOTH panels.
#      Otherwise it stays off and the paper reports the diagnosis rather than a repair.
# Output: overlay_engine_results{,_holdout}.json
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from es_integral import converged_es, levels_and_body
from scipy import stats
from arch import arch_model
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True); rng=np.random.default_rng(20260906)
PANEL=sys.argv[sys.argv.index("--panel")+1] if "--panel" in sys.argv else "canon200"
assert PANEL in ("canon200","holdout")
TAUS=[0.01,0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975,0.99]; ALPHAS=[0.01,0.025]
ZX=['logsig','zl1','absz5','zstd21','fracdn5']; SUBN=20; P0=0.025; GAMMAS=[0.02,0.05]
ES_LEGACY={}   # superseded 20-node ES, for the old-vs-new record
HGB=dict(max_iter=250,max_depth=3,learning_rate=0.06,random_state=0); BB=4000; MBLK=10.0
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
def conf_ostat(sc,tau):
    n=len(sc); k=min(max(int(math.ceil((n+1)*tau)),1),n); return float(np.sort(np.asarray(sc,float))[k-1])
def _llb(pp,k0,k1):
    if k0+k1==0: return 0.0
    if pp<=0: return 0.0 if k1==0 else -1e300
    if pp>=1: return 0.0 if k0==0 else -1e300
    return k0*math.log(1-pp)+k1*math.log(pp)
def kupiec_pn(x,T,p):
    if T==0 or x==T: return None
    if x==0: return round(float(1-stats.chi2.cdf(-2*_llb(p,T,0),1)),4)
    pi=x/T; lr=-2*(_llb(p,T-x,x)-_llb(pi,T-x,x)); return round(float(1-stats.chi2.cdf(max(lr,0),1)),4)
def christ(b,p):
    b=b.astype(int); T=len(b); x=int(b.sum())
    if x==0: return None
    n00=n01=n10=n11=0
    for i in range(1,T):
        a_,c_=b[i-1],b[i]
        if a_==0 and c_==0:n00+=1
        elif a_==0 and c_==1:n01+=1
        elif a_==1 and c_==0:n10+=1
        else:n11+=1
    pi=x/T; pi0=n01/max(n00+n01,1); pi1=n11/max(n10+n11,1)
    lu=-2*(_llb(p,T-x,x)-_llb(pi,T-x,x))
    li=-2*(_llb(pi,n00+n10,n01+n11)-(_llb(pi0,n00,n01)+_llb(pi1,n10,n11)))
    return round(float(1-stats.chi2.cdf(max(lu+li,0),2)),4)
def kupiec_pool(br,a):
    N=len(br); nb=int(br.sum()); ph=nb/N if N else 0.0
    if nb==0 or nb==N: return ph,float('nan')
    LR=-2*((nb*math.log(a)+(N-nb)*math.log(1-a))-(nb*math.log(ph)+(N-nb)*math.log(1-ph)))
    return ph,float(1-stats.chi2.cdf(LR,1))
src="holdout_panel_2000_2013.csv" if PANEL=="holdout" else "crsp_panel_returns.csv"
rr=pd.read_csv(os.path.join(P,src),dtype={'permno':'int32'})
rr['date']=pd.to_datetime(rr['date']); rr['ret']=pd.to_numeric(rr['ret'],errors='coerce')*100.0
cnt=rr.groupby('permno')['ret'].count().sort_values(ascending=False); names=cnt[cnt>=1500].index.tolist()[:200]
def feats(y,sig,mu,dts):
    z=(y-mu)/np.maximum(sig,1e-6); df=pd.DataFrame({'y':y,'sig':sig,'z':z,'date':dts})
    df['logsig']=np.log(np.maximum(df['sig'],1e-6)); df['zl1']=df['z'].shift(1)
    df['absz5']=df['z'].abs().rolling(5,min_periods=3).mean().shift(1)
    df['zstd21']=df['z'].rolling(21,min_periods=8).std().shift(1)
    df['fracdn5']=(df['y']<0).rolling(5,min_periods=3).mean().shift(1)
    df['mk63']=df['z'].rolling(63,min_periods=30).kurt().shift(1)
    return df
TR=[]; CAL=[]; rows=[]
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
    df=feats(y,np.sqrt(s2),mu,dts); df['mu']=mu; df['nu']=nu; df['tsc']=math.sqrt(nu/(nu-2)) if nu>2 else 1.0
    df['idx']=np.arange(n); ok=df.dropna(subset=ZX+['mk63'])
    trn=ok[ok['idx']<cp]; cal=ok[(ok['idx']>=cp)&(ok['idx']<sp)]; tst=ok[ok['idx']>=sp]
    if len(tst)<60 or len(cal)<60: continue
    TR.append(trn[ZX+['z']]); CAL.append(cal[ZX+['z']]); t2=tst.copy(); t2['permno']=pn; rows.append(t2)
TE=pd.concat(rows).reset_index(drop=True); TRc=pd.concat(TR); CALc=pd.concat(CAL)
lg("panel[%s] %d names %d rows %.0fs"%(PANEL,TE.permno.nunique(),len(TE),time.time()-t0))
ztr=TRc['z'].values; uu=float(np.quantile(ztr,P0)); exc=uu-ztr[ztr<uu]
xi,_,bt=stats.genpareto.fit(exc,floc=0.0)
evt=lambda t:(uu-(bt/xi)*((t/P0)**(-xi)-1.0) if abs(xi)>1e-6 else uu-bt*math.log(P0/t))
X=TE[ZX].values; Xc=CALc[ZX].values
ZQ={}; ZQc={}
for t in TAUS:
    m=HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB).fit(TRc[ZX].values,ztr)
    ZQ[t]=m.predict(X); ZQc[t]=m.predict(Xc)
SUB={}; SUBc={}
for a in ALPHAS:
    SUB[a]=[]; SUBc[a]=[]
    for j in range(SUBN):
        mz=HistGradientBoostingRegressor(loss='quantile',quantile=a*(j+0.5)/SUBN,**HGB).fit(TRc[ZX].values,ztr)
        SUB[a].append(mz.predict(X)); SUBc[a].append(mz.predict(Xc))
zcal=CALc['z'].values
MU=TE['mu'].values; SIG=TE['sig'].values; NU=TE['nu'].values; TSC=TE['tsc'].values
Y=TE['y'].values; DTS=TE['date'].values; PERM=TE['permno'].values
di,ud=pd.factorize(pd.to_datetime(DTS),sort=True); di=np.asarray(di); Dn=len(ud)
rk=np.full(len(Y),-1); fm=np.isfinite(TE['mk63'].values)
rk[fm]=pd.qcut(pd.Series(TE['mk63'].values[fm]),10,labels=False,duplicates='drop').values+1
REG={'overall':np.ones(len(Y),bool),'top_mk63_decile':rk==10}
ENG={}; SH={}
for a in ALPHAS:
    st=np.sort(np.stack([np.minimum(SUB[a][j],evt(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
    zq=np.maximum(np.minimum(ZQ[a],evt(a)),st[:,-1])
    # CONVERGED ES. The overlay shifts both VaR and ES by a constant, so the shift itself is unchanged and
    # only the ES level moves; the pre-committed 'within noise of the accuracy layer' rule is re-graded on it.
    _lev,_QB=levels_and_body(a,SUB[a],lambda u:HistGradientBoostingRegressor(loss='quantile',quantile=u,**HGB).fit(TRc[ZX].values,ztr).predict(X),subn=SUBN)
    _es20=np.minimum(st.mean(axis=1),zq-1e-6)
    es=converged_es(a,_lev,_QB,evt,(uu,float(bt),float(xi),P0),M=2000,var_z=zq)
    ES_LEGACY[str(a)]={'es_20node':round(float(np.nanmean(_es20)),5),'es_converged':round(float(np.nanmean(es)),5),'ratio':round(float(np.nanmean(es/_es20)),5)}
    stc=np.sort(np.stack([np.minimum(SUBc[a][j],evt(a*(j+0.5)/SUBN)) for j in range(SUBN)],axis=1),axis=1)
    zqc=np.maximum(np.minimum(ZQc[a],evt(a)),stc[:,-1])
    ENG[a]=(zq,es)
    SH[a]={'body':conf_ostat(zcal-ZQc[a],a),'engine':conf_ostat(zcal-zqc,a)}
lg("GPD u=%.3f xi=%.3f beta=%.3f | shifts %s %.0fs"%(uu,xi,bt,{str(a):{k:round(v,4) for k,v in SH[a].items()} for a in ALPHAS},time.time()-t0))
order=np.argsort(di,kind='mergesort')
def aci_shift(a,gam):
    """c_{t+1} = c_t - gam (b_t - a); causal, b_t from date t breaches under the shift in force on date t."""
    zq=ENG[a][0]; c=0.0; out=np.zeros(len(Y)); path=[]
    for d in range(Dn):
        m=order[di[order]==d]
        out[m]=c
        b=float(np.mean(Y[m]<=MU[m]+SIG[m]*(zq[m]+c)))
        path.append(c); c=c-gam*(b-a)
    return out,np.array(path)
VAR={}
for a in ALPHAS:
    zq,es=ENG[a]
    VAR[(a,'engine')]=(MU+SIG*zq,MU+SIG*es)
    for k in ('body','engine'):
        s_=SH[a][k]; VAR[(a,'overlay_static_'+k)]=(MU+SIG*(zq+s_),MU+SIG*(es+s_))
    for gam in GAMMAS:
        cs,path=aci_shift(a,gam)
        VAR[(a,'overlay_aci_%g'%gam)]=(MU+SIG*(zq+cs),MU+SIG*(es+cs))
        lg("  aci a=%g gamma=%g: c range [%.4f, %.4f] final %.4f %.0fs"%(a,gam,path.min(),path.max(),path[-1],time.time()-t0))
    VAR[(a,'garch_t')]=(MU+SIG*stats.t.ppf(a,NU)/TSC,MU+SIG*t_es(a,NU)/TSC)
IDX=np.empty((BB,Dn),dtype=np.int64); _s=rng.integers(0,Dn,size=(BB,Dn)); _n=rng.random((BB,Dn))<(1.0/MBLK)
IDX[:,0]=_s[:,0]
for t_ in range(1,Dn): IDX[:,t_]=np.where(_n[:,t_],_s[:,t_],(IDX[:,t_-1]+1)%Dn)
def dagg(v): o=np.zeros(Dn); np.add.at(o,di,np.asarray(v,float)); return o
def bratio(S,C):
    cc=C[IDX].sum(axis=1); return np.where(cc>1e-12,S[IDX].sum(axis=1)/np.maximum(cc,1e-12),np.nan)
def dmv(a1,a2,msk):
    msk=msk&np.isfinite(a1)&np.isfinite(a2)
    if msk.sum()<100: return None
    g=pd.DataFrame({'d':(a1-a2)[msk],'dt':di[msk]}).groupby('dt')['d'].mean()
    return {'mean_diff':round(float(np.mean((a1-a2)[msk])),5),'DM_t':nw_t(g.values),'n':int(msk.sum())}
MODELS=['engine','overlay_static_body','overlay_static_engine']+['overlay_aci_%g'%g for g in GAMMAS]+['garch_t']
OUT={'es_convention':{'engine_rows':'converged integral (es_integral.converged_es, M=2000, 60 fitted body levels): body interpolated on [a/40,a], pooled GPD exact at every node, sub-floor region in closed form. VaR is the committed construction and is unchanged, so no VaR-only statistic and no pinball number can move.','closed_form_and_empirical_rows':'unaffected','superseded_20node_values':ES_LEGACY},'note':'Stage 4 with the shift estimated on the engine itself, static and adaptive (Gibbs-Candes, the '
     'rates of job_fz_aci.py). overlay_static_body is the committed construction, kept so the double count is '
     'visible. DM_t>0 means the row is WORSE than the named reference. Panel: '+PANEL+'.',
     'predictions_written_before_run':'P1 adaptive within 0.3pt of nominal at 2.5% on both panels; P2 adaptive '
     'FZ0 within noise of the accuracy layer (|DM|<2); P3 overlay_static_engine stays mis-signed out of era; '
     'P4 overlay_static_body worst on the date-clustered test at 2.5%. DECISION: Stage 4 on only if P1 and P2 '
     'both hold on both panels.',
     'panel':PANEL,'n_names':int(TE.permno.nunique()),'n_test':int(len(Y)),'n_dates':int(Dn),
     'gpd':{'u':round(uu,4),'xi':round(float(xi),4),'beta':round(float(bt),4)},
     'shifts':{str(a):{k:round(v,4) for k,v in SH[a].items()} for a in ALPHAS},'per_alpha':{}}
for a in ALPHAS:
    A={}; L={m:fz0(Y,*VAR[(a,m)],a) for m in MODELS}
    for m in MODELS:
        v,e=VAR[(a,m)]; b=(Y<=v).astype(int)
        brate,kp=kupiec_pool(b.astype(float),a)
        kl=[];cl=[]
        for pn_ in np.unique(PERM):
            msk=PERM==pn_
            kv=kupiec_pn(int(b[msk].sum()),int(msk.sum()),a)
            if kv is not None: kl.append(kv)
            cv=christ(b[msk],a)
            if cv is not None: cl.append(cv)
        fdf=pd.DataFrame({'b':b,'dt':di}).groupby('dt')['b'].mean(); dct=nw_t(fdf.values-a)
        z=(Y-MU)/SIG; zq=(v-MU)/SIG; zes=(e-MU)/SIG
        tt=(z<=zq).astype(float)*z/(a*zes); obs=1.0-tt.sum()/len(tt)
        se=np.nanstd(1.0-bratio(dagg(tt),dagg(np.ones(len(tt)))))
        mm=z<=zq; erm=np.where(mm,z-zes,0.0); nb=int(mm.sum())
        o2=erm.sum()/nb if nb>=10 else float('nan')
        se2=np.nanstd(bratio(dagg(erm),dagg(mm.astype(float)))) if nb>=10 else 0.0
        A[m]={'breach':round(brate,4),'dev_from_nominal_pt':round(100*(brate-a),3),
              'kupiec_pooled_p':round(kp,4),'kupiec_pername_passrate':round(float(np.mean([x>0.05 for x in kl])),3) if kl else None,
              'christoffersen_passrate':round(float(np.mean([x>0.05 for x in cl])),3) if cl else None,
              'dateclustered_NW_t':dct,'dateclustered_PASS':(abs(dct)<1.96) if dct is not None else None,
              'AS_Z2':round(float(obs),4),'AS_Z2_p':round(float(2*(1-stats.norm.cdf(abs(obs)/se))),4) if se>0 else None,
              'MF_exres':round(float(o2),4) if nb>=10 else None,
              'MF_p':round(float(2*(1-stats.norm.cdf(abs(o2)/se2))),4) if se2>0 else None,
              'meanFZ0':round(float(np.nanmean(L[m])),5)}
        for ref in ('engine','garch_t'):
            if m!=ref: A[m]['vs_'+ref]={r:dmv(L[m],L[ref],msk2) for r,msk2 in REG.items()}
        lg("  a=%g %-22s breach %.4f (%+.2fpt) dclust %6s PASS %-5s FZ0 %.5f"%(a,m,brate,100*(brate-a),dct,str(A[m]['dateclustered_PASS']),A[m]['meanFZ0']))
    OUT['per_alpha'][str(a)]=A
fn="overlay_engine_results%s.json"%("" if PANEL=="canon200" else "_"+PANEL)
json.dump(OUT,open(os.path.join(P,fn),"w"),indent=2); lg("OVERLAYDONE[%s] %.0fs -> %s"%(PANEL,time.time()-t0,fn))
