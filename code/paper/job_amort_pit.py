# job_amort_pit.py -- THE AMORTIZATION NUMBERS THE PAPER CITES, RERUN WITH POINT-IN-TIME CHARACTERISTICS.
# crsp_panel_chars.csv is built with look-ahead: annvol is each name's FULL-SAMPLE realized volatility
# (correlation 1.0000 with it, exact to 2dp for 720 of 720 names), logmcap comes from a market-cap pull dated
# 2024-12-31, and beta is estimated on the full sample. amort_full.py reads all three, and its output backs
# the paper's +1.0%/+0.5% ablation and its 6-10% first-month edge. This job reruns those numbers with
# characteristics a desk could actually have, on the same split and seed, so old and new are like-for-like.
#
# POINT-IN-TIME REPLACEMENTS
#   annvol_pit  trailing 250-day realized volatility through t-1, annualised (min 60 observations)
#   beta_pit    trailing 250-day OLS beta on the equal-weighted panel return, through t-1 (min 60)
#   sector      static and genuinely known at listing: kept unchanged
#   logmcap     DROPPED. Market capitalisation needs prices and shares outstanding, which are not in the
#               returns panel, so it cannot be rebuilt point-in-time on this host. Rebuilding it needs a
#               WRDS pull of mcap as of each date (or as of listing for day-one rows). Its absence makes the
#               point-in-time arm slightly weaker than a fully rebuilt one would be, so the comparison below
#               is conservative against the point-in-time model, not in its favour.
#
# ALSO DEFINED HERE FOR THE FIRST TIME: the transfer win rate. The paper cites "59--79%" on held-out names,
# but no script in the repository computes a win rate and no result file contains one; the project's own
# HALLUCINATION_SWEEP_2026-09-03 flagged this figure as unverified pending packaged results. This job defines
# it explicitly as the fraction of held-out NAMES whose amortized mean pinball is below the own-history
# benchmark's, and reports it for both characteristic sets so the cited range can be checked or replaced.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (2026-10-02), decisions included:
#  P1. The transfer win rate on held-out names stays above 55% with point-in-time characteristics. If it
#      falls below 55%, transfer to unseen names does not survive the removal of look-ahead and the
#      abstract's "transfers to newly listed assets" has to go.
#  P2. The own-dynamics ablation still dominates: removing realized volatility costs more than removing all
#      characteristics, under both characteristic sets. This is the claim the paper actually leans on.
#  P3. The characteristics-only arm degrades materially when the look-ahead is removed -- its pinball rises
#      by more than 3% -- because annvol is close to the scale it is being asked to predict.
#  P4. The first-month (d15_30) edge over own history survives at more than half its published size, i.e.
#      above +5% against the published +9.7%, since the ablation attributes the transfer to own dynamics.
# Output: amort_pit_results.json
import os, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
t0=time.time(); lg=lambda s:print(s,flush=True)
TAUS=[0.05,0.10,0.25,0.50,0.75,0.90,0.95]; T3=[0.05,0.5,0.95]
HGB_A=dict(max_iter=250,max_depth=4,learning_rate=0.06,max_bins=128,random_state=0)
HGB_B=dict(max_iter=150,max_depth=4,learning_rate=0.07,max_bins=128,random_state=0)
def pin(y,q,t): e=y-q; return np.where(e>=0,t*e,(t-1)*e)
r=pd.read_csv(os.path.join(P,"crsp_panel_returns.csv"),dtype={'permno':'int32'})
r['date']=pd.to_datetime(r['date']); r['ret']=pd.to_numeric(r['ret'],errors='coerce').astype(float)*100.0
r=r.sort_values(['permno','date']).reset_index(drop=True)
ch=pd.read_csv(os.path.join(P,"crsp_panel_chars.csv"))
g=r.groupby('permno',sort=False); r['age']=g.cumcount()
r['lag1']=g['ret'].shift(1); r['abs1']=r['lag1'].abs()
r['rv5']=g['ret'].transform(lambda x:x.rolling(5,min_periods=3).std().shift(1))
r['rv21']=g['ret'].transform(lambda x:x.rolling(21,min_periods=8).std().shift(1))
r['mean21']=g['ret'].transform(lambda x:x.rolling(21,min_periods=8).mean().shift(1))
r['dn']=(r['lag1']<0).astype(float); r['ret2']=r['ret']**2
cs=g['ret'].cumsum()-r['ret']; cs2=g['ret2'].cumsum()-r['ret2']; cnt=r['age'].values.astype(float)
with np.errstate(invalid='ignore',divide='ignore'):
    om=np.where(cnt>0,cs/np.maximum(cnt,1),0.0); ov=np.where(cnt>1,cs2/np.maximum(cnt,1)-om**2,np.nan)
r['own_mean']=om; r['own_std']=np.sqrt(np.clip(ov,1e-6,None))
# ---- point-in-time characteristics
r['annvol_pit']=g['ret'].transform(lambda x:x.rolling(250,min_periods=60).std().shift(1))*math.sqrt(252)/100.0
mkt=r.groupby('date')['ret'].transform('mean')          # equal-weighted panel return, same date
r['_m']=mkt
cov=g.apply(lambda d: d['ret'].rolling(250,min_periods=60).cov(d['_m']).shift(1)).reset_index(level=0,drop=True)
var=g['_m'].transform(lambda x:x.rolling(250,min_periods=60).var().shift(1))
r['beta_pit']=(cov/var.replace(0,np.nan)).clip(-3,5)
lg("point-in-time characteristics built %.0fs"%(time.time()-t0))
ch['logmcap']=np.log(np.maximum(pd.to_numeric(ch['mcap_mm'],errors='coerce').fillna(300.0),1.0))
for c in ['sector','beta','annvol']: ch[c]=pd.to_numeric(ch[c],errors='coerce')
r=r.merge(ch[['permno','logmcap','sector','beta','annvol']],on='permno',how='left')
r['sector']=r['sector'].fillna(-1); r['beta']=r['beta'].fillna(1.0); r['annvol']=r['annvol'].fillna(0.3)
r['beta_pit']=r['beta_pit'].fillna(1.0); r['annvol_pit']=r['annvol_pit'].fillna(0.3)
r['rv5']=r['rv5'].fillna(r['rv21']).fillna(2.0); r['rv21']=r['rv21'].fillna(2.0); r['mean21']=r['mean21'].fillna(0.0)
r=r.dropna(subset=['lag1']); r['abs1']=r['abs1'].fillna(r['abs1'].median())
XC_OLD=['lag1','abs1','rv5','rv21','mean21','dn','age','logmcap','sector','beta','annvol']
XC_PIT=['lag1','abs1','rv5','rv21','mean21','dn','age','sector','beta_pit','annvol_pit']
SUB_OLD={'ALL':list(range(11)),'chars_only':[6,7,8,9,10],'own_recent_only':[0,1,2,3,4,5],
 'no_realized_vol':[0,1,4,5,6,7,8,9,10],'no_lags':[2,3,4,6,7,8,9,10],'no_chars':[0,1,2,3,4,5,6],'no_age':[0,1,2,3,4,5,7,8,9,10]}
SUB_PIT={'ALL':list(range(10)),'chars_only':[6,7,8,9],'own_recent_only':[0,1,2,3,4,5],
 'no_realized_vol':[0,1,4,5,6,7,8,9],'no_lags':[2,3,4,6,7,8,9],'no_chars':[0,1,2,3,4,5,6],'no_age':[0,1,2,3,4,5,7,8,9]}
rng=np.random.default_rng(11); names=r['permno'].unique(); rng.shuffle(names)   # same split and seed as amort_full.py
hold=set(names[:int(len(names)*0.4)]); tr=r[~r.permno.isin(hold)]; te=r[r.permno.isin(hold)].reset_index(drop=True)
lg("panel %s rows, %d names (%d held out); train %s test %s %.0fs"%(
   '{:,}'.format(len(r)),len(names),len(hold),'{:,}'.format(len(tr)),'{:,}'.format(len(te)),time.time()-t0))
y=te['ret'].values; age=te['age'].values.astype(float); ownm=te['own_mean'].values; owns=te['own_std'].values
perm=te['permno'].values
tn={0.05:-1.645,0.10:-1.282,0.25:-0.674,0.50:0.0,0.75:0.674,0.90:1.282,0.95:1.645}
BUCK=[('d15_30',15,30),('d30_60',30,60),('d60_120',60,120),('d120_250',120,250),('d250_500',250,500),('d500_1000',500,1000),('d1000_2700',1000,2700)]
OUT={'note':'The cited amortization numbers rerun with point-in-time characteristics beside the published '
     'look-ahead ones, same split and seed. logmcap is dropped from the point-in-time arm because market cap '
     'cannot be rebuilt from the returns panel; that makes the comparison conservative against the '
     'point-in-time model. The transfer win rate is DEFINED here (fraction of held-out names whose amortized '
     'mean pinball beats the own-history benchmark) because no script in the repository computes one.',
     'predictions_written_before_run':'P1 win rate stays above 55% point-in-time; P2 own-dynamics ablation '
     'still dominates under both sets; P3 chars_only degrades by more than 3%; P4 the d15_30 edge keeps more '
     'than half its published size (above +5% against +9.7%).',
     'n_rows':int(len(r)),'n_names':int(len(names)),'n_holdout':len(hold),'arms':{}}
for arm,XC,SUB in (('published_lookahead',XC_OLD,SUB_OLD),('point_in_time',XC_PIT,SUB_PIT)):
    Xtr=tr[XC].values.astype('float32'); ytr=tr['ret'].values; Xte=te[XC].values.astype('float32')
    AM={}
    for t in TAUS:
        AM[t]=HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB_A).fit(Xtr,ytr).predict(Xte)
    lg("  [%s] amortized trained %.0fs"%(arm,time.time()-t0))
    qA=lambda t:AM[t]; qP=lambda t:ownm+np.clip(owns,1e-3,None)*tn[t]
    def L(qf,mask): return float(np.mean([pin(y[mask],qf(t)[mask],t) for t in TAUS]))
    curve={}
    for bn,lo,hi in BUCK:
        m=(age>=lo)&(age<hi)&(age>=12)
        if m.sum()<50: continue
        a_,p_=L(qA,m),L(qP,m)
        curve[bn]=dict(n=int(m.sum()),amortized=round(a_,4),own_param=round(p_,4),amort_vs_own=round(a_/p_,4),
                       edge_pct=round(100*(1-a_/p_),2))
    # transfer win rate, defined per held-out name
    la=np.mean([pin(y,AM[t],t) for t in TAUS],axis=0)
    lp=np.mean([pin(y,ownm+np.clip(owns,1e-3,None)*tn[t],t) for t in TAUS],axis=0)
    df=pd.DataFrame({'pn':perm,'a':la,'p':lp,'age':age})
    wn=df.groupby('pn').apply(lambda d: d['a'].mean()<d['p'].mean())
    wr_all=float(wn.mean())
    yg=df[df['age']<250]; wn_y=yg.groupby('pn').apply(lambda d: d['a'].mean()<d['p'].mean())
    # DATE win rate: the share of test dates on which the amortized model's CROSS-NAME mean pinball beats
    # own history. This is the statistic the paper already reports elsewhere ("lower loss on 74% of dates"),
    # so it reads consistently, and unlike a per-name rate it needs no definition to be understood. The
    # per-name rate is kept beside it, with its definition attached.
    dd=pd.DataFrame({'dt':te['date'].values,'a':la,'p':lp,'age':age})
    gall=dd.groupby('dt')[['a','p']].mean()
    dwr_all=float((gall['a']<gall['p']).mean())
    dwr_buck={}
    for bn,lo,hi in BUCK:
        mb=(age>=lo)&(age<hi)&(age>=12)
        if mb.sum()<50: continue
        gb=dd[mb].groupby('dt')[['a','p']].mean()
        nper=dd[mb].groupby('dt').size()
        dwr_buck[bn]={'n_dates':int(len(gb)),'median_names_per_date':int(nper.median()),
                      'date_win_rate':round(float((gb['a']<gb['p']).mean()),4)}
    gy=dd[dd['age']<250].groupby('dt')[['a','p']].mean()
    dwr_young=float((gy['a']<gy['p']).mean())
    abl={}
    for nm,cols in SUB.items():
        tot=0.0
        for t in T3:
            tot+=np.mean(pin(y,HistGradientBoostingRegressor(loss='quantile',quantile=t,**HGB_B).fit(Xtr[:,cols],ytr).predict(Xte[:,cols]),t))
        abl[nm]=round(float(tot/len(T3)),4)
    base=abl['ALL']
    OUT['arms'][arm]={'features':XC,'age_curve':curve,
        'transfer_win_rate_all_names':round(wr_all,4),'transfer_win_rate_young_names':round(float(wn_y.mean()),4),
        'n_heldout_names_scored':int(len(wn)),
        'win_rate_definitions':{
            'per_name':'fraction of held-out names whose amortized mean pinball over all test rows of that name is below the own-history benchmark',
            'per_date':'fraction of test dates on which the cross-name mean pinball of the amortized model is below that of own history, held-out names only'},
        'date_win_rate_overall':round(dwr_all,4),'date_win_rate_young_names':round(dwr_young,4),
        'date_win_rate_by_age':dwr_buck,'n_test_dates':int(len(gall)),
        'ablation':{k:{'pinball':v,'pct_vs_ALL':round((v/base-1)*100,2)} for k,v in sorted(abl.items(),key=lambda x:x[1])}}
    lg("  [%s] win rate per-name %.3f (young %.3f) per-date %.3f (young %.3f) | by age %s"%(arm,wr_all,float(wn_y.mean()),dwr_all,dwr_young,json.dumps({k:v['date_win_rate'] for k,v in dwr_buck.items()})))
    lg("  [%s] ablation %s %.0fs"%(arm,
        json.dumps({k:v['pct_vs_ALL'] for k,v in OUT['arms'][arm]['ablation'].items()}),time.time()-t0))
json.dump(OUT,open(os.path.join(P,"amort_pit_results.json"),"w"),indent=2)
lg("AMORTPITDONE %.0fs"%(time.time()-t0))
