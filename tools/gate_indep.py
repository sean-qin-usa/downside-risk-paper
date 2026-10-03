# gate_indep.py -- INDEPENDENT recomputation of the score-gated-scale comparison from the per-row forecasts
# that job_scale_gate.py saves. Nothing here is imported from that job: FZ0 and the Newey-West t are written
# from scratch, and the only inputs are the saved panel and the job's own result file for comparison. It exists
# because the gate's margin over always-running-the-robust-scale is small enough that the sign needed a check
# by code that could not inherit the same mistake.
# Reads scale_gate_panel.csv.gz from GBC_PROJ (not committed: it is a per-row panel) and
# results/paper/scale_gate_results.json from the repo. Set GBC_PROJ to override the project path.
import os, numpy as np, pandas as pd, math, json, warnings; warnings.filterwarnings('ignore')
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
REPO=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
THR=13.0953
A=pd.read_csv(os.path.join(P,"scale_gate_panel.csv.gz"))
A['date']=pd.to_datetime(A['date']); A['year']=A['date'].dt.year
gate=(A['mk'].values>=THR); Y=A['y'].values
print("rows %s | dates %d"%('{:,}'.format(len(A)),A['date'].nunique()))
print("test-era fire rate by year:")
for yr,g in A.assign(g=gate).groupby('year'):
    print("   %d: %5.2f%% of %s rows"%(yr,100*g['g'].mean(),'{:,}'.format(len(g))))
print("   overall %.2f%%"%(100*gate.mean()))
def FZ(y,v,e,a):                       # independent FZ0
    v=np.where(v>-1e-8,-1e-8,v); e=np.where(e>v,v,e)
    ind=(y<=v).astype(float)
    return -(ind*(v-y))/(a*e)+v/e+np.log(-e)-1.0
def NW(x,L=10):                        # independent Newey-West t
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x); m=x.mean(); d=x-m
    s=np.dot(d,d)/n
    for k in range(1,L+1): s+=2.0*(1.0-k/(L+1.0))*np.dot(d[k:],d[:-k])/n
    return m/math.sqrt(s/n)
dk=A['date'].values
r=pd.Series(A['mk'].values).rank(method='first')
rk=(pd.qcut(r,10,labels=False)+1).values
REG={'overall':np.ones(len(Y),bool),'top_decile':rk==10,'deciles_1to9':(rk>=1)&(rk<=9)}
ref=json.load(open(os.path.join(REPO,"results","paper","scale_gate_results.json")))
for a in (0.01,0.025):
    V={}
    for f in ('garch_t','bteg'):
        V['engine_'+f]=(A['v_engine_%s_%g'%(f,a)].values,A['e_engine_%s_%g'%(f,a)].values)
        V['param_'+f]=(A['v_param_%s_%g'%(f,a)].values,A['e_param_%s_%g'%(f,a)].values)
    V['gate']=(np.where(gate,V['engine_bteg'][0],V['engine_garch_t'][0]),
               np.where(gate,V['engine_bteg'][1],V['engine_garch_t'][1]))
    L={m:FZ(Y,*V[m],a) for m in V}
    print()
    print("alpha %.3f   %-16s %12s %12s %10s"%(a,"model","independent","job-reported","abs diff"))
    for m in ['gate','engine_garch_t','engine_bteg','param_garch_t','param_bteg']:
        mine=float(np.nanmean(L[m])); theirs=ref['per_alpha'][str(a)][m]['meanFZ0']
        print("            %-16s %12.5f %12.5f %10.2e"%(m,mine,theirs,abs(mine-theirs)))
    print("   gate vs always-engine_bteg by region (DM>0 => engine_bteg worse than gate):")
    for rg,msk in REG.items():
        d=(L['engine_bteg']-L['gate'])[msk]
        s=pd.DataFrame({'d':d,'dt':dk[msk]}).groupby('dt')['d'].mean().values
        print("      %-14s DM %+6.2f   n=%s"%(rg,NW(s),'{:,}'.format(int(msk.sum()))))
    print("   every fixed rule vs gate, overall (DM>0 => that rule worse):")
    for m in ['engine_garch_t','engine_bteg','param_garch_t','param_bteg']:
        d=(L[m]-L['gate'])
        s=pd.DataFrame({'d':d,'dt':dk}).groupby('dt')['d'].mean().values
        print("      %-16s DM %+6.2f"%(m,NW(s)))
