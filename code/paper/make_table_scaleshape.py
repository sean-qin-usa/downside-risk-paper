# make_table_scaleshape.py -- the Beta-t-EGARCH block for the scale-shape decomposition table (tab:rgarch),
# from robust_engine_results_taq30.json (job_robust_engine.py --panel taq30).
#
# The published table has two scales: the daily GARCH-t core and a proper HHS Realized GARCH, with DM taken
# against the realized scale carrying an unconditional residual quantile. This adds a THIRD scale, Beta-t-EGARCH,
# fitted per name by maximum likelihood on daily returns alone, on the same 30 large caps and the same test rows,
# with DM on the same footing. Two rows of context are printed first so the reproduction of the published levels
# can be checked before the new block is pasted in.
#
# One published row form is not computed for the new scale: the realized block's "+ EVT tail" row (a GPD tail with
# no learner). The engine rows here always combine body and EVT, as equation (5) does.
# Usage: python make_table_scaleshape.py [results_dir] [outdir]
import os, sys, json, math
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
RD=sys.argv[1] if len(sys.argv)>1 else os.path.join(P,"results","paper")
OUT=sys.argv[2] if len(sys.argv)>2 else os.path.join(P,"tables"); os.makedirs(OUT,exist_ok=True)
fn='robust_engine_results_taq30.json'
src=os.path.join(RD,fn) if os.path.exists(os.path.join(RD,fn)) else os.path.join(P,fn)
R=json.load(open(src)); F=R['fz0']; REF='rgarch_uncond'
CTX=[('engine','Daily core (GARCH-$t$ $+$ EVT), reproduction'),
     (REF,'Realized scale $+$ residual quantile (reference)'),
     ('body_rg','\\quad $+$ pooled shape learner'),
     ('engine_rg','\\quad $+$ shape $+$ EVT')]
NEW=[('bteg_uncond','Beta-$t$-EGARCH scale $+$ residual quantile'),
     ('bteg','\\quad $+$ parametric $t$ tail (the benchmark form)'),
     ('body_bteg','\\quad $+$ pooled shape learner'),
     ('engine_bteg','\\quad $+$ shape $+$ EVT')]
def row(k,lab):
    o1=F['0.025'].get(k); o2=F['0.01'].get(k)
    if not o1 or not o2: return '%s & --- & --- & --- & --- \\\\'%lab
    def dm(o):
        d=o.get('vs_'+REF)
        return '---' if k==REF or not d or d.get('DM_t') is None else '$%+.1f$'%d['DM_t']
    return '%s & $%.3f$ & %s & $%.3f$ & %s \\\\'%(lab,o1['meanFZ0'],dm(o1),o2['meanFZ0'],dm(o2))
L=['\\begin{tabular}{lcccc}','\\toprule','Model & FZ0 $2.5\\%$ & DM & FZ0 $1\\%$ & DM \\\\','\\midrule']
for k,lab in CTX: L.append(row(k,lab))
L+=['\\midrule']
for k,lab in NEW: L.append(row(k,lab))
L+=['\\bottomrule','\\end{tabular}']
bp=R.get('bteg_param_medians',{})
L+=['\\begin{tablenotes}\\footnotesize',
 ('\\item Added scale: Beta-$t$-EGARCH \\citep{harveychakravarty2008}, $\\lambda_{t+1} = \\omega + \\phi(\\lambda_t-\\omega) '
  '+ \\kappa u_t$ with $u_t = (\\nu+1)e_t^2/(\\nu+e_t^2) - 1$ the score of the Student-$t$ log-likelihood, a '
  'martingale difference bounded in $[-1,\\nu]$, fitted per name by maximum likelihood on the training window. '
  'Median fitted $\\phi = %.3f$, $\\kappa = %.3f$, $\\nu = %.2f$; the implied bound on one shock is '
  '$\\kappa\\nu = %.3f$ in log scale, a factor of $%.2f$. It uses daily returns only, so unlike the realized '
  'scale it is available on every name in the panel. %d names, %s test rows, same rows for every model; DM is '
  'date-clustered Newey--West(10) against the realized scale with an unconditional residual quantile, and a '
  'positive DM is worse than that benchmark. Result file \\texttt{%s}; predictions in the script header.')%(
   bp.get('phi',0),bp.get('kap',0),bp.get('nu',0),bp.get('score_bound',0),
   math.exp(bp.get('score_bound',0)),R['n_names'],'{:,}'.format(R['n_test']),fn.replace('_','\\_')),
 '\\end{tablenotes}']
open(os.path.join(OUT,'tab_scaleshape_bteg.tex'),'w').write('\n'.join(L)+'\n')
print("Scale-shape decomposition, 30 large caps, same test rows (FZ0, lower better; DM vs realized+uncond)")
print("  %-52s %9s %7s %9s %7s"%('model','FZ0 2.5%','DM','FZ0 1%','DM'))
for k,lab in CTX+[('--','')]+NEW:
    if k=='--': print('  '+'-'*86); continue
    o1=F['0.025'].get(k); o2=F['0.01'].get(k)
    if not o1: print("  %-52s %9s"%(lab,'---')); continue
    d1=o1.get('vs_'+REF); d2=o2.get('vs_'+REF)
    print("  %-52s %9.3f %7s %9.3f %7s"%(lab.replace('\\quad','  ').replace('$','').replace('\\%','%'),
      o1['meanFZ0'],('%+.1f'%d1['DM_t']) if d1 and d1.get('DM_t') is not None else '---',
      o2['meanFZ0'],('%+.1f'%d2['DM_t']) if d2 and d2.get('DM_t') is not None else '---'))
print("\nwrote %s"%os.path.join(OUT,'tab_scaleshape_bteg.tex'))
