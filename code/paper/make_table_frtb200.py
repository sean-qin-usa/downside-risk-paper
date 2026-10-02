# make_table_frtb200.py -- tab:frtb regenerated on the 200-name panel of Table 1, from
# frtb_table_200_results.json (job_frtb200.py). Column structure and row order are the committed table's;
# only the panel changes, so the FRTB battery and the frontier finally share one set of names and rows.
# Prints a side-by-side against the committed frtb_table_results.json so every moved figure is visible.
# Usage: python make_table_frtb200.py [results_dir] [outdir]
import os, sys, json
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
RD=sys.argv[1] if len(sys.argv)>1 else os.path.join(P,"results","paper")
OUT=sys.argv[2] if len(sys.argv)>2 else os.path.join(P,"tables"); os.makedirs(OUT,exist_ok=True)
NEW=json.load(open(os.path.join(RD,"frtb_table_200_results.json")))
try: OLD=json.load(open(os.path.join(RD,"frtb_table_results.json")))
except Exception: OLD=None
ORDER=[('resid_hybrid_ML','Residual-hybrid (GBM)'),('hybrid_EVT','\\quad $+$ EVT tail'),
       ('garch_t','GARCH(1,1)-$t$'),('gjr_skewt','GJR-GARCH-skew-$t$'),
       ('fhs_pername','GARCH-FHS (per-name)'),('fhs','GARCH-FHS (pooled)'),
       ('fhs_roll500','GARCH-FHS (rolling 500d)'),('ewma_rm','EWMA (RiskMetrics)'),
       ('hist_sim','Historical simulation')]
pm=NEW['per_model']; best=min(pm,key=lambda m:pm[m]['avg_pinball'])
def kp(v): return '0.00' if v is not None and v<0.005 else ('---' if v is None else '%.3g'%v)
L=['\\begin{tabular}{lcccc}','\\toprule',
   'Model & Avg.\\ pinball & ES$_{97.5}$ pred.\\ vs realized & Breach$_{99}$ & Kupiec$_{99}$ $p$ \\\\','\\midrule',
   '\\emph{Ideal value} & \\emph{(smaller better)} & \\emph{(pred.\\ $=$ realized)} & \\emph{1.00\\%} & \\emph{$>0.05$ (pass)} \\\\','\\midrule']
for k,lab in ORDER:
    o=pm.get(k)
    if not o: continue
    pin=('\\textbf{%.4f}'%o['avg_pinball']) if k==best else ('%.4f'%o['avg_pinball'])
    L.append('%s & %s & $%.2f$ / $%.2f$ & %.2f\\%% & %s \\\\'%(
        lab,pin,o['ES975_pred_true'],o['ES975_realized_ownVaR'],100*o['breach99'],kp(o['kupiec99_p'])))
L+=['\\bottomrule','\\end{tabular}',
 '\\begin{tablenotes}\\footnotesize',
 ('\\item %d names, %s test asset-days over %d dates -- the panel of Table~1, so the battery and the frontier '
  'share one set of rows. Twelve quantile levels; ES$_{97.5}$ is the exact tail integral of each model'
  "'"'s own quantile function, and the realized column is the mean return below that model'
  "'"'s own VaR, a '
  'model-dependent conditioning set kept as a diagnostic. Source: \\texttt{frtb\\_table\\_200\\_results.json}.')%(
   NEW['n_names'],'{:,}'.format(NEW['n_test_rows']),NEW['n_dates']),
 '\\end{tablenotes}']
open(os.path.join(OUT,'tab_frtb200.tex'),'w').write('\n'.join(L)+'\n')
print("tab:frtb on the %d-name panel (committed table is %s names)\n"%(
    NEW['n_names'], OLD['n_names'] if OLD and 'n_names' in OLD else '?'))
print("  %-26s %18s %18s %14s"%('model','avg pinball','breach99','Kupiec99 p'))
print("  %-26s %18s %18s %14s"%('','old -> new','old -> new','old -> new'))
for k,lab in ORDER:
    n=pm.get(k); o=(OLD or {}).get('per_model',{}).get(k)
    if not n: continue
    f=lambda a,b,fmt: ('%s -> %s'%((fmt%a) if a is not None else '--',(fmt%b)))
    print("  %-26s %18s %18s %14s"%(lab.replace('\\quad ','  ').replace('$','').replace('\\','')[:26],
        f(o['avg_pinball'] if o else None,n['avg_pinball'],'%.4f'),
        f(100*o['breach99'] if o else None,100*n['breach99'],'%.2f'),
        f(o['kupiec99_p'] if o else None,n['kupiec99_p'],'%.3g')))
print("\nwrote %s"%os.path.join(OUT,'tab_frtb200.tex'))
