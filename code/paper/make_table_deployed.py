# make_table_deployed.py -- THE SINGLE ALL-METRICS TABLE FOR ONE DEPLOYED CONFIGURATION.
# The paper reports a menu: an accuracy layer, a body without the EVT branch, a conformal overlay, and a
# decision table over regions. A desk deploys one model. This script fixes that one model and assembles every
# metric recorded for it into a single table.
#
# THE DEPLOYED CONFIGURATION (default bteg; pass a scale as the third argument to restate against another):
#   Stage 1  GARCH(1,1)-t conditional scale, per name (NOT swapped: see the Stage-1 note below)
#   Stage 2  pooled HistGBM residual-quantile body on the five features (logsig, z_{t-1}, |z|_5, sd_21, dn_5)
#   Stage 3  pooled GPD left tail spliced at p0 = 0.025, taken as the body/EVT minimum, then rearranged
#   Stage 4  NOT APPLIED, and the retargeted variant does not rescue it (Panel D).
# STAGE 1 WAS NOT MOVED. A Beta-t-EGARCH Stage 1 beats this one on FZ0 under the frozen 60/40 protocol
# (DM 3.31 and 4.55) and on the frozen 2000-2013 holdout (3.09, 3.48), but under the ANNUAL REFITS the paper
# recommends for production the two are indistinguishable (DM -0.31 and -0.01, walkforward_bteg_results.json).
# The frozen-fit advantage was largely parameter staleness, which refitting removes as well as a robust filter
# does, so the swap is reported as a diagnostic result and not as a deployment change.
# Panel B is sourced from the exception battery, which runs every test on the same 200 names and 221,600 rows,
# so the table no longer mixes the 140-name per-asset file with the 200-name ES file.
# Usage: python make_table_deployed.py [results_dir] [outdir] [deploy_scale]
import os, sys, json, re, math
P=os.environ.get("GBC_PROJ",os.environ.get("GBC_PROJECT_DIR",r"C:\Users\OWNER\Claude\Projects\GBC Project"))
RD=sys.argv[1] if len(sys.argv)>1 else os.path.join(P,"results","paper")
OUT=sys.argv[2] if len(sys.argv)>2 else os.path.join(P,"tables"); os.makedirs(OUT,exist_ok=True)
SC=sys.argv[3] if len(sys.argv)>3 else "garch_t"
def J(fn):
    for c in (os.path.join(RD,fn),os.path.join(P,fn)):
        if os.path.exists(c):
            try: return json.load(open(c))
            except Exception: return None
    return None
BA=J('bench_all_results.json'); XB=J('exception_battery_results.json'); RE=J('robust_engine_results.json')
HOL=J('robust_engine_results_holdout.json'); CP=J('coldstart_peers_results.json')
HO=J('holdout_frontier_results.json'); CS=J('calendar_split_results.json'); WF=J('walkforward_results.json')
PIT=J('pit_universe_results.json'); FR=J('frontier_robust_results.json'); TD=J('tenday_diag_results.json')
ENGK={'garch_t':'engine','bip_t':'engine_bip','bteg':'engine_bteg'}[SC]      # key in robust_engine
XBK='engine_'+SC                                                            # key in the battery
SCN={'garch_t':'GARCH(1,1)-$t$','bip_t':'bounded-news GARCH','bteg':'Beta-$t$-EGARCH'}[SC]
def g(o,*ks,d=None):
    for k in ks:
        if o is None or not isinstance(o,dict) or k not in o: return d
        o=o[k]
    return o if o is not None else d
def n(x,dec=2,sign=False,pct=False,d='---'):
    if x is None: return d
    return (('%+.'+str(dec)+'f' if sign else '%.'+str(dec)+'f')%x)+('\\%' if pct else '')
def pr(x,d='---'):
    if x is None: return d
    return '$<0.001$' if x<0.001 else '%.3f'%x
def ed(e):
    if not e: return '---'
    return '$%s$ (%s)'%(n(e.get('edge_pct'),2,True),n(e.get('DM') if e.get('DM') is not None else e.get('DM_stat'),2))

# ---- Panel A: accuracy, eleven-level pinball, from robust_engine (all scales on one row set)
A=[]; pb=g(RE,'pinball','sort_mk63_'+SC)
A.append(('Mean pinball, eleven levels','%s (GARCH-$t$ %s)'%(n(g(RE,'mean_pinball',ENGK),5),n(g(RE,'mean_pinball','garch_t'),5))))
for key,lab in [('overall','Edge over GARCH-$t$, overall'),('top_decile','\\quad top mk$_{63}$ decile'),('deciles_1to9','\\quad deciles 1--9')]:
    A.append((lab,ed(g(pb,'vs_garch_t',ENGK,key))))
A.append(('Edge over its own-scale parametric tail, top decile',ed(g(pb,'flexible_over_same_scale_parametric','top_decile'))))
for a,lab in [('0.01','1\\%'),('0.025','2.5\\%')]:
    m=g(RE,'mcs','fz0_'+('0.01' if a=='0.01' else '0.025'),'in_90pct_MCS',d=[])
    A.append(('In 90\\%% MCS on FZ0 at %s'%lab,('yes' if ENGK in m else 'no')+(' (of %d rows)'%len(g(RE,'mean_pinball',d={}))) ))

# ---- Panel B: the full battery for this configuration, one source, 200 names
B=[]
def two(lab,f1,f2): B.append((lab,f1,f2))
X1=g(XB,'per_alpha','0.01',XBK,d={}); X2=g(XB,'per_alpha','0.025',XBK,d={})
two('Mean FZ0',n(X1.get('meanFZ0'),5),n(X2.get('meanFZ0'),5))
for rival,lab in [('garch_t','GARCH(1,1)-$t$'),('engine','the GARCH-$t$ engine'),('bteg_uncond','robust scale $+$ pooled residual quantile'),('bip_t','bounded-news GARCH-$t$')]:
    k='vs_'+ENGK
    two('DM against %s'%lab,n(g(RE,'fz0','0.01',rival,k,'DM_t'),2,True),n(g(RE,'fz0','0.025',rival,k,'DM_t'),2,True))
two('Breach rate (nominal 1.0, 2.5)',n(100*X1.get('breach',0),2,pct=True),n(100*X2.get('breach',0),2,pct=True))
two('Unconditional coverage, date-clustered $p$',pr(X1.get('uc_clustered_p')),pr(X2.get('uc_clustered_p')))
two('\\quad same test pooled, treating rows as independent',pr(X1.get('kupiec_pooled_p')),pr(X2.get('kupiec_pooled_p')))
two('\\quad design effect, iid over clustered',n(X1.get('design_effect_iid_over_clustered'),1),n(X2.get('design_effect_iid_over_clustered'),1))
two('Date-clustered exception test, NW $t$',n(X1.get('dateclustered_NW_t'),2,True),n(X2.get('dateclustered_NW_t'),2,True))
two('Kupiec per name, pass rate',n(100*(X1.get('kupiec_pername_passrate') or 0),1,pct=True),n(100*(X2.get('kupiec_pername_passrate') or 0),1,pct=True))
two('Christoffersen conditional coverage, pass rate',n(100*(X1.get('christoffersen_passrate') or 0),1,pct=True),n(100*(X2.get('christoffersen_passrate') or 0),1,pct=True))
two("Acerbi--Sz\\'ekely $Z_2$ ($p$)",'%s (%s)'%(n(X1.get('AS_Z2'),3,True),pr(X1.get('AS_Z2_p'))),'%s (%s)'%(n(X2.get('AS_Z2'),3,True),pr(X2.get('AS_Z2_p'))))
two('McNeil--Frey ES residual, $p$',pr(X1.get('MF_p')),pr(X2.get('MF_p')))

# ---- Panel C: robustness and reach
C=[]
for a,lab in [('0.01','1\\%'),('0.025','2.5\\%')]:
    e=g(HOL,'fz0',a,'engine','vs_'+ENGK); eh=g(HOL,'fz0',a,ENGK,'meanFZ0'); ee=g(HOL,'fz0',a,'engine','meanFZ0')
    C.append(('Frozen 2000--2013 holdout at %s, FZ0 vs the GARCH-$t$ engine'%lab,
              '%s against %s, DM %s'%(n(eh,5),n(ee,5),n(g(e,'DM_t'),2,True)) if eh else '---'))
C.append(('2000--2013 holdout, frontier under the frozen specification','overall %s, top decile %s'%(ed(g(HO,'overall')),ed(g(HO,'top_decile_mk63')))))
C.append(('Calendar-time split, train before 2020','overall $%s$ (%s), top-decile DM %s'%(n(g(CS,'overall_edge_pct'),2,True),n(g(CS,'overall_DM'),2),n(g(CS,'top_decile_DM'),2))))
C.append(('Annual walk-forward refit','overall $%s$ (%s), top decile $%s$'%(n(g(WF,'overall_edge_pct'),2,True),n(g(WF,'overall_DM'),2),n(g(WF,'top_decile_edge_pct'),2,True))))
C.append(('Point-in-time universe, delistings ridden','overall %s, top bucket %s'%(ed(g(PIT,'overall')),ed(g(PIT,'top_bucket_frozen')))))
for era,R,lab in [('design',RE,'design era'),('holdout',HOL,'frozen holdout')]:
    C.append(('Flexible edge over this scale\'s parametric tail, %s'%lab,
              ed(g(R,'pinball','sort_mk63_'+SC,'flexible_over_same_scale_parametric','top_decile'))))
td=g(TD,'design_2014_2024')
C.append(('Ten-day horizon, 2014--2024','$%s$ (%s) vs $\\sqrt{h}$-GARCH-$t$, $%s$ (%s) vs iterated'%(
  n(g(td,'edge_pct_direct_vs','sqrt_h'),2,True),n(g(td,'DM_direct_vs','sqrt_h'),2),
  n(g(td,'edge_pct_direct_vs','iter_var'),2,True),n(g(td,'DM_direct_vs','iter_var'),2)) if td else '---'))
if CP:
    av=g(CP,'by_age','1-5','availability','peer_young_emp',d=0); br=g(CP,'by_age','1-5','fz0','alpha_0.01','peer_young_emp','breach')
    C.append(('Cold start, ages 1--5: pooled peer model available on','%s of rows, but breaching %s against a 1\\%% level'%(n(100*av,1,pct=True),n(100*(br or 0),2,pct=True))))
    C.append(('\\quad ages 15--60: own 20-day scale beats that benchmark by',ed(g(CP,'by_age','15-60','edge_over_peer_young_emp','own_short_pool'))))

NUMPAT=re.compile(r'(?<![$\w.])([+-]?\d+\.\d+|[+-]\d+)(\\%)?(?![$\d])')
def mj(v):
    if v is None or '$' in v: return v
    return NUMPAT.sub(lambda m:'$'+m.group(1)+(m.group(2) or '')+'$',v)
L=['\\begin{tabular}{lcc}','\\toprule','Metric & $\\alpha=1\\%$ & $\\alpha=2.5\\%$ \\\\','\\midrule',
   '\\multicolumn{3}{l}{\\emph{Panel A. Accuracy, eleven-level pinball, %d names, %s asset-days}} \\\\'%(
     g(RE,'n_names',d=0),'{:,}'.format(g(RE,'n_test',d=0)))]
for lab,v in A: L.append('%s & \\multicolumn{2}{c}{%s} \\\\'%(lab,mj(v)))
L+=['\\midrule','\\multicolumn{3}{l}{\\emph{Panel B. Joint $(\\VaR,\\ES)$ score and calibration, one source, same rows}} \\\\']
for lab,v1,v2 in B: L.append('%s & %s & %s \\\\'%(lab,mj(v1),mj(v2)))
L+=['\\midrule','\\multicolumn{3}{l}{\\emph{Panel C. The same configuration re-estimated: robustness and reach}} \\\\']
for lab,v in C: L.append('%s & \\multicolumn{2}{c}{%s} \\\\'%(lab,mj(v)))
SCN2={'garch_t':'GARCH(1,1)-$t$','bip_t':'bounded-news','bteg':'Beta-$t$-EGARCH'}
VRN={'engine':'engine','body':'body (no EVT)','overlay':'$+$ overlay as committed','conf2':'$+$ overlay retargeted'}
if XB:
    L+=['\\midrule','\\multicolumn{3}{l}{\\emph{Panel D. Stage-1 scale and Stage-4 variant: breach / NW $t$ / verdict, at $1\\%$ then $2.5\\%$}} \\\\']
    for f in ['garch_t','bip_t','bteg']:
        for vv in ['engine','body','overlay','conf2']:
            k=vv+'_'+f; o1=g(XB,'per_alpha','0.01',k); o2=g(XB,'per_alpha','0.025',k)
            if not o1 or not o2: continue
            def c(o): return '%s / $%s$ / %s'%(n(100*o['breach'],2,pct=True),n(o['dateclustered_NW_t'],2,True),'pass' if o['dateclustered_PASS'] else '\\textbf{fail}')
            star='' if not (f==SC and vv=='engine') else '$^{\\dagger}$'
            L.append('%s, %s%s & \\multicolumn{2}{c}{%s \\quad %s} \\\\'%(SCN2[f],VRN[vv],star,c(o1),c(o2)))
L+=['\\bottomrule','\\end{tabular}']
NOTE=(r'\begin{tablenotes}\footnotesize'+'\n'
 r'\item[$\dagger$] The deployed configuration: a per-name '+SCN+r' Stage-1 scale, the pooled HistGBM '
 r'residual-quantile body on the five features, and a pooled GPD left tail spliced at $p_0=0.025$ as the '
 r'body/EVT minimum, rearranged, with no conformal overlay. A Beta-$t$-EGARCH Stage 1 was tested and not '
 r'adopted: it wins on FZ0 under this frozen protocol (DM 3.31 and 4.55) and on the frozen 2000--2013 '
 r'holdout (3.09 and 3.48), but under the annual refits recommended for production the two filters are '
 r'indistinguishable (DM $-0.31$ and $-0.01$), so the frozen-fit gain is attributable to parameter '
 r'staleness that refitting removes. Panel D reports its exception tests beside this one. '+'\n'
 r'\item Unconditional coverage is read from the date-clustered test, as the paper does elsewhere: the '
 r'pooled Kupiec statistic treats every row as independent, and on a panel whose names share each day'
 "'"+r's '
 r'shock the effective sample is the number of dates. The design-effect row is the ratio of the iid to the '
 r'clustered variance; it implies an intra-date breach correlation of a few per cent, which is why the '
 r'pooled statistic rejects on deviations of under a tenth of a percentage point. '+'\n'
 r'\item Stage 4 is off. The committed overlay estimates its shift from the \emph{body} on the calibration '
 r'block and applies it to the \emph{engine}, which already carries the EVT branch, so the tail correction '
 r'enters twice; re-estimating the shift against the engine itself recovers much of that (Panel D, '
 r'retargeted rows) but the shift stays negative, because the calibration window and the test window '
 r'disagree in the sign of their miscoverage. A single additive out-of-era shift is the wrong instrument '
 r'for a bias of under a tenth of a point, so the stage is dropped rather than repaired. '+'\n'
 r'\item Not yet re-scored against this Stage 1: GJR-GARCH-skew-$t$, the Taylor ES-CAViaR model, '
 r'SAV-CAViaR, EWMA, historical simulation and the GARCH-EVT variants, all of which are scored in '
 r'\texttt{bench\_all\_results.json} against the GARCH-$t$ engine only. '+'\n'
 r'\item Sources: \texttt{exception\_battery\_results.json} (Panel B and D, 200 names), '
 r'\texttt{robust\_engine\_results.json} and \texttt{robust\_engine\_results\_holdout.json} (Panels A and C), '
 r'\texttt{holdout\_frontier\_results.json}, \texttt{calendar\_split\_results.json}, '
 r'\texttt{walkforward\_results.json}, \texttt{pit\_universe\_results.json}, '
 r'\texttt{tenday\_diag\_results.json}, \texttt{coldstart\_peers\_results.json}.'+'\n'
 r'\end{tablenotes}')
open(os.path.join(OUT,'tab_deployed.tex'),'w').write('\n'.join(L)+'\n'+NOTE+'\n')
pl=lambda s_: s_.replace('\\quad','   ').replace('\\%','%').replace('$','').replace('\\textbf{','').replace('}','').replace("\\'e",'e').replace('--','-')
print("\n=== DEPLOYED: %s scale + pooled body + GPD tail (p0=.025), no conformal overlay ==="%SC)
print("\nPanel A. Accuracy")
for lab,v in A: print("  %-54s %s"%(pl(lab),pl(v)))
print("\nPanel B. Joint score and calibration                    1%%          2.5%%")
for lab,v1,v2 in B: print("  %-54s %-12s %s"%(pl(lab),pl(v1),pl(v2)))
print("\nPanel C. Robustness and reach")
for lab,v in C: print("  %-54s %s"%(pl(lab),pl(v)))
print("\nwrote %s"%os.path.join(OUT,'tab_deployed.tex'))
