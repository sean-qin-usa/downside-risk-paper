# make_synthetic_truth_results.py -- consolidate the runs of job_synthetic_truth.py into the one file Table
# OA.16 cites, results/paper/synthetic_truth_results.json. Reads synthetic_truth_{garch,bteg}.json from the
# project directory (written by that job) and folds in the three placebo runs plus the quadrature size check
# that motivated the ES convention. Run from the repo root after the generator:
#     python job_synthetic_truth.py garch && python job_synthetic_truth.py bteg
#     python make_synthetic_truth_results.py
# Set GBC_PROJ to override the project path. The placebo rows are carried as literals because those runs score
# pinball only; their FZ0 was never computed and the file says so rather than leaving a silent gap.
import os, json
P = os.environ.get("GBC_PROJ", os.environ.get("GBC_PROJECT_DIR", r"C:\Users\OWNER\Claude\Projects\GBC Project"))
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "paper")

PLACEBOS = {
    'placebo_1d':  (221400, {'param_bteg':0.396006,'engine_bteg':0.396553,'engine_garch_t':0.398549,'param_garch_t':0.399877},  5.68, 9.96, -2.42, -1.92, -6.82),
    'placebo_5d':  (220600, {'param_bteg':0.398659,'engine_bteg':0.399508,'engine_garch_t':0.401627,'param_garch_t':0.403212},  5.54, 9.47, -3.60, -3.43, -7.70),
    'placebo_20d': (217600, {'param_bteg':0.401262,'engine_bteg':0.402204,'engine_garch_t':0.404210,'param_garch_t':0.405798},  4.43, 9.51, -2.98, -3.14, -5.59),
}

def main():
    out = {
      'note':'Synthetic-truth and placebo tests for the frontier claim. Two data-generating processes with a '
             'known answer plus three shift placebos on the real panel. Engine ES uses the CONVERGED integral '
             '(es_integral: body interpolated on [alpha/40, alpha], pooled GPD exact at every node, sub-floor '
             'region in closed form); parametric ES is closed form. The 20-node values are kept beside it '
             'because the engine-versus-truth FZ0 gaps are smaller than the 20-node error, so their sign was '
             'not identified under the committed convention.',
      'generated_by':'code/paper/job_synthetic_truth.py, consolidated by code/paper/make_synthetic_truth_results.py',
      'what_each_test_answers':{
        'dgp_garch':'data generated from a per-name GARCH(1,1)-t. The TRUE model is param_garch_t. A frontier '
                    'that is an artifact of the flexible shape would show the engine beating the truth; it must not.',
        'dgp_bteg':'the mirror image, generated from Beta-t-EGARCH. The true model is param_bteg.',
        'placebo_k':'real panel, every forecast scored against the return k days later. Diagnostic value is '
                    'LIMITED and stated as such: both models share the same GARCH scale, so a shape advantage '
                    'survives any such shift. These rows bound how much of the frontier a pure timing artifact '
                    'could explain, they do not test for one.'},
      'dgps':{}, 'placebos':{},
      'quadrature_size_check':{
        'note':'why the ES convention had to be fixed before any of this was read. McNeil-Frey test size in '
               'simulation under a correctly specified model, 5% nominal.',
        'MF_size_exact_ES':0.060,'MF_size_20node_ES':0.225,
        'engine_MF_p_canon200_1pct':{'20_nodes':0.0673,'200_nodes':0.3822,'2000_nodes':0.3832},
        'conclusion':'the "McNeil-Frey rejects everywhere" pattern is substantially a quadrature artifact; the '
                     'size of the test with the 20-node ES is 22.5% against a nominal 5%.'}}
    for m in ('garch','bteg'):
        d = json.load(open(os.path.join(P, 'synthetic_truth_%s.json' % m)))
        true_scale = 'garch_t' if m == 'garch' else 'bteg'
        rec = {'dgp':d['dgp'],'true_model':'param_'+true_scale,'n_names':d['n_names'],'n_test':d['n_test'],
               'test_length_per_name':d['test_length_per_name'],
               'mean_pinball_11tau':d['mean_pinball_11tau'],
               'frontier_DM_engine_over_own_parametric':d['frontier_DM_engine_over_own_parametric'],
               'cross_DM_engine_garch_t_over_param_bteg':d['cross_DM_engine_garch_t_over_param_bteg'],
               'fz0':{}}
        for a in ('0.01','0.025'):
            f = d['fz0'][a]; sub = {}
            for sc in ('garch_t','bteg'):
                cv = f['engine_%s__convergedES' % sc]
                sub[sc] = {'param':f['param_'+sc],'engine_ES_20node':f['engine_'+sc],'engine_ES_converged':cv,
                           'engine_minus_param_converged':round(cv - f['param_'+sc], 6),
                           'is_true_scale':sc == true_scale,
                           'ES_ratio_conv_over_20node':f['engine_%s__ES_ratio_conv_over_20node' % sc],
                           'relK_halfK_vs_K':f['engine_%s__relK_halfK_vs_K' % sc],
                           'breach_control_identical_both_conventions':f['engine_%s__breach_control' % sc]}
            rec['fz0'][a] = sub
        out['dgps'][m] = rec
    for tag,(nrows,pl,dmg,dmgt,dmb,dmbt,cross) in PLACEBOS.items():
        out['placebos'][tag] = {'n_names':200,'n_test':nrows,'mean_pinball_11tau':pl,
          'frontier_DM_engine_over_own_parametric':{'garch_t':{'overall':dmg,'top_mk63_decile':dmgt},
                                                    'bteg':{'overall':dmb,'top_mk63_decile':dmbt}},
          'cross_DM_engine_garch_t_over_param_bteg':cross,
          'fz0':None,'fz0_note':'not computed: these runs were scored on pinball only. The placebo is not '
                                'diagnostic for the reason stated above, so an FZ0 column would add no evidence.'}
    out['reading']=(
      'In both DGPs the flexible engine is indistinguishable from the TRUE parametric model on FZ0 -- it loses '
      'by 2.7e-4 and 1.0e-5 when GARCH-t is true, and by 2.9e-4 at 1% while edging ahead by 1.8e-4 at 2.5% when '
      'Beta-t-EGARCH is true -- while against a MISSPECIFIED scale it wins by 1.6e-3 to 6.4e-3, six to five '
      'hundred times larger. The frontier therefore appears only against misspecification, which is what the '
      'claim requires, and the estimator costs essentially nothing when the parametric form is correct. The '
      'placebos keep an edge on the GARCH-t scale (DM 4.4-5.7 overall, 9.5-10.0 in the top decile) but this is '
      'expected and not evidence of a timing artifact: both models share the same conditional scale, so any '
      'shift that preserves the scale preserves the shape advantage.')
    fn = os.path.normpath(os.path.join(OUTDIR, 'synthetic_truth_results.json'))
    json.dump(out, open(fn, 'w'), indent=2)
    print('wrote %s' % fn)

if __name__ == '__main__':
    main()
