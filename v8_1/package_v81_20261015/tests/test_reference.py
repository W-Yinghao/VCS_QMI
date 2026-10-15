import copy, json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import spearmanr
from core import (array1, simplex, histogram, tails, moments, distribution_metrics,
                  mse_decomposition, spearman, kendall_b, pairwise_accuracy,
                  selection_metrics, decision_agreement, swap_decomposition,
                  cluster_bootstrap, validate_rows, summarize)
from audit_logs import inspect_log, inspect_source
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('bad',[[],[float('nan')],[float('inf')],[[1,2]]])
def test_bad_array(bad):
    with pytest.raises(ValueError): array1(bad)

@pytest.mark.parametrize('bad',[[.2]*4,[.2,.2,.2,.2,.1],[-.1,.1,.2,.3,.5],[0,0,0,0,2]])
def test_bad_simplex(bad):
    with pytest.raises(ValueError): simplex(bad)

@pytest.mark.parametrize('bad',[[],[0],[6],[2.5],[1,float('nan')]])
def test_bad_votes(bad):
    with pytest.raises(ValueError): histogram(bad)

@pytest.mark.parametrize('v',[[1],[1,2,3],[1,1,3,3,5,5],[2,4,5]])
def test_votes_tail_identity(v):
    h=histogram(v); assert 1+sum(tails(h))==pytest.approx(np.mean(v));assert moments(h)['mu']==pytest.approx(np.mean(v))

@pytest.mark.parametrize('seed',range(5))
def test_rps_identity(seed):
    rng=np.random.default_rng(seed);q=rng.dirichlet(np.ones(5));h=rng.dirichlet(np.ones(5))
    d=distribution_metrics(q,h)
    assert d['rps_random_rater']==pytest.approx(d['cdf_fit_mse']+d['vote_constant'])
    assert d['rps_identity_residual']<1e-14

@pytest.mark.parametrize('seed',range(5))
def test_weighted_decomposition(seed):
    r=np.random.default_rng(seed);a=r.uniform(1,5,20);b=r.uniform(1,5,20);w=r.random(20)
    d=mse_decomposition(a,b,w);assert abs(d['identity_residual'])<1e-12

def test_uniform_shift_only_bias():
    d=mse_decomposition([2,3,4],[1,2,3]);assert d['mse']==1;assert d['centered_mse']==0

def test_ranking_same_not_same_mse():
    assert spearman([1,2,3],[2,4,6])==pytest.approx(1)
    assert mse_decomposition([2,4,6],[1,2,3])['centered_mse']>0

@pytest.mark.parametrize('x,y',[([1,1,2],[2,2,1]),([3,2,1],[1,2,3]),([0,2,3,2],[1,4,2,4])])
def test_spearman_against_scipy(x,y):
    assert spearman(x,y)==pytest.approx(spearmanr(x,y).statistic)

@pytest.mark.parametrize('x,y',[([1],[1]),([2,2,2],[1,2,3]),([1,2,3],[2,2,2])])
def test_constant_correlations_none(x,y):
    assert spearman(x,y) is None;assert kendall_b(x,y) is None

def test_pairwise_ties():
    d=pairwise_accuracy([1,1,1],[1,2,3]);assert d['pair_accuracy']==.5;assert d['informative_pairs']==3

def test_pairwise_all_label_ties():
    d=pairwise_accuracy([1,2,3],[2,2,2]);assert d['pair_accuracy'] is None;assert d['informative_pairs']==0

def test_choice_respects_first_tie():
    assert selection_metrics([2,2,1],[1,5,3])['selected_human_mean']==1

def test_high_agreement_from_constants_is_visible():
    d=decision_agreement([1,1,1],[2,2,2]);assert d['top1_agreement']==1;assert d['top_tie_a']==1;assert d['spearman'] is None

def test_swap_identity():
    d=swap_decomposition([1,0,0],[0,1,-.5]);assert abs(d['identity_residual'])<1e-14

def test_swap_no_disagreement():
    d=swap_decomposition([1,1],[0,0]);assert d['delta_when_disagree'] is None;assert d['identity_residual']==0

def test_swap_invalid_same_choice_difference():
    with pytest.raises(ValueError):swap_decomposition([1],[.1])

def test_cluster_bootstrap_deterministic():
    a=cluster_bootstrap([0,1,2],['a','a','b'],reps=99);b=cluster_bootstrap([0,1,2],['a','a','b'],reps=99)
    assert a==b;assert a['n_clusters']==2;assert a['n_pools']==3

def test_cluster_bootstrap_one():
    assert cluster_bootstrap([1,2],['x','x'],reps=20)['ci95'] is None

def test_ensemble_is_not_mean_decision():
    scores=[[5,1,3],[1,4,3]];target=[5,4,1]
    mean_quality=np.mean([selection_metrics(s,target)['selected_human_mean'] for s in scores])
    ens=selection_metrics(np.mean(scores,axis=0),target)['selected_human_mean']
    assert ens != mean_quality

def base_row():
    return dict(dataset='synthetic',role='DEV',method='M',seed=0,checkpoint_id='c',prompt_id='p',pool_id='p',image_id='i',candidate_index=0,score=3.,q=[.2]*5,votes=[1,3,5])

@pytest.mark.parametrize('role',['GENAI_RANK','GENAI_BENCH_CONFIRM','TIFA160','TEST'])
def test_protected_roles(role):
    r=base_row();r['role']=role
    with pytest.raises(ValueError,match='Role not allowed'):validate_rows([r],labelled=True)

def test_label_free_rejects_labels():
    with pytest.raises(ValueError,match='forbidden'):validate_rows([base_row()],labelled=False)

def test_label_free_rejects_dummy_hist():
    r=base_row();r.pop('votes');r['hist']=[.2]*5
    with pytest.raises(ValueError,match='forbidden'):validate_rows([r],labelled=False)

def test_label_free_external_allowed():
    r=base_row();r.pop('votes');r['role']='GENAI_RANK';assert validate_rows([r],labelled=False)

def test_q_score_consistency():
    r=base_row();r['score']=4.
    with pytest.raises(ValueError,match='disagree'):validate_rows([r],labelled=True)

def test_duplicate_row():
    r=base_row()
    with pytest.raises(ValueError,match='Duplicate'):validate_rows([r,r],labelled=True)

def test_hist_mean_consistency():
    r=base_row();r['human_mean']=1
    with pytest.raises(ValueError,match='mismatch'):validate_rows([r],labelled=True)

def test_native_score_no_rating_mse():
    r=base_row();r.pop('q');r['score_kind']='native_score';r['score']=.12
    out=summarize([r],labelled=True);assert 'mse' not in out['per_pool'][0]

def test_missing_candidates_refused():
    a=base_row();b=copy.deepcopy(a);b['image_id']='j';b['candidate_index']=1;c=copy.deepcopy(a);c['method']='N'
    with pytest.raises(ValueError,match='candidate sets'):summarize([a,b,c],labelled=True)

def test_rows_with_hist_mean_only():
    r=base_row();r.pop('votes');r['human_mean']=3
    out=summarize([r],labelled=True);assert out['per_pool'][0]['mse']==0

def test_epoch_zero_after_training():
    d=inspect_log({'arm':'U-CS','selected_epoch':0,'epochs_log':[{'epoch':0,'steps':20,'images_seen':140}]})
    assert d['completed_epoch_count_if_zero_based']==1;assert d['not_equivalent_to_preupdate_init']

def test_epoch_unknown_without_log():
    assert inspect_log({'selected_epoch':0})['status']=='insufficient_log_evidence'

def test_epoch_count_is_not_amp_success():
    r=inspect_log({'selected_epoch':0,'epochs_log':[{'epoch':0,'steps':2}]})
    assert r['successful_optimizer_updates'].startswith('unknown')

def test_static_source_records_order(tmp_path):
    p=tmp_path/'train.py';p.write_text('def cmd_train():\n for epoch in range(5):\n  scaler.step(opt)\n  m=cal_metrics(predict(model))\n  torch.save(m, "x")\n')
    d=inspect_source(p); assert d['calls'][0]['function']=='scaler.step';assert len(d['sha256'])==64

def test_cli_demo(tmp_path):
    f=tmp_path/'rows.jsonl';o=tmp_path/'out.json'
    subprocess.run([sys.executable,str(ROOT/'reference/make_demo.py'),'--out',str(f)],check=True,capture_output=True)
    subprocess.run([sys.executable,str(ROOT/'reference/diagnose.py'),'--rows',str(f),'--label-mode','development','--out',str(o)],check=True,capture_output=True)
    d=json.loads(o.read_text());assert d['n_rows']==36;assert len(d['model_levels'])==3
    p=subprocess.run([sys.executable,str(ROOT/'reference/diagnose.py'),'--rows',str(f),'--label-mode','development','--out',str(o)],capture_output=True)
    assert p.returncode!=0

def test_adapt_v8_schema(tmp_path):
    pred=tmp_path/'p.jsonl';man=tmp_path/'m.jsonl';out=tmp_path/'out.jsonl'
    pred.write_text(json.dumps({'image_id':'x','prompt_id':'p','mu':3.,'q':[.2]*5,'u':[.8,.6,.4,.2]})+'\n')
    man.write_text(json.dumps({'image_id':'x','prompt_id':'p','role':'DEV','overall_votes':[1,3,5]})+'\n')
    args=[sys.executable,str(ROOT/'reference/adapt_v8.py'),'--pred',str(pred),'--public-train-manifest',str(man),'--role','DEV','--method','U-CS','--seed','0','--checkpoint-id','c','--out',str(out)]
    subprocess.run(args,check=True,capture_output=True);r=json.loads(out.read_text());assert r['human_mean']==3

def test_nonzero_mean_bound():
    a=np.array([.8,.6,.4,.2]);b=np.array([.7,.5,.3,.1]); assert (sum(a-b))**2<=16*np.mean((a-b)**2)+1e-12

def test_different_hist_same_mean_refused():
    a=base_row();b=copy.deepcopy(a);b['method']='N';b['votes']=[3,3,3]
    with pytest.raises(ValueError,match='histograms'):summarize([a,b],labelled=True)
