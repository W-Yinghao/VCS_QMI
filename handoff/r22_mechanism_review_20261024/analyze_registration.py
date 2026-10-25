"""Read-only audit of all six registered CAL term diagnostics; no GPU or image reads."""
import hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OUT=BASE/'handoff/r22_watch10m_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def mean(v):return statistics.mean(v)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def cos(a,b):return dot(a,b)/(norm(a)*norm(b)) if norm(a)*norm(b)>0 else None
def grouped_mean(rr,fn):
    # Offset mean within perturbation/window, then window within scene.
    g=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in rr:
        x=fn(r)
        if x is not None:g[r['scene']][r['core']][r['perturbation']].append(x)
    return mean(mean(mean(mean(v) for v in per.values()) for per in cores.values()) for cores in g.values()) if g else None

def audit():
    freeze=sha(HERE/'FREEZE.json')
    for p,h in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/p)==h,p
    cp=read(HERE/'checks/registration.json');assert cp['status']=='PASS' and cp['freeze_sha256']==freeze
    sources=read(HERE/'SOURCE_LOCK.json')['geometry'];result={};cost=[]
    for ds in ('bf','epfl'):
        units={};ids=None;truth=None
        for arm in ('vcs_exact','bce','nce'):
            p=HERE/'runs'/f'reg_terms_{ds}_{arm}_1001';c=read(p/'COMPLETE.json')
            assert c['status']=='COMPLETE' and c['freeze_sha256']==freeze and c['ds']==ds and c['arm']==arm and c['seed']==1001
            src=sources[f'{ds}_{arm}_1001'];assert src==c['source']
            assert sha(Path(src['checkpoint']))==src['sha256'] and sha(Path(src['completion']))==src['completion_sha256']
            assert sha(p/'POSE_TERMS.jsonl')==c['pose_terms_sha256'] and sha(p/'TOPK_RERANK.jsonl')==c['rerank_sha256']
            rr=rows(p/'POSE_TERMS.jsonl');tt=rows(p/'TOPK_RERANK.jsonl');trace=read(p/'INPUT_TRACE.json')
            assert set(c['roles_read'])=={'CAL'} and {x['role'] for x in trace}=={'CAL'}
            assert len(list(p.glob('ATTEMPT_*.json')))==1 and len(rr)==c['rows']
            keys=[(r['core'],r['perturbation'],r['radius'],r['axis'],r['sign']) for r in rr]
            assert len(keys)==len(set(keys)) and len({r['core'] for r in rr})==4
            assert len(tt)==8 and {(x['core'],x['perturbation']) for x in tt}=={(r['core'],r['perturbation']) for r in rr}
            if ids is None:ids=keys;truth=[r['error_before'] for r in rr]
            else:assert ids==keys and truth==[r['error_before'] for r in rr]
            for r in rr:
                t=r['terms'];assert set(t)=={'P1','negative_P2_half','negative_Q1','negative_Q2_half','POS_QUAD','J'}
                assert r['radius'] in (1.,4.,16.) and r['axis'] in (0,1,2) and r['sign'] in (-1,1)
                assert math.isclose(t['J']['value'],sum(t[k]['value'] for k in list(t)[:4]),abs_tol=2e-6,rel_tol=2e-5)
                for i in range(3):assert math.isclose(t['J']['gradient'][i],sum(t[k]['gradient'][i] for k in list(t)[:4]),abs_tol=2e-5,rel_tol=1e-3)
                for v in t.values():
                    assert all(math.isfinite(x) for x in [v['value'],v['gradient_norm'],v['error_after'],*v['gradient'],*v['updated_pose']])
                    assert math.isclose(v['error_delta'],v['error_after']-r['error_before'],abs_tol=1e-12)
                    assert math.isclose(v['gradient_norm'],norm(v['gradient']),abs_tol=1e-7,rel_tol=1e-5)
            names=list(rr[0]['terms'])
            def summarize(sub):
                return {k:{'error_delta':grouped_mean(sub,lambda r:r['terms'][k]['error_delta']),
                    'improved_fraction':grouped_mean(sub,lambda r:float(r['terms'][k]['error_delta']<0)),
                    'gradient_norm':grouped_mean(sub,lambda r:r['terms'][k]['gradient_norm']),
                    'norm_ratio_to_P1':grouped_mean(sub,lambda r:r['terms'][k]['gradient_norm']/r['terms']['P1']['gradient_norm'] if r['terms']['P1']['gradient_norm'] else None),
                    'gradient_cosine_to_P1':grouped_mean(sub,lambda r:cos(r['terms'][k]['gradient'],r['terms']['P1']['gradient'])),
                    'zero_gradients':sum(r['terms'][k]['gradient_norm']==0 for r in sub)} for k in names}
            rerank={}
            for name in ('P1','J'):
                diffs=[];orders=[]
                for r in tt:
                    t=r['readouts'][name];cs=t['candidates'];assert len(cs)==3
                    assert t['exact_order']==sorted(range(3),key=lambda n:cs[n]['exact_value'],reverse=True)
                    assert all(math.isfinite(c[k]) for c in cs for k in ('coarse_value','exact_value'))
                    diffs.extend(abs(c['exact_value']-c['coarse_value']) for c in cs);orders.append(t['exact_order'])
                rerank[name]={'problems':len(tt),'any_rank_changed':sum(x!=[0,1,2] for x in orders),'top1_changed':sum(x[0]!=0 for x in orders),'mean_abs_value_gap':mean(diffs),'max_abs_value_gap':max(diffs)}
            units[arm]=dict(rows=len(rr),registered_max_rows=144,excluded_out_of_bounds=144-len(rr),core_ids=sorted({r['core'] for r in rr}),
                summary=summarize(rr),by_radius={str(rad):summarize([r for r in rr if r['radius']==rad]) for rad in (1.,4.,16.)},
                by_scene={scene:summarize([r for r in rr if r['scene']==scene]) for scene in sorted({r['scene'] for r in rr})},
                paired_increment={'positive_minus_cos':grouped_mean(rr,lambda r:r['terms']['POS_QUAD']['error_delta']-r['terms']['P1']['error_delta']),
                'full_minus_positive':grouped_mean(rr,lambda r:r['terms']['J']['error_delta']-r['terms']['POS_QUAD']['error_delta'])},
                topk=rerank,input_trace_sha256=sha(p/'INPUT_TRACE.json'),completion_sha256=sha(p/'COMPLETE.json'))
            cost.append(dict(job=c['job'],seconds=c['seconds'],ds=ds,arm=arm,device=c['environment']['gpu_name'],driver=c['environment']['precision']['nvidia_smi_name_driver']))
        result[ds]=units
    r=dict(status='AUDIT_PASS',utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=freeze,datasets=result,cost=dict(records=cost,gpu_hours=sum(x['seconds'] for x in cost)/3600),
        scope='All6 units, seed1001 only; each4CAL windows x2perturbations x3radii x6axis/sign offsets minus host-bound exclusions; no learning curve or generalization evaluation.',
        limitations=['CAL descriptive diagnostic; offsets/perturbations are not independent seeds. No causal claim of full-training improvement.',
        'Fresh Adam first step rescales coordinates: gradient attenuation is not proportional update attenuation. Isolated negative terms are diagnostic controls, not deployable objectives.',
        'Small new CAL coarse passes, not historical top-k; exact re-ranking only within original top3 cannot establish exact-full-grid optimum.',
        'Receipt runtime excludes scheduler startup and source training; CPU/gates shared elsewhere, not counted again.'])
    path=OUT/'ANALYSIS_REG_TERMS_ALL6_20261024.json';path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(path)
    for ds,units in result.items():
        for a,u in units.items():print(ds,a,'rows',u['rows'],'deltas',{k:round(u['summary'][k]['error_delta'],6) for k in ['P1','POS_QUAD','J']},'ratios',{k:round(u['summary'][k]['norm_ratio_to_P1'],5) for k in ['POS_QUAD','J']},'increments',u['paired_increment'],'topk',u['topk'])
    print('cost',r['cost'])

if __name__=='__main__':audit()
