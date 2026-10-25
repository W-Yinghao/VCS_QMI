"""Audit complete fixed-CAL trajectory series; standard-library analysis only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as S

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PACK=BASE/'handoff/r22_reg_trajectory_v2_20261025'
ARMS=('vcs_exact','bce','nce');SEEDS=(1001,1002,1003)
OBJS=('P1','POS_QUAD','J');STEPS=(0,1,5,10,25,50,100)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def equal(a,b):return math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)
def avgscene(pairs):
    groups=defaultdict(list)
    for scene,value in pairs:groups[scene].append(value)
    return {s:S.mean(v) for s,v in groups.items()}
def norm(x):return math.sqrt(sum(v*v for v in x))
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main(ds):
    freeze=read(PACK/'FREEZE.json');fh=sha(PACK/'FREEZE.json')
    for rel,h in freeze['files'].items():assert sha(BASE/rel)==h,rel
    inp=read(PACK/'INPUT_LOCK.json')[ds];source=read(PACK/'SOURCE_LOCK.json')
    expected={(c,i) for c in inp['cores'] for i in (0,1)}
    specs=[r for r in read(PACK/'JOBS_full.json') if r['group']==f'other_reg_trajectory_{ds}']
    assert len(specs)==9
    assert all(Path(r['completion']).exists() for r in specs),'Wait for all9 matched units'
    units={};receipt_hashes={};flat=[];scene_values={};origins=Counter();input_ids={};local_ratio=[]
    thresholds=(9,42) if ds=='bf' else (2.25,10.5)
    for spec in specs:
        out=Path(spec['completion']).parent;r=read(out/'COMPLETE.json');key=f"{r['arm']}_{r['seed']}"
        assert r['status']=='COMPLETE' and r['freeze_sha256']==fh and not r['gate'] and r['ds']==ds
        assert r['n_problems']==len(expected) and r['adam_steps']==100 and r['starts']==3 and r['record_steps']==list(STEPS)
        assert r['roles_read']==['CAL'] and r['source_unchanged'] and r['source']==source[f'{ds}_{key}']
        assert sha(r['source']['checkpoint'])==r['source']['sha256']
        assert sha(r['source']['completion'])==r['source']['completion_sha256']
        for name,field in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256')]:assert sha(out/name)==r[field]
        assert {x['role'] for x in read(out/'INPUT_TRACE.json')}=={'CAL'}
        attempts=list(out.glob('ATTEMPT_*.json'));assert len(attempts)==1 and not list(out.glob('FAILURE_*.json'))
        assert read(attempts[0])['job']==r['job']
        rows=[json.loads(s) for s in (out/'TRAJECTORIES.jsonl').read_text().splitlines()]
        assert len(rows)==len(expected) and {(q['core'],q['perturbation']) for q in rows}==expected
        metrics={o:defaultdict(list) for o in OBJS};curves={o:defaultdict(list) for o in OBJS}
        for q in rows:
            finite(q);scene=q['scene'];assert scene==inp['scenes'][q['core']] and q['n_pos']==4096
            pid=(q['core'],q['perturbation']);signature=(q['truth'],q['position_sha256'])
            if pid in input_ids:assert input_ids[pid]==signature,(key,pid)
            input_ids[pid]=signature
            assert sha_bytes(json.dumps(q['candidates'],sort_keys=True).encode())==q['start_sha256']
            origins[q['start_origin']]+=1
            initial=[p['initial_pose'] for p in q['objectives']['P1']['paths']]
            for obj in OBJS:
                z=q['objectives'][obj];paths=z['paths'];assert len(paths)==3
                assert [p['initial_pose'] for p in paths]==initial
                assert z['selected_rank']==max(range(3),key=lambda k:paths[k]['final_energy'])
                assert equal(z['mean_three_start_error_native'],S.mean(p['final_error_native'] for p in paths))
                assert equal(z['selected_error_native'],paths[z['selected_rank']]['final_error_native'])
                metrics[obj]['path_error'].append((scene,z['mean_three_start_error_native']))
                metrics[obj]['selected_error'].append((scene,z['selected_error_native']))
                for title,t in zip(('strict','loose'),thresholds):metrics[obj][title].append((scene,float(z['selected_error_native']<t)))
                for k,path in enumerate(paths):
                    tr=path['trajectory'];assert [t['step'] for t in tr]==list(STEPS)
                    assert [t['adam_step'] for t in tr]==list(STEPS)
                    assert path['final_pose']==tr[-1]['pose'] and equal(path['final_energy'],tr[-1]['values'][obj])
                    assert equal(path['final_error_native'],tr[-1]['error_native'])
                    assert tr[0]['pose']==initial[k]
                    for t in tr:
                        assert equal(t['error_native']*(4 if ds=='epfl' else 1),t['error_full'])
                        v=t['values'];assert equal(v['POS_QUAD'],v['P1']+v['negative_P2_half'])
                        assert equal(v['J'],sum(v[a] for a in ('P1','negative_P2_half','negative_Q1','negative_Q2_half')))
                        curves[obj][t['step']].append((scene,t['error_native']))
                        flat.append(dict(dataset=ds,arm=r['arm'],seed=r['seed'],core=q['core'],scene=scene,perturbation=q['perturbation'],objective=obj,start=k,step=t['step'],error_native=t['error_native'],energy=t['values'][obj]))
                    if obj=='P1':
                        g=tr[0]['gradients'];den=norm(g['P1'])
                        if den>0:local_ratio.append(norm(g['POS_QUAD'])/den)
            # All objectives see identical terms/gradients at the shared initial pose.
            for k in range(3):
                ref=q['objectives']['P1']['paths'][k]['trajectory'][0]
                for obj in ('POS_QUAD','J'):
                    cur=q['objectives'][obj]['paths'][k]['trajectory'][0]
                    for term in ref['values']:assert equal(ref['values'][term],cur['values'][term])
        summary={}
        for obj in OBJS:
            scene_values[(key,obj)]={m:avgscene(v) for m,v in metrics[obj].items()}
            summary[obj]={m:S.mean(v.values()) for m,v in scene_values[(key,obj)].items()}
            summary[obj]['curve_mean_three_starts']={str(s):S.mean(avgscene(v).values()) for s,v in curves[obj].items()}
        units[key]=dict(arm=r['arm'],seed=r['seed'],job=r['job'],seconds=r['seconds'],source=r['source'],environment=r['environment'],summaries=summary)
        receipt_hashes[str((out/'COMPLETE.json').relative_to(BASE))]=sha(out/'COMPLETE.json')
    assert set(units)=={f'{a}_{s}' for a in ARMS for s in SEEDS}
    contrasts={};group_means={}
    for arm in ARMS:
        group_means[arm]={o:{m:S.mean(units[f'{arm}_{s}']['summaries'][o][m] for s in SEEDS) for m in ('path_error','selected_error','strict','loose')} for o in OBJS}
        contrasts[arm]={}
        for name,a,b in [('positive_square','POS_QUAD','P1'),('reference_terms','J','POS_QUAD'),('full_vs_cosine','J','P1')]:
            per_seed={};per_scene={}
            for seed in SEEDS:
                key=f'{arm}_{seed}';am=scene_values[(key,a)];bm=scene_values[(key,b)]
                diff={sc:am['path_error'][sc]-bm['path_error'][sc] for sc in am['path_error']}
                per_scene[str(seed)]=diff
                per_seed[str(seed)]={m:S.mean(am[m][sc]-bm[m][sc] for sc in am[m]) for m in am}
            scene_avg={sc:S.mean(v[sc] for v in per_scene.values()) for sc in next(iter(per_scene.values()))}
            mean=S.mean(scene_avg.values());loso={sc:S.mean(v for other,v in scene_avg.items() if other!=sc) for sc in scene_avg}
            negatives=sum(d['path_error']<0 for d in per_seed.values());positives=sum(d['path_error']>0 for d in per_seed.values())
            direction='lower_error' if mean<0 else 'higher_error' if mean>0 else 'tie'
            passed=(negatives>=2 and all(v<0 for v in loso.values())) if mean<0 else (positives>=2 and all(v>0 for v in loso.values())) if mean>0 else False
            contrasts[arm][name]=dict(comparison=f'{a}-{b}',primary_mean_native=mean,per_seed=per_seed,per_scene_native=per_scene,
                leave_one_scene_out_mean_native=loso,negative_seeds=negatives,positive_seeds=positives,direction=direction,registered_directional_signal=passed,
                scenes_negative=sum(v<0 for v in scene_avg.values()),scenes_positive=sum(v>0 for v in scene_avg.values()))
    seconds=sum(u['seconds'] for u in units.values())
    result=dict(status='COMPLETE_REVIEW',utc=datetime.now(timezone.utc).isoformat(),dataset=ds,freeze_sha256=fh,
        frozen_files_verified=len(freeze['files']),complete_units=9,source_checkpoint_step=30000,n_scenes=len(set(inp['scenes'].values())),n_problems_per_unit=len(expected),
        updates_per_path=100,paths_per_unit=9*len(expected),receipt_hashes=receipt_hashes,units=units,scene_equal_group_means=group_means,
        contrasts=contrasts,start_origins=dict(origins),initial_positive_to_cosine_gradient_norm_ratio=dict(mean=S.mean(local_ratio),median=S.median(local_ratio),min=min(local_ratio),max=max(local_ratio)),
        cost=dict(receipt_gpu_hours=seconds/3600,shared_new_gate_seconds=11.343969345092773,old_failed_gate_scheduler_seconds=8,
            note='Two new gates and two old failed gates shared once across BF/EPFL, not charged twice. CPU/source training/earlier mechanism diagnoses separate; receipt is not scheduler allocation.'),
        attempt_note='Frozen runner writes STARTED attempt only. Matching COMPLETE with verified hashes and absence of FAILURE is success authority; exactly one attempt/unit.',
        limitations=['CAL reused repeatedly; development mechanism evidence only.','Three seeds; perturbations and starts are correlated within scene. No pooled BF/EPFL mean.','Same starts within each checkpoint/problem, while representations can generate different coarse starts. Do not interpret across-training-arm comparisons as identical-start interventions.','Positive-square and combined reference removals are mechanism controls, not complete VCS-QMI.','100-step path effects include Adam dynamics and changing points; initial gradient norms alone cannot explain final error.','Reference comparison combines Q1 and Q2; it does not identify their separate causal effects.','Own-energy terminal selection adds ranking to path effects; report separately.'])
    write(HERE/f'ANALYSIS_REG_TRAJECTORY_{ds.upper()}_20261025.json',result)
    with (HERE/f'REG_TRAJECTORY_{ds.upper()}_CURVES_20261025.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[f'{ds.upper()} 固定CAL配准100步轨迹9/9完整分析（2026-10-25）',
           f'全部3训练目标×3seed、每项{len(expected)}问题、每问题3共同COS起点×3精修目标×100步Adam；源30k模型不训练。冻结/源/数据身份/位置/实际100更新/选择逻辑/有限值/访问域与单次尝试核验通过。',
           '主指标为同3起点末端误差均值，scene等权，以下负差代表改善；单位为数据集原生像素。']
    for arm in ARMS:
        lines.append(f'训练表示 {arm}:')
        for name in ('positive_square','reference_terms','full_vs_cosine'):
            c=contrasts[arm][name];sv=[c['per_seed'][str(s)]['path_error'] for s in SEEDS]
            selected=[c['per_seed'][str(s)]['selected_error'] for s in SEEDS]
            loso=list(c['leave_one_scene_out_mean_native'].values())
            lines.append(f"  {c['comparison']}: 主差{c['primary_mean_native']:+.6f}，逐seed{sv}；各自能量选末端后的误差差均值{S.mean(selected):+.6f}；LOSO范围[{min(loso):+.6f},{max(loso):+.6f}]；预登记方向信号{c['registered_directional_signal']}（{c['direction']}）。")
        lines.append('  各读出scene等权(三起点误差/自身选择误差/严格%/宽松%): '+ '; '.join(f'{o}={group_means[arm][o]["path_error"]:.4f}/{group_means[arm][o]["selected_error"]:.4f}/{100*group_means[arm][o]["strict"]:.2f}/{100*group_means[arm][o]["loose"]:.2f}' for o in OBJS))
    lines += [f'成本：本组完成回执{seconds/3600:.6f}GPUh；共享新门禁11.343969秒、旧失败门禁调度8秒跨两数据集只计一次，CPU/源训练另计。',
              '不能从初始梯度变小直接推出最终任务变化；实际Adam轨迹与末端候选排序已分别保留。Q1/Q2此处联合删除，不作各自因果结论。',
              '全部scene保留，LOSO仅描述敏感性；CAL反复使用、扰动/起点相关、跨表示起点可能不同、搜索非确定性均保留。',
              '完成另一数据集后并列判断，不从单数据集或单seed提前改A1/训练配方。无TEST、新模型训练或途中参数修改。']
    (HERE/f'ANALYSIS_REG_TRAJECTORY_{ds.upper()}_20261025.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],dataset=ds,cost_gpu_hours=seconds/3600,means=group_means,
        primary={a:{n:{k:v for k,v in c.items() if k in ['primary_mean_native','direction','registered_directional_signal','negative_seeds']} for n,c in cc.items()} for a,cc in contrasts.items()}),ensure_ascii=False))

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ds',choices=['bf','epfl'],required=True);main(p.parse_args().ds)
