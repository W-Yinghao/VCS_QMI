"""Audit the complete same-job direction x norm experiment, one dataset at a time."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as S

from analyze_reg_trajectory_20261025 import read, sha, equal, avgscene, norm, finite, write

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PACK = BASE/'handoff/r22_reg_gradfactor_paired_20261025'
ENCODERS = ('vcs_exact','bce','nce')
SEEDS = (1001,1002,1003)
METHODS = ('P1_NATIVE','POS_NATIVE','P1_DIR_POS_NORM','POS_DIR_P1_NORM')
FACTORS = dict(zip(METHODS, [('P1','P1'),('POS_QUAD','POS_QUAD'),('P1','POS_QUAD'),('POS_QUAD','P1')]))
STEPS = (0,1,5,10,25,50,100)
WEIGHTS = {
    'direction_POS_minus_P1':dict(zip(METHODS,[-.5,.5,-.5,.5])),
    'norm_POS_minus_P1':dict(zip(METHODS,[-.5,.5,.5,-.5])),
    'interaction':dict(zip(METHODS,[1,1,-1,-1])),
    'native_POS_minus_P1':dict(zip(METHODS,[-1,1,0,0])),
    'P1_direction_with_POS_norm_minus_P1':dict(zip(METHODS,[-1,0,1,0])),
    'POS_direction_with_P1_norm_minus_POS':dict(zip(METHODS,[0,-1,0,1])),
}


def main(ds):
    specs=[j for j in read(PACK/'JOBS_full.json') if j['group']==f'other_reg_gradpair_{ds}']
    assert len(specs)==9
    missing=[j['name'] for j in specs if not Path(j['completion']).exists()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_COMPLETE_SERIES',dataset=ds,missing=missing)));return
    freeze=read(PACK/'FREEZE.json');fh=sha(PACK/'FREEZE.json')
    for rel,h in freeze['files'].items():assert sha(BASE/rel)==h,rel
    for name in ('CPU_CHECK.json','STATIC_CHECK.json'):
        d=read(PACK/name);assert d['status']=='PASS' and d['freeze_sha256']==fh
    admission=read(PACK/'ADMISSION.json');assert 'full' in admission['groups']
    gates=[]
    for data in ('bf','epfl'):
        p=PACK/'runs'/f'gate_{data}'/'COMPLETE.json';r=read(p)
        assert r['status']=='COMPLETE' and r['freeze_sha256']==fh and r['roles_read']==['CAL']
        assert sha(p)==admission['gpu_gate_hashes'][data]
        gates.append(dict(ds=data,job=r['job'],seconds=r['seconds']))
    inputs=read(PACK/'INPUT_LOCK.json')[ds];source=read(PACK/'SOURCE_LOCK.json');baseline=read(PACK/'BASELINE_LOCK.json')
    expected={(c,i) for c in inputs['cores'] for i in (0,1)}
    thresholds=(9,42) if ds=='bf' else (2.25,10.5)
    units={};scene_values={};flat=[];input_ids={};receipts={};drifts=[];historical=[]
    for spec in specs:
        out=Path(spec['completion']).parent;r=read(out/'COMPLETE.json');key=f"{r['arm']}_{r['seed']}";fullkey=f'{ds}_{key}'
        assert r['status']=='COMPLETE' and r['freeze_sha256']==fh and r['ds']==ds and not r['gate']
        assert r['n_problems']==len(expected) and r['adam_steps']==100 and r['starts']==3 and r['arms']==list(METHODS)
        assert r['paired_reference_paths']==6*len(expected) and r['historical_reference_reuse'] is False
        assert r['roles_read']==['CAL'] and r['source_unchanged'] and r['source']==source[fullkey]
        assert sha(r['source']['checkpoint'])==r['source']['sha256']
        assert sha(r['source']['completion'])==r['source']['completion_sha256']
        assert r['baseline']==baseline[fullkey]
        previous=read(r['baseline']['completion'])
        assert sha(r['baseline']['completion'])==r['baseline']['completion_sha256']
        assert sha(r['baseline']['trajectories'])==r['baseline']['trajectories_sha256']
        for k in ('host','gpu_name','torch','numpy','cuda_runtime','cudnn','precision'):
            assert r['environment'][k]==previous['environment'][k]
        oldrows={(q['core'],q['perturbation']):q for q in map(json.loads,Path(r['baseline']['trajectories']).read_text().splitlines())}
        for name,k in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256'),('HISTORICAL_DIFFERENCES.json','historical_differences_sha256')]:assert sha(out/name)==r[k]
        assert {q['role'] for q in read(out/'INPUT_TRACE.json')}=={'CAL'}
        attempts=list(out.glob('ATTEMPT_*.json'));assert len(attempts)==1 and not list(out.glob('FAILURE_*.json'))
        assert read(attempts[0])['job']==r['job']
        olddiff=read(out/'HISTORICAL_DIFFERENCES.json');assert len(olddiff)==r['paired_reference_paths']
        historical.extend(dict(unit=key,**v) for v in olddiff)
        rows=list(map(json.loads,(out/'TRAJECTORIES.jsonl').read_text().splitlines()))
        assert len(rows)==len(expected) and {(q['core'],q['perturbation']) for q in rows}==expected
        metrics={m:defaultdict(list) for m in METHODS};curves={m:defaultdict(list) for m in METHODS}
        scales={m:[] for m in METHODS}
        for q in rows:
            finite(q);pid=(q['core'],q['perturbation']);old=oldrows[pid];scene=q['scene']
            assert scene==inputs['scenes'][q['core']] and old['n_pos']==4096
            for k in ('truth','start_sha256','position_sha256'):assert q[k]==old[k]
            signature=(q['truth'],q['position_sha256'])
            if pid in input_ids:assert input_ids[pid]==signature
            input_ids[pid]=signature
            initial=[p['initial_pose'] for p in old['objectives']['P1']['paths']]
            assert list(q['arms'])==list(METHODS)
            for method in METHODS:
                z=q['arms'][method];paths=z['paths'];assert len(paths)==3
                assert [p['initial_pose'] for p in paths]==initial
                assert z['selected_by_common_cosine']==max(range(3),key=lambda k:paths[k]['final_cosine'])
                assert equal(z['mean_three_start_error_native'],S.mean(p['final_error_native'] for p in paths))
                assert equal(z['selected_error_native'],paths[z['selected_by_common_cosine']]['final_error_native'])
                metrics[method]['path_error'].append((scene,z['mean_three_start_error_native']))
                metrics[method]['selected_error'].append((scene,z['selected_error_native']))
                for name,t in zip(('strict','loose'),thresholds):metrics[method][name].append((scene,float(z['selected_error_native']<t)))
                for k,path in enumerate(paths):
                    assert path['arm']==method and (path['direction'],path['norm_source'])==FACTORS[method]
                    tr=path['trajectory'];assert [x['step'] for x in tr]==list(STEPS)==[x['adam_step'] for x in tr]
                    assert path['final_pose']==tr[-1]['pose'] and equal(path['final_error_native'],tr[-1]['error_native'])
                    assert equal(path['final_cosine'],tr[-1]['values']['P1']) and tr[0]['pose']==initial[k]
                    ref=q['arms']['P1_NATIVE']['paths'][k]['trajectory'][0]
                    for name in ref['values']:assert math.isclose(ref['values'][name],tr[0]['values'][name],abs_tol=1e-6,rel_tol=1e-5)
                    for x in tr:
                        assert equal(x['error_native']*(4 if ds=='epfl' else 1),x['error_full'])
                        v=x['values'];assert equal(v['POS_QUAD'],v['P1']+v['negative_P2_half'])
                        assert equal(v['J'],sum(v[a] for a in ('P1','negative_P2_half','negative_Q1','negative_Q2_half')))
                        target=x['gradient_norms'][path['norm_source']];direction=x['gradient_norms'][path['direction']]
                        actual=norm(x['applied_ascent_gradient'])
                        assert math.isclose(actual,target,abs_tol=1e-8,rel_tol=5e-5)
                        if direction>1e-12:assert math.isclose(x['scale'],target/direction,abs_tol=1e-8,rel_tol=2e-5)
                        duplicate=[v*x['scale'] for v in x['gradients'][path['direction']]]
                        drift=max(abs(a-b) for a,b in zip(duplicate,x['applied_ascent_gradient']))
                        drifts.append(dict(unit=key,core=q['core'],perturbation=q['perturbation'],method=method,start=k,step=x['step'],max_abs=drift,relative_to_actual_norm=drift/max(actual,1e-12)))
                        scales[method].append(x['scale']);curves[method][x['step']].append((scene,x['error_native']))
                        flat.append(dict(dataset=ds,encoder=r['arm'],seed=r['seed'],core=q['core'],scene=scene,perturbation=q['perturbation'],method=method,start=k,step=x['step'],error_native=x['error_native'],terminal_selection_energy_P1=x['values']['P1'],scale=x['scale'],actual_ascent_norm=actual))
        summaries={}
        for method in METHODS:
            scene_values[(key,method)]={m:avgscene(v) for m,v in metrics[method].items()}
            summaries[method]={m:S.mean(v.values()) for m,v in scene_values[(key,method)].items()}
            summaries[method]['curve']={str(s):S.mean(avgscene(v).values()) for s,v in curves[method].items()}
            summaries[method]['recorded_scale_range']=[min(scales[method]),max(scales[method])]
        units[key]=dict(encoder=r['arm'],seed=r['seed'],job=r['job'],seconds=r['seconds'],environment=r['environment'],summaries=summaries)
        receipts[str((out/'COMPLETE.json').relative_to(BASE))]=sha(out/'COMPLETE.json')
    assert set(units)=={f'{a}_{s}' for a in ENCODERS for s in SEEDS}
    contrasts={};means={}
    for encoder in ENCODERS:
        contrasts[encoder]={}
        means[encoder]={m:{k:S.mean(units[f'{encoder}_{s}']['summaries'][m][k] for s in SEEDS) for k in ('path_error','selected_error','strict','loose')} for m in METHODS}
        for name,weights in WEIGHTS.items():
            per_seed={};per_scene={}
            for seed in SEEDS:
                key=f'{encoder}_{seed}';ref=scene_values[(key,'P1_NATIVE')]
                values={metric:{sc:sum(w*scene_values[(key,m)][metric][sc] for m,w in weights.items()) for sc in ref[metric]} for metric in ref}
                per_seed[str(seed)]={metric:S.mean(v.values()) for metric,v in values.items()};per_scene[str(seed)]=values['path_error']
            scene_avg={sc:S.mean(v[sc] for v in per_scene.values()) for sc in next(iter(per_scene.values()))}
            mean=S.mean(scene_avg.values());loso={sc:S.mean(v for s,v in scene_avg.items() if s!=sc) for sc in scene_avg}
            neg=sum(v['path_error']<0 for v in per_seed.values());pos=sum(v['path_error']>0 for v in per_seed.values())
            signal=(neg>=2 and all(v<0 for v in loso.values())) if mean<0 else (pos>=2 and all(v>0 for v in loso.values())) if mean>0 else False
            contrasts[encoder][name]=dict(weights=weights,primary_mean_native=mean,per_seed=per_seed,per_scene_native=per_scene,leave_one_scene_out_mean_native=loso,negative_seeds=neg,positive_seeds=pos,registered_directional_signal=signal)
        for seed in SEEDS:
            c=contrasts[encoder]
            assert equal(c['direction_POS_minus_P1']['per_seed'][str(seed)]['path_error']+c['norm_POS_minus_P1']['per_seed'][str(seed)]['path_error'],c['native_POS_minus_P1']['per_seed'][str(seed)]['path_error'])
    report=dict(status='COMPLETE_REVIEW',utc=datetime.now(timezone.utc).isoformat(),dataset=ds,freeze_sha256=fh,analyzer_sha256=sha(Path(__file__)),
        frozen_files_verified=len(freeze['files']),complete_units=9,n_problems_per_unit=len(expected),paths_per_problem=12,updates_per_path=100,
        receipt_hashes=receipts,units=units,scene_equal_group_means=means,contrasts=contrasts,
        repeated_backward_diagnostic=dict(max_abs=max(v['max_abs'] for v in drifts),max_relative_to_actual_norm=max(v['relative_to_actual_norm'] for v in drifts),worst=sorted(drifts,key=lambda v:v['relative_to_actual_norm'],reverse=True)[:12]),
        historical_native_trajectory_differences=historical,
        cost=dict(receipt_gpu_hours=sum(v['seconds'] for v in units.values())/3600,shared_gate_records=gates,shared_gate_seconds=sum(v['seconds'] for v in gates),old_failed_reuse_gate_scheduler_seconds=10,
                  note='Shared successful gates and failed reuse gates counted once across BF/EPFL. Original trajectories, models, prior import failures and CPU cost separately retained. Receipt time is not scheduler allocation.'),
        limitations=['Repeated CAL development data; three seeds; correlated starts and perturbations. No TEST or confirmatory claim.',
            'Same four arms per job and same starts per source. Between encoders, original coarse starts may differ.',
            'The actual ascent vector is applied_ascent_gradient. Full-term gradients are a separate autograd call and can differ slightly; first-pass norms/scale audited, repeated-backward drift disclosed.',
            'Gradient composition at each changing current pose includes Adam history and nonlinear interactions; not an additive decomposition of training or generalization.',
            'Primary path means and common-P1 candidate selection are distinct; do not choose starts by truth.',
            'No fullJ arm here: positive-only/crossed paths are mechanism controls, not fullVCS-QMI.',
            'Historical native trajectories are not reused as comparators after failed reproduction gate; drift retained descriptively.'])
    write(HERE/f'ANALYSIS_GRADPAIR_{ds.upper()}_20261025.json',report)
    with (HERE/f'GRADPAIR_{ds.upper()}_CURVES_20261025.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=[f'{ds.upper()} 同作业方向×范数四臂9/9完整分析（固定CAL；2026-10-25）',
           f'3个训练目标×3seed；每项{len(expected)}问题×3共同起点×4臂×100真实Adam更新。源30k模型不训练。',
           '冻结、源、数据/位置/起点、完整预算、实际梯度范数/比例、CAL边界、单次尝试和统一末P1选择已核验。',
           '主指标scene等权三起点终点误差；下列为原生像素，负为误差降低。']
    for encoder in ENCODERS:
        lines.append('训练表示 '+encoder)
        for name,c in contrasts[encoder].items():
            loso=list(c['leave_one_scene_out_mean_native'].values())
            lines.append(f'  {name}: mean={c["primary_mean_native"]:+.7f}; seeds='+str([c['per_seed'][str(s)]['path_error'] for s in SEEDS])+f'; LOSO=[{min(loso):+.7f},{max(loso):+.7f}]; signal={c["registered_directional_signal"]}')
        lines.append('  四臂均值(路径/统一P1选择/严格%/宽松%): '+ '; '.join(f'{m}={means[encoder][m]["path_error"]:.5f}/{means[encoder][m]["selected_error"]:.5f}/{100*means[encoder][m]["strict"]:.3f}/{100*means[encoder][m]["loose"]:.3f}' for m in METHODS))
    lines += [f'本组正式回执{report["cost"]["receipt_gpu_hours"]:.6f}GPUh；两数据集共享成功门禁17.338887秒及失败复用门禁调度10秒只计一次。',
              '方向/范数两主效应之和等于原生POS−P1；交互另报，不能据此唯一归因训练收益。配对在同一作业内计算，旧轨迹仅为输入和描述性偏差参照。',
              '实际更新向量与再次autograd记录不是逐位同一值，差异已留存；不改冻结门槛，不把重复求导作实际应用梯度。',
              '两数据集分别解读；CAL反复使用、3seed、起点/扰动相关，不作为独立泛化验证。']
    (HERE/f'ANALYSIS_GRADPAIR_{ds.upper()}_20261025.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],dataset=ds,gpu_hours=report['cost']['receipt_gpu_hours'],means=means,
        contrasts={a:{k:{f:c[f] for f in ('primary_mean_native','negative_seeds','positive_seeds','registered_directional_signal')} for k,c in v.items()} for a,v in contrasts.items()}),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ds',choices=['bf','epfl'],required=True);main(p.parse_args().ds)
