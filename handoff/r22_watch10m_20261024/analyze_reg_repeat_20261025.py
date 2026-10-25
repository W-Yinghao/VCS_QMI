"""Audit all18 balanced-repeat diagnostics; no partial scientific conclusions."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics as S
from analyze_reg_trajectory_20261025 import read,sha,write,finite,norm,equal,avgscene
from analyze_gradpair_20261025 import METHODS,FACTORS,STEPS,WEIGHTS,ENCODERS,SEEDS

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
PACK=BASE/'handoff/r22_reg_repeat_20261025'
CONTRASTS={k:WEIGHTS[k] for k in ('direction_POS_minus_P1','norm_POS_minus_P1','interaction','native_POS_minus_P1')}


def audit_unit(out,fh,gate=False):
    r=read(out/'COMPLETE.json');ds,encoder,seed=r['ds'],r['arm'],r['seed'];key=f'{ds}_{encoder}_{seed}'
    assert r['status']=='COMPLETE' and r['freeze_sha256']==fh and r['gate']==gate
    inputs=read(PACK/'INPUT_LOCK.json')[ds];source=read(PACK/'SOURCE_LOCK.json')[key];baseline=read(PACK/'BASELINE_LOCK.json')[key]
    expected=[(c,i) for c in (inputs['cores'][:1] if gate else inputs['cores']) for i in ((0,) if gate else (0,1))]
    assert r['n_problems']==len(expected) and r['n_paths']==len(expected)*48 and r['repeats']==4
    assert r['starts']==3 and r['adam_steps']==100 and r['arms']==list(METHODS)
    assert r['roles_read']==['CAL'] and r['source_unchanged'] and r['torch_rng_unchanged'] and r['balanced_order']
    assert r['source']==source and r['baseline']==baseline
    for p,k in [('checkpoint','sha256'),('completion','completion_sha256')]:assert sha(source[p])==source[k]
    for p,k in [('completion','completion_sha256'),('trajectories','trajectories_sha256')]:assert sha(baseline[p])==baseline[k]
    old=read(baseline['completion'])
    for k in ('host','gpu_name','torch','numpy','cuda_runtime','cudnn','precision'):assert r['environment'][k]==old['environment'][k]
    oldrows={(q['core'],q['perturbation']):q for q in map(json.loads,Path(baseline['trajectories']).read_text().splitlines())}
    for name,k in [('TRAJECTORIES.jsonl','trajectory_sha256'),('INPUT_TRACE.json','input_trace_sha256')]:assert sha(out/name)==r[k]
    assert {q['role'] for q in read(out/'INPUT_TRACE.json')}=={'CAL'}
    attempts=list(out.glob('ATTEMPT_*.json'));assert len(attempts)==1 and not list(out.glob('FAILURE_*.json'))
    assert read(attempts[0])['job']==r['job'] and read(attempts[0])['freeze_sha256']==fh
    rows=list(map(json.loads,(out/'TRAJECTORIES.jsonl').read_text().splitlines()))
    assert [(q['core'],q['perturbation']) for q in rows]==expected
    metrics=defaultdict(lambda:defaultdict(list));curves=defaultdict(list);rank=defaultdict(list);spreads=[];flat=[];ids={}
    threshold=(9,42) if ds=='bf' else (2.25,10.5)
    for problem_index,q in enumerate(rows):
        finite(q);pid=(q['core'],q['perturbation']);prev=oldrows[pid];scene=q['scene'];assert scene==inputs['scenes'][q['core']]
        for k in ('truth','start_sha256','position_sha256'):assert q[k]==prev[k]
        assert prev['n_pos']==4096
        ids[pid]=(q['truth'],q['position_sha256'])
        initial=[p['initial_pose'] for p in prev['objectives']['P1']['paths']]
        assert [rep['repeat'] for rep in q['repeats']]==list(range(4))
        for i in range(4):assert {rep['order'][i] for rep in q['repeats']}==set(METHODS)
        for rep in q['repeats']:
            n=rep['repeat'];offset=(problem_index+n)%4;order=METHODS[offset:]+METHODS[:offset]
            assert rep['order']==list(order)==list(rep['arms'])
            for position,method in enumerate(order):
                a=rep['arms'][method];paths=a['paths'];assert a['position']==position and len(paths)==3
                assert [p['initial_pose'] for p in paths]==initial
                chosen=max(range(3),key=lambda k:paths[k]['final_cosine']);assert chosen==a['selected_by_common_cosine']
                assert equal(a['mean_three_start_error_native'],S.mean(p['final_error_native'] for p in paths))
                assert equal(a['selected_error_native'],paths[chosen]['final_error_native'])
                vals=dict(path_error=a['mean_three_start_error_native'],selected_error=a['selected_error_native'],
                    strict=float(a['selected_error_native']<threshold[0]),loose=float(a['selected_error_native']<threshold[1]))
                for metric,value in vals.items():metrics[(method,n)][metric].append((scene,value))
                rank[(method,position)].append((scene,vals['path_error']))
                for k,path in enumerate(paths):
                    assert path['arm']==method and (path['direction'],path['norm_source'])==FACTORS[method]
                    tr=path['trajectory'];assert [x['step'] for x in tr]==list(STEPS)==[x['adam_step'] for x in tr]
                    assert path['final_pose']==tr[-1]['pose'] and equal(path['final_error_native'],tr[-1]['error_native'])
                    assert equal(path['final_cosine'],tr[-1]['values']['P1']) and tr[0]['pose']==initial[k]
                    ref=prev['objectives']['P1']['paths'][k]['trajectory'][0]
                    for name,value in ref['values'].items():assert math.isclose(value,tr[0]['values'][name],abs_tol=1e-6,rel_tol=1e-5)
                    for name,values in ref['gradients'].items():
                        assert all(math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-5) for a,b in zip(values,tr[0]['gradients'][name]))
                    for x in tr:
                        assert equal(x['error_native']*(4 if ds=='epfl' else 1),x['error_full'])
                        v=x['values'];assert equal(v['POS_QUAD'],v['P1']+v['negative_P2_half'])
                        assert equal(v['J'],sum(v[a] for a in ('P1','negative_P2_half','negative_Q1','negative_Q2_half')))
                        target=x['gradient_norms'][path['norm_source']];direction=x['gradient_norms'][path['direction']]
                        actual=norm(x['applied_ascent_gradient']);assert math.isclose(actual,target,abs_tol=1e-8,rel_tol=5e-5)
                        if direction>1e-12:assert math.isclose(x['scale'],target/direction,abs_tol=1e-8,rel_tol=2e-5)
                        curves[(method,n,x['step'])].append((scene,x['error_native']))
                        flat.append(dict(dataset=ds,encoder=encoder,seed=seed,core=q['core'],scene=scene,perturbation=q['perturbation'],repeat=n,
                            method=method,order_position=position,start=k,step=x['step'],error_native=x['error_native'],P1=v['P1'],scale=x['scale'],actual_ascent_norm=actual))
        for method in METHODS:
            for k in range(3):
                values=[rep['arms'][method]['paths'][k]['final_error_native'] for rep in q['repeats']]
                spreads.append(dict(core=q['core'],scene=scene,perturbation=q['perturbation'],method=method,start=k,values=values,
                    mean=S.mean(values),sd=S.stdev(values),range=max(values)-min(values)))
    summaries={};scenes={}
    for method in METHODS:
        summaries[method]={};scenes[method]={}
        for n in range(4):
            per={metric:avgscene(values) for metric,values in metrics[(method,n)].items()}
            scenes[method][str(n)]=per
            summaries[method][str(n)]={metric:S.mean(v.values()) for metric,v in per.items()}
            summaries[method][str(n)]['curve']={str(step):S.mean(avgscene(curves[(method,n,step)]).values()) for step in STEPS}
    position={m:{str(i):S.mean(avgscene(rank[(m,i)]).values()) for i in range(4)} for m in METHODS}
    return dict(dataset=ds,encoder=encoder,seed=seed,job=r['job'],seconds=r['seconds'],environment=r['environment'],
        receipt_sha256=sha(out/'COMPLETE.json'),n_paths=r['n_paths'],summaries=summaries,scene_values=scenes,within_path_spreads=spreads,
        execution_position_path_means=position),flat,ids


def main(check_gates=False,dataset=None):
    jobs=read(PACK/'JOBS_full.json');assert len(jobs)==18
    datasets=(dataset,) if dataset else ('bf','epfl')
    jobs=[j for j in jobs if any(j['group']==f'other_reg_repeat_{ds}' for ds in datasets)]
    expected_units=9*len(datasets);assert len(jobs)==expected_units
    expected_paths=sum((16 if ds=='bf' else 20)*9*48 for ds in datasets)
    slug=dataset.upper() if dataset else 'ALL18'
    if not check_gates:
        missing=[j['name'] for j in jobs if not Path(j['completion']).exists()]
        if missing:
            print(json.dumps(dict(status='WAITING_FOR_COMPLETE_SERIES',datasets=datasets,expected=expected_units,complete=expected_units-len(missing),missing=missing)));return
    fh=sha(PACK/'FREEZE.json');freeze=read(PACK/'FREEZE.json')
    for rel,h in freeze['files'].items():assert sha(BASE/rel)==h,rel
    for name in ('CPU_CHECK.json','STATIC_CHECK.json'):
        r=read(PACK/name);assert r['status']=='PASS' and r['freeze_sha256']==fh
    admission=read(PACK/'ADMISSION.json');assert 'full' in admission['groups']
    gates=[]
    for ds in ('bf','epfl'):
        out=PACK/'runs'/f'gate_{ds}';u,_,_=audit_unit(out,fh,True)
        assert sha(out/'COMPLETE.json')==admission['gpu_gate_hashes'][ds]
        gates.append(dict(dataset=ds,job=u['job'],seconds=u['seconds'],paths=u['n_paths']))
    if check_gates:
        write(HERE/'REPEAT_ANALYZER_GATE_CHECK_20261025.json',dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(Path(__file__)),freeze_sha256=fh,gates=gates))
        print('ANALYZER_GATE_AUDIT PASS (both datasets,96paths,672 recorded steps)');return
    units={};flat=[];identities={}
    for spec in jobs:
        u,rows,ids=audit_unit(Path(spec['completion']).parent,fh);key=f"{u['dataset']}_{u['encoder']}_{u['seed']}"
        for pid,value in ids.items():
            ident=(u['dataset'],*pid)
            if ident in identities:assert identities[ident]==value
            identities[ident]=value
        units[key]=u;flat.extend(rows)
    assert len(units)==expected_units and sum(u['n_paths'] for u in units.values())==expected_paths
    results={}
    for ds in datasets:
        results[ds]={}
        for encoder in ENCODERS:
            subset={str(s):units[f'{ds}_{encoder}_{s}'] for s in SEEDS};contrasts={}
            for name,w in CONTRASTS.items():
                seed_stats={};allscenes={}
                for seed,u in subset.items():
                    reps=[];ss=[]
                    for n in range(4):
                        sc=u['scene_values'];values={scene:sum(w[m]*sc[m][str(n)]['path_error'][scene] for m in METHODS) for scene in sc[METHODS[0]][str(n)]['path_error']}
                        ss.append(values);reps.append(S.mean(values.values()))
                    seed_stats[seed]=dict(replicates=reps,mean=S.mean(reps),sd=S.stdev(reps),range=max(reps)-min(reps))
                    allscenes[seed]={scene:S.mean(v[scene] for v in ss) for scene in ss[0]}
                seedmeans=[v['mean'] for v in seed_stats.values()];mean=S.mean(seedmeans)
                aggregate=[S.mean(v['replicates'][n] for v in seed_stats.values()) for n in range(4)]
                sceneavg={sc:S.mean(v[sc] for v in allscenes.values()) for sc in next(iter(allscenes.values()))}
                loso={sc:S.mean(v for k,v in sceneavg.items() if k!=sc) for sc in sceneavg}
                sign=1 if mean>0 else -1 if mean<0 else 0
                criteria=dict(all3_seed_means_same_sign=all(sign*v>0 for v in seedmeans),all4_aggregate_repeats_same_sign=all(sign*v>0 for v in aggregate),
                    stable_loso=all(sign*v>0 for v in loso.values()),mean_exceeds_repeat_spread=min(abs(v) for v in seedmeans)>=2*max(v['sd'] for v in seed_stats.values()))
                contrasts[name]=dict(mean_native=mean,seed_mean_sd=S.stdev(seedmeans),per_seed=seed_stats,aggregate_replicates=aggregate,per_scene=sceneavg,
                    leave_one_scene_out=loso,criteria=criteria,precision_qualified_directional_signal=all(criteria.values()))
            means={m:{metric:S.mean(u['summaries'][m][str(n)][metric] for u in subset.values() for n in range(4)) for metric in ('path_error','selected_error','strict','loose')} for m in METHODS}
            results[ds][encoder]=dict(method_means=means,contrasts=contrasts)
    report=dict(status='COMPLETE_AUDIT_PENDING_INTERPRETATION',utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=fh,analyzer_sha256=sha(Path(__file__)),
        frozen_files_verified=len(freeze['files']),datasets=datasets,complete_units=expected_units,paths=expected_paths,adam_updates=expected_paths*100,units=units,results=results,
        cost=dict(formal_receipt_gpu_hours=sum(u['seconds'] for u in units.values())/3600,shared_gate_seconds=sum(g['seconds'] for g in gates),gates=gates,
            failures='No failed attempts in reviewed units; previous gradient reuse failures and prior studies remain separate, not zero cost.',note='Receipt elapsed excludes scheduler/startup overhead; CPU check cost separate. Both gates are shared by BF/EPFL and must be counted once when combining reports.'),
        limitations=['Repeated-development CAL; three seeds; correlated repeats/starts/perturbations; no TEST/confirmatory claim.',
            'Descriptive precision screen is not a significance test; repeated measurements do not create new independent seeds.',
            'Balanced order does not isolate CUDA/root causes or all between-job variation. Four-arm paths exclude fullJ and are not completeVCS objectives.',
            'Scene-equal path means and common-P1-selected success are separate outcomes; no truth-based selection or dataset pooling.',
            'Actual applied gradients differ from second-autograd diagnostic gradients; first-pass norms/scales audited.'])
    write(HERE/f'ANALYSIS_REG_REPEAT_{slug}_20261025.json',report)
    with (HERE/f'REG_REPEAT_{slug}_CURVES_20261025.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=[f'固定代码平衡顺序重复诊断{slug} {expected_units}/{expected_units}完整核验；4重复×4臂×3起点×100步，{expected_paths}条路径。','主终点scene等权平均三起点原生误差；负为误差降低。各数据集分别报告；精度方向筛选非显著性检验。']
    for ds in results:
        for encoder,v in results[ds].items():
            lines.append(ds+' '+encoder)
            for name,c in v['contrasts'].items():
                lines.append(f'  {name}: mean={c["mean_native"]:+.8f}; seedmeans={[s["mean"] for s in c["per_seed"].values()]}; repeatSD={[s["sd"] for s in c["per_seed"].values()]}; criteria={c["criteria"]}; precision_signal={c["precision_qualified_directional_signal"]}')
    lines.append('成本：'+json.dumps(report['cost'],ensure_ascii=False));lines.append('完整解释/下一步须结合各seed、曲线、每scene、重复波动与统一P1选点结果后补充，再登记reviewed。')
    (HERE/f'ANALYSIS_REG_REPEAT_{slug}_20261025.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],cost=report['cost'],results=results),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-gates',action='store_true');p.add_argument('--ds',choices=['bf','epfl']);a=p.parse_args();main(a.check_gates,a.ds)
