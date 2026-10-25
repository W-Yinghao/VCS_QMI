"""Fixed240-epoch positive-quadratic missing-middle comparison; FIT/CAL records only."""
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as S

from analyze_reg_trajectory_20261025 import read, sha, write, finite

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
PACK=BASE/'handoff/r22_kd_positive_control_v3_20261025'
ANCHORS=('ce','dkd');SEEDS=(31415,16180,57721);METHODS=('cosine','positive_quadratic','vcs')


def scientific_config(c):
    c=copy.deepcopy(c)
    for k in ('name','notes'):c.pop(k,None)
    c['loss']['decision']['method']='ONLY_INTENDED_DIFFERENCE'
    return c


def summarize(history):
    best=max(history,key=lambda x:(x['cal']['top1'],-x['cal']['nll'],-x['epoch']))
    return dict(final_top1=history[-1]['cal']['top1'],last10_top1=S.mean(x['cal']['top1'] for x in history[-10:]),
                best_top1=best['cal']['top1'],best_epoch_1based=best['epoch']+1,
                final_nll=history[-1]['cal']['nll'],final_ece=history[-1]['cal']['ece_15bins'],
                curve={str(e):history[e-1]['cal']['top1'] for e in (40,80,120,150,180,210,240)},
                last10_sparse_probes=dict(positive_cosine=S.mean(x['dependence']['positive_cosine'] for x in history[-10:]),
                    aux_over_anchor=S.mean(x['gradient_probe']['decision_space']['aux_over_anchor'] for x in history[-10:]),
                    gradient_cosine=S.mean(x['gradient_probe']['decision_space']['cosine'] for x in history[-10:])))


def main():
    jobs=read(PACK/'JOBS_full.json');assert len(jobs)==6
    missing=[j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_COMPLETE_SERIES',missing=missing)));return
    freeze=read(PACK/'FREEZE.json');fh=sha(PACK/'FREEZE.json')
    for rel,h in freeze['files'].items():assert sha(BASE/rel)==h,rel
    for n in ('CPU_CHECK.json','STATIC_CHECK.json'):
        r=read(PACK/n);assert r['status']=='PASS' and r['freeze_sha256']==fh
    assert 'full' in read(PACK/'ADMISSION.json')['groups']
    gates=[]
    for seed in (31415,16180):
        g=read(PACK/'gates'/f'{seed}.json');assert g['status']=='COMPLETE' and g['freeze_sha256']==fh
        for anchor in ANCHORS:
            q=PACK/'runs'/f'gate_{anchor}_{seed}';r=read(q/'COMPLETE.json')
            assert r['technical'] and r['epochs']==1 and r['freeze_sha256']==fh and not r['test_read']
            assert sha(q/'history.json')==r['history_sha256']
            gates.append(dict(anchor=anchor,seed=seed,job=r['job'],seconds=r['seconds']))
    locks=read(PACK/'SOURCE_LOCK.json')['controls'];units={};flat=[];identities={};new_seconds=[];old_seconds=[]
    for j in jobs:
        q=Path(j['completion']).parent;r=read(q/'COMPLETE.json');anchor,seed=r['anchor'],r['seed'];key=f'{anchor}_{seed}'
        assert anchor==j['anchor'] and seed==j['seed'] and r['status']=='COMPLETE' and not r['technical']
        assert r['freeze_sha256']==fh and r['epochs']==240 and not r['test_read']
        assert sha(j['config'])==r['config_sha256']
        cfg=read(j['config']);attempt=read(q/'ATTEMPT.json');preflight=read(q/'PREFLIGHT.json')
        assert attempt['status']=='COMPLETE' and attempt['job']==r['job'] and preflight['status']=='PASS'
        assert attempt['seconds']>=r['seconds'] and attempt['freeze_sha256']==fh
        assert preflight['initial_sha256']==r['student_initial_sha256'] and preflight['gpu']==r['gpu'] and preflight['environment']==r['environment']
        assert r['guard']['allowed_opens_in_this_process']>0
        dirs={'positive_quadratic':q}
        for method in ('cosine','vcs'):
            ref=locks[key][method];assert sha(ref['receipt'])==ref['sha256']
            assert str(ref['receipt'])==j['control_receipts'][method]
            dirs[method]=Path(ref['receipt']).parent
        data={};u={}
        for method in METHODS:
            p=dirs[method];receipt=read(p/'COMPLETE.json');run=read(p/'run.json');history=read(p/'history.json')
            assert receipt['status']=='COMPLETE' and run['status']=='complete' and run['epochs_completed']==240
            assert len(history)==240 and [x['epoch'] for x in history]==list(range(240))
            assert sha(p/'history.json')==receipt['history_sha256']
            assert run['teacher_frozen_verified'] and not run['test_loaded_during_train']
            assert run['train_data_role']=='FIT' and run['selection_data_role']=='CAL' and not run['teacher_training']
            assert run['parameters_auxiliary']==0 and scientific_config(run['config'])==scientific_config(cfg)
            assert run['config']['loss']['decision']==dict(method=method,weight=10)
            assert run['config']['training']==dict(batch_size=64,epochs=240,workers=4)
            assert run['config']['optimizer']['milestones']==[150,180,210]
            assert run['environment']==r['environment'] and receipt['gpu']==r['gpu']
            assert run['student_initial_state_sha256']==r['student_initial_sha256']
            assert run['student_selection_rule']=='max_CAL_top1_then_min_CAL_nll_then_earliest_epoch'
            for name,k in [('best.pt','best_checkpoint_sha256'),('best_cal_metrics.json','best_cal_metrics_sha256'),('best_cal_predictions.csv','best_cal_predictions_sha256')]:assert sha(p/name)==run[k]
            if method=='positive_quadratic':
                assert sha(p/'last.pt')==receipt['last_checkpoint_sha256'] and run['config']==cfg
                assert read(p/'resolved_config.json')==cfg
            for x in history:
                finite(x);assert x['fit_n_seen']==44992 and x['cal']['n']==5000
                ep=x['epoch']+1
                expected=.05*(.1**sum(ep>=m for m in (150,180,210)))
                assert math.isclose(x['next_lr'],expected,rel_tol=1e-12,abs_tol=1e-15)
                assert x['dependence']['method']==method
                flat.append(dict(anchor=anchor,seed=seed,method=method,epoch_1based=ep,cal_top1=x['cal']['top1'],cal_nll=x['cal']['nll'],fit_loss=x['fit_loss'],next_lr=x['next_lr'],sample_order_sha256=x['sample_order_sha256'],positive_cosine=x['dependence']['positive_cosine'],aux_over_anchor=x['gradient_probe']['decision_space']['aux_over_anchor'],gradient_cosine=x['gradient_probe']['decision_space']['cosine']))
            identity={k:run[k] for k in ('teacher_initial_state_sha256','student_initial_state_sha256','source_sha256','split_sha256','environment','teacher_provenance')}
            identity['gpu']=receipt['gpu'];identity['epoch_order']=[x['sample_order_sha256'] for x in history]
            if key in identities:assert identity==identities[key],(key,method)
            identities[key]=identity
            summary=summarize(history)
            if method=='positive_quadratic':
                assert math.isclose(summary['final_top1'],receipt['final_top1'],abs_tol=1e-12)
            else:
                assert math.isclose(summary['best_top1'],receipt['metrics']['top1'],abs_tol=1e-12)
            selected=read(p/'best_cal_metrics.json')
            assert math.isclose(summary['best_top1'],selected['top1'],abs_tol=1e-12)
            assert summary['best_epoch_1based']==selected['selected_epoch']+1
            u[method]=dict(summary,receipt_sha256=sha(p/'COMPLETE.json'),history_sha256=sha(p/'history.json'),gpu=receipt['gpu'],seconds=receipt['seconds'])
            data[method]=history
            if method!='positive_quadratic':old_seconds.append(receipt['seconds'])
        units[key]=dict(anchor=anchor,seed=seed,job=r['job'],seconds=r['seconds'],new_attempt=attempt,methods=u,
                        optimizer_steps=168720,fit_examples_per_epoch=44992,initial_gpu=preflight['gpu'],cpu_capability=preflight['cpu_capability'])
        new_seconds.append(r['seconds'])
    for seed in SEEDS:
        # Input order, source and initialization also match across the CE/DKD anchors.
        assert identities[f'ce_{seed}']==identities[f'dkd_{seed}']
    means={};contrasts={}
    for anchor in ANCHORS:
        means[anchor]={m:{k:S.mean(units[f'{anchor}_{s}']['methods'][m][k] for s in SEEDS) for k in ('final_top1','last10_top1','best_top1','final_nll','final_ece')} for m in METHODS}
        contrasts[anchor]={}
        for name,a,b in [('positive_minus_cosine','positive_quadratic','cosine'),('full_minus_positive','vcs','positive_quadratic'),('full_minus_cosine','vcs','cosine')]:
            summary={}
            for metric in ('final_top1','last10_top1','best_top1'):
                per={str(s):100*(units[f'{anchor}_{s}']['methods'][a][metric]-units[f'{anchor}_{s}']['methods'][b][metric]) for s in SEEDS}
                summary[metric]=dict(per_seed_pp=per,mean_pp=S.mean(per.values()),sd_pp=S.stdev(per.values()),positive_seeds=sum(x>0 for x in per.values()),negative_seeds=sum(x<0 for x in per.values()))
            d=summary['final_top1'];mean=d['mean_pp'];late=summary['last10_top1']['mean_pp']
            summary['registered_directional_signal']=abs(mean)>=.2 and ((mean>0 and d['positive_seeds']>=2 and late>0) or (mean<0 and d['negative_seeds']>=2 and late<0))
            summary['curve_mean_pp']={str(e):100*S.mean(units[f'{anchor}_{s}']['methods'][a]['curve'][str(e)]-units[f'{anchor}_{s}']['methods'][b]['curve'][str(e)] for s in SEEDS) for e in (40,80,120,150,180,210,240)}
            contrasts[anchor][name]=summary
        for s in SEEDS:
            c=contrasts[anchor];assert math.isclose(c['positive_minus_cosine']['final_top1']['per_seed_pp'][str(s)]+c['full_minus_positive']['final_top1']['per_seed_pp'][str(s)],c['full_minus_cosine']['final_top1']['per_seed_pp'][str(s)],abs_tol=1e-12)
    report=dict(status='COMPLETE_REVIEW_WITH_DRIVER_PROVENANCE_LIMITATION',utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=fh,
        analyzer_sha256=sha(Path(__file__)),frozen_files_verified=len(freeze['files']),new_complete=6,reused_controls=12,
        units=units,method_means=means,paired_contrasts_pp=contrasts,
        matching='Exact initial state, teacher, source/split, environment/software/precision, same GPU model, every epoch input order/n_seen/LR, all scientific config except loss method; no in-flight changes.',
        cost=dict(new_receipt_gpu_hours=sum(new_seconds)/3600,shared_gate_seconds=sum(g['seconds'] for g in gates),gate_receipts=gates,
            reused_control_receipt_gpu_hours=sum(old_seconds)/3600,old_failed_import_gate_scheduler_seconds=5,
            prior_initialization_cpu_failures='1063545/1063547 retained; scheduler elapsed unavailable, not zero.',
            note='New six each one COMPLETE attempt, gates shared once. Reused controls and old failures separate; receipt excludes unknown scheduler/startup overhead.'),
        limitations=['All FIT/CAL development evidence; CAL repeatedly used, three seeds, no TEST or confirmatory claim.',
            'Protocol asks driver recording; frozen runner environment/receipts omit driver and host, so exact driver matching cannot be independently verified. GPU model/software/precision match, not bitwise run identity.',
            'Fixedlambda10 mechanism comparison, not optimally tuned positive-only loss; it is not full VCS-QMI.',
            'Differences telescope numerically but optimization paths interact. Do not describe independent additive causal contributions or universal harmful Q.',
            'Dependence fields are final FIT batch; gradients first FIT batch per epoch. Sparse probes do not characterize full training distributions.',
            'BestCAL checkpoint secondary; primary fixedepoch240 and last10 separate.'])
    write(HERE/'ANALYSIS_KD_POSITIVE_ALL6_20261025.json',report)
    with (HERE/'KD_POSITIVE_ALL18_CURVES_20261025.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['KD正对二次机制控制6/6完整分析；另复用12个cosine/fullVCS对照。',
           '六项均240轮、168720更新、lambda10；逐epoch输入顺序/LR、教师/初始化/数据划分、GPU型号及软件精度均匹配。',
           '主指标固定240轮CAL top1；以下为百分比及百分点。']
    for anchor in ANCHORS:
        lines.append(anchor.upper())
        for m,v in means[anchor].items():lines.append(f'  {m}: final={100*v["final_top1"]:.4f}%; last10={100*v["last10_top1"]:.4f}%; best={100*v["best_top1"]:.4f}%.')
        for n,v in contrasts[anchor].items():
            d=v['final_top1'];lines.append(f'  {n}: final {d["mean_pp"]:+.4f}pp, perseed {d["per_seed_pp"]}; last10 {v["last10_top1"]["mean_pp"]:+.4f}pp; best {v["best_top1"]["mean_pp"]:+.4f}pp; frozen directional signal={v["registered_directional_signal"]}.')
    lines += ['驱动溯源限制：协议要求记录，但执行回执/环境没有driver/host字段，不能独立验证驱动完全匹配；不把同型GPU等同逐位可复现。',
              '分项差值可相加为总差，但不是独立可加的因果贡献；不同训练路径交互，不能称Q普遍有害。正对二次是机制对照而非完整VCS。',
              '成本：'+json.dumps(report['cost'],ensure_ascii=False)]
    (HERE/'ANALYSIS_KD_POSITIVE_ALL6_20261025.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],means=means,contrasts=contrasts,cost=report['cost']),ensure_ascii=False))


if __name__=='__main__':main()
