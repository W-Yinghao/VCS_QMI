"""Read-only audit of complete NCR posterior experiments and their existing DEV reference."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
SEEDS=(901,902,903)
ARMS=('NCR_A','vcs','bce')
RECALLS=('i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10')


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def near(a,b): assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9),(a,b)


def audit():
    automatic=read(BASE/'handoff/r22_monitor_codex_20261023/ANALYSIS.json')['ncr']
    assert automatic['complete']==6
    units={a:{} for a in ARMS}; signatures={}; dev_signature=None; recorded_attempts=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=(BASE/f'outputs_v9_n2p_20261020/V9N2P_f30k_r04_s{seed}_NCR_A' if arm=='NCR_A' else
                    BASE/f'outputs_v9_r22/ncr_vcs/R22B_f30k_r04_s{seed}_NCR_A_R22_{arm.upper()}_posterior')
            result=read(folder/'RESULT.json'); info=read(folder/'unit.json'); fwd=read(folder/'dev/FORWARD_SUMMARY.json')
            frozen=read(folder/'adapter/B0_FROZEN.json')
            assert result['seed']==seed and result['schedule']=='full' and result['rho']==.4
            assert result['candidates_expected']==result['candidates_scored']==45 and not result['missing_candidates']
            assert info['schedule_values']==dict(warmup_epoch=5,num_epochs=40,lr_update=30) and info['lr_multiplier']==1
            assert sha(folder/'adapter/B0_FROZEN.json')==info['frozen_config_sha256']
            for rel,digest in info['effective_code_sha256'].items(): assert sha(folder/'code/ncr'/rel)==digest,rel
            em=info['export_manifest']
            assert sha(folder/'v9_export/v9_observed_t2i.npy')==em['observed_sha256']
            assert sha(folder/'v9_export/v9_anchor_mask.npy')==em['anchor_sha256']
            signature=(em['observed_sha256'],em['anchor_sha256'],info['frozen_config_sha256'])
            if seed in signatures: assert signatures[seed]==signature
            signatures[seed]=signature
            for trace in result['read_trace'].values():
                assert not trace['denied'] and not trace['test_or_eval_only_opened']
                assert sha(trace['trace'])==trace['trace_sha256']
            dsig=(fwd['image_order_sha256'],fwd['caption_order_sha256'],
                  sorted((Path(x['path']).name,x['sha256']) for x in fwd['data_reads']))
            if dev_signature is not None: assert dsig==dev_signature
            dev_signature=dsig
            assert fwd['dev_identity']['n_images']==1014 and fwd['dev_identity']['n_captions']==5070
            assert not fwd['missing'] and len(fwd['candidates'])==45
            for r in fwd['ids'].values(): assert sha(r['path'])==r['sha256']
            cmap=frozen['candidate_maps']['ncr']['full']; trajectory=result['trajectory']
            assert [x['candidate_id'] for x in trajectory]==[x['candidate_id'] for x in cmap]
            assert {x['candidate_id'] for x in rows(folder/'candidates/candidates.jsonl')}=={x['candidate_id'] for x in cmap}
            for c,t in zip(cmap,trajectory):
                r=read(folder/'dev/full_dev'/f"{c['phase']}_{c['epoch']:02d}.json")
                assert r['status']=='SCORE_METRICS_VERIFIED_ONLY' and r['candidate_id']==c['candidate_id']
                assert r['split']=='dev' and r['test_read_by_this_program'] is False
                assert r['n_images']==1014 and r['n_captions']==5070
                near(sum(r['metrics'][k] for k in RECALLS),r['metrics']['rsum'])
                near(r['metrics']['rsum'],t['fullDEV_rsum'])
            best=max(trajectory,key=lambda x:(x['fullDEV_rsum'],-[y['candidate_id'] for y in cmap].index(x['candidate_id'])))
            selected=result['primary_fullDEV_best']; assert selected['candidate_id']==best['candidate_id']
            near(selected['fullDEV']['rsum'],best['fullDEV_rsum'])
            assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
            main=[x['fullDEV_rsum'] for x in trajectory if x['candidate_id'].startswith('main:')]
            assert len(main)==40;near(main[-1],result['last']['fullDEV']['rsum']);near(st.mean(main[-5:]),result['mean_last5_rsum'])
            for stage in ('train','forward','score'): assert read(folder/f'{stage}_receipt.json')['rc']==0
            train_receipt=read(folder/'train_receipt.json')
            assert train_receipt['gpu']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
            item=dict(best=selected['fullDEV']['rsum'],selected_candidate=selected['candidate_id'],selected_recalls=selected['fullDEV'],
                last=main[-1],last5=st.mean(main[-5:]),last10=st.mean(main[-10:]),trajectory=trajectory,
                stage_means={f'{lo+1}-{hi}':st.mean(main[lo:hi]) for lo,hi in [(0,10),(10,20),(20,30),(30,40)]},
                source=str(folder/'RESULT.json'),source_sha256=sha(folder/'RESULT.json'),job=train_receipt['slurm_job'],
                costs=result['costs'],gpu=train_receipt['gpu'],host=train_receipt['host'],environment=fwd['env'],
                selected_checkpoint_sha256=selected['checkpoint_sha256'],shared_input_hashes=signature,
                checkpoint_note='Selected checkpoint rehashed; all45 scored rows/metadata audited, not all45 large checkpoint bytes.')
            if arm!='NCR_A':
                rr=read(folder/'R22_RESULT.json');assert rr['selected']['fullDEV']==selected['fullDEV']
                role=info['r22_roles_manifest'];assert role['clean_labels_read'] is False
                assert sha(role['output']['file'])==role['output']['sha256']
                ledger=rows(folder/'official_out/r22/posterior.jsonl')
                assert [x['epoch'] for x in ledger]==list(range(40)) and all(x['arm']==arm for x in ledger)
                posts=[x[m] for x in ledger for m in ('A','B')]
                assert len(posts)==rr['posterior']['n_epoch_models']==80
                assert sum(not x['deployed'] for x in posts)==rr['posterior']['n_fallbacks']==0
                for x in posts:
                    assert x['config']==posts[0]['config'] and x['kind']==arm
                    assert x['rho']['gap']>=.01 and 0<x['rho']['raw']<1
                roles_trace=rr['posterior']['roles_trace']
                assert not roles_trace['denied'] and not roles_trace['test_or_eval_only_opened']
                assert sha(roles_trace['trace'])==roles_trace['trace_sha256']
                item['posterior']=dict(epoch_models=80,fallbacks=0,configuration=posts[0]['config'],
                    rho_mean=st.mean(x['rho']['raw'] for x in posts),rho_min=min(x['rho']['raw'] for x in posts),rho_max=max(x['rho']['raw'] for x in posts),
                    split_agreement_mean=st.mean(x['agreement']['split_agree_frac'] for x in posts),
                    split_agreement_min=min(x['agreement']['split_agree_frac'] for x in posts),
                    posterior_total_seconds=sum(x['seconds'] for x in ledger),
                    fitting_seconds=sum(x['fit_seconds'] for x in posts),q_scoring_seconds=sum(x['q_score_seconds'] for x in posts),
                    ledger_sha256=sha(folder/'official_out/r22/posterior.jsonl'))
                item['archived_attempts']=[str(p) for p in (folder/'attempts').glob('attempt_*')]
                recorded_attempts.extend(item['archived_attempts'])
                auto=next(x for x in automatic['units'] if x['seed']==seed and x['arm']==arm)
                near(item['best'],auto['rsum'])
                assert info['effective_code_sha256']==read(BASE/f'outputs_v9_r22/ncr_vcs/R22B_f30k_r04_s901_NCR_A_R22_VCS_posterior/unit.json')['effective_code_sha256']
            units[arm][str(seed)]=item
    paired={}
    for a,b in [('vcs','NCR_A'),('vcs','bce'),('bce','NCR_A')]:
        row={}
        for metric in ('best','last','last5','last10'):
            differences=[units[a][str(s)][metric]-units[b][str(s)][metric] for s in SEEDS]
            row[metric]=dict(differences=differences,mean=st.mean(differences),positive_seeds=sum(x>0 for x in differences))
        paired[a+'-'+b]=row
    assert len(recorded_attempts)==1 and 's903' in recorded_attempts[0]
    failed=read(BASE/'handoff/r22_monitor_codex_20261023/NCR_ORIGINAL_TIMEOUT_RECEIPT.json')
    assert failed['job']=='1056273' and 'JobState=TIMEOUT' in failed['scontrol']
    hh,mm,ss=map(int,re.search(r'RunTime=(\d+):(\d+):(\d+)',failed['scontrol']).groups());failed_seconds=3600*hh+60*mm+ss
    successful=sum(r['costs'][stage]['wall_seconds'] for a in ('vcs','bce') for r in units[a].values() for stage in ('train','forward','score'))
    baseline=sum(r['costs'][stage]['wall_seconds'] for r in units['NCR_A'].values() for stage in ('train','forward','score'))
    return dict(utc=datetime.now(timezone.utc).isoformat(),status='AUDIT_PASS',series='original_NCR',units=units,
        means={a:{k:st.mean(r[k] for r in units[a].values()) for k in ('best','last','last5','last10')} for a in ARMS},
        paired=paired,cost=dict(successful_six_stage_receipts_gpu_hours=successful/3600,
          failed_original_scheduler_gpu_hours=failed_seconds/3600,known_six_plus_failed_gpu_hours=(successful+failed_seconds)/3600,
          reused_baseline_stage_receipts_gpu_hours=baseline/3600,failed_attempts=recorded_attempts,
          scope='Successful train/forward/score stages run in GPU allocations, plus original TIMEOUT scheduler allocation. Not all startup/export/gates costs are measured; baseline separately reported.'),
        audit=dict(all9_units_45_candidates=True,selected_checkpoint_hashes=True,all_scored_recall_sums=True,
          source_and_input_export_hashes=True,read_trace_hashes=True,roles_and_posterior_ledger=True,existing_paired_summary_agrees=True),
        limitations=['DEV best-of45 selection; not independent confirmation or TEST.',
          'Matched seed, exported input/roles and schedule, but no recorded initial-tensor or every-minibatch hash. Co-teaching splits intentionally diverge after posterior changes.',
          'All RTX6000PRO; jobs/nodes differ, reused baseline and cudnn.benchmark are not bitwise reproducible.',
          'Reference Q is same-role anchor re-pairing excluding own image, not the exact unconditional full product.',
          'Sparse scalar posterior summaries do not prove clean-pair calibration, hard-positive retention, or gradient causality.',
          'Failed original903 run retained and charged; retry unchanged recipe, no extra retries.'])


if __name__=='__main__':
    r=audit();p=HERE/'ANALYSIS_NCR_ALL3_20261024.json';p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','means','paired','cost')},ensure_ascii=False))
