"""Freeze CR6 (after acceptance ALL_PASS). Writes PROTOCOL_FROZEN_CR6.json (chmod 444), UNIT_REGISTRY, DEDUP_CR6.json.
Refuses to overwrite an existing frozen protocol. Package configs are mapped 1:1 (30 new + 18 references)."""
from __future__ import annotations
import csv, glob, hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402
import cr6_source as SRC                                                     # noqa: E402


ACC_RULE = ('see ADDENDUM revision_2 (single RTX acceptance 1090989; release A1/B RTX6000PRO|L40S, A2 RTX6000PRO|L40S|V100-32GB). Original: ADDENDUM (registered before any formal training, coordinator request 2026-11-06): acceptances per GPU type. L40S job 1090959 = history mode '
            '(same-job checks + bitwise vs CR1/CR5 L40S history); RTX6000PRO = portable mode (same-job reruns of the frozen CR1 estimator, CR5 student and CR1 '
            'eval bridge instead of history + all TAIL checks); V100-32GB = portable --no-tail (A1/A2 only; TAIL never on V100). Each matched group is one job on '
            'one GPU, submitted only to accepted types valid for its kind; CR5 FROZEN references are reused only if the 250-step replay in the group job is '
            'bitwise equal to the stored run (i.e. same numeric path as the L40S history), otherwise the matched small-G reference is retrained in that group job.')


def fp(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    accs = sys.argv[1:]; As = [K.read_json(x) for x in accs]
    if not any(A['status'] == 'ALL_PASS' for A in As): raise SystemExit('no acceptance ALL_PASS')
    acc = next(x for x, A in zip(accs, As) if A['status'] == 'ALL_PASS'); A = K.read_json(acc)
    if K.FROZEN.exists(): raise SystemExit('PROTOCOL_FROZEN_CR6.json exists; register an amendment instead')
    pkg = K.TREE/'package'; cr5 = K.read_json(K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json')
    rows = list(csv.DictReader(open(pkg/'configs/MATRIX_NEW.csv'))); refs = list(csv.DictReader(open(pkg/'configs/REFERENCES.csv')))
    assert len(rows) == 30 and len(refs) == 18
    units, references = {}, {}
    for r in refs:
        c, rel, s = r['condition'], r['relation'], int(r['head_seed'])
        orig = f'CR5_{c}_{rel}_COMPLETE_PC_s{s}'; cu = cr5['units'][orig]
        assert cu['head_seed'] == s and cu['sampler_seed'] == int(r['sampler_seed']) and cu['relation'] == rel
        res = Path(cu['result_dir']); uc = K.read_json(res/'UNIT_COMPLETE.json')
        od = Path(uc['reused_dir']) if uc.get('status') == 'REUSE_SAME_JOB_RECT' else res
        cfgp = pkg/'configs/units'/f"{r['unit_id']}.json"
        groups = ([f'A2_Flickr60_t{s}'] if c == 'Flickr60' else []) + ([f'B_{c}_t1002'] if s == 1002 else [])
        references[r['unit_id']] = dict(condition=c, relation=rel, head_seed=s, sampler_seed=int(r['sampler_seed']), scope='FROZEN', consumer='COMPLETE_PC',
                                        graph_path=cu['graph_path'], graph_sha256=K.sha256(cu['graph_path']), original_unit=orig, original_dir=str(od),
                                        original_ckpt_sha256=K.read_json(od/'DONE.json')['checkpoint_sha256'], single_positive_identity=uc.get('status') == 'REUSE_SAME_JOB_RECT',
                                        groups=groups, package_config_sha256=K.sha256(cfgp), plan='REUSE_IF_SAME_GPU_250_REPLAY_BITWISE_AND_ENDPOINT_IDENTITY_ELSE_RERUN_IN_GROUP')
    for r in rows:
        u = r['unit_id']; cfgp = pkg/'configs/units'/f'{u}.json'; base = dict(stage=r['stage'], condition=r['condition'], package_config_sha256=K.sha256(cfgp), steps=8000)
        if r['stage'] == 'A1_ESTIMATOR':
            s = SRC.NEW_SEEDS[r['expert']]; assert s['head_seed'] == int(r['head_seed']) and s['sampler_seed'] == int(r['sampler_seed'])
            base.update(source_objective=r['source_objective'], expert=r['expert'], source_seed=2002, head_seed=s['head_seed'], sampler_seed=s['sampler_seed'],
                        group='A1', out=str(K.OUT/'A1'/'experts'/r['source_objective']/r['expert']), cloned_from=f"CR1_Flickr60_EST_{r['source_objective']}_{r['expert']}_s1001")
        elif r['stage'] == 'A2_SOURCE_STUDENT':
            base.update(relation=r['relation'], source_set='NEW_SRC2002', head_seed=int(r['head_seed']), sampler_seed=int(r['sampler_seed']), scope='FROZEN',
                        consumer='COMPLETE_PC', group=f"A2_Flickr60_t{r['head_seed']}", graph='A1/graphs/<relation>_src2002.npz (bound by sha in GRAPH_REGISTRY_SRC2002.json at build time)')
            assert int(r['sampler_seed']) == 20261104201 + int(r['head_seed']) - 1001
        else:
            cu = cr5['units'][f"CR5_{r['condition']}_{r['relation']}_COMPLETE_PC_s1002"]
            base.update(relation=r['relation'], source_set='ORIGINAL_CR4', head_seed=1002, sampler_seed=20261104202, adapter_seed=1002, scope='TAIL_LORA8',
                        consumer='COMPLETE_PC', graph_path=cu['graph_path'], graph_sha256=K.sha256(cu['graph_path']), baseline_ref_id=r['baseline_ref_id'],
                        group=f"B_{r['condition']}_t1002")
        units[u] = base
    # dedup: config fingerprints vs every earlier CR registry (CR1-CR5P)
    seen = {}
    for p in [K.CR1_TREE/'PROTOCOL_FROZEN_CR1.json', K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json', K.CR5P_TREE/'PROTOCOL_FROZEN_CR5_P1105.json',
              K.CR4_TREE/'PROTOCOL_FROZEN_CR4.json', K.CR3_TREE/'PROTOCOL_FROZEN_CR3.json']:
        z = K.read_json(p)
        for uid, i in (z.get('units') or {}).items():
            if isinstance(i, dict) and 'graph_path' in i and Path(i['graph_path']).exists():
                key = fp(dict(c=i.get('condition'), g=K.sha256(i['graph_path']), h=i.get('head_seed', i.get('student_seed', 1001)), s=i.get('sampler_seed', 20261104201),
                              consumer=i.get('consumer', i.get('kind')), scope='FROZEN'))
                seen.setdefault(key, []).append(f'{p.name}:{uid}')
    cr1_est = {f"{u}": 'seed1001' for u in K.read_json(K.CR1_TREE/'PROTOCOL_FROZEN_CR1.json')['unit_config_sha256'] if '_EST_' in u}
    dd = {}
    for u, i in units.items():
        if i['stage'] == 'A1_ESTIMATOR':
            dd[u] = dict(fingerprint=fp(dict(c=i['condition'], o=i['source_objective'], e=i['expert'], h=i['head_seed'], s=i['sampler_seed'])),
                         matches=[], note=f'CR1 estimators exist only with R1=1001/R2=[1001,e,51] ({len(cr1_est)} units); new seeds -> new')
        elif i['stage'] == 'B_STUDENT_SCOPE':
            dd[u] = dict(fingerprint=fp(dict(c=i['condition'], g=i['graph_sha256'], h=1002, s=20261104202, consumer='COMPLETE_PC', scope='TAIL_LORA8', a=1002)),
                         matches=[], note='no earlier CR TAIL scope; NR4/NR5/NR6 TAIL runs use other objectives/candidates/no fixed CR graph')
        else:
            dd[u] = dict(fingerprint=None, matches=[], note='graph is new (built from the new source set); cannot equal any earlier graph sha (checked after build)')
    for rid, i in references.items():
        key = fp(dict(c=i['condition'], g=i['graph_sha256'], h=i['head_seed'], s=i['sampler_seed'], consumer='COMPLETE_PC', scope='FROZEN'))
        dd[rid] = dict(fingerprint=key, matches=seen.get(key, []), mapped_to=i['original_unit'])
    K.write_json(K.TREE/'DEDUP_CR6.json', dict(units=dd, rule='full config + asset-hash fingerprint; references map to the CR5 units; 30 new units have no earlier equivalent'))
    code = {}
    for p in sorted(glob.glob(str(K.TREE/'code/site/*.py'))) + sorted(glob.glob(str(K.TREE/'sbatch/*'))) + sorted(glob.glob(str(K.TREE/'configs/graphs/*.json'))):
        code[os.path.relpath(p, K.TREE)] = K.sha256(p)
    ext = [K.CR1_TREE/'code/site/cr1_prod.py', K.CR1_TREE/'code/site/eval_student.py', *sorted((K.CR1_TREE/'code/src/cr1').glob('*.py')),
           K.CR2_TREE/'code/site/build_relation_pkg.py', K.CR2_TREE/'code/reference/cr2_relations.py',
           K.CR3_TREE/'code/pkg/reference/build_controls.py', K.CR3_TREE/'code/pkg/reference/support_complete.py',
           K.CR4_TREE/'code/pkg/reference/positive_exclusive.py', K.CR4_TREE/'code/pkg/reference/support_complete.py', K.CR4_TREE/'code/site/cr4_student.py',
           K.CR5_TREE/'code/site/cr5_student.py', K.NR4_CODE/'nr4/tail_encoder.py', K.NR4_CODE/'nr4/ref_adapters.py', K.NR4_CODE/'nr4/evaluate.py', K.NR4_CODE/'nr4/train.py',
           K.NR1_CODE/'nr1/evaluate.py', K.NR1_CODE/'nr1/io.py', K.NR1_CODE/'nr1/models.py', K.NR2_CODE/'nr2/reference.py']
    for p in ext: code[str(p)] = K.sha256(p)
    data = {}
    for c in K.CONDS:
        for p in (K.bank_path(c), Path(str(K.bank_path(c)).replace('.npz', '.binding.json')), K.prep_dir(c)/'PREP.json', K.prep_dir(c)/'graph.npz'):
            data[str(p)] = K.sha256(p)
        dv, tr = K.dev_paths(c); data[str(dv/'manifest.json')] = K.sha256(dv/'manifest.json')
        for s in ('fit', 'dev'):
            q = K.NR4_RAW/K.DATASET[c]/s
            for f in ('RAW_MANIFEST.json', 'HIDDEN_MANIFEST.json'):
                if (q/f).exists(): data[str(q/f)] = K.sha256(q/f)
    for i in list(references.values()) + [u for u in units.values() if 'graph_path' in u]: data[i['graph_path']] = i['graph_sha256']
    for o in ('J', 'BCE'):
        for e in K.EXPERTS: data[str(K.CR1/'Flickr60'/'experts'/o/e/'DONE.json')] = K.sha256(K.CR1/'Flickr60'/'experts'/o/e/'DONE.json')
    large = {}
    for c in K.CONDS:
        d = K.NR4_RAW/K.DATASET[c]
        hm = K.read_json(d/'fit'/'HIDDEN_MANIFEST.json')
        large[str(d/'fit'/'img_h10_f32.npy')] = dict(sha256_from_manifest=hm['img_h10_sha256'], bytes=os.path.getsize(d/'fit'/'img_h10_f32.npy'))
        large[str(d/'fit'/'txt_h10_f32.npy')] = dict(sha256_from_manifest=hm['txt_h10_sha256'], bytes=os.path.getsize(d/'fit'/'txt_h10_f32.npy'))
        rm = K.read_json(d/'dev'/'RAW_MANIFEST.json')
        large[str(d/'dev'/'pixels_u8.npy')] = dict(sha256_from_manifest=rm['pixels_sha256'], bytes=os.path.getsize(d/'dev'/'pixels_u8.npy'))
        large[str(d/'dev'/'tokens_i32.npy')] = dict(sha256_from_manifest=rm['tokens_sha256'], bytes=os.path.getsize(d/'dev'/'tokens_i32.npy'))
    fz = dict(protocol_id=K.PROTOCOL_ID, name_check='CR6_SOURCE_REPLICATE_AND_STUDENT_SCOPE_V1 not used by any earlier server registry (grep of handoff/*.json and outputs before registration)',
              evidence_commit=K.EVIDENCE, package=dict(zip='/home/infres/yinwang/CS_QMI/VCS_CR5_Review_CR6_Server_20261105.zip',
              zip_sha256='c4ce17b58ed103d57a4f6bd167b4042f0c4d23f9ea722722d133dee3ae13e072', manifest_sha256=K.sha256(pkg/'MANIFEST_SHA256.json'),
              run_checks='12/12 unittest OK + summarize_cr5 (CPU job 1090928, env vcs_aigiqa_ir torch 2.9.0 pandas 3.0.6, scratch copy)'),
              user_instruction='准备CR6并提交', frozen_utc=K.utc(), acceptance=dict(at_freeze=[dict(file=x, sha256=K.sha256(x), status=B['status'], mode=B.get('mode'), gpu=B.get('gpu_name'), validates=B.get('validates_groups'), job=B['job']) for x, B in zip(accs, As)],
                              release_rule=ACC_RULE, addendum=dict(file=str(K.TREE/'ADDENDUM_CR6_ACCEPTANCE_BY_GPU_TYPE.json'), sha256=K.sha256(K.TREE/'ADDENDUM_CR6_ACCEPTANCE_BY_GPU_TYPE.json'))),
              source_rng_roles=SRC.__doc__, new_source_seed_map=SRC.NEW_SEEDS, units=units, references=references,
              matrix=dict(new=30, A1=12, A2=6, B=12, references=18), endpoints=list(K.SAVES), main_endpoint=8000,
              gpu_policy=dict(release='a group is submitted only to GPU types whose CR6 acceptance (history or portable mode) passed and validates that group kind; acceptances passing after freeze are appended to ACCEPTANCE_REGISTRY.json (sha-bound) by the driver',A1='one job (12 fits + 2 score exports) on one GPU; RTX6000PRO or L40S',
                              A2='one job per student seed (3 new + 6 refs) on one GPU; RTX6000PRO, L40S or V100-32GB (S1/V1 passed on V100S in 1090990)',
                              B='one job per condition (6 refs + 6 TAIL; in-job TAIL re-check only off RTX) on one GPU; RTX6000PRO or L40S (TAIL never on V100)',
                              resubmission='an incomplete group may be resubmitted (exact resume); GROUP_ENV mismatch aborts'),
              evaluation='full DEV, base Top100 rerank by G float64 (NR1 protocol), all queries, fixed step 8000 (curves 0..8000 reported); secondary student-cosine and current-G full gallery; no TEST',
              code_sha256=code, data_sha256=data, large_data_bound_by_manifest=large, dedup='DEDUP_CR6.json', test_read=False, no_result_thresholds=True,
              not_done=['no new NCE source', 'no CC source refit', 'new-source graphs never used in B', 'no LoRA rank/depth/lr changes'])
    K.write_json(K.FROZEN, fz); os.chmod(K.FROZEN, 0o444)
    print('FROZEN', K.sha256(K.FROZEN))


if __name__ == '__main__': main()
