"""One-time stdlib freeze; no torch import, training or evaluation."""
from common import BASE,HERE,OLD,OUT,ARMS,SEEDS,read,write,sha

def main():
    assert not (HERE/'FREEZE.json').exists(), 'Never rebuild a frozen batch'
    long = BASE/'handoff/r22_geometry_long_v2_20261024'
    audit = BASE/'handoff/r22_next_round_20261025/A1_COVERAGE_AUDIT_20261025.json'
    assert read(audit)['missing_a1_units'] == 12
    locked = {}; nodes = {}
    for ds in ('bf','epfl'):
        for arm in ARMS:
            for seed in SEEDS:
                key = f'{ds}_{arm}_{seed}'; directory = long/'runs'/key; p = directory/'COMPLETE.json'; done = read(p)
                assert done['status'] == 'COMPLETE' and done['final_steps'] == 30000 and done['new_updates'] == 27000
                assert not (OLD/'runs'/('a1_'+key)).exists()
                env = done['environment']; group = ds+'_'+env['host']
                assert env['host'] in ('node50','node57','node59')
                assert set(done['roles_read']) <= {'FIT','CAL','DEV'}
                paths = [p,directory/'TRAIN.jsonl',directory/'checkpoints/step_30000.pt',
                         directory/'dev_30000/DEV.jsonl',directory/'dev_30000/DEV_SUMMARY.json']
                for step in (10000,20000,30000):
                    paths += [directory/f'cal_{step}/CAL.jsonl',directory/f'cal_{step}/CAL_SUMMARY.json']
                for kind in ('checkpoint','metadata','completion','source_train'):
                    q = done['source'][kind]; assert sha(q) == done['source'][kind+'_sha256']; paths.append(q)
                assert sha(directory/'TRAIN.jsonl') == done['train_sha256']
                assert sha(directory/'checkpoints/step_30000.pt') == done['final_checkpoint_sha256']
                locked[key] = dict(ds=ds,arm=arm,seed=seed,a0_completion=str(p),environment=env,gate_group=group,
                                   a0_read_evidence='Completion roles_read; original A0 runner did not export a separate INPUT_TRACE.json',
                                   files={str(q):sha(q) for q in paths})
                nodes[group] = dict(ds=ds,host=env['host'],partition=env['slurm_partition'],driver=env['precision']['nvidia_smi_name_driver'])
    assert len(locked) == 12 and len(nodes) == 5
    write(HERE/'SOURCE_LOCK.json',dict(units=locked,gate_groups=nodes,audit=str(audit),audit_sha256=sha(audit)))
    protocol = dict(status='FROZEN_BEFORE_NEW_RESULTS',theory='Original Variational Cauchy Schwarz divergence unchanged',
        hypothesis='Known-transform native FIT resampling changes fixed-budget registration task performance and objective gaps; missing-seed completion tests whether the observed pilot behavior persists.',
        scope='Twelve new A1 continuations: BF/EPFL x exact-VCS/BCE/NCE x seeds1002/1003. Reuse twelve completed A0 fixed30000 controls. Retain six pilot1001 A1/A0 pairs separately; no repeated successful training.',
        matching='Same source model/AdamW/sampler/global RNG at3000 and batch metadata/LR at all27000 updates. Each new unit pinned to original A0 node, GPU, driver, Python, torch, numpy, CUDA, cuDNN, threads and full effective precision. Gate or full preflight mismatch fails closed.',
        science='Reuse original geometry_a1.step, GeometryMatchedBatches and native warp/preprocessing/pullback unchanged. 50% of16 scenes requested; angle30deg, translationdisk51px,16support tries,margins2/1; independent geometry RNG and unchanged crop/positions/permutation. Invalid geometry retains original crop and is logged. Common unadapted3000 prefix.',
        budget=dict(final_steps=30000,new_updates_per_unit=27000,lr='3e-4 through15000,3e-5 through24000,3e-6 through30000',
                    cal_steps=[10000,20000,30000],dev_steps=[30000],expected_gpu_hours=[9,12],historical_estimate_gpu_hours=2*read(BASE/'handoff/r22_watch10m_20261024/ANALYSIS_A1_ALL6_20261025.json')['cost']['new_receipt_gpu_hours'],
                    full_jobs=12,full_hard_hours_each=8,gpu_gates=5,gate_hard_hours_each=1,cpu_hard_minutes=30,output_max_gib=4,
                    accounting='Count all new attempts, five gates once, CPU separately, old A0 and shared prefixes separately; unknown scheduler costs not zero.'),
        gates='CPU all12 real restore/firstfive matched baseline batches/LR boundaries/privategeometry RNG/support/identitypullback; five dataset-node GPU gates cover all12 actualsourcecells, three real updates each, nativepath equivalence, actual modelchange, optimizer+RNG roundtrip, CAL posegradient, FIT/CAL guardtrace and exactA0 environment. Allfive pass before anyfull admission.',
        endpoints=dict(primary='Fixed30000 COS DEV strict/loose success and percore/scene-equal tail. A1-A0 paired differences by dataset and target; VCS-minus-BCE/NCE interaction. No dataset pooling or checkpoint selection.',
            secondary='CAL10000/20000/30000 curves, median and tail error, active/requested/fallback counts, perseed input consistency, walltime and allocated GPU cost.',
            descriptive_signal='On the new matched1002/1003 subset, both seeds improve and mean strict>=1pp, or both loose improve and mean loose>=1pp without either strict dropping>1pp. This is a descriptive screen, not significance or permanent stopping. For objective-specificity also report VCS-control levels and adaptation interactions; a control decline alone is not a VCS gain.',
            history='Show allthree-seed descriptive table and the matchednewtwo-seed table separately; pilot1001 fiveof6 driver-mismatched pairs cannot be used as clean adaptation evidence. DEV reuse and nondeterministic cross-run pose search remain, perturbations are not independent seeds.'),
        stopping='All12 full units complete fixedbudget regardless interim efficacy. Stop only a technically invalid unit, retain attempts, no automatic retry, no changed lambda/steps/endpoint/driver pins. Technical repair needs new freeze.',
        operations=dict(pool='Existing1062146 only',watch='Existing1062667 only',no_test=True,no_public_push=True,no_delete=True,no_subagents=True,
                        output=str(OUT),home_limit_gib=1450,home_cache_max_seconds=7200,projects_floor_bytes=300_000_000_000,additional_output_reserve_gib=4))
    write(HERE/'PROTOCOL.json',protocol)
    jobs = {'gates': [], 'full': []}
    def job(name, partition, node, hours, command, completion, group):
        return dict(name=name,args=['sbatch','--partition='+partition,'--nodelist='+node,'--time='+hours,'--job-name='+name,
                    str(HERE/'gpu.sbatch'),*command],completion=str(completion),group=group,freeze=str(HERE/'FREEZE.json'))
    for group, rec in sorted(nodes.items()):
        jobs['gates'].append(job('r22_a1c_gate_'+group,rec['partition'],rec['host'],'01:00:00',['gate','--group',group],
                                OUT/'gates'/group/'COMPLETE.json','other_a1_complete_gate'))
    for key, rec in locked.items():
        jobs['full'].append(job('r22_a1c_'+key,rec['environment']['slurm_partition'],rec['environment']['host'],'08:00:00',
                               ['run','--key',key],OUT/'runs'/key/'COMPLETE.json','other_a1_complete_full'))
    write(HERE/'JOBS_SPEC.json',jobs)
    files = {}
    for rel, digest in read(OLD/'FREEZE.json')['files'].items():
        p = BASE/rel; assert sha(p) == digest, p; files[str(p)] = digest
    for p in [OLD/'FREEZE.json',audit,BASE/'handoff/r22_watch10m_20261024/ANALYSIS_A1_ALL6_20261025.json',
              BASE/'handoff/r22_submission_coord_20261024/coord.py',*HERE.glob('*.py'),*HERE.glob('*.sbatch'),*HERE.glob('*.json')]:
        assert p.name != 'FREEZE.json' or p.parent != HERE
        files[str(p)] = sha(p)
    write(HERE/'FREEZE.json',dict(schema='R22_A1_COMPLETION_V1',files=dict(sorted(files.items()))))
    print('FROZEN',sha(HERE/'FREEZE.json'),len(files),'files,12full,5gates')

if __name__ == '__main__': main()
