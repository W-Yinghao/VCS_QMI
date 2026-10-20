"""BF1_CAPACITY_EXT_K8_K32 unit runner (one Slurm GPU job = one unit).

Runs the frozen, accepted BF1 pipeline (handoff/v10_bf1_20261017/source = b8f59ea: FIT 3000 steps -> 12 checkpoints with full CAL
-> CAL lexicographic checkpoint choice -> DEV32/DEV24 at that checkpoint) for one family / seed with the MF head width K = 8 or 32
and the family's inherited K16 CAL selection.  Exactly two things differ from BF1 and both are patched here, never in the frozen tree:
  1. run.config(ident): BF1 hard-codes K=16; here K comes from the unit identity (verified: the TrainConfig equals BF1's with K
     replaced, field by field).
  2. run.load_checkpoint: deserialised on CPU (map_location='cpu'; the only token changed) so module/optimizer load_state_dict
     place tensors by their own policy and AdamW keeps its scalar step on CPU, as in an uninterrupted run (review 2026-10-19).
Around BF1 run_unit: admission against the 30 L40S GPU-h ceiling, the Slurm-StartTime cost clock, step-log truncation before a
resume, a separate identity/output root, and the complete-unit audit.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import capacity_common as C


def import_bf1():
    """Load the frozen BF1 run.py with its own common.py (both named as in BF1) from the frozen tree only."""
    scripts = C.BF1_SOURCE / 'scripts' / 'v10_bf1'
    sys.path.insert(0, str(C.BF1_SOURCE / 'src'))
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('common', scripts / 'common.py')
    common = importlib.util.module_from_spec(spec)
    sys.modules['common'] = common
    spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location('bf1_run', scripts / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    sys.modules['bf1_run'] = runner
    spec.loader.exec_module(runner)
    C.require(Path(runner.A.__file__).resolve() == (C.BF1_SOURCE / 'src' / 'v10' / 'adapter.py').resolve()
              and Path(runner.A.MF.__file__).resolve() == (C.BF1_SOURCE / 'src' / 'v10' / 'mf_vcs.py').resolve(),
              'Scientific modules were not imported from the frozen BF1 tree')
    return runner


def capacity_config_factory(bf1):
    A = bf1.A

    def config(ident):
        return A.TrainConfig(method=ident['method'], seed=ident['seed'], encoder_lr=ident['lr'], head_lr=ident['lr'],
                             K=ident['K'], n_sc=16, n_pos=128, crop=160, m1=A.M1Config(max_iterations=ident['cap'] or 30))
    return config


def require_only_k_differs(bf1, config, ident):
    ours, theirs = dataclasses.asdict(config(ident)), dataclasses.asdict(bf1.config(ident))
    C.require(theirs['K'] == 16 and ours['K'] == ident['K'] and ours == {**theirs, 'K': ident['K']},
              'Capacity TrainConfig differs from BF1 config in more than K')
    return ours


def cpu_first_loader(bf1):
    """BF1 run.load_checkpoint (b8f59ea) with map_location=t.device -> map_location='cpu'; check_control/smoke_k compare the source."""
    import torch
    A, read, sha = bf1.A, bf1.read, bf1.sha

    def load_checkpoint(t,path,ident,restore_rng=True):
        path=Path(path);meta=read(path.with_suffix('.json'))
        if meta['identity']!=ident or sha(path)!=meta['sha256']:
            raise RuntimeError('Checkpoint identity or bytes mismatch: '+str(path))
        st=torch.load(path,map_location='cpu',weights_only=True)
        if st['identity']!=ident or st['config']!=dataclasses.asdict(t.cfg):
            raise RuntimeError('Checkpoint configuration mismatch')
        for m,s in zip(t.modules(),st['modules'],strict=True):m.load_state_dict(s,strict=True)
        t.opt.load_state_dict(st['optimizer']);t.n_steps=st['step'];t.sampler.step=st['sampler_step']
        t.sampler.g.bit_generator.state=st['sampler_state'];t.g_extra.bit_generator.state=st['extra_rng']
        if restore_rng:
            torch.set_rng_state(st['torch_rng'].cpu())
            if t.device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda_rng']])
        if t.n_steps!=t.sampler.step or A.state_hash(*t.modules())!=st['model_state_hash']:
            raise RuntimeError('Restored checkpoint state mismatch')
        return meta
    return load_checkpoint


def capacity_identity(bf1, unit, prereg, capacity_freeze_sha):
    selection, dependency = bf1.checked_selection(unit['family'])
    f = prereg['families'][unit['family']]
    C.require(dependency == f['selection']['sha256'] and selection['selected']['candidate'] == f['candidate'],
              'Inherited K16 selection differs from the pre-registration')
    cd = selection['selected']['candidate']
    ident = bf1.identity(unit['family'], unit['seed'], cd['lr'], cd['cap'], 'FINAL_DEV', dependency=dependency)
    C.require(ident['schema'] == 'v10.bf1.unit.v1' and ident['K'] == 16 and ident['method'] == unit['method'], 'BF1 identity changed')
    ident.update(schema=C.SCHEMA_UNIT, K=unit['K'], extension=C.NAME, preregistration_sha256=C.sha(C.BUNDLE / 'PREREGISTRATION.json'),
                 capacity_freeze_sha256=capacity_freeze_sha, result_scope='CAPACITY_EXTENSION_SEPARATE_TABLE')
    C.require(ident == C.expected_identity(unit, prereg, capacity_freeze_sha), 'Identity differs from the pre-registration reconstruction')
    return ident


def git_head(path):
    r = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def main(key):
    C.require(os.environ.get('SLURM_JOB_ID'), 'Scientific execution requires Slurm')
    job = os.environ['SLURM_JOB_ID']
    freeze_sha = os.environ.get('V10_CAPACITY_FREEZE')
    prereg = C.validate_prereg(C.frozen(freeze_sha))
    C.checks_passed(freeze_sha)
    unit = C.unit_by_key(prereg, key)
    C.require(os.environ.get('SLURM_JOB_NAME') == unit['job_name'], 'Job name must be ' + unit['job_name'])
    C.require(os.environ.get('SLURM_JOB_PARTITION') == 'L40S', 'Pre-registered hardware is the L40S partition')
    limit = C.slurm_seconds(os.environ.get('V10K_TIME_LIMIT_RAW', ''))
    C.require(limit == unit['time_limit_seconds'], f"Slurm time limit {limit}s != pre-registered {unit['time_limit_seconds']}s")
    allocation = C.Allocation(unit, job, C.Clock.from_env())
    state = 'FAIL'
    try:
        directory = C.OUT / unit['key']
        done = (directory / 'COMPLETE.json').exists()
        if done:
            C.verify_result(unit, prereg, freeze_sha, require_receipt=False)
        admission = C.admit(unit, job, limit + C.RESERVE_MARGIN_SECONDS, already_complete=done)
        if admission['decision'] == 'ALREADY_COMPLETE':
            state = 'ALREADY_COMPLETE'
            print('ALREADY_COMPLETE', key, flush=True)
            return 0
        if admission['decision'] != 'ADMITTED':
            raise C.Refused(admission['decision'], f"committed={admission['committed_other_seconds']:.0f}s "
                            f"reserve={admission['reserved_seconds']}s ceiling={C.CEILING_SECONDS}s")
        os.environ['V10_BF1_EXPECT_FREEZE'] = prereg['bf1_basis']['freeze']['sha256']
        bf1 = import_bf1()
        bf1.freeze()
        bf1.require_gpu_gate()
        C.require(bf1.A.code_hash() == C.CORE_HASH and bf1.A.host_code_hash() == C.HOST_HASH, 'Accepted BF core/host changed')
        bf1.A.check_job_pin()
        env = bf1.runtime_check('cuda')
        C.require(env['git_commit'] == C.BF1_COMMIT and env['git_dirty_code'] == '' and env['code_hash_src_scripts_v10'] == C.CORE_HASH
                  and env['host'] not in C.ALLOWED_NODES_EXCLUDED, 'Frozen BF1 tree or node is not the pre-registered one')
        ident = capacity_identity(bf1, unit, prereg, freeze_sha)
        config = capacity_config_factory(bf1)
        resolved = require_only_k_differs(bf1, config, ident)
        bf1.config = config
        bf1.load_checkpoint = cpu_first_loader(bf1)
        C.require(not (directory / 'IDENTITY.json').exists() or C.read(directory / 'IDENTITY.json') == ident,
                  'Refusing to touch a unit directory that holds another identity')
        resume = C.prepare_resume(directory, job)
        if resume:
            allocation.record('RESUME_PREPARED', **resume)
        C.write(C.BUNDLE / 'runtime' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit, 'identity': ident,
                'train_config': resolved, 'environment': env, 'admission': admission, 'resume': resume,
                'capacity_commit': git_head(C.REPO), 'capacity_freeze_sha256': freeze_sha,
                'patches': ['run.config: K from identity', "run.load_checkpoint: map_location='cpu'"]})
        allocation.record('RUNTIME_VERIFIED', gpu=env['gpu_name'], node=env['host'], resume=resume)
        print('RUNTIME_VERIFIED', key, env['gpu_name'], env['host'], 'K=' + str(unit['K']), flush=True)
        bf1.run_unit(ident, directory, allocation, env)
        C.verify_result(unit, prereg, freeze_sha, require_receipt=False)
        state = 'COMPLETE'
        print('UNIT_COMPLETE', key, flush=True)
        return 0
    except C.Refused as refusal:
        state = refusal.code
        C.write(C.BUNDLE / 'refusals' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': key, 'code': refusal.code,
                                                        'detail': refusal.detail})
        print('REFUSED', key, refusal, flush=True)
        return C.EXIT_REFUSED
    except BaseException as error:
        state = 'INTERRUPTED' if isinstance(error, InterruptedError) else 'FAIL'
        C.write(C.BUNDLE / 'failures' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': key, 'error': repr(error),
                                                        'traceback': traceback.format_exc()})
        raise
    finally:
        allocation.record('END', status=state)


def interrupted(signum, frame):
    raise InterruptedError(f'Slurm signal {signum}; resume from the last verified checkpoint if resubmitted')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    raise SystemExit(main(sys.argv[1]))
