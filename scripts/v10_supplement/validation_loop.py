"""Single owner: CPU numerics -> CUDA/real-host gate -> seven full CAL validations.

No scientific release, held-data read, cancellation, retry, or duplicated BF1 job.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from validate import verify_manifest
from v10.supplement.contract import FAMILIES, read, sha
from v10.supplement.records import lock, write, utc


def require(ok, why):
    if not ok: raise RuntimeError(why)


def queue():
    p = subprocess.run(['squeue', '-u', 'yinwang', '-r', '-h', '-o', '%i|%j|%T|%q|%b|%R'],
                       capture_output=True, text=True, timeout=45)
    require(p.returncode == 0, 'Queue query failed: ' + p.stderr)
    result = []
    for s in p.stdout.splitlines():
        fields = s.split('|', 5); require(len(fields) == 6, 'Malformed queue row')
        result.append(dict(zip(('id','name','state','qos','gres','reason'), fields)))
    return result


def capacity(rows):
    pending = [r for r in rows if r['qos'] == 'normal' and r['state'] == 'PENDING'
               and r['gres'] not in ('N/A', 'gres/gpu:0') and r['name'].startswith(('v9', 'v10'))]
    return sum(r['qos'] == 'normal' for r in rows) < 29 and len(pending) < 8


def main():
    pin, manifest = verify_manifest(); root = pin.parent
    source = Path(manifest['source_root'])
    ledger = {'utc': utc(), 'controller': os.environ['SLURM_JOB_ID'], 'status': 'CPU_NUMERICS',
              'gate': {'stage': 'gate', 'status': 'PLANNED'},
              'full': [{'stage': 'full', 'family': f, 'status': 'PLANNED'} for f in FAMILIES]}
    with lock(root/'LOOP.lock'):
        require(not (root/'LEDGER.json').exists(), 'Existing validation loop ledger; do not duplicate')
        write(root/'LEDGER.json', ledger)
        env = dict(os.environ, V10_TEST_DEVICE='cpu',
                   PYTHONPATH=str(source/'src')+':'+str(source))
        with (root/'CPU_NUMERICS.txt').open('w') as stream:
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                       'tests/v10/test_supplement.py', 'tests/v10/test_v10_adapter.py',
                       'tests/v10/test_mf_vcs_core.py', 'tests/v10/test_t00_manifest.py'],
                       cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
        cpu = {'utc': utc(), 'status': 'PASS' if result.returncode == 0 else 'FAIL',
               'exit_code': result.returncode, 'job': os.environ['SLURM_JOB_ID'],
               'code_sha256': manifest['code_sha256'], 'log_sha256': sha(root/'CPU_NUMERICS.txt')}
        write(root/'CPU_NUMERICS.json', cpu)
        require(result.returncode == 0, 'CPU numerical/regression tests failed')
        # The formal runner must reject an unvalidated release with a real nonzero exit.
        draft = root/'INVALID_RELEASE_FIXTURE.json'; write(draft, {'status': 'DRAFT'})
        bad_env = dict(env, V10_SUPPLEMENT_EXPECT_RELEASE=sha(draft))
        with (root/'RELEASE_REJECTION.txt').open('w') as stream:
            refusal = subprocess.run([sys.executable, 'scripts/v10_supplement/run.py', 'cal', 'MI_native',
                                      '--release', str(draft)], cwd=source, env=bad_env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        require(refusal.returncode != 0 and 'has not been validated/frozen' in (root/'RELEASE_REJECTION.txt').read_text(),
                'Formal runner did not reject invalid release for the expected reason')
        ledger['status'] = 'WAIT_CUDA_GATE'; write(root/'LEDGER.json', ledger)

        def submit(unit):
            verify_manifest()
            name = 'v10val_gate' if unit['stage'] == 'gate' else 'v10val_f' + str(FAMILIES.index(unit['family']))
            require(unit['status'] == 'PLANNED' and not unit.get('job'), 'Duplicate validation submission')
            args = ['sbatch', '--parsable', '--job-name='+name,
                    '--export=ALL,V10_VALIDATION_SOURCE='+str(source)+',V10_VALIDATION_MANIFEST='+str(pin)+
                    ',V10_VALIDATION_SHA256='+os.environ['V10_VALIDATION_SHA256'],
                    str(source/'slurm/v10_supplement/validate.sbatch'), unit['stage']]
            if unit['stage'] == 'full': args += ['--family', unit['family']]
            unit.update(status='SUBMIT_INTENT', command=args, utc=utc()); write(root/'LEDGER.json', ledger)
            r = subprocess.run(args, text=True, capture_output=True, timeout=45)
            unit['submission'] = {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
            write(root/'LEDGER.json', ledger)
            if r.returncode != 0 and not r.stdout.strip() and 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
                unit['status']='PLANNED'; write(root/'LEDGER.json',ledger); return
            require(r.returncode == 0 and re.fullmatch(r'\d+(;[^\s]+)?', r.stdout.strip()),
                    'Ambiguous/failed submission; manual reconciliation required')
            unit.update(status='SUBMITTED', job=r.stdout.strip().split(';')[0]); write(root/'LEDGER.json',ledger)
            print('SUBMITTED', name, unit['job'], flush=True)

        def observe(unit, rows):
            if unit['status'] != 'SUBMITTED' or any(r['id'] == unit['job'] for r in rows): return
            unit.setdefault('absent_unix', time.time())
            receipt = root/'exits'/(unit['job']+'.json')
            if not receipt.exists() and time.time()-unit['absent_unix'] < 300: return
            try:
                r = read(receipt)
                require(r['job'] == unit['job'] and r['exit_code'] == 0 and
                        r['validation_manifest_sha256'] == os.environ['V10_VALIDATION_SHA256'], 'Missing successful exit receipt')
                reports = ['GATE.json'] if unit['stage'] == 'gate' else ['T02.json', 'RUNNER.json']
                for name in reports:
                    p = root/'runs'/unit['job']/name; report=read(p)
                    require(report['status'] == 'PASS' and report['code_sha256'] == manifest['code_sha256'], 'Report not verified PASS')
                    if unit['stage']=='gate': require({f['family'] for f in report['families']} == set(FAMILIES), 'Incomplete gate coverage')
                    else: require(report['family']==unit['family'], 'Report family mismatch')
                    unit.setdefault('reports',{})[name]={'path':str(p),'sha256':sha(p)}
                unit['status']='COMPLETE_VERIFIED'
            except Exception as e:
                unit.update(status='FAILED_REVIEW_REQUIRED', error=repr(e))
            write(root/'LEDGER.json',ledger)

        for _ in range(94*120):
            try: rows = queue()
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                write(root/'STATUS.json',{'utc':utc(),'status':'QUEUE_RETRY_NO_SUBMISSION','error':repr(e)})
                time.sleep(30); continue
            observe(ledger['gate'], rows)
            for unit in ledger['full']: observe(unit, rows)
            require(ledger['gate']['status'] != 'FAILED_REVIEW_REQUIRED', 'CUDA/real-host gate failed; full validations not submitted')
            eligible = [ledger['gate']] if ledger['gate']['status'] != 'COMPLETE_VERIFIED' else ledger['full']
            for unit in eligible:
                if unit['status'] == 'PLANNED':
                    if not capacity(queue()): break
                    submit(unit)
            failed = [u['family'] for u in ledger['full'] if u['status']=='FAILED_REVIEW_REQUIRED']
            complete = sum(u['status']=='COMPLETE_VERIFIED' for u in ledger['full'])
            pending = sum(u['status']=='PLANNED' for u in ledger['full'])
            ledger['status']='FULL_VALIDATION_ACTIVE' if ledger['gate']['status']=='COMPLETE_VERIFIED' else 'WAIT_CUDA_GATE'
            write(root/'STATUS.json',{'utc':utc(),'status':ledger['status'],'controller':ledger['controller'],
                  'gate':ledger['gate'],'full_complete':complete,'full_planned':pending,'failed':failed,
                  'shared_normal_gpu_cap':8,'target_project_pending':8,'scientific_release_created':False})
            if complete + len(failed) == len(FAMILIES):
                ledger['status']='VALIDATIONS_PASS_REVIEW_BEFORE_SCIENTIFIC_RELEASE' if not failed else 'REVIEW_REQUIRED'
                write(root/'LEDGER.json',ledger); write(root/'FINISHED.json',ledger)
                require(not failed, 'Full validation failures: '+str(failed)); return
            write(root/'LEDGER.json',ledger)
            time.sleep(30)
        raise RuntimeError('Validation controller exceeded 94 hours')


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        root=Path(os.environ['V10_VALIDATION_MANIFEST']).parent
        write(root/'ERROR.json',{'utc':utc(),'status':'FAIL','error':repr(error),'traceback':traceback.format_exc()})
        raise
