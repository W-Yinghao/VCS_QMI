"""Write FREEZE.json: every package file the runs read, the shared read-trace guard, the N2M experiment receipt and the reused controls."""
from common import HERE, BASE, N2M, EXP, TRACE, SEEDS, sha, write
files = [HERE / n for n in ('common.py', 'run.py', 'gpu.sbatch', 'cpu.sbatch', 'PROTOCOL.json')] + [TRACE, EXP, N2M / 'EXPERIMENT.sha256', N2M / 'runtime/RUNTIME.json']
for s in SEEDS:
    for a in ('VCS', 'ConstantU', 'ShuffleVCS'):
        files += [N2M / 'runs' / f's{s}__{a}' / n for n in ('full_result.json', 'train_log.json', 'run_manifest.json', 'identity.json')]
key = lambda p: str(p.relative_to(BASE)) if BASE in p.parents else str(p)
write(HERE / 'FREEZE.json', dict(files={key(p): sha(p) for p in files},
      note='N2M package code and runtime files are verified at run time against the frozen experiment identity; training inputs against its unit hashes.'))
print(len(files), 'files frozen')
