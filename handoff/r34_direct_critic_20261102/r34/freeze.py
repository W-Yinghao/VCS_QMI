"""Write PROTOCOL_FROZEN.json (before any formal unit): configs, input locks, cache manifests, code sha, gate."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDS, OUT, ROOT, PACKAGE, sha256, write_json, read_json, utc


def main():
    gate = read_json(ROOT / 'GPU_GATE.json'); assert gate['ALL_PASS']
    code = {f.name: sha256(f) for f in sorted((ROOT / 'r34').glob('*.py'))}
    assert all(gate['code_sha256'].get(k) == v for k, v in code.items() if k not in ('freeze.py', 'summarize.py')), 'code changed after gate'
    units = read_json(ROOT / 'configs' / 'UNITS.json')
    caches = {}
    for cond in CONDS:
        man = read_json(OUT / 'cache' / cond / 'CACHE_MANIFEST.json')
        caches[cond] = dict(manifest=str(OUT / 'cache' / cond / 'CACHE_MANIFEST.json'), manifest_sha256=sha256(OUT / 'cache' / cond / 'CACHE_MANIFEST.json'),
                            parent=man['parent'], observed_map=man['observed_map'], native_dev=man['native_dev'], cache_gpu=man['env'])
    out = dict(line='R34 DIRECT_CRITIC_RETRIEVAL_v1', frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS',
               design_package=dict(path=str(PACKAGE), sha256_manifest=sha256(PACKAGE / 'SHA256.json'), plan_sha256=sha256(PACKAGE / 'docs/SERVER_PLAN_CN.md'),
                                   matrix_sha256=sha256(PACKAGE / 'MATRIX.json')),
               units=units, config_sha256={u: sha256(ROOT / 'configs' / f'{u}.json') for u in units},
               configs={u: read_json(ROOT / 'configs' / f'{u}.json') for u in units},
               input_locks=dict(caches=caches, locks=read_json(OUT / 'locks' / 'LOCKS.json'), locks_sha256=sha256(OUT / 'locks' / 'LOCKS.json')),
               pq_rules=read_json(ROOT / 'PQ_RULES_AND_F1_MISMATCHES.json'), code_sha256=code,
               gate=dict(file='GPU_GATE.json', sha256=sha256(ROOT / 'GPU_GATE.json'), job=gate['job'], gpu=gate['gpu']),
               cpu_check=read_json(ROOT / 'CPU_CHECK.json'),
               primary_endpoint='fixed step-2000 full-DEV R@1/5/10 both directions + rSum (upstream IN2R i2t/t2i on peer-mean g); no checkpoint selection',
               test_read=False, formal_execution='one GPU job per condition running its 6 units sequentially (same GPU per condition), normal QOS, nice 0')
    write_json(ROOT / 'PROTOCOL_FROZEN.json', out)
    print('FROZEN', sha256(ROOT / 'PROTOCOL_FROZEN.json'))


if __name__ == '__main__':
    main()
