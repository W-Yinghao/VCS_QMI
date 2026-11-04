"""Freeze the R50 amendment-A3 re-runs (PROTOCOL_FROZEN_R50_A3.json; the original PROTOCOL_FROZEN_R50.json is NOT edited). Binds the A3 code of r50f / r50c,
the unchanged configs and parent manifests of the re-run units, the A1/A2/A3 amendment files and the A3 smokes (GPU_SMOKE_R50.json entries, ALL_PASS).
check_frozen reads FROZEN_FILES_R50 = (A3, original) and uses the first file that lists all requested units.
Usage: python -I -B tools/freeze_r50_a3.py <smoke_job> [<smoke_job> ...]"""
import glob, hashlib, json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
UNITS = ['R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_NCE', 'R50_A_CC_R31H1001_PRESENT_ONLY_both_joint_NCE']
base = rj(W / 'PROTOCOL_FROZEN_R50.json'); sm = rj(W / 'GPU_SMOKE_R50.json'); smk = {j: sm[j] for j in sys.argv[1:]}
assert smk and all(v['ALL_PASS'] for v in smk.values()), {j: v['ALL_PASS'] for j, v in smk.items()}
out = W / 'PROTOCOL_FROZEN_R50_A3.json'; assert not out.exists()
for u in UNITS:
    assert sh(W / 'configs' / f'{u}.json') == base['config_sha256'][u], f'config changed: {u}'
out.write_text(json.dumps(dict(
    line='R50 amendment A3 freeze (re-runs of the two JOINT NCE units with the A3 check E); everything else as PROTOCOL_FROZEN_R50.json', protocol_id='FIT_FIXED_CENTER_V1',
    frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='FROZEN_BEFORE_A3_RERUNS', base_freeze_sha256=sh(W / 'PROTOCOL_FROZEN_R50.json'),
    amendments={f: sh(W / f) for f in ('R50_AMENDMENT_A1.json', 'R50_AMENDMENT_A2.json', 'R50_AMENDMENT_A3.json')},
    units=UNITS, config_sha256={u: base['config_sha256'][u] for u in UNITS},
    code_sha256_by_tree={t: {Path(f).name: sh(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r50f', 'r50c')}, code_not_bound=[],
    registered_diffs={p: sh(W / p) for p in ('R50_DIFF_R50F.patch', 'R50_DIFF_R50C.patch')},
    tool_sha256={p: sh(W / 'tools' / p) for p in ('r50_patch.py', 'r50_src/r50ext.py', 'r50_diag_e.py', 'freeze_r50_a3.py', 'r50_smoke_check.py')},
    parent_cache_manifest_sha256={m: h for m, h in base['parent_cache_manifest_sha256'].items()},
    gpu_gate=dict(waived_by_user=False, smoke_jobs={j: dict(ALL_PASS=v['ALL_PASS'], units={u: x['passed'] for u, x in v['units'].items()}) for j, v in smk.items()},
                  gpu_smoke_sha256=sh(W / 'GPU_SMOKE_R50.json')),
    tie_policy=base.get('tie_policy'), test_read=False), indent=1, ensure_ascii=False))
print('R50 A3 frozen', UNITS)
