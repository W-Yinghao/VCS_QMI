"""Write R30_V11_DIFF.patch (migration diff code_r30 v1 -> code_r30v11) and R30_DIFF_VS_R29.patch (v1.1 new files vs code_r29),
and code_r30v11/INSTALL_MANIFEST.json (sha256 of every snapshot .py)."""
import difflib, hashlib, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, 'code_r30v11'); V1 = os.path.join(ROOT, 'code_r30'); BASE = '/home/infres/yinwang/CS_QMI/handoff/r29_listwise_20261101/code_r29'
NEW = ['r30_align.py', 'r30_calibrate.py', 'r30_co_train.py', 'r30_run.py', 'r30_state.py']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
py = lambda d: {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith('.py')}
files, base, v1 = py(CODE), py(BASE), py(V1)
changed = sorted(f for f in files if f in base and files[f] != base[f]); added = sorted(set(files) - set(base)); removed = sorted(set(base) - set(files))
assert not removed and not changed and added == NEW, (changed, added, removed)
mig = []
for f in sorted(set(files) | set(v1)):
    a = open(os.path.join(V1, f)).read().splitlines(True) if f in v1 else []
    b = open(os.path.join(CODE, f)).read().splitlines(True) if f in files else []
    if a != b:
        mig += difflib.unified_diff(a, b, 'code_r30/' + f if f in v1 else '/dev/null', 'code_r30v11/' + f)
open(os.path.join(ROOT, 'R30_V11_DIFF.patch'), 'w').writelines(mig)
full = []
for f in added:
    full += difflib.unified_diff([], open(os.path.join(CODE, f)).read().splitlines(True), '/dev/null', 'code_r30v11/' + f)
open(os.path.join(ROOT, 'R30_V11_DIFF_VS_R29.patch'), 'w').writelines(full)
m = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), base_snapshot=BASE, v1_snapshot=V1,
         base_files_sha256=base, changed_vs_base=changed, added_vs_base=added, removed_vs_base=removed, unchanged_byte_identical=sorted(base),
         migration_diff='../R30_V11_DIFF.patch', migration_diff_sha256=sha(os.path.join(ROOT, 'R30_V11_DIFF.patch')),
         diff_vs_r29='../R30_V11_DIFF_VS_R29.patch', diff_vs_r29_sha256=sha(os.path.join(ROOT, 'R30_V11_DIFF_VS_R29.patch')),
         v1_changed=sorted(f for f in files if f in v1 and files[f] != v1[f]), v1_added=sorted(set(files) - set(v1)), files_sha256=files)
json.dump(m, open(os.path.join(CODE, 'INSTALL_MANIFEST.json'), 'w'), indent=1)
print(json.dumps({k: m[k] for k in ('added_vs_base', 'v1_changed', 'v1_added', 'migration_diff_sha256')}))
