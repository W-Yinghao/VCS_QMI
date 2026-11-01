"""Write R29_DIFF.patch (vs the R27 snapshot code_v12) and code_r29/INSTALL_MANIFEST.json (sha256 of every snapshot .py)."""
import difflib, hashlib, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, 'code_r29'); BASE = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/code_v12'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
files = {f: sha(os.path.join(CODE, f)) for f in sorted(os.listdir(CODE)) if f.endswith('.py')}
base = {f: sha(os.path.join(BASE, f)) for f in sorted(os.listdir(BASE)) if f.endswith('.py')}
changed = sorted(f for f in files if f in base and files[f] != base[f]); added = sorted(set(files) - set(base)); removed = sorted(set(base) - set(files))
assert not removed and not changed and added == ['r29_align.py', 'r29_co_train.py', 'r29_run.py'], (changed, added, removed)
patch = []
for f in added:
    patch += difflib.unified_diff([], open(os.path.join(CODE, f)).read().splitlines(True), '/dev/null', 'code_r29/' + f)
for dst in (os.path.join(ROOT, 'R29_DIFF.patch'), os.path.join(CODE, 'R29_DIFF.patch')):
    open(dst, 'w').writelines(patch)
m = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), base_snapshot=BASE,
         base_files_sha256=base, changed_vs_base=changed, added_vs_base=added, removed_vs_base=removed,
         unchanged_byte_identical=sorted(base), diff_patch='R29_DIFF.patch', diff_patch_sha256=sha(os.path.join(CODE, 'R29_DIFF.patch')),
         files_sha256=files)
json.dump(m, open(os.path.join(CODE, 'INSTALL_MANIFEST.json'), 'w'), indent=1)
print(json.dumps({k: m[k] for k in ('changed_vs_base', 'added_vs_base', 'diff_patch_sha256')}))
