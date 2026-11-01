"""Fill PARENT_BINDINGS.json (copy of templates/PARENT_BINDINGS.template.json) from <out>/parents/<parent_id>/PARENT.json:
state path, file sha256 (re-hashed), training-state fingerprint, code snapshot (INSTALL_MANIFEST sha256), data split / noise /
environment digests. A parent is BOUND only if its file hash matches, it is a full 5-epoch formal warmup produced by this code
snapshot, and its identity matches the WARMUP_PARENTS row. Built fresh in R30 (no historical boundary exists), so
parent_reuse_verified stays false. Usage: bind_parents.py OUT_BASE ROOT PACKAGE_DIR"""
import csv, hashlib, json, os, sys, time
out_base, root, pkg = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()
tpl = json.load(open(os.path.join(pkg, 'templates', 'PARENT_BINDINGS.template.json')))
rows = {r['parent_id']: r for r in csv.DictReader(open(os.path.join(root, 'v11_protocol', 'WARMUP_PARENTS.csv')))}
man_path = os.path.join(root, 'code_r30v11', 'INSTALL_MANIFEST.json')
man = json.load(open(man_path))
cells = json.load(open(os.path.join(root, 'CELLS_R30.json')))
for p in tpl['parents']:
    pid = p['parent_id']
    d = os.path.join(out_base, 'parents', pid)
    rec_p = os.path.join(d, 'PARENT.json')
    if not os.path.isfile(rec_p):
        p['status'] = 'UNBOUND'; p['note'] = 'parent not built yet'
        continue
    rec = json.load(open(rec_p))
    row = rows[pid]
    cell = cells[f"f{int(row['noise_percent']):02d}_s{int(row['seed'])}"]
    fsha = sha(rec['state_path'])
    checks = dict(file_sha=fsha == rec['sha256'], code=rec['code_sha256'] == man['files_sha256'], formal=rec['max_iters'] == 0,
                  epochs=len(rec['warmup_dev']) == int(row['warmup_epochs']), source=rec['warmup_source'] == row['warmup_source'],
                  seed=rec['identity']['seed'] == int(row['seed']), noise=abs(rec['identity']['noise_ratio'] * 100 - int(row['noise_percent'])) < 1e-9,
                  noise_file=rec['identity']['noise_file'] == cell['noise_file'], gpu=rec['gpu'] == cell['block_gpu'],
                  noise_digest=rec['data']['noise_digest'] == sha(cell['noise_file']))
    p.update(state_path=rec['state_path'], boundary_sha256=fsha, training_state_fingerprint=rec['training_state_fingerprint'],
             source_code_commit=f"code_r30v11 INSTALL_MANIFEST sha256 {sha(man_path)}", data_split_digest=rec['data']['data_split_digest'],
             noise_digest=rec['data']['noise_digest'], env_digest=rec['env_digest'], parent_reuse_verified=False,
             built_by_job=rec['job'], gpu=rec['gpu'], host=rec['host'], checks=checks,
             status='BOUND' if all(checks.values()) else 'INVALID',
             note='built once in R30 (no historical warmup boundary exists: R29 ran with --prefix_epoch -1)')
tpl['bound_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(tpl, open(os.path.join(root, 'PARENT_BINDINGS.json'), 'w'), indent=1)
print(json.dumps({p['parent_id']: p['status'] for p in tpl['parents']}, indent=1))
