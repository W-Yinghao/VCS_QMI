"""R30 v1.1 parents (parents/<parent_id>) vs the R29 warmup of the same cell and source (H -> R29 h_native, L -> R29
l_listwise_vcs): per warmup epoch the full-DEV readout (six recalls, rSum, per-peer rSum) and the R29_WARMUP_DIAG.json
entries; bitwise equality and max abs difference; GPU types of both (bitwise only expected on the same GPU type; R29 saved no
warmup state, so model parameters cannot be compared). Usage: compare_r29_warmup.py OUT_BASE ROOT"""
import csv, glob, json, os, sys
out_base, root = sys.argv[1], sys.argv[2]
cells = json.load(open(os.path.join(root, 'CELLS_R30.json')))
res = {}
for r in csv.DictReader(open(os.path.join(root, 'v11_protocol', 'WARMUP_PARENTS.csv'))):
    pid = r['parent_id']
    d30 = os.path.join(out_base, 'parents', pid)
    if not os.path.isfile(os.path.join(d30, 'PARENT.json')):
        continue
    cell = cells[f"f{int(r['noise_percent']):02d}_s{int(r['seed'])}"]
    d29 = cell['r29_reference'][r['warmup_source']]
    b = json.load(open(os.path.join(d30, 'PARENT.json')))
    h29 = json.load(open(os.path.join(d29, 'EPOCHS.json')))['warmup']
    run29 = json.load(open(sorted(glob.glob(os.path.join(d29, 'RUN_*.json')))[0]))['env']
    ep, mx, eq = [], 0.0, True
    for e, (x, y) in enumerate(zip(b['warmup_dev'], [w['dev'] for w in h29])):
        keys = sorted(k for k in x if isinstance(x[k], (int, float)))
        diff = {k: x[k] - y[k] for k in keys}
        same = all(x[k] == y[k] for k in keys)
        eq &= same; mx = max(mx, max(abs(v) for v in diff.values()))
        ep.append(dict(epoch=e, r30_rsum=x['rsum'], r29_rsum=y['rsum'], bitwise=same, diff=diff))
    g30 = json.load(open(os.path.join(d30, 'R29_WARMUP_DIAG.json'))); g29 = json.load(open(os.path.join(d29, 'R29_WARMUP_DIAG.json')))
    dk = sorted(set(g30) & set(g29)); dmx, deq = 0.0, True
    for k in dk:
        for f in g30[k]:
            if isinstance(g30[k][f], (int, float)) and f in g29[k]:
                deq &= g30[k][f] == g29[k][f]; dmx = max(dmx, abs(g30[k][f] - g29[k][f]))
    res[pid] = dict(r29_run=d29, r30_gpu=b['gpu'], r30_host=b['host'], r30_job=b['job'], r29_gpu=run29['gpu'], r29_host=run29['host'], r29_job=run29['job'],
                    same_gpu_type=b['gpu'] == run29['gpu'], dev_bitwise_all_epochs=eq, dev_max_abs_diff=mx, epochs=ep,
                    diag_entries=len(dk), diag_bitwise=deq, diag_max_abs_diff=dmx, parent_sha256=b['sha256'])
json.dump(res, open(os.path.join(root, 'WARMUP_VS_R29.json'), 'w'), indent=1)
print(json.dumps({c: {k: v[k] for k in ('same_gpu_type', 'dev_bitwise_all_epochs', 'dev_max_abs_diff', 'diag_bitwise', 'diag_max_abs_diff')} for c, v in res.items()}, indent=1))
