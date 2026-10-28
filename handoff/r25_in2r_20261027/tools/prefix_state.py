"""COMMON_PREFIX_STATE.json: content list, path, size, sha256 of every group prefix (no training, CPU)."""
import glob, hashlib, json, os, torch
ROOT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups'
out = {}
for p in sorted(glob.glob(f'{ROOT}/*/prefix/prefix_state.pt')):
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    st = torch.load(p, map_location='cpu', weights_only=False)
    comp = {k: sorted(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in st.items() if k not in ('history', 'all_loss')}
    qa = st['model_A']['queue']; qb = st['model_B']['queue']
    out[p.split('/')[-3]] = dict(path=p, bytes=os.path.getsize(p), sha256=h, prefix_epoch=st['epoch'], best_rsum_at_prefix=st['best_rsum'],
        components=comp, queue_A=dict(ptr=qa['ptr'], is_full=qa['is_full']), queue_B=dict(ptr=qb['ptr'], is_full=qb['is_full']),
        optimizer_param_groups={m: len(st[m]['optimizer']['param_groups']) for m in ('model_A', 'model_B')},
        dev_epoch0=st['history']['cotrain'][-1]['dev'], warmup_dev_rsum=[w['dev']['rsum'] for w in st['history']['warmup']],
        cudnn_benchmark=st['cudnn_benchmark'],
        definition='native 5 warmup epochs + native co-training epoch 0 (A trains with empty B queue -> no A noisy correction in epoch 0); saved at the end of epoch 0, after which both peers have both prototype directions')
json.dump(out, open('/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/COMMON_PREFIX_STATE.json', 'w'), indent=1, default=str)
print({k: (v['bytes'], v['queue_A'], v['queue_B']) for k, v in out.items()})
