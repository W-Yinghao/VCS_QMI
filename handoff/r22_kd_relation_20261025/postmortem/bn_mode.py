"""Post-mortem: train-mode vs eval-mode accuracy of final relation checkpoints and a control on fixed FIT batches (no training)."""
import json, sys
from pathlib import Path
BASE = Path('/home/infres/yinwang/CS_QMI'); sys.path.insert(0, str(BASE / 'handoff/r22_other_apps_20261023/engine'))
import guard, torch
from kd import runner as K
torch.set_num_threads(4)
runs = {'control_dkd_none_31415': BASE / 'handoff/r22_other_apps_20261023/runs/kd_full_dkd_none_l0_s31415',
        'rel_direct_31415': BASE / 'handoff/r22_kd_relation_20261025/runs/kd_rel_direct_dkd_s31415',
        'rel_vcs_57721': BASE / 'handoff/r22_kd_relation_20261025/runs/kd_rel_vcs_dkd_s57721',
        'rel_bce_31415': BASE / 'handoff/r22_kd_relation_20261025/runs/kd_rel_bce_dkd_s31415'}
cfg = json.loads((runs['control_dkd_none_31415'] / 'resolved_config.json').read_text()); guard.install(guard.kd_roots(cfg))
man = K.read_manifest(cfg['data']['manifest']); fit = K.RoleDataset(cfg['data']['root'], man, 'fit', augment=False, seed=0)
batches = [b for _, b in zip(range(8), K._loader(fit, cfg, False, 0))]
out = {}
for name, d in runs.items():
    c = json.loads((d / 'resolved_config.json').read_text()); ck = torch.load(d / 'last.pt', map_location='cpu', weights_only=True)
    m = K.create_model(c['student'], man['num_classes'], c['seed']); K.feature_dim(m); m.load_state_dict(ck['student'])
    acc = {}
    for mode in ('train', 'eval'):
        m.train(mode == 'train'); hit = n = 0; fn = []
        with torch.no_grad():
            for x, y, _ in batches:
                lg, f = m(x); hit += int((lg.argmax(1) == y).sum()); n += len(y); fn.append(float(f.norm(dim=1).mean()))
        acc[mode] = dict(top1=hit / n, feat_norm=sum(fn) / len(fn))
    bn = [mod for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    acc['bn_running_var_max'] = max(float(b.running_var.max()) for b in bn); acc['bn_weight_absmax'] = max(float(b.weight.abs().max()) for b in bn)
    out[name] = acc; print(name, acc, flush=True)
(BASE / 'handoff/r22_kd_relation_20261025/postmortem/BN_MODE.json').write_text(json.dumps(out, indent=1))
