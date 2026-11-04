"""R50 replay control comparison: state at the replay step of (a) R50 replay jobs and (b) the R47-code control (R50_REPLAYCTL_*) of the same reference config,
tensor by tensor (heads / nulls / student). Writes ROOT/REPLAY_R50_CONTROL_COMPARE.json. Usage: python -I -B tools/r50_control_compare.py <ref_unit> <gate_dir_ctl> <gate_dir_r50> [<gate_dir_r50> ...]"""
import json, sys, time
from pathlib import Path
import torch
W = Path(__file__).resolve().parents[1]; ref = sys.argv[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
def load(d, u, step=250):
    return torch.load(Path(d) / u / f'ckpt_step{step}.pt', map_location='cpu', weights_only=False)
def cmp(a, b):
    n = m = 0; tot = 0
    for part in ('heads', 'nulls', 'student'):
        if a.get(part) is None:
            continue
        for k in a[part]:
            tot += 1
            if not torch.equal(a[part][k], b[part][k]):
                n += 1; m = max(m, float((a[part][k].double() - b[part][k].double()).abs().max()))
    return dict(tensors=tot, not_bitwise=n, max_abs=m)
ctl = load(sys.argv[2], f'R50_REPLAYCTL_{ref}'); orig = torch.load(OUT / 'runs_r47' / ref / 'ckpt_step250.pt', map_location='cpu', weights_only=False)
res = dict(reference=ref, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), control_dir=sys.argv[2], control_vs_reference_ckpt=cmp(ctl, orig), r50={})
for g in sys.argv[3:]:
    a = load(g, f'R50_REPLAY_{ref}'); res['r50'][g] = dict(vs_control=cmp(a, ctl), vs_reference_ckpt=cmp(a, orig))
p = W / 'REPLAY_R50_CONTROL_COMPARE.json'; agg = json.loads(p.read_text()) if p.exists() else {}; agg[ref] = res; p.write_text(json.dumps(agg, indent=1))
print(json.dumps(res))
