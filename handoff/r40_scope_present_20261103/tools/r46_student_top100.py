"""R46-B secondary readout (zero training, post-hoc): for each finished round-2 unit (and its round-1 unit for reference), the two-stage DEV with the
round-2 student's OWN top-100 as stage 1: DEV student vectors (parent centring), peer-mean student cosine -> top-100 tables (EV.dev_tables) and
'rest' order; the unit's final head reranks the top-100 by peer-mean g (the trainer's readout_r36 with the stage-1 tables / order swapped).
Primary (frozen-H top-100) is the trainer's own readout and is only copied here. No TEST. Usage: python -B tools/r46_student_top100.py <tree> <unit,unit,...>"""
import json, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, UNITS = sys.argv[1], sys.argv[2].split(',')
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('trainer'); import train_unit as TU, evals as EV, r39ext as R9
from data35 import shard_mm
from heads import build_heads, NullSlots
TU.set_determinism()
dev = torch.device('cuda'); res = {}
outp = W / 'R46_STUDENT_TOP100.json'
agg = json.loads(outp.read_text()) if outp.exists() else dict(units={})
ctxs = {}
for u in UNITS:
    cfg = TU.cfg_of(u); d0 = C.RUNS / u; ck = torch.load(d0 / 'ckpt_step8000.pt', map_location=dev, weights_only=False); assert ck['step'] == 8000
    key = (cfg['cond'], (cfg.get('candidate_file') or {}).get('sha256'))
    if key not in ctxs:
        ctxs = {key: TU.Ctx(cfg['cond'], dev, need_local=False, cand_file=cfg.get('candidate_file'))}; ctxs[key].ensure_joint()
    ctx = ctxs[key]; d = ctx.d
    heads = build_heads(cfg['structure'], d['a0b0']['a0'], d['a0b0']['b0'], cfg['init_seed']).to(dev); heads.load_state_dict(ck['heads'])
    nulls = NullSlots().to(dev); nulls.load_state_dict(ck['nulls'])
    stud = R9.build_students(ctx.joint.ck, dev, ('img', 'txt')); stud.load_state_dict(ck['student']); stud.train()
    ctx.joint.clear_eval()
    sims = sum(shard_mm(ctx.joint.table(stud[p], p, 'dev', 'img'), ctx.joint.table(stud[p], p, 'dev', 'cap')) for p in C.PEERS).cpu().numpy() / 2
    d2 = dict(d, native_dev=sims, dev_tab=EV.dev_tables(sims))
    rec, _, _ = TU.readout_r36(heads, nulls, None, TU._shallow(ctx, d=d2), final=True, stud=stud)
    prim = json.loads((d0 / 'RESULT.json').read_text())['final']['dev_retrieval']['rsum']
    stud_only = EV.full_metrics(sims)[0]['rsum']
    agg['units'][u] = dict(rsum_student_top100=rec['dev_retrieval']['rsum'], rsum_primary=prim, rsum_student_cosine_only=stud_only, gpu=torch.cuda.get_device_name(0))
    print('R46_STUDENT_TOP100', u, json.dumps(agg['units'][u]), flush=True)
    ctx.joint.clear_eval()
outp.write_text(json.dumps(agg, indent=1))
