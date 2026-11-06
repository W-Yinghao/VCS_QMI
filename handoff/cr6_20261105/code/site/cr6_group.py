"""CR6 matched group = ONE job on ONE GPU (GROUP_ENV pinned; a resubmitted attempt must land on the identical GPU/driver/stack).
  A1                 : 12 estimator fits (Flickr60, J/BCE x E01..E23, source seed 2002) + J/BCE cross-fit score exports (CR1 score)
  A2 --seed s        : Flickr60 x student s: 3 new-source COMPLETE_PC students (CR5 trainer) + the 6 old-source FROZEN references
  B  --condition c   : c x student 1002: in-job TAIL acceptance on this GPU, 6 FROZEN references, 6 TAIL_LORA8 PC students
References: reuse the CR5 run only if (i) binding identity (graph sha, head/sampler seed, consumer path, endpoint checkpoint sha) and
(ii) a 250-step replay in THIS job is bitwise identical to the stored step-250 model; otherwise retrain the reference (8000 steps) here.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402

JOB, HOST = K.job()


def pin_env(g, meta):
    import torch  # noqa
    e = K.gpu_env(); ge = g/'GROUP_ENV.json'
    if ge.exists():
        prev = K.read_json(ge)
        if {k: prev[k] for k in e} != e: raise RuntimeError(f'GROUP ENV MISMATCH {prev} vs {e}')
    else: K.write_json(ge, dict(e, **meta, first_job=JOB, host=HOST, utc=K.utc()))
    with open(g/'ATTEMPTS.txt', 'a') as f: f.write(f"{JOB} {HOST} {e['gpu_name']}, {e['driver']} {K.utc()}\n")
    return e


def complete(d, **kw):
    K.write_json(Path(d)/'UNIT_COMPLETE.json', dict(job=JOB, host=HOST, utc=K.utc(), **kw))


def run_A1(fz):
    import cr6_source as SRC
    g = K.OUT/'A1'; g.mkdir(parents=True, exist_ok=True); e = pin_env(g, dict(group='A1'))
    for u, i in sorted(fz['units'].items()):
        if i['stage'] != 'A1_ESTIMATOR': continue
        d = K.OUT/'A1'/'experts'/i['source_objective']/i['expert']
        if (d/'DONE.json').exists(): continue
        t = time.perf_counter()
        r = SRC.fit(K.bank_path('Flickr60'), K.prep_dir('Flickr60'), i['source_objective'], i['expert'], d, 'cuda', i['head_seed'], i['sampler_seed'], unit=u)
        K.write_json(d/'COST.json', dict(unit=u, wall_seconds=time.perf_counter()-t, env=e)); print('A1_DONE', u, flush=True)
    for S in ('J', 'BCE'):
        p = K.OUT/'A1'/'scores'/f'{S}.npz'
        if Path(str(p)+'.json').exists(): continue
        t = time.perf_counter()
        r = subprocess.run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/cr1_prod.py'), 'score', '--bank', str(K.bank_path('Flickr60')), '--prepared', str(K.prep_dir('Flickr60')),
                            '--source', S, '--models-root', str(K.OUT/'A1'/'experts'), '--out', str(p), '--device', 'cuda'], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-3000:])
        K.write_json(K.OUT/'A1'/'scores'/f'{S}_COST.json', dict(wall_seconds=time.perf_counter()-t, env=e)); print('SCORE_DONE', S, flush=True)
    complete(g, group='A1', status='COMPLETE', env=e); print('GROUP_COMPLETE A1', flush=True)


def verify_or_rerun_ref(rid, i, g):
    import cr5_student as S5
    import cr6_eval as EV
    d = g/'refs'/rid; d.mkdir(parents=True, exist_ok=True)
    if (d/'UNIT_COMPLETE.json').exists(): return K.read_json(d/'UNIT_COMPLETE.json')
    src = Path(i['original_dir']); bank, prep = str(K.bank_path(i['condition'])), str(K.prep_dir(i['condition']))
    done = K.read_json(src/'DONE.json')
    ident = dict(graph_sha=K.sha256(i['graph_path']) == i['graph_sha256'] == done['relation_sha256'], head_seed=done['head_seed'] == i['head_seed'],
                 sampler_seed=done['sampler_seed'] == i['sampler_seed'], endpoint_sha=K.sha256(src/'step_008000.pt') == done['checkpoint_sha256'] == i['original_ckpt_sha256'],
                 consumer=done['consumer'] in (('COMPLETE_PC',) if not i.get('single_positive_identity') else ('COMPLETE_PC', 'RECT')))
    v = d/'replay250'
    if not (v/'step_000250.pt').exists():
        with K.stop_after(250): S5.train(bank, prep, i['graph_path'], v, rid, 'COMPLETE_PC', i['head_seed'], i['sampler_seed'], 'cuda')   # production T_max=8000
    x = torch.load(v/'step_000250.pt', map_location='cpu', weights_only=False)['model']; y = torch.load(src/'step_000250.pt', map_location='cpu', weights_only=False)['model']
    bit = all(torch.equal(x[k], y[k]) for k in x); dmax = max(float((x[k]-y[k]).abs().max()) for k in x)
    ok = bit and all(ident.values())
    K.write_json(d/'REUSE_RECEIPT.json', dict(ref=rid, original_unit=i['original_unit'], original_dir=str(src), identity=ident, replay250_bitwise=bit, replay250_max_abs=dmax,
                 decision='REUSE_VERIFIED' if ok else 'RERUN_IN_GROUP', group_env=K.read_json(g/'GROUP_ENV.json'), original_gpu=done.get('gpu'), utc=K.utc()))
    if ok: train_dir = src
    else:
        train_dir = d/'train'
        if not (train_dir/'DONE.json').exists(): S5.train(bank, prep, i['graph_path'], train_dir, rid, 'COMPLETE_PC', i['head_seed'], i['sampler_seed'], 'cuda')
    EV.evaluate_unit(i['condition'], train_dir, d/'eval')
    u = dict(ref=rid, status='REUSE_VERIFIED' if ok else 'RERUN_IN_GROUP', checkpoint_dir=str(train_dir))
    complete(d, **u); print('REF', rid, u['status'], flush=True)
    return u


def run_A2(fz, seed):
    import cr5_student as S5
    import cr6_eval as EV
    gname = f'A2_Flickr60_t{seed}'; g = K.OUT/'groups'/gname; g.mkdir(parents=True, exist_ok=True); e = pin_env(g, dict(group=gname))
    reg = K.read_json(K.OUT/'A1'/'GRAPH_REGISTRY_SRC2002.json')['graphs']
    for rid, i in sorted(fz['references'].items()):
        if gname in i['groups']: verify_or_rerun_ref(rid, i, g)
    for u, i in sorted(fz['units'].items()):
        if i['stage'] != 'A2_SOURCE_STUDENT' or i['head_seed'] != seed: continue
        d = K.OUT/'students'/'Flickr60'/u
        if (d/'UNIT_COMPLETE.json').exists(): continue
        gr = reg[f"{i['relation']}_src2002"]; gp = gr['path']
        if K.sha256(gp) != gr['relation_sha256']: raise RuntimeError('new graph changed')
        if not (d/'DONE.json').exists():
            r = S5.train(str(K.bank_path('Flickr60')), str(K.prep_dir('Flickr60')), gp, d, u, 'COMPLETE_PC', i['head_seed'], i['sampler_seed'], 'cuda')
        EV.evaluate_unit('Flickr60', d, d)
        complete(d, unit=u, status='COMPLETE', env=e, graph_sha256=gr['relation_sha256']); print('UNIT_COMPLETE', u, flush=True)
    complete(g, group=gname, status='COMPLETE', env=e); print('GROUP_COMPLETE', gname, flush=True)


def run_B(fz, cond):
    import cr6_tail as TL
    import cr6_eval as EV
    import acceptance_cr6 as AC
    gname = f'B_{cond}_t1002'; g = K.OUT/'groups'/gname; g.mkdir(parents=True, exist_ok=True); e = pin_env(g, dict(group=gname))
    units = sorted((u for u, i in fz['units'].items() if i['stage'] == 'B_STUDENT_SCOPE' and i['condition'] == cond), key=lambda u: fz['units'][u]['relation'])
    for rid, i in sorted(fz['references'].items()):
        if gname in i['groups']: verify_or_rerun_ref(rid, i, g)
    hid = None
    if not all((K.OUT/'students'/cond/u/'UNIT_COMPLETE.json').exists() for u in units):
        acc = g/'TAIL_ACCEPTANCE_IN_JOB.json'
        jgraph = fz['units'][[u for u in units if fz['units'][u]['relation'] == 'J_KEEP_TOP1'][0]]['graph_path']
        S0 = TL.TailStudent(cond, jgraph, 1002, 1002, 'cuda'); hid = S0.hid
        rtx_ok = 'RTX PRO 6000' in e['gpu_name'] and any(v.get('validates') and v.get('partition') == 'RTX6000PRO' for v in (K.read_json(K.TREE/'ACCEPTANCE_REGISTRY.json').values() if (K.TREE/'ACCEPTANCE_REGISTRY.json').exists() else []))
        if rtx_ok and not acc.exists(): K.write_json(acc, dict(status='COVERED_BY_RTX_ACCEPTANCE', env=e, hidden_load_seconds=hid.load_seconds, ram=hid.ram, utc=K.utc()))
        if not rtx_ok and (not acc.exists() or {k: K.read_json(acc)['env'][k] for k in e} != e):
            AC.R.clear(); AC.light_tail_checks(cond, jgraph, g, 'cuda', student=S0, prefix='T')
            ok = all(v['ok'] for v in AC.R.values())
            K.write_json(acc, dict(status='PASS' if ok else 'FAIL', checks=dict(AC.R), env=e, hidden_load_seconds=hid.load_seconds, ram=hid.ram, utc=K.utc()))
            if not ok: raise RuntimeError('IN-JOB TAIL ACCEPTANCE FAILED; no TAIL training on this GPU')
        del S0; torch.cuda.empty_cache()
    for u in units:
        i = fz['units'][u]; d = K.OUT/'students'/cond/u
        if (d/'UNIT_COMPLETE.json').exists(): continue
        if K.sha256(i['graph_path']) != i['graph_sha256']: raise RuntimeError('old graph changed')
        if not (d/'DONE.json').exists():
            S = TL.TailStudent(cond, i['graph_path'], i['head_seed'], i['adapter_seed'], 'cuda', hid=hid)
            r = TL.train(cond, i['graph_path'], d, u, i['head_seed'], i['sampler_seed'], i['adapter_seed'], 'cuda', student=S)
            refd = K.read_json(g/'refs'/i['baseline_ref_id']/'UNIT_COMPLETE.json')['checkpoint_dir']
            rd = K.read_json(Path(refd)/'DONE.json').get('same_draw_receipt', {})
            K.write_json(d/'SAME_DRAW_VS_REFERENCE.json', dict(tail=r.get('same_draw_receipt'), reference=rd, reference_dir=refd,
                                                               equal=(r.get('same_draw_receipt') or {}).get('edge_hits_sha256') == rd.get('edge_hits_sha256')))
            del S; torch.cuda.empty_cache()
        EV.evaluate_unit(cond, d, d)
        complete(d, unit=u, status='COMPLETE', env=e); print('UNIT_COMPLETE', u, flush=True)
    complete(g, group=gname, status='COMPLETE', env=e); print('GROUP_COMPLETE', gname, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('group', choices=['A1', 'A2', 'B']); ap.add_argument('--seed', type=int); ap.add_argument('--condition')
    ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    fz = K.frozen_check()
    sys.path.insert(0, str(K.CR5_TREE/'code'/'site'))
    if a.dry_run:
        print('DRY_RUN_OK', a.group, a.seed, a.condition, len(fz['units']), len(fz['references'])); return
    {'A1': lambda: run_A1(fz), 'A2': lambda: run_A2(fz, a.seed), 'B': lambda: run_B(fz, a.condition)}[a.group]()


if __name__ == '__main__': main()
