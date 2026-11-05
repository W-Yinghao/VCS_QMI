"""CR4 matched-group runner: one Slurm job per condition, all its cells on the single GPU it was given (CR3_A1 principle).

Steps: frozen-hash check -> GROUP_ENV (pinned; later attempts must match) -> reference receipt (250-step CE replay with the CR4
trainer and RECT replay with the unchanged CR3 trainer, compared bitwise with the stored CR3 step-250 checkpoints; reuse CR3
references only if GPU/driver/torch/CUDA/cuDNN equal CR3's group env AND both replays are bitwise, else rerun this condition's
CR3 RECT+COMPLETE references with the unchanged CR3 code) -> identity check for single-positive graphs (full-graph check +
250-step PC trajectory == CR3 COMPLETE bitwise -> REUSE, else train new) -> train/evaluate PC units (separate eval process).
"""
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr4_student as S4                                            # noqa: E402
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr3_20261104/code/site')
import cr3_student as S3                                            # noqa: E402 unchanged CR3 trainer (references only)
from cr1.io import read_json, write_json, sha256                   # noqa: E402

TREE = HERE.parents[1]; OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104')
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR3S = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/students_A1')
PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'; EVAL = '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site/eval_student.py'
CCD = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'; NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
REPLAY = {'Flickr00': 'OBS', 'Flickr60': 'J_KEEP_TOP1', 'CC152K_RECOVERED_20261104': 'J_KEEP_TOP1'}
JOB = os.environ.get('SLURM_JOB_ID'); HOST = socket.gethostname()
utc = lambda: time.strftime('%FT%TZ', time.gmtime())


def env():
    drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split('\n')[0].strip()
    return {'gpu_name': torch.cuda.get_device_name(), 'driver': drv, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}


def frozen(fz):
    for rel, h in fz['code_sha256'].items():
        if sha256(Path(rel) if rel.startswith('/') else TREE/rel) != h: raise RuntimeError(f'frozen code changed: {rel}')
    for p, h in fz['data_sha256'].items():
        if sha256(p) != h: raise RuntimeError(f'frozen data changed: {p}')
    for u, h in fz['unit_config_sha256'].items():
        if sha256(TREE/'package'/'configs'/'units'/f'{u}.json') != h: raise RuntimeError(f'unit config changed {u}')


def evaluate(d, c):
    dev, tr = (f'{CCD}/cache/dev', f'{CCD}/evaluator_only/dev_positives.json') if c.startswith('CC152K') else (f'{NR1}/cache/dev', f'{NR1}/evaluator_only/dev_positives.json')
    p = subprocess.run([PY, '-I', '-B', EVAL, '--no-frozen-check', '--student', str(d), '--dev-cache', dev, '--truth', tr, '--prepared', str(CR1/c/'prepared'), '--device', 'cuda'], capture_output=True, text=True)
    (Path(d)/'EVAL_STDOUT.txt').write_text(p.stdout + p.stderr[-4000:])
    if p.returncode: raise RuntimeError(f'eval failed {d}: {p.stderr[-2000:]}')


def same(p1, p2):
    a = torch.load(p1, map_location='cpu', weights_only=False)['model']; b = torch.load(p2, map_location='cpu', weights_only=False)['model']
    return all(torch.equal(a[k], b[k]) for k in a), max(float((a[k]-b[k]).abs().max()) for k in a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--condition', required=True); a = ap.parse_args(); c = a.condition
    fz = read_json(TREE/'PROTOCOL_FROZEN_CR4.json'); frozen(fz)
    units = [u for u, i in fz['units'].items() if i['condition'] == c]
    g = OUT/'groups'/c; g.mkdir(parents=True, exist_ok=True); e = env(); ge = g/'GROUP_ENV.json'
    if ge.exists():
        prev = read_json(ge)
        if {k: prev[k] for k in e} != e: raise RuntimeError(f'GROUP ENV MISMATCH {prev} vs {e}: stop (no mixed-GPU matched group)')
    else: write_json(ge, dict(e, condition=c, first_job=JOB, host=HOST, utc=utc(), units=units))
    bank, prep = str(CR1/'banks'/c/'fit_bank.npz'), str(CR1/c/'prepared')
    # ---------------- reference receipt
    rr = g/'REFERENCE_RECEIPT.json'
    if not rr.exists():
        cr3env = read_json(CR3S/c/'GROUP_ENV.json'); env_eq = {k: cr3env[k] for k in e} == e
        r = REPLAY[c]; gp = fz['graph_paths'][r] if r in fz['graph_paths'] else None
        gp = next(i['graph_path'] for i in fz['units'].values() if i['condition'] == c and i['relation'] == r)
        rp = g/'replay'; rp.mkdir(exist_ok=True)
        S4.train(bank, prep, gp, rp/'CE', 'REPLAY', 'CE', 'cuda', steps=250, saves=(0, 250), draw_log=rp/'CE_draws.jsonl')
        S3.train(bank, prep, gp, rp/'RECT', 'REPLAY', 'RECT', 'cuda', steps=250, check_frozen=False, saves=(0, 250))
        S4.train(bank, prep, gp, rp/'PC', 'REPLAY', 'PC', 'cuda', steps=250, saves=(0, 250), draw_log=rp/'PC_draws.jsonl')
        ce_bit, ce_d = same(rp/'CE'/'step_000250.pt', CR3S/c/f'CR3_{c}_{r}_COMPLETE_s1001'/'step_000250.pt')
        re_bit, re_d = same(rp/'RECT'/'step_000250.pt', CR3S/c/f'CR3_{c}_{r}_RECT_s1001'/'step_000250.pt')
        draws_eq = (rp/'CE_draws.jsonl').read_text() == (rp/'PC_draws.jsonl').read_text()
        dec = 'REUSE_CR3_REFERENCES' if (env_eq and ce_bit and re_bit) else 'RERUN_CR3_REFERENCES_IN_THIS_GROUP'
        write_json(rr, dict(condition=c, replay_graph=r, this_env=e, cr3_env={k: cr3env[k] for k in e}, env_equal=env_eq,
                            complete_ce_replay_bitwise=ce_bit, complete_ce_replay_max_abs=ce_d, rect_replay_bitwise=re_bit, rect_replay_max_abs=re_d,
                            pc_vs_ce_same_draws_250=draws_eq, decision=dec, job=JOB, utc=utc()))
    rec = read_json(rr); print('REFERENCE', rec['decision'], flush=True)
    if rec['decision'].startswith('RERUN'):
        for u in units:
            i = fz['units'][u]
            for cons in ('RECT', 'COMPLETE'):
                ru = f"CR3_{c}_{i['relation']}_{cons}_s1001"; d = OUT/'refs_rerun'/c/ru
                if (d/'UNIT_COMPLETE.json').exists(): continue
                d.mkdir(parents=True, exist_ok=True)
                if not (d/'DONE.json').exists(): S3.train(bank, prep, i['graph_path'], d, ru, cons, 'cuda', check_frozen=False)
                evaluate(d, c); write_json(d/'UNIT_COMPLETE.json', {'unit': ru, 'job': JOB, 'utc': utc(), 'env': e}); print('REF_RERUN', ru, flush=True)
    # ---------------- PC units (identity checks first)
    for u in units:
        i = fz['units'][u]; d = OUT/'students'/c/u
        if (d/'UNIT_COMPLETE.json').exists(): print('skip', u, flush=True); continue
        d.mkdir(parents=True, exist_ok=True)
        with open(d/'ATTEMPTS.txt', 'a') as f: f.write(f"{JOB} {HOST} {e['gpu_name']}, {e['driver']} {utc()}\n")
        if i['plan'].startswith('IDENTITY_CHECK'):
            m = load_npz(i['graph_path']).tocsr(); rows = np.diff(m.indptr); cols = np.diff(m.tocsc().indptr)
            single = bool((rows == 1).all() and ((cols == 0) | (cols == 1)).all())
            t = d/'identity_trajectory'
            if not (t/'step_000250.pt').exists(): S4.train(bank, prep, i['graph_path'], t, u, 'PC', 'cuda', steps=250, saves=(0, 250))
            cr3c = CR3S/c/f"CR3_{c}_{i['relation']}_COMPLETE_s1001"
            bit, dmax = same(t/'step_000250.pt', cr3c/'step_000250.pt'); env_eq = {k: read_json(CR3S/c/'GROUP_ENV.json')[k] for k in e} == e
            ok = single and bit and env_eq
            write_json(d/'IDENTITY_RECEIPT.json', dict(unit=u, all_text_rows_single_positive=bool((rows == 1).all()), all_image_columns_at_most_one=bool(((cols == 0) | (cols == 1)).all()),
                                                       pc_equals_ce_structurally=single, trajectory250_bitwise_vs_CR3_COMPLETE=bit, trajectory250_max_abs=dmax, env_equal_CR3=env_eq,
                                                       decision='REUSE_CR3_COMPLETE' if ok else 'TRAIN_NEW_FALLBACK', reused_dir=str(cr3c) if ok else None))
            if ok:
                write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'REUSE_CR3_COMPLETE', 'reused_dir': str(cr3c), 'job': JOB, 'utc': utc()}); print('IDENTITY_REUSE', u, flush=True); continue
        if not (d/'DONE.json').exists():
            r = S4.train(bank, prep, i['graph_path'], d, u, 'PC', 'cuda'); r['group_env'] = e; write_json(d/'DONE.json', r)
        evaluate(d, c); write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'COMPLETE', 'job': JOB, 'utc': utc(), 'env': e}); print('UNIT_COMPLETE', u, flush=True)
    print('GROUP_COMPLETE', c, flush=True)


if __name__ == '__main__': main()
