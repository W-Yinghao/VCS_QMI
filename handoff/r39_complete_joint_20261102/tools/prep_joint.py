"""R39 JOINT prep (package docs/SERVER_PLAN_CN.md section 3), one dataset per call (--tree r39f | r39c):
 1. CC152K only: tokens_cc.npz = data.process_caption(vocab, caption, drop=False) of train/dev captions (native eval tokenisation,
    the R35 Flickr tokens.npz rule).
 2. For both peers of the TARGET parent: student encoders (r39ext.Student = native EncoderImage/EncoderText with the parent weights,
    BN running stats, dropout off, train-mode one-layer GRUs) encode every FIT and DEV row of the native eval layout; the native
    per-batch centring vector (128 sequential rows, per modality) is stored per batch: center_<split>_<kind>_peer<P>.npy.
 3. Equivalence vs the frozen native cache: student vector l2norm(e - mu[batch]) vs cached vector (max/mean abs, bitwise
    fraction), and the DEV native retrieval recomputed from the student vectors vs the cache.
 4. Functional check: 20 updates of one JOINT unit config (r39 trainer run_unit, scratch dir OUT/prep/<ds>/joint_smoke, no
    readout): step-0 cache equivalence with grad, update-1 student/frozen checks, deterministic kernels, seconds per update.
Usage: python -I -B tools/prep_joint.py --tree r39f|r39c [--data-path DIR (CC staged copy)] [--smoke-unit UNIT]"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tree', required=True, choices=('r39f', 'r39c'))
    ap.add_argument('--data-path', default=''); ap.add_argument('--smoke-unit', default=''); ap.add_argument('--smoke-steps', type=int, default=20)
    ap.add_argument('--smoke-only', action='store_true', help='keep the existing JOINT_PREP_MANIFEST.json / centring files; rerun only the functional check')
    a = ap.parse_args()
    sys.path.insert(0, str(W / a.tree))
    import torch
    import common as C
    from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, install_guards, guard_record, sha256, write_json, read_json, utc
    install_guards(role='gate')
    import train_unit as TU
    TU.set_determinism()
    import r39ext as R9
    from data35 import load_cond
    import evals as EV
    t_all = time.time(); dev = torch.device('cuda')
    pdir = C.R39_PREP / C.DS_KEY; pdir.mkdir(parents=True, exist_ok=True)
    files = {}
    if a.smoke_only:
        assert (pdir / 'JOINT_PREP_MANIFEST.json').exists()
    if not a.smoke_only:
        # ---------------------------------------------------------------- 1. tokens (CC152K)
        if C.DS_KEY == 'cc':
            if not C.TOKENS_FILE.exists():
                sys.path.insert(1, str(C.NATIVE_CODE))
                import data as D
                from vocab import deserialize_vocab
                vocab = deserialize_vocab(os.path.join(str(C.VOCAB), f'{C.DATA_NAME}_vocab.json')); vocab.add_word('<mask>')
                dp = a.data_path or str(C.DATA_SRC.parent)
                out = {}
                for split in ('train', 'dev'):
                    caps, _ = D.get_dataset(dp, C.DATA_NAME, split)
                    toks = [D.process_caption(vocab, c, False).long().numpy() for c in caps]
                    L = max(len(t) for t in toks); arr = np.zeros((len(toks), L), np.int64)
                    for k, t in enumerate(toks):
                        arr[k, :len(t)] = t
                    out[f'{split}_tok'] = arr; out[f'{split}_len'] = np.array([len(t) for t in toks], np.int64)
                assert len(out['train_len']) == N_TRAIN_CAP and len(out['dev_len']) == N_DEV_CAP
                ck = torch.load(C.PARENT_CKPT, map_location='cpu', weights_only=False)
                assert len(vocab) == int(ck['opt'].vocab_size) == ck['model_A'][1]['embed.weight'].shape[0]
                np.savez(C.TOKENS_FILE, **out)
            tok_rule = 'R39 prep: data.process_caption(vocab, caption, drop=False), native cc152k vocab + <mask>'
        else:
            tok_rule = 'R35 prep tokens.npz: data.process_caption(vocab, caption, drop=False) (read-only)'
        tokens = dict(path=str(C.TOKENS_FILE), sha256=sha256(C.TOKENS_FILE), rule=tok_rule)
        # ---------------------------------------------------------------- 2. centring vectors (parent encoders, native row layout)
        d = load_cond(C.TARGET_COND, dev)
        ck = R9.load_parent(dev)
        stud = R9.build_students(ck, dev)
        tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); dv = np.load(C.RAW_DEV_IMS, mmap_mode='r')
        T = np.load(C.TOKENS_FILE)
        ims = dict(fit=torch.from_numpy(np.ascontiguousarray(tr)), dev=torch.from_numpy(np.ascontiguousarray(dv)))   # dev: all rows (row r = image r // stride)
        tok = dict(fit=torch.from_numpy(T['train_tok']).to(dev), dev=torch.from_numpy(T['dev_tok']).to(dev))
        ln = dict(fit=torch.from_numpy(T['train_len']).to(dev), dev=torch.from_numpy(T['dev_len']).to(dev))
        nrow = dict(fit=N_TRAIN_CAP, dev=N_DEV_CAP)
        eq = {}; t_enc = time.time()
        for p in PEERS:
            sp = stud[p]
            for split in ('fit', 'dev'):
                with torch.no_grad():
                    # images: encode the distinct image rows of the native layout (FIT: train_ims row = image id; DEV: every dev_ims row)
                    src = ims[split]; n_src = src.shape[0]; E_img = torch.empty(n_src, 1024, device=dev)
                    for i in range(0, n_src, 512):
                        E_img[i:i + 512] = sp.enc_img(src[i:i + 512].to(dev))
                    E_cap = torch.empty(nrow[split], 1024, device=dev)
                    for i in range(0, nrow[split], 2048):
                        E_cap[i:i + 2048] = sp.enc_cap(tok[split][i:i + 2048], ln[split][i:i + 2048])
                rows = np.arange(nrow[split])
                row_img_src = rows // R9.PER_CAP if split == 'fit' else rows       # source row of the image of native row r
                ei = E_img[torch.as_tensor(row_img_src, device=dev)].double().cpu().numpy(); ec = E_cap.double().cpu().numpy()
                starts = np.arange(0, nrow[split], R9.NATIVE_BATCH); cnt = np.diff(np.append(starts, nrow[split]))[:, None]
                mu_i = (np.add.reduceat(ei, starts, axis=0) / cnt).astype(np.float32); mu_c = (np.add.reduceat(ec, starts, axis=0) / cnt).astype(np.float32)
                for kind, mu in (('img', mu_i), ('cap', mu_c)):
                    fn = pdir / f'center_{split}_{kind}_peer{p}.npy'; np.save(fn, mu); files[fn.name] = dict(sha256=sha256(fn), shape=list(mu.shape))
                # 3. equivalence of the student vector (item k) vs the cache
                n_img = N_TRAIN_IMG if split == 'fit' else N_DEV_IMG
                item_img_row = np.arange(n_img) * (1 if split == 'fit' else C.DEV_IMS_ROW_STRIDE)
                with torch.no_grad():
                    for kind, E, src_rows, n in (('img', E_img, item_img_row, n_img), ('cap', E_cap, np.arange(nrow[split]), nrow[split])):
                        ids = torch.arange(n, device=dev)
                        mu = torch.from_numpy(np.load(pdir / f'center_{split}_{kind}_peer{p}.npy')).to(dev)
                        u = R9.native_vse().l2norm(E[torch.as_tensor(src_rows, device=dev)] - mu[R9.batch_index(kind, ids)])
                        ref = d[(split, kind, p)]; dl = (u - ref).abs()
                        eq[f'{split}_{kind}_{p}'] = dict(n=int(n), max_abs=float(dl.max()), mean_abs=float(dl.mean()), frac_rows_bitwise=float((dl == 0).all(1).float().mean()),
                                                         min_cos=float((u * ref).sum(1).min()))
                        if split == 'dev':
                            eq.setdefault('_dev_u', {})[(kind, p)] = u
        # DEV native retrieval from the student vectors (peer mean of the 100x100-sharded native score)
        from data35 import shard_mm
        sims = sum(shard_mm(eq['_dev_u'][('img', p)], eq['_dev_u'][('cap', p)]) for p in PEERS).cpu().numpy() / 2
        m_stu, _, rk_stu = EV.full_metrics(sims); m_nat, _, rk_nat = EV.full_metrics(d['native_dev'])
        eq.pop('_dev_u')
        eq['dev_native_from_student_vectors'] = dict(rsum=m_stu['rsum'], native_rsum=m_nat['rsum'], max_abs_sim_diff=float(np.abs(sims - d['native_dev']).max()),
                                                     n_i2t_rank_diff=int((rk_stu['ranks_i'] != rk_nat['ranks_i']).sum()), n_t2i_rank_diff=int((rk_stu['ranks_t'] != rk_nat['ranks_t']).sum()))
        summ = dict(max_abs_over_all=max(v['max_abs'] for k, v in eq.items() if isinstance(v, dict) and 'max_abs' in v), encode_seconds=time.time() - t_enc)
        man = dict(line='R39 JOINT prep (package VCS_R38_Review_R39_Design_20261102 section 3)', created_utc=utc(), ds=C.DS_KEY, target_cond=C.TARGET_COND,
                   parent=dict(ckpt=str(C.PARENT_CKPT), model_best_sha256=sha256(C.PARENT_CKPT), epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum'])),
                   target_cache_manifest_sha256=d['manifest_sha256'], native_code=dict(dir=str(C.NATIVE_CODE), vse_sha256=sha256(C.NATIVE_CODE / 'vse.py')),
                   raw=dict(fit=str(C.RAW_FIT_IMS), dev=str(C.RAW_DEV_IMS), dev_row_stride=C.DEV_IMS_ROW_STRIDE),
                   tokens=tokens, files=files,
                   centring_rule='native forward_emb per-batch mean (128 sequential rows of the native eval layout, per modality) of the PARENT encoder output, '
                                 'stored per batch and held fixed for the student (registered compatibility patch)',
                   modes='BN eval (running stats), dropout off, caption BiGRU + both GPO GRUs in train mode (cuDNN RNN backward; 1 layer, dropout 0)',
                   equivalence=eq, equivalence_summary=summ, test_guard=guard_record(),
                   env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, deterministic=torch.are_deterministic_algorithms_enabled()))
        write_json(pdir / 'JOINT_PREP_MANIFEST.json', man)
        print('R39_JOINT_PREP equivalence', json.dumps(summ), json.dumps(eq['dev_native_from_student_vectors']), flush=True)
        del ims, tok, ln, stud
        torch.cuda.empty_cache()
    # ---------------------------------------------------------------- 4. functional check (scratch dir; not a formal unit)
    if a.smoke_unit:
        cfg = TU.cfg_of(a.smoke_unit); cfg['_config_sha256'] = sha256(C.ROOT / 'configs' / f'{a.smoke_unit}.json')
        assert cfg['encoder'] == 'joint_global' and not cfg['oracle']
        cfg['reference_unit'] = None   # scratch run stops at step 20: the R38 batch chain is only defined at the formal readout steps
        ctx = TU.Ctx(cfg['cond'], dev, need_local=False)
        if cfg['supervision'] == 'COMPLETE_XCHECK':
            ctx.ensure_completion()
        ctx.ensure_joint()
        sdir = pdir / 'joint_smoke' / a.smoke_unit
        import shutil
        shutil.rmtree(sdir, ignore_errors=True)
        t0 = time.time()
        r = TU.run_unit(dict(cfg), ctx, out_dir=sdir, steps=a.smoke_steps, ckpt_steps=[0, a.smoke_steps], no_readout=True)
        hist = r['hist']; sec = hist['train_seconds']
        smoke = dict(unit=a.smoke_unit, steps=a.smoke_steps, train_seconds=sec, seconds_per_update=sec / a.smoke_steps, wall=time.time() - t0,
                     step0_cache_equivalence=hist['r39'].get('step0_cache_equivalence'), one_step_check=hist['r39'].get('one_step_check'),
                     update1_student_grad_norm=hist['r39'].get('update1_student_grad_norm'), grad_split=hist['r39'].get('grad_split_windows'),
                     peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(dev)), gpu=torch.cuda.get_device_name(0), utc=utc(),
                     note='scratch functional check inside the prep job (user waived the standalone GPU gate); not a formal unit, never reused')
        write_json(pdir / 'JOINT_SMOKE.json', smoke)
        print('R39_JOINT_SMOKE seconds/update', smoke['seconds_per_update'], 'eq', json.dumps(smoke['step0_cache_equivalence'] and smoke['step0_cache_equivalence']['passed']),
              'one_step', json.dumps(smoke['one_step_check'] and smoke['one_step_check']['passed']), flush=True)
    print('R39_JOINT_PREP_DONE elapsed_s', time.time() - t_all)


if __name__ == '__main__':
    main()
