"""Build the frozen TEST manifest (no TEST file is opened here).

Usage:
  build_manifest.py --initial      # writes and freezes TEST_MANIFEST.json (all finished units); only once
  build_manifest.py --increment    # writes and freezes TEST_MANIFEST_INC_<utc>.json with the finished units whose checkpoint
                                   # sha256 is in no earlier frozen manifest; exit code 3 (nothing written) if there are none
Freezing = chmod 444 + a 'manifest_frozen' line (path, sha256) appended to TEST_ACCESS_LOG.jsonl. Earlier manifests are
the 'manifest_frozen' entries of that log (each re-verified by sha256). Run roots come from ROOTS.json.

A unit = a run dir with DONE.json and model_best.pth.tar under the line roots below (smoke / prefix / calib / parent dirs
are listed as skipped). Selection is the existing DEV selection: model_best.pth.tar = best full-DEV epoch written during
training. Exception (pre-recorded rule, R25 FULLDEV.json of 2026-10-29): SPS S0 uses the full-DEV best checkpoint_37,
because SPS's own model_best.pth.tar was selected on its native 200-image DEV subset.
Per unit: checkpoint sha256, checkpoint epoch / best_rsum, DONE.json, the logged DEV entry of that epoch (six recalls),
training RUN_*.json environments and recorded code sha256, the training snapshot(s) matching those sha256, and whether
the evaluation copy (eval_code/) is byte-identical for every evaluation-relevant file.
"""
import os, sys, glob, json, hashlib, re, time, datetime, io

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDOFF = '/home/infres/yinwang/CS_QMI/handoff'
DATA = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data'
VOCAB = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab'
GLOVE = '/projects/EEG-foundation-model/yinghao/models/glove840b/glove840b_r25_subset.npz'
EVAL_FILES = ('evaluation.py', 'vse.py', 'model.py', 'data.py', 'vocab.py', 'utils.py', 'glove_shim.py')
NRCS_DIR = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
NRCS_EVAL_FILES = ('nrcs/test_entry.py', 'nrcs/gpo_adapter.py', 'upstream/vse.py', 'upstream/glove_shim.py',
                   'upstream/evaluation_909e5329_orig.py', 'package/src/native_ratio/models.py')
SPS_EVAL_FILES = ('evaluation.py', 'model.py', 'data.py', 'vocab.py', 'utils.py', 'InfoNCELoss.py')
# author-original IN2R data.py (commit 909e5329); differs from the eval copy only in the training-time caption-drop branch
KNOWN_DATA_PY = {'80b7d497be1f2df2d90d4695888caf36642de2146f8e865abd655d8b6d0865d8':
                 'author-original IN2R data.py at training; the evaluation copy differs only in process_caption(drop=True) '
                 '(training-time token deletion, R25 patches/in2r_data.py.diff); dev/test loading uses drop=False'}

REG = json.load(open(os.path.join(HERE, 'ROOTS.json')))
OUT9 = REG['outputs_root']
LINES = REG['lines']
ROOTS = sorted({e['glob'].split('/')[0] for e in LINES})
EXCLUDE_ARMS = REG['exclude_arms']
SNAPSHOT_GLOBS = REG['snapshot_globs']


def sha_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def append_log(path, rec):
    import fcntl
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + '\n'); f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def parse_cell(cell):
    m = re.match(r'^(f|c|cc)(\d\d)?_s(\d+)$', cell)
    if not m:
        return None
    kind, noise, seed = m.group(1), m.group(2), int(m.group(3))
    if kind == 'cc':
        return dict(dataset='cc152k', data_name='cc152k_precomp', noise='real', seed=seed)
    return dict(dataset='f30k' if kind == 'f' else 'coco', data_name='f30k_precomp' if kind == 'f' else 'coco_precomp',
                noise=int(noise), seed=seed)


def snapshots():
    out = {}
    for g in SNAPSHOT_GLOBS:
        for d in sorted(glob.glob(os.path.join(HANDOFF, g))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, 'evaluation.py')):
                out[d] = {os.path.basename(p): sha_file(p) for p in glob.glob(os.path.join(d, '*.py'))}
    return out


def match_snapshots(recorded, snaps):
    full = [d for d, fs in snaps.items() if all(fs.get(k) == v for k, v in recorded.items())]
    if full:
        return full, {}
    best = max(snaps, key=lambda d: sum(snaps[d].get(k) == v for k, v in recorded.items()))
    return [], dict(closest=best, mismatched=sorted(k for k, v in recorded.items() if snaps[best].get(k) != v))


def logged_dev(run_dir, ck_epoch, ck_best):
    e = json.load(open(os.path.join(run_dir, 'EPOCHS.json')))
    co = e['cotrain'] if isinstance(e, dict) else e
    ent = [x for x in co if x.get('epoch') == ck_epoch]
    note = 'EPOCHS.json cotrain entry with epoch == checkpoint epoch'
    if not ent or abs(ent[-1]['dev']['rsum'] - ck_best) > 1e-9:
        ent2 = [x for x in co if abs(x['dev']['rsum'] - ck_best) <= 1e-9]
        if ent2:
            ent, note = ent2, 'EPOCHS.json cotrain entry whose rsum == checkpoint best_rsum (epoch field did not match)'
    if not ent:
        return None, 'no EPOCHS.json entry matches checkpoint epoch/best_rsum', None
    x = ent[-1]
    best_all = max(x2['dev']['rsum'] for x2 in co)
    keep = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum', 'peer0_rsum', 'peer1_rsum', 'n_images', 'n_captions')
    d = {k: x['dev'][k] for k in keep if k in x['dev']}
    for k in ('epoch', 'raw_epoch', 'completed_cotrain_epochs'):
        if k in x:
            d['entry_' + k] = x[k]
    return d, note, dict(n_cotrain_entries=len(co), max_cotrain_dev_rsum=best_all)


def run_records(run_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(run_dir, 'RUN_*.json'))):
        r = json.load(open(p))
        env = r.get('env', {})
        recs.append(dict(file=os.path.basename(p), gpu=env.get('gpu'), host=env.get('host'), job=env.get('job'),
                         torch=env.get('torch'), code_sha256=env.get('code_sha256') or r.get('code_sha256') or {}))
    return recs


def r30_label(run_dir):
    p = os.path.join(run_dir, 'R30_IDENTITY.json')
    if not os.path.exists(p):
        return None, {}
    u = json.load(open(p))['unit']
    lab = f"{u['warmup_source']}->{u['continuation_objective']}"
    if u.get('scale_rule') not in ('native', '1', None, ''):
        lab += f"(scale={u['scale_rule']})"
    lab += f" [{u['co_train_epochs']}ep]"
    return lab, {k: u.get(k) for k in ('unit_id', 'warmup_source', 'continuation_objective', 'scale_rule', 'co_train_epochs',
                                        'execution_class', 'stage_role', 'parent_id')}


def main():
    import torch
    addendum = '--increment' in sys.argv
    assert addendum or '--initial' in sys.argv, 'use --initial or --increment'
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_path = os.path.join(HERE, f'TEST_MANIFEST_INC_{stamp}.json' if addendum else 'TEST_MANIFEST.json')
    log_p = os.path.join(HERE, 'TEST_ACCESS_LOG.jsonl')
    prior_shas, prior_stat, prior_manifests = set(), set(), []
    if os.path.exists(log_p):
        for line in open(log_p):
            ev = json.loads(line)
            if ev.get('event') == 'manifest_frozen':
                mp = ev['manifest']
                assert sha_file(mp) == ev['sha256'], f'frozen manifest changed: {mp}'
                prior_manifests.append(dict(file=mp, sha256=ev['sha256']))
                for pu in json.load(open(mp))['units']:
                    prior_shas.add(pu['checkpoint_sha256'])
                    prior_stat.add((pu['checkpoint'], pu['checkpoint_bytes'], pu.get('checkpoint_mtime_ns')))
    if addendum:
        assert prior_manifests, 'addendum requires an earlier frozen manifest'
    elif prior_manifests:
        raise SystemExit('a manifest is already frozen; use --increment')

    eval_code = {f: sha_file(os.path.join(HERE, 'eval_code', f)) for f in EVAL_FILES}
    sps_code = {f: sha_file(os.path.join(HERE, 'eval_code_sps', f)) for f in SPS_EVAL_FILES}
    nrcs_code = {f: sha_file(os.path.join(NRCS_DIR, f)) for f in NRCS_EVAL_FILES}
    snaps = snapshots()
    units, skipped = [], []
    claimed = set()
    for e in LINES:
        line, g, oracle, source = e['line'], e['glob'], e['oracle'], e['source']
        for d in sorted(glob.glob(os.path.join(OUT9, g))):
            if not os.path.isdir(d):
                continue
            ck = os.path.join(d, 'model_best.pth.tar')
            if e['layout'] == 'r30flat':
                m = re.match(r'^r30[a-z0-9]*_(f\d\d_s\d+)_(\w+)$', os.path.basename(d))
                cell, arm = (m.group(1), m.group(2)) if m else (None, os.path.basename(d))
            else:
                cell, arm = os.path.basename(os.path.dirname(d)), os.path.basename(d)
            if not os.path.exists(ck):
                continue
            claimed.add(ck)
            if arm in EXCLUDE_ARMS:
                skipped.append(dict(path=d, reason=EXCLUDE_ARMS[arm])); continue
            if parse_cell(cell or '') is None:
                skipped.append(dict(path=d, reason=f'not a formal cell dir ({cell})')); continue
            if not os.path.exists(os.path.join(d, 'DONE.json')):
                skipped.append(dict(path=d, reason='no DONE.json (still training / not finished)')); continue
            units.append(dict(line=line, cell=cell, arm=arm, run_dir=d, checkpoint=ck, oracle_diagnostic=oracle, source=source, host=e['host'], registry_snapshot=e.get('snapshot')))
    # SPS S0 external reference (R25 dir): full-DEV-selected checkpoint per FULLDEV.json
    fx = REG['fixed_units'][0]
    sps_dir = os.path.join(OUT9, fx['run_dir'])
    fd = json.load(open(os.path.join(sps_dir, 'FULLDEV.json')))
    sps_ck = os.path.join(sps_dir, f"checkpoint_{fd['best']['epoch']}.pth.tar")
    claimed.add(os.path.join(sps_dir, 'model_best.pth.tar'))
    skipped.append(dict(path=os.path.join(sps_dir, 'model_best.pth.tar'),
                        reason='SPS native model_best (selected on SPS native 200-image DEV subset); replaced by the full-DEV '
                               f"selected checkpoint_{fd['best']['epoch']} recorded in FULLDEV.json (rule: max full-DEV rSum, earliest tie)"))
    units.append(dict(line='R25-SPS', cell='f40_s1001', arm='S0_sps', run_dir=sps_dir, checkpoint=sps_ck, oracle_diagnostic=False,
                      source='r25', host='sps'))
    # every other model_best under the roots
    for r in ROOTS:
        for p in sorted(glob.glob(os.path.join(OUT9, r, '**', 'model_best.pth.tar'), recursive=True)):
            if p not in claimed:
                skipped.append(dict(path=os.path.dirname(p), reason='smoke / gate / parent / diagnostic dir (not a formal unit)'))

    out_units = []
    for u in units:
        t0 = time.time()
        st = os.stat(u['checkpoint'])
        if (u['checkpoint'], st.st_size, st.st_mtime_ns) in prior_stat:
            continue  # unchanged file already in a frozen manifest (path, size, mtime); no re-hash
        u['checkpoint_mtime_ns'] = st.st_mtime_ns
        raw = open(u['checkpoint'], 'rb').read()
        u['checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()
        u['checkpoint_bytes'] = len(raw)
        if u['checkpoint_sha256'] in prior_shas:
            continue
        ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        del raw
        opt = vars(ckpt['opt'])
        u['checkpoint_epoch'] = int(ckpt['epoch'])
        u['checkpoint_best_rsum'] = float(ckpt.get('best_rsum', float('nan')))
        u['checkpoint_keys'] = sorted(k for k in ckpt if k != 'opt')
        u['opt'] = {k: opt.get(k) for k in ('data_name', 'noise_ratio', 'seed', 'batch_size', 'vocab_path', 'num_epochs', 'embed_size',
                                            'noise_file', 'arm', 'module_name')}
        cellinfo = parse_cell(u['cell'])
        u.update({k: cellinfo[k] for k in ('dataset', 'noise', 'seed')})
        checks = dict(data_name_matches_cell=opt.get('data_name') == cellinfo['data_name'],
                      seed_matches_cell=int(opt.get('seed', -1)) == cellinfo['seed'])
        if cellinfo['noise'] != 'real':
            checks['noise_matches_cell'] = abs(float(opt.get('noise_ratio', -1)) * 100 - cellinfo['noise']) < 1e-6
        if u['host'] == 'in2r':
            done = json.load(open(os.path.join(u['run_dir'], 'DONE.json')))
            u['done'] = done
            checks['done_best_equals_checkpoint_best'] = abs(done['best_rsum'] - u['checkpoint_best_rsum']) < 1e-9
            dev, note, extra = logged_dev(u['run_dir'], u['checkpoint_epoch'], u['checkpoint_best_rsum'])
            u['logged_dev'], u['logged_dev_source'], u['logged_dev_context'] = dev, note, extra
            checks['logged_entry_found'] = dev is not None
            if dev is not None and extra:
                checks['selected_epoch_is_max_dev'] = abs(extra['max_cotrain_dev_rsum'] - dev['rsum']) < 1e-9
            u['training_runs'] = run_records(u['run_dir'])
            recs = [r['code_sha256'] for r in u['training_runs'] if r['code_sha256']]
            u['training_snapshot'] = []
            ev = {}
            for r in u['training_runs']:
                if r['code_sha256']:
                    full, partial = match_snapshots(r['code_sha256'], snaps)
                    u['training_snapshot'].append(dict(run=r['file'], matches=full, **partial))
                    for f in EVAL_FILES:
                        rec = r['code_sha256'].get(f)
                        if rec is None:
                            ev.setdefault(f, 'not recorded')
                        elif rec == eval_code[f]:
                            ev.setdefault(f, 'identical')
                        elif f == 'data.py' and rec in KNOWN_DATA_PY:
                            ev[f] = KNOWN_DATA_PY[rec]
                        else:
                            ev[f] = f'DIFFERENT (recorded {rec})'
            u['eval_code_vs_training'] = ev
            checks['eval_code_ok'] = bool(recs) and not any(str(v).startswith('DIFFERENT') for v in ev.values())
            if os.path.exists(os.path.join(u['run_dir'], 'R30_IDENTITY.json')):
                lab, ident = r30_label(u['run_dir'])
                u['arm_label'], u['r30_identity'] = lab, ident
            u['epochs_done'] = done.get('epochs_done')
        elif u['host'] == 'nrcs':  # NRCS-v1 GPO two-tower views (r32_native_ratio_20261101/nrcs/test_entry.py views)
            done = json.load(open(os.path.join(u['run_dir'], 'DONE.json')))
            u['done'] = done
            checks['done_best_equals_checkpoint_best'] = abs(done['best_rsum'] - u['checkpoint_best_rsum']) < 1e-9
            u['logged_dev'], u['logged_dev_source'] = done['logged_dev'], done['logged_dev_source']
            u['logged_dev_context'] = dict(checkpoint_rule=done['checkpoint_rule'], source_checkpoint_sha256=done['source_checkpoint_sha256'])
            checks['logged_entry_found'] = True
            checks['source_checkpoint_sha256_matches'] = sha_file(ckpt['source_checkpoint']) == done['source_checkpoint_sha256']
            u['training_runs'] = [dict(file='DONE.json', gpu=done.get('gpu_model'))]
            u['eval_code_vs_training'] = 'NRCS evaluator files (manifest eval_code_nrcs) are the training adapter files'
            checks['eval_code_ok'] = all(sha_file(os.path.join(NRCS_DIR, f)) == s for f, s in nrcs_code.items())
            u['epochs_done'] = done['epochs_done']
        else:  # SPS
            row = [r for r in fd['rows'] if r['epoch'] == u['checkpoint_epoch']][-1]
            u['logged_dev'] = {k: row[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum', 'n_images', 'n_captions')}
            u['logged_dev_source'] = 'FULLDEV.json row of this epoch (tools/sps_fulldev_eval.py, written 2026-10-29)'
            u['logged_dev_context'] = dict(full_dev_best=fd['best'], rule=fd['rule'], native_best_rsum_200img=row.get('native_best_rsum'))
            checks['selected_epoch_is_max_dev'] = fd['best']['epoch'] == u['checkpoint_epoch']
            u['training_runs'] = [dict(file='GPU.txt', gpu=open(os.path.join(u['run_dir'], 'GPU.txt')).read().strip().splitlines()[-1])]
            u['eval_code_vs_training'] = 'SPS sps_code snapshot (R25 dir) copied byte-identical to eval_code_sps/'
            checks['eval_code_ok'] = all(sha_file(os.path.join(HANDOFF, 'r25_in2r_20261027/sps_code', f)) == s for f, s in sps_code.items())
            u['epochs_done'] = 40
        u['checks'] = checks
        u['all_checks_pass'] = all(checks.values())
        u['unit_id'] = f"{u['line']}/{u['cell']}/{u['arm']}"
        u['manifest_seconds'] = round(time.time() - t0, 2)
        out_units.append(u)
        print(u['unit_id'], u['checkpoint_sha256'][:12], u['checkpoint_epoch'], round(u['checkpoint_best_rsum'], 3), u['all_checks_pass'], flush=True)
        del ckpt

    split_files = {}
    for dn, splits in (('f30k_precomp', ('dev', 'test')), ('coco_precomp', ('dev', 'testall')), ('cc152k_precomp', ('dev', 'test'))):
        ext = 'tsv' if dn == 'cc152k_precomp' else 'txt'
        for s in splits:
            for f in (f'{s}_caps.{ext}', f'{s}_ims.npy'):
                p = os.path.join(DATA, dn, f)
                split_files[f'{dn}/{f}'] = dict(path=p, bytes=os.stat(p).st_size)
    man = dict(
        kind='increment' if addendum else 'initial',
        created_utc=datetime.datetime.utcnow().isoformat() + 'Z', job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
        authorization='user 2026-11-01: 「现在就准备好，我们看看在测试集上我们现在各种方法，最好的结果是怎么样的，runfill提交」 (one-shot TEST read of DEV-selected checkpoints)',
        prior_manifests=prior_manifests,
        selection_rule='checkpoint = existing model_best.pth.tar = best full-DEV epoch written during training (SPS S0: full-DEV best '
                       'checkpoint from FULLDEV.json). No checkpoint / epoch / method / hyper-parameter is chosen with TEST.',
        protocol=dict(
            similarity='peer-averaged: (sim_A + sim_B) / 2, VSE forward_sim (dot product of batch-centred, L2-normalised embeddings); '
                       'encode with the checkpoint opt.batch_size, shuffle off (host evaluation.encode_data)',
            dev_recompute='identical to r25_co_train.validate: encode_data per peer, every per_captions-th image row, '
                          'shard_attn_scores(shard_size=100), i2t/t2i of the peer mean; gate: max |recomputed - logged| <= 0.1 pp over the six recalls',
            f30k='TEST = test split (1,000 images / 5,000 captions), evalrank(split="test", fold5=False) equivalent, shard 1000',
            coco='TEST = testall (5,000 images / 25,000 captions): fold5=True (five 1K folds of rows [5000i, 5000(i+1)), images every 5th, '
                 'mean of per-fold recalls and rSum) and fold5=False (5K); one encode of testall per peer as in evalrank, shard 1000',
            cc152k='TEST = test split (1,000 images / 1,000 captions), per_captions = 1, shard 1000',
            sps='SPS S0: SPS own code (SGRAF, SGR similarity), same peer-averaged protocol; DEV recompute = tools/sps_fulldev_eval.py path'),
        eval_code=dict(dir=os.path.join(HERE, 'eval_code'), sha256=eval_code,
                       source='/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code (byte-identical evaluation.py / vse.py / model.py / '
                              'data.py / vocab.py / utils.py / glove_shim.py in every R25-R30 snapshot)'),
        eval_code_nrcs=dict(dir=NRCS_DIR, sha256=nrcs_code, note='NRCS-v1 GPO two-tower evaluator (host nrcs); pointwise, frozen BN, no batch centering'),
        eval_code_sps=dict(dir=os.path.join(HERE, 'eval_code_sps'), sha256=sps_code,
                           source='/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/sps_code'),
        glove_npz=dict(path=GLOVE, bytes=os.stat(GLOVE).st_size, note='only used to construct EncoderText before load_state_dict'),
        vocab_dir=VOCAB,
        data=dict(root=DATA, files=split_files, note='TEST files are only stat()ed here; their sha256 is computed by the evaluation jobs after the access-log entry'),
        snapshots_scanned=sorted(snaps),
        units=out_units,
        skipped=skipped if not addendum else [s for s in skipped if 'no DONE' in s['reason']],
    )
    if addendum and not out_units:
        print('no new finished units; nothing written', flush=True)
        sys.exit(3)
    man['registry'] = dict(file=os.path.join(HERE, 'ROOTS.json'), sha256=sha_file(os.path.join(HERE, 'ROOTS.json')))
    tmp = out_path + '.tmp'
    json.dump(man, open(tmp, 'w'), indent=1)
    os.replace(tmp, out_path)
    os.chmod(out_path, 0o444)
    append_log(log_p, dict(event='manifest_frozen', utc=datetime.datetime.utcnow().isoformat() + 'Z', manifest=out_path,
                           sha256=sha_file(out_path), kind=man['kind'], n_units=len(out_units), job=os.environ.get('SLURM_JOB_ID'),
                           note='frozen before any TEST file read for these units'))
    print(f'wrote+froze {out_path}: {len(out_units)} units, {len(man["skipped"])} skipped', flush=True)


if __name__ == '__main__':
    main()
