"""R43 B: R37 B-TRUST subsets of the three 80% IN2R-native Flickr parents (R33 H0 f80 s1001/s1002/s1003) + their R43 prep dirs (r43f tree).
Rule = R37 prep.trust_cond verbatim (R35 trust_n60 rule): native IN2R co_train.eval_train on the frozen two peers of the parent (python /
numpy / torch seeds 1001, native get_loader('warmup') with the parent's own noise_ratio / noise_file, workers 0), subset = mean(prob_A,
prob_B) >= 0.5, computed once, never updated. The native clean labels returned by get_loader are discarded unread. The native code is the R31
IN2R copy (co_train.py byte-identical to the R30 code R37 used; R33 code_h0 byte-identical to the R31 copy).
R43 prep dir of each parent = byte copy of its R41 / R42 candidate table (sha-verified against the source manifest) + its calibration_init /
candidates entries copied from the source manifest + trust_<cond>.npz. The R41 / R42 prep dirs are only read.
No test file is opened (TEST guard, role gate). Usage: python -B tools/r43_build_trust.py --data-path <staged dir containing f30k_precomp>"""
from __future__ import annotations
import argparse, os, random, shutil, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r43f'))
import common as C
from common import N_TRAIN_CAP, N_TRAIN_IMG, VOCAB, install_guards, guard_record, sha256, write_json, read_json, utc, assert_no_oracle_loaded

SOURCE_PREP = {'f80in': C.R41_PARENTS / 'prep', 'f80in_s1002': C.R42_PARENTS / 'prep' / 'f80in_s1002', 'f80in_s1003': C.R42_PARENTS / 'prep' / 'f80in_s1003'}


def trust_cond(cond, data_path, workers=0):
    """R37 prep.trust_cond with the parent dir / native code of the IN2R-native parent."""
    import torch
    sys.path.insert(1, str(C.CODE_IN2R))
    import co_train as N
    import vse as V
    import data as D
    from vocab import deserialize_vocab
    c = C.CONDS[cond]; hdir = Path(c['parent_dir'])
    ck = torch.load(hdir / 'model_best.pth.tar', map_location='cpu', weights_only=False)
    opt = ck['opt']; opt.data_path = data_path; opt.vocab_path = str(VOCAB); opt.workers = workers
    assert opt.noise_file == c['noise_file'] and float(opt.noise_ratio) == 0.8 and sha256(c['noise_file']) == c['noise_sha256']
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    models = []
    for key in ('model_A', 'model_B'):
        m = V.VSE(opt, vocab.word2idx); m.load_state_dict(ck[key]); models.append(m)
    caps, ims = D.get_dataset(opt.data_path, opt.data_name, 'train')
    random.seed(1001); np.random.seed(1001); torch.manual_seed(1001)
    loader, data_size, _clean_labels_discarded = D.get_loader(caps, ims, vocab, 'warmup', opt.batch_size, workers, opt.noise_ratio, opt.noise_file)
    del _clean_labels_discarded
    t0 = time.time()
    prob_A, prob_B, _ = N.eval_train(opt, models[0], models[1], loader, data_size, [[], []], None, 0)
    prob_A = np.asarray(prob_A, np.float64); prob_B = np.asarray(prob_B, np.float64)
    mean = (prob_A + prob_B) / 2
    subset = np.nonzero(mean >= 0.5)[0].astype(np.int64)
    obs = np.load(c['noise_file']).astype(np.int64)
    info = dict(n_records=int(data_size), n_subset=int(len(subset)), retention=float(len(subset) / data_size),
                n_images_in_subset=int(len(np.unique(obs[subset]))),
                captions_per_image_in_subset=dict(zip(*[x.tolist() for x in np.unique(np.bincount(obs[subset], minlength=N_TRAIN_IMG), return_counts=True)])),
                prob_A_mean=float(prob_A.mean()), prob_B_mean=float(prob_B.mean()), agreement_A_B_ge_half=float(((prob_A >= .5) == (prob_B >= .5)).mean()),
                seconds=time.time() - t0, seeds=dict(python=1001, numpy=1001, torch=1001), loader="native get_loader('warmup'): shuffle, region/caption augmentation, workers=%d" % workers,
                rule='native co_train.eval_train (eval_loss, min-max normalised, GaussianMixture(2,max_iter=10,tol=1e-2,reg_covar=5e-4), prob of low-mean component); subset = mean(prob_A,prob_B) >= 0.5; computed once, never updated',
                gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID'))
    assert data_size == N_TRAIN_CAP
    return dict(prob_A=prob_A, prob_B=prob_B, subset=subset), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-path', required=True); ap.add_argument('--conds', default='f80in,f80in_s1002,f80in_s1003')
    a = ap.parse_args()
    install_guards(role='gate'); assert_no_oracle_loaded()
    for cond in a.conds.split(','):
        c = C.CONDS[cond]; pdir = Path(c['prep_dir'])
        if (pdir / 'PREP_MANIFEST.json').exists():
            print('R43 prep already built', cond); continue
        pdir.mkdir(parents=True, exist_ok=True)
        src = SOURCE_PREP[cond]; sm = read_json(src / 'PREP_MANIFEST.json'); cname = f'candidates_{cond}.npz'
        assert sha256(src / cname) == sm['files'][cname]['sha256']
        shutil.copyfile(src / cname, pdir / cname); assert sha256(pdir / cname) == sm['files'][cname]['sha256']
        files = {cname: dict(sm['files'][cname], copied_from=str(src / cname))}
        T, info = trust_cond(cond, a.data_path)
        tname = c['trust_file']; np.savez(pdir / tname, **T); files[tname] = dict(sha256=sha256(pdir / tname), keys=sorted(T))
        man = dict(line='R43 B prep (R41/R42 candidate table + calibration copied byte-identical; R37 B-TRUST subset added)', created_utc=utc(),
                   source_prep_manifest=dict(path=str(src / 'PREP_MANIFEST.json'), sha256=sha256(src / 'PREP_MANIFEST.json')),
                   r35_prep_manifest_sha256=sm['r35_prep_manifest_sha256'], shared_from_r35=sm['shared_from_r35'],
                   calibration_init={cond: sm['calibration_init'][cond]}, candidates={cond: sm['candidates'][cond]}, trust={cond: info}, files=files,
                   test_guard=guard_record())
        write_json(pdir / 'PREP_MANIFEST.json', man)
        print('R43_TRUST_DONE', cond, {k: info[k] for k in ('n_subset', 'retention', 'n_images_in_subset', 'agreement_A_B_ge_half', 'seconds')}, flush=True)
    print('R43_BUILD_TRUST_DONE')


if __name__ == '__main__':
    main()
