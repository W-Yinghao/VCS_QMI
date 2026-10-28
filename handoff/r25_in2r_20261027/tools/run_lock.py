"""Write RUN_LOCK.json (source / patch / data / noise / GloVe / environment / config / roles)."""
import hashlib, json, os, subprocess, sys, glob, datetime

PKG = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027'
SRC = '/projects/EEG-foundation-model/yinghao/models/r25_src'
DATA = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data'


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(f'{PKG}/patches', exist_ok=True)
    patches = {}
    for name, orig in (('in2r_vse.py.diff', f'{SRC}/IN2R/vse.py'),):
        d = subprocess.run(['diff', '-u', orig, f'{PKG}/code/vse.py'], capture_output=True, text=True).stdout
        open(f'{PKG}/patches/{name}', 'w').write(d); patches[name] = hashlib.sha256(d.encode()).hexdigest()
    for f in ('train.py', 'run.py'):
        d = subprocess.run(['diff', '-u', '--strip-trailing-cr', f'{SRC}/SPS/code/{f}', f'{PKG}/sps_code/{f}'], capture_output=True, text=True).stdout
        open(f'{PKG}/patches/sps_{f}.diff', 'w').write(d); patches[f'sps_{f}.diff'] = hashlib.sha256(d.encode()).hexdigest()
    lock = {
        'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'design_package': {'path': '/home/infres/yinwang/CS_QMI/VCS_Retrieval_R25_Server_Design_20261027.zip',
                           'sha256': sha('/home/infres/yinwang/CS_QMI/VCS_Retrieval_R25_Server_Design_20261027.zip')},
        'author_sources': {'IN2R': {'repo': 'https://github.com/liuyyy111/IN2R', 'commit': '909e53293ec3c633823597b8215ce23a71796d69',
                                    'files': {os.path.basename(p): sha(p) for p in sorted(glob.glob(f'{SRC}/IN2R/*.py'))}},
                           'SPS': {'repo': 'https://github.com/C-TeaRanger/SPS', 'commit': 'b985abe9936df95fff6d7f2745c408b88336c93c',
                                   'files': {os.path.basename(p): sha(p) for p in sorted(glob.glob(f'{SRC}/SPS/code/*.py'))}},
                           'read_only': True},
        'r25_code': {os.path.relpath(p, PKG): sha(p) for p in sorted(glob.glob(f'{PKG}/code/*.py') + glob.glob(f'{PKG}/sps_code/*.py') + glob.glob(f'{PKG}/tools/*.py'))},
        'patches': patches,
        'data': {'root': DATA, 'files': {f: sha(f'{DATA}/data/f30k_precomp/{f}') for f in ('train_ims.npy', 'train_caps.txt', 'dev_ims.npy', 'dev_caps.txt')},
                 'vocab': {f: sha(f'{DATA}/vocab/{f}') for f in ('f30k_precomp_vocab.json',)},
                 'train': '29,000 images / 145,000 captions (FIT role; full training pool, no clean anchors)',
                 'dev': 'full DEV 1,014 images / 5,070 captions (dev_ims.npy has 5,070 rows: image repeated per caption); native IN2R validates on all of it',
                 'test': 'test_ims.npy/test_caps.txt NOT read and NOT staged to nodes; explicit tools/test_eval.py entry only after the lock',
                 'staging': 'each GPU job copies the four FIT/DEV files to node-local $TMPDIR and checks sizes (I/O only)'},
        'noise': json.load(open(f'{PKG}/NOISE_LOCK.json')),
        'glove': json.load(open(f'{PKG}/GLOVE_BUILD.json')),
        'config_code_version': {'batch_size': 128, 'embed_size': 1024, 'optimizer': 'AdamW(lr 5e-4, wd 1e-4) native', 'grad_clip': '2.0 on encoder params only (native; refiners excluded)',
                                'warmup_epochs': 5, 'main_epochs': 40, 'lr_update': 15, 'k': 5, 'i2t': True, 't2i': True, 'queue_size': 65536,
                                'p_threshold': 0.5, 'tau_sce': 0.05, 'precision': 'fp32, no AMP; torch defaults matmul TF32 off, cudnn TF32 on; cudnn.benchmark True (native)',
                                'workers': 0, 'checkpoint_rule': 'best full-DEV rSum of the peer-averaged similarity over co-training epochs 0..39 (native rule)'},
        'roles': {'FIT': 'train split only (noisy pool); C1/c_D calibration uses 32 FIT batches per peer at the common prefix',
                  'DEV': 'full DEV for epoch selection and critic selection', 'TEST': 'not read in this stage'},
        'environment': {'python': '/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python', 'torch': '2.9.0+cu128', 'sklearn': '1.7.2', 'nltk': '3.10.3', 'numpy': '2.2.6'},
    }
    json.dump(lock, open(f'{PKG}/RUN_LOCK.json', 'w'), indent=1)
    print('written', len(lock['r25_code']), 'code files')


if __name__ == '__main__':
    main()
