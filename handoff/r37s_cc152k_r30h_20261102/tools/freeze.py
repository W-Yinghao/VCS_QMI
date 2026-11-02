#!/usr/bin/env python3
"""Write code_r37s/INSTALL_MANIFEST.json, R37S_ADAPTATION.json and the frozen PROTOCOL.json (before any GPU job).
Refuses to overwrite a frozen PROTOCOL.json bound to different code."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code_r37s'
R30 = Path('/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101')
A1X = Path('/home/infres/yinwang/CS_QMI/handoff/r27_a1cross_20261030/code_a1x')
UTC = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = {p.name: sha(p) for p in sorted(CODE.glob('*.py'))}
r30 = {p.name: sha(p) for p in sorted((R30 / 'code_r30v11').glob('*.py'))}
changed = sorted(f for f in files if files[f] != r30.get(f))
assert sorted(files) == sorted(r30), 'file set differs from code_r30v11'
assert changed == ['r27_interface.py', 'r30_co_train.py', 'r30_run.py'], changed
assert files['r27_interface.py'] == sha(A1X / 'r27_interface.py'), 'r27_interface.py is not the R27-A1X file'
diff = subprocess.run(['diff', '-u', '-x', 'INSTALL_MANIFEST.json', '-x', '__pycache__', str(R30 / 'code_r30v11'), str(CODE)],
                      capture_output=True, text=True).stdout
(ROOT / 'R37S_ADAPTATION_DIFF.patch').write_text(diff)

man = dict(created_utc=UTC, base_snapshot=str(R30 / 'code_r30v11'), base_manifest_sha256=sha(R30 / 'code_r30v11' / 'INSTALL_MANIFEST.json'),
           files_sha256=files, changed_vs_base=changed)
(CODE / 'INSTALL_MANIFEST.json').write_text(json.dumps(man, indent=1) + '\n')

adapt = dict(
    label='R30-style adaptation (CC152K data adaptation only); NOT byte-identical to code_r30v11',
    base='code_r30v11 (R30 v1.1 + amendment A1 snapshot, gated 1080158)', base_files_sha256=r30, files_sha256=files,
    diff_file='R37S_ADAPTATION_DIFF.patch', diff_sha256=sha(ROOT / 'R37S_ADAPTATION_DIFF.patch'),
    unchanged_files=sorted(f for f in files if f not in changed),
    changes=[
        dict(file='r30_run.py', change="assert data_name == 'f30k_precomp' -> data_name in (f30k_precomp, cc152k_precomp)",
             reason='R30 entry refused any dataset but Flickr; the driver below it (r25_co_train / data.py / evaluation) is the IN2R code '
                    'that natively supports cc152k_precomp (im_div 1, per_captions 1, .tsv captions, noise_ratio forced 0 by r25_run).'),
        dict(file='r30_run.py', change="PEAK_<job>.json also records r27_interface.Ctx.gather_mode", reason='report-only (as R27-A1X r27_run.py)'),
        dict(file='r30_co_train.py', change="data_digests(): caption files train_caps.tsv / dev_caps.tsv for cc152k_precomp",
             reason='PARENT.json data digest would otherwise try to hash the non-existent Flickr .txt caption files (FileNotFoundError at the boundary save).'),
        dict(file='r30_co_train.py', change="dev_ranks() (E10 report-only rank pass): per_captions = 1 for cc152k_precomp (was hard-coded 5)",
             reason='Flickr 5-captions-per-image assumption; CC152K has exactly one caption per image. Same rule as r25_co_train.validate / evaluation.py.'),
        dict(file='r27_interface.py', change='byte-identical copy of R27-A1X code_a1x/r27_interface.py: Ctx.image_rows() host-gather of observed-image rows',
             reason='CC152K train features are 41.2 GiB fp32; the A1 re-encoding put the full array on the GPU. image_rows returns exactly the rows '
                    'gpu_images()[cols] returns (same bytes, same order); auto mode = GPU-resident iff array + 16 GiB <= GPU memory '
                    '(RTX PRO 6000 / H100: gpu, L40S: cpu). Gated bitwise for R27-A1X (GATE_A1X.json, Flickr gather 1077656).')],
    not_changed=['training recipe / options (R30 H parent + H0 continuation: warmup 5, K 5, i2t+t2i on, 40 co-train epochs, native hinge)',
                 'A1 noisy interface (other peer selects candidate identities from its own ID-bearing queue, current peer re-encodes)',
                 'GMM split / eval_train / LR schedule / DEV-best rule / save-resume code'],
    cc152k_structure='one image per caption: t2i_index = arange(150000), im_div 1, DEV 1000 images x 1000 captions; no record//5 and no '
                     'multi-positive assumption is used (observed-ID equality in report-only readouts reduces to the diagonal).',
    created_utc=UTC)
(ROOT / 'R37S_ADAPTATION.json').write_text(json.dumps(adapt, indent=1) + '\n')

pf = ROOT / 'PROTOCOL.json'
if pf.exists():
    old = json.loads(pf.read_text())
    if old.get('frozen') and old.get('code_files_sha256') != files:
        sys.exit('PROTOCOL.json frozen for another code snapshot; refuse to overwrite')
prot = dict(
    version='R37S-v1-20261102', frozen=True, frozen_utc=UTC, line='R37S same-recipe CC152K (R30-style H parent)',
    purpose='separate dataset effect from training-recipe effect: CC152K parent trained with the R30 v1.1 H recipe (vs R37C on the R31-H parent)',
    data=dict(data_name='cc152k_precomp', observed_pairs='real observed CC152K pairs, no injected noise (noise_ratio 0, noise_file none)',
              splits='train + dev only (node-local staged copies); TEST never staged, never read'),
    seed=1001,
    recipe=dict(code='code_r37s = code_r30v11 + R37S_ADAPTATION.json', init='seed-determined untrained init (no R26 prefix)',
                parent=dict(parent_id='wp_cc_s1001_H_e5', warmup_epochs=5, warmup_source='H (R29 h_native = native VSE warmup)',
                            boundary='R30 boundary: after warmup epoch 5 of both peers + full-DEV validation (parent_state.pt, PARENT.json)'),
                continuation=dict(unit_id='r37s_cc_s1001_H0', objective='H (native hinge, r30 scale native)', mode='fork from the parent',
                                  co_train_epochs=40, noisy_interface='A1 (arm a1_iface, lam 0)', k=5, i2t=True, t2i=True,
                                  milestone='E10 non-overwritable state_completed10.pt + E10_RECORD.json (as R30)'),
                hyper='r25_run defaults (batch 128, lr 5e-4, lr_update 15, margin 0.2, grad_clip 2, queue 65536, p_threshold 0.5, workers 0, fp32 no AMP)'),
    checkpoint_rule='model_best.pth.tar = co-train epoch (0..39) with the strictly highest peer-averaged full-DEV rSum (first maximum on ties; '
                    'best_rsum starts at 0 at the boundary, warmup epochs never eligible) -- the native r25_co_train rule used by R30 H0',
    hardware=dict(submit='--qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0',
                  block='GPU_BLOCK.json = GPU type of the first attempt; parent warmup, continuation and every resume attempt on that type'),
    resume='self-resubmission on timeout (same partition/time/nice), at most 4 attempts in total; continuation resumes its own state_last.pt; '
           'a warmup interrupted before the boundary save restarts the warmup from the seed (deterministic init)',
    test='no TEST evaluation registered or run by this line (main session decides with the user)',
    code_files_sha256=files, adaptation_sha256=sha(ROOT / 'R37S_ADAPTATION.json'))
pf.write_text(json.dumps(prot, indent=1) + '\n')
print('frozen', sha(pf), 'changed', changed)
