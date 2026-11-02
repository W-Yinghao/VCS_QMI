"""R37C parent binding (CPU, reads the R31 parent dir read-only) -> ROOT/PARENT_BINDING.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'r37c'))
from common import ROOT, CONDS, CODE_NATIVE, R31_ROOT, sha256, read_json, write_json, utc

c = CONDS['cc']; P = Path(c['parent_dir'])
run = read_json(P / c['run_json']); done = read_json(P / 'DONE.json'); ep = read_json(P / 'EPOCHS.json')
best = max(ep['cotrain'], key=lambda e: e['dev']['rsum'])
att = [l.rstrip('\n').split('\t') for l in open(P / 'ATTEMPTS.tsv')]
code_now = {f: sha256(CODE_NATIVE / f) for f in run['env']['code_sha256']}
write_json(ROOT / 'PARENT_BINDING.json', dict(
    created_utc=utc(), line='R37C (CC152K block of R37, registered as its own line); development experiment, no TEST read',
    user_decision='2026-11-02: CC152K six units use R31 H0_default/cc_s1001 as the common frozen parent; no new R30-H training; R27 A1X kept as a separate strong-host check (not run now, not mixed)',
    series_note='DIFFERENT SERIES from the Flickr R30-H parents of R36/R37 (R30 a1_iface H arm, 5-epoch H warmup + 40 co-train, R30 code_r30v11): this parent is the R31 IN2R native-fidelity run with the author run.py defaults (warmup 1, K 1, i2t/t2i off), R31 code = byte copy of R25 IN2R code (author IN2R-code 909e5329). Gains are reported relative to this parent only; no absolute rSum comparison with Flickr.',
    conditions={'cc': dict(parent_unit=c['h_unit'], parent_dir=str(P), parent_seed=1001, dataset='CC152K', noise='real (uninjected); noise_ratio 0, noise_file none',
                           model_best=str(P / 'model_best.pth.tar'), model_best_sha256=sha256(P / 'model_best.pth.tar'),
                           run_json=str(P / c['run_json']), run_json_sha256=sha256(P / c['run_json']),
                           recipe={k: run['opt'][k] for k in ('warmup_epoch', 'k', 'i2t', 't2i', 'num_epochs', 'lr_update', 'learning_rate', 'batch_size', 'noise_ratio', 'noise_file', 'noisy_obj', 'seed', 'p_threshold', 'soft_margin', 'queue_size')},
                           num_epochs=run['opt']['num_epochs'], epochs_done=done['epochs_done'], status=done['status'],
                           best_rsum_logged=done['best_rsum'], best_epoch=best['epoch'], best_epoch_dev=best['dev'],
                           train_attempts=[dict(attempt=a[0], utc=a[1], job=a[2], host=a[4], gpu=a[5], start=a[6]) for a in att],
                           best_epoch_validated_on='attempt 2 (epochs resumed from state_last on RTX PRO 6000 node61; epoch %d >= resume point)' % best['epoch'],
                           code_dir=str(CODE_NATIVE), code_sha256_equal_to_run=all(code_now[f] == h for f, h in run['env']['code_sha256'].items()),
                           r31_code_manifest=str(R31_ROOT / 'CODE_SHA256.txt'),
                           cache_dir=c['cache_dir'], prep_dir=c['prep_dir'],
                           artifacts='R37C cache (native DEV reproduction required exactly) + R37C prep (locks rebuilt for 150000 records, calibration, top-32 tables, B-TRUST); nothing carried over from Flickr')},
    not_used=dict(R30_H='no CC152K R30-H parent exists; no extra R30-H training submitted (user)', A1X='R27 A1X CC152K parents: separate later block only if this one is positive; not run now')))
print('PARENT_BINDING written')
