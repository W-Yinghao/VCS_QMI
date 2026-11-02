"""Write PARENT_BINDING.json (recipe comparison of the candidate parents vs the R36 H parent r30_f60_s1001_H0) and
OLD_TASKS_AND_DEDUP.json. Reads RUN/EPOCHS/DONE/FORK json + hashes model_best; no data, no TEST."""
import csv, glob, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'r37'))
from common import ROOT, CONDS, R30_RUNS, write_json, read_json, utc, sha256

O9 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9')
IGNORE = {'data_path', 'output_dir', 'seed', 'r30_parent', 'r30_parent_fingerprint', 'r30_parent_id', 'r30_parent_sha', 'resume', 'r30_unit_id', 'r30_require_gpu'}


def run_of(d):
    f = sorted(glob.glob(str(d / 'RUN_*.json')))[0]; return f, read_json(f)


def main():
    ref_dir = R30_RUNS / 'r30_f60_s1001_H0'; ref_f, ref = run_of(ref_dir)
    out = dict(created_utc=utc(), reference=dict(unit='r30_f60_s1001_H0', run_json=ref_f, note='R36 H parent (Flickr60 seed1001)'), conditions={}, cc152k={})
    for cond, c in CONDS.items():
        d = R30_RUNS / c['h_unit']; f, r = run_of(d)
        diff = {k: [ref['opt'].get(k), r['opt'].get(k)] for k in set(ref['opt']) | set(r['opt']) if ref['opt'].get(k) != r['opt'].get(k)}
        recipe_diff = {k: v for k, v in diff.items() if k not in IGNORE}
        par = read_json(d / 'FORK_REPORT.json'); ep = read_json(d / 'EPOCHS.json'); done = read_json(d / 'DONE.json')
        best = max(ep['cotrain'], key=lambda x: x['dev']['rsum'])
        pw = Path(r['opt']['r30_parent']).parent; pf, prun = run_of(pw)
        pref = run_of(Path(ref['opt']['r30_parent']).parent)[1]
        pdiff = {k: [pref['opt'].get(k), prun['opt'].get(k)] for k in set(pref['opt']) | set(prun['opt']) if pref['opt'].get(k) != prun['opt'].get(k)}
        out['conditions'][cond] = dict(
            parent_unit=c['h_unit'], parent_seed=c['parent_seed'], model_best=str(d / 'model_best.pth.tar'), model_best_sha256=sha256(d / 'model_best.pth.tar'),
            run_json=f, objective=r['opt'].get('r30_objective'), arm=r['opt']['arm'], source_warmup=r['opt'].get('r30_source_warmup'), mode=r['opt'].get('r30_mode'),
            num_epochs=r['opt']['num_epochs'], epochs_done=done['epochs_done'], best_rsum_logged=done['best_rsum'], best_epoch=best['epoch'],
            noise_file=r['opt']['noise_file'], noise_sha256=sha256(r['opt']['noise_file']), train_gpu=r['env']['gpu'],
            code_sha256_equal_to_reference=r['env']['code_sha256'] == ref['env']['code_sha256'],
            opt_differences_vs_reference=diff, recipe_differences_vs_reference=recipe_diff,
            warmup_parent=dict(dir=str(pw), run_json=pf, gpu=prun['env'].get('gpu'), code_equal_to_reference_warmup=prun['env']['code_sha256'] == pref['env']['code_sha256'],
                               opt_differences_vs_reference_warmup={k: v for k, v in pdiff.items() if k not in IGNORE}, fork_components_equal=par.get('components_equal')),
            same_series=(not recipe_diff and r['env']['code_sha256'] == ref['env']['code_sha256'] and prun['env']['code_sha256'] == pref['env']['code_sha256']
                         and not {k: v for k, v in pdiff.items() if k not in IGNORE} and done['epochs_done'] == 40),
            cache_dir=c['cache_dir'], prep_dir=c['prep_dir'],
            artifacts='R34 cache + R35 prep reused read-only (R36 binding)' if cond == 'n60' else 'new R37 cache (L40S) + R37 prep (job r37_prep)')
    # CC152K candidate
    cd = O9 / 'r27_a1cross_20261030/runs/cc_s1001/a1_iface'; f, r = run_of(cd); ep = read_json(cd / 'EPOCHS.json')
    diff = {k: [ref['opt'].get(k), r['opt'].get(k)] for k in set(ref['opt']) | set(r['opt']) if ref['opt'].get(k) != r['opt'].get(k)}
    code_diff = sorted(k for k in set(ref['env']['code_sha256']) | set(r['env']['code_sha256']) if ref['env']['code_sha256'].get(k) != r['env']['code_sha256'].get(k))
    core = ('co_train.py', 'data.py', 'evaluation.py', 'model.py', 'vse.py', 'noisy_loss.py', 'train.py')
    out['cc152k'] = dict(
        candidate=str(cd), run_json=f, model_best_sha256=sha256(cd / 'model_best.pth.tar'), best_rsum_logged=read_json(cd / 'DONE.json')['best_rsum'],
        arm=r['opt']['arm'], resumed_from=ep['resumed_from'], opt_differences_vs_reference={k: v for k, v in diff.items() if k not in ('data_path', 'output_dir')},
        code_files_differing_from_reference=code_diff, core_host_files_identical=all(ref['env']['code_sha256'][k] == r['env']['code_sha256'][k] for k in core),
        cotrain_epoch0_train_record=ep['cotrain'][0].get('train'),
        findings=[
            'lineage: R26 common prefix (r26_dist_20261027/prefix/cc_s1001: 5 warmup epochs + co-train epoch 0 run by the R26 prefix job) -> R27 code_a1x a1_iface epochs 1-39; '
            'the R30 H series instead forks a 5-epoch H warmup boundary (wp_*_H_e5) into 40 co-train epochs of arm a1_iface with r30_objective H (code_r30v11)',
            'co-train epoch 0 of the CC model is the prefix epoch (logged i2t_no_proto/t2i_no_proto counters, the R26 prefix host), not an a1_iface/R30-H epoch',
            'code snapshot differs (code_a1x: r27_interface.py/r27_run.py differ, adds r27x_* files; no r30_* files), core IN2R host files identical',
            'opt lacks every r30_* field (r30_objective/r30_source_warmup/r30_mode/r30_scale/r30_tau)'],
        same_series=False,
        decision='NOT bound. Package C rule: if a same-series complete parent is unavailable, report the missing parent and the needed new parent training; '
                 'do not substitute a different state. The 6 CC152K units are not run; the 18 Flickr units are submitted.',
        needed=dict(
            parent_training='CC152K seed1001 R30-series H parent: R30 v1.1 entry (code_r30v11) source_warmup=H boundary (5 warmup epochs, mode boundary, stage parent) '
                            'then mode=fork r30_objective=H arm a1_iface 40 co-train epochs, CC152K real noise (noise_file none, ratio 0), DEV-best model_best',
            est_gpu_hours='~5.8 GPU h on RTX6000PRO (R27 CC a1_iface: warmup 1050 s + co-train 19973 s; peak 45 GiB with GPU gather) ',
            code_port='R34 build_cache / R35 prep / R36-R37 trainer and evaluator are Flickr-specific (145000 records = 29000 images x 5, record//5 DEV truth, 1014x5070 DEV); '
                      'CC152K needs an explicit image<->caption identity map (one caption per image, DEV 1000x1000), its own candidate tables, trust subset and DEV evaluation; '
                      'no injected mismatch and no oracle'))
    write_json(ROOT / 'PARENT_BINDING.json', out)
    print(json.dumps({k: dict(same_series=v['same_series'], recipe_diff=v['recipe_differences_vs_reference'], best=v['best_rsum_logged'], ep=v['best_epoch']) for k, v in out['conditions'].items()}, indent=1))
    print('CC same_series', out['cc152k']['same_series'], out['cc152k']['code_files_differing_from_reference'])


if __name__ == '__main__':
    main()
