"""Audit 3 extension for the Flickr seed1002/1003 extension (R27-D/O/O3/CS snapshots): same per-query DEV rank export as
audit_ranks.py (unchanged, imported). Only units with DONE.json + model_best are exported; done units are skipped by
model_best sha256 (audit_ranks rule). Usage: audit_ranks_seedext.py OUT_ROOT DATA_PATH. DEV only; TEST is never read."""
import os
import audit_ranks as AR

EXT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_seedext_20261031'
ROOTS = {f'{EXT}/r27d/runs': ('d1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i'),
         f'{EXT}/r27o/runs': ('o1_rescue_native_t2i', 'o2_rescue_vcs_t2i'),
         f'{EXT}/r27o3/runs': ('o3_true_target_t2i',),
         f'{EXT}/r27cs/runs': ('bn_cross_t2i', 'cn_mix_t2i')}
CELLS = ('f40_s1002', 'f40_s1003', 'f60_s1002', 'f60_s1003')


def units():
    out = [(c, a, f'{r}/{c}/{a}') for r, arms in ROOTS.items() for c in CELLS for a in arms]
    return [(c, a, d) for c, a, d in out if os.path.exists(f'{d}/DONE.json') and os.path.exists(f'{d}/model_best.pth.tar')]


AR.units = units
if __name__ == '__main__':
    AR.main()
