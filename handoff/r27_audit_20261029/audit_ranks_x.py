"""Audit 3 extension: same per-query DEV rank export as audit_ranks.py (unchanged, imported), for the R27-S / R27-D /
R27-O / R27-O3 / R27-CS units (Flickr seed1001 cells). Only units with DONE.json + model_best are exported; done units are
skipped by model_best sha256 (audit_ranks rule). Usage: audit_ranks_x.py OUT_ROOT DATA_PATH. DEV only; TEST is never read."""
import os, sys
import audit_ranks as AR

ROOTS = {'/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs':
             ('s1_balanced_dual', 's2_single_free_vcs', 's3_single_cap_delta', 's4_single_cap_vcs', 's5_single_cap_bce', 's6_single_cap_snce'),
         '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_directional_20261030/runs': ('d1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i'),
         '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_oracle_20261030/runs': ('o1_rescue_native_t2i', 'o2_rescue_vcs_t2i'),
         '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_oracle3_20261031/runs': ('o3_true_target_t2i',),
         '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_candsrc_20261031/runs': ('bn_cross_t2i', 'cn_mix_t2i')}
CELLS = ('f40_s1001', 'f60_s1001')


def units():
    out = [(c, a, f'{r}/{c}/{a}') for r, arms in ROOTS.items() for c in CELLS for a in arms]
    return [(c, a, d) for c, a, d in out if os.path.exists(f'{d}/DONE.json') and os.path.exists(f'{d}/model_best.pth.tar')]


AR.units = units
if __name__ == '__main__':
    AR.main()
