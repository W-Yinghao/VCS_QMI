"""NR3 per-condition input map. Flickr inputs are the sha-verified /projects copies of the home NR1/NR2 trees
(inputs_flickr/COPY_RECEIPT.json); COCO60 (NR2C) and CC152K_RECOVERED (NR1 recovered snapshot, NR2D tables) are read
in place on /projects. Nothing here is written by NR3 except OUT."""
from pathlib import Path

OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104')
FL = OUT/'inputs_flickr'
NR2C = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104')
CC = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104')
NR2D = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104')
CONDS = ('Flickr00', 'Flickr60', 'Flickr80', 'COCO60', 'CC152K_RECOVERED')


def cond_paths(c):
    if c.startswith('Flickr'):
        return dict(fit=FL/'cache/fit', dev=FL/'cache/dev', observed=FL/f'data/observed_{c}.tsv', calibration=FL/f'calibration/{c}.json',
                    tables=FL/'proposal/base_top100_fit.npz', truth=FL/'evaluator_only/dev_positives.json', baseline=FL/'baseline/float64/METRICS.json',
                    clean_observed=FL/'data/observed_Flickr00.tsv', parent=lambda o: FL/'parents'/f'NR2_HREF_{c}_G_{o}_s1001',
                    nr2_config=lambda o: f'locked_configs/NR2_HREF_{c}_G_{o}_s1001.json')
    if c == 'COCO60':
        return dict(fit=NR2C/'cache/fit', dev=NR2C/'cache/dev', observed=NR2C/'data/observed_COCO60.tsv', calibration=NR2C/'calibration/COCO60.json',
                    tables=NR2C/'proposal/base_top100_fit.npz', truth=NR2C/'evaluator_only/dev_positives.json', baseline=NR2C/'baseline/float64/METRICS.json',
                    clean_observed=NR2C/'data/observed_COCO00.tsv', parent=lambda o: NR2C/'runs'/f'NR2C_HREF_COCO60_G_{o}_s1001',
                    nr2_config=lambda o: f'locked_configs_nr2c/NR2C_HREF_COCO60_G_{o}_s1001.json')
    if c == 'CC152K_RECOVERED':
        return dict(fit=CC/'cache/fit', dev=CC/'cache/dev', observed=CC/'data/observed_CC152K_RECOVERED_20261104.tsv', calibration=CC/'calibration.json',
                    tables=NR2D/'proposal_cc/base_top100_fit.npz', truth=CC/'evaluator_only/dev_positives.json', baseline=CC/'baseline/METRICS.json',
                    clean_observed=None, parent=lambda o: NR2D/'runs_cc'/f'NR2D_HREF_CC152K_RECOVERED_G_{o}_s1001',
                    nr2_config=lambda o: f'locked_configs_nr2d/NR2D_HREF_CC152K_RECOVERED_G_{o}_s1001.json')
    raise KeyError(c)
