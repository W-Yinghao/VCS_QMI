"""NR4 path resolution: NR3 cond_paths (read-only) + NR4 raw/hidden arrays and NR3-B reference runs."""
from pathlib import Path
from nr3.paths import cond_paths, OUT as NR3_OUT
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104')
DS = {'Flickr00': 'flickr', 'Flickr60': 'flickr', 'COCO60': 'coco', 'CC152K_RECOVERED': 'cc'}


def paths(cond, ref_unit=None):
    p = dict(cond_paths(cond)); p['prefix'] = NR3_OUT/'prep'/cond/'prefix'/'PREFIX_step_008000.pt'
    p['hidden'] = OUT/'raw'/DS[cond]/'fit'; p['raw_dev'] = OUT/'raw'/DS[cond]/'dev'
    if ref_unit: p['nr3_run'] = NR3_OUT/'runs'/ref_unit
    return p
