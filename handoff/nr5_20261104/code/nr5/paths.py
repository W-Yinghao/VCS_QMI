"""NR5 paths: NR4 paths (NR3 cond_paths + NR4 raw/hidden arrays, read-only) + NR5 output root."""
from pathlib import Path
from nr4.paths import paths as nr4_paths, OUT as NR4_OUT
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104')


def paths(cond):
    return nr4_paths(cond)
