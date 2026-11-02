"""R35 B-ORACLE ground truth (pre-injection FIT record -> image identity). Importable ONLY inside the oracle
runner process (train_oracle.py sets R35_ORACLE_PROCESS=1 before importing); every other R35 process installs a
meta-path blocker that refuses this import, and asserts the module is not loaded. Diagnostic arm only."""
import os
import sys
import numpy as np

if os.environ.get('R35_ORACLE_PROCESS') != '1' or not os.path.basename(sys.argv[0]).startswith('train_oracle'):
    raise ImportError('oracle_truth may only be imported by train_oracle.py')

N_TRAIN_CAP = 145000


def truth_map():
    """Flickr30K precomp layout: caption record k belongs to image k//5 (the native data.py _t2i_index)."""
    return np.arange(N_TRAIN_CAP, dtype=np.int64) // 5
