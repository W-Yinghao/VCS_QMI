"""Evaluator helper: native H DEV ranks of one parent with that tree's upstream evaluator (EV.full_metrics on the cached native DEV
sims) -> npz. Usage: python -I -B tools/native_ranks.py <tree dir> <cond> <out.npz>"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from common import CONDS, install_guards
install_guards(role='audit')
import evals as EV
c = CONDS[sys.argv[2]]
nat = np.load(Path(c['cache_dir']) / 'native_dev_sims.npy')
m, tb, rk = EV.full_metrics(nat)
np.savez(sys.argv[3], **rk)
Path(sys.argv[3] + '.json').write_text(json.dumps(dict(metrics=m, per_cap=int(EV.PER_CAP) if hasattr(EV, 'PER_CAP') else 5)))
