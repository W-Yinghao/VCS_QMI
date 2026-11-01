"""Adapter to the R30 production hard-comparison function (code_r30/r30_align.hard_pair_rows), used by margin_diag.py only
when the R30 GPU gate reports all_pass true. Loaded from its file (read-only); its `import r29_align` resolves to the
already-imported code_r29/r29_align (byte-identity checked here)."""
import hashlib, importlib.util, os, sys

import json as _json
_ROOT = '/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101'
_gate = _json.load(open(os.path.join(_ROOT, 'GPU_GATE.json')))
CODE_R30 = None
for _d in ('code_r30v11', 'code_r30'):   # the snapshot whose INSTALL_MANIFEST the passing gate certifies
    _m = os.path.join(_ROOT, _d, 'INSTALL_MANIFEST.json')
    if os.path.isfile(_m) and _json.load(open(_m))['files_sha256'] == _gate.get('code_files_sha256'):
        CODE_R30 = os.path.join(_ROOT, _d); break
assert CODE_R30 is not None, 'no code snapshot matches the R30 GPU gate'
_f = os.path.join(CODE_R30, 'r30_align.py')
_sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
_r29_in_r30 = os.path.join(CODE_R30, 'r29_align.py')
if os.path.isfile(_r29_in_r30):
    assert _sha(_r29_in_r30) == _sha(sys.modules['r29_align'].__file__), 'code_r30/r29_align.py differs from code_r29'
_spec = importlib.util.spec_from_file_location('r30_align_prod', _f)
R30 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R30)
SOURCE = f'{os.path.basename(CODE_R30)}/r30_align.hard_pair_rows sha256 {_sha(_f)}'


def pair_rows(scores, obs, kind):
    r, v, _, _ = R30.hard_pair_rows(scores, obs, kind)
    return r, v
