#!/usr/bin/env python3
"""NR2D: COCO seed units are bound by PROTOCOL_FROZEN_NR2D.json; CC152K_RECOVERED units by PROTOCOL_FROZEN_NR2D_CC.json.
Refuse to run unless every listed hash and the unit config hash match."""
import hashlib, json, sys
from pathlib import Path
N = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
u = sys.argv[1]
fz = json.loads((N/('PROTOCOL_FROZEN_NR2D_CC.json' if 'CC152K' in u else 'PROTOCOL_FROZEN_NR2D.json')).read_text())
bad = [p for p, h in fz['hashes'].items() if sha(p) != h]
if u not in fz['units'] or sha(N/f'locked_configs_nr2d/{u}.json') != fz['units'][u]: bad.append('config:' + u)
if bad: sys.exit(f'FROZEN_MISMATCH {bad}')
print('frozen ok', u)
