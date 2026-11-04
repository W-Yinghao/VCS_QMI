#!/usr/bin/env python3
"""Refuse to run a formal unit unless code/config/cache/calibration/table hashes equal PROTOCOL_FROZEN_NR2C.json."""
import hashlib, json, sys
from pathlib import Path
N = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
fz = json.loads((N/'PROTOCOL_FROZEN_NR2C.json').read_text()); u = sys.argv[1]
bad = [p for p, h in fz['hashes'].items() if sha(p) != h]
if u not in fz['units'] or sha(N/f'locked_configs_nr2c/{u}.json') != fz['units'][u]: bad.append('config:' + u)
if bad: sys.exit(f'FROZEN_MISMATCH {bad}')
print('frozen ok', u)
