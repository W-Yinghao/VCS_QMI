#!/usr/bin/env bash
# Package steps 4-5 after both FIT smokes PASS: the six A0/A1 pairs (one allocation each, 03:00:00, nice 50) and the two
# seed-1001 diagnostics (01:00:00, nice 50), sequentially through submit.sh (normal < 28 wait, 21 GPU-h ledger check).
# All twelve arms are submitted regardless of any seed's result.  A refused submission stops here; rerun later skips nothing
# automatically - submit the remaining lines by hand after checking ledger.py.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
R=/home/infres/yinwang/CS_QMI/handoff/v10_app_round_20261021
for f in "$R/smoke_BF/SERVER_SMOKE.json" "$R/smoke_EPFL/SERVER_SMOKE.json"; do
  python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"]!="PASS")' "$f" || { echo "smoke not PASS: $f"; exit 1; }
done
for ds in BF EPFL; do
  for s in 1001 1002 1003; do
    "$D/submit.sh" "r21_v10_pair_${ds,,}_s$s" 03:00:00 50 "$D/gpu.sbatch" pair "$R/resolved/${ds}_M1_s$s.json" "$R/pairs/${ds}_M1_s$s" || exit
  done
done
for ds in BF EPFL; do
  "$D/submit.sh" "r21_v10_diag_${ds,,}_s1001" 01:00:00 50 "$D/gpu.sbatch" diag "$R/resolved/${ds}_M1_s1001.json" "$R/diag_${ds}" || exit
done
