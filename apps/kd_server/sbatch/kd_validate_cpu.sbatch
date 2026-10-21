#!/bin/bash
# S1 (CPU, no GPU): env lock check, package verification before/after, the package's own CPU validation
# (scripts/run_validation.py --with-smokes), and a CPU selftest of the G0 driver on the synthetic fixture.
# Exit code covers env + verify + package validation; the driver selftest is reported in S1_SUMMARY.json.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
export KD_BATCH_SCRIPT="$0"
cd "$KD_ROOT" || exit 2
JOB=${SLURM_JOB_ID:-manual}
OUT="$KD_ROOT/validation/s1_slurm$JOB"
mkdir -p "$OUT"
kd_ledger start --kind validate --tag s1
"$KD_PY" "$KD_SERVER/tools/env_check.py" --out "$OUT/env_check.json" > /dev/null; rc_env=$?
"$KD_PY" -m vcs_common.verify_package --root "$KD_PKG" > "$OUT/verify_package_before.json"; rc_v1=$?
"$KD_PY" "$KD_PKG/scripts/run_validation.py" --output "$OUT/package_validation" --with-smokes; rc_val=$?
"$KD_PY" "$KD_SERVER/g0/g0_driver.py" --out "$OUT/driver_selftest" --device cpu --selftest; rc_self=$?
"$KD_PY" -m vcs_common.verify_package --root "$KD_PKG" > "$OUT/verify_package_after.json"; rc_v2=$?
rc=0
for x in $rc_env $rc_v1 $rc_val $rc_v2; do [ "$x" -ne 0 ] && rc=1; done
"$KD_PY" - "$OUT" "$rc_env" "$rc_v1" "$rc_val" "$rc_self" "$rc_v2" "$rc" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); env, v1, val, self_, v2, rc = map(int, sys.argv[2:])
def read(p):
    try: return json.loads(Path(p).read_text())
    except Exception as e: return {"unreadable": str(e)}
val_sum = read(out / "package_validation" / "VALIDATION_SUMMARY.json")
st = read(out / "driver_selftest" / "G0_SUMMARY.json")
summary = {"schema": "kd_server_s1_summary_v1", "exit_code": rc,
  "env_check": "PASS" if env == 0 else "FAIL",
  "verify_package_before": read(out / "verify_package_before.json").get("status"),
  "verify_package_after": read(out / "verify_package_after.json").get("status"),
  "package_validation": {"status": val_sum.get("status"), "commands": [(c.get("name"), c.get("passed"), c.get("tests_completed")) for c in val_sum.get("commands", [])]},
  "driver_selftest": {"rc": self_, "overall": st.get("overall"), "stages": {k: v.get("status") for k, v in st.get("stages", {}).items()},
                      "meaning": "driver mechanics on the synthetic CPU fixture; not package evidence"}}
(out / "S1_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
EOF
kd_ledger end --kind validate --tag s1 --rc "$rc" --stage done
exit "$rc"
