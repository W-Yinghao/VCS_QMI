#!/usr/bin/env bash
#SBATCH --job-name=p138_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P138 CPU gate: P138 tests + regression (P114 all-view JS, v2 core, P126, P133) + config-hash invariance against a HEAD worktree.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P138_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
"$PYTHON" -m pytest -q tests/test_vcs_ssl_p138.py 2>&1 | tee $G/pytest_p138.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_vcs_ssl_p114.py tests/test_vcs_ssl_v2.py tests/test_vcs_ssl_p126.py tests/test_vcs_ssl_p133.py tests/test_ssl_core.py 2>&1 | tee $G/pytest_regression.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
WT=$(mktemp -d)/head; git worktree add -q --detach "$WT" HEAD
"$PYTHON" scripts/p138_config_hashes.py --src src --out $G/hashes_worktree.json
"$PYTHON" scripts/p138_config_hashes.py --src "$WT/src" --out $G/hashes_head.json
git worktree remove --force "$WT"
"$PYTHON" - $G <<'PY' || rc=1
import json, sys
g = sys.argv[1]; a = json.load(open(f"{g}/hashes_worktree.json")); b = json.load(open(f"{g}/hashes_head.json"))
common = [k for k in b if not b[k].startswith("ERROR")]
changed = [k for k in common if a.get(k) != b[k]]
new_only = [k for k in a if k not in common and not a[k].startswith("ERROR")]
res = {"loaded_under_head": len(common), "changed": changed, "loaded_only_in_worktree": new_only}
json.dump(res, open(f"{g}/hash_invariance.json", "w"), indent=1)
print(f"hash invariance: {len(common)} configs loaded under HEAD, changed {len(changed)}; new configs only loadable in the worktree: {len(new_only)}")
sys.exit(1 if changed else 0)
PY
echo "{\"gate_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
