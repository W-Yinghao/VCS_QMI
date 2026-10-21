#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Package CPU validation in the AIGIQA env (README "最小启动"): verify_package + scripts/run_validation.py --with-smokes,
# plus the AIGIQA-relevant commands of that runner in separate interpreters, so the AIG evidence is complete even if
# a KD step stops the combined runner. Synthetic CPU fixtures only; small outputs in the worktree evidence dir.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
OUT=$AIG_SERVER/evidence/cpu_validation/$SLURM_JOB_ID
mkdir -p "$OUT"
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" > "$OUT/env_check.json" 2>&1 || exit $?
cd "$AIG_PKG"
"$AIG_PY" -m vcs_common.verify_package --root . > "$OUT/verify_package.json"; echo "verify_package rc=$?" | tee "$OUT/rc_verify_package"
"$AIG_PY" scripts/run_validation.py --output "$OUT/full" --with-smokes > "$OUT/run_validation.stdout" 2>&1; echo "run_validation rc=$?" | tee "$OUT/rc_run_validation"
"$AIG_PY" "$AIG_SERVER/cpu_subset.py" --pkg "$AIG_PKG" --out "$OUT/aig_subset"; echo "aig_subset rc=$?" | tee "$OUT/rc_aig_subset"
# Real ImageReward/BLIP architecture, RANDOM weights, synthetic images, CPU: env + adapter + G0-script wiring (not a G0 result)
"$AIG_PY" "$AIG_SERVER/g0_cpu_dryrun.py" --pkg "$AIG_PKG" --bert-local "$AIG_BERT_LOCAL" --evidence "$OUT/g0_cpu_dryrun"; echo "g0_cpu_dryrun rc=$?" | tee "$OUT/rc_g0_cpu_dryrun"
