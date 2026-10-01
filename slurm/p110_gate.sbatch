#!/usr/bin/env bash
#SBATCH --job-name=p110_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P110 gate: unit tests + V1 / V2 / V3 smoke (reduced sizes, --smoke) on two encoders + aggregate; evidence -> reports/P110_GATE_<job>/.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P110_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G
banner "P110 gate -> $G"
"$PYTHON" -m pytest -q tests/test_p110.py 2>&1 | tee $G/pytest_test_p110.txt; t=${PIPESTATUS[0]}
T0=$(date +%s); "$PYTHON" scripts/p110_v.py --runs P35_vcs_a5_views4_800ep_seed0,P104_G2_views4_800ep_seed0 --modules v1,v2,v3 --smoke --out $G/smoke 2>&1 | tee $G/smoke_log.txt; s=${PIPESTATUS[0]}; echo "smoke wall seconds $(( $(date +%s) - T0 ))" | tee $G/time_smoke.txt
"$PYTHON" scripts/p110_aggregate.py --in $G/smoke --out $G/smoke_aggregate; g=$?
# V3 design check at full size (one encoder, one replicate): does the class-specific cue create differing shortcut use across the pool?
T1=$(date +%s); "$PYTHON" scripts/p110_v.py --runs P104_G2_views4_800ep_seed0 --modules v3 --v3-replicates 1 --out $G/v3check 2>&1 | tee $G/v3check_log.txt; c=${PIPESTATUS[0]}
echo "v3check wall seconds $(( $(date +%s) - T1 ))" | tee -a $G/time_smoke.txt

echo "{\"pytest_rc\": $t, \"smoke_rc\": $s, \"aggregate_rc\": $g, \"v3check_rc\": $c}" > $G/summary.json; cat $G/summary.json
echo "exit=$(( t | s | g | c )) end=$(date -u +%FT%TZ)"; exit $(( t | s | g | c ))
