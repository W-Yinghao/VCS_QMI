#!/usr/bin/env bash
#SBATCH --job-name=p156_gate_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P156 CPU gate: tests -> block-B matching table -> smoke cells (2 per block, tiny) -> aggregate on the smoke cells.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
SM=/home/infres/yinwang/CS_QMI/outputs/P156_SMOKE_${SLURM_JOB_ID}
banner "P156 gate: tests"
"$PYTHON" -m pytest -q tests/test_p156.py tests/test_vcs_estim_p134.py -p no:cacheprovider 2>&1 | tail -25; rt=${PIPESTATUS[0]}
echo "tests_rc=$rt"
banner "P156 gate: matching table"
"$PYTHON" scripts/p156_sep.py match --out reports/P156_S_MATCHING; rm=$?
banner "P156 gate: smoke cells -> $SM"
rs=0
for b in A E C D Dp B B2; do "$PYTHON" scripts/p156_sep.py run --seed 100 --out "$SM" --blocks $b --limit 2 --smoke --cpu || rs=1; done
"$PYTHON" scripts/p156_aggregate.py --dir "$SM" --out "$SM/aggregate" | head -30; ra=${PIPESTATUS[0]}
echo "tests_rc=$rt match_rc=$rm smoke_rc=$rs aggregate_rc=$ra end=$(date -u +%FT%TZ)"
[ $rt -eq 0 ] && [ $rm -eq 0 ] && [ $rs -eq 0 ] && [ $ra -eq 0 ]
