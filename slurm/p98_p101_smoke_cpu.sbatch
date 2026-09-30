#!/usr/bin/env bash
#SBATCH --job-name=p98_p101_smoke
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p98_p101_smoke_%j.out
# CPU gate for P98 / P101: tests + a smoke of each script on one checkpoint (tiny roles, 2 probe epochs / 60 critic updates).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8
banner "P98 / P101 CPU gate"
S=/home/infres/yinwang/CS_QMI/outputs/P98_P101_smoke_${SLURM_JOB_ID}; mkdir -p $S
"$PYTHON" -m pytest -q tests/test_p98_p101.py -p no:cacheprovider 2>&1 | tail -3; r1=${PIPESTATUS[0]}
"$PYTHON" scripts/probe_robustness.py --smoke --cpu --workers 4 --out $S/p98_smoke --runs P91_c100_simclr_views4_800ep_seed0:epoch_800.pt,P35_vcs_a5_views4_800ep_seed0:epoch_800.pt; r2=$?
"$PYTHON" -m vcs_estim.observation_frozen --run-dir /home/infres/yinwang/CS_QMI/outputs/P35_vcs_a5_views4_800ep_seed0 --ckpt epoch_800 --out $S/p101_ref.json --smoke --cpu --workers 4; r3=$?
"$PYTHON" -m vcs_estim.observation_frozen --run-dir /home/infres/yinwang/CS_QMI/outputs/P35_vcs_a5_views4_800ep_seed0 --ckpt epoch_400 --out $S/p101_ep400.json --ref-json $S/p101_ref.json --smoke --cpu --workers 4; r4=$?
"$PYTHON" -m vcs_estim.observation_frozen --stage aggregate --inputs $S/p101_ep400.json $S/p101_ref.json --out $S/p101_agg.md; r5=$?
echo "rc tests=$r1 p98=$r2 p101_ref=$r3 p101_ep400=$r4 agg=$r5 end=$(date -u +%FT%TZ)"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ] && [ $r4 -eq 0 ] && [ $r5 -eq 0 ]
