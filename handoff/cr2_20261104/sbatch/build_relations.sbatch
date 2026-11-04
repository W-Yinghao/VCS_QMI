#!/usr/bin/env bash
# usage: sbatch --job-name=cr2_build_<COND> build_relations.sbatch <COND>   (deployable CSR graphs; no truth)
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr2_20261104/sbatch/common_env.sh
C=$1; echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
for P in BASE J BCE NCE; do for POL in KEEP_SOFT KEEP_TOP1; do
  U=CR2_${C}_${P}_${POL}_s1001
  $CR2_PY -I -B site/build_relation_pkg.py --config $CR2_TREE/package/configs/units/$U.json --bank $CR1_OUT/banks/$C/fit_bank.npz \
     --candidates $CR1_OUT/$C/prepared/graph.npz --parent-graph $CR1_OUT/$C/relations/${P}_CAP.npz --out $CR2_OUT/relations/$C/$U.npz > /dev/null
  echo "BUILT $U"
done; done
for POL in SUPPORT_OBS RANDOM_SUPPORT; do
  U=CR2_${C}_${POL}_s1001
  $CR2_PY -I -B site/build_relation_pkg.py --config $CR2_TREE/package/configs/units/$U.json --bank $CR1_OUT/banks/$C/fit_bank.npz \
     --candidates $CR1_OUT/$C/prepared/graph.npz --out $CR2_OUT/relations/$C/$U.npz > /dev/null
  echo "BUILT $U"
done
echo "EXIT_OK $(date -u +%FT%TZ)"
