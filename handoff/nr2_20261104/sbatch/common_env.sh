# sourced by NR2 sbatch scripts (same environment/precision policy as NR1 common_env.sh)
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export HF_HOME=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NR2_CODE=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code
export NR2_H=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104
export NR1_OUT=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104
export NR2_OUT=/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104
export NR1_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
export PYTHONPATH=$NR2_CODE/src
cd $NR2_CODE
