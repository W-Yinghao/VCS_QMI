# sourced by NR1 sbatch scripts
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export HF_HOME=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NR1_CODE=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code
export NR1_OUT=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104
export NR1_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
export NR1_REV=75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
export NR1_SNAPSHOT=$HF_HOME/hub/models--google--siglip2-base-patch16-224/snapshots/$NR1_REV
export PYTHONPATH=$NR1_CODE/src
cd $NR1_CODE
