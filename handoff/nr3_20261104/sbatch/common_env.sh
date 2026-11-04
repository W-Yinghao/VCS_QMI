# sourced by NR3 sbatch scripts (NR2 environment/precision policy; NR2 code imported read-only)
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export NR3_H=/home/infres/yinwang/CS_QMI/handoff/nr3_20261104
export NR3_OUT=/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104
export NR2_CODE=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code
export NR1_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
export PYTHONPATH=$NR2_CODE/src:$NR3_H/code
cd $NR3_H/code
