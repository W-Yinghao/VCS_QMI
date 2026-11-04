# sourced by NR4 sbatch scripts (NR1 environment/precision policy; NR2/NR3 code imported read-only)
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export HF_HOME=/projects/EEG-foundation-model/yinghao/home_moved_20261104/outputs_nr1/NR1_20261104/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NR4_H=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104
export NR4_OUT=/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104
export NR2_CODE=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code
export NR3_CODE=/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/code
export NR1_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
export PYTHONPATH=$NR2_CODE/src:$NR3_CODE:$NR4_H/code
cd $NR4_H/code
