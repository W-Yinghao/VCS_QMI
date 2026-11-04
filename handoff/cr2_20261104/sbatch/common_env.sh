# sourced by CR2 sbatch scripts
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export CR2_TREE=/home/infres/yinwang/CS_QMI/handoff/cr2_20261104
export CR2_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104
export CR1_TREE=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104
export CR1_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104
export CR2_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
cd $CR2_TREE/code
