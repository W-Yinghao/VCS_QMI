# sourced by CR5 sbatch scripts
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export CR5_TREE=/home/infres/yinwang/CS_QMI/handoff/cr5_20261105
export CR5_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105
export CR1_TREE=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104
export CR1_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104
export CR5_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
cd $CR5_TREE/code
export CR2_TREE=/home/infres/yinwang/CS_QMI/handoff/cr2_20261104
export CR2_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104
export CR3_TREE=/home/infres/yinwang/CS_QMI/handoff/cr3_20261104
export CR3_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104
export CR4_TREE=/home/infres/yinwang/CS_QMI/handoff/cr4_20261104
export CR4_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104
