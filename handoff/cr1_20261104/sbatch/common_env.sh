# sourced by CR1 sbatch scripts
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export HF_HOME=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CR1_TREE=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104
export CR1_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104
export CR1_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
export CR1_REV=75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
export NR1_CODE=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code
export PYTHONPATH=$CR1_TREE/code/src:$CR1_TREE/code
cd $CR1_TREE/code
