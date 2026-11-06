# sourced by CR6 sbatch scripts
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export HF_HOME=/projects/EEG-foundation-model/yinghao/home_moved_20261104/outputs_nr1/NR1_20261104/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CR6_TREE=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105
export CR6_OUT=/projects/EEG-foundation-model/yinghao/outputs_cr6/CR6_20261105
export CR6_PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
cd $CR6_TREE/code/site
