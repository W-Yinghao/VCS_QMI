# Site paths of the V9 N2M execution on this server (sourced by the job helpers; no side effects).
N2M_PACKAGE=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019
N2M_BASE=/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017
N2M_OUT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1
N2M_PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
N2M_LOGS=/home/infres/yinwang/CS_QMI/slurm_logs
N2M_OLD_OUTPUT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k
# All new arms, the fixed-loss export and the verification share one frozen environment (package rule): RTX6000PRO
# (node58/59/61: NVIDIA RTX PRO 6000 Blackwell, driver 615.71.09 on all three, probed 2026-10-19 11:08 UTC).
N2M_PARTITION=RTX6000PRO
