# Site paths of V9 N2P phases C0/C1 (sourced by the sbatch files; no side effects).
N2PC_REPO=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c
N2PC_PKG=$N2PC_REPO/v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020
N2PC_SERVER=$N2PC_REPO/v9_n2p/c_server
# Output root (proposed: home; /projects is below its 1.2 TB free floor).  Override by exporting N2PC_ROOT
# before sbatch; c1_student.py takes the same path through --root.
N2PC_ROOT=${N2PC_ROOT:-/home/infres/yinwang/CS_QMI/outputs_v9_n2p/c_20261020}
N2PC_RECORDS=${N2PC_RECORDS:-/home/infres/yinwang/CS_QMI/handoff/v9_n2p_20261020/rich}
N2PC_OLD=/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4
N2PC_DATA=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
N2PC_VOCAB=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json
# C0: v9_sgr (Python 3.10.20, Torch 2.9.0+cu128, NumPy 2.2.6, nltk 3.10.3): extract_sgr.py tokenises with nltk;
# this env trained the frozen teachers and wrote the historical scores (precheck 1042133).
N2PC_PY_C0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
# C1: the N2F/N2M student env (Python 3.12.12, Torch 2.9.0+cu128, NumPy 2.5.2; reads the frozen token cache).
N2PC_PY_C1=/home/infres/yinwang/CS_QMI/env/bin/python
