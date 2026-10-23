# V9-N2R32 site paths (sourced by the sbatch files; no side effects).  Same data, token cache, HOST_LOCK and
# Python environments as the closed N2P C0/C1 (ssl_pilot_v9_n2p_c/v9_n2p/c_server/env.sh).
N2R_REPO=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2r32
V9_PKG=$N2R_REPO/v9_n2r32/package/v9                 # package v9/ as delivered (commit c66822c6)
N2R_SERVER=$N2R_REPO/v9_n2r32/server
V9_REPO=$N2R_REPO                                    # the 13 SOURCE_LOCK files are byte-identical to ea0535b5
N2P_AFFINE_ROOT=/home/infres/yinwang/CS_QMI/outputs_v9_n2p/c_20261020          # old affine C0/C1, read only
N2R_ROOT=${N2R_ROOT:-/home/infres/yinwang/CS_QMI/outputs_v9_n2p/round_20261021_n2r32}
V9_HISTORY=/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4
V9_DATA=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
V9_VOCAB=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json
V9_TOKENS=/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/source/token_cache   # frozen N2F token cache
# CPU fits/exports/decision: the env of the original affine C0 fit 1052397 (Python 3.10.20, NumPy 2.2.6, Torch 2.9.0).
PY_CPU=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
# GPU students: the N2F/N2M/C1 student env (Python 3.12.12, Torch 2.9.0+cu128, NumPy 2.5.2).
PY_GPU=/home/infres/yinwang/CS_QMI/env/bin/python

# One start and one end line per job in $N2R_ROOT/cost/JOBS.tsv (sacct is unavailable).  A job killed before
# its end line has unknown cost and is kept as such.
job_record() {
  mkdir -p "$N2R_ROOT/cost"
  local gpu=""; command -v nvidia-smi >/dev/null && gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-nojob}" "${SLURM_JOB_NAME:-}" \
    "${SLURM_JOB_PARTITION:-}" "$(hostname)" "$gpu" "${2:-}" >> "$N2R_ROOT/cost/JOBS.tsv"
}
