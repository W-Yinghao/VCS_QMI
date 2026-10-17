# Sourced by slurm/v10_t03/*.sbatch (the T03 equivalent of slurm/v10/v10_header.sh).  t03_header <cpu|gpu> <committed script path>
# [mode]: prints the job header (host, GPU, start UTC, git commit and branch, commit at submission, T03 code hash (src/v10 +
# scripts/v10_t03), the BF code hash of src/v10 + scripts/v10 for reference, the job pin, sha of the batch script that runs, mode,
# and - owner decision 2026-10-17 section 3 - CUBLAS_WORKSPACE_CONFIG and the two TF32 override variables, '<unset>' when unset;
# nothing is set or changed here) and refuses to run when (a) pinned / test paths have uncommitted changes; (b) the job was not
# submitted through slurm/v10_t03/submit.sh (V10_T03_EXPECT_CODE unset); (c) the pinned paths changed since submission; (d) the
# batch script SLURM runs ($0, the spool copy) differs from the committed file; (e) gpu job without a visible GPU.
t03_header() {
  local kind=$1 script=$2 mode=${3:-} R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03 commit branch code bf pin dirty gpu ran committed
  local PIN_PATHS="src/v10 scripts/v10_t03 slurm/v10_t03 v10/protocol/T03_UNITS.csv v10/package_v10_server_v1_20261016/synthetic/src/v10_synthetic.py v10/package_v10_server_v1_20261016/synthetic/tests/check_synthetic.py"
  local DIRTY_PATHS="$PIN_PATHS tests/v10_t03 src/v9/reg"
  commit=$(git -C "$R" rev-parse HEAD)
  branch=$(git -C "$R" rev-parse --abbrev-ref HEAD)
  code=$(cd "$R" && git ls-files src/v10 scripts/v10_t03 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  bf=$(cd "$R" && git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  # shellcheck disable=SC2086
  pin=$(cd "$R" && git ls-files $PIN_PATHS | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  # shellcheck disable=SC2086
  dirty=$(git -C "$R" status --porcelain -- $DIRTY_PATHS | wc -l)
  ran=$(sha256sum "$0" | cut -c1-16)
  committed=$(git -C "$R" show "HEAD:$script" 2>/dev/null | sha256sum | cut -c1-16)
  gpu=none
  if [ "$kind" = gpu ]; then gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1); fi
  echo "host=$(hostname) gpu=${gpu:-NONE} start=$(date -u +%FT%TZ) commit=${commit} branch=${branch} submit_commit=${V10_T03_SUBMIT_COMMIT:-UNSET} t03_code=${code} bf_code=${bf} pin=${pin} expect=${V10_T03_EXPECT_CODE:-UNSET} script=${script} script_sha=${ran} committed_script_sha=${committed} mode=${mode:-NONE} dirty_files=${dirty} job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} node=${SLURM_JOB_NODELIST:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA}"
  echo "precision_env CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG-<unset>} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE-<unset>}"
  if [ "$dirty" -ne 0 ]; then
    # shellcheck disable=SC2086
    echo "[v10_t03] REFUSED: uncommitted changes under $DIRTY_PATHS:"; git -C "$R" status --porcelain -- $DIRTY_PATHS
    return 1
  fi
  if [ -z "${V10_T03_EXPECT_CODE:-}" ]; then echo "[v10_t03] REFUSED: V10_T03_EXPECT_CODE unset - submit with slurm/v10_t03/submit.sh"; return 1; fi
  if [ "$pin" != "$V10_T03_EXPECT_CODE" ]; then echo "[v10_t03] REFUSED: pinned paths changed since submission (pin $pin != $V10_T03_EXPECT_CODE)"; return 1; fi
  if [ "$ran" != "$committed" ]; then echo "[v10_t03] REFUSED: the running batch script ($0) differs from HEAD:$script"; return 1; fi
  if [ "$kind" = gpu ] && [ -z "$gpu" ]; then echo "[v10_t03] REFUSED: no GPU visible"; return 1; fi
  return 0
}
