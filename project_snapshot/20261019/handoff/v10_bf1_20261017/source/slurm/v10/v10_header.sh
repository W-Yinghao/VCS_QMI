# Sourced by slurm/v10/*.sbatch.  v10_header <cpu|gpu> <committed script path> [mode]: prints the job header (host, GPU name, start
# UTC, git commit, commit at submission, code hash of src/v10 + scripts/v10, job pin, sha of the batch script that runs, the job's
# mode, and - owner decision 2026-10-17 section 3 - CUBLAS_WORKSPACE_CONFIG and the two TF32 override variables, '<unset>' when
# unset; nothing is set or changed here) and refuses
# to run when (a) the code paths have uncommitted changes; (b) the job was not submitted through slurm/v10/submit.sh
# (V10_EXPECT_CODE unset); (c) src/v10 + scripts/v10 + slurm/v10 changed since submission (pin != V10_EXPECT_CODE); (d) the
# batch script SLURM runs ($0, the spool copy) differs from the committed file; (e) gpu job without a visible GPU.
# Commits that touch only results / docs (v10/T01, v10/T02, ...) do not change the pin, so a pending job still starts.
# Code hash = git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16 (= adapter.code_hash()).
v10_header() {
  local kind=$1 script=$2 mode=${3:-} R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10 commit code pin dirty gpu ran committed
  commit=$(git -C "$R" rev-parse HEAD)
  code=$(cd "$R" && git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  pin=$(cd "$R" && git ls-files src/v10 scripts/v10 slurm/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  dirty=$(git -C "$R" status --porcelain -- src/v10 scripts/v10 slurm/v10 tests/v10 src/v9/reg | wc -l)
  ran=$(sha256sum "$0" | cut -c1-16)
  committed=$(git -C "$R" show "HEAD:$script" 2>/dev/null | sha256sum | cut -c1-16)
  gpu=none
  if [ "$kind" = gpu ]; then gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1); fi
  echo "host=$(hostname) gpu=${gpu:-NONE} start=$(date -u +%FT%TZ) commit=${commit} submit_commit=${V10_SUBMIT_COMMIT:-UNSET} code=${code} pin=${pin} expect=${V10_EXPECT_CODE:-UNSET} script=${script} script_sha=${ran} committed_script_sha=${committed} mode=${mode:-NONE} dirty_code_files=${dirty} job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} node=${SLURM_JOB_NODELIST:-NA} cuda_visible=${CUDA_VISIBLE_DEVICES:-NA}"
  echo "precision_env CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG-<unset>} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE-<unset>}"
  if [ "$dirty" -ne 0 ]; then
    echo "[v10] REFUSED: uncommitted changes under src/v10 scripts/v10 slurm/v10 tests/v10 src/v9/reg:"; git -C "$R" status --porcelain -- src/v10 scripts/v10 slurm/v10 tests/v10 src/v9/reg
    return 1
  fi
  if [ -z "${V10_EXPECT_CODE:-}" ]; then echo "[v10] REFUSED: V10_EXPECT_CODE unset - submit with slurm/v10/submit.sh"; return 1; fi
  if [ "$pin" != "$V10_EXPECT_CODE" ]; then echo "[v10] REFUSED: src/v10 scripts/v10 slurm/v10 changed since submission (pin $pin != $V10_EXPECT_CODE)"; return 1; fi
  if [ "$ran" != "$committed" ]; then echo "[v10] REFUSED: the running batch script ($0) differs from HEAD:$script"; return 1; fi
  if [ "$kind" = gpu ] && [ -z "$gpu" ]; then echo "[v10] REFUSED: no GPU visible"; return 1; fi
  return 0
}
