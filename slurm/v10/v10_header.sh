# Sourced by slurm/v10/*.sbatch.  v10_header <cpu|gpu>: prints the job header (host, GPU name, start UTC, git commit, code hash of
# src/v10 + scripts/v10) and refuses to run when the code paths have uncommitted changes (jobs run committed code only).
# Code hash = git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16 (= adapter.code_hash()).
v10_header() {
  local kind=$1 R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10 commit code dirty gpu
  commit=$(git -C "$R" rev-parse HEAD)
  code=$(cd "$R" && git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  dirty=$(git -C "$R" status --porcelain -- src/v10 scripts/v10 slurm/v10 src/v9/reg | wc -l)
  gpu=none
  if [ "$kind" = gpu ]; then gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1); fi
  echo "host=$(hostname) gpu=${gpu:-NONE} start=$(date -u +%FT%TZ) commit=${commit} code=${code} dirty_code_files=${dirty} job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} cuda_visible=${CUDA_VISIBLE_DEVICES:-NA}"
  if [ "$dirty" -ne 0 ]; then
    echo "[v10] REFUSED: uncommitted changes under src/v10 scripts/v10 slurm/v10 src/v9/reg:"; git -C "$R" status --porcelain -- src/v10 scripts/v10 slurm/v10 src/v9/reg
    return 1
  fi
  if [ "$kind" = gpu ] && [ -z "$gpu" ]; then echo "[v10] REFUSED: no GPU visible"; return 1; fi
  return 0
}
