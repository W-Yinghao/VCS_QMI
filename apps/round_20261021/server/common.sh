# Sourced by the round-21 apps wrappers. Server adaptation only: paths, offline mode, no bytecode, records.
# Nothing here changes TF32/precision, threads or any training setting.
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/server/paths.env
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH   # run_one.py / kd_grad_diag.py put the installed stage first on sys.path themselves
r21_record() {  # $1 = output json, $2 = phase (start|end), $3 = exit code (end only); python = $PY
  local out=$1 phase=$2 rc=${3:-} gpu
  mkdir -p "$(dirname "$out")"
  gpu=$(nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
  "$PY" -I - "$out" "$phase" "$rc" "$gpu" <<'PY'
import json, os, sys, time
out, phase, rc, gpu = sys.argv[1:5]
rec = json.load(open(out)) if os.path.exists(out) else {}
keys = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_JOB_QOS",
        "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_RESTART_COUNT"]
rec[phase] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unix": time.time(),
              "slurm": {k: os.environ.get(k) for k in keys}, "nvidia_smi": gpu}
if phase == "end":
    rec["end"]["exit_code"] = int(rc) if rc else None
    if "start" in rec:
        rec["end"]["elapsed_seconds"] = rec["end"]["unix"] - rec["start"]["unix"]
tmp = out + f".{os.getpid()}.tmp"
json.dump(rec, open(tmp, "w"), indent=1, sort_keys=True)
os.replace(tmp, out)
PY
}
