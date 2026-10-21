# Sourced by the AIGIQA wrappers. Server adaptation only: paths, offline mode, no bytecode in the package tree.
# Nothing here changes TF32/precision, threads or any training setting (package INSTALL_CN.md section 5).
AIG_SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$AIG_SERVER_DIR/paths.env"
export PYTHONPATH="$AIG_PKG:$AIG_PKG/shared"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# Offline assets: ImageReward's BLIP calls BertTokenizer.from_pretrained('bert-base-uncased'); HF_HOME holds a
# linked cache of the verified local bert-base-uncased tokenizer files (built by stage_data.py).
export HF_HOME="$AIG_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
aig_slurm_record() {  # $1 = output json path, $2 = phase (start/end), $3 = exit code (end only)
  local out=$1 phase=$2 rc=${3:-}
  mkdir -p "$(dirname "$out")"
  local gpu
  gpu=$(nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
  "$AIG_PY" -I - "$out" "$phase" "$rc" "$gpu" <<'PY'
import json, os, sys, time
out, phase, rc, gpu = sys.argv[1:5]
rec = json.load(open(out)) if os.path.exists(out) else {}
keys = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_JOB_NODELIST", "SLURM_JOB_QOS",
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
