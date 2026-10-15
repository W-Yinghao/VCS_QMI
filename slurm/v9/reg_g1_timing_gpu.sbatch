#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g1_timing
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 GPU timing + remaining code-path smokes, in a SEPARATE root (outputs_v9/reg_timing) so nothing here is reused by the formal units:
#   (a) CoMIR and kernel smokes on REG-P1's smoke cache (code path only);  (b) one real unit end-to-end (train 3 000 steps, CAL selection over 12
#   checkpoints, DEV) with --n-pert 3 (3 of the 10 perturbations per image) to measure training time and s / registration; the full-unit cost is
#   extrapolated to 1 280 registrations.  Results are timing records, not results of the protocol.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; export V9_REG_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v9/reg_timing
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) root=$V9_REG_ROOT"
rc_all=0
for u in V9-G1-comir-frac0.25-eps0.2-s901 V9-G1-kernel-frac0.25-eps0.2-s901; do
  t0=$(date +%s); "$PYTHON" scripts/v9/reg_g1.py run --unit "$u" --smoke --n-pert 2; rc=$?; echo "[smoke-exit] $u rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc_all=$rc
done
t0=$(date +%s); "$PYTHON" scripts/v9/reg_g1.py run --unit V9-G1-vcs-frac0.25-eps0.2-s901 --n-pert 3; rc=$?; echo "[timing-exit] V9-G1-vcs-frac0.25-eps0.2-s901 n_pert=3 rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc_all=$rc
"$PYTHON" - <<'PY'
import json, glob
r = json.load(open(glob.glob("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_timing/units/V9-G1-vcs-frac0.25-eps0.2-s901/p2/results/vcs_s901.json")[0]); t = json.load(open("/projects/EEG-foundation-model/yinghao/outputs_v9/reg_timing/units/V9-G1-vcs-frac0.25-eps0.2-s901/run/train.json"))
n_cal = sum(len(v) for k, v in r["cal"].items() if "#" not in k); n_dev = sum(len(v) for k, v in r["dev"].items() if "#" not in k)
sel_secs = sum(x for x in [])  # placeholder
print(f"[timing] train {t['seconds']:.0f} s ({t['steps']} steps, fit {len(t['fit'])} cores, realised mismatch {t['realised_mismatch_fraction']:.3f}); registration total {r['seconds_total']:.0f} s for 12 x {n_cal} CAL + {n_dev} DEV; DEV {r['dev_sec_per_registration']:.2f} s/reg; CAL/DEV success<42 {r['cal_success42']:.3f}/{r['dev_success42']:.3f}; peak {r['peak_gib']:.2f} GiB")
per = r['seconds_total'] / (12 * n_cal + n_dev); print(f"[timing] mean s/registration over the selection + DEV = {per:.2f}; full unit (12 x 80 + 320 = 1280 regs) ~ {per * 1280 / 60:.0f} min + training {t['seconds'] / 60:.1f} min")
PY
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
