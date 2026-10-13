#!/usr/bin/env bash
#SBATCH --job-name=r7e2g_group
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Owner decisions 2026-10-13 (RTX PRO 6000 cards are hard to get, longer jobs wanted; per-unit L40S jobs grouped): runs E2 units — selector
# units or quality shards — one after the other on this GPU.  Each unit is named by its frozen per-unit job name (r7e2s_<tag> / r7e2q_<bench>_<i>)
# and looked up in the given FROZEN lines file (written by scripts/r7_e2_plan.py); the group runs that line's frozen per-unit wrapper with
# exactly its positional arguments ('bash slurm/r7_e2_sel.sbatch train …' / 'bash slurm/r7_e2_gpu.sbatch run …').  A unit whose marker (its #MARK)
# exists is skipped; a failing unit does not stop the group; inner 'exit=' lines are renamed 'unit_exit=' so the log keeps one final exit line.
# Exit = the number of units without a marker at the end (0 only if every unit is complete).  Packaging only: units, code, rows, rules unchanged.
# The --partition / --time / --job-name of a group come from its line (slurm/e2_rest_grouped.sh); the resources match the per-unit wrappers.
# Usage: sbatch ... slurm/e2_unit_group.sbatch <frozen lines file> <unit job name> [<unit job name> ...]      (positional arguments only)
cd "${E2_REPO:-/home/infres/yinwang/CS_QMI/ssl_pilot}" || exit 1                                   # E2_REPO: dry-run tests only
F=$1; shift; fail=0
echo "group host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD 2>/dev/null) lines=$F units: $*"
[ -f "$F" ] || { echo "[group] lines file $F missing"; echo "exit=$# end=$(date -u +%FT%TZ)"; exit $#; }
for u in "$@"; do
  [[ "$u" =~ ^r7e2[sq]_[A-Za-z0-9_.]+$ ]] || { echo "[group] $u: not an E2 selector / quality unit name"; fail=$((fail + 1)); continue; }
  line=$(grep -F -- "--job-name=$u " "$F" | head -1)
  [ -n "$line" ] || { echo "[group] $u: no line in $F"; fail=$((fail + 1)); continue; }
  m=${line##* #MARK }; read -r -a T <<< "${line%% #MARK *}"; w=""; args=()
  for ((k = 0; k < ${#T[@]}; k++)); do
    if [[ "${T[k]}" =~ ^slurm/r7_e2_(sel|gpu)\.sbatch$ ]]; then w=${T[k]}; args=("${T[@]:k+1}"); break; fi
  done
  [ -n "$w" ] && [ -n "$m" ] && [ "$m" != "$line" ] || { echo "[group] $u: not a frozen E2 selector / quality line"; fail=$((fail + 1)); continue; }
  if [ -f "$m" ]; then echo "[group] $u: already complete"; continue; fi
  echo "[group] $u: start $(date -u +%FT%TZ): bash $w ${args[*]}"
  bash "$w" "${args[@]}" 2>&1 | sed -u 's/^exit=/unit_exit=/'; rc=${PIPESTATUS[0]}
  if [ -f "$m" ]; then echo "[group] $u: done (rc $rc) $(date -u +%FT%TZ)"; else echo "[group] $u: NO marker (rc $rc)"; fail=$((fail + 1)); fi
done
echo "exit=$fail end=$(date -u +%FT%TZ)"; exit $fail
