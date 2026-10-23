#!/usr/bin/env bash
# submit.sh <job-name> <HH:MM:SS> <nice> <sbatch script> [script args...]
# Waits until this user's normal-QoS job count is below 28 (shared with the other round agents), refuses a GPU job that could
# take the round past 21 GPU hours (ledger.py), then sbatch and append to ledger/SUBMITTED.tsv.  No dependency, no hold.
set -uo pipefail
name=$1 limit=$2 nice=$3 script=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
L=/home/infres/yinwang/CS_QMI/handoff/v10_app_round_20261021/ledger
until [ "$(squeue -u "$USER" -h -o %q 2>/dev/null | grep -c normal)" -lt 28 ] 2>/dev/null; do sleep 30; done
kind=cpu
case "$script" in
  *gpu.sbatch) kind=gpu
    if ! python3 "$D/ledger.py" --add "$limit" > /dev/null; then echo "BUDGET: $name ($limit) would exceed 21 GPU h; not submitted"; exit 4; fi ;;
esac
id=$(sbatch --parsable --job-name="$name" --time="$limit" --nice="$nice" "$script" "$@") || { echo "sbatch failed for $name"; exit 5; }
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$name" "$limit" "$kind" "$(date -u +%FT%TZ)" "$*" >> "$L/SUBMITTED.tsv"
echo "$name $id"
