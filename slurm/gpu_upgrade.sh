#!/usr/bin/env bash
# Owner 2026-10-04: "你来按运行时间的快慢来选择GPU，H100和L40S也可以选择" (replaces slurm/l40s_to_pro.sh).
# Policy: units start on any allowed GPU (RTX6000PRO / H100 / L40S, node51 excluded); every POLL s this script moves the running unit that gains the
# most to a faster idle GPU.  Measured epoch times (A-P3 4v/B256, 2026-10-04): healthy RTX6000PRO 20 s, H100 35 s, L40S 39 s, and the throttled
# RTX6000PRO GPUs in SLOW (nvidia-smi throttle reason 0x88 = HW slowdown + power brake, ~255 W vs ~450 W) 44 s.
# Gain of moving unit j to idle GPU g = remaining_epochs(j) x (t_j - t_g) - OVERHEAD; a move happens only if the best gain >= MIN_SAVE.
# Move: (1) my pending GPU jobs temporarily exclude the node being vacated (an ExcNodeList / Partition / ReqNodeList update keeps EligibleTime;
# hold/release would reset it), (2) requeue j (same id; it resumes from last.pt at an epoch boundary — atomic save, TERM forwarded by run_fg) pinned
# to g's partition and node, (3) once j runs, or after WAIT s, j and the pending jobs get the normal Partition / ReqNodeList / ExcNodeList back.
# Run on the login node: nohup setsid bash slurm/gpu_upgrade.sh >/dev/null 2>&1 &   — log slurm_logs/gpu_upgrade.log; stop by PID (never pkill -f).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
POLL=${POLL:-300}; WAIT=${WAIT:-420}; OVERHEAD=${OVERHEAD:-360}; MIN_SAVE=${MIN_SAVE:-1800}
SLOW=${SLOW:-"node60:0 node60:1 node61:1"}; T_SLOW=${T_SLOW:-44}
declare -A T_PART=([RTX6000PRO]=${T_RTX:-20} [H100]=${T_H100:-35} [L40S]=${T_L40S:-39})
PARTS="RTX6000PRO,H100,L40S"; EXCL="node51"
LOGS=/home/infres/yinwang/CS_QMI/slurm_logs; OUT=/home/infres/yinwang/CS_QMI/outputs; LOG=$LOGS/gpu_upgrade.log
PY=/home/infres/yinwang/CS_QMI/env/bin/python
COOLDOWN=${COOLDOWN:-1800}; declare -A MOVED   # a unit moved in the last COOLDOWN s is not moved again
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

expand_idx() {  # "0-1,3" -> "0 1 3"; "N/A" -> ""
  local part a b out=""
  for part in $(tr ',' ' ' <<<"$1"); do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}; out="$out $(seq -s ' ' "$a" "$b")"
    elif [[ "$part" =~ ^[0-9]+$ ]]; then out="$out $part"; fi
  done
  echo $out
}
gpu_time() {  # partition node idx -> seconds per epoch
  [[ " $SLOW " == *" $2:$3 "* ]] && { echo "$T_SLOW"; return; }
  echo "${T_PART[$1]:-99}"
}
free_gpus() {  # lines "partition node idx t" for idle GPUs on usable nodes with 8 CPUs / 40 GB free
  local p node info tot ca ct rm am used i
  for p in RTX6000PRO H100 L40S; do
    for node in $(sinfo -h -p "$p" -N -o "%N" | sort -u); do
      [[ ",$EXCL," == *",$node,"* ]] && continue
      info=$(scontrol show node -d "$node")
      grep -qE "State=[^ ]*(DRAIN|DOWN|MAINT|RESERVED|FAIL)" <<<"$info" && continue
      tot=$(grep -o "Gres=gpu:[^ ]*" <<<"$info" | grep -oE "[0-9]+(\(|$)" | head -1 | tr -d '(')
      ca=$(grep -oE "CPUAlloc=[0-9]+" <<<"$info" | cut -d= -f2); ct=$(grep -oE "CPUTot=[0-9]+" <<<"$info" | cut -d= -f2)
      rm=$(grep -oE "RealMemory=[0-9]+" <<<"$info" | cut -d= -f2); am=$(grep -oE "AllocMem=[0-9]+" <<<"$info" | cut -d= -f2)
      [ $(( ct - ca )) -ge 8 ] && [ $(( rm - am )) -ge 40960 ] || continue
      used=" $(expand_idx "$(grep -oE "GresUsed=[^ ]*" <<<"$info" | grep -oE "IDX:[^)]*" | cut -d: -f2)") "
      for i in $(seq 0 $(( ${tot:-0} - 1 ))); do
        [[ "$used" == *" $i "* ]] && continue
        echo "$p $node $i $(gpu_time "$p" "$node" "$i")"
      done
    done
  done
}
run_id_of() { grep -h -m1 -oE "unit [^ ]+: cfg" "$LOGS"/unit_*_"$1".out 2>/dev/null | head -1 | awk '{print $2}' | tr -d ':'; }
remaining_of() { "$PY" -c "import json,yaml;s=json.load(open('$OUT/$1/status.json'));c=yaml.safe_load(open('$OUT/$1/config.resolved.yaml'));print(int(c['train']['epochs'])-int(s['completed_epoch']))" 2>/dev/null; }
running_units() {  # lines "jobid partition node idx t remaining run_id"
  local j p n idx rid rem
  while read -r j p n; do
    [[ ",$PARTS," == *",$p,"* ]] || continue
    rid=$(run_id_of "$j"); [ -z "$rid" ] && continue
    rem=$(remaining_of "$rid"); [ -z "$rem" ] && continue
    idx=$(scontrol show job -d "$j" | grep -oE "IDX:[0-9]+" | head -1 | cut -d: -f2)
    echo "$j $p $n ${idx:-0} $(gpu_time "$p" "$n" "${idx:-0}") $rem $rid"
  done < <(squeue -h -u "$USER" -t R -o "%i %P %N")
}
pending_gpu_jobs() { for p in $(squeue -h -u "$USER" -t PD -o "%i"); do s=$(squeue -h -j "$p" -o "%.80P" | tr -d ' '); [[ "$s" =~ RTX6000PRO|H100|L40S ]] && echo "$p"; done; }

log "start pid $$ poll=${POLL}s wait=${WAIT}s overhead=${OVERHEAD}s min_save=${MIN_SAVE}s times RTX=${T_PART[RTX6000PRO]} H100=${T_PART[H100]} L40S=${T_PART[L40S]} slow=${T_SLOW} (${SLOW})"
while :; do
  F=$(free_gpus)
  if [ -n "$F" ]; then
    R=$(running_units)
    best=""; bestgain=0
    while read -r j jp jn jidx jt rem rid; do
      [ -z "$j" ] && continue
      [ $(( $(date +%s) - ${MOVED[$j]:-0} )) -lt "$COOLDOWN" ] && continue
      while read -r gp gn gidx gt; do
        [ -z "$gp" ] && continue
        gain=$(( rem * (jt - gt) - OVERHEAD ))
        [ "$gain" -gt "$bestgain" ] && { bestgain=$gain; best="$j $jp $jn $jidx $jt $rem $rid $gp $gn $gidx $gt"; }
      done <<<"$F"
    done <<<"$R"
    if [ -n "$best" ] && [ "$bestgain" -ge "$MIN_SAVE" ]; then
      read -r j jp jn jidx jt rem rid gp gn gidx gt <<<"$best"
      pend=$(pending_gpu_jobs)
      for p in $pend; do scontrol update jobid="$p" ExcNodeList="$EXCL,$jn"; done
      log "move $j ($rid, $rem epochs left) $jp/$jn:gpu$jidx (${jt}s) -> $gp/$gn:gpu$gidx (${gt}s), est. saving $(( bestgain / 60 )) min; $(wc -w <<<"$pend") pending jobs exclude $jn meanwhile"
      MOVED[$j]=$(date +%s)
      if scontrol requeue "$j"; then
        for i in $(seq 1 20); do [ "$(squeue -h -j "$j" -o %T)" = "PENDING" ] && break; sleep 3; done
        scontrol update jobid="$j" Partition="$gp" ReqNodeList="$gn" ExcNodeList="$EXCL"
        t=0; while [ "$t" -lt "$WAIT" ]; do sleep 30; t=$((t + 30)); [ "$(squeue -h -j "$j" -o %T)" = "RUNNING" ] && break; done
        log "$j after ${t}s: $(squeue -h -j "$j" -o '%T %P %N %r')"
      else
        log "requeue of $j refused"
      fi
      [ "$(squeue -h -j "$j" -o %T)" = "PENDING" ] && scontrol update jobid="$j" Partition="$PARTS" ReqNodeList= ExcNodeList="$EXCL"
      for p in $pend; do [ "$(squeue -h -j "$p" -o %T 2>/dev/null)" = "PENDING" ] && scontrol update jobid="$p" ExcNodeList="$EXCL"; done
      log "pending jobs restored (Partition $PARTS, ExcNodeList $EXCL)"
      sleep 60; continue   # look again soon: there may be another useful move
    fi
  fi
  sleep "$POLL"
done
