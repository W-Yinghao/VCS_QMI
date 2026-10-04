#!/usr/bin/env bash
# Owner 2026-10-04: "你来按运行时间的快慢来选择GPU，H100和L40S也可以选择" — v6 (v4's all-node exclusion made SLURM flag jobs BadConstraints).
# Measured epoch times (A-P3 4v/B256, 2026-10-04): healthy RTX6000PRO 20 s, H100 35 s, L40S 39 s; throttled RTX6000PRO GPUs (nvidia-smi throttle
# reason 0x88 = HW slowdown + power brake, ~255 W vs ~450 W) 44 s: node60 GPU 0 and 1 (node excluded for every job), node61 GPU 1 (cannot be
# excluded per GPU).  SLURM fills a multi-partition job in partition order RTX6000PRO > H100 > L40S, i.e. fastest first.
#
# Every cycle (POLL s; 60 s while a requeued unit waits):
#  * W = my pending GPU jobs with Restarts > 0 (moved units waiting; a requeue resets EligibleTime, so they would otherwise queue behind every newer
#    job).  While W is non-empty the other pending GPU jobs are soft-blocked: Partition = an allowed partition that has no idle GPU now (H100 >
#    RTX6000PRO > L40S) — satisfiable, so no BadConstraints; a Partition / ExcNodeList / ReqNodeList update keeps EligibleTime (hold/release resets
#    it).  W jobs keep all partitions, so the next free quota slot goes to the oldest W job on the fastest idle GPU.  When W is empty, everything
#    is back to Partition RTX6000PRO,H100,L40S, ExcNodeList node51,node60.
#  * Move: gain(j, g) = remaining_epochs(j) x (t_j - t_g) - OVERHEAD for running unit j and idle GPU g; best gain >= MIN_SAVE, j not moved in the
#    last COOLDOWN s.  W empty: others soft-blocked (partition without idle GPU, not g's; none -> skip), j requeued pinned to g's partition + node,
#    WAIT s, then j back to all partitions.  W non-empty: j is requeued unpinned and joins W (the freed slot goes to the oldest W job, fastest GPU).
#    A requeued unit resumes from last.pt at an epoch boundary (atomic save every epoch; TERM forwarded by run_fg); SLURM delays its start ~2 min.
# DRY=1: log the decisions only (slurm_logs/gpu_upgrade_dry.log), change nothing.
# Run on the login node: nohup setsid bash slurm/gpu_upgrade.sh >/dev/null 2>&1 &   — log slurm_logs/gpu_upgrade.log; stop by PID (never pkill -f).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
POLL=${POLL:-300}; WAIT=${WAIT:-480}; OVERHEAD=${OVERHEAD:-360}; MIN_SAVE=${MIN_SAVE:-1800}; COOLDOWN=${COOLDOWN:-1800}; DRY=${DRY:-0}
SLOW=${SLOW:-"node60:0 node60:1 node61:1"}; T_SLOW=${T_SLOW:-44}
declare -A T_PART=([RTX6000PRO]=${T_RTX:-20} [H100]=${T_H100:-35} [L40S]=${T_L40S:-39})
PARTS="RTX6000PRO,H100,L40S"; EXCL="node51,node60"
LOGS=/home/infres/yinwang/CS_QMI/slurm_logs; OUT=/home/infres/yinwang/CS_QMI/outputs
LOG=$LOGS/gpu_upgrade.log; [ "$DRY" = 1 ] && LOG=$LOGS/gpu_upgrade_dry.log
PY=/home/infres/yinwang/CS_QMI/env/bin/python
declare -A MOVED
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
act() { if [ "$DRY" = 1 ]; then log "DRY: $*"; else "$@"; fi; }   # every state change goes through act

expand_idx() {  # "0-1,3" -> "0 1 3"; "N/A" -> ""
  local part a b out=""
  for part in $(tr ',' ' ' <<<"$1"); do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}; out="$out $(seq -s ' ' "$a" "$b")"
    elif [[ "$part" =~ ^[0-9]+$ ]]; then out="$out $part"; fi
  done
  echo $out
}
gpu_time() { [[ " $SLOW " == *" $2:$3 "* ]] && { echo "$T_SLOW"; return; }; echo "${T_PART[$1]:-99}"; }   # partition node idx -> s/epoch
free_gpus() {  # "partition node idx t" for idle GPUs on allowed nodes with 8 CPUs / 40 GB free
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
      for i in $(seq 0 $(( ${tot:-0} - 1 ))); do [[ "$used" == *" $i "* ]] || echo "$p $node $i $(gpu_time "$p" "$node" "$i")"; done
    done
  done
}
run_id_of() { grep -h -m1 -oE "unit [^ ]+: cfg" "$LOGS"/unit_*_"$1".out 2>/dev/null | head -1 | awk '{print $2}' | tr -d ':'; }
remaining_of() { "$PY" -c "import json,yaml;s=json.load(open('$OUT/$1/status.json'));c=yaml.safe_load(open('$OUT/$1/config.resolved.yaml'));print(int(c['train']['epochs'])-int(s['completed_epoch']))" 2>/dev/null; }
running_units() {  # "jobid partition node idx t remaining run_id"
  local j p n idx rid rem
  while read -r j p n; do
    [[ ",$PARTS," == *",$p,"* ]] || continue
    rid=$(run_id_of "$j"); [ -z "$rid" ] && continue
    rem=$(remaining_of "$rid"); [ -z "$rem" ] && continue
    idx=$(scontrol show job -d "$j" | grep -oE "IDX:[0-9]+" | head -1 | cut -d: -f2)
    echo "$j $p $n ${idx:-0} $(gpu_time "$p" "$n" "${idx:-0}") $rem $rid"
  done < <(squeue -h -u "$USER" -t R -o "%i %P %N")
}
jf() { scontrol show job "$1" 2>/dev/null | grep -oE "$2=[^ ]*" | head -1 | cut -d= -f2; }   # job field
pending_gpu_jobs() { local p; for p in $(squeue -h -u "$USER" -t PD -o "%i"); do [[ "$(jf "$p" Partition)" =~ RTX6000PRO|H100|L40S ]] && echo "$p"; done; }
pick_soft() { local p; for p in H100 RTX6000PRO L40S; do [ "$p" = "$2" ] && continue; grep -q "^$p " <<<"$1" || { echo "$p"; return; }; done; }  # $1 free table, $2 avoid
soft_block() { local s=$1 p; shift; for p in "$@"; do [ "$(jf "$p" Partition)" = "$s" ] || act scontrol update jobid="$p" Partition="$s"; done; }
normal() { local p; for p in "$@"; do { [ "$(jf "$p" Partition)" = "$PARTS" ] && [ "$(jf "$p" ExcNodeList)" = "$EXCL" ] && [ "$(jf "$p" ReqNodeList)" = "(null)" ]; } || act scontrol update jobid="$p" Partition="$PARTS" ReqNodeList= ExcNodeList="$EXCL"; done; }

log "start pid $$ v6 dry=$DRY poll=${POLL}s wait=${WAIT}s overhead=${OVERHEAD}s min_save=${MIN_SAVE}s cooldown=${COOLDOWN}s RTX=${T_PART[RTX6000PRO]} H100=${T_PART[H100]} L40S=${T_PART[L40S]} slow=${T_SLOW} (${SLOW}) excl=${EXCL}"
last_w="-"
while :; do
  pend=$(pending_gpu_jobs); W=""; others=""
  for p in $pend; do if [ "$(jf "$p" Restarts)" -gt 0 ] 2>/dev/null; then W="$W $p"; else others="$others $p"; fi; done
  F=$(free_gpus)
  normal $W
  if [ -n "$W" ]; then
    s=$(pick_soft "$F" ""); soft_block "${s:-H100}" $others
    [ "$W" != "$last_w" ] && log "waiting requeued units:$W — $(wc -w <<<"$others") other pending jobs soft-blocked (Partition ${s:-H100})"
  else
    normal $others
    [ "$last_w" != "" ] && [ "$last_w" != "-" ] && log "no requeued unit waiting — pending jobs back to $PARTS / excl $EXCL"
  fi
  last_w="$W"
  best=""; bestgain=0
  if [ -n "$F" ]; then
    R=$(running_units)
    while read -r j jp jn jidx jt rem rid; do
      [ -z "$j" ] && continue
      [ $(( $(date +%s) - ${MOVED[$j]:-0} )) -lt "$COOLDOWN" ] && continue
      while read -r gp gn gidx gt; do
        [ -z "$gp" ] && continue
        gain=$(( rem * (jt - gt) - OVERHEAD ))
        [ "$gain" -gt "$bestgain" ] && { bestgain=$gain; best="$j $jp $jn $jidx $jt $rem $rid $gp $gn $gidx $gt"; }
      done <<<"$F"
    done <<<"$R"
  fi
  if [ -n "$best" ] && [ "$bestgain" -ge "$MIN_SAVE" ]; then
    read -r j jp jn jidx jt rem rid gp gn gidx gt <<<"$best"
    if [ -n "$W" ]; then   # rotation: j joins the waiting set; the oldest waiting unit takes the freed slot on the fastest idle GPU
      log "rotate $j ($rid, $rem epochs left, $jp/$jn:gpu$jidx ${jt}s): fastest idle GPU $gp/$gn:gpu$gidx (${gt}s) goes to the oldest waiting unit; est. saving $(( bestgain / 60 )) min"
      MOVED[$j]=$(date +%s); act scontrol requeue "$j"
      sleep 90; continue
    fi
    s=$(pick_soft "$F" "$gp")
    if [ -z "$s" ]; then log "move of $j skipped: no partition without idle GPUs to soft-block into (target $gp)"; sleep "$POLL"; continue; fi
    log "move $j ($rid, $rem epochs left) $jp/$jn:gpu$jidx (${jt}s) -> $gp/$gn:gpu$gidx (${gt}s), est. saving $(( bestgain / 60 )) min; $(wc -w <<<"$others") pending jobs soft-blocked (Partition $s) meanwhile"
    soft_block "$s" $others; MOVED[$j]=$(date +%s)
    if act scontrol requeue "$j"; then
      if [ "$DRY" != 1 ]; then
        for i in $(seq 1 20); do [ "$(squeue -h -j "$j" -o %T)" = "PENDING" ] && break; sleep 3; done
        scontrol update jobid="$j" Partition="$gp" ReqNodeList="$gn" ExcNodeList="node51"
        t=0; while [ "$t" -lt "$WAIT" ]; do sleep 30; t=$((t + 30)); [ "$(squeue -h -j "$j" -o %T)" = "RUNNING" ] && break; done
        log "$j after ${t}s: $(squeue -h -j "$j" -o '%T %P %N %r')"
        [ "$(squeue -h -j "$j" -o %T)" = "PENDING" ] && { normal "$j"; log "$j did not start on $gn — waits as a requeued unit (others stay soft-blocked)"; }
      fi
    fi
    sleep 30; continue
  fi
  if [ -n "$W" ]; then sleep 60; else sleep "$POLL"; fi
done
