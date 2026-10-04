#!/usr/bin/env bash
# Owner 2026-10-04 "如果你看到PRO6000有可用slot，就把L40S上的转过去".  Every POLL s: if an RTX6000PRO GPU is idle (its node also has 8 CPUs and
# 40 GB free) and one of my training units runs on L40S with >= MIN_REMAIN epochs left, move the unit with the most epochs left:
#   1. my pending jobs temporarily drop L40S from their partition list (a partition update keeps EligibleTime; hold/release would reset it), so the
#      quota slot freed in step 2 cannot go to an L40S start;
#   2. requeue the unit (same job id) restricted to RTX6000PRO — it resumes from last.pt at an epoch boundary (atomic save, TERM forwarded by run_fg);
#   3. once the unit runs, or after WAIT s, every pending job (the unit too, if still pending) gets L40S back.
# Run on the login node: nohup setsid bash slurm/l40s_to_pro.sh >/dev/null 2>&1 &   — log slurm_logs/l40s_to_pro.log; stop by PID (never pkill -f).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
POLL=${POLL:-300}; MIN_REMAIN=${MIN_REMAIN:-100}; WAIT=${WAIT:-420}
LOGS=/home/infres/yinwang/CS_QMI/slurm_logs; OUT=/home/infres/yinwang/CS_QMI/outputs; LOG=$LOGS/l40s_to_pro.log
PY=/home/infres/yinwang/CS_QMI/env/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

free_pro() {  # idle RTX6000PRO GPUs on usable nodes with room for one unit
  local n=0 node info tot used ca ct rm am
  for node in $(sinfo -h -p RTX6000PRO -N -o "%N" | sort -u); do
    info=$(scontrol show node "$node")
    grep -qE "State=[^ ]*(DRAIN|DOWN|MAINT|RESERVED|FAIL)" <<<"$info" && continue
    tot=$(grep -o "Gres=gpu:[^ ]*" <<<"$info" | grep -oE "[0-9]+(\(|$)" | head -1 | tr -d '(')
    used=$(grep -o "AllocTRES=[^ ]*" <<<"$info" | grep -o "gres/gpu=[0-9]*" | grep -o "[0-9]*$")
    ca=$(grep -oE "CPUAlloc=[0-9]+" <<<"$info" | cut -d= -f2); ct=$(grep -oE "CPUTot=[0-9]+" <<<"$info" | cut -d= -f2)
    rm=$(grep -oE "RealMemory=[0-9]+" <<<"$info" | cut -d= -f2); am=$(grep -oE "AllocMem=[0-9]+" <<<"$info" | cut -d= -f2)
    [ $(( ${tot:-0} - ${used:-0} )) -ge 1 ] && [ $(( ct - ca )) -ge 8 ] && [ $(( rm - am )) -ge 40960 ] && n=$((n + 1))
  done
  echo $n
}
run_id_of() { grep -h -m1 -oE "unit [^ ]+: cfg" "$LOGS"/unit_*_"$1".out 2>/dev/null | head -1 | awk '{print $2}' | tr -d ':'; }
remaining_of() { "$PY" -c "import json,yaml;s=json.load(open('$OUT/$1/status.json'));c=yaml.safe_load(open('$OUT/$1/config.resolved.yaml'));print(int(c['train']['epochs'])-int(s['completed_epoch']))" 2>/dev/null; }
parts_of() { squeue -h -j "$1" -o "%.80P" 2>/dev/null | tr -d ' '; }

log "start pid $$ poll=${POLL}s min_remain=${MIN_REMAIN} wait=${WAIT}s"
while :; do
  f=$(free_pro)
  if [ "${f:-0}" -ge 1 ]; then
    best=""; bestrem=0; bestrid=""
    for jid in $(squeue -h -u "$USER" -t R -p L40S -o "%i"); do
      rid=$(run_id_of "$jid"); [ -z "$rid" ] && continue
      rem=$(remaining_of "$rid"); [ -z "$rem" ] && continue
      [ "$rem" -gt "$bestrem" ] && { best=$jid; bestrem=$rem; bestrid=$rid; }
    done
    if [ -n "$best" ] && [ "$bestrem" -ge "$MIN_REMAIN" ]; then
      pend=""
      for p in $(squeue -h -u "$USER" -t PD -o "%i"); do
        s=$(parts_of "$p"); [[ "$s" == *L40S* ]] || continue
        ns=$(sed -e 's/,L40S//' -e 's/L40S,//' <<<"$s"); [ -n "$ns" ] && [ "$ns" != "L40S" ] && scontrol update jobid="$p" Partition="$ns" && pend="$pend $p"
      done
      log "free RTX6000PRO GPUs=$f: moving $best ($bestrid, $bestrem epochs left) off L40S; L40S dropped from $(wc -w <<<"$pend") pending jobs"
      if scontrol requeue "$best"; then
        for i in $(seq 1 20); do [ "$(squeue -h -j "$best" -o %T)" = "PENDING" ] && break; sleep 3; done
        scontrol update jobid="$best" Partition=RTX6000PRO
        t=0; while [ $t -lt "$WAIT" ]; do sleep 30; t=$((t + 30)); [ "$(squeue -h -j "$best" -o %T)" = "RUNNING" ] && break; done
        log "$best after ${t}s: $(squeue -h -j "$best" -o '%T %P %N %r')"
      else
        log "requeue of $best refused"
      fi
      for p in $pend $best; do
        s=$(squeue -h -j "$p" -t PD -o "%.80P" 2>/dev/null | tr -d ' '); [ -z "$s" ] && continue
        if [ "$p" = "$best" ]; then scontrol update jobid="$p" Partition=RTX6000PRO,H100,L40S
        elif [[ "$s" != *L40S* ]]; then scontrol update jobid="$p" Partition="$s,L40S"; fi
      done
      log "L40S restored on pending jobs"
    fi
  fi
  sleep "$POLL"
done
