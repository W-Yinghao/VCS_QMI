#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13; R7-E1 production_gate, D-E1-4, D-E1-8 as read in execution note e1_2): wait for the HiPrune validation smoke;
# feed its 12 frozen unit lines (feed_gated.sh, CAP 27, GATE 10; Nice 300 via the running nice_filler) only if SMOKECHECK passes, both attention
# classes are FlashAttention2, and the steady-state per-item time (ms_median) is within 1.5 x the frozen estimate; otherwise hold for the coordinator.
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/after_e1_smokes.log; E=/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40/eval_e1; S=$L/r7e1_smoke_hiprune_1037517.out
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$ (hiprune only; mmtok fed by the coordinator)"
until [ -f $S ] && grep -qE "^exit=|CANCELLED|TIME LIMIT" $S; do sleep 30; done
v=$(/home/infres/yinwang/CS_QMI/env/bin/python -I - "$S" 2>&1 <<'PY'
import json, sys
t = open(sys.argv[1]).read(); est = {"gqa_dev": 0.19, "chartqa_val": 0.36}
if "exit=0" not in t or "SMOKECHECK" not in t: print("HOLD no SMOKECHECK or non-zero exit"); sys.exit()
i = t.rindex("SMOKECHECK"); d, _ = json.JSONDecoder().raw_decode(t[t.index("{", i):])
if d.get("pass") is not True: print("HOLD SMOKECHECK pass is not true: " + json.dumps(d.get("problems"))[:300]); sys.exit()
bad = {}
for b, e in est.items():
    c = d["benches"][b]["configs"]["ext:hiprune:0.25"]; a = c.get("attn_built", {})
    if "FlashAttention2" not in str(a.get("vision_block0")) or "FlashAttention2" not in str(a.get("llm_layer0")): bad[b + "_attn"] = a
    if c.get("ms_median") is None or c["ms_median"] / 1000 > 1.5 * e: bad[b + "_rate"] = c.get("ms_median")
print("HOLD " + json.dumps(bad) if bad else "GO " + json.dumps({b: d["benches"][b]["configs"]["ext:hiprune:0.25"]["ms_median"] for b in est}))
PY
)
say "hiprune smoke $(grep -E '^exit=' $S | tail -1): $v"
if [[ "$v" == GO* ]]; then
  grep -E "job-name=r7e1_hiprune_" $E/r7_e1_lines.txt > $E/feed_hiprune.txt; say "hiprune: $(wc -l < $E/feed_hiprune.txt) lines -> feed_gated"
  cd /home/infres/yinwang/CS_QMI/ssl_pilot && CAP=27 GATE=10 nohup setsid bash slurm/feed_gated.sh $E/feed_hiprune.txt >/dev/null 2>&1 &
fi; say "done"
