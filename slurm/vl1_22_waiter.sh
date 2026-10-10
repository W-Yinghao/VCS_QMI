#!/usr/bin/env bash
# VL1-22 chain: VL1-21 eval (gate D) -> Table C smoke -> the 12 Table C jobs.  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/feed_normal.log; EV=reports/VL1/VL1_21_detector_eval.json
P=/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals; T=/home/infres/yinwang/CS_QMI/outputs/VL1_22
until [ -f $EV ]; do sleep 300; done
python3 -c "import json,sys; sys.exit(0 if json.load(open('$EV'))['gate_AP_ge_0.33'] else 1)" || { echo "$(date -u +%FT%TZ) VL1-22: gate D FAILED - Table C not run" >> $LOG; exit 1; }
for d in refcocog refcoco refcocoplus; do [ -f $P/${d}_DEV.pt ] || { echo "$(date -u +%FT%TZ) VL1-22: missing $P/${d}_DEV.pt" >> $LOG; exit 1; }; done
echo "sbatch --parsable --time=01:00:00 --export=ALL,VL_DATASET=refcocog --job-name=vl1_22_smoke slurm/vl1_22_tableC.sbatch --backbone clip_b16 --seeds 0 --smoke" > slurm/vl1_22_smoke.txt
bash slurm/feed_normal.sh slurm/vl1_22_smoke.txt
until ls $L/vl1_22_smoke_*.out >/dev/null 2>&1 && grep -q "^exit=" $L/vl1_22_smoke_*.out; do sleep 120; done
grep -q "^exit=0" $L/vl1_22_smoke_*.out && [ -f $T/refcocog_clip_b16_vcs_s0_smoke.json ] || { echo "$(date -u +%FT%TZ) VL1-22: smoke FAILED - not fed" >> $LOG; exit 1; }
bash slurm/feed_normal.sh slurm/vl1_22_feed.txt
