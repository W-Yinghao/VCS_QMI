Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR5 site notes
- Design registered by the main session under the user quote 「下来这两个你都自己来分析分析结果，设计新一轮的方向，不要让GPU空着」 (reason in PROTOCOL_FROZEN_CR5.json).
- Matrix 48 = {Flickr60, CC152K_RECOVERED} x {OBS, J_MASS_OBS, BASE/J/BCE/NCE_KEEP_TOP1} x {RECT, COMPLETE_PC} x seeds {1002, 1003};
  seed rule: head seed = s, sampler seed = 20261104201 + (s - 1001); s1001 taken from CR4 (PC cells + same-group RECT refs_rerun).
- code/site/cr5_student.py: seeded copy of the CR3 RECT step and CR4 PC step (CR3/CR4 hard-code seed 1001); acceptance R1/R2 checks that
  seed 1001 reproduces cr3_student RECT and cr4_student PC bitwise in the same job.
- One job per condition x seed (CR3_A1/CR4_A1 rule): each RECT/PC pair on the same GPU. CC OBS/J_MASS_OBS PC: single-positive check +
  250-step PC vs same-job RECT bitwise -> reuse RECT, else train new.
- Driver (CPU, no --dependency): acceptance ALL_PASS -> freeze -> CPU dry-run -> 4 group jobs -> summarize_cr5.py (RESULTS_CR5.md).
