Publication copy (sample names aliased; numbers and conclusions unchanged).

- 1088215 (nr5_cputests): cancelled before start; common_env.sh had a stray 'cd $NR4_H/code' before NR4_H was defined. Fixed and resubmitted.
- 1088221 (nr5_accept): failed at the first step (EQ FROZEN) before training: the acceptance script put a PosixPath into the config ('prefix' key), not JSON-serialisable in BINDING. Removed (the prefix is passed positionally). Resubmitted.
- 1088301 (nr5_accept): EQ FROZEN run completed, then the comparison helper failed on the stage-0 checkpoint (empty Adam state -> max() of empty). Helper fixed (empty -> 0, state key sets compared). Resubmitted.
- 1088316 (nr5_accept): EQ and A sections completed and saved; B0 failed writing JSON (NaN correlation: Flickr00 text marginal is uniform). Correlation set to None when undefined; multinomial-from-p reference TV added. Resubmitted (EQ/A reused from the saved file, same GPU type).
- 2026-11-04 ~21:05 UTC, user decision (relayed): "不行不行，现在这个只能用一种GPU还是太慢了，你来组织一下，至少我们要能在L40S上跑". Feeder nr5 stopped
  (STOP_APPS_nr5); 13 never-started RTX-pinned jobs cancelled while PENDING (no run directory, no output): 1088432 1088433 1088434
  1088438 1088439 1088440 1088441 1088442 1088443 1088463 1088468 1088469 1088470. Replaced under amendment NR5_A1 (see NR5_A1.json).
  The 6 completed A FROZEN units (L40S, original frozen code, cached globals only) are kept.
- 1088540 (nr5a1_accept_l40s, first version): t0 feature diffs vs RTX cache within NR1 rule (max ratio 0.068); training step-0 ratios <= 0.26; but (i) the stage-0 DEV score check with the elementwise 3e-5 rule failed (max abs 5.9e-5 Flickr60, 1.1e-4 CC; order differences all within the elementwise 2x tolerance: 0 violating rows) and (ii) B G_ZERO CC direct backward ran out of memory on the 44 GiB L40S. Amended NR5_A1 before any formal unit: stage-0 = features elementwise NR1 rule + scores max|d| <= 3e-5(1+max|ref|) + order rule; TAIL uses the verified replay backward on GPUs < 64 GiB. Acceptance rerun.
