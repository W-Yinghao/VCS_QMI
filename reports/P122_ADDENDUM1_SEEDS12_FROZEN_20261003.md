# P122 addendum 1 — encoder seeds 1–2 for the cells meeting the frozen positive-structural-gain rule — FROZEN 2026-10-03T20:08:37Z

Frozen rule (P122 prereg, freeze decision 2): a structural increment counts as positive only if its paired interval excludes 0 for both VCS and JS; positive →
encoder seeds 1–2.  Seed-0 result (`P122_V6_EVIDENCE_REPORT_20261003.md`): met by CIFAR-10 cells — under the common-strong law all five encoders (recipe VCS,
A-P3, A-P3 strong, SimCLR, SimCLR strong), under the common-standard law A-P3 strong, SimCLR strong and SimCLR (borderline); none on CIFAR-100.
**Submitted:** the same five CIFAR-10 encoder families at seeds 1 and 2, both measurement laws (every encoder qualifies under at least one law; running both
laws keeps the cross-evaluation complete).  Identical code, pools, indices, candidate library and budget as job 1020304; one GPU job.  Read per seed and pooled
over seeds 0–2 (encoder seed = the unit; EVAL pairs nest inside it).  No CIFAR-100 follow-up (no cell met the rule).
