# P112 addendum 1 — follow-up seeds by the frozen rule — FROZEN 2026-10-03T08:07:57Z

Screen (seed 0, linear-val; G2 (2, κ 0.5) references): CIFAR-10 standard — (1.5, 0.5) 88.06, (3, 0.5) 88.48, (2, 0.25) 88.72, (2, 0.75) 88.64 vs G2 88.66;
strong augmentation — (2, 0.25) 87.88, **(2, 0.75) 88.22**, (3, 0.5) 86.92 vs G2-strong 86.86; CIFAR-100 — (2, 0.25) 59.84, (2, 0.75) 58.34 vs G2 59.46,
(3, 0.5) still running.  Frozen rule: a cell beating its setting's G2 reference by ≥ 0.5 linear gets seeds 1–2 → **strong (2, 0.75)** (+1.36) and
**strong (2, 0.25)** (+1.02).  No CIFAR-10-standard or CIFAR-100 cell qualifies so far (CIFAR-100 (3, 0.5) is judged when it completes).
Submitted: those two cells × seeds 1, 2 (each = the P111 G2-strong config of that seed with only (a, b) changed; verified by diff).  Paired reading vs
P111 G2-strong seeds 0–2 (86.86 / 87.70 / 87.80) when complete.  (These became eligible while the session was down; submitted at the next check.)
