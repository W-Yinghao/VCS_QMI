# Pre-registration — the method's ceiling on CIFAR-10 SSL, compute unconstrained (P43), frozen 2026-09-26 before launch

Owner (2026-09-26): "现在先不要考虑步数、计算预算，我们就看这个新的方法在 SSL 上能做到最好什么程度" — stop matching compute or steps; find the best
the method reaches.  Recipe fixed by P19–P40: cosine critic tanh(a⟨z1,z2⟩+b), K = 8, negative partner detached, a0 = 5, multi-view J
(averaged over all view pairs), AdamW 1e-3, warm-up 10, cosine → 1 %, wd 1e-4, projector 512/128.  Best so far: 2 views / 800 ep 85.54;
4 views / 800 ep running (kNN 84.16 at epoch 400, above the 2-view 800-ep endpoint and above the SimCLR control).

The remaining levers that raised the ceiling were schedule length and views (with enough steps).  Units (single seed; chained over the 23 h
walltime with `slurm/submit_chain.sh`, resume from `last.pt` at an epoch boundary — the trainer's SIGTERM handling and the unit script's
resume branch, both exercised before):
| unit | views | epochs | B | steps | ≈ wall | question |
|---|---|---|---|---|---|---|
| a5_views4_1600ep | 4 | 1600 | 256 | 281 k | 30 h (2 links) | does the schedule gain continue beyond 800 epochs? (2-view: 200 → 800 gave +4) |
| a5_views8_800ep | 8 | 800 | 256 | 140 k | 27 h (2 links) | with a long schedule, do 28 pairs per image beat 6? (P38's 8-view deficit was a steps effect if P37's new units confirm) |
| a5_views4_b128_800ep | 4 | 800 | 128 | 281 k | 18 h (2 links) | twice the updates of the 4-view 800-ep run at the same epochs |
Running already and part of the same question: `P35_vcs_a5_views4_800ep` (seeds 0/1/2), `P35_vcs_a5_views4_400ep`, `P35_vcs_a5_800ep` (seeds 1/2),
`P37_vcs_a5_views8_200ep`, `P37_vcs_a5_views8_b128_100ep`.

## Reading
Absolute linear-val / kNN / h-rank at the final checkpoint plus the kNN curve at every logged epoch; no HELPS/HURTS threshold — the question
is the maximum.  A run whose kNN curve has flattened before its end (last two points within 0.2) is read as "schedule saturated".  The best
configuration gets seeds 1/2 afterwards (owner's rule: the base gets seeds).  Selection split as always; official test once at the very end.
Configs and sha256: `configs/HPARAM_O_SHA256.json`; job ids (all chain links) in `job_ids.json`.
