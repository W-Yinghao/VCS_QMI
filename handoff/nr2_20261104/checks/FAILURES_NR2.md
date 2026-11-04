Publication copy (sample names aliased; numbers and conclusions unchanged).

# NR2 technical failures (kept on record; no tolerance was loosened)

| job | stage | cause | fix |
|---|---|---|---|
| 1086421 | real_checks, after 9/9 PASS | check-script bug: invariance baseline dict keyed by objective only (KeyError-type unpack) | key by (objective, mode); production code unchanged |
| 1086422 | real_checks, after 14/14 PASS, trainer resume smoke | PRODUCTION bug in code/src/nr2/train.py resume: torch.load(map_location=cuda) moved saved CUDA RNG states to GPU; torch.cuda.set_rng_state_all needs CPU ByteTensor (same line exists in NR1 train.py, never exercised there because NR1 units never resumed) | `.cpu()` on restored CUDA RNG states (NR2 trainer only; NR1 files untouched); `.detach()` in step-0 contract read |
| 1086535 | nr2c_prep, FIT Top100 table build (after extraction/calibration/baseline OK) | build_base_topk guard: exact-score tie at position 100 extends beyond the 28-item window (COCO has up to 159 identical captions) | site/build_tables_nr2c.py passes margin=400 to the unchanged build_base_topk (same tie rule and guard); guard not loosened |
