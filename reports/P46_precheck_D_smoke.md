# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T08:25:04Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 20; critic/classifier: MLP(512+2→128→128→1), 50 Adam steps; repeats 3 (n ≥ 2000: 3).

| case | s | n | R | vcs_hoeff | vcs_perm | hsic_perm | c2st | mean J_eval | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 100 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | -0.0831 ± 0.0577 | 0.592 | 0.507 |
| s0 | 0 | 200 | 3 | 0.00 | 0.33 | 0.00 | 0.00 | -0.0530 ± 0.0275 | 0.419 | 0.514 |
| s0.1 | 0.1 | 100 | 3 | 0.00 | 0.67 | 0.00 | 0.00 | -0.0273 ± 0.0113 | 0.592 | 0.510 |
| s0.1 | 0.1 | 200 | 3 | 0.00 | 0.33 | 0.00 | 0.00 | -0.0461 ± 0.0098 | 0.419 | 0.519 |
| cond_label_only | 0 | 200 | 3 | 0.00 | 0.33 | 0.33 | 0.00 | -0.0233 ± 0.0105 | 0.419 | 0.499 |
| cond_label_only (unconditional test) | | 200 | 3 | 0.00 | 1.00 | 1.00 | 0.67 | | | |
| cond_label_colour0.05 | 0.05 | 200 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | -0.0282 ± 0.0133 | 0.419 | 0.503 |
| cond_label_colour0.05 (unconditional test) | | 200 | 3 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
