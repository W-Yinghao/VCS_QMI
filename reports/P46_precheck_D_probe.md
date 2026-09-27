# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T08:31:50Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 100; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 3 (n ≥ 2000: 3).

| case | s | n | R | vcs_hoeff | vcs_perm | hsic_perm | c2st | mean J_eval | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|
| s0.1 | 0.1 | 1000 | 3 | 0.00 | 1.00 | 1.00 | 1.00 | -0.3296 ± 0.0169 | 0.187 | 0.569 |
| s0.1 | 0.1 | 5000 | 3 | 0.00 | 1.00 | 1.00 | 1.00 | -0.3092 ± 0.0105 | 0.084 | 0.576 |
| cond_label_colour0.05 | 0.05 | 2000 | 3 | 0.00 | 1.00 | 0.67 | 0.33 | -0.3908 ± 0.0240 | 0.132 | 0.512 |
| cond_label_colour0.05 (unconditional test) | | 2000 | 3 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
