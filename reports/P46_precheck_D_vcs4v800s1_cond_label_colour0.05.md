# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed1, epoch_800.pt) — 2026-09-27T14:59:56Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_label_colour0.05 | 0.05 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.16 | 0.00 | -0.0049 ± 0.0065 | 0.265 | 0.503 |
| cond_label_colour0.05 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.92 | | | |
| cond_label_colour0.05 | 0.05 | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.92 | 0.76 | 0.04 | 0.0006 ± 0.0020 | 0.132 | 0.504 |
| cond_label_colour0.05 (unconditional test) | | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
