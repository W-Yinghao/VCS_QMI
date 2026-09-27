# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:34:37Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_label_only | 0 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.02 | 0.00 | -0.0037 ± 0.0061 | 0.265 | 0.501 |
| cond_label_only (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.80 | | | |
| cond_label_only | 0 | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.04 | 0.00 | -0.0004 ± 0.0008 | 0.132 | 0.501 |
| cond_label_only (unconditional test) | | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
