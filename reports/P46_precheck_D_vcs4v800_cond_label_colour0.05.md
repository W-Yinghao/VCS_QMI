# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T13:15:13Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_label_colour0.05 | 0.05 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 0.12 | 0.00 | -0.0034 ± 0.0060 | 0.265 | 0.502 |
| cond_label_colour0.05 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.88 | | | |
| cond_label_colour0.05 | 0.05 | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.96 | 0.50 | 0.06 | 0.0008 ± 0.0019 | 0.132 | 0.505 |
| cond_label_colour0.05 (unconditional test) | | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
