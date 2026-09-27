# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed1, epoch_800.pt) — 2026-09-27T14:54:50Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.05 | 0.05 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.08 | 0.00 | -0.0361 ± 0.0771 | 0.592 | 0.504 |
| s0.05 | 0.05 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.10 | 0.00 | -0.0249 ± 0.0500 | 0.419 | 0.504 |
| s0.05 | 0.05 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.36 | 0.40 | 0.00 | -0.0054 ± 0.0119 | 0.265 | 0.502 |
| s0.05 | 0.05 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.83 | 0.90 | 0.06 | -0.0008 ± 0.0055 | 0.187 | 0.505 |
| s0.05 | 0.05 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.17 | 0.0044 ± 0.0033 | 0.132 | 0.506 |
| s0.05 | 0.05 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.87 | 0.0086 ± 0.0025 | 0.084 | 0.521 |
