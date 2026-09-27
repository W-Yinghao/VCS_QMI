# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed1, epoch_800.pt) — 2026-09-27T15:00:37Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.2 | 0.2 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.80 | 0.85 | 0.03 | -0.0184 ± 0.0664 | 0.592 | 0.512 |
| s0.2 | 0.2 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.24 | 0.0183 ± 0.0387 | 0.419 | 0.524 |
| s0.2 | 0.2 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.88 | 0.0603 ± 0.0254 | 0.265 | 0.560 |
| s0.2 | 0.2 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0907 ± 0.0146 | 0.187 | 0.595 |
| s0.2 | 0.2 | 2000 | 30 | 0.03 | 0.00 | 0.00 | 0.03 | 1.00 | 1.00 | 1.00 | 0.1130 ± 0.0097 | 0.132 | 0.619 |
| s0.2 | 0.2 | 5000 | 30 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1335 ± 0.0071 | 0.084 | 0.645 |
