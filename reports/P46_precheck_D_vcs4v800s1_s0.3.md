# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed1, epoch_800.pt) — 2026-09-27T15:00:32Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.3 | 0.3 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.91 | 0.99 | 0.08 | -0.0085 ± 0.0743 | 0.592 | 0.521 |
| s0.3 | 0.3 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.49 | 0.0545 ± 0.0390 | 0.419 | 0.543 |
| s0.3 | 0.3 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.98 | 0.1049 ± 0.0260 | 0.265 | 0.596 |
| s0.3 | 0.3 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1377 ± 0.0156 | 0.187 | 0.632 |
| s0.3 | 0.3 | 2000 | 30 | 1.00 | 0.03 | 0.37 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1609 ± 0.0096 | 0.132 | 0.654 |
| s0.3 | 0.3 | 5000 | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1868 ± 0.0061 | 0.084 | 0.676 |
