# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:33:46Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 1 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.98 | 0.1331 ± 0.0695 | 0.592 | 0.659 |
| s1 | 1 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1817 ± 0.0470 | 0.419 | 0.680 |
| s1 | 1 | 500 | 100 | 0.10 | 0.00 | 0.00 | 0.11 | 1.00 | 1.00 | 1.00 | 0.2313 ± 0.0292 | 0.265 | 0.692 |
| s1 | 1 | 1000 | 100 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2539 ± 0.0174 | 0.187 | 0.703 |
| s1 | 1 | 2000 | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2682 ± 0.0088 | 0.132 | 0.718 |
| s1 | 1 | 5000 | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2882 ± 0.0067 | 0.084 | 0.728 |
