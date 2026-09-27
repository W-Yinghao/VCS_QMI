# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:31:35Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.75 | 0.75 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | 0.0372 ± 0.0796 | 0.592 | 0.558 |
| s0.75 | 0.75 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.90 | 0.0956 ± 0.0426 | 0.419 | 0.601 |
| s0.75 | 0.75 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1508 ± 0.0322 | 0.265 | 0.649 |
| s0.75 | 0.75 | 1000 | 100 | 0.46 | 0.00 | 0.04 | 0.52 | 1.00 | 1.00 | 1.00 | 0.1839 ± 0.0173 | 0.187 | 0.670 |
| s0.75 | 0.75 | 2000 | 30 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2031 ± 0.0097 | 0.132 | 0.685 |
| s0.75 | 0.75 | 5000 | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2295 ± 0.0074 | 0.084 | 0.704 |
