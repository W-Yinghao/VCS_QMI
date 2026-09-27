# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:34:55Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.5 | 0.5 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.38 | 0.29 | 0.00 | -0.0242 ± 0.0724 | 0.592 | 0.506 |
| s0.5 | 0.5 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.82 | 0.84 | 0.05 | -0.0053 ± 0.0370 | 0.419 | 0.509 |
| s0.5 | 0.5 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 1.00 | 0.34 | 0.0141 ± 0.0227 | 0.265 | 0.520 |
| s0.5 | 0.5 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.93 | 0.0355 ± 0.0114 | 0.187 | 0.545 |
| s0.5 | 0.5 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0503 ± 0.0083 | 0.132 | 0.564 |
| s0.5 | 0.5 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0644 ± 0.0057 | 0.084 | 0.595 |
