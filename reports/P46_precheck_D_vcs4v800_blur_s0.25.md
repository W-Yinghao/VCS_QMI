# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:26:36Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.25 | 0.25 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.03 | 0.00 | -0.0255 ± 0.0573 | 0.592 | 0.501 |
| s0.25 | 0.25 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.06 | 0.00 | -0.0134 ± 0.0295 | 0.419 | 0.501 |
| s0.25 | 0.25 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.11 | 0.00 | -0.0057 ± 0.0090 | 0.265 | 0.501 |
| s0.25 | 0.25 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.32 | 0.24 | 0.00 | -0.0025 ± 0.0048 | 0.187 | 0.502 |
| s0.25 | 0.25 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.73 | 0.00 | -0.0004 ± 0.0021 | 0.132 | 0.502 |
| s0.25 | 0.25 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.97 | 1.00 | 0.20 | 0.0010 ± 0.0010 | 0.084 | 0.504 |
