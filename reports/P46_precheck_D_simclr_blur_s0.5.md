# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-27T18:42:21Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.5 | 0.5 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | 0.07 | 0.00 | -0.0311 ± 0.0542 | 0.592 | 0.505 |
| s0.5 | 0.5 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.09 | 0.00 | -0.0228 ± 0.0489 | 0.419 | 0.504 |
| s0.5 | 0.5 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.45 | 0.30 | 0.01 | -0.0052 ± 0.0180 | 0.265 | 0.504 |
| s0.5 | 0.5 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.94 | 0.79 | 0.08 | 0.0002 ± 0.0060 | 0.187 | 0.506 |
| s0.5 | 0.5 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.97 | 0.33 | 0.0049 ± 0.0040 | 0.132 | 0.510 |
| s0.5 | 0.5 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.97 | 0.0122 ± 0.0032 | 0.084 | 0.530 |
