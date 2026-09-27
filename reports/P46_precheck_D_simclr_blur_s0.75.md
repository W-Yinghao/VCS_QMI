# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-27T18:46:19Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.75 | 0.75 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.39 | 0.31 | 0.00 | -0.0273 ± 0.0560 | 0.592 | 0.507 |
| s0.75 | 0.75 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.78 | 0.80 | 0.03 | -0.0154 ± 0.0528 | 0.419 | 0.511 |
| s0.75 | 0.75 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.48 | 0.0199 ± 0.0211 | 0.265 | 0.526 |
| s0.75 | 0.75 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.98 | 0.0405 ± 0.0142 | 0.187 | 0.557 |
| s0.75 | 0.75 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0679 ± 0.0088 | 0.132 | 0.583 |
| s0.75 | 0.75 | 5000 | 30 | 0.80 | 0.07 | 0.07 | 0.80 | 1.00 | 1.00 | 1.00 | 0.0905 ± 0.0070 | 0.084 | 0.617 |
