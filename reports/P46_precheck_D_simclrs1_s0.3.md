# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed1, epoch_200.pt) — 2026-09-27T18:52:27Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.3 | 0.3 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.09 | 0.01 | -0.0253 ± 0.0461 | 0.592 | 0.503 |
| s0.3 | 0.3 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.23 | 0.00 | -0.0132 ± 0.0241 | 0.419 | 0.502 |
| s0.3 | 0.3 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.64 | 0.49 | 0.01 | -0.0049 ± 0.0179 | 0.265 | 0.504 |
| s0.3 | 0.3 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 0.91 | 0.14 | 0.0027 ± 0.0067 | 0.187 | 0.508 |
| s0.3 | 0.3 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.43 | 0.0114 ± 0.0041 | 0.132 | 0.513 |
| s0.3 | 0.3 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0255 ± 0.0030 | 0.084 | 0.552 |
