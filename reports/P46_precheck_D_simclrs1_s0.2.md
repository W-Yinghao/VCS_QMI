# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed1, epoch_200.pt) — 2026-09-27T18:51:33Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.2 | 0.2 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.03 | 0.00 | -0.0270 ± 0.0488 | 0.592 | 0.503 |
| s0.2 | 0.2 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.09 | 0.00 | -0.0149 ± 0.0242 | 0.419 | 0.501 |
| s0.2 | 0.2 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.32 | 0.19 | 0.00 | -0.0051 ± 0.0098 | 0.265 | 0.502 |
| s0.2 | 0.2 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.52 | 0.03 | -0.0014 ± 0.0050 | 0.187 | 0.502 |
| s0.2 | 0.2 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.90 | 0.07 | 0.0033 ± 0.0028 | 0.132 | 0.504 |
| s0.2 | 0.2 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.83 | 0.0103 ± 0.0018 | 0.084 | 0.522 |
