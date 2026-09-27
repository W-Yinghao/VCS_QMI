# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed1, epoch_200.pt) — 2026-09-27T18:50:32Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.1 | 0.1 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.06 | 0.00 | -0.0275 ± 0.0479 | 0.592 | 0.500 |
| s0.1 | 0.1 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.04 | 0.02 | -0.0156 ± 0.0239 | 0.419 | 0.503 |
| s0.1 | 0.1 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.06 | 0.00 | -0.0067 ± 0.0109 | 0.265 | 0.502 |
| s0.1 | 0.1 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.12 | 0.01 | -0.0031 ± 0.0046 | 0.187 | 0.501 |
| s0.1 | 0.1 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.30 | 0.00 | -0.0010 ± 0.0023 | 0.132 | 0.500 |
| s0.1 | 0.1 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.80 | 0.13 | 0.0010 ± 0.0008 | 0.084 | 0.504 |
