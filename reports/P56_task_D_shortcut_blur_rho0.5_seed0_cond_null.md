# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.5_seed0, model.pt) — 2026-09-27T20:46:59Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_null | 0 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.08 | -0.0251 ± 0.0437 | 0.419 | 0.509 |
| cond_null (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 | 0.00 | | | |
| cond_null | 0 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.02 | -0.0094 ± 0.0133 | 0.265 | 0.507 |
| cond_null (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.02 | 0.04 | | | |
| cond_null | 0 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | -0.0046 ± 0.0072 | 0.187 | 0.503 |
| cond_null (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.02 | | | |
