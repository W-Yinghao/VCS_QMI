# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.8_seed1, model.pt) — 2026-09-27T16:20:12Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_null | 0 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.06 | -0.0154 ± 0.0272 | 0.419 | 0.508 |
| cond_null (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.12 | 0.04 | | | |
| cond_null | 0 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.10 | 0.04 | -0.0066 ± 0.0116 | 0.265 | 0.505 |
| cond_null (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 | 0.02 | | | |
| cond_null | 0 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.10 | 0.02 | -0.0029 ± 0.0078 | 0.187 | 0.503 |
| cond_null (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | | | |
