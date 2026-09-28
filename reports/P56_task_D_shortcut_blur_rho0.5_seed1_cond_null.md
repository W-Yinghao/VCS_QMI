# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.5_seed1, model.pt) — 2026-09-27T20:54:57Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_null | 0 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.06 | 0.04 | -0.0149 ± 0.0303 | 0.419 | 0.506 |
| cond_null (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.08 | 0.00 | | | |
| cond_null | 0 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.02 | 0.02 | -0.0077 ± 0.0170 | 0.265 | 0.503 |
| cond_null (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.04 | 0.00 | | | |
| cond_null | 0 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 0.04 | 0.02 | -0.0026 ± 0.0070 | 0.187 | 0.504 |
| cond_null (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.04 | 0.02 | | | |
