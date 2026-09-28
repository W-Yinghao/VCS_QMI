# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.95_seed0, model.pt) — 2026-09-27T20:59:06Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.2057 ± 0.0435 | 0.419 | 0.689 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.24 | 0.02 | 0.16 | 0.26 | 1.00 | 1.00 | 1.00 | 0.2503 ± 0.0235 | 0.265 | 0.708 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.18 | 0.00 | 0.10 | 0.20 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 1000 | 50 | 1.00 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2667 ± 0.0145 | 0.187 | 0.715 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | | | |
