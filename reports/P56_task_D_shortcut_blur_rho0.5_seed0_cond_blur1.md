# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.5_seed0, model.pt) — 2026-09-27T20:43:04Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.18 | 0.24 | 0.0118 ± 0.0496 | 0.419 | 0.524 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.14 | 0.18 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.94 | 0.70 | 0.0670 ± 0.0257 | 0.265 | 0.558 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.52 | 0.62 | | | |
| cond_blur1 | 1 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1037 ± 0.0172 | 0.187 | 0.617 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.98 | | | |
