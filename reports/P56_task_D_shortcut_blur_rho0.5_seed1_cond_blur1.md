# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.5_seed1, model.pt) — 2026-09-27T20:48:13Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.46 | 0.14 | 0.0275 ± 0.0348 | 0.419 | 0.511 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 0.38 | 0.12 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.98 | 0.72 | 0.0647 ± 0.0156 | 0.265 | 0.554 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.92 | 0.78 | | | |
| cond_blur1 | 1 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | 0.0989 ± 0.0150 | 0.187 | 0.609 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | | | |
