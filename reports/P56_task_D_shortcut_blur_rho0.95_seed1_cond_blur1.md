# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho0.95_seed1, model.pt) — 2026-09-28T01:36:53Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.2325 ± 0.0394 | 0.419 | 0.707 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.52 | 0.02 | 0.40 | 0.52 | 1.00 | 1.00 | 1.00 | 0.2681 ± 0.0212 | 0.265 | 0.717 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.32 | 0.00 | 0.26 | 0.38 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 1000 | 50 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2817 ± 0.0146 | 0.187 | 0.721 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | | | |
