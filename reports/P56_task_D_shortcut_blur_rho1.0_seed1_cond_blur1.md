# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho1_seed1, model.pt) — 2026-09-28T01:40:42Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.2571 ± 0.0363 | 0.419 | 0.713 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.84 | 0.02 | 0.78 | 0.80 | 1.00 | 1.00 | 1.00 | 0.2811 ± 0.0201 | 0.265 | 0.723 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.58 | 0.04 | 0.54 | 0.56 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 1000 | 50 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2918 ± 0.0157 | 0.187 | 0.728 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | | | |
