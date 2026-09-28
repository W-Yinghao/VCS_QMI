# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_blur_rho1_seed0, model.pt) — 2026-09-28T01:39:54Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_blur1 | 1 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.2522 ± 0.0379 | 0.419 | 0.712 |
| cond_blur1 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 500 | 50 | 0.72 | 0.04 | 0.70 | 0.72 | 1.00 | 1.00 | 1.00 | 0.2765 ± 0.0264 | 0.265 | 0.727 |
| cond_blur1 (unconditional test) | | 500 | 50 | 0.76 | 0.04 | 0.76 | 0.70 | 1.00 | 1.00 | 1.00 | | | |
| cond_blur1 | 1 | 1000 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2926 ± 0.0135 | 0.187 | 0.730 |
| cond_blur1 (unconditional test) | | 1000 | 50 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | | | |
