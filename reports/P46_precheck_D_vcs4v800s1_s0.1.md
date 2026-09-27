# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed1, epoch_800.pt) — 2026-09-27T14:58:30Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0.1 | 0.1 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.24 | 0.01 | -0.0391 ± 0.0803 | 0.592 | 0.504 |
| s0.1 | 0.1 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.61 | 0.68 | 0.01 | -0.0161 ± 0.0439 | 0.419 | 0.507 |
| s0.1 | 0.1 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.95 | 1.00 | 0.15 | 0.0061 ± 0.0171 | 0.265 | 0.512 |
| s0.1 | 0.1 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.76 | 0.0211 ± 0.0109 | 0.187 | 0.530 |
| s0.1 | 0.1 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.97 | 0.0358 ± 0.0082 | 0.132 | 0.550 |
| s0.1 | 0.1 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0470 ± 0.0050 | 0.084 | 0.578 |
