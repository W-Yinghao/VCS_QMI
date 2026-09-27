# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T18:24:08Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.01 | 0.00 | -0.0357 ± 0.0757 | 0.592 | 0.502 |
| s0 | 0 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.02 | 0.00 | -0.0113 ± 0.0190 | 0.419 | 0.501 |
| s0 | 0 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.00 | -0.0059 ± 0.0093 | 0.265 | 0.501 |
| s0 | 0 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | 0.06 | 0.00 | -0.0032 ± 0.0047 | 0.187 | 0.501 |
| s0 | 0 | 2000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.07 | 0.00 | -0.0014 ± 0.0018 | 0.132 | 0.501 |
| s0 | 0 | 5000 | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.03 | 0.00 | -0.0003 ± 0.0005 | 0.084 | 0.500 |
