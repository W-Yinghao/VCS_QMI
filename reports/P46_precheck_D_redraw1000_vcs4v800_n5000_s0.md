# Pre-check D — independence-test power on planted nuisances (P35_vcs_a5_views4_800ep_seed0, epoch_800.pt) — 2026-09-27T23:22:26Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 1000). **Null N re-drawn per repeat (--redraw-null-n, addendum 6).**

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 5000 | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.04 | 0.01 | -0.0005 ± 0.0008 | 0.084 | 0.500 |
