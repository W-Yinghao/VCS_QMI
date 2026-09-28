# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-28T01:02:29Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 1000). **Null N re-drawn per repeat (--redraw-null-n, addendum 6).**

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 10000 | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 | 0.01 | -0.0002 ± 0.0004 | 0.059 | 0.500 |
