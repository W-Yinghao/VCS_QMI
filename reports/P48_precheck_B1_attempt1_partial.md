# Pre-check B1 — closed-form linear-class critic vs trained tanh critic on frozen features — 2026-09-27T08:19:24Z

5 000 selection images, two train-distribution views (fixed RNG), 2 500 fit / 2 500 eval pairs, K = 8 cyclic-shift negatives per pair in each split. J values are on the eval split; closed-form T = w*ᵀφ has no tanh (|T| > 1 fraction reported in the JSON); tanh(c·w*ᵀφ) fits only the scalar c on the fit split.

| run | trained critic | class | dim | ridge λ | J_fit closed | J_eval closed | J_eval tanh(c·w*ᵀφ) | c | neural J_eval (same pairs) |
|---|---|---|---|---|---|---|---|---|---|
| P5_vcs_seed0 | ordered_concat | P1 | 129 | 1e-06 | 0.6859 | 0.6869 | 0.7803 | 2.57 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P1 | 129 | 0.0001 | 0.6776 | 0.6809 | 0.7807 | 2.57 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P1 | 129 | 0.01 | 0.6601 | 0.6655 | 0.7677 | 2.57 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P2 | 257 | 1e-06 | 0.8085 | 0.8051 | 0.8845 | 3.46 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P2 | 257 | 0.0001 | 0.8048 | 0.8049 | 0.8860 | 3.46 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P2 | 257 | 0.01 | 0.7914 | 0.7940 | 0.8766 | 2.98 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P3 | 513 | 1e-06 | 0.8170 | 0.8066 | 0.8844 | 3.46 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P3 | 513 | 0.0001 | 0.8111 | 0.8075 | 0.8866 | 3.46 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | P3 | 513 | 0.01 | 0.7923 | 0.7937 | 0.8774 | 2.98 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | H1 | 513 | 1e-06 | 0.7526 | 0.7362 | 0.8154 | 2.57 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | H1 | 513 | 0.0001 | 0.7417 | 0.7357 | 0.8094 | 2.57 | 0.8988 |
| P5_vcs_seed0 | ordered_concat | H1 | 513 | 0.01 | 0.6872 | 0.6905 | 0.7775 | 2.98 | 0.8988 |
