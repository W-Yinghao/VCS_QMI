# Pre-check B1 — closed-form linear-class critic vs trained tanh critic on frozen features — 2026-09-27T08:25:34Z

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
| P18_vcs_crit_cosine_seed0 | cosine | P1 | 129 | 1e-06 | 0.8737 | 0.8740 | 0.9656 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P1 | 129 | 0.0001 | 0.8737 | 0.8741 | 0.9656 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P1 | 129 | 0.01 | 0.8690 | 0.8713 | 0.9648 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P2 | 257 | 1e-06 | 0.8762 | 0.8747 | 0.9682 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P2 | 257 | 0.0001 | 0.8762 | 0.8747 | 0.9686 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P2 | 257 | 0.01 | 0.8687 | 0.8688 | 0.9721 | 6.24 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P3 | 513 | 1e-06 | 0.8801 | 0.8746 | 0.9671 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P3 | 513 | 0.0001 | 0.8801 | 0.8747 | 0.9675 | 5.38 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | P3 | 513 | 0.01 | 0.8736 | 0.8710 | 0.9721 | 6.24 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | H1 | 513 | 1e-06 | 0.8405 | 0.8280 | 0.9323 | 4.64 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | H1 | 513 | 0.0001 | 0.8395 | 0.8303 | 0.9323 | 4.64 | 0.9694 |
| P18_vcs_crit_cosine_seed0 | cosine | H1 | 513 | 0.01 | 0.7592 | 0.7630 | 0.9041 | 5.38 | 0.9694 |
| P24_vcs_cos_negdetach_seed0 | cosine | P1 | 129 | 1e-06 | 0.8009 | 0.7967 | 0.9016 | 3.46 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P1 | 129 | 0.0001 | 0.7990 | 0.7953 | 0.8988 | 3.46 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P1 | 129 | 0.01 | 0.7599 | 0.7613 | 0.8839 | 4.00 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P2 | 257 | 1e-06 | 0.8628 | 0.8578 | 0.9498 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P2 | 257 | 0.0001 | 0.8624 | 0.8583 | 0.9502 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P2 | 257 | 0.01 | 0.8536 | 0.8510 | 0.9467 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P3 | 513 | 1e-06 | 0.8700 | 0.8599 | 0.9504 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P3 | 513 | 0.0001 | 0.8672 | 0.8617 | 0.9523 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | P3 | 513 | 0.01 | 0.8538 | 0.8508 | 0.9465 | 4.64 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | H1 | 513 | 1e-06 | 0.8351 | 0.8204 | 0.9066 | 3.46 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | H1 | 513 | 0.0001 | 0.8338 | 0.8231 | 0.9076 | 3.46 | 0.9267 |
| P24_vcs_cos_negdetach_seed0 | cosine | H1 | 513 | 0.01 | 0.7827 | 0.7808 | 0.8779 | 4.00 | 0.9267 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P1 | 129 | 1e-06 | 0.8658 | 0.8634 | 0.9572 | 5.38 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P1 | 129 | 0.0001 | 0.8655 | 0.8634 | 0.9567 | 5.38 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P1 | 129 | 0.01 | 0.6815 | 0.6830 | 0.9469 | 8.38 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P2 | 257 | 1e-06 | 0.8942 | 0.8906 | 0.9771 | 7.23 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P2 | 257 | 0.0001 | 0.8941 | 0.8909 | 0.9772 | 7.23 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P2 | 257 | 0.01 | 0.8925 | 0.8913 | 0.9774 | 6.24 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P3 | 513 | 1e-06 | 0.8971 | 0.8906 | 0.9774 | 6.24 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P3 | 513 | 0.0001 | 0.8954 | 0.8917 | 0.9775 | 7.23 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | P3 | 513 | 0.01 | 0.8929 | 0.8910 | 0.9775 | 6.24 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | H1 | 513 | 1e-06 | 0.9068 | 0.8884 | 0.9635 | 5.38 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | H1 | 513 | 0.0001 | 0.9066 | 0.8894 | 0.9632 | 5.38 | 0.9712 |
| P35_vcs_a5_views4_800ep_seed0 | cosine | H1 | 513 | 0.01 | 0.8710 | 0.8663 | 0.9491 | 4.64 | 0.9712 |
| P5_simclr_seed0 | None | P1 | 129 | 1e-06 | 0.9113 | 0.9118 | 0.9416 | 2.57 | — |
| P5_simclr_seed0 | None | P1 | 129 | 0.0001 | 0.9113 | 0.9118 | 0.9416 | 2.57 | — |
| P5_simclr_seed0 | None | P1 | 129 | 0.01 | 0.9088 | 0.9105 | 0.9410 | 2.57 | — |
| P5_simclr_seed0 | None | P2 | 257 | 1e-06 | 0.9179 | 0.9173 | 0.9439 | 2.57 | — |
| P5_simclr_seed0 | None | P2 | 257 | 0.0001 | 0.9178 | 0.9173 | 0.9439 | 2.57 | — |
| P5_simclr_seed0 | None | P2 | 257 | 0.01 | 0.8955 | 0.8948 | 0.9308 | 2.57 | — |
| P5_simclr_seed0 | None | P3 | 513 | 1e-06 | 0.9210 | 0.9157 | 0.9438 | 2.57 | — |
| P5_simclr_seed0 | None | P3 | 513 | 0.0001 | 0.9208 | 0.9160 | 0.9440 | 2.57 | — |
| P5_simclr_seed0 | None | P3 | 513 | 0.01 | 0.9006 | 0.8943 | 0.9331 | 2.98 | — |
| P5_simclr_seed0 | None | H1 | 513 | 1e-06 | 0.9024 | 0.8921 | 0.9434 | 3.46 | — |
| P5_simclr_seed0 | None | H1 | 513 | 0.0001 | 0.9024 | 0.8926 | 0.9429 | 3.46 | — |
| P5_simclr_seed0 | None | H1 | 513 | 0.01 | 0.8842 | 0.8814 | 0.9283 | 3.46 | — |
