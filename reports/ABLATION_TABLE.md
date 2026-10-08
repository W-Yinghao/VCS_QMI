# SSL exploration — appendix ablation table (development split, from results_v7.json)

Parents: A-P3 seed 0 = P107_AP3_views4_800ep_seed0 (89.06) and P107_AP3_c100_views4_800ep_seed0 (60.20).  Single-seed cells are descriptive; multi-seed lines have their own reports.

| line | run | dataset | family | change vs A-P3 seed 0 | linear | kNN | Δ linear | Δ kNN |
|---|---|---|---|---|---|---|---|---|
| P112 scorer scale sensitivity | P112_std_a1.5_k0.5_seed0 | cifar10 | VCS | scorer.a 2.0→1.5; scorer.b -1.0→-0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.06 | 86.64 | -1.00 | -0.66 |
| P112 scorer scale sensitivity | P112_std_a2_k0.25_seed0 | cifar10 | VCS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.72 | 86.88 | -0.34 | -0.42 |
| P112 scorer scale sensitivity | P112_std_a2_k0.75_seed0 | cifar10 | VCS | scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.64 | 86.82 | -0.42 | -0.48 |
| P112 scorer scale sensitivity | P112_std_a3_k0.5_seed0 | cifar10 | VCS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.48 | 86.34 | -0.58 | -0.96 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.25_seed0 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 87.88 | 84.56 | -1.18 | -2.74 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.25_seed1 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 87.70 | 85.08 | -1.36 | -2.22 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.25_seed2 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 87.34 | 85.16 | -1.72 | -2.14 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.75_seed0 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.22 | 85.74 | -0.84 | -1.56 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.75_seed1 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 88.28 | 85.86 | -0.78 | -1.44 |
| P112 scorer scale sensitivity | P112_strong_a2_k0.75_seed2 | cifar10 | VCS | crop_min 0.2→0.08; scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 87.96 | 85.56 | -1.10 | -1.74 |
| P112 scorer scale sensitivity | P112_strong_a3_k0.5_seed0 | cifar10 | VCS | crop_min 0.2→0.08; scorer.a 2.0→3.0; scorer.b -1.0→-1.5; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 86.92 | 84.64 | -2.14 | -2.66 |
| P112 scorer scale sensitivity | P112_c100_a2_k0.25_seed0 | cifar100 | VCS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 59.84 | 55.64 | -0.36 | -0.28 |
| P112 scorer scale sensitivity | P112_c100_a2_k0.75_seed0 | cifar100 | VCS | scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 58.34 | 51.24 | -1.86 | -4.68 |
| P112 scorer scale sensitivity | P112_c100_a3_k0.5_seed0 | cifar100 | VCS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5; pairing.pair_scope all_view_tokens→cross_view_k; pairing.negative_detach False→True | 59.66 | 54.24 | -0.54 | -1.68 |
| P115 augmentation factors | P115_AP3_croponly_views4_800ep_seed0 | cifar10 | VCS | crop_min 0.2→0.08 | 88.80 | 86.42 | -0.26 | -0.88 |
| P115 augmentation factors | P115_AP3_jitteronly_views4_800ep_seed0 | cifar10 | VCS | (seed / stage only) | 88.78 | 87.78 | -0.28 | +0.48 |
| P115 augmentation factors | P115_simclr_croponly_views4_800ep_seed0 | cifar10 | SimCLR | crop_min 0.2→0.08; scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 89.86 | 88.68 | +0.80 | +1.38 |
| P115 augmentation factors | P115_simclr_jitteronly_views4_800ep_seed0 | cifar10 | SimCLR | scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 88.24 | 88.48 | -0.82 | +1.18 |
| P126 curved scorer | P126_CV_AP3_c10_lamm025_views4_800ep_seed0 | cifar10 | VCS | scorer.affine_mode fixed→fixed_curved; scorer.curvature_lambda None→-0.25 | 89.08 | 87.44 | +0.02 | +0.14 |
| P126 curved scorer | P126_CV_AP3_c10_lamp025_views4_800ep_seed0 | cifar10 | VCS | scorer.affine_mode fixed→fixed_curved; scorer.curvature_lambda None→0.25 | 88.98 | 87.12 | -0.08 | -0.18 |
| P126 curved scorer | P126_CV_AP3_c100_lamm025_views4_800ep_seed0 | cifar100 | VCS | scorer.affine_mode fixed→fixed_curved; scorer.curvature_lambda None→-0.25 | 59.34 | 55.70 | -0.86 | -0.22 |
| P126 curved scorer | P126_CV_AP3_c100_lamp025_views4_800ep_seed0 | cifar100 | VCS | scorer.affine_mode fixed→fixed_curved; scorer.curvature_lambda None→0.25 | 59.86 | 56.26 | -0.34 | +0.34 |
| P129 tuned JS scorer | P129_JS_c10_a1.5_k0.5_seed0 | cifar10 | JS | scorer.a 2.0→1.5; scorer.b -1.0→-0.75 | 88.50 | 87.26 | -0.56 | -0.04 |
| P129 tuned JS scorer | P129_JS_c10_a2_k0.25_seed0 | cifar10 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 89.26 | 87.68 | +0.20 | +0.38 |
| P129 tuned JS scorer | P129_JS_c10_a2_k0.25_seed1 | cifar10 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 88.60 | 87.48 | -0.46 | +0.18 |
| P129 tuned JS scorer | P129_JS_c10_a2_k0.25_seed2 | cifar10 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 89.06 | 87.02 | +0.00 | -0.28 |
| P129 tuned JS scorer | P129_JS_c10_a2_k0.75_seed0 | cifar10 | JS | scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75 | 88.52 | 87.06 | -0.54 | -0.24 |
| P129 tuned JS scorer | P129_JS_c10_a3_k0.25_seed0 | cifar10 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-0.75; scorer.kappa 0.5→0.25 | 88.28 | 87.12 | -0.78 | -0.18 |
| P129 tuned JS scorer | P129_JS_c10_a3_k0.5_seed0 | cifar10 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 88.60 | 87.06 | -0.46 | -0.24 |
| P129 tuned JS scorer | P129_JS_c100_a1.5_k0.5_seed0 | cifar100 | JS | scorer.a 2.0→1.5; scorer.b -1.0→-0.75 | 58.86 | 54.48 | -1.34 | -1.44 |
| P129 tuned JS scorer | P129_JS_c100_a2_k0.25_seed0 | cifar100 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 59.58 | 56.42 | -0.62 | +0.50 |
| P129 tuned JS scorer | P129_JS_c100_a2_k0.25_seed1 | cifar100 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 60.18 | 56.02 | -0.02 | +0.10 |
| P129 tuned JS scorer | P129_JS_c100_a2_k0.25_seed2 | cifar100 | JS | scorer.b -1.0→-0.5; scorer.kappa 0.5→0.25 | 59.18 | 55.52 | -1.02 | -0.40 |
| P129 tuned JS scorer | P129_JS_c100_a2_k0.75_seed0 | cifar100 | JS | scorer.b -1.0→-1.5; scorer.kappa 0.5→0.75 | 58.24 | 52.20 | -1.96 | -3.72 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.25_seed0 | cifar100 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-0.75; scorer.kappa 0.5→0.25 | 60.24 | 57.12 | +0.04 | +1.20 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.5_lr0.5x_seed0 | cifar100 | JS | lr 0.001→0.0005; scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 60.22 | 55.02 | +0.02 | -0.90 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.5_lr2x_seed0 | cifar100 | JS | lr 0.001→0.002; scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 60.26 | 56.62 | +0.06 | +0.70 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.5_seed0 | cifar100 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 60.72 | 56.18 | +0.52 | +0.26 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.5_seed1 | cifar100 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 60.44 | 56.28 | +0.24 | +0.36 |
| P129 tuned JS scorer | P129_JS_c100_a3_k0.5_seed2 | cifar100 | JS | scorer.a 2.0→3.0; scorer.b -1.0→-1.5 | 60.30 | 56.48 | +0.10 | +0.56 |
| P133 momentum keys | P133_simclr_moco_noqueue_views4_800ep_seed0 | cifar10 | SimCLR | scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1; pairing.momentum_encoder False→True; pairing.moco_consistent False→True; pairing.moco_use_queue None→False | 87.68 | 87.36 | -1.38 | +0.06 |
| P133 momentum keys | P133_vcs_moco_noqueue_views4_800ep_seed0 | cifar10 | VCS | pairing.pair_scope all_view_tokens→cross_view_k; pairing.momentum_encoder False→True; pairing.moco_consistent False→True; pairing.moco_use_queue None→False | 89.04 | 87.74 | -0.02 | +0.44 |
| P135 learning rate | P135_js_c10_lr0.5x_seed0 | cifar10 | JS | lr 0.001→0.0005 | 88.28 | 87.14 | -0.78 | -0.16 |
| P135 learning rate | P135_js_c10_lr2x_seed0 | cifar10 | JS | lr 0.001→0.002 | 88.86 | 87.56 | -0.20 | +0.26 |
| P135 learning rate | P135_vcs_c10_lr0.5x_seed0 | cifar10 | VCS | lr 0.001→0.0005 | 88.72 | 87.02 | -0.34 | -0.28 |
| P135 learning rate | P135_vcs_c10_lr2x_seed0 | cifar10 | VCS | lr 0.001→0.002 | 88.94 | 87.84 | -0.12 | +0.54 |
| P135 learning rate | P135_js_c100_lr0.5x_seed0 | cifar100 | JS | lr 0.001→0.0005 | 59.30 | 54.66 | -0.90 | -1.26 |
| P135 learning rate | P135_js_c100_lr2x_seed0 | cifar100 | JS | lr 0.001→0.002 | 59.36 | 55.90 | -0.84 | -0.02 |
| P135 learning rate | P135_js_c100_lr2x_seed1 | cifar100 | JS | lr 0.001→0.002 | 59.50 | 56.24 | -0.70 | +0.32 |
| P135 learning rate | P135_js_c100_lr2x_seed2 | cifar100 | JS | lr 0.001→0.002 | 58.60 | 55.06 | -1.60 | -0.86 |
| P135 learning rate | P135_vcs_c100_lr0.5x_seed0 | cifar100 | VCS | lr 0.001→0.0005 | 60.38 | 55.20 | +0.18 | -0.72 |
| P135 learning rate | P135_vcs_c100_lr2x_seed0 | cifar100 | VCS | lr 0.001→0.002 | 60.52 | 56.68 | +0.32 | +0.76 |
| P136 schedule / views / batch | P136_AP3_c10_b512_seed0 | cifar10 | VCS | batch_images 256→512 | 88.58 | 87.28 | -0.48 | -0.02 |
| P136 schedule / views / batch | P136_AP3_c10_ep1600_seed0 | cifar10 | VCS | epochs 800→1600 | 88.68 | 87.74 | -0.38 | +0.44 |
| P136 schedule / views / batch | P136_AP3_c10_v8_seed0 | cifar10 | VCS | views 4→8 | 89.20 | 87.82 | +0.14 | +0.52 |
| P136 schedule / views / batch | P136_AP3_c100_b512_seed0 | cifar100 | VCS | batch_images 256→512 | 59.36 | 55.14 | -0.84 | -0.78 |
| P136 schedule / views / batch | P136_AP3_c100_ep1600_seed0 | cifar100 | VCS | epochs 800→1600 | 59.56 | 56.88 | -0.64 | +0.96 |
| P136 schedule / views / batch | P136_AP3_c100_v8_seed0 | cifar100 | VCS | views 4→8 | 59.62 | 57.14 | -0.58 | +1.22 |
| P137 projector width | P137_HEAD512_js_c10_seed0 | cifar10 | JS | projector 512->512->128→512->512->512 | 88.90 | 87.44 | -0.16 | +0.14 |
| P137 projector width | P137_HEAD512_simclr_c10_seed0 | cifar10 | SimCLR | projector 512->512->128→512->512->512; scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 87.56 | 87.98 | -1.50 | +0.68 |
| P137 projector width | P137_HEAD512_vcs_c10_seed0 | cifar10 | VCS | projector 512->512->128→512->512->512 | 88.42 | 86.96 | -0.64 | -0.34 |
| P137 projector width | P137_HEAD512_js_c100_seed0 | cifar100 | JS | projector 512->512->128→512->512->512 | 59.82 | 55.22 | -0.38 | -0.70 |
| P137 projector width | P137_HEAD512_simclr_c100_seed0 | cifar100 | SimCLR | projector 512->512->128→512->512->512; scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 58.02 | 56.58 | -2.18 | +0.66 |
| P137 projector width | P137_HEAD512_vcs_c100_seed0 | cifar100 | VCS | projector 512->512->128→512->512->512 | 59.88 | 55.86 | -0.32 | -0.06 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_js_c10_seed0 | cifar10 | JS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 88.80 | 87.38 | -0.26 | +0.08 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_js_c10_seed1 | cifar10 | JS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 88.74 | 87.64 | -0.32 | +0.34 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_js_c10_seed2 | cifar10 | JS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 88.40 | 87.20 | -0.66 | -0.10 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_vcs_c10_seed0 | cifar10 | VCS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 89.22 | 87.88 | +0.16 | +0.58 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_vcs_c10_seed1 | cifar10 | VCS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 88.88 | 87.74 | -0.18 | +0.44 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_vcs_c10_seed2 | cifar10 | VCS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 89.06 | 87.56 | +0.00 | +0.26 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_js_c100_seed0 | cifar100 | JS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 59.14 | 54.74 | -1.06 | -1.18 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_js_c100_seed1 | cifar100 | JS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 59.46 | 54.68 | -0.74 | -1.24 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_vcs_c100_seed0 | cifar100 | VCS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 60.00 | 55.38 | -0.20 | -0.54 |
| P138 K = 16 sampled pairs | P138_PAIR_K16_vcs_c100_seed1 | cifar100 | VCS | pairing.pair_scope all_view_tokens→sampled_image_shifts_all_view_pairs; pairing.k 8→16 | 60.62 | 55.60 | +0.42 | -0.32 |
| P145 pair contamination (STRESS) | P145_STRESS10_js_c10_seed0 | cifar10 | JS | (seed / stage only) | 88.44 | 87.24 | -0.62 | -0.06 |
| P145 pair contamination (STRESS) | P145_STRESS10_simclr_c10_seed0 | cifar10 | SimCLR | scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 88.12 | 87.54 | -0.94 | +0.24 |
| P145 pair contamination (STRESS) | P145_STRESS10_vcs_c10_seed0 | cifar10 | VCS | (seed / stage only) | 89.08 | 87.80 | +0.02 | +0.50 |
| P145 pair contamination (STRESS) | P145_STRESS10_js_c100_seed0 | cifar100 | JS | (seed / stage only) | 59.94 | 55.62 | -0.26 | -0.30 |
| P145 pair contamination (STRESS) | P145_STRESS10_js_c100_seed1 | cifar100 | JS | (seed / stage only) | 58.82 | 54.66 | -1.38 | -1.26 |
| P145 pair contamination (STRESS) | P145_STRESS10_js_c100_seed2 | cifar100 | JS | (seed / stage only) | 59.82 | 55.46 | -0.38 | -0.46 |
| P145 pair contamination (STRESS) | P145_STRESS10_simclr_c100_seed0 | cifar100 | SimCLR | scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 57.36 | 56.34 | -2.84 | +0.42 |
| P145 pair contamination (STRESS) | P145_STRESS10_simclr_c100_seed1 | cifar100 | SimCLR | scorer.temperature None→0.2; pairing.pair_scope all_view_tokens→cross_view_k; pairing.k 8→1 | 58.16 | 56.76 | -2.04 | +0.84 |
| P145 pair contamination (STRESS) | P145_STRESS10_vcs_c100_seed0 | cifar100 | VCS | (seed / stage only) | 60.30 | 55.82 | +0.10 | -0.10 |
| P145 pair contamination (STRESS) | P145_STRESS10_vcs_c100_seed1 | cifar100 | VCS | (seed / stage only) | 59.78 | 55.50 | -0.42 | -0.42 |
| P145 pair contamination (STRESS) | P145_STRESS10_vcs_c100_seed2 | cifar100 | VCS | (seed / stage only) | 59.72 | 56.24 | -0.48 | +0.32 |
| P147 batch 512 + lr scaling | P147_AP3_c10_b512_lr2x_seed0 | cifar10 | VCS | lr 0.001→0.002; batch_images 256→512 | 88.64 | 87.46 | -0.42 | +0.16 |
| P147 batch 512 + lr scaling | P147_AP3_c100_b512_lr2x_seed0 | cifar100 | VCS | lr 0.001→0.002; batch_images 256→512 | 59.72 | 56.30 | -0.48 | +0.38 |
| P148 weight decay | P148_AP3_c10_wd1e-5_seed0 | cifar10 | VCS | (seed / stage only) | 88.92 | 87.24 | -0.14 | -0.06 |
| P148 weight decay | P148_AP3_c10_wd5e-4_seed0 | cifar10 | VCS | (seed / stage only) | 89.28 | 87.80 | +0.22 | +0.50 |
| P148 weight decay | P148_AP3_c100_wd1e-5_seed0 | cifar100 | VCS | (seed / stage only) | 60.52 | 55.80 | +0.32 | -0.12 |
| P148 weight decay | P148_AP3_c100_wd5e-4_seed0 | cifar100 | VCS | (seed / stage only) | 60.46 | 56.06 | +0.26 | +0.14 |
