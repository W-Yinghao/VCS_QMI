Publication copy (sample names aliased; numbers and conclusions unchanged).

# L2RM nearest-neighbour baseline — integration checklist (prepared only; nothing launched)

Package item 9 / EVALUATION_AND_DECISIONS §4. CR1 CAP is **not** an L2RM reproduction and must not carry its name.

## Official source (local clone, read-only)
- Path: `/projects/EEG-foundation-model/yinghao/code_v9/L2RM`, origin `https://github.com/hhc1997/L2RM`,
  HEAD `a766c22a7cc7c2253805e212acc001a9d535c095` (2024-03-28). Working tree clean except `__pycache__/`.
- Env already on server: `/projects/EEG-foundation-model/yinghao/envs/v9_l2rm` (torch 2.9.0+cu128, POT 0.9.7.post1, pomegranate 0.14.8).
  README requires Python 3.8 / torch 1.12 / patched pomegranate (`rayleizhu/pomegranate`); the server env differs → any run must be
  labelled "server env, not README env".

## Official inputs
- NCR precomputed region features (`f30k_precomp`, `coco_precomp`, `cc152k_precomp`; 36×2048 BUTD regions) + NCR vocab (`./vocab`),
  i.e. **not** SigLIP2. On server: `/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data`.
- Noise index `noise_index/{f30k,coco}_precomp_{0.2,0.4,0.6,0.8}.npy` (DECL/RCL setting); f30k 0.6 sha256
  `5d8b61c1cf3656f1991f28b61d7928abe1d1240652f4ed65aa3c414fb90e5f28`. CR1 Flickr60 uses the R25 NOISE_LOCK map
  (`d81939538fb4…`, via NR1 DATA_BINDING) — a different noise draw; a matched comparison must state which index is used.
- Scripts: `train_f30k.sh` (SGR, bs128, reg 0.01, 35 ep, lr 2e-4, lr_cost 2e-6, lr_update 15, queue 128, warmup 5),
  `train_cc152.sh` (reg 0.07, 40 ep, warmup 10, noise_ratio 0.0), `opt.py` (`--rho` default 0.1 = partial transport mass).

## Mechanism interfaces to keep in an "adapted" control (if CR1 later shows a VCS-specific gain)
1. Clean/noisy split each epoch: Beta-mixture (pomegranate) on per-pair loss (`main_L2RM.split_data_by_loss/BMM_pred`).
   Its `labels_all` (noise truth) is only printed — must stay evaluator-only in any adaptation.
2. Self-supervised cost: `FeedForward_Network` (128×128, ReLU, init −I) on the batch similarity matrix, trained by
   inserting 1..10% "key" clean pairs into the image queue and minimising Σ supervision·cost (`train`, lines ~128–147).
3. Partial OT with dummy row/column: `SGRAF.partial_ot` (xi = 100·max C, A = max C, total mass s = `rho`) solved by
   log-domain Sinkhorn (`sinkhorn_log_domain`, reg = `--reg`); diagonal mask `M[I,J] = 1 if noise_ratio == 0 else 0`
   (uses the declared noise ratio — a known-hyperparameter dependence to disclose).
4. Rematched targets enter a symmetric-KL loss on noisy pairs (`method='Sym_KLDIV'`) plus triplet loss on clean pairs.
5. Evaluation: `evaluation.py` (`evalrank`, SGR shard scores, `i2t/t2i` with 5 captions/image). Earlier project audits
   (R44/R50) found batch-layout effects in IN²R-family evaluators — re-verify any upstream evaluator before use.

## Not done here (needs a new decision)
- No L2RM training/eval was started. A same-SigLIP2 "adapted L2RM" (cost net + partial OT on G scores) and an official-resource
  reproduction would be separate registered lines; CR1 results are not mixed with them.
