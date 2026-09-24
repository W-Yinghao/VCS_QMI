# P2 smoke gate (job 1007756)


## vcs — smoke_vcs_1007756

- [PASS] vcs status COMPLETED: status=COMPLETED reason=None
- [PASS] vcs first-step gradients: {"grad_norm_encoder": 0.019414531722808742, "grad_norm_projector": 0.0066280926781731005, "grad_norm_critic": 0.00976737832510553}
- [PASS] vcs encoder params changed after step 1
- [PASS] vcs all logged losses/gradients finite: 3 logged steps
- [PASS] vcs peak memory recorded: alloc 2793.33642578125 MB reserved 3524.0 MB
- [PASS] vcs J_raw within [-3,1]: first [5.386249085859163e-06] last [0.22719722986221313]
- [PASS] vcs J == 1 - R
- [PASS] vcs shifts in [1,B-1]
- [PASS] vcs in-training eval ran and left training RNG untouched: knn_epoch_000: kNN 36.54% h_rank 2.941489863916485, knn_epoch_002: kNN 30.30% h_rank 1.6837765828374935
- last step: loss -0.22720, step 134 ms, data wait 0 ms, lr 1.00e-05; steady images/s 1908.2342881935285

## simclr — smoke_simclr_1007756

- [PASS] simclr status COMPLETED: status=COMPLETED reason=None
- [PASS] simclr first-step gradients: {"grad_norm_encoder": 7.9797366892979555, "grad_norm_projector": 3.418288402596534, "grad_norm_critic": null}
- [PASS] simclr encoder params changed after step 1
- [PASS] simclr all logged losses/gradients finite: 3 logged steps
- [PASS] simclr peak memory recorded: alloc 2787.31103515625 MB reserved 3516.0 MB
- [PASS] simclr in-training eval ran and left training RNG untouched: knn_epoch_000: kNN 36.54% h_rank 2.941489863916485, knn_epoch_002: kNN 43.04% h_rank 13.735576668144041
- last step: loss 4.17252, step 128 ms, data wait 0 ms, lr 1.00e-05; steady images/s 1967.9475171926003

## vicreg — smoke_vicreg_1007756

- [PASS] vicreg status COMPLETED: status=COMPLETED reason=None
- [PASS] vicreg first-step gradients: {"grad_norm_encoder": 23.260522109334286, "grad_norm_projector": 19.761757788022948, "grad_norm_critic": null}
- [PASS] vicreg encoder params changed after step 1
- [PASS] vicreg all logged losses/gradients finite: 3 logged steps
- [PASS] vicreg peak memory recorded: alloc 2787.31494140625 MB reserved 3516.0 MB
- [PASS] vicreg in-training eval ran and left training RNG untouched: knn_epoch_000: kNN 36.54% h_rank 2.941489863916485, knn_epoch_002: kNN 37.84% h_rank 4.244603550887657
- last step: loss 18.37620, step 129 ms, data wait 0 ms, lr 1.00e-05; steady images/s 1984.6929679483399

## GPU resume check (VCS, stop@50 -> resume to 100 vs continuous)

- [PASS] interrupted run resumed to COMPLETED: status=COMPLETED
- max |Δparam| after 100 steps: {'encoder_state': 165.10752868652344, 'projector_state': 19.391801834106445, 'critic_state': 0.042101092636585236}
- max |Δloss| over resumed steps [50]..[99]: 0.0779469907283783
- resume events: [{'path': '/home/infres/yinwang/CS_QMI/outputs/smoke_vcs_interrupted_1007756/checkpoints/last.pt', 'sha256': '0da61e694d4fe7024464c6cad0c6c4cd42564c3090f8490e6b227da5f63b8991', 'completed_epoch': 1, 'optimizer_step': 50, 'utc': '2026-09-24T15:37:52Z', 'slurm_job_id': '1007756'}]
- bitwise identical on this GPU: False (epoch-boundary resume level; CUDA nondeterminism may prevent exactness — reported, not gated)
- [PASS] resume produced a full 100-step trajectory

**GATE PASS** — P3 may start

