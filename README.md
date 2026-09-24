# VCS-QMI CIFAR-10 SSL pilot — server implementation

Implements the CLIs required by `VCS_QMI_SSL_Server_Agent_Spec_v1.md` on top of the collaborator reference
`reference/ssl_core.py` (imported verbatim, never re-implemented).  Read `STARTER_README.md` and the spec first.

## Resolved paths (P0)

| placeholder | value |
|---|---|
| `REPO_ROOT` | `/home/infres/yinwang/CS_QMI/ssl_pilot` (git, branch `main`) |
| `DATA_ROOT` | `/home/infres/yinwang/CS_QMI/data/cifar10` — `cifar-10-batches-py` is a symlink to the user's existing copy `/projects/EEG-foundation-model/yinghao/FMCA-AV/cifar10` (official batch files, md5-verified against the CIFAR page; only the 5 train batches are read, `test_batch` never) |
| `OUTPUT_ROOT` | `/home/infres/yinwang/CS_QMI/outputs` |
| `MANIFEST_ROOT` | `/home/infres/yinwang/CS_QMI/manifests` (`cifar10_dev45k_val5k.json` shared by every run) |
| Python env | `/home/infres/yinwang/CS_QMI/env` (uv venv, CPython 3.12.12, torch 2.9.0+cu128, torchvision 0.24.0) |

`slurm/common.sh` exports these; the CLIs refuse to run with unresolved `${...}` placeholders.

## Layout → spec §14 mapping

| spec | here |
|---|---|
| `data/cifar.py, splits.py, transforms.py` | `src/vcs_ssl/data/{cifar,splits,transforms,datasets}.py` |
| `models/backbone.py, projector.py, critic.py` | `src/vcs_ssl/models/` (`critic.py` only validates config and returns the reference `PairCritic`) |
| `losses/vcs.py, simclr.py, vicreg.py`, `pairing.py` | `reference/ssl_core.py` (collaborator) + `src/vcs_ssl/objectives.py` (dispatch only) |
| `train.py, preflight.py, evaluate.py, diagnostics.py, checkpoint.py` | `src/vcs_ssl/{train,preflight,evaluate,diagnostics,checkpoint}.py` (+ `schedule.py`, `optim.py`, `config.py`, `summarize.py`, `prepare_data.py`) |
| `configs/` | frozen starter YAMLs (SHA-256 verified against `PACKAGE_SHA256.json`) |
| `tests/` | `test_ssl_core.py` (29 reference tests, unchanged) + `test_integration.py` (server integration tests) |
| `outputs/<run_id>/` | as in spec §14 (`config.resolved.yaml`, `environment.json`, `manifest.json(.sha256)`, `checkpoints/`, `logs/steps.jsonl|epochs.jsonl`, `evaluations/`, `artifacts/view_examples.png`, `status.json`, `summary.json`, `failure/`) |
| scheduling | `slurm/*.sbatch`, `slurm/submit_pipeline.sh` (job ids recorded in `reports/job_ids.json`) |

## Commands (run inside SLURM allocations; see `slurm/`)

```bash
source slurm/common.sh
$PYTHON -m pytest -q                                                   # 29 reference + integration tests
$PYTHON -m vcs_ssl.prepare_data --root "$DATA_ROOT"                    # verify official archive/batches (md5/sha256)
$PYTHON -m vcs_ssl.preflight --config configs/cifar10_pilot_vcs.yaml --create-manifest
$PYTHON -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --smoke-steps 100 --smoke-epoch-steps 50
$PYTHON -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --run-id P3_vcs_seed0
$PYTHON -m vcs_ssl.evaluate --run-dir "$OUTPUT_ROOT/P3_vcs_seed0" --checkpoint epoch_020.pt --protocol pilot
$PYTHON -m vcs_ssl.summarize --stage P3_pilot --output reports/P4_ssl_pilot_results_table.md
```

Pipeline: `slurm/submit_pipeline.sh` submits P0 → P1 → P2 → P3(vcs → simclr → vicreg, one GPU job at a time) → P4 with
`afterok` dependencies, so a failing gate blocks everything downstream.  GPU partitions: `A100,H100,L40S,RTX6000PRO`
(never P100/V100), one GPU, FP32, TF32 off.

## Engineering choices not in the spec (disclosed)

- Smoke runs (`--smoke-steps N --smoke-epoch-steps M`) use their own horizon `T=N`, warmup `W=round(N·350/3500)`, and
  pseudo-epochs of `M` steps (fresh shuffle each pseudo-epoch) so checkpoint/resume can be exercised on real data.
- Critic init uses a separate `torch.manual_seed(seed·1000003+7919)` stream; encoder/projector init is `torch.manual_seed(seed)`
  and therefore bit-identical across the three methods (hashes in `run_manifest.json`).
- Loader generator seed = `seed`; pairing generator seed = `seed + 100003` (config `rng_seed_offset`).
- In-training kNN/spectrum/critic diagnostics run on a fresh model copy loaded from the just-written checkpoint, under
  `torch.random.fork_rng`, with dedicated eval generators; training RNG fingerprints are asserted unchanged afterwards.
- Resume is at epoch boundaries only (`last.pt`); mid-epoch progress is discarded and the cost is recorded.
