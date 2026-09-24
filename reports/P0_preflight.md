# P0 preflight — PASS

UTC: 2026-09-24T15:34:53Z

## Repo
- root: `/home/infres/yinwang/CS_QMI/ssl_pilot`
- branch/commit: `main` / `76cebd619b4da834e38c633a2dd3e2fe913ec88e`
- dirty: True (2 files)

## Environment
- python: `3.12.12 (main, Oct 31 2025, 23:02:31) [Clang 21.1.4 ]`
- torch: `2.9.0+cu128` (CUDA build 12.8, cuDNN 91002)
- torchvision: `0.24.0+cu128`
- numpy: `2.5.2`
- host: `node60`  SLURM job `1007754` partition `RTX6000PRO`
- GPUs: [{'index': 0, 'name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'total_memory_mb': 97343, 'capability': '12.0', 'multi_processor_count': 188}]
- nvidia-smi: ['615.71.09, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB']
- precision flags (defaults before policy): {'cudnn_benchmark': False, 'cudnn_deterministic': False, 'cudnn_allow_tf32': True, 'matmul_allow_tf32': False, 'float32_matmul_precision': 'highest', 'default_dtype': 'torch.float32', 'deterministic_algorithms': False}

## Resources
- {'cpu_count': 96, 'slurm_cpus': '8', 'slurm_mem': '32768', 'disk_output_root': {'total_gb': 85573.705078125, 'free_gb': 28149.14453125, 'inodes_free': 59033033135}, 'disk_data_root': {'total_gb': 85573.705078125, 'free_gb': 28149.14453125, 'inodes_free': 59033033135}}

## Paths
- {'REPO_ROOT': '/home/infres/yinwang/CS_QMI/ssl_pilot', 'DATA_ROOT': '/home/infres/yinwang/CS_QMI/data/cifar10', 'OUTPUT_ROOT': '/home/infres/yinwang/CS_QMI/outputs', 'MANIFEST_ROOT': '/home/infres/yinwang/CS_QMI/manifests', 'SLURM_JOB_ID': '1007754', 'SLURM_JOB_PARTITION': 'RTX6000PRO'}

## Config
- path: `/home/infres/yinwang/CS_QMI/ssl_pilot/configs/cifar10_pilot_vcs.yaml`
- file_sha256: `6621d513c45fc9d487228264e80f262f51ee0511bddaaf24aa0b1381bd56199e`
- config_hash: `cfe89416b54381ff870379c9d6cb734f1111c8fbe6d23ffb6f57426af6c8407a`
- method: `vcs_qmi`
- resolved_data_root: `/home/infres/yinwang/CS_QMI/data/cifar10`
- resolved_output_root: `/home/infres/yinwang/CS_QMI/outputs`
- resolved_manifest: `/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json`
- policy_checks: `PASS`

## Data
- n_train: 50000 (official train partition only; test partition not read)
- first10 labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3] (expected [6, 9, 9, 4, 1, 1, 2, 7, 8, 3])
- file hashes:
  - data_batch_1: md5 `c99cafc152244af753f735de768cd75f` sha256 `54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd`
  - data_batch_2: md5 `d4bba439e000b95fd0a9bffe97cbabec` sha256 `766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8`
  - data_batch_3: md5 `54ebc095f3ab1f0389bbae665268c751` sha256 `0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9`
  - data_batch_4: md5 `634d18415352ddfa80567beed471001a` sha256 `3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b`
  - data_batch_5: md5 `482c414d41f54cd18b22e5b47cb7c3cb` sha256 `d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154`
  - batches.meta: md5 `5ff9c542aee3614f3951f8cda6e48888` sha256 `f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb`

## Manifest
- path: `/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json`
- status: `CREATED`
- sha256: `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`
- n_fit: `45000`
- n_selection: `5000`
- fit_class_counts: `[4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500]`
- selection_class_counts: `[500, 500, 500, 500, 500, 500, 500, 500, 500, 500]`
- fit_first5: `[0, 1, 2, 3, 5]`
- selection_first5: `[4, 6, 29, 49, 66]`

## GPU shape check
- {'h': [4, 512], 'p_raw': [4, 128], 'params': {'encoder': 11168832, 'projector': 328832, 'critic': 394753}}

## To implement / mapping
- all CLIs implemented in this repo (train/evaluate/preflight/summarize/prepare_data). An unrelated SSL harness exists in ~/FMCA-AV (different objective/protocol); only its CIFAR-10 raw batch files are reused (read-only, md5-verified), no code/config is inherited from it (spec §1.1).
