"""Write AMENDMENT_CR3_A1.json (chmod 444). Does not edit PROTOCOL_FROZEN_CR3.json."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
acc_l40s, acc_rtx = sys.argv[1], str(TREE/'acceptance'/'ACCEPTANCE_CR3_1088291.json')
dst = TREE/'AMENDMENT_CR3_A1.json'
if dst.exists(): raise FileExistsError(dst)
A = json.load(open(acc_l40s)); assert A['all_pass'], 'L40S acceptance not ALL_PASS'
rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv'))); groups = {}
for r in rows: groups.setdefault(r['condition'], []).append(r['unit_id'])
fz = json.load(open(TREE/'PROTOCOL_FROZEN_CR3.json'))
am = {'amendment_id': 'CR3_A1', 'protocol_id': 'CR3_FIXED_GRAPH_SUPPORT_COMPLETE_V1', 'utc': time.strftime('%FT%TZ', time.gmtime()),
      'base_protocol_sha256': sha(TREE/'PROTOCOL_FROZEN_CR3.json'),
      'user_decision_verbatim': '不行不行，现在这个只能用一种GPU还是太慢了，你来组织一下，至少我们要能在L40S上跑',
      'change': 'GPU lock is per matched group instead of global: each condition (all its graphs x RECT/COMPLETE) runs as ONE Slurm job, '
                'cells sequential on the single GPU it receives; partition RTX6000PRO,L40S,H100, exclude node51,node53,node60. '
                'GROUP_ENV.json pins GPU name/driver/torch/CUDA/cuDNN; a later attempt on a different env stops. Within-condition comparisons are same-GPU; '
                'cross-condition comparisons may be cross-GPU (reported).',
      'unchanged': 'science, graphs, consumers, init, optimizer, sampler, steps, evaluator, cr3_student.py bytes, unit configs, data hashes',
      'reason': 'RTX PRO 6000 partition saturated (all 6 GPUs held by other users for 13-22 h); choice made without reading any CR3 DEV result',
      'groups': groups, 'n_jobs': len(groups),
      'superseded_runs': {'CR3_Flickr00_OBS_RECT_s1001': 'job 1088304 (RTX) kept on record under students/; rerun inside its condition job (students_A1/)',
                          'cancelled_pending_RTX_only_jobs': '1088305-1088315, 1088334 (never ran)'},
      'reuse': 'all 12 historical RECT cells remain RERUN (receipts of 1088291); no historical endpoint is reused',
      'code_sha256': {str(p.relative_to(TREE)): sha(p) for p in [TREE/'code'/'site'/'cr3_group_a1.py', TREE/'sbatch'/'group_a1.sbatch', TREE/'code'/'site'/'acceptance_cr3_a1.py']},
      'numeric_env': {'torch': A['checks']['GPU_RECORD']['torch'], 'cuda': A['checks']['GPU_RECORD']['cuda'], 'cudnn': A['checks']['GPU_RECORD']['cudnn'],
                      'determinism': A['checks']['GPU_RECORD']['determinism'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
      'acceptance': {'RTX_PRO_6000': {'file': acc_rtx, 'sha256': sha(acc_rtx), 'all_pass': True},
                     A['checks']['GPU_RECORD']['gpu']: {'file': acc_l40s, 'sha256': sha(acc_l40s), 'all_pass': True}},
      'test_read': False}
dst.write_text(json.dumps(am, indent=1)+'\n'); os.chmod(dst, 0o444); print(sha(dst))
