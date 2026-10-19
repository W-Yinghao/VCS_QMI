"""Freeze a committed local clone and the accepted BF1 execution contract."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from common import ROOT,REPO,FAMILIES,candidates,digest,read,sha,utc,write


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()


def main():
    assert REPO==ROOT/'source', 'Freeze the isolated committed clone, not a live development tree'
    assert not git('status','--porcelain'), 'Freeze requires a clean clone'
    assert not (ROOT/'FROZEN.json').exists(), 'Preserve the existing freeze; use a versioned revision'
    d=read(ROOT/'DATA_LOCK.json');assert d['status']=='PASS' and d['counts']=={'FIT':32,'CAL':8,'DEV':32}
    base=read(REPO/'v10/T02/T02_M1_RELATIVE_TIMING_cuda.json')['environment']
    precision_keys=('float32_matmul_precision','cuda_matmul_fp32_precision_raw','backends_fp32_precision_raw',
        'cudnn_fp32_precision_raw','cudnn_conv_fp32_precision','cudnn_rnn_fp32_precision_raw',
        'deterministic_algorithms','deterministic_algorithms_warn_only','cudnn_benchmark','cudnn_deterministic')
    protocol={'schema':'v10.bf1.full.v1','status':'FROZEN_EXECUTION_GATE_REQUIRED','utc':utc(),
        'source_commit':git('rev-parse','HEAD'),'accepted_core_hash':'e9b4239450c4e2eb',
        'parent_protocol_sha256':sha(REPO/'v10/protocol/V10_PROTOCOL_P2_20261017.json'),
        'data':{'manifest_sha256':d['manifest_sha256'],'data_lock_sha256':sha(ROOT/'DATA_LOCK.json'),
                'roles':d['counts'],'CONFIRM':0,'TEST134':'HOLD','EPFL':'NOT_ACTIVATED'},
        'preprocessing':'unchanged pinned R6/SGR-independent BF host; 512px x4 downsample; RGB robust normalization and log1p SHG',
        'families':list(FAMILIES),'candidates':{f:candidates(f) for f in FAMILIES},
        'train':{'steps':3000,'checkpoint_interval':250,'fit_fraction':1.0,'n_sc':16,'n_pos':128,'crop':160,
                 'K':16,'temperature':1,'ridge':0,'w_decay':0,'other_decay':1e-4,'clip':1,
                 'tuning_seed':1017,'final_seeds':[1001,1002,1003],'optimizer':'AdamW, tied learning rates',
                 'rng':'unchanged HostBatchSampler/default_rng(1000+seed); model torch seed; save all RNG and optimizer states'},
        'reference':'within-scene V; ordinary permutation including fixed points',
        'm1':'relative_zero_head; tolerance=0.01*G_b(0); caps 10/30/100; warm start; no floor; convergence recorded',
        'geometry':{'searcher':'unchanged R6-REG-P2 optimise_rigid','reference_scope':'same positions and fixed permutation per core/perturbation',
                    'perturbations_per_core':10,'cal_cores':8,'dev_cores':32,'support':'unchanged BOX1/BOX2, warp-then-encode',
                    'registration_energy':{'MF_VCS_perm1':'J_perm1','MF_VCS_M0':'exact V J','MF_VCS_M1':'exact V J','MLP_VCS_perm1':'J_perm1'},
                    'checkpoint':'saved encoder/features/W pair; no refit for evaluation'},
        'selection':{'order':['success42 descending','success9 descending','median error ascending','checkpoint step ascending','candidate order ascending'],
                     'role':'CAL only','final_rule':'hyperparameters from tuning selection; each final seed selects its own CAL checkpoint',
                     'completeness':'fixed-steps selection uses only full 3000-step candidates with all 12 complete CAL readouts; budget-truncated candidates retained separately'},
        'budget':{'cal_max_candidates_per_family':9,'cal_allocated_L40S_gpu_hours_per_family':8,
                  'includes':'training, checkpoints, full CAL evaluation and allocated job startup/overhead; failed attempts count',
                  'watchdog':'8h minus 30s bookkeeping reserve; budget exhaustion is retained, never silently expanded',
                  'cost_view':'checkpoint timestamps and measured phase accounting; scheduler entire-allocation reconciliation separate',
                  'final_cost':'all three final seeds including their CAL and DEV; charged separately from tuning cap'},
        'precision':{'settings':{k:base['precision'][k] for k in precision_keys},'env':base['precision']['env'],
                     'environment':{k:base[k] for k in ('torch','cuda_runtime','cudnn','torch_threads')}},
        'execution':{'formal_trainings':30,'CAL':18,'FINAL_DEV':12,'array_families':list(FAMILIES),'gpu_concurrency':2,
                     'gpu_type':'NVIDIA L40S','family_flow':'all frozen CAL candidates -> verified selection -> 3 full final seeds',
                     'failure':'retain and stop affected family; complete reuse checks configuration and artifact hashes',
                     'validation':'Slurm real FIT/CAL checkpoint/resume/search checks, excluded from scientific comparison'},
        'reporting':{'primary':'DEV32 per seed','sensitivity':'DEV24 excluding FIT/CAL TMA positions',
                     'units':'core, paired 3 training seeds; 10 perturbations are not 10 independent cores',
                     'limitations':'development benchmark; remaining external/native baselines required before full comparative claims'}}
    write(ROOT/'PROTOCOL.json',protocol)
    tracked=git('ls-files','src/v10','scripts/v10','slurm/v10','tests/v10','src/v9/reg','src/v9/__init__.py',
        'scripts/v10_bf1','slurm/v10_bf1','v10/T00','v10/T01','v10/T02','v10/protocol','v10/decisions').splitlines()
    frozen={'schema':'v10.bf1.freeze.v1','utc':utc(),'source_commit':git('rev-parse','HEAD'),
        'source_root':str(REPO),'source_files':{p:sha(REPO/p) for p in tracked if p},
        'run_inputs':{p:sha(ROOT/p) for p in ('PROTOCOL.json','DATA_LOCK.json')},
        'core_code_unchanged':True,'driver_validation_required':True}
    write(ROOT/'FROZEN.json',frozen)
    print(json.dumps({'source_commit':frozen['source_commit'],'freeze_sha256':sha(ROOT/'FROZEN.json'),
                      'source_files':len(frozen['source_files']),'formal_trainings':30}))


if __name__=='__main__':main()
