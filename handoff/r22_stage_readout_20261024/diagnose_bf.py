"""Read-only CAL score identity diagnostic after failed gate1062609; no selection."""
import importlib.util
import json
import os
from pathlib import Path
import time
import types

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_stage', HERE/'run.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
M.torch.set_num_threads(4)
started=time.time();records=[]
freeze=HERE/'DIAGNOSTIC_FREEZE.json'
for rel,h in json.loads(freeze.read_text())['files'].items():assert M.R.sha(M.BASE/rel)==h
M.verify_freeze()
for arm in M.ARMS:
    t,u,guard,H,source=M.build('bf',arm,1001,'cuda')
    old={(r['core'],r['perturbation']):r for r in M.R.rows(source['cal_rows'])}
    with M.R.CP.evaluation_state(t):
        cores=sorted(guard.included('CAL'))[:2]
        session=M.R.make_session(u,t,guard,H,'CAL',cores,'cuda')
        for core in cores:
            energy,_=session.energy(core,0);inner=M.R.PE.inner(energy)
            inner.j_critic=types.MethodType(M.T.cosine_energy,inner)
            truth=session.perturbations[core][0];inner.geometry_readout='COS'
            before=energy.value(truth[0],truth[1:])
            checks=M.compare_gate(energy,inner,truth)
            wrapped=M.StageEnergy(energy,inner,'COS','COS')
            after=wrapped.value(truth[0],truth[1:])
            row=dict(arm=arm,core=core,truth=truth,stored=old[core,0]['E']['COS'],
                     before=before,after=after,difference=after-old[core,0]['E']['COS']['F_true'],
                     model_hash=M.A.state_hash(t.eU,t.eV),source=source,checks=checks)
            records.append(M.R.plain(row));print(json.dumps(M.R.plain(row)),flush=True)
    del t;M.torch.cuda.empty_cache()
result=dict(status='COMPLETE',freeze_sha256=M.R.sha(freeze),job=os.environ['SLURM_JOB_ID'],seconds=time.time()-started,
            records=records,environment=M.R.environment(),numerical_roles=['CAL'],label='Technical score-identity diagnostic, not task result')
M.R.write(HERE/'DIAGNOSTIC_COMPLETE.json',result)
