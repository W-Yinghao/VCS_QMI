"""Real-host execution checks; these small checks are never scientific evidence."""
from __future__ import annotations
import os
import traceback
import torch
from common import ROOT,FAMILIES,Allocation,identity,lock,sha,utc,write
from run import (A,check_data,evaluate,load_checkpoint,new_trainer,plain,runtime_check,
                 save_checkpoint)


def main():
    allocation=Allocation('SHARED_EXECUTION_VALIDATION','VALIDATION')
    status='FAIL';checks=[]
    try:
        env=runtime_check('cuda');check_data({'FIT','CAL'})
        guard=A.DataGuard();fit=A.fit_cores(guard,1,1117)
        imgs=A.load_learned_inputs(guard,fit,{'FIT'},'BF1 execution validation full FIT residency','cuda')
        for family in FAMILIES:
            ident=identity(family,1117,1e-4,10 if family=='MF_VCS_M1' else None,'VALIDATION')
            directory=ROOT/'validation'/os.environ['SLURM_JOB_ID']/family
            directory.mkdir(parents=True,exist_ok=True)
            t=new_trainer(ident,guard,'cuda',imgs)
            t.step();cp=directory/'pair.pt';meta=save_checkpoint(t,cp,ident)
            t.step();expected=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in t.modules()]
            expected_sampler=t.sampler.g.bit_generator.state
            load_checkpoint(t,cp,ident);t.step()
            errors=[]
            for m,ref in zip(t.modules(),expected,strict=True):
                for k,v in m.state_dict().items():
                    comp=A.compare(v.detach().cpu(),ref[k],'float32')
                    # Independently assert tolerance instead of depending on a reporting key.
                    torch.testing.assert_close(v.detach().cpu(),ref[k],atol=1e-5,rtol=1e-4)
                    errors.append(comp)
            assert t.n_steps==2 and t.sampler.g.bit_generator.state==expected_sampler
            bad={**ident,'lr':3e-4}
            try:load_checkpoint(t,cp,bad)
            except RuntimeError:pass
            else:raise AssertionError('Checkpoint identity guard did not reject a changed lr')
            meta=save_checkpoint(t,cp,ident)
            cal=guard.included('CAL')[:1]
            ev=evaluate(t,guard,'CAL',cal,[0],directory/'CAL.jsonl',ident,meta,allocation)
            t.step()  # evaluation must restore trainability, including M1 W being solver-owned
            assert t.n_steps==3
            checks.append({'family':family,'status':'PASS','resume_parameter_comparisons':plain(errors),
                           'full_search_registration':ev['metrics'],'post_eval_training_step':3})
            del t;torch.cuda.empty_cache()
        try:guard.check(guard.held()[0],{'FIT','CAL','DEV'},'negative test: held TEST')
        except PermissionError:pass
        else:raise AssertionError('Held TEST guard did not reject access')
        out={'status':'PASS','utc':utc(),'job_id':os.environ['SLURM_JOB_ID'],
             'freeze_sha256':sha(ROOT/'FROZEN.json'),'environment':env,'checks':checks,
             'full_fit_cores_loaded':32,'held_test_read':False,'data_reads':guard.reads,
             'purpose':'execution validation only, excluded from CAL selection and scientific evidence'}
        write(ROOT/'GPU_GATE.json',out);status='PASS'
        print('BF1_GPU_GATE_PASS',flush=True)
    except Exception as e:
        write(ROOT/'GPU_GATE_FAILURE.json',{'status':'FAIL','utc':utc(),'job_id':os.environ.get('SLURM_JOB_ID'),
            'freeze_sha256':sha(ROOT/'FROZEN.json'),'checks':checks,'exception':repr(e),'traceback':traceback.format_exc()})
        raise
    finally:allocation.record('END',status=status)


if __name__=='__main__':
    with lock(ROOT/'GPU_GATE.lock'):main()
