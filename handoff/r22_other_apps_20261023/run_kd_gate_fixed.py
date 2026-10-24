"""Technical gate-only repair: query feature dimension on the model's current device."""
import os
from common import HERE,BASE,read,write,sha
import run_kd as R

def feature_dim(model):
    previous=model.training
    model.eval()
    param=next(model.parameters())
    try:
        with R.torch.no_grad():
            _,features=model(R.torch.zeros(2,3,32,32,device=param.device,dtype=param.dtype))
    finally:model.train(previous)
    return features.shape[1]

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    R.verify()
    for rel,digest in read(HERE/'GATE_REPAIR_FREEZE.json')['files'].items():assert sha(BASE/rel)==digest,rel
    gate=read(HERE/'KD_CPU_CHECK.json');assert gate['status']=='PASS' and gate['freeze_sha256']==sha(HERE/'FREEZE.json')
    R.K.feature_dim=feature_dim
    def record(path,value):
        value.update(training_freeze_sha256=sha(HERE/'FREEZE.json'),freeze_sha256=sha(HERE/'GATE_REPAIR_FREEZE.json'),
                     technical_repair='Feature-dimension probe uses model device; training engine unchanged.',
                     failed_gate_job='1061572',gate_wrapper_sha256=sha(__file__))
        write(path,value)
    R.write=record
    R.gate()
