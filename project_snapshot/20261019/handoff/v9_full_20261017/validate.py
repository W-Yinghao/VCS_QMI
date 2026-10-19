"""Execution gate only, never scientific evidence. All numerical checks run inside Slurm."""
from common import ROOT, frozen, oracle_pool, read, verify_input, write
import sys


def validate():
    m=frozen()
    for path in m['input_sha256']:
        verify_input(m,path)
    groups=read(ROOT/'GROUPS.json')
    assert len(groups)==18 and sum(len(g['units']) for g in groups)==33
    assert len({g['id'] for g in groups})==18
    sys.path.insert(0,str(ROOT/'source/src'))
    import torch
    from v9.n2.sgr_host import hinge_per_anchor
    # Hand-computed two-pair hinge: positive .8,.7; negatives .6,.9.
    sim=torch.tensor([[.8,.6],[.9,.7]],dtype=torch.float64,requires_grad=True)
    a,b=hinge_per_anchor(sim,.2,True)
    assert torch.allclose(a+b,torch.tensor([.3,.5],dtype=torch.float64),atol=1e-12)
    v=torch.tensor([1.,.25],dtype=torch.float64)
    mean=(v*(a+b)).mean()/v.mean(); summ=(v*(a+b)).sum()/v.mean()
    assert torch.allclose(summ,2*mean)
    g1=torch.autograd.grad(mean,sim,retain_graph=True)[0];g2=torch.autograd.grad(summ,sim)[0]
    assert torch.allclose(g2,2*g1)
    rows=[{'pair_id':'a','role':'teacher_warmup','observed_image_id':'0','caption_id':'0'},
          {'pair_id':'u1','role':'U','observed_image_id':'2','caption_id':'1'},
          {'pair_id':'u2','role':'U','observed_image_id':'2','caption_id':'2'}]
    labs={'a':{'mismatched':False},'u1':{'mismatched':True},'u2':{'mismatched':False}}
    pool,w=oracle_pool(rows,labs,False);assert len(pool)==3 and w=={'u1':0.,'u2':1.}
    pool,w=oracle_pool(rows,labs,True);assert len(pool)==2 and w=={'u2':1.}
    try:oracle_pool([dict(rows[0],mismatched=False)],labs,False)
    except ValueError:pass
    else:raise AssertionError('Leak guard missing')
    from summarize import collapse
    log={'epochs':[{'epoch':4,'dev':{'rsum':400}},{'epoch':5,'dev':{'rsum':100}},{'epoch':6,'dev':{'rsum':80}}],
         'steps':[{'epoch':e,'step':i,'u_risk_unnormalised':.4} for e in (5,6) for i in range(0,1000,20)]}
    assert collapse(log)['severe_collapse']
    for e in log['epochs'][1:]:e['dev']['rsum']=300
    assert not collapse(log)['severe_collapse']
    write(ROOT/'VALIDATION.json',{'status':'PASS','groups':18,'full_units':33,'input_files_verified':len(m['input_sha256']),
                                'numeric_checks':'hand-computed hinge, scale/gradient ratio, oracle pool isolation, collapse readout',
                                'scientific_evidence':False,'torch':torch.__version__})
    print('VALIDATION PASS',flush=True)


if __name__=='__main__':validate()
