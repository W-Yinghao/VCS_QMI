"""Positive-only quadratic mechanism control, never labeled complete VCS-QMI."""
import argparse,copy,math,os,sys,time,traceback
from pathlib import Path
from common import BASE,HERE,OLD,read,write,sha,verify
sys.path.insert(0,str(OLD/'engine'))
import guard
import torch
from kd import runner as K,decision as D
from kd.losses import DistillationLoss
ORIGINAL=D.decision_loss

def decision_loss(student,teacher,method):
    if method!='positive_quadratic':return ORIGINAL(student,teacher,method)
    u,v=D.coordinates(student),D.coordinates(teacher.detach());p=(u*v).sum(1)
    loss=-(p-.5*p.square()).mean()
    with torch.no_grad():
        scores=u@v.T;mask=~torch.eye(len(u),dtype=torch.bool,device=u.device)
        diagnostics=dict(method=method,positive_cosine=float(p.mean()),product_cosine=float(scores[mask].mean()),
            positive_abs_T_ge099=float((p.abs()>=.99).float().mean()),raw_auxiliary=float(loss),
            raw_j=float(D.quadratic_u(u,v)),q='NOT IN LOSS: positive-only mechanism control; full J logged descriptively')
    return loss,diagnostics

D.decision_loss=decision_loss

def check():
    freeze=verify();torch.set_num_threads(4);torch.manual_seed(25)
    s=torch.randn(9,11,dtype=torch.float64,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
    val,_=decision_loss(s,t,'positive_quadratic');p=(D.coordinates(s)*D.coordinates(t.detach())).sum(1)
    expected=(-p+.5*p.square()).mean();assert torch.equal(val,expected)
    gs,gt=torch.autograd.grad(val,(s,t),retain_graph=True,allow_unused=True);assert gt is None
    assert torch.equal(gs,torch.autograd.grad(expected,s,retain_graph=True)[0])
    assert torch.autograd.gradcheck(lambda x:decision_loss(x,t,'positive_quadratic')[0],(s,))
    assert torch.isfinite(gs).all()
    for method in ('cosine','vcs','bce'):
        a,_=decision_loss(s,t,method);b,_=ORIGINAL(s,t,method)
        assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,s,retain_graph=True)[0],torch.autograd.grad(b,s,retain_graph=True)[0])
    scores=D.coordinates(s)@D.coordinates(t.detach()).T;mask=~torch.eye(len(s),dtype=torch.bool)
    ref=(scores[mask]+.5*scores[mask].square()).mean();full,_=ORIGINAL(s,t,'vcs')
    assert torch.allclose(full,val+ref,atol=1e-12,rtol=1e-12)
    shifted,_=decision_loss(s+torch.randn(9,1,dtype=torch.float64),t+torch.randn(9,1,dtype=torch.float64),'positive_quadratic')
    assert torch.allclose(val,shifted,atol=1e-12,rtol=1e-12)
    perm=torch.randperm(11);permuted,_=decision_loss(s[:,perm],t[:,perm],'positive_quadratic')
    assert torch.allclose(val,permuted,atol=1e-12,rtol=1e-12)
    # Real input/initialization checks for all six configurations, one CPU update
    # through the original DistillationLoss; never train on CAL or open TEST.
    rows=[]
    for j in read(HERE/'JOBS_SPEC.json')['full']:
        cfg=K.read_config(j['config']);guard.install(guard.kd_roots(cfg));manifest=K.read_manifest(cfg['data']['manifest'])
        source=read(j['control_receipts']['vcs']);assert source['status']=='COMPLETE'
        model=K.create_model(cfg['student'],manifest['num_classes'],cfg['seed']);dim=K.feature_dim(model)
        cpu_initial=K.state_digest(model)
        repeat=K.create_model(cfg['student'],manifest['num_classes'],cfg['seed']);K.feature_dim(repeat)
        assert K.state_digest(repeat)==cpu_initial;del repeat
        teacher,tdim,_=K._teacher(cfg,manifest,torch.device('cpu'));teacher_hash=K.state_digest(teacher)
        fit=K.RoleDataset(cfg['data']['root'],manifest,'fit',augment=True,seed=cfg['seed'])
        images,labels,ids=next(iter(K._loader(fit,cfg,True,0)));model.train()
        with torch.no_grad():lt,ft=teacher(images)
        ls,fs=model(images);losses=DistillationLoss(dim,tdim,cfg['loss'])
        loss,_=losses(ls,lt,fs,ft,labels,0);assert not list(losses.parameters())
        opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=.0005);opt.zero_grad();loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is None for p in teacher.parameters());opt.step();assert K.state_digest(teacher)==teacher_hash
        rows.append(dict(name=j['name'],seed=cfg['seed'],loss=float(loss.detach()),initial_sha256=cpu_initial,source_initial_sha256=source['student_initial_sha256'],cpu_matches_source=cpu_initial==source['student_initial_sha256'],cpu_capability=torch.backends.cpu.get_cpu_capability(),teacher_sha256=teacher_hash))
        print('CPU_PATH',j['name'],'PASS',flush=True)
    write(HERE/'CPU_CHECK.json',dict(status='PASS',freeze_sha256=freeze,rows=rows,guard=guard.status(),checks=['positive-only formula and gradcheck','teacher stop-gradient','old objectives bitwise unchanged','positive plus reference equals full -J','translation/class-coordinate permutation invariance','six real FIT CPU updates and repeat initialization identity; exact source initialization enforced on assigned GPU host before every training run']))

def execute(config,folder,technical=False):
    freeze=verify();cpu=read(HERE/'CPU_CHECK.json');assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
    cfg=K.read_config(config);out=HERE/'runs'/folder
    assert not out.exists(),'No implicit resume/retry or duplicate successful run'
    if not technical:
        admission=read(HERE/'ADMISSION.json');assert 'full' in admission['groups'] and admission['freeze_sha256']==freeze
    guard.install(guard.kd_roots(cfg));start=time.time();out.mkdir(parents=True)
    write(out/'ATTEMPT.json',dict(status='RUNNING',job=os.environ.get('SLURM_JOB_ID'),started=start,freeze_sha256=freeze))
    try:
        # CPU instruction paths can change last-bit random initialization.
        # Exact source matching is mandatory on the assigned execution host,
        # before K.train performs any update (and checked again after training).
        manifest=K.read_manifest(cfg['data']['manifest']);K._device(cfg)
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        probe=K.create_model(cfg['student'],manifest['num_classes'],cfg['seed']);K.feature_dim(probe)
        probe_sha=K.state_digest(probe);del probe
        anchor='ce' if cfg['loss']['anchor']=='none' else 'dkd'
        refs=read(HERE/'SOURCE_LOCK.json')['controls'][f'{anchor}_{cfg["seed"]}']
        for method in ('cosine','vcs'):
            c=read(refs[method]['receipt']);cr=read(Path(refs[method]['receipt']).parent/'run.json')
            assert probe_sha==c['student_initial_sha256'],'Initial state mismatch before training'
            assert torch.cuda.get_device_name()==c['gpu']
            assert K.environment()==cr['environment']
        write(out/'PREFLIGHT.json',dict(status='PASS',initial_sha256=probe_sha,cpu_capability=torch.backends.cpu.get_cpu_capability(),gpu=torch.cuda.get_device_name(),environment=K.environment()))
        r=K.train(config,out);assert r['status']=='complete' and r['teacher_frozen_verified'] and not r['test_loaded_during_train']
        hist=read(out/'history.json');assert len(hist)==cfg['training']['epochs']
        seed=cfg['seed'];anchor='ce' if cfg['loss']['anchor']=='none' else 'dkd'
        controls=read(HERE/'SOURCE_LOCK.json')['controls'][f'{anchor}_{seed}']
        for method in ('cosine','vcs'):
            cp=Path(controls[method]['receipt']);c=read(cp);ch=read(cp.parent/'history.json');cr=read(cp.parent/'run.json')
            assert r['student_initial_state_sha256']==c['student_initial_sha256']
            assert r['teacher_initial_state_sha256']==cr['teacher_initial_state_sha256']
            assert r['environment']==cr['environment'],'Frozen software/precision environment differs'
            assert torch.cuda.get_device_name()==c['gpu'],'Within-seed GPU model differs'
            for x,y in zip(hist,ch):
                assert x['sample_order_sha256']==y['sample_order_sha256'] and x['fit_n_seen']==y['fit_n_seen']==44992
                assert x['next_lr']==y['next_lr']
        write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=freeze,seed=seed,anchor=anchor,ds='kd_positive_control',summaries={},technical=technical,
            epochs=len(hist),job=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-start,config_sha256=sha(config),
            student_initial_sha256=r['student_initial_state_sha256'],gpu=torch.cuda.get_device_name(),environment=K.environment(),
            history_sha256=sha(out/'history.json'),best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt'),
            final_top1=hist[-1]['cal']['top1'],guard=guard.status(),test_read=False,label='Positive-only mechanism control, NOT complete VCS-QMI; development CAL'))
        write(out/'ATTEMPT.json',dict(status='COMPLETE',job=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-start,freeze_sha256=freeze))
    except BaseException as e:
        write(out/'ATTEMPT.json',dict(status='FAILED',job=os.environ.get('SLURM_JOB_ID'),seconds=time.time()-start,error=repr(e),traceback=traceback.format_exc(),freeze_sha256=freeze));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','gate','run']);p.add_argument('--seed',type=int);p.add_argument('--name');a=p.parse_args()
    if a.mode=='check':check()
    elif a.mode=='gate':
        for anchor in ('ce','dkd'):execute(HERE/'configs'/f'gate_{anchor}_{a.seed}.json',f'gate_{anchor}_{a.seed}',True)
        write(HERE/'gates'/f'{a.seed}.json',dict(status='COMPLETE',freeze_sha256=verify(),seed=a.seed,job=os.environ.get('SLURM_JOB_ID'),summaries={},technical=True))
    else:execute(HERE/'configs'/f'{a.name}.json',a.name)

