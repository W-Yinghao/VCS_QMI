"""Task-coordinate alignment; all arms use centered, normalized class logits without a projection head."""
import torch
import torch.nn.functional as F

def coordinates(x):
    return F.normalize(x-x.mean(1,keepdim=True),dim=1,eps=1e-8)

def quadratic_u(u,v):
    n=u.shape[0]
    if n<2:raise ValueError('U-statistic requires at least two samples')
    p=(u*v).sum(1)
    q1=(u.sum(0).dot(v.sum(0))-p.sum())/(n*(n-1))
    q2=((u.T@u)*(v.T@v)).sum()-p.square().sum()
    q2=q2/(n*(n-1))
    return (p-.5*p.square()).mean()-q1-.5*q2

def decision_loss(student,teacher,method):
    u,v=coordinates(student),coordinates(teacher.detach())
    p=(u*v).sum(1)
    if method=='cosine':loss=(1-p).mean()
    elif method=='vcs':loss=-quadratic_u(u,v)
    elif method=='bce':
        n=len(u);scores=u@v.T;mask=~torch.eye(n,dtype=torch.bool,device=u.device)
        loss=-torch.log(((1+p)/2).clamp(1e-6,1-1e-6)).mean()-torch.log(((1-scores[mask])/2).clamp(1e-6,1-1e-6)).mean()
    else:raise ValueError(method)
    with torch.no_grad():
        scores=u@v.T;mask=~torch.eye(len(u),dtype=torch.bool,device=u.device)
        diagnostics=dict(method=method,positive_cosine=float(p.mean()),product_cosine=float(scores[mask].mean()),
            positive_abs_T_ge099=float((p.abs()>=.99).float().mean()),raw_auxiliary=float(loss),
            raw_j=float(quadratic_u(u,v)),q='off-diagonal ordered cross-instance U-statistic')
    return loss,diagnostics
