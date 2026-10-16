"""Server-side numerical contract tests for the real mf_vcs PyTorch backend.

Run explicitly in an existing PyTorch environment. This file does not install
packages, submit jobs, train data, or claim GPU timing. Its JSON report is only
written after successful execution. --device cuda is an optional correctness
check, not a benchmark. Primary-runtime local execution was unavailable.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from mf_vcs import (SceneBatch, QuadraticHead, SimplexBilinearCritic,
                    dense_objective_terms, objective_terms, partition_scenes,
                    profiled_objective, sampled_permutation_terms, solve_box_head)


def run(device):
    torch.manual_seed(20261016)
    device = torch.device(device)
    records = []
    def check(name, error, limit, note=""):
        err = float(error)
        row = {"name":name,"error":err,"limit":limit,"passed":bool(torch.isfinite(torch.tensor(err)) and err <= limit),"note":note}
        records.append(row)
        if not row["passed"]:
            raise AssertionError(row)
    def expect_error(name, fun, error=(ValueError, NotImplementedError)):
        try:
            fun()
        except error:
            check(name,0,0)
        else:
            raise AssertionError(f"{name}: expected {error}")
    def rand_features(n,r,s,dtype):
        za = torch.randn(n,r,device=device,dtype=dtype,requires_grad=True)
        zb = torch.randn(n,s,device=device,dtype=dtype,requires_grad=True)
        return za,zb,za.softmax(-1),zb.softmax(-1)
    for dtype in (torch.float64,torch.float32):
        value_tol,grad_tol = (2e-12,2e-11) if dtype == torch.float64 else (2e-5,3e-5)
        for mode in ("v","u","independent"):
            za,zb,a,b=rand_features(6,3,2,dtype)
            w=torch.empty(3,2,device=device,dtype=dtype).uniform_(-.8,.8).requires_grad_()
            leaves=[za,zb,w]
            if mode=="independent":
                zqa=torch.randn(7,3,device=device,dtype=dtype,requires_grad=True)
                zqb=torch.randn(9,2,device=device,dtype=dtype,requires_grad=True)
                scene=SceneBatch(a,b,mode,zqa.softmax(-1),zqb.softmax(-1))
                leaves.extend([zqa,zqb])
            else:
                scene=SceneBatch(a,b,mode)
            for ridge in (0.,.07):
                exact=objective_terms([scene],w,weights=[1.],ridge=ridge)
                for chunk,checkpointed in ((None,False),(2,False),(2,True)):
                    dense=dense_objective_terms([scene],w,weights=[1.],ridge=ridge,chunk_size=chunk,checkpoint_blocks=checkpointed)
                    check(f"value_{dtype}_{mode}_{ridge}_{chunk}_{checkpointed}",(exact.f-dense.f).abs(),value_tol)
                    ga=torch.autograd.grad(exact.f,leaves,retain_graph=True)
                    gb=torch.autograd.grad(dense.f,leaves,retain_graph=True)
                    check(f"gradient_{dtype}_{mode}_{ridge}_{chunk}_{checkpointed}",max(float((x-y).abs().max()) for x,y in zip(ga,gb)),grad_tol)
                head=QuadraticHead([scene],weights=[1.],ridge=ridge)
                hvalue,hgrad=head.value_and_gradient(w)
                true_grad=torch.autograd.grad(exact.f,w,create_graph=True,retain_graph=True)[0]
                direction=torch.randn_like(w)
                true_hv=-torch.autograd.grad((true_grad*direction).sum(),w,retain_graph=True)[0]
                check(f"head_value_{dtype}_{mode}_{ridge}",(hvalue-exact.f).abs(),value_tol)
                check(f"head_gradient_{dtype}_{mode}_{ridge}",(hgrad-true_grad).abs().max(),grad_tol)
                check(f"head_HVP_{dtype}_{mode}_{ridge}",(head.hvp(direction)-true_hv).abs().max(),grad_tol)
    za,zb,a,b=rand_features(5,3,2,torch.float64)
    w=torch.empty(3,2,device=device,dtype=torch.float64).uniform_(-.7,.7).requires_grad_()
    scene=SceneBatch(a,b,"v")
    perms=torch.tensor(list(itertools.permutations(range(5))),device=device,dtype=torch.long)
    sampled=sampled_permutation_terms([scene],w,weights=[1.],permutations=[perms])
    full=objective_terms([scene],w,weights=[1.])
    check("all_ordinary_permutations_V_value",(sampled.f-full.f).abs(),2e-12)
    gs=torch.autograd.grad(sampled.f,(za,zb,w),retain_graph=True)
    gf=torch.autograd.grad(full.f,(za,zb,w),retain_graph=True)
    check("all_ordinary_permutations_V_gradients",max(float((x-y).abs().max()) for x,y in zip(gs,gf)),2e-11)
    # Two local scenes, unequal sizes, explicit fixed weights; pooling is not used.
    za,zb,a,b=rand_features(9,3,2,torch.float64)
    ids=torch.tensor([4,4,4,8,8,8,8,8,8],device=device,dtype=torch.long)
    grouped=partition_scenes(a,b,ids,q_mode="v")
    weighted=objective_terms(grouped,w,weights=[.3,.7],ridge=.1)
    dense=dense_objective_terms(grouped,w,weights=[.3,.7],ridge=.1)
    check("grouped_values",(weighted.f-dense.f).abs(),2e-12)
    gx=torch.autograd.grad(weighted.f,(za,zb,w),retain_graph=True)
    gy=torch.autograd.grad(dense.f,(za,zb,w),retain_graph=True)
    check("grouped_gradients",max(float((x-y).abs().max()) for x,y in zip(gx,gy)),2e-11)
    # Snapshot must be independent of later caller storage mutation.
    clone_a,clone_b=a.detach().clone(),b.detach().clone()
    original=SceneBatch(clone_a,clone_b,"v")
    snap=QuadraticHead([original],weights=[1.])
    old=snap.hvp(w.detach()).clone()
    clone_a.fill_(1/clone_a.shape[1])
    clone_b.fill_(1/clone_b.shape[1])
    check("head_snapshot_not_aliased",(old-snap.hvp(w.detach())).abs().max(),0)
    # Box-active hard features test, then a general fixed-scene head.
    hard=torch.eye(2,device=device,dtype=torch.float64)
    for scenes,weights,label in (([SceneBatch(hard,hard,"u")],[1.],"active"),(grouped,[.3,.7],"weighted")):
        for ridge in (0.,.05):
            head=QuadraticHead(scenes,weights=weights,ridge=ridge)
            solved=solve_box_head(head,tolerance=1e-8,max_iterations=30000)
            check(f"solver_gap_{label}_{ridge}",solved.fw_gap,1e-8)
            check(f"solver_box_{label}_{ridge}",max(0,float(solved.w.abs().max())-1),0)
            terms=objective_terms(scenes,solved.w,weights=weights,ridge=ridge)
            check(f"solver_report_F_{label}_{ridge}",abs(float(terms.f)-solved.f),2e-12)
            check(f"solver_report_J_{label}_{ridge}",abs(float(terms.j)-solved.j),2e-12)
            trial=torch.empty_like(solved.w).uniform_(-1,1)
            trial_f,_=head.value_and_gradient(trial)
            check(f"FW_feasible_improvement_{label}_{ridge}",max(0,solved.f-float(trial_f)-float(head.fw_gap(trial))),2e-12)
    # M1 returns same-tensor fixed-head partial gradients of F with ridge.
    profile=profiled_objective(grouped,weights=[.3,.7],ridge=.2,tolerance=1e-9,max_iterations=30000)
    manual=objective_terms(grouped,profile.solution.w.detach(),weights=[.3,.7],ridge=.2)
    gp=torch.autograd.grad(profile.terms.f,(za,zb),retain_graph=True)
    gm=torch.autograd.grad(manual.f,(za,zb),retain_graph=True)
    check("M1_same_tensor_F_gradients",max(float((x-y).abs().max()) for x,y in zip(gp,gm)),0)
    check("M1_head_is_detached",1 if profile.solution.w.requires_grad else 0,0)
    check("M1_outer_inner_F_agree",abs(float(profile.terms.f)-profile.solution.f),2e-12)
    # A separate scalar perturbation recomputes each true regularized profile;
    # compare its central difference with the original-tensor envelope partial.
    base_logits=za.detach().clone()
    direction=torch.randn_like(base_logits)
    fixed_b=b.detach().clone()
    def scalar_profile(theta):
        one=SceneBatch((base_logits+theta*direction).softmax(-1),fixed_b,"v")
        return profiled_objective([one],weights=[1.],ridge=.2,tolerance=1e-11,max_iterations=30000)
    theta=torch.zeros((),device=device,dtype=torch.float64,requires_grad=True)
    at_zero=scalar_profile(theta)
    envelope=torch.autograd.grad(at_zero.terms.f,theta)[0]
    hstep=1e-4
    fp=scalar_profile(torch.tensor(hstep,device=device,dtype=torch.float64)).terms.f
    fm=scalar_profile(torch.tensor(-hstep,device=device,dtype=torch.float64)).terms.f
    check("M1_ridge_F_actual_profile_finite_difference",(envelope-(fp-fm)/(2*hstep)).abs(),2e-7,"One smooth regularized example; not a general approximate-envelope theorem")
    for name,fun in (
        ("negative_weights_rejected",lambda:objective_terms(grouped,w,weights=[-.2,1.2])),
        ("unnormalized_weights_rejected",lambda:objective_terms(grouped,w,weights=[1.,1.])),
        ("tensor_weights_rejected",lambda:objective_terms(grouped,w,weights=torch.tensor([.3,.7]))),
        ("forbidden_edges_rejected",lambda:SceneBatch(a,b,forbidden_edges=torch.ones(9,9))),
        ("U_singleton_rejected",lambda:SceneBatch(a[:1],b[:1],"u")),
        ("independent_missing_pool_rejected",lambda:SceneBatch(a,b,"independent")),
        ("out_of_box_rejected",lambda:objective_terms(grouped,torch.full_like(w,1.01),weights=[.3,.7])),
        ("negative_simplex_rejected",lambda:SceneBatch(-a,b)),
        ("unnormalized_simplex_rejected",lambda:SceneBatch(2*a,b)),
        ("ordinary_for_U_rejected",lambda:sampled_permutation_terms([SceneBatch(a,b,"u")],w,weights=[1.],permutations=[torch.arange(9,device=device)[None,:]])),
    ):
        expect_error(name,fun)
    critic=SimplexBilinearCritic(4,5,3,2).to(device=device,dtype=torch.float64)
    with torch.no_grad():
        critic.W.uniform_(-3,3)
        critic.project_head_()
    x=torch.randn(11,4,device=device,dtype=torch.float64)
    y=torch.randn(11,5,device=device,dtype=torch.float64)
    check("critic_global_pair_bound",max(0,float(critic(x,y).abs().max())-1),2e-14)
    # Deterministic insufficient-budget status is explicit and not accepted by M1.
    hard_head=QuadraticHead([SceneBatch(hard,hard,"u")],weights=[1.])
    unfinished=solve_box_head(hard_head,max_iterations=0)
    check("solver_budget_exhausted_flag",1 if unfinished.converged else 0,0)
    expect_error("M1_incomplete_solve_rejected",lambda:profiled_objective([SceneBatch(hard,hard,"u")],weights=[1.],max_iterations=0),RuntimeError)
    report={"status":"passed","scope":"PyTorch value/gradient/HVP/head/M1 correctness; no training or timing", "device":str(device),"torch_version":torch.__version__,"n_checks":len(records),"checks":records}
    path=Path(__file__).with_name("MATH_TORCH_AUDIT_"+str(device).replace(":","_")+".json")
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","n_checks":len(records),"report":str(path)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    run(args.device)
