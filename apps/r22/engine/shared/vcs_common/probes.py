"""Mathematical/cost probes shared by KD and AIGIQA (no task-performance claims).

Run: python -m vcs_common.probes math --output path.json
     python -m vcs_common.probes benchmark --device cuda --output cost.json
     python -m vcs_common.probes variance --output sampling.json
"""
from __future__ import annotations

import argparse
import copy
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dependence import DependenceRegularizer, sampled_quadratic, sampled_indices
from .mf_backend import SceneBatch, objective_terms, dense_objective_terms
from .runtime import runtime_record, write_json, synchronize, sha256


def _value_gradient(fn, tensors):
    value = fn(*tensors)
    grads = torch.autograd.grad(value, tensors)
    return float(value.detach()), grads


def toy_task_counterexample() -> dict:
    """Analytic distributions; not a trained image-model benchmark."""
    outputs = []
    for factor in (0, 1):
        p = np.zeros((4, 2), dtype=np.float64)
        for y in range(2):
            for nuisance in range(2):
                u = 2*y+nuisance
                p[u, (y, nuisance)[factor]] = 0.25
        q = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
        m = 0.5*(p+q)
        eta = (p-q)/(p+q)
        s = float((m*eta**2).sum())
        outputs.append(dict(retained_factor="task_label" if factor == 0 else "independent_nuisance",
                            s=s, bayes_task_accuracy=1.0 if factor == 0 else 0.5))
    return dict(scope="exact_discrete_probability_calculation", results=outputs,
                conclusion="Same S can retain different task information; no task guarantee")


def run_math() -> dict:
    torch.manual_seed(19261019)
    checks = []
    for dtype in (torch.float64, torch.float32):
        for qmode in ("u", "v"):
            x = torch.randn(11, 7, dtype=dtype, requires_grad=True)
            y = torch.randn(11, 5, dtype=dtype, requires_grad=True)
            w = (torch.rand(7, 5, dtype=dtype)*1.8-0.9).requires_grad_()
            def f(a, b, c):
                return objective_terms([SceneBatch(a.softmax(-1), b.softmax(-1), q_mode=qmode)],
                                       c, weights=[1.0]).j
            def dense(a, b, c):
                return dense_objective_terms([SceneBatch(a.softmax(-1), b.softmax(-1), q_mode=qmode)],
                                             c, weights=[1.0], chunk_size=4, checkpoint_blocks=True).j
            v, g = _value_gradient(f, (x, y, w))
            vd, gd = _value_gradient(dense, (x, y, w))
            error = max(float((aa-bb).abs().max()) for aa, bb in zip(g, gd))
            # Rounded fp32 inputs compared to an independent fp64 evaluation.
            xr, yr, wr = [t.detach().double().requires_grad_() for t in (x, y, w)]
            vr, gr = _value_gradient(dense, (xr, yr, wr))
            reference_error = max(float((aa.double()-bb).abs().max()) for aa, bb in zip(g, gr))
            tol = 2e-12 if dtype == torch.float64 else 2e-6
            if abs(v-vd) > tol or error > tol or abs(v-vr) > tol or reference_error > tol:
                raise AssertionError("MF/dense value or all-parameter gradient mismatch")
            checks.append(dict(name="moment_dense_value_and_xyz_gradient", dtype=str(dtype),
                q_mode=qmode, value_error=abs(v-vd), gradient_error=error,
                fp64_reference_error=reference_error, tolerance=tol, passed=True))
    # Exact enumeration over all cyclic shifts proves U marginal expectation.
    a = torch.randn(9, 4, dtype=torch.float64).softmax(-1).requires_grad_()
    b = torch.randn(9, 6, dtype=torch.float64).softmax(-1).requires_grad_()
    w = (torch.rand(4, 6, dtype=torch.float64)*2-1).requires_grad_()
    idx = (torch.arange(9)[None]+torch.arange(1,9)[:,None]) % 9
    js, _ = sampled_quadratic(a,b,w,idx)
    gs = torch.autograd.grad(js, (a,b,w))
    je = objective_terms([SceneBatch(a,b,q_mode="u")],w,weights=[1.0]).j
    ge = torch.autograd.grad(je,(a,b,w))
    if not torch.allclose(js,je,atol=2e-14,rtol=2e-12):
        raise AssertionError("Cyclic U mean mismatch")
    if any(not torch.allclose(u,v,atol=2e-14,rtol=2e-12) for u,v in zip(gs,ge)):
        raise AssertionError("Cyclic U mean gradient mismatch")
    checks.append(dict(name="all_nonzero_shifts_match_U_value_and_gradient", passed=True,
                       sampler="uniform_nonzero_cyclic_shift_not_uniform_derangements"))
    # Exact identity S-J for a fixed finite P/Q and arbitrary bounded critic.
    p = np.array([[0.34,0.04,0.02],[0.03,0.22,0.05],[0.01,0.04,0.25]])
    q = p.sum(1,keepdims=True) @ p.sum(0,keepdims=True)
    m = (p+q)/2
    eta = (p-q)/(p+q)
    h = np.array([[0.8,-0.2,-0.5],[-0.3,0.7,-0.4],[-0.6,0.1,0.9]])
    s = float((m*eta**2).sum())
    j = float(((p-q)*h-m*h*h).sum())
    error = float((m*(h-eta)**2).sum())
    if not np.isclose(s-j,error,atol=1e-14,rtol=0):
        raise AssertionError("Fixed-theory identity failed")
    checks.append(dict(name="population_S_minus_J_identity", s=s, j=j,
                       mixture_squared_error=error, error=abs(s-j-error),passed=True))
    # Relative solver must not stop at W=0 when a nonzero G0 is < old 1e-4.
    x = torch.randn(24,4,dtype=torch.float64)*0.005
    y = x[:,:3] + torch.randn(24,3,dtype=torch.float64)*0.001
    reg = DependenceRegularizer(4,3,method="vcs_m1",rank=4,solver_max_iter=30).double()
    before = reg.W.detach().clone()
    loss = reg(x,y)
    loss.backward()
    diag = copy.deepcopy(reg.diagnostics)
    if diag["solver_zero_gap"] <= 0 or diag["solver_iterations"] == 0 or torch.equal(before,reg.W):
        raise AssertionError("M1 incorrectly stopped at nonstationary zero head")
    checks.append(dict(name="M1_relative_zero_reference_nonzero_update", **diag, passed=True))
    # Matched posterior BCE algebra, incl. equal P/Q priors (not count-weighted).
    reg = DependenceRegularizer(4,3,method="bce",rank=4).double()
    with torch.no_grad(): reg.W.uniform_(-0.8,0.8)
    actual = reg(x,y)
    a,b = reg.head.features(x,y)
    h = a@reg.W@b.T
    mask = ~torch.eye(len(x),dtype=torch.bool)
    raw_bce = -0.5*(torch.log((1+h.diagonal())/2).mean()+torch.log((1-h[mask])/2).mean())
    expected = 2*(raw_bce-math.log(2))
    if not torch.allclose(actual,expected,rtol=1e-12,atol=1e-14):
        raise AssertionError("Matched-BCE posterior/weight convention mismatch")
    checks.append(dict(name="balanced_BCE_same_bounded_posterior_family", objective_scale=2,
                       raw_balanced_bce=float(raw_bce.detach()), passed=True))
    # Fixed analytic scale matches -J first and second derivatives at W=0.
    zero_w = torch.zeros_like(reg.W, requires_grad=True)
    aa,bb = a.detach(),b.detach()
    def matched(w):
        hh=aa@w@bb.T
        return -(torch.log1p(hh.diagonal()).mean()+torch.log1p(-hh[mask]).mean())
    def quadratic(w):
        hh=aa@w@bb.T
        return -(hh.diagonal().mean()-hh[mask].mean()-.5*(hh.diagonal().square().mean()+hh[mask].square().mean()))
    if not torch.allclose(torch.autograd.functional.jacobian(matched,zero_w),torch.autograd.functional.jacobian(quadratic,zero_w),atol=1e-13,rtol=1e-12):
        raise AssertionError("Normalized BCE/CS local first derivative mismatch")
    if not torch.allclose(torch.autograd.functional.hessian(matched,zero_w),torch.autograd.functional.hessian(quadratic,zero_w),atol=1e-13,rtol=1e-12):
        raise AssertionError("Normalized BCE/CS local curvature mismatch")
    checks.append(dict(name="normalized_BCE_minusJ_gradient_and_hessian_at_zero",passed=True))
    return dict(status="PASS", checks=checks, check_count=len(checks),
                task_counterexample=toy_task_counterexample(), runtime=runtime_record(),
                coverage="CPU math and gradient; no real data, pretrained models, CUDA or application efficacy")


def run_benchmark(device: str, sizes: list[int], ranks: list[int], repeats: int, warmup: int) -> dict:
    """Measure shared branch including projections+backward, no optimizer/encoder.

    All methods use same generated feature tensors/state. CUDA max allocation is
    measured per method; report peak including persistent input/model, plus delta.
    This is a MICRO benchmark; application trainers must report whole-run cost.
    """
    if repeats < 3 or warmup < 1:
        raise ValueError("Need >=3 measured repetitions and >=1 warmup")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no substitution of CPU results")
    rows = []
    for n in sizes:
        for rank in ranks:
            torch.manual_seed(20261019 + n + rank)
            # Higher-dimensional real feature interface, same for all variants.
            x0 = torch.randn(n,128,device=dev)
            y0 = torch.randn(n,96,device=dev)
            base = DependenceRegularizer(128,96,rank=rank).to(dev)
            with torch.no_grad(): base.W.uniform_(-0.2,0.2)
            state = copy.deepcopy(base.state_dict())
            for method in ("vcs_exact","vcs_dense","vcs_sampled","bce","infonce","vcs_m1"):
                reg = DependenceRegularizer(128,96,method=method,rank=rank,
                    dense_chunk_size=min(128,n),dense_checkpoint=True,solver_max_iter=30).to(dev)
                reg.load_state_dict(state)
                def once():
                    reg.zero_grad(set_to_none=True)
                    # Reset the SAME feasible start every repetition: reusing
                    # the already-solved head on repeated fixed inputs would
                    # make later M1 timings misleadingly approach zero steps.
                    with torch.no_grad(): reg.W.copy_(state["head.W"])
                    x=x0.detach().requires_grad_();y=y0.detach().requires_grad_()
                    reg.prepare_step(x,y)
                    loss=reg(x,y)
                    loss.backward()
                try:
                    for _ in range(warmup): once()
                    synchronize(dev)
                    initial_memory = 0
                    if dev.type=="cuda":
                        initial_memory=torch.cuda.memory_allocated(dev)
                        torch.cuda.reset_peak_memory_stats(dev)
                    timing=[]
                    for _ in range(repeats):
                        synchronize(dev);start=time.perf_counter()
                        once()
                        synchronize(dev);timing.append(time.perf_counter()-start)
                    peak=torch.cuda.max_memory_allocated(dev) if dev.type=="cuda" else None
                    rows.append(dict(n=n,rank=rank,method=method,status="COMPLETE",
                        median_seconds=statistics.median(timing),min_seconds=min(timing),max_seconds=max(timing),
                        peak_allocated_bytes=peak,allocation_baseline_bytes=initial_memory if peak is not None else None,
                        peak_increment_bytes=peak-initial_memory if peak is not None else None,
                        solver=reg.diagnostics if method=="vcs_m1" else None))
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(n=n,rank=rank,method=method,status="OOM"))
                del reg
                if dev.type=="cuda": torch.cuda.empty_cache()
    return dict(status="MEASURED", scope="dependence_projection_forward_backward_microbenchmark",
                includes_encoder=False,includes_optimizer=False, Q="U", dtype="float32",
                repeated_inputs=True, head_start_reset_each_repeat=True,
                sampling_uncertainty_is_not_independent_data=True,
                repeats=repeats,warmup=warmup,rows=rows,runtime=runtime_record())


def run_variance(snapshot_npz: str | None = None, draws: list[int] | None = None,
                 repetitions: int = 128, seed: int = 20261019) -> dict:
    """Conditional negative-pair sampling variation at ONE fixed critic/batch.

    Optional NPZ keys a,b,w contain simplex features and feasible head from a
    FIT checkpoint. No images/labels, encoder updates or new data are sampled.
    Derivatives are with respect to a,b,w (before any encoder Jacobian).
    Finite Monte Carlo bias is reported, never used as an equality gate.
    """
    if repetitions < 2:
        raise ValueError("At least two independent sampling repetitions required")
    draws = [1, 8, 32] if draws is None else draws
    if not draws or any(k < 1 for k in draws) or len(set(draws)) != len(draws):
        raise ValueError("Draw counts must be distinct positive integers")
    torch.manual_seed(seed)
    if snapshot_npz:
        with np.load(snapshot_npz, allow_pickle=False) as z:
            if set(z.files) != {"a", "b", "w"}:
                raise ValueError("Snapshot must contain exactly a,b,w; do not include TEST labels")
            a,b,w = [torch.as_tensor(np.array(z[k]), dtype=torch.float64) for k in ("a","b","w")]
        origin = {"kind": "provided_fixed_features", "snapshot_sha256": sha256(snapshot_npz)}
    else:
        latent = torch.randn(64, 8, dtype=torch.float64)
        a = (latent @ torch.randn(8, 12, dtype=torch.float64) / 2).softmax(-1)
        b = (latent @ torch.randn(8, 10, dtype=torch.float64) / 2).softmax(-1)
        w = torch.rand(12, 10, dtype=torch.float64)*1.2-.6
        origin = {"kind": "synthetic_correlated_simplex_fixture", "n": 64,
                  "application_performance_evidence": False}
    if a.ndim != 2 or b.ndim != 2 or w.shape != (a.shape[1], b.shape[1]) or len(a) != len(b) or len(a) < 2:
        raise ValueError("Incompatible snapshot shapes or singleton batch")
    if not all(bool(torch.isfinite(t).all()) for t in (a,b,w)) or bool(w.abs().max() > 1):
        raise ValueError("Snapshot must be finite and head feasible")
    for features in (a,b):
        if bool(features.min() < 0) or not torch.allclose(features.sum(-1), torch.ones(len(a), dtype=features.dtype), atol=2e-6, rtol=2e-6):
            raise ValueError("Snapshot a,b must be simplex rows")
    a,b,w = [t.detach().requires_grad_() for t in (a,b,w)]
    def vector(grads):
        return torch.cat([g.detach().reshape(-1) for g in grads])
    exact = objective_terms([SceneBatch(a,b,q_mode="u")], w, weights=[1.0]).j
    exact_value = float(exact.detach())
    exact_gradient = vector(torch.autograd.grad(exact, (a,b,w)))
    exact_norm = float(exact_gradient.norm())
    rows = []
    for count in draws:
        generator = torch.Generator(device="cpu").manual_seed(seed+count*100003)
        values, gradients = [], []
        for _ in range(repetitions):
            indices = sampled_indices(len(a), count, "u", device=a.device, generator=generator)
            sample,_ = sampled_quadratic(a,b,w,indices)
            values.append(float(sample.detach()))
            gradients.append(vector(torch.autograd.grad(sample,(a,b,w))))
        values = np.asarray(values, dtype=np.float64)
        gradients = torch.stack(gradients)
        centered = gradients - gradients.mean(0)
        grad_delta = gradients - exact_gradient[None]
        squared_l2_error = grad_delta.square().sum(1)
        rows.append(dict(draws=count, repetitions=repetitions,
            mean_j=float(values.mean()), exact_j=exact_value,
            monte_carlo_j_bias=float(values.mean()-exact_value),
            j_sample_variance=float(values.var(ddof=1)),
            j_mean_squared_error=float(np.mean((values-exact_value)**2)),
            gradient_mean_bias_l2=float((gradients.mean(0)-exact_gradient).norm()),
            gradient_variance_trace=float(centered.square().sum()/(repetitions-1)),
            gradient_mean_squared_l2_error=float(squared_l2_error.mean()),
            gradient_relative_rms_error=float(squared_l2_error.mean().sqrt()/exact_norm) if exact_norm > 0 else None,
            gradient_cosine_median=float(F.cosine_similarity(gradients,exact_gradient[None],dim=1).median()) if exact_norm > 0 else None))
    return dict(status="MEASURED", scope="conditional_negative_pair_sampling_only",
        origin=origin, q_mode="u", sampler="uniform_nonzero_cyclic_shift",
        exact_negative_pair_sampling_variance=0.0, exact_gradient_l2=exact_norm,
        derivative_scope=["simplex_a","simplex_b","W"], dtype="float64",
        n=int(len(a)), rank_a=int(a.shape[1]), rank_b=int(b.shape[1]),
        seed=seed, rows=rows, runtime=runtime_record(),
        interpretation="The exact branch removes conditional Q-sampling variation, not data/augmentation/optimization variation. Empirical variances need not be monotone in finite repetitions.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    mathp=sub.add_parser("math");mathp.add_argument("--output",required=True)
    bench=sub.add_parser("benchmark");bench.add_argument("--output",required=True)
    bench.add_argument("--device",default="cuda")
    bench.add_argument("--sizes",nargs="+",type=int,default=[32,128,512,2048])
    bench.add_argument("--ranks",nargs="+",type=int,default=[16,32,64])
    bench.add_argument("--repeats",type=int,default=20);bench.add_argument("--warmup",type=int,default=3)
    variance=sub.add_parser("variance");variance.add_argument("--output",required=True)
    variance.add_argument("--snapshot-npz")
    variance.add_argument("--draws",nargs="+",type=int,default=[1,8,32])
    variance.add_argument("--repetitions",type=int,default=128);variance.add_argument("--seed",type=int,default=20261019)
    args=parser.parse_args()
    torch.set_num_threads(1) if args.command in {"math","variance"} else None
    if args.command=="math":
        result=run_math()
    elif args.command=="variance":
        result=run_variance(args.snapshot_npz,args.draws,args.repetitions,args.seed)
    else:
        result=run_benchmark(args.device,args.sizes,args.ranks,args.repeats,args.warmup)
    write_json(args.output,result)
    print(result["status"],Path(args.output).resolve())


if __name__=="__main__":
    main()
