"""Explicit local head-only benchmark; never submits jobs or installs packages.

Default = small smoke grid, CPU, fp32. Full N/rank grid requires --full.
Use --device cuda only in an authorized existing GPU environment. CUDA absence
is an error, never a silent CPU fallback or fabricated GPU measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from mf_vcs import (SceneBatch, dense_objective_terms, draw_ordinary_permutations,
                    objective_terms, sampled_permutation_terms)


def run(args):
    device=torch.device(args.device)
    if device.type not in ("cpu","cuda"):
        raise SystemExit("Only CPU/CUDA benchmark devices are supported")
    if device.type=="cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no GPU benchmark was run")
    if args.repeats<1 or args.warmup<0 or args.block_size<1:
        raise SystemExit("repeats/block-size must be positive; warmup must be nonnegative")
    dtype=getattr(torch,args.dtype)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    grid=[(n,k) for n in (128,512,2048,8192) for k in (8,16,32)] if args.full else [(32,4),(128,8)]
    atol=3e-5 if dtype==torch.float32 else 3e-11
    out=Path(args.output) if args.output else Path(__file__).with_name("BENCHMARK_TORCH_"+str(device).replace(":","_")+"_"+args.dtype+("_full" if args.full else "_smoke")+".json")
    report={"status":"running","scope":"Head-only synthetic V objective; encoder/input construction excluded; no task-quality comparison",
            "torch_version":torch.__version__,"device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            "dtype":args.dtype,"seed":args.seed,"full_grid":args.full,"grid":grid,"warmup":args.warmup,"repeats":args.repeats,
            "negative_target":"within-one-scene V; ordinary permutations retain fixed points",
            "timing":"per-call forward+autograd.grad(A,B,W), wall clock; CUDA synchronized before/after each call",
            "memory":"CUDA allocated/reserved peaks after method warmup; CPU memory not measured",
            "validation":"Inputs validated before timing; check_w/check_permutations disabled inside timed calls",
            "sampling":"Fixed ordinary permutations across timing repeats; perm1 is first row of perm8; no pairing-variance claim",
            "dense_block":"non-reentrant checkpoint on each row block, recomputed during backward; all eligible scores evaluated",
            "rows":[]}
    def save():
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    def synchronize():
        if device.type=="cuda":
            torch.cuda.synchronize(device)
    def release():
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
    for n,k in grid:
        # A/B are fixed simplex-feature leaves for every method in this cell.
        a=torch.randn(n,k,device=device,dtype=dtype,generator=generator).softmax(-1).detach().requires_grad_()
        b=torch.randn(n,k,device=device,dtype=dtype,generator=generator).softmax(-1).detach().requires_grad_()
        w=torch.empty(k,k,device=device,dtype=dtype).uniform_(-.8,.8,generator=generator).requires_grad_()
        scenes=(SceneBatch(a,b,"v"),)
        leaves=(a,b,w)
        all_perms=draw_ordinary_permutations(scenes,8,generator=generator)
        one_perm=(all_perms[0][:1],)
        # Explicit initial checks, identical tensors, no model/optimizer update.
        objective_terms(scenes,w,weights=[1.])
        sampled_permutation_terms(scenes,w,weights=[1.],permutations=all_perms)
        methods={
            "perm1":lambda:sampled_permutation_terms(scenes,w,weights=[1.],permutations=one_perm,check_w=False,check_permutations=False).j,
            "perm8":lambda:sampled_permutation_terms(scenes,w,weights=[1.],permutations=all_perms,check_w=False,check_permutations=False).j,
            "dense":lambda:dense_objective_terms(scenes,w,weights=[1.],check_w=False).j,
            "dense_block_checkpoint":lambda:dense_objective_terms(scenes,w,weights=[1.],chunk_size=min(n,args.block_size),checkpoint_blocks=True,check_w=False).j,
            "moments":lambda:objective_terms(scenes,w,weights=[1.],check_w=False).j,
        }
        def evaluate(fun):
            value=fun()
            grads=torch.autograd.grad(value,leaves)
            return value,grads
        ref_f,ref_g=evaluate(methods["moments"])
        reference_value=float(ref_f.detach().cpu())
        reference_grads=tuple(g.detach().cpu().clone() for g in ref_g)
        del ref_f,ref_g
        release()
        for name,fun in methods.items():
            value=grads=None
            row={"n":n,"rank":k,"method":name,"block_size":min(n,args.block_size) if "block" in name else None,
                 "status":"running","objective_comparison":"exact V" if name in ("dense","dense_block_checkpoint","moments") else "stochastic fixed-permutation realization; error vs V is sampling deviation"}
            try:
                value,grads=evaluate(fun)
                value_error=abs(float(value.detach().cpu())-reference_value)
                gradient_errors={label:float((g.detach().cpu()-ref).abs().max()) for label,g,ref in zip(("A","B","W"),grads,reference_grads)}
                row.update({"J":float(value.detach().cpu()),"reference_J":reference_value,"value_absolute_error_vs_V":value_error,"gradient_max_absolute_errors_vs_V":gradient_errors})
                del value,grads
                exact=name in ("dense","dense_block_checkpoint","moments")
                correct=value_error<=atol and max(gradient_errors.values())<=atol
                if exact and not correct:
                    row.update({"status":"correctness_failed","absolute_tolerance":atol})
                    report["rows"].append(row)
                    save()
                    release()
                    continue
                release()
                for _ in range(args.warmup):
                    value,grads=evaluate(fun)
                    del value,grads
                synchronize()
                if device.type=="cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline_allocated=torch.cuda.memory_allocated(device)
                    baseline_reserved=torch.cuda.memory_reserved(device)
                durations=[]
                for _ in range(args.repeats):
                    synchronize()
                    started=time.perf_counter()
                    value,grads=evaluate(fun)
                    synchronize()
                    durations.append(1000*(time.perf_counter()-started))
                    del value,grads
                row.update({"status":"completed","forward_backward_ms_median":statistics.median(durations),"forward_backward_ms_min":min(durations),"forward_backward_ms_all":durations,
                            "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                            "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
                            "cuda_baseline_allocated_bytes":baseline_allocated if device.type=="cuda" else None,
                            "cuda_baseline_reserved_bytes":baseline_reserved if device.type=="cuda" else None})
            except Exception as exc:
                message=str(exc)
                is_oom=isinstance(exc,torch.cuda.OutOfMemoryError) or "out of memory" in message.lower() or "can't allocate memory" in message.lower()
                row.update({"status":"oom" if is_oom else "error","exception_type":type(exc).__name__,"message":message[:1500]})
            # Also release outputs if a CUDA synchronization failed after a
            # successful forward/backward assignment, before continuing a cell.
            value=grads=None
            report["rows"].append(row)
            save()
            print(json.dumps({"n":n,"rank":k,"method":name,"status":row["status"]}),flush=True)
            release()
        del methods,leaves,scenes,a,b,w,all_perms,one_perm,reference_grads
        release()
    failures=[r for r in report["rows"] if r["status"] in ("error","correctness_failed")]
    report["status"]="completed_with_errors" if failures else "completed"
    report["oom_count"]=sum(r["status"]=="oom" for r in report["rows"])
    save()
    print(json.dumps({"status":report["status"],"report":str(out),"oom_count":report["oom_count"]}))
    if failures:
        raise SystemExit(2)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--dtype",choices=("float32","float64"),default="float32")
    parser.add_argument("--full",action="store_true",help="Explicitly select the N={128,512,2048,8192}, rank={8,16,32} grid")
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--block-size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=20261016)
    parser.add_argument("--output")
    run(parser.parse_args())
