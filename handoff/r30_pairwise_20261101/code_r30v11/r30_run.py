"""R30 v1.1 entry point: pinned r25_run arguments plus the R30 stage arguments. Noisy branch always the R27 A1 interface
(arm a1_iface, lam 0). No TEST at the end.
  --r30_stage warmup : --r30_parent_id, --r30_source_warmup H|L: one parent state from the seed-determined untrained init
                       -> parent_state.pt (read-only) + PARENT.json
  --r30_stage unit   : --r30_unit_id, --r30_source_warmup, --r30_objective H|L|N|B|PN|PV|PB, --r30_scale native|1|fit_static,
                       --r30_coef (fit_static only), --r30_mode fork|resume, --r30_parent + --r30_parent_sha + --r30_parent_id
                       (+ --r30_parent_fingerprint); fork reads the parent (opt.resume = parent), resume needs --resume <own state>
  --r30_milestone    : completed co-train epoch of the non-overwritable checkpoint (10; smoke only may use another value)
  --r30_require_gpu  : refuse to run unless torch.cuda.get_device_name(0) equals this (hardware block)
"""
import os, json, glob, hashlib, argparse, time
import torch
import r25_run
import r30_align as A30


def parse(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--r30_stage', required=True, choices=('warmup', 'unit'))
    p.add_argument('--r30_source_warmup', required=True, choices=('H', 'L'))
    p.add_argument('--r30_parent_id', required=True)
    p.add_argument('--r30_unit_id', default=None)
    p.add_argument('--r30_objective', default=None, choices=A30.OBJECTIVES)
    p.add_argument('--r30_scale', default=None, choices=('native', '1', 'fit_static'))
    p.add_argument('--r30_coef', default=None, type=float)
    p.add_argument('--r30_mode', default=None, choices=('fork', 'resume'))
    p.add_argument('--r30_parent', default=None)
    p.add_argument('--r30_parent_sha', default=None)
    p.add_argument('--r30_parent_fingerprint', default=None)
    p.add_argument('--r30_milestone', default=10, type=int)
    p.add_argument('--r30_require_gpu', default='')
    a, rest = p.parse_known_args(argv)
    opt = r25_run.parse(rest)
    assert opt.noisy_obj == 'sce' and opt.link == 'c0', 'R25 hook flags are not used by R30'
    assert opt.data_name == 'f30k_precomp'
    opt.arm, opt.lam = 'a1_iface', 0.0
    for k, v in vars(a).items():
        setattr(opt, k, v)
    opt.r30_tau = A30.TAU
    if a.r30_stage == 'warmup':
        assert a.r30_objective is None and a.r30_coef is None and a.r30_parent is None and a.r30_mode is None and not opt.resume
        opt.r29_arm = {'H': 'h_native', 'L': 'l_listwise_vcs'}[a.r30_source_warmup]   # the R29 warmup recipe of the source
    else:
        assert a.r30_unit_id and a.r30_objective and a.r30_scale and a.r30_mode and a.r30_parent and a.r30_parent_sha
        if a.r30_scale == '1' and a.r30_coef is None:
            opt.r30_coef = 1.0
        A30.check_term(opt.r30_objective, opt.r30_scale, opt.r30_coef)
        opt.r29_arm = None
        opt.r30_parent = os.path.abspath(a.r30_parent)
        if a.r30_mode == 'fork':
            assert not opt.resume, 'fork reads the parent; --resume is for mode resume'
            opt.resume = opt.r30_parent
        else:
            assert opt.resume and os.path.abspath(opt.resume) != opt.r30_parent, 'resume needs this branch state'
    return opt


def env_record(opt):
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                dtype='float32 (no AMP)', matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                code_sha256={os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob(os.path.join(here, '*.py')))})


def main():
    opt = parse()
    os.makedirs(opt.output_dir, exist_ok=True)
    r25_run.seed_all(opt.seed)
    env = env_record(opt)
    if opt.r30_require_gpu and env['gpu'] != opt.r30_require_gpu:
        raise SystemExit(f"R30: GPU {env['gpu']} != hardware block GPU {opt.r30_require_gpu}")
    json.dump({'opt': vars(opt), 'env': env}, open(os.path.join(opt.output_dir, f"RUN_{env['job']}.json"), 'w'), indent=1)
    import r30_co_train
    t0 = time.time()
    r30_co_train.main(opt)
    json.dump({'peak_mem_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30, 'peak_mem_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
               'wall_seconds': time.time() - t0, 'gpu': env['gpu'], 'job': env['job']},
              open(os.path.join(opt.output_dir, f"PEAK_{env['job']}.json"), 'w'), indent=1)


if __name__ == '__main__':
    main()
