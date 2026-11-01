"""R29 entry point: pinned r25_run arguments plus --r29_arm. The noisy branch is always the R27 A1 interface (arm a1_iface,
lam 0); the run starts from the common untrained seed-determined initialization unless --resume names this unit's own
state_last.pt (the unit runner enforces same-unit resume). No TEST at the end."""
import os, json, glob, hashlib, argparse, time
import torch
import r25_run
import r29_align as AL


def parse(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--r29_arm', required=True, choices=AL.ARMS)
    a, rest = p.parse_known_args(argv)
    opt = r25_run.parse(rest)
    assert opt.noisy_obj == 'sce' and opt.link == 'c0', 'R25 hook flags are not used by R29'
    assert opt.data_name == 'f30k_precomp'
    opt.arm, opt.lam, opt.r29_arm = 'a1_iface', 0.0, a.r29_arm
    opt.r29_tau = AL.TAU
    return opt


def main():
    opt = parse()
    os.makedirs(opt.output_dir, exist_ok=True)
    r25_run.seed_all(opt.seed)
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
               dtype='float32 (no AMP)',
               code_sha256={os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob(os.path.join(here, '*.py')))})
    json.dump({'opt': vars(opt), 'env': env}, open(os.path.join(opt.output_dir, f"RUN_{env['job']}.json"), 'w'), indent=1)
    import r29_co_train
    t0 = time.time()
    r29_co_train.main(opt)
    json.dump({'peak_mem_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30, 'peak_mem_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
               'wall_seconds': time.time() - t0, 'gpu': env['gpu'], 'job': env['job']},
              open(os.path.join(opt.output_dir, f"PEAK_{env['job']}.json"), 'w'), indent=1)


if __name__ == '__main__':
    main()
