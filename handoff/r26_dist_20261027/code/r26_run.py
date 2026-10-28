"""R26 entry point: r25_run arguments (pinned copy) plus --arm and --c_J. No TEST at the end."""
import os, sys, json, glob, hashlib, argparse
import torch
import r25_run
from r26_graph import ARMS


def parse(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--c_J', required=True, type=float)
    a, rest = p.parse_known_args(argv)
    assert a.c_J > 0
    opt = r25_run.parse(rest)
    assert opt.noisy_obj == 'sce' and opt.link == 'c0', 'R25 hook flags are not used by R26'
    opt.arm, opt.c_J = a.arm, a.c_J
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
    import time
    import r26_co_train
    t0 = time.time()
    r26_co_train.main(opt)
    json.dump({'peak_mem_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30, 'peak_mem_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
               'wall_seconds': time.time() - t0, 'gpu': env['gpu'], 'job': env['job']},
              open(os.path.join(opt.output_dir, f"PEAK_{env['job']}.json"), 'w'), indent=1)


if __name__ == '__main__':
    main()
