"""R25 entry point: run.py arguments of IN2R-code plus the R25 hook arguments. No TEST at the end."""
import os, sys, random, argparse, json
import numpy as np
import torch


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data")
    p.add_argument("--data_name", default="f30k_precomp")
    p.add_argument("--vocab_path", default="/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab")
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--num_epochs", default=40, type=int)
    p.add_argument("--lr_update", default=15, type=int)
    p.add_argument("--learning_rate", default=0.0005, type=float)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--log_step", default=500, type=int)
    p.add_argument("--grad_clip", default=2.0, type=float)
    p.add_argument("--margin", default=0.2, type=float)
    p.add_argument("--img_dim", default=2048, type=int)
    p.add_argument("--dropout_rate", default=0.3, type=float)
    p.add_argument("--word_dim", default=300, type=int)
    p.add_argument("--embed_size", default=1024, type=int)
    p.add_argument("--num_layers", default=1, type=int)
    p.add_argument("--bi_gru", action="store_false")
    p.add_argument("--no_imgnorm", action="store_true")
    p.add_argument("--no_txtnorm", action="store_true")
    p.add_argument("--noise_file", required=True)
    p.add_argument("--noise_ratio", default=0.4, type=float)
    p.add_argument("--no_co_training", action="store_true")
    p.add_argument("--warmup_epoch", default=5, type=int)  # R25 code-version config (run.py default 1)
    p.add_argument("--warmup_model_path", default="")
    p.add_argument("--p_threshold", default=0.5, type=float)
    p.add_argument("--soft_margin", default="exponential")
    p.add_argument("--seed", default=1001, type=int)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--queue_size", default=65536, type=int)
    p.add_argument("--k", default=5, type=int)  # R25 config (run.py default 1)
    p.add_argument("--no_i2t", dest="i2t", action="store_false")  # R25: both directions ON by default
    p.add_argument("--no_t2i", dest="t2i", action="store_false")
    # R25 hook
    p.add_argument("--noisy_obj", default="sce", choices=["sce", "vcs", "bce"])
    p.add_argument("--link", default="c0", choices=["c0", "c1"])
    p.add_argument("--c1_a", default=None, type=float)
    p.add_argument("--c1_b", default=None, type=float)
    p.add_argument("--c_D", default=1.0, type=float)
    p.add_argument("--resume", default="")
    p.add_argument("--prefix_epoch", default=0, type=int)
    p.add_argument("--stop_after_prefix", action="store_true")
    p.add_argument("--max_iters", default=0, type=int, help="smoke only")
    opt = p.parse_args(argv)
    if opt.noisy_obj != "sce":
        assert opt.c_D > 0
        if opt.link == "c1":
            assert opt.c1_a is not None and opt.c1_a > 0 and opt.c1_b is not None
    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
    return opt


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # native


def main():
    opt = parse()
    os.makedirs(opt.output_dir, exist_ok=True)
    seed_all(opt.seed)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
               matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
               cudnn_benchmark=torch.backends.cudnn.benchmark, dtype="float32 (no AMP)")
    import glob, hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    env['code_sha256'] = {os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob(os.path.join(here, '*.py')))}
    json.dump({'opt': vars(opt), 'env': env}, open(os.path.join(opt.output_dir, f"RUN_{env['job']}.json"), "w"), indent=1)
    import r25_co_train
    r25_co_train.main(opt)


if __name__ == "__main__":
    main()
