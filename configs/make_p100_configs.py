"""P100 — momentum-encoder key queue (MoCo-style) for the VCS a5 recipe and the tuned SimCLR 4-view recipe, 8x (4 views, B 256, 800 epochs),
seeds 0-2.  Each config = its frozen 8x base with only the pairing block changed: negative_source 'queue', queue true, queue_size 4096,
momentum_encoder true, momentum_m 0.99; stage P100_momentum_queue.

    python configs/make_p100_configs.py [--write]      # writes configs/cifar10_hpQ_*.yaml, configs/P100_SHA256.json, slurm/p100_units.txt, slurm/p100_lines.txt
"""
import hashlib, json, sys
from pathlib import Path
import yaml

R = Path(__file__).resolve().parents[1]
BASES = {"vcs": "configs/cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml", "simclr": "configs/cifar10_hpN_simclr_views4_800ep_seed{s}.yaml"}
out, units = {}, []
for m, tpl in BASES.items():
    for s in (0, 1, 2):
        base = tpl.format(s=s)
        c = yaml.safe_load(open(R / base))
        assert c["train"]["epochs"] == 800 and c["views"]["count"] == 4 and c["run"]["seed"] == s
        c["run"]["stage"] = "P100_momentum_queue"
        c["pairing"].update({"negative_source": "queue", "queue": True, "queue_size": 4096, "momentum_encoder": True, "momentum_m": 0.99})
        name = f"cifar10_hpQ_{m}_mq_views4_800ep_seed{s}.yaml"
        txt = yaml.safe_dump(c, sort_keys=False)
        run_id = f"P100_{m}_mq_views4_800ep_seed{s}"
        out[name] = {"sha256": hashlib.sha256(txt.encode()).hexdigest(), "base": base, "run_id": run_id,
                     "changes": {"pairing.negative_source": "queue", "pairing.queue": True, "pairing.queue_size": 4096,
                                 "pairing.momentum_encoder": True, "pairing.momentum_m": 0.99, "run.stage": "P100_momentum_queue"}}
        units.append((run_id, f"configs/{name}", "epoch_800.pt"))
        if "--write" in sys.argv:
            (R / "configs" / name).write_text(txt)
        print(f"{run_id:34s} configs/{name}  {out[name]['sha256'][:16]}  <- {base}")
if "--write" in sys.argv:
    (R / "configs/P100_SHA256.json").write_text(json.dumps(out, indent=1))
    (R / "slurm/p100_units.txt").write_text("".join(f"{a} {b} {c}\n" for a, b, c in units))
    (R / "slurm/p100_lines.txt").write_text("# P100 momentum queue, 8x, fast GPUs, normal QOS\n" + "".join(
        f"sbatch --parsable --partition=RTX6000PRO,H100 --exclude=node60 --job-name={a[5:]} --export=ALL,CFG={b},RUN_ID={a},FINAL_CKPT={c} slurm/run_unit.sbatch\n"
        for a, b, c in units))
    print("wrote configs, P100_SHA256.json, slurm/p100_units.txt, slurm/p100_lines.txt")
