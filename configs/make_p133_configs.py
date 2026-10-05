"""P133 — MoCo-consistent momentum-key pairs (the follow-up proposed in the P100 report §3; owner 2026-10-04 "全部提交").
CIFAR-10, 4 views, B 256, 800 epochs, seed 0 (owner rule: one full seed first; seeds 1-2 only by the pre-stated rule).
  VCS   : the A-P3 seed-0 config (fixed f = 2s - 1, original J) with pair_scope / all_view_chunk dropped and the P133 pairing block;
  SimCLR: the P41 tuned 4-view SimCLR seed-0 config (tau 0.2) with the same P133 pairing block.
Variants (pairing block = negative_source 'queue', queue true, momentum_encoder true, moco_consistent true, plus):
  q4096m99  : queue_size 4096, momentum_m 0.99            — FAILED the constant-online-map gate (job 1021175); kept for the record, never launched
  q1024m999 : queue_size 1024, momentum_m 0.999           — variant (b): lower-staleness queue (~4 steps of view-0 keys)
  noqueue   : momentum_m 0.99, moco_use_queue false        — variant (a): negatives = current-batch momentum keys only (queue_size unused)
run.stage P133_moco_consistent.

    python configs/make_p133_configs.py [--write]   # configs/cifar10_hpMC_*.yaml, configs/P133_SHA256.json, slurm/p133_lines_<variant>.txt
"""
import hashlib, json, sys
from pathlib import Path
import yaml

R = Path(__file__).resolve().parents[1]
STAGE = "P133_moco_consistent"
BASES = {"vcs": "configs/cifar10_hpY_AP3_views4_800ep_seed{s}.yaml", "simclr": "configs/cifar10_hpN_simclr_views4_800ep_seed{s}.yaml"}
SEEDS = (0,)
COMMON = {"negative_source": "queue", "queue": True, "momentum_encoder": True, "moco_consistent": True}
VARIANTS = {"q4096m99": {"queue_size": 4096, "momentum_m": 0.99},
            "q1024m999": {"queue_size": 1024, "momentum_m": 0.999},
            "noqueue": {"queue_size": 1024, "momentum_m": 0.99, "moco_use_queue": False}}
NAME = {"q4096m99": "", "q1024m999": "_q1024m999", "noqueue": "_noqueue"}   # the first (failed) variant keeps its original file / run names

out, lines = {}, {v: [] for v in VARIANTS}
for v, extra in VARIANTS.items():
    block = {**COMMON, **extra}
    for m, tpl in BASES.items():
        for s in SEEDS:
            base = tpl.format(s=s)
            c = yaml.safe_load(open(R / base))
            assert c["train"]["epochs"] == 800 and c["views"]["count"] == 4 and c["run"]["seed"] == s and c["train"]["batch_size_images"] == 256
            if m == "vcs":
                cr = c["model"]["critic"]
                assert c["run"]["method"] == "vcs_qmi" and cr["affine_mode"] == "fixed" and float(cr["cosine_scale_init"]) == 2.0 and float(cr["cosine_bias_init"]) == -1.0
                assert c["objective"]["loss"] == "negative_J" and c["pairing"].get("pair_scope") == "all_view_tokens"
                for k in ("pair_scope", "all_view_chunk"):
                    c["pairing"].pop(k, None)
            else:
                assert c["run"]["method"] == "simclr_matched" and float(c["objective"]["simclr_temperature"]) == 0.2
                assert "pair_scope" not in c["pairing"]
            c["run"]["stage"] = STAGE
            c["pairing"].update(block)
            name = f"cifar10_hpMC_{m}{NAME[v]}_views4_800ep_seed{s}.yaml"
            txt = yaml.safe_dump(c, sort_keys=False)
            run_id = f"P133_{m}_moco{NAME[v]}_views4_800ep_seed{s}"
            out[name] = {"sha256": hashlib.sha256(txt.encode()).hexdigest(), "variant": v, "base": base, "run_id": run_id,
                         "changes": {**{f"pairing.{k}": val for k, val in block.items()}, "run.stage": STAGE,
                                     **({"pairing.pair_scope": "removed", "pairing.all_view_chunk": "removed"} if m == "vcs" else {})}}
            lines[v].append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p133_{m}{NAME[v]}_s{s} "
                            f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if "--write" in sys.argv:
                (R / "configs" / name).write_text(txt)
            print(f"{v:10s} {run_id:44s} configs/{name}  {out[name]['sha256'][:16]}")
if "--write" in sys.argv:
    (R / "configs/P133_SHA256.json").write_text(json.dumps({"stage": STAGE, "configs": out}, indent=1))
    GATE = {"q4096m99": "FAILED gate 1021175 (max J +0.320)", "q1024m999": "FAILED gate 1021714 (max J +0.025 > 2 SE 0.0025)",
            "noqueue": "PASSED gate 1021714 (max J -0.0003)"}
    for v in ("q1024m999", "noqueue"):
        body = "\n".join(lines[v]) if v == "noqueue" else "\n".join("# " + l for l in lines[v])
        (R / f"slurm/p133_lines_{v}.txt").write_text(f"# P133 variant {v}: {GATE[v]}; seed 0 — normal QOS (never runfill); RTX6000PRO / H100 / L40S; "
                                                     "node51, node52, node60 excluded\n" + body + "\n")
    (R / "slurm/p133_lines.txt").write_text("# P133 = momentum-encoder keys WITHOUT a queue (variant noqueue; gate job 1021714 passed).  q4096m99 (gate 1021175) and "
                                            "q1024m999 (gate 1021714) FAILED - never launch them.\n" + "\n".join(lines["noqueue"]) + "\n")
    print("wrote configs, P133_SHA256.json, slurm/p133_lines{,_q1024m999,_noqueue}.txt")
