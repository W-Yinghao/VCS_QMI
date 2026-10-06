"""P138 — v7 V7-PAIR stage 2: K = 16 sampled image shifts x all view pairs, VCS (A-P3) and matched JS-AP3, CIFAR-10 and CIFAR-100, seed 0.
Each config = the method's / dataset's seed-0 parent with ONLY: pairing.pair_scope -> 'sampled_image_shifts_all_view_pairs', pairing.k 8 -> 16
(= number of distinct image shifts per step), pairing.all_view_chunk removed (unused: only selected pairs are scored), run.stage.
(a, kappa) = (2, 0.5), lr 1e-3, full gradients, same RNG roles (shifts come from the dedicated pair generator, unused by the parents).
    python configs/make_p138_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P138_v7_pair_k16"
K = 16
PARENTS = {("vcs", "c10"): "cifar10_hpY_AP3_views4_800ep_seed0.yaml", ("vcs", "c100"): "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml",
           ("js", "c10"): "cifar10_hpJS_AP3_views4_800ep_seed0.yaml", ("js", "c100"): "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for (m, ds), parent in PARENTS.items():
        c = yaml.safe_load(open(ROOT / "configs" / parent))
        assert c["pairing"]["pair_scope"] == "all_view_tokens" and c["pairing"]["negative_detach"] is False
        c["pairing"]["pair_scope"] = "sampled_image_shifts_all_view_pairs"; c["pairing"]["k"] = K; c["pairing"].pop("all_view_chunk", None)
        c["run"]["stage"] = STAGE
        tag = f"{m}_{ds}"; run_id = f"P138_PAIR_K16_{tag}_seed0"
        name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpPK16_{tag}_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "method": m, "k": K}; print(run_id, sha[:16])
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p138_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write:
            (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P138_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p138_lines.txt").write_text("# P138 V7-PAIR stage 2 (K=16 sampled image shifts x all view pairs) — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
