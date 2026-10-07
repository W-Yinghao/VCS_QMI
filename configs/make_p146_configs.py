"""P146 — v7 §7.1 direct-CS route-transfer controls on CIFAR-100: the P87 / P88 S-Kernel (RFF critic on the same J, m 4096, bandwidth multiple
0.5) and CS-K-native (classical kernel CS-QMI objective, bandwidth multiple 0.5) designs, seed 0, 4 views, 800 epochs.  Each config = the CIFAR-100
recipe-VCS base (P91, cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml) + exactly the field changes that turn the CIFAR-10 recipe base
(cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml) into the P87 8x config of that route (verified diff; FIT-side median-heuristic bandwidth calibration at
step 0 unchanged; the multiples are the CIFAR-10 dev-grid selections, transferred, not re-screened — disclosed).
    python configs/make_p146_configs.py [--write]
"""
import argparse, copy, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P146_v7_direct_cs_c100"
C10_BASE = "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"
C100_BASE = "cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml"
ROUTES = {"skernel": "cifar10_hpS_skernel_m4096_bw0.5_views4_800ep_seed0.yaml", "kcs": "cifar10_hpS_kcs_bw0.5_views4_800ep_seed0.yaml"}


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".")) if isinstance(v, dict) else out.__setitem__(p + k, v)
    return out


def setp(d, key, v):
    ks = key.split("."); cur = d
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = v


def build(route: str) -> tuple[dict, dict]:
    b10 = flat(yaml.safe_load(open(ROOT / "configs" / C10_BASE))); r10 = flat(yaml.safe_load(open(ROOT / "configs" / ROUTES[route])))
    assert set(b10) <= set(r10), "route configs only add or change fields"
    diff = {k: r10[k] for k in r10 if b10.get(k, object()) != r10[k] and k != "run.stage"}
    c = copy.deepcopy(yaml.safe_load(open(ROOT / "configs" / C100_BASE)))
    for k, v in diff.items():
        setp(c, k, v)
    c["run"]["stage"] = STAGE
    return c, diff


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for route in ROUTES:
        c, diff = build(route)
        run_id = f"P146_{route}_c100_views4_800ep_seed0"; name = f"cifar100_hpDCS_{route}_views4_800ep_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "c100_base": C100_BASE, "c10_route": ROUTES[route], "applied_diff": diff}
        print(run_id, sha[:16], diff)
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p146_{route} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P146_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1, default=str))
        (ROOT / "slurm" / "p146_lines.txt").write_text("# P146 v7 §7.1 direct-CS controls, CIFAR-100 — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
