"""S line (package v2 Server Spec §6–7; Plan §6) — configs and unit files for
  P87  S-CS: VCS-N (existing) vs S-Kernel (rff_tanh critic, same J / pairing / detach) vs CS-K-native (classical kernel CS-QMI, native)
  P89  S4:   method x augmentation strength (SimCLR strong-aug seeds 0-2, VCS strong-aug seeds 1-2; standard cells and VCS strong seed 0 reused)
  P91  CIFAR-100 independent confirmation (frozen recipes, 2x then 8x, 3 seeds)
Every YAML is derived from an existing frozen config (the exact file is recorded), with the listed fields changed and nothing else.

    python configs/make_s_line_configs.py [--p87-dev] [--p89] [--p91] [--gate] [--p87-8x M BW KBW] [--write]

Dry run prints the units; --write writes YAMLs, refreshes configs/S_LINE_SHA256.json (merged per stage) and the slurm/s_units_*.txt files.
--p87-8x is run only after the P87 dev selection (M = rff_features, BW = rff bandwidth multiple, KBW = kernel-CS bandwidth multiple).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
SLURM = ROOT / "slurm"
SHA_JSON = CFG / "S_LINE_SHA256.json"
CIFAR100 = {"name": "cifar100", "root": "/home/infres/yinwang/CS_QMI/data/cifar100", "val_per_class": 50,
            "manifest": "/home/infres/yinwang/CS_QMI/manifests/cifar100_dev45k_val5k.json"}
BASE = {
    "vcs_1x": "cifar10_hpM_a5_views4_b128_100ep_vcs_seed0.yaml",          # 1x: 4 views, B 128, 100 ep (83.19 +- 0.40, P39 seeds 0-2)
    "vcs_2x": "cifar10_hpK_a5_views4_vcs_seed{s}.yaml",                    # 2x: 4 views, B 256, 200 ep
    "vcs_8x": "cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml",              # 8x: 4 views, B 256, 800 ep (87.01 +- 0.53)
    "vcs_8x_strong": "cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml",  # P43 strong aug (87.78, seed 0)
    "simclr_2x": "cifar10_hpN_simclr_views4_seed{s}.yaml", "simclr_8x": "cifar10_hpN_simclr_views4_800ep_seed{s}.yaml",
    "vicreg_2x": "cifar10_hpN_vicreg_views4_seed{s}.yaml", "vicreg_8x": "cifar10_hpN_vicreg_views4_800ep_seed{s}.yaml",
}


def load(name: str) -> dict:
    return yaml.safe_load((CFG / name).read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{x:g}"


def skernel(c: dict, m: int, bw: float) -> dict:
    """S-Kernel: only the critic class changes (fixed RFF of [z1; z2] + trainable read-out, tanh); J, K, detach, everything else as VCS-N."""
    crit = c["model"]["critic"]
    crit["input"] = "rff_tanh"; crit["rff_features"] = int(m); crit["rff_bandwidth_multiple"] = float(bw)
    return {"model.critic.input": "rff_tanh", "model.critic.rff_features": int(m), "model.critic.rff_bandwidth_multiple": float(bw)}


def kcs(c: dict, bw: float) -> dict:
    """CS-K-native: classical kernel CS-QMI on z_l2, native autodiff, no critic, no negative branch; multi-view averaging kept."""
    c["run"]["method"] = "cs_kernel_native"
    c["model"]["critic"]["enabled"] = False
    c["objective"]["target"] = "classical_kernel_CS_QMI"; c["objective"]["loss"] = "negative_kernel_cs"
    c["objective"]["kernel_cs_bandwidth_multiple"] = float(bw); c["objective"]["kernel_cs_chunk"] = 0
    c["pairing"]["k"] = 1; c["pairing"]["negative_detach"] = False
    c["evaluation"]["critic_validation"]["enabled"] = False
    return {"run.method": "cs_kernel_native", "model.critic.enabled": False, "objective.target": "classical_kernel_CS_QMI",
            "objective.loss": "negative_kernel_cs", "objective.kernel_cs_bandwidth_multiple": float(bw), "objective.kernel_cs_chunk": 0,
            "pairing.k": 1, "pairing.negative_detach": False, "evaluation.critic_validation.enabled": False}


def cifar100(c: dict) -> dict:
    d = c["data"]
    d["name"] = CIFAR100["name"]; d["root"] = CIFAR100["root"]; d["val_per_class"] = CIFAR100["val_per_class"]; d["manifest"] = CIFAR100["manifest"]
    return {"data.name": "cifar100", "data.root": CIFAR100["root"], "data.val_per_class": 50, "data.manifest": CIFAR100["manifest"]}


def unit(stage: str, base: str, name: str, run_id: str, cfg: dict, changes: dict, final: str, note: str = "") -> dict:
    return {"stage": stage, "base_config": base, "config": f"configs/{name}", "run_id": run_id, "final_ckpt": final, "changes": changes,
            "note": note, "_cfg": cfg}


def gen_p87_dev() -> list[dict]:
    st = "P87_s_cs_dev"; out = []
    for m in (256, 1024, 4096):
        for bw in (0.5, 1.0, 2.0):
            c = load(BASE["vcs_1x"]); c["run"]["stage"] = st; ch = skernel(c, m, bw)
            out.append(unit(st, BASE["vcs_1x"], f"cifar10_hpS_skernel_m{m}_bw{fmt(bw)}_1x_seed0.yaml", f"P87_skernel_m{m}_bw{fmt(bw)}_1x_seed0", c, ch, "epoch_100.pt",
                            "S-Kernel dev grid (1x); selection rule in the P87 prereg"))
    for bw in (0.5, 1.0, 2.0):
        c = load(BASE["vcs_1x"]); c["run"]["stage"] = st; ch = kcs(c, bw)
        out.append(unit(st, BASE["vcs_1x"], f"cifar10_hpS_kcs_bw{fmt(bw)}_1x_seed0.yaml", f"P87_kcs_bw{fmt(bw)}_1x_seed0", c, ch, "epoch_100.pt",
                        "CS-K-native dev grid (1x)"))
    return out


def gen_p87_8x(m: int, bw: float, kbw: float) -> list[dict]:
    st = "P87_s_cs"; out = []
    for s in (0, 1, 2):
        b = BASE["vcs_8x"].format(s=s)
        c = load(b); c["run"]["stage"] = st; ch = skernel(c, m, bw)
        out.append(unit(st, b, f"cifar10_hpS_skernel_m{m}_bw{fmt(bw)}_views4_800ep_seed{s}.yaml", f"P87_skernel_m{m}_bw{fmt(bw)}_views4_800ep_seed{s}", c, ch,
                        "epoch_800.pt", "S-CS table A: S-Kernel at 8x (selected on the 1x dev grid)"))
    for s in (0, 1, 2):
        b = BASE["vcs_8x"].format(s=s)
        c = load(b); c["run"]["stage"] = st; ch = kcs(c, kbw)
        out.append(unit(st, b, f"cifar10_hpS_kcs_bw{fmt(kbw)}_views4_800ep_seed{s}.yaml", f"P87_kcs_bw{fmt(kbw)}_views4_800ep_seed{s}", c, ch,
                        "epoch_800.pt", "S-CS table B: CS-K-native at 8x (selected on the 1x dev grid)"))
    return out


def aug_diff(std: dict, strong: dict) -> dict:
    """Field-level difference of two views blocks (the strong block must differ from the standard one only in crop scale and jitter)."""
    def flat(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, pre + k + ".")
            else:
                yield pre + k, v
    a, b = dict(flat(std)), dict(flat(strong))
    return {k: {"standard": a.get(k), "strong": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def gen_p89() -> list[dict]:
    st = "P89_s4_aug_interaction"; out = []
    strong = load(BASE["vcs_8x_strong"])["views"]
    std = load(BASE["vcs_8x"].format(s=0))["views"]
    diff = aug_diff(std, strong)
    assert set(diff) == {"random_resized_crop.scale", "color_jitter.brightness", "color_jitter.contrast", "color_jitter.saturation", "color_jitter.hue"}, diff
    for s in (0, 1, 2):
        b = BASE["simclr_8x"].format(s=s)
        c = load(b); c["run"]["stage"] = st; c["views"] = copy.deepcopy(strong)
        out.append(unit(st, b, f"cifar10_hpS_simclr_views4_800ep_augstrong_seed{s}.yaml", f"P89_simclr_views4_800ep_augstrong_seed{s}", c,
                        {"views": "hpO strong block verbatim", "views_diff_vs_standard": diff}, "epoch_800.pt",
                        "S4: tuned SimCLR 4v/800ep with the P43 strong-augmentation block (whole block copied)"))
    for s in (1, 2):
        b = BASE["vcs_8x_strong"]
        c = load(b); c["run"]["stage"] = st; c["run"]["seed"] = s
        out.append(unit(st, b, f"cifar10_hpS_a5_views4_800ep_augstrong_vcs_seed{s}.yaml", f"P89_vcs_a5_views4_800ep_augstrong_seed{s}", c,
                        {"run.seed": s, "run.stage": st}, "epoch_800.pt", "S4: VCS strong aug seeds 1-2 (seed 0 = P43_vcs_a5_views4_800ep_augstrong_seed0 reused)"))
    return out


def gen_p91() -> list[dict]:
    st = "P91_cifar100"; out = []
    for meth, key in (("vcs_a5", "vcs"), ("simclr", "simclr"), ("vicreg", "vicreg")):
        for budget, ep, final in (("2x", "200ep", "epoch_200.pt"), ("8x", "800ep", "epoch_800.pt")):
            for s in (0, 1, 2):
                b = BASE[f"{key}_{budget}"].format(s=s)
                c = load(b); c["run"]["stage"] = st; ch = cifar100(c)
                out.append(unit(st, b, f"cifar100_hpS_{meth}_views4_{ep}_seed{s}.yaml", f"P91_c100_{meth}_views4_{ep}_seed{s}", c, ch, final,
                                f"CIFAR-100 {budget}: frozen CIFAR-10 recipe unchanged except the dataset block"))
    return out


def gen_gate() -> list[dict]:
    """Two-view smoke configs for the CPU gate only (never a unit)."""
    st = "S_LINE_GATE"; out = []
    c = load(BASE["vcs_8x"].format(s=0)); c["run"]["stage"] = st; c["views"]["count"] = 2; c["train"]["batch_size_images"] = 64; ch = skernel(c, 256, 1.0)
    out.append(unit(st, BASE["vcs_8x"].format(s=0), "cifar10_hpS_gate_skernel_2v.yaml", "gate_skernel_2v", c, {**ch, "views.count": 2, "train.batch_size_images": 64}, "last.pt", "gate only"))
    c = load(BASE["vcs_8x"].format(s=0)); c["run"]["stage"] = st; c["views"]["count"] = 2; c["train"]["batch_size_images"] = 64; ch = kcs(c, 1.0)
    out.append(unit(st, BASE["vcs_8x"].format(s=0), "cifar10_hpS_gate_kcs_2v.yaml", "gate_kcs_2v", c, {**ch, "views.count": 2, "train.batch_size_images": 64}, "last.pt", "gate only"))
    return out


UNIT_FILES = {"P87_s_cs_dev": "s_units_P87_dev.txt", "P87_s_cs": "s_units_P87_8x.txt", "P89_s4_aug_interaction": "s_units_P89.txt",
              "P91_cifar100:2x": "s_units_P91_2x.txt", "P91_cifar100:8x": "s_units_P91_8x.txt"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p87-dev", action="store_true"); ap.add_argument("--p89", action="store_true"); ap.add_argument("--p91", action="store_true")
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--p87-8x", nargs=3, metavar=("M", "BW", "KBW"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    units: list[dict] = []
    if a.p87_dev:
        units += gen_p87_dev()
    if a.p87_8x:
        units += gen_p87_8x(int(a.p87_8x[0]), float(a.p87_8x[1]), float(a.p87_8x[2]))
    if a.p89:
        units += gen_p89()
    if a.p91:
        units += gen_p91()
    if a.gate:
        units += gen_gate()
    if not units:
        ap.error("nothing selected")
    for u in units:
        print(f"{u['stage']:<24} {u['run_id']:<52} {u['config']:<62} {u['final_ckpt']}  <- {u['base_config']}")
    print(f"{len(units)} units")
    if not a.write:
        print("dry run (pass --write)"); return 0
    reg = json.loads(SHA_JSON.read_text()) if SHA_JSON.is_file() else {"stages": {}}
    by_stage: dict[str, list[dict]] = {}
    for u in units:
        p = CFG / Path(u["config"]).name
        p.write_text(yaml.safe_dump(u["_cfg"], sort_keys=False, allow_unicode=True), encoding="utf-8")
        u["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        by_stage.setdefault(u["stage"], []).append({k: v for k, v in u.items() if k != "_cfg"})
    for st, us in by_stage.items():
        reg["stages"][st] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": us, "files": {Path(x["config"]).name: x["sha256"] for x in us}}
    reg["cifar100"] = CIFAR100
    SHA_JSON.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    for key, fname in UNIT_FILES.items():
        st, _, sub = key.partition(":")
        if st not in by_stage:
            continue
        us = [u for u in by_stage[st] if not sub or f"_{ {'2x': '200ep', '8x': '800ep'}[sub] }_" in u["run_id"]]
        (SLURM / fname).write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in us), encoding="utf-8")
        print(f"wrote {SLURM / fname} ({len(us)} units)")
    print(f"wrote {SHA_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
