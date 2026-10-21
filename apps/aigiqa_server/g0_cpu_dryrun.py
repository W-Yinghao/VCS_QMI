"""CPU wiring dry run of g0_preflight.py on the REAL ImageReward/BLIP architecture with RANDOM weights and SYNTHETIC images.

Not a G0 result and not evidence about AGIQA-3K. It shows, before any download, that this env runs ImageReward's BLIP
(ViT-L/16 + BERT with cross-attention) under the chosen torch build, that the package's ImageRewardCMIM adapter wires up
(_freeze_half layer names, 1024/768/1024 features), and that every G0 code path executes. The med_config fixture holds the
standard BLIP values (the staged med_config.json is verified separately by its pinned git blob). All fixtures live in
node-local tmp and are deleted; only DRYRUN JSON files are copied to --evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_data  # noqa: E402

BLIP_MED_CONFIG = {"architectures": ["BertModel"], "attention_probs_dropout_prob": 0.1, "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
                   "hidden_size": 768, "initializer_range": 0.02, "intermediate_size": 3072, "layer_norm_eps": 1e-12,
                   "max_position_embeddings": 512, "model_type": "bert", "num_attention_heads": 12, "num_hidden_layers": 12,
                   "pad_token_id": 0, "type_vocab_size": 2, "vocab_size": 30524, "encoder_width": 768, "add_cross_attention": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", required=True); ap.add_argument("--bert-local", required=True); ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="aig_g0dry_"))
    evidence = Path(a.evidence); evidence.mkdir(parents=True, exist_ok=True)
    try:
        rng = np.random.default_rng(11)
        img = tmp / "images"; img.mkdir()
        rows = []
        for i in range(30):
            for j in range(3):
                name = f"dry{j}_{i:03d}.jpg"
                Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(img / name, quality=90)
                rows.append([name, f"synthetic prompt number {i} with token{i * 7 % 13}", "a", "b", "s",
                             f"{1 + j + .02 * i:.4f}", "0.5", f"{1.2 + .6 * j + .03 * i:.4f}", "0.5"])
        src = tmp / "full.csv"
        with open(src, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "prompt", "adj1", "adj2", "style", "mos_quality", "std_quality", "mos_align", "std_align"])
            w.writerows(rows)
        prepared = tmp / "prepared"
        subprocess.run([sys.executable, "-m", "aigiqa.cli", "prepare", "--dataset", "AGIQA-3k", "--source-csv", str(src),
                        "--image-root", str(img), "--output", str(prepared)], cwd=a.pkg, check=True, capture_output=True)
        med = tmp / "med_config_SYNTHETIC_FIXTURE.json"
        med.write_text(json.dumps(BLIP_MED_CONFIG, indent=2) + "\n")
        hf_home = tmp / "hf_home"
        c = stage_data.Checks()
        stage_data.tokenizer_cache(c, hf_home, Path(a.bert_local), dry=False)
        os.environ["HF_HOME"] = str(hf_home)
        from ImageReward.ImageReward import ImageReward as RewardModel
        torch.manual_seed(0)
        ckpt = tmp / "ImageReward_RANDOM_INIT_FIXTURE.pt"
        torch.save(RewardModel(med_config=str(med), device="cpu").state_dict(), ckpt)
        base = json.loads(Path(a.pkg, "aigiqa/configs/base_alignment.json").read_text())
        base["model"].update(checkpoint=str(ckpt), checkpoint_sha256=stage_data.sha256(ckpt), med_config=str(med),
                             med_config_sha256=stage_data.sha256(med))
        base.update(data_dir=str(prepared), output_dir=str(tmp / "unused_runs"), device="cpu")
        (tmp / "base.json").write_text(json.dumps(base, indent=2))
        p = subprocess.run([sys.executable, str(HERE / "g0_preflight.py"), "--base-config", str(tmp / "base.json"), "--out", str(tmp / "g0"),
                            "--synthetic-dry-run"], cwd=a.pkg, env={**os.environ, "HF_HOME": str(hf_home)}, capture_output=True, text=True)
        (evidence / "g0_dryrun.stdout.txt").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
        res = tmp / "g0" / "G0_RESULT.json"
        if res.exists():
            shutil.copy(res, evidence / "G0_SYNTHETIC_DRYRUN_RESULT.json")
        print(json.dumps({"rc": p.returncode, "tokenizer_checks_ok": c.ok, "result": str(evidence / "G0_SYNTHETIC_DRYRUN_RESULT.json")}))
        sys.exit(p.returncode)
    finally:
        for d in tmp.rglob("sealed"):
            os.chmod(d, 0o700)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
