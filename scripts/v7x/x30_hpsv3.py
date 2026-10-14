"""V7X-30 B5: HPSv3 (github.com/MizzenAI/HPSv3, pip hpsv3 1.0.0; weights MizzenAI/HPSv3 local copy) on the GenAI-Rank pool: HPSv3RewardInferencer.reward,
score = mu (the first output, as the release example).  Config: the package's HPSv3_7B.yaml with two recorded overrides — model_name_or_path = the local
Qwen/Qwen2-VL-7B-Instruct copy (offline) and disable_flash_attn2 = true (flash-attn not installed).  Preference score, not a Bernoulli probability.
    rank --inputs I --out F [--shard k/n] [--limit N]"""
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

MODELS = Path("/projects/EEG-foundation-model/yinghao/models"); RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank")


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--inputs", required=True); r.add_argument("--out", required=True); r.add_argument("--shard", default="0/1"); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--image-root", default=None); a = ap.parse_args(argv)
    import torch
    import yaml
    import hpsv3
    from hpsv3 import HPSv3RewardInferencer
    cfg = yaml.safe_load((Path(hpsv3.__file__).parent / "config" / "HPSv3_7B.yaml").read_text())
    cfg.update({"model_name_or_path": str(MODELS / "Qwen2-VL-7B-Instruct"), "disable_flash_attn2": True})
    local = Path(os.environ.get("V7X_LOCAL", "/tmp")); local.mkdir(parents=True, exist_ok=True); cp = local / "HPSv3_7B_v7x.yaml"; cp.write_text(yaml.safe_dump(cfg))
    k, n = (int(x) for x in a.shard.split("/")); rows = [json.loads(l) for i, l in enumerate(open(a.inputs)) if l.strip() and i % n == k][: a.limit or None]
    out = Path(a.out); done = {json.loads(l)["image_id"] for l in open(out) if l.strip() and "image_id" in l} if out.exists() else set(); todo = [r for r in rows if r["image_id"] not in done]
    t0 = time.time(); inf = HPSv3RewardInferencer(config_path=str(cp), checkpoint_path=str(MODELS / "HPSv3" / "HPSv3.safetensors"), device="cuda"); zs = {}
    with open(out, "a") as fh:
        if not done:
            fh.write(json.dumps({"header": {"method": "B5 HPSv3 (mu)", "config_overrides": {"model_name_or_path": "local Qwen2-VL-7B-Instruct", "disable_flash_attn2": True},
                                            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "load_s": time.time() - t0}}) + "\n")
        for s in range(0, len(todo), 8):
            chunk = todo[s:s + 8]; paths = []
            for c in chunk:
                if a.image_root:
                    paths.append(str(Path(a.image_root) / c["image_id"])); continue
                g = c["image_id"].split("/")[0]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(RANK / f"{g}.zip")); p = local / "img" / c["image_id"]
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(c["image_id"]))
                paths.append(str(p))
            with torch.no_grad():
                mu = inf.reward(paths, [c["prompt"] for c in chunk])[:, 0].float().cpu().tolist()
            for c, v in zip(chunk, mu):
                fh.write(json.dumps({"prompt_id": c["prompt_id"], "generator_id": c["generator_id"], "image_id": c["image_id"], "candidate_index": c["candidate_index"], "score": float(v)}) + "\n")
            if (s // 8) % 100 == 0:
                fh.flush(); print(f"[hpsv3] {s + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
