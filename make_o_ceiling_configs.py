"""Wave O (stage P43_vcs_ceiling): owner 2026-09-26 — ignore compute budgets and step matching; find how far the method goes.
Long / wide runs of the final recipe (cosine critic, K = 8, negative detach, a0 = 5), single seed, chained over the 23 h walltime."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
STAGE = "P43_vcs_ceiling"
def sched(c, ep, cks, kes):
    c["train"]["epochs"] = ep; c["logging"]["checkpoint_epochs"] = cks; c["evaluation"]["knn_epochs"] = kes; c["evaluation"]["linear_epochs_of_pretrain"] = [ep]
V4_800 = ROOT / "configs" / "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"
V8_100 = ROOT / "configs" / "cifar10_hpL_a5_views8_100ep_vcs_seed0.yaml"
units, files = [], {}
# 1) 4 views, 1600 epochs
c = yaml.safe_load(V4_800.read_text()); assert c["views"]["count"] == 4
sched(c, 1600, [200, 400, 800, 1200, 1600], [0, 50, 100, 200, 400, 800, 1200, 1600])
specs = [("a5_views4_1600ep", c, "epoch_1600.pt", 2)]
# 2) 8 views, 800 epochs
c = yaml.safe_load(V8_100.read_text()); assert c["views"]["count"] == 8 and c["train"]["batch_size_images"] == 256
sched(c, 800, [100, 200, 400, 600, 800], [0, 20, 50, 100, 200, 400, 600, 800])
specs.append(("a5_views8_800ep", c, "epoch_800.pt", 2))
# 3) 4 views, B = 128, 800 epochs
c = yaml.safe_load(V4_800.read_text()); c["train"]["batch_size_images"] = 128
specs.append(("a5_views4_b128_800ep", c, "epoch_800.pt", 2))
for label, cc, final, links in specs:
    cc["run"]["stage"] = STAGE
    p = ROOT / "configs" / f"cifar10_hpO_{label}_vcs_seed0.yaml"; p.write_text(yaml.safe_dump(cc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "views": cc["views"]["count"], "epochs": cc["train"]["epochs"], "batch": cc["train"]["batch_size_images"], "config": f"configs/{p.name}", "run_id": f"P43_vcs_{label}_seed0", "final_ckpt": final, "chain_links": links})
(ROOT / "configs" / "HPARAM_O_SHA256.json").write_text(json.dumps({"stage": STAGE, "units": units, "files": files}, indent=2))
print("\n".join(f"{u['run_id']} views={u['views']} ep={u['epochs']} B={u['batch']} links={u['chain_links']} {files[Path(u['config']).name][:12]}" for u in units))
