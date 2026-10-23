"""R22-B: anchor role per training caption (int8; -1 = U, 0 teacher_warmup, 1 critic_fit, 2 critic_select, 3 reference_moments) from the
V9 shared stage (pairs.observed_pairs.json + roles.json, train roles only; no clean labels, no eval_only file).  Cross-checked against the
V9 export used by the trainer (anchor mask and observed mapping).  Run under n2p_guard.py policy train.

    python r22_export_roles.py --shared <stage> --v9-export <unit>/v9_export --out <unit>/r22_export
"""
import argparse
import hashlib
import json
import os

import numpy as np

CODES = {"U": -1, "teacher_warmup": 0, "critic_fit": 1, "critic_select": 2, "reference_moments": 3}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--v9-export", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    obs_p, roles_p = os.path.join(a.shared, "pairs.observed_pairs.json"), os.path.join(a.shared, "roles.json")
    rows = json.load(open(obs_p))
    roles = json.load(open(roles_p))
    anchor = np.load(os.path.join(a.v9_export, "v9_anchor_mask.npy")).astype(bool)
    t2i = np.load(os.path.join(a.v9_export, "v9_observed_t2i.npy"))
    n = len(anchor)
    code = np.full(n, -2, dtype=np.int8)
    img_role = {i: r for r in CODES if r != "U" for i in roles["A"][r]}
    for r in rows:
        cid = int(r["caption_id"])
        code[cid] = CODES[r["role"]]
        if int(r["observed_image_id"]) != int(t2i[cid]):
            raise SystemExit(f"observed mapping differs from the V9 export at caption {cid}")
        if r["role"] != "U" and img_role.get(r["observed_image_id"]) != r["role"]:
            raise SystemExit(f"caption {cid}: role {r['role']} differs from roles.json for image {r['observed_image_id']}")
    if (code == -2).any() or len(rows) != n:
        raise SystemExit("not every caption has a role")
    if not np.array_equal(code >= 0, anchor):
        raise SystemExit("roles disagree with the V9 anchor mask")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "r22_anchor_roles.npy")
    np.save(out, code)
    counts = {r: int((code == c).sum()) for r, c in CODES.items()}
    images = {r: len(set(t2i[code == c].tolist())) for r, c in CODES.items() if r != "U"}
    man = {"schema": "r22-anchor-roles-v1", "shared": a.shared, "inputs": {"observed_pairs_sha256": sha(obs_p), "roles_sha256": sha(roles_p),
           "v9_anchor_mask_sha256": sha(os.path.join(a.v9_export, "v9_anchor_mask.npy")),
           "v9_observed_t2i_sha256": sha(os.path.join(a.v9_export, "v9_observed_t2i.npy"))},
           "codes": CODES, "caption_counts": counts, "anchor_image_counts": images, "roles_hashes": roles["hashes"],
           "output": {"file": out, "sha256": sha(out)}, "clean_labels_read": False}
    json.dump(man, open(os.path.join(a.out, "r22_roles_manifest.json"), "w"), indent=1)
    print(json.dumps({"caption_counts": counts, "anchor_image_counts": images}))


if __name__ == "__main__":
    main()
