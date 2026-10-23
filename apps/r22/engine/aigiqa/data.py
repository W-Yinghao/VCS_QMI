from __future__ import annotations

import hashlib
import io
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .manifest import read_csv, safe_image_path


class IQADataset(Dataset):
    def __init__(self, csv_path, image_root, target=None, image_size=224, verify_images=True):
        self.rows = read_csv(csv_path)
        self.image_root, self.target, self.image_size = image_root, target, image_size
        self.verify_images = verify_images
        if not self.rows:
            raise ValueError(f"Empty dataset: {csv_path}")
        ids = [r["sample_id"] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample ids")
        if target is None and any("quality_mos" in r or "alignment_mos" in r for r in self.rows):
            raise ValueError("Prediction must use label-free *_inputs.csv")
        if target not in {None, "quality", "alignment"}:
            raise ValueError("target must be quality or alignment")
        self.mean = torch.tensor([.48145466, .4578275, .40821073]).view(3, 1, 1)
        self.std = torch.tensor([.26862954, .26130258, .27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = safe_image_path(self.image_root, row["image"])
        blob = path.read_bytes()
        if self.verify_images and hashlib.sha256(blob).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image bytes changed since preparation: {row['sample_id']}")
        with Image.open(io.BytesIO(blob)) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        # Same deterministic direct resize as the reference. No horizontal flip:
        # a flip can change the truth of left/right statements in a prompt.
        tensor = (tensor - self.mean) / self.std
        tensor = F.interpolate(tensor[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        result = {"image": tensor, "prompt": row["prompt"], "sample_id": row["sample_id"],
                  "prompt_group": row["prompt_group"], "dataset": row["dataset"]}
        if self.target:
            raw_mos = float(row[self.target + "_mos"])
            result["label"] = torch.tensor(raw_mos / 5.0, dtype=torch.float32)
            result["raw_mos"] = torch.tensor(raw_mos, dtype=torch.float64)
        return result
