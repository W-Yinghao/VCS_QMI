"""Download public pretrained weights for the R23 SOTA-matcher baselines (user approval 2026-10-26 「都可以」) into
/projects/EEG-foundation-model/yinghao/models/<name>/ and write MANIFEST.json (url, sha256, bytes, license, utc)."""
import hashlib, json, shutil, sys, time, urllib.request
from pathlib import Path

M = Path('/projects/EEG-foundation-model/yinghao/models')
SRC = M / 'sota_src'
W = [  # (name, file, url, license)
    ('lightglue', 'superpoint_v1.pth', 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth',
     'SuperPoint weights: Magic Leap academic/non-commercial research license'),
    ('lightglue', 'superpoint_lightglue_v0-1_arxiv.pth', 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth', 'Apache-2.0'),
    ('lightglue', 'disk_lightglue_v0-1_arxiv.pth', 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth', 'Apache-2.0'),
    ('lightglue', 'aliked_lightglue_v0-1_arxiv.pth', 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth', 'Apache-2.0'),
    ('lightglue', 'aliked-n16.pth', 'https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth', 'BSD-3-Clause (ALIKED)'),
    ('lightglue', 'disk_depth-save.pth', 'https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth', 'Apache-2.0 (DISK)'),
    ('roma', 'roma_outdoor.pth', 'https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth', 'MIT (RoMa)'),
    ('roma', 'dinov2_vitl14_pretrain.pth', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'Apache-2.0 (DINOv2)'),
    ('minima', 'minima_lightglue.pth', 'https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth',
     'Apache-2.0 (MINIMA); uses SuperPoint (non-commercial)'),
    ('minima', 'minima_loftr.ckpt', 'https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt', 'Apache-2.0 (MINIMA)'),
    ('minima', 'minima_roma.pth', 'https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth', 'Apache-2.0 (MINIMA)'),
    ('dinov2', 'dinov2_vits14_pretrain.pth', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', 'Apache-2.0 (DINOv2)'),
    ('dinov2', 'dinov2_vitb14_pretrain.pth', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth', 'Apache-2.0 (DINOv2)'),
]
COPIES = [('xfeat', 'xfeat.pt', SRC / 'accelerated_features/weights/xfeat.pt', 'Apache-2.0 (XFeat, repo copy)'),
          ('xfeat', 'xfeat-lighterglue.pt', SRC / 'accelerated_features/weights/xfeat-lighterglue.pt', 'Apache-2.0 (XFeat LighterGlue, repo copy)')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def main():
    man = {}
    for name, fn, url, lic in W:
        d = M / name; d.mkdir(parents=True, exist_ok=True); p = d / fn
        if not p.exists():
            tmp = p.with_suffix(p.suffix + '.part')
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f: shutil.copyfileobj(r, f, 1 << 22)
            tmp.replace(p)
        man.setdefault(name, []).append(dict(file=fn, url=url, sha256=sha(p), bytes=p.stat().st_size, license=lic))
        print(name, fn, p.stat().st_size, flush=True)
    for name, fn, src, lic in COPIES:
        d = M / name; d.mkdir(parents=True, exist_ok=True); p = d / fn
        if not p.exists(): shutil.copy2(src, p)
        man.setdefault(name, []).append(dict(file=fn, url='repo copy: ' + str(src), sha256=sha(p), bytes=p.stat().st_size, license=lic))
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for name, files in man.items():
        (M / name / 'MANIFEST.json').write_text(json.dumps(dict(utc=utc, purpose='R23 SOTA matcher baselines (DEV only)', files=files), indent=1) + '\n')
    print('DONE', sum(len(v) for v in man.values()))


if __name__ == '__main__':
    sys.exit(main())
