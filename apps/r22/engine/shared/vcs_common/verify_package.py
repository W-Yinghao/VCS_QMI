"""Verify frozen package files without importing application dependencies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    root=root.resolve()
    manifest=json.loads((root/"PACKAGE_MANIFEST.json").read_text())
    errors=[]
    seen=set()
    for item in manifest["files"]:
        relative=item["path"]
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            errors.append({"path":relative,"reason":"invalid_or_duplicate_path"});continue
        seen.add(relative)
        if not path.is_file():
            errors.append({"path":relative,"reason":"missing"});continue
        data=path.read_bytes()
        if len(data)!=item["size"] or hashlib.sha256(data).hexdigest()!=item["sha256"]:
            errors.append({"path":relative,"reason":"content_mismatch"})
    # Newly inserted executable modules can change imports without touching an
    # existing file. Reject such source overlays, but permit runtime outputs.
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".py",".sh",".sbatch"):
            rel=path.relative_to(root).as_posix()
            if rel not in seen and not any(part in ("runs","outputs","external",".venv") for part in path.relative_to(root).parts):
                errors.append({"path":rel,"reason":"unlisted_executable_source"})
    return {"status":"PASS" if not errors else "FAIL","file_count":len(seen),"errors":errors,
            "manifest_sha256":hashlib.sha256((root/"PACKAGE_MANIFEST.json").read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",default=".")
    args=p.parse_args();result=verify(Path(args.root));print(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
