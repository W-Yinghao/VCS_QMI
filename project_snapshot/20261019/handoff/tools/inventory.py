"""Read-only source inventory for the CS_QMI handover; no model/data evaluation."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff" / "evidence"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archives = []
    for path in sorted(ROOT.glob("*.zip")):
        rec = {"path": path.name, "sha256": sha(path.read_bytes()), "members": []}
        with zipfile.ZipFile(path) as z:
            rec["crc_failure"] = z.testzip()
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"unsafe archive member: {name}")
                data = z.read(info)
                entry = {"path": str(name), "bytes": len(data), "sha256": sha(data)}
                if name.suffix.lower() == ".md":
                    content = data.decode("utf-8", errors="replace")
                    entry["headings"] = [s for s in content.splitlines() if s.startswith("#")]
                    target = OUT / "archive_texts" / path.stem / str(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                rec["members"].append(entry)
        rec["file_count"] = len(rec["members"])
        archives.append(rec)
    (OUT / "archive_inventory.json").write_text(json.dumps(archives, indent=2, ensure_ascii=False) + "\n")
    trees = []
    for path in sorted(ROOT.glob("ssl_pilot*")):
        if not (path / ".git").exists():
            continue
        def git(*args):
            return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
        files = git("ls-files").splitlines()
        trees.append({"path": path.name, "branch": git("branch", "--show-current"),
                      "head": git("rev-parse", "HEAD"), "status": git("status", "--short"),
                      "tracked_files": len(files), "files": files})
    (OUT / "worktree_inventory.json").write_text(json.dumps(trees, indent=2, ensure_ascii=False) + "\n")
    from pypdf import PdfReader
    pdfs = []
    for path in sorted(ROOT.glob("*.pdf")):
        reader = PdfReader(str(path))
        content = "\n\n".join(f"PAGE {i + 1}\n{p.extract_text()}" for i, p in enumerate(reader.pages))
        target = OUT / "pdf_texts" / (path.stem + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        pdfs.append({"path": path.name, "sha256": sha(path.read_bytes()), "pages": len(reader.pages),
                     "extracted_text": str(target.relative_to(ROOT))})
    (OUT / "pdf_inventory.json").write_text(json.dumps(pdfs, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"archives": len(archives), "archive_files": sum(x["file_count"] for x in archives),
                      "crc_failures": [x["path"] for x in archives if x["crc_failure"]],
                      "worktrees": len(trees), "pdfs": pdfs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
