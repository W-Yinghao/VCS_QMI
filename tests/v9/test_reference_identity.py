"""The copies of the package reference used by src/v9 (and the verbatim test copies) are byte-identical to the package files."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "v9" / "package_v9_20261016"
PAIRS = [("reference/core.py", "src/v9/reference_core.py"), ("reference/torch_losses.py", "src/v9/reference_torch_losses.py"),
         ("tests/test_core.py", "tests/v9/test_pkg_core.py"), ("tests/test_torch.py", "tests/v9/test_pkg_torch.py")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_byte_identity():
    for src, dst in PAIRS:
        assert sha(PKG / src) == sha(ROOT / dst), f"{dst} differs from the package's {src}"


def test_package_manifest_lists_reference():
    text = (PKG / "MANIFEST.sha256").read_text()
    assert "reference/core.py" in text and sha(PKG / "reference/core.py") in text
