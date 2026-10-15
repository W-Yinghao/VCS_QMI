"""tests/v9: the package's own test files (test_pkg_*.py, verbatim copies) import `reference.*` from the package directory; everything else uses src/v9."""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "v9" / "package_v9_20261016"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
