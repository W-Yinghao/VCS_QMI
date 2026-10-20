#!/usr/bin/env python3
"""Entry point for the unpacked package; name avoids shadowing the v9n2m package."""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from v9n2m.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
