"""Bind the frozen supplement runner of one verified source tree into this process (read-only).

The frozen trees are imported, never edited; the caller has already verified every
source file against the frozen RELEASE.json / validation MANIFEST (common.verify_*).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from .common import require


def _under(module, root):
    filename = getattr(module, "__file__", None)
    return filename is not None and Path(filename).resolve().is_relative_to(Path(root).resolve())


def import_runner(source_root, module_name="run"):
    """Import <source_root>/scripts/v10_supplement/run.py as `module_name` with the tree's own v10 package.

    Refuses when a v10 package from any other tree is already imported, so a diagnostics
    process can never mix frozen and non-frozen implementations.
    """
    source_root = Path(source_root).resolve()
    src = source_root / "src"
    for name, module in tuple(sys.modules.items()):
        if name in ("v10", "v9", "r6_reg_pilot", "r6_reg_p2") or name.startswith(("v10.", "v9.")):
            require(_under(module, src), f"Module {name} was imported from outside the frozen tree {source_root}")
    if module_name in sys.modules:
        require(_under(sys.modules[module_name], source_root), f"Module {module_name!r} belongs to another tree")
        return sys.modules[module_name]
    for entry in (str(source_root / "scripts/v10_supplement"), str(src)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(module_name, source_root / "scripts/v10_supplement/run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_script(source_root, relative, module_name):
    path = Path(source_root).resolve() / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def initialize(runner, source_root):
    """runner.initialize_science(), then prove every bound scientific module comes from the frozen tree."""
    runner.initialize_science()
    src = Path(source_root).resolve() / "src"
    for label, module in (("adapter", runner.A), ("checkpoints", runner.CP),
                          ("training", sys.modules[runner.new_trainer.__module__]),
                          ("registration", sys.modules[runner.RegistrationSession.__module__]),
                          ("host_g1", runner.A.G), ("host_P", runner.A.P), ("host_Q", runner.A.Q)):
        require(_under(module, src), f"{label} module is not from the frozen tree: {getattr(module, '__file__', None)}")
    return runner
