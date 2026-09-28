# S2b (CS_QMI P77) entry point: registers the VCSS2B method without editing solo/methods/__init__.py or main_pretrain.py
# (both are part of the frozen P75 patch, sha256 prefix ee399078ae0a4c7b, and are loaded at start by the P75 units).
# main_pretrain.py is then executed as __main__ (runpy), so Hydra resolves --config-path exactly as for the P75 units
# (file-relative; an imported module would make Hydra look for a Python package instead).
# Run from the solo-learn root exactly like main_pretrain.py:
#     python main_pretrain_s2b.py --config-path scripts/pretrain/cifar --config-name s2b_vcs.yaml seed=0 ...
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solo.methods import METHODS  # noqa: E402
from solo.methods.vcs_s2b import VCSS2B  # noqa: E402

METHODS["vcs_s2b"] = VCSS2B  # the same dict object main_pretrain imports

if __name__ == "__main__":
    runpy.run_path(os.path.join(HERE, "main_pretrain.py"), run_name="__main__")
