"""SLURM helpers and code identities: syntax, queue rules (all v10_ jobs <= 2, normal QOS <= 29), the pinned paths shared by the
shell helpers and t03_common, the shell pin formula = tree_hash, and the BF code identity left untouched by the T03 branch."""
from __future__ import annotations

import re
import subprocess

import pytest

import t03_common as C

SL = C.REPO / "slurm" / "v10_t03"


@pytest.mark.parametrize("name", ("submit.sh", "t03_header.sh", "t03_smoke.sbatch"))
def test_shell_syntax(name):
    assert subprocess.run(["bash", "-n", str(SL / name)], capture_output=True).returncode == 0


def test_smoke_sbatch_directives():
    s = (SL / "t03_smoke.sbatch").read_text()
    d = dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", s, flags=re.M))
    assert d["job-name"] == "v10_t03_smoke" and d["partition"] == "cpu-high" and d["nice"] == "0"
    assert "gres" not in d and "exclude" not in d
    assert "t03_header cpu slurm/v10_t03/t03_smoke.sbatch" in s and "--block smoke" in s and "--root-seed 20261016" in s
    assert "V10_T03_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10/T03" in s


def test_submit_queue_rules():
    s = (SL / "submit.sh").read_text()
    assert "grep -c '^v10_'" in s and '"$n_v10" -ge 2' in s and '"$n_normal" -ge 29' in s
    assert "grep -c normal" in s and "V10_T03_EXPECT_CODE" in s and "git ls-files --error-unmatch" in s


def _shell_paths(text, var):
    return re.search(rf'{var}="([^"]+)"', text).group(1).replace("$PIN_PATHS", "").split()


def test_pinned_paths_agree_everywhere():
    for name in ("submit.sh", "t03_header.sh"):
        t = (SL / name).read_text()
        assert tuple(_shell_paths(t, "PIN_PATHS")) == C.T03_PIN_PATHS
        assert tuple(_shell_paths(t, "PIN_PATHS") + _shell_paths(t, "DIRTY_PATHS")) == C.DIRTY_PATHS


def test_shell_pin_formula_equals_tree_hash():
    paths = " ".join(C.T03_PIN_PATHS)
    out = subprocess.run(f"git ls-files {paths} | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16", shell=True, cwd=C.REPO,
                         capture_output=True, text=True).stdout.strip()
    assert out == C.tree_hash(C.T03_PIN_PATHS)
    from v10 import adapter as A
    assert A.code_hash() == C.tree_hash(C.BF_CODE_PATHS)


def test_t03_branch_leaves_the_bf_code_untouched():
    base = C.git("merge-base", "HEAD", "v10/mf-vcs")
    assert not base.startswith("<git error")
    changed = C.git("diff", "--name-only", base, "HEAD", "--", "src/v10", "scripts/v10", "slurm/v10", "tests/v10")
    assert changed == "", changed
    dirty = C.git("status", "--porcelain", "--", "src/v10", "scripts/v10", "slurm/v10", "src/v9")
    assert dirty == "", dirty
