"""CPU checks for the N2P A/B adapter (no research data read except the official vocab; fixture data only; no GPU).

  1. frozen candidate maps: NCR 45 / L2RM 40, warm-up first, gate maps are prefixes
  2. both patch layers apply to fresh copies of the clones and the patched files compile (official envs)
  3. anchor-off native path unchanged after both layers (scripts/v9/n2_degradation.py check_ncr / check_l2rm, fixture)
  4. read guard: TEST and EVAL_ONLY opens refused, train_* refused under the dev policy, dev_* allowed, trace written
  5. CPU fixture smoke of the real driver (prepare + train) for NCR and L2RM with anchors ON: co-training / rematching epoch runs
     through the patched split path, every candidate saved, native TEST skipped, no TEST / EVAL_ONLY file opened
  6. gate control (L2RM smoke unit): (a) a gate whose health fails returns non-zero from select; (b) a full unit is refused before
     anything is written unless --gate-report is this family's B0_PASS gate with the same frozen config / patches / forward+guard

    python -I run_cpu_checks.py --work <empty scratch dir> --out <CPU_CHECKS.json>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_ab")
AB = REPO / "v9_n2p/ab"
PY = "/home/infres/yinwang/CS_QMI/env/bin/python"
FROZEN = json.loads((AB / "b0/B0_FROZEN.json").read_text())
ENVS, CLONES = FROZEN["paths"]["envs"], FROZEN["paths"]["clones"]
WORDS = "a man woman dog red shirt is in the on with playing standing street ball two people white black sitting girl boy".split()


def layered_copy(b, dst):
    shutil.copytree(CLONES[b], dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output", "noise_index", "*.png", "cc152k"))
    for p in (REPO / f"v9/patches/{b}_anchor_assisted.diff", AB / f"patches/{b}_n2p_devonly.diff"):
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(p)], cwd=dst, capture_output=True, text=True)
        if r.returncode:
            return r.stdout + r.stderr
    return None


def check_maps():
    m = FROZEN["candidate_maps"]
    ok = (len(m["ncr"]["full"]) == 45 and len(m["l2rm"]["full"]) == 40
          and all(c["order"] == i for b in m for s in m[b] for i, c in enumerate(m[b][s]))
          and all(m[b]["gate"] == m[b]["full"][:len(m[b]["gate"])] for b in m)
          and [c["phase"] for c in m["ncr"]["full"][:6]] == ["warmup"] * 5 + ["main"])
    return {"pass": ok, "counts": {b: {s: len(m[b][s]) for s in m[b]} for b in m}}


def check_patches_and_native(work):
    out = {}
    for b in ("ncr", "l2rm"):
        td = work / f"degr_{b}"
        shutil.copytree(CLONES[b], td / "native", ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output", "noise_index", "*.png", "cc152k"))
        err = layered_copy(b, td / "patched")
        if err:
            out[b] = {"pass": False, "patch_error": err}
            continue
        comp = subprocess.run([ENVS[b] + "/bin/python", "-m", "compileall", "-q", str(td / "patched")], capture_output=True, text=True)
        code = ("import sys, json, numpy as np; from pathlib import Path; sys.path.insert(0, %r); import n2_degradation as D; "
                "lay = D.fixture_layout(seed=0); td = Path(%r); nf = td / 'noise.npy'; np.save(nf, lay['noise_index']); "
                "print('RESULT ' + json.dumps(D.check_%s(td, lay, nf, 0)))") % (str(REPO / "scripts/v9"), str(td), b)
        r = subprocess.run([ENVS[b] + "/bin/python", "-c", code], capture_output=True, text=True, cwd=td)
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        res = json.loads(line[-1][7:]) if line else {"error": r.stderr[-2000:]}
        out[b] = {"pass": comp.returncode == 0 and bool(res.get("native_path_identical")), "compile_rc": comp.returncode, "anchor_off": res}
    return out


def fixture_data(work):
    rng = np.random.default_rng(0)
    d = work / "data/f30k_precomp"
    d.mkdir(parents=True)
    cap = lambda: " ".join(rng.choice(WORDS, size=int(rng.integers(5, 10)))) + " ."
    np.save(d / "train_ims.npy", rng.standard_normal((200, 36, 16)).astype(np.float32))
    (d / "train_caps.txt").write_text("".join(cap() + "\n" for _ in range(1000)))
    dev = np.repeat(rng.standard_normal((1014, 36, 16)).astype(np.float32), 5, axis=0)
    np.save(d / "dev_ims.npy", dev)
    (d / "dev_caps.txt").write_text("".join(cap() + "\n" for _ in range(5070)))
    (d / "dev_ids.txt").write_text("".join(f"{1000 + i // 5}\n" for i in range(5070)))
    np.save(d / "test_ims.npy", rng.standard_normal((50, 36, 16)).astype(np.float32))      # present but must stay unread
    (d / "test_caps.txt").write_text("".join(cap() + "\n" for _ in range(50)))
    (work / "eval_only").mkdir()
    (work / "eval_only/pairs.eval_only.json").write_text("{}")
    return d


def check_guard(work, data):
    g = AB / "b0/n2p_guard.py"
    probe = work / "probe.py"
    probe.write_text("import sys\nopen(sys.argv[1]).read()\n")
    res = {}
    for name, policy, target, expect_ok in (("dev_file_dev_policy", "dev", data / "dev_caps.txt", True),
                                            ("train_file_dev_policy", "dev", data / "train_caps.txt", False),
                                            ("test_file_train_policy", "train", data / "test_caps.txt", False),
                                            ("eval_only_train_policy", "train", work / "eval_only/pairs.eval_only.json", False)):
        trace = work / f"guard_{name}.jsonl"
        r = subprocess.run([PY, "-I", str(g), str(trace), policy, str(data), str(probe), str(target)], capture_output=True, text=True)
        rows = [json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
        hit = [x for x in rows if x["real"] == str(target)]
        ok = (r.returncode == 0) == expect_ok and bool(hit) and (hit[-1]["decision"] == ("allow" if expect_ok else "DENY"))
        res[name] = {"pass": ok, "rc": r.returncode, "decision": hit[-1]["decision"] if hit else None}
    return {"pass": all(v["pass"] for v in res.values()), "cases": res}


def check_smoke(work, data):
    sys.path.insert(0, str(REPO / "scripts/v9"))
    from n2_official_smoke import shared_stage
    shared = work / "shared"
    shared_stage(shared, 200, 0.4, 901)
    res = {}
    for b in ("ncr", "l2rm"):
        f = json.loads(json.dumps(FROZEN))
        f["paths"]["data_dir"], f["paths"]["shared_stage"] = str(data), str(shared)
        f["schedules"][b]["gate"] = {"warmup_epoch": 1, "num_epochs": 1, "lr_update": f["schedules"][b]["gate"]["lr_update"]}
        fa = f["schedules"][b]["fixed_args"]
        for k, v in (("--batch_size", "16"), ("--workers", "0"), ("--queue_length", "16")):
            if k in fa:
                fa[fa.index(k) + 1] = v
        fa += ["--img_dim", "16", "--embed_size", "32", "--sim_dim", "16", "--word_dim", "16"]
        f["candidate_maps"][b]["gate"] = f["candidate_maps"][b]["full"][:1] + [c for c in f["candidate_maps"][b]["full"] if c["candidate_id"] == "main:0"]
        f["candidate_maps"][b]["gate"][1] = dict(f["candidate_maps"][b]["gate"][1], order=1)
        fz = work / f"frozen_{b}.json"
        fz.write_text(json.dumps(f))
        unit = work / f"unit_{b}"
        env = {**os.environ, "N2P_CPU_SMOKE": "1", "CUDA_VISIBLE_DEVICES": ""}
        rcs = {}
        for step in ("prepare", "train"):
            r = subprocess.run([PY, "-I", str(AB / "b0/n2p_baseline.py"), step, "--baseline", b, "--schedule", "gate", "--seed", "901",
                                "--rho", "0.4", "--unit", str(unit), "--frozen", str(fz)], capture_output=True, text=True, env=env)
            rcs[step] = r.returncode
            (work / f"smoke_{b}_{step}.log").write_text(r.stdout + r.stderr)
        log = (unit / "train.log").read_text(errors="replace") if (unit / "train.log").exists() else ""
        led = [json.loads(x)["candidate_id"] for x in (unit / "candidates/candidates.jsonl").read_text().splitlines()] if (unit / "candidates/candidates.jsonl").exists() else []
        trace = [json.loads(x) for x in (unit / "train_trace.jsonl").read_text().splitlines()] if (unit / "train_trace.jsonl").exists() else []
        data_opened = sorted({Path(x["real"]).name for x in trace if x["dataset_file"]})
        r = {"rc": rcs, "candidates": led, "test_skipped": "[N2P] native end-of-training TEST evaluation disabled" in log,
             "observed_index_loaded": "load V9 observed index" in log, "anchor_term_lines": log.count("[V9] anchor-supervised term"),
             "traceback": "Traceback" in log, "dataset_files_opened": data_opened,
             "denied": [x for x in trace if x["decision"] == "DENY"]}
        r["pass"] = (rcs == {"prepare": 0, "train": 0} and led == ["warmup:0", "main:0"] and r["test_skipped"] and r["observed_index_loaded"]
                     and r["anchor_term_lines"] >= 1 and not r["traceback"] and not r["denied"]
                     and set(data_opened) <= {"train_ims.npy", "train_caps.txt", "dev_ims.npy", "dev_caps.txt"})
        res[b] = r
    return res


def check_gate_control(work):
    env = {**os.environ, "N2P_CPU_SMOKE": "1", "CUDA_VISIBLE_DEVICES": ""}
    fz, gate = work / "frozen_l2rm.json", work / "unit_l2rm"
    drv = lambda step, sched, unit, *extra: subprocess.run([PY, "-I", str(AB / "b0/n2p_baseline.py"), step, "--baseline", "l2rm", "--schedule", sched,
                                                            "--seed", "901", "--rho", "0.4", "--unit", str(unit), "--frozen", str(fz), *extra],
                                                           capture_output=True, text=True, env=env).returncode
    rc = {s: drv(s, "gate", gate) for s in ("forward", "score", "select")}
    rep_status = json.loads((gate / "B0_DEV_ADAPTER_REPORT.json").read_text())["status"] if (gate / "B0_DEV_ADAPTER_REPORT.json").exists() else None
    t1 = {"rc": rc, "report_status": rep_status, "pass": rc["forward"] == 0 and rc["score"] == 0 and rc["select"] == 6 and rep_status == "B0_FAIL"}
    copy = work / "gate_copy"                                         # same gate unit with its status set to B0_PASS
    shutil.copytree(gate / "adapter", copy / "adapter")
    shutil.copy(gate / "unit.json", copy / "unit.json")
    r = json.loads((gate / "B0_DEV_ADAPTER_REPORT.json").read_text())
    r["status"] = "B0_PASS"
    (copy / "B0_DEV_ADAPTER_REPORT.json").write_text(json.dumps(r))
    cases = {}
    def case(name, unit, *extra, expect):
        code = drv("prepare", "full", unit, *extra)
        cases[name] = {"rc": code, "unit_written": unit.exists(), "pass": (code == 0 and unit.exists()) if expect == "accept" else (code == 7 and not unit.exists())}
    case("no_gate_report", work / "full_a", expect="refuse")
    case("gate_report_B0_FAIL", work / "full_b", "--gate-report", str(gate / "B0_DEV_ADAPTER_REPORT.json"), expect="refuse")
    case("gate_B0_PASS_same_identity", work / "full_c", "--gate-report", str(copy / "B0_DEV_ADAPTER_REPORT.json"), expect="accept")
    u = json.loads((copy / "unit.json").read_text())
    (copy / "unit.json").write_text(json.dumps(dict(u, frozen_config_sha256="0" * 64)))
    case("gate_B0_PASS_other_frozen_config", work / "full_d", "--gate-report", str(copy / "B0_DEV_ADAPTER_REPORT.json"), expect="refuse")
    (copy / "unit.json").write_text(json.dumps(u))
    (copy / "adapter/n2p_forward.py").write_text((copy / "adapter/n2p_forward.py").read_text() + "\n# changed\n")
    case("gate_B0_PASS_other_forward_script", work / "full_e", "--gate-report", str(copy / "B0_DEV_ADAPTER_REPORT.json"), expect="refuse")
    t2 = {"cases": cases, "pass": all(c["pass"] for c in cases.values())}
    return {"select_nonzero_on_failed_gate": t1, "full_requires_matching_B0_PASS": t2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=False)
    data = fixture_data(work)
    res = {"frozen_maps": check_maps(), "patches_and_native_path": check_patches_and_native(work), "guard": check_guard(work, data),
           "cpu_smoke": check_smoke(work, data)}
    res["gate_control"] = check_gate_control(work)
    flat = [res["frozen_maps"]["pass"], res["guard"]["pass"]] + [v["pass"] for v in res["patches_and_native_path"].values()] + \
           [v["pass"] for v in res["cpu_smoke"].values()] + [v["pass"] for v in res["gate_control"].values()]
    res["status"] = "PASS" if all(flat) else "FAIL"
    res["scope"] = "CPU fixture checks only; not a GPU gate, not B0 PASS"
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"status": res["status"], **{k: (v.get("pass") if isinstance(v, dict) and "pass" in v else {kk: vv.get("pass") for kk, vv in v.items()}) for k, v in res.items() if isinstance(v, dict)}}))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
