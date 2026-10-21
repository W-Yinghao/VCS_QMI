"""V9 N2P C1: paired rich-VCS / rich-BCE SGR students through the original v9.n2.train.WeightedStudent.

Minimal adapter for package v9/rich_input/README_zh.md section 6 and ROUND_PROTOCOL v9.rich_student:
  * weights only through the package's student_handoff.validated_handoff (original read_weight_lock plus
    cache/selection/export/observed identities), and only a file that PILOT_DECISION.json verified;
    A weight 1, U weight from the lock, normaliser (n_A + sum_U w) / n_pool fixed over the whole pool;
  * HostConfig from the frozen HOST_LOCK, checked against the protocol recipe; loss_scale "sum";
    the protocol's deterministic settings are applied and their effective values recorded;
  * one Slurm job per seed runs the two arms in the protocol order on one GPU, so both arms share the
    seed, the initial model, the per-epoch batch order and the environment; only the weights differ;
  * reads train + the full DEV split (1014 images / 5070 captions); there is no TEST loader here.

  python -I -B c1_student.py accept    --root ROOT [--seed 901] [--steps 50]   small technical acceptance
  python -I -B c1_student.py run       --root ROOT --seed SEED                 both arms, 40 epochs each
  python -I -B c1_student.py summarize --root ROOT                             paired DEV readout
Exit status: 0 done, 3 gate closed, 4 check failed, 9 forbidden file access, 76 environment differs
from the seed's earlier attempt (resubmit on the same partition).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG = REPO / "v9_n2p" / "VCS_V9_N2P_V10_Review_Server_Package_v1_20261020"
N2M_SRC = REPO / "v9_n2m" / "VCS_V9_N2M_Server_Package_v1_20261019" / "src"
OLD = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4")
DATA = Path("/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp")
VOCAB = Path("/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json")
TOKENS = Path("/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/source/token_cache")   # frozen N2F token cache
HOST_LOCK = REPO / "reports" / "V9" / "HOST_LOCK.yaml"
ARMS = {"RICH_VCS": "rich_vcs", "RICH_BCE": "rich_bce"}
SEEDS = (901, 902, 903)
ELIGIBLE = "ELIGIBLE_PENDING_TECHNICAL_ACCEPTANCE"
PROTOCOL_POOL = {"n_pool": 145000, "n_A": 14500, "n_U": 130500, "steps_per_epoch": 1133, "last_batch": 104}
PAIRED_KEYS = ("gpu", "driver", "torch", "cuda", "cudnn", "python", "numpy")
RECALLS = ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")

for _p in (HERE, PKG / "v9" / "rich_input", REPO / "src"):
    sys.path.insert(0, str(_p))
import trace_open  # noqa: E402


class Refused(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, obj, overwrite=True):
    path = Path(path)
    if not overwrite and path.exists():
        raise Refused(4, f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")          # a non-finite loss stays visible in a FAIL record
    tmp.replace(path)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def protocol():
    return json.loads((PKG / "config" / "ROUND_PROTOCOL.json").read_text())["v9"]["rich_student"]


# ------------------------------------------------------------------------------------------- gate and environment
def read_gate(root, need_acceptance):
    p = root / "c0" / "PILOT_DECISION.json"
    if not p.is_file():
        raise Refused(3, f"missing {p}")
    d = json.loads(p.read_text())
    if d.get("status") != ELIGIBLE or d.get("full_training_eligible") is not True:
        raise Refused(3, f"pilot decision status {d.get('status')!r}: no C1 student run")
    verified = {(int(w["seed"]), w["arm"]): w["sha256"] for w in d.get("verified_weight_identities", [])}
    gate = {"decision": str(p), "decision_sha256": sha256(p), "decision_status": d["status"]}
    if need_acceptance:
        a = root / "c1" / "C1_ACCEPTANCE.json"
        acc = json.loads(a.read_text()) if a.is_file() else {}
        if acc.get("status") != "PASS":
            raise Refused(3, f"technical acceptance {a} is missing or not PASS")
        if acc.get("gate", {}).get("decision_sha256") != gate["decision_sha256"]:
            raise Refused(3, f"technical acceptance {a} was made under a different PILOT_DECISION")
        gate.update(acceptance=str(a), acceptance_sha256=sha256(a))
    return gate, verified


def configure_precision(spec):
    """Protocol settings before any CUDA work; returns the effective values (must equal the protocol)."""
    if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", spec["CUBLAS_WORKSPACE_CONFIG"]) != spec["CUBLAS_WORKSPACE_CONFIG"]:
        raise Refused(4, f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']!r} differs from the protocol")
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.use_deterministic_algorithms(spec["deterministic_algorithms"])
    torch.backends.cudnn.deterministic = spec["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = spec["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = spec["cuda_matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = spec["cudnn_tf32"]
    effective = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                 "cudnn_deterministic": torch.backends.cudnn.deterministic,
                 "cudnn_benchmark": torch.backends.cudnn.benchmark,
                 "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                 "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                 "mixed_precision": False}          # v9.n2.train has no autocast / GradScaler
    if effective != spec:
        raise Refused(4, f"effective precision {effective} differs from the protocol {spec}")
    return {**effective, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_threads": torch.get_num_threads()}


def environment():
    import numpy
    import torch
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        driver = None
    return {"host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(), "numpy": numpy.__version__,
            "job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION")}


def check_paired_environment(runs, seed, env):
    """Every attempt of a seed pair (both arms, any resubmission) must share GPU model and software."""
    for arm in ARMS:
        p = runs / f"s{seed}__{arm}" / "attempts.jsonl"
        for line in (p.read_text().splitlines() if p.is_file() else []):
            rec = json.loads(line)
            if rec.get("event") == "start":
                diff = {k: [rec["environment"].get(k), env.get(k)] for k in PAIRED_KEYS if rec["environment"].get(k) != env.get(k)}
                if diff:
                    raise Refused(76, f"seed {seed}: this allocation differs from an earlier attempt of the pair {diff}; "
                                      "resubmit on the same partition/node type")


def attempt_log(out, rec):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attempts.jsonl", "a") as f:
        f.write(json.dumps({"utc": utc(), "job_id": os.environ.get("SLURM_JOB_ID"), **rec}) + "\n")


# ------------------------------------------------------------------------------------------- inputs and checks
def load_context():
    """Original loaders: train + DEV from the frozen token cache, HostConfig from HOST_LOCK."""
    from v9.n2.data import eval_arrays, load_precomp, load_vocab
    from v9.n2.fixtures import collate_pairs
    from v9.n2.teacher import load_host_lock
    spec = protocol()
    vocab = load_vocab(VOCAB)
    cfg = load_host_lock(HOST_LOCK, "f30k_precomp", len(vocab))
    expected = {"num_epochs": spec["epochs"], "warmup_epochs": spec["warmup_epochs"], "batch_size": spec["batch_size"],
                "learning_rate": spec["learning_rate"], "lr_update": spec["lr_update"], "grad_clip": spec["grad_clip"],
                "margin": spec["margin"], "img_dim": 2048, "word_dim": 300, "embed_size": 1024, "sim_dim": 256,
                "num_layers": 1, "bi_gru": True, "num_region": 36, "sgr_step": 3}
    actual = {k: getattr(cfg, k) for k in expected}
    if actual != expected or cfg.to_verify or spec["optimizer"] != "torch.optim.Adam" or spec["loss_scale"] != "sum":
        raise Refused(4, f"host recipe differs from the protocol: {actual} vs {expected}; to_verify={cfg.to_verify}")
    tag = hashlib.sha256(json.dumps(sorted(vocab.items())).encode()).hexdigest()[:16]     # data.py cache key
    for split in ("train", "dev"):
        if not (TOKENS / f"f30k_precomp_{split}_caps_tokens_{tag}.json").is_file():
            raise Refused(4, f"frozen {split} token cache missing in {TOKENS}; refusing to tokenise into it")
    train = load_precomp(DATA, "train", vocab, cache_dir=TOKENS)
    dev = load_precomp(DATA, "dev", vocab, cache_dir=TOKENS)
    if (train["n_images"], len(train["captions"]), dev["n_images"], len(dev["captions"])) != (29000, 145000, 1014, 5070):
        raise Refused(4, "loader did not return the full train and DEV splits")
    dev_images, dev_captions = eval_arrays(dev)
    return {"cfg": cfg, "collate": partial(collate_pairs, train["images"], train["captions"]),
            "dev_images": dev_images, "dev_captions": dev_captions}


def arm_inputs(root, seed, arm, verified):
    from student_handoff import validated_handoff
    c0 = root / "c0" / f"s{seed}"
    observed = OLD / f"s{seed}" / "pairs.observed_pairs.json"
    paths = {"cache": c0 / "cache", "selection": c0 / "fit" / "FIT_SELECTION.json",
             "export_summary": c0 / "export" / "EXPORT_SUMMARY.json", "observed": observed}
    weights, lock, receipt = validated_handoff(REPO, paths["cache"], paths["selection"], paths["export_summary"],
                                               observed, ARMS[arm])
    if verified.get((seed, ARMS[arm])) != receipt["weight_sha256"]:
        raise Refused(4, f"s{seed}/{arm}: weight file is not one PILOT_DECISION.json verified")
    return {"weights": weights, "lock": lock, "receipt": receipt, "rows": json.loads(observed.read_text()),
            "paths": {**{k: str(v) for k, v in paths.items()}, "weights": lock["file"]}}


def initial_model_sha256(cfg, seed):
    """The trainer builds its host the same way (seed_everything + build_host on CPU, then .to(device))."""
    from v9.n2.teacher import build_host, seed_everything
    seed_everything(seed)
    host, h = build_host(cfg), hashlib.sha256()
    for name, t in sorted(host.state_dict().items()):
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_config(cfg, lr_mult):
    import torch
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=cfg.learning_rate * lr_mult)    # as in train.py
    conf = {"class": "torch.optim.Adam", **{k: list(v) if isinstance(v, tuple) else v for k, v in opt.defaults.items()}}
    return conf, canonical(conf)


def pool_checks(student, rows, weights, seed, cfg, expect=None):
    """A = 1, U = lock values, fixed full-pool normaliser, update count from the trainer's own batch iterator."""
    import numpy as np
    from v9.n2.teacher import iterate_batches
    is_u = np.array([r["role"] == "U" for r in rows])
    u_ids = [r["pair_id"] for r in rows if r["role"] == "U"]
    w_u = np.array([weights[p] for p in u_ids], dtype=np.float64)
    n, n_a, n_u = len(rows), int((~is_u).sum()), int(is_u.sum())
    normaliser = (n_a + math.fsum(w_u.tolist())) / n
    order, sizes = hashlib.sha256(), []
    for epoch in range(cfg.num_epochs):
        batches = list(iterate_batches(n, cfg.batch_size, np.random.default_rng([seed, 9020, epoch])))
        sizes.append([len(b) for b in batches])
        for b in batches:
            order.update(np.asarray(b, dtype="<i8").tobytes())
    checks = {"A_weight_exactly_one": bool(np.all(student.v[~is_u] == 1.0)),
              "U_weights_equal_lock": bool(np.array_equal(student.v[is_u], w_u)) and set(weights) == set(u_ids),
              "normaliser_full_pool": math.isclose(student.mean_v_full, normaliser, rel_tol=1e-12, abs_tol=0.0),
              "n_pool": n, "n_A": n_a, "n_U": n_u, "sum_w_U": math.fsum(w_u.tolist()), "mean_v_full_pool": student.mean_v_full,
              "steps_per_epoch": sorted({len(s) for s in sizes}), "last_batch_sizes": sorted({s[-1] for s in sizes}),
              "expected_updates": sum(len(s) for s in sizes), "data_order_sha256": order.hexdigest(),
              "drop_rule": "iterate_batches drops only singleton batches; the 104-row last batch is kept"}
    ok = checks["A_weight_exactly_one"] and checks["U_weights_equal_lock"] and checks["normaliser_full_pool"]
    if expect is not None:
        ok = ok and (n, n_a, n_u) == (expect["n_pool"], expect["n_A"], expect["n_U"]) \
            and checks["steps_per_epoch"] == [expect["steps_per_epoch"]] and checks["last_batch_sizes"] == [expect["last_batch"]] \
            and checks["expected_updates"] == expect["updates"]
    checks["ok"] = bool(ok)
    return checks


def require_clean_trace(trace_hits):
    hits = trace_hits() if trace_hits is not None else []
    if hits:
        raise Refused(9, f"forbidden file access before a completion record: {hits}")


def train_arm(ctx, inputs, seed, out, epochs, device, stop_epoch_steps=None, expect=None, collate=None, trace_hits=None):
    from v9.n2.train import WeightedStudent

    class Student(WeightedStudent):
        def validate(self, host):              # runs before each epoch's checkpoints and before done.json
            result = super().validate(host)
            require_clean_trace(trace_hits)
            return result

    spec = protocol()
    student = Student(ctx["cfg"], inputs["rows"], inputs["weights"], collate or ctx["collate"], ctx["dev_images"],
                              ctx["dev_captions"], out, seed, spec["lr_mult"], device=device, weight_lock=inputs["lock"],
                              formal=True, loss_scale=spec["loss_scale"], stop_epoch_steps=stop_epoch_steps)
    checks = pool_checks(student, inputs["rows"], inputs["weights"], seed, ctx["cfg"], expect)
    if not checks["ok"]:
        raise Refused(4, f"pool checks failed: {checks}")
    return checks, student.fit(epochs, log_every=20)


def log_checks(out, epochs, steps_per_epoch):
    log, done = json.loads((out / "train_log.json").read_text()), json.loads((out / "done.json").read_text())
    eps = log["epochs"]
    best = {"rsum": -1.0, "epoch": -1}
    for r in eps:                                       # strictly greater keeps the earliest of equal epochs
        if r["dev"]["rsum"] > best["rsum"]:
            best = {"rsum": r["dev"]["rsum"], "epoch": r["epoch"]}
    checks = {"epochs_complete": [r["epoch"] for r in eps] == list(range(epochs)),
              "every_epoch_full": all(r["steps_in_epoch"] == steps_per_epoch for r in eps),
              "updates": sum(r["steps_in_epoch"] for r in eps),
              "finite": all(math.isfinite(r["dev"]["rsum"]) and math.isfinite(r["mean_loss"]) for r in eps),
              "best_is_earliest_max_and_matches_done": done["best"] == best, "best": best}
    checks["ok"] = all(checks[k] for k in ("epochs_complete", "every_epoch_full", "finite", "best_is_earliest_max_and_matches_done")) \
        and checks["updates"] == epochs * steps_per_epoch
    return checks, done


# ------------------------------------------------------------------------------------------- commands
def accept(root, seed, steps, attempt_dir, trace_hits=None):
    """Small real-stack check on one seed: both arms, `steps` updates of epoch 0, then one full DEV pass."""
    spec = protocol()
    gate, verified = read_gate(root, need_acceptance=False)
    precision = configure_precision(spec["precision"])
    import torch
    if not torch.cuda.is_available():
        raise Refused(4, "an allocated CUDA device is required")
    ctx = load_context()
    report = {"schema": "v9-n2p-c1-acceptance-v1", "utc": utc(), "seed": seed, "steps_per_arm": steps, "gate": gate,
              "environment": environment(), "precision_effective": precision, "arms": {}}
    for arm in spec["schedule"][str(seed)]:
        out = attempt_dir / f"s{seed}__{arm}"
        if out.exists():
            raise Refused(4, f"{out} exists; acceptance always starts from an empty directory")
        inputs, batches = arm_inputs(root, seed, arm, verified), []

        def recording_collate(rows, real=ctx["collate"]):
            if len(batches) < steps:
                batches.append(hashlib.sha256("\n".join(r["pair_id"] for r in rows).encode()).hexdigest())
            return real(rows)

        t0 = time.time()
        checks, _ = train_arm(ctx, inputs, seed, out, 1, "cuda", stop_epoch_steps=steps,
                              expect={**PROTOCOL_POOL, "updates": spec["updates"]}, collate=recording_collate,
                              trace_hits=trace_hits)
        epoch0 = json.loads((out / "train_log.json").read_text())["epochs"][0]
        report["arms"][arm] = {"handoff_receipt": inputs["receipt"], "pool_checks": checks,
                               "initial_model_sha256": initial_model_sha256(ctx["cfg"], seed),
                               "batch_sha256_first_steps": batches, "steps_run": epoch0["steps_in_epoch"],
                               "mean_loss": epoch0["mean_loss"], "dev": epoch0["dev"],
                               "checkpoints_written": (out / "best.pt").is_file() and (out / "last.pt").is_file(),
                               "seconds": round(time.time() - t0, 1)}
    a, b = (report["arms"][x] for x in spec["schedule"][str(seed)])
    report["paired"] = {"same_initial_model": a["initial_model_sha256"] == b["initial_model_sha256"],
                        "same_batches_first_steps": a["batch_sha256_first_steps"] == b["batch_sha256_first_steps"]
                        and len(a["batch_sha256_first_steps"]) == steps,
                        "same_data_order_all_epochs": a["pool_checks"]["data_order_sha256"] == b["pool_checks"]["data_order_sha256"],
                        "weights_differ": a["handoff_receipt"]["weight_sha256"] != b["handoff_receipt"]["weight_sha256"]}
    report["checks_pass"] = all(report["paired"].values()) and all(
        x["pool_checks"]["ok"] and x["steps_run"] == steps and math.isfinite(x["mean_loss"])
        and math.isfinite(x["dev"]["rsum"]) and x["checkpoints_written"] for x in (a, b))
    return report


def run_seed(root, seed, trace_hits=None):
    spec = protocol()
    gate, verified = read_gate(root, need_acceptance=True)
    precision = configure_precision(spec["precision"])
    import torch
    if not torch.cuda.is_available():
        raise Refused(4, "an allocated CUDA device is required")
    env, runs = environment(), root / "c1" / "runs"
    check_paired_environment(runs, seed, env)
    ctx, results = load_context(), {}
    cfg = ctx["cfg"]
    init_sha = initial_model_sha256(cfg, seed)
    opt_conf, opt_sha = optimizer_config(cfg, spec["lr_mult"])
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    for order, arm in enumerate(spec["schedule"][str(seed)]):
        out = runs / f"s{seed}__{arm}"
        if (out / "unit_result.json").is_file():
            results[arm] = json.loads((out / "unit_result.json").read_text())["status"]
            continue
        inputs = arm_inputs(root, seed, arm, verified)
        t0 = time.time()
        attempt_log(out, {"event": "start", "arm": arm, "environment": env, "precision_effective": precision,
                          "resumes_from_last_pt": (out / "last.pt").is_file()})
        try:
            checks, _ = train_arm(ctx, inputs, seed, out, cfg.num_epochs, "cuda", expect={**PROTOCOL_POOL, "updates": spec["updates"]},
                                  trace_hits=trace_hits)
            post, done = log_checks(out, cfg.num_epochs, PROTOCOL_POOL["steps_per_epoch"])
            if not post["ok"]:
                raise Refused(4, f"s{seed}/{arm}: completed log fails the checks {post}")
            require_clean_trace(trace_hits)                     # unit_result COMPLETE only after a clean trace
        except BaseException as exc:
            status = "INTERRUPTED" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "FAILED"
            attempt_log(out, {"event": "end", "status": status, "elapsed_s": round(time.time() - t0, 1), "error": repr(exc)})
            raise
        attempt_log(out, {"event": "end", "status": "COMPLETE", "elapsed_s": round(time.time() - t0, 1)})
        backend = {**precision, **{k: env[k] for k in PAIRED_KEYS}}
        write_json(out / "unit_result.json", {
            "schema": "v9-n2p-c1-unit-v1", "unit_id": f"V9N2P_f30k_r04_s{seed}_{arm}", "seed": seed, "arm": arm,
            "critic": ARMS[arm], "order": order, "status": "COMPLETE_VERIFIED", "gate": gate, "inputs": inputs["paths"],
            "handoff_receipt": inputs["receipt"], "weight_lock_sha256": sha256(inputs["paths"]["weights"] + ".lock.json"),
            "initial_model_sha256": init_sha, "data_order_sha256": checks["data_order_sha256"],
            "optimizer": opt_conf, "optimizer_sha256": opt_sha, "backend": backend, "backend_sha256": canonical(backend),
            "environment": env, "pool_checks": checks, "log_checks": post, "done": done,
            "artifacts_sha256": {n: sha256(out / n) for n in ("best.pt", "last.pt", "train_log.json", "run_manifest.json", "done.json")},
            "recipe": {k: spec[k] for k in ("model", "epochs", "warmup_epochs", "batch_size", "learning_rate", "lr_mult", "lr_update",
                                            "grad_clip", "margin", "optimizer", "loss_scale", "normalizer", "weights_from_step",
                                            "negative_pool", "updates", "primary")},
            "source": {"repo_head": head, "adapter_sha256": sha256(__file__), "train_py_sha256": sha256(REPO / "src/v9/n2/train.py")},
            "test_read": False})
        results[arm] = "COMPLETE_VERIFIED"
    return results


PAIR_VALIDITY = ("same_initial_model", "same_data_order", "same_backend", "same_gpu")


def rich_gate(units, pairs):
    """ROUND_PROTOCOL v9.analysis.new_rich_gate (investment signal only).  Computed only from three complete,
    fully paired seeds whose severe-collapse status is known for all six units; otherwise null with a status."""
    deltas = [p["vcs_minus_bce_best_rsum"] for p in pairs]
    collapse = {arm: sum(units.get(f"s{s}__{arm}", {}).get("severe_collapse") == "YES" for s in SEEDS) for arm in ARMS}
    unknown = sorted(f"s{s}__{a}" for s in SEEDS for a in ARMS if units.get(f"s{s}__{a}", {}).get("severe_collapse") not in ("YES", "NO"))
    invalid = [p["seed"] for p in pairs if not all(p.get(k) is True for k in PAIR_VALIDITY)]
    gate = {"complete_pairs": len(pairs), "invalid_pairs": invalid,
            "mean_vcs_minus_bce_best_rsum": sum(deltas) / len(deltas) if deltas else None,
            "positive_seeds": sum(d > 0 for d in deltas), "severe_collapse_yes": collapse,
            "severe_collapse_unknown": len(unknown), "severe_collapse_unknown_units": unknown, "investment_signal": None}
    if invalid:
        gate["status"] = "PAIRING_INVALID"
    elif len(pairs) < 3 or unknown:
        gate["status"] = "INCOMPLETE"
    else:
        gate["status"] = "COMPUTED"
        gate["investment_signal"] = gate["mean_vcs_minus_bce_best_rsum"] >= 3.0 and gate["positive_seeds"] >= 2 \
            and collapse["RICH_VCS"] <= collapse["RICH_BCE"]
    return gate


def summarize(root):
    sys.path.insert(0, str(N2M_SRC))
    from v9n2m.aggregate import collapse_diagnostic        # the frozen N2F/N2M severe-collapse rule
    spec, units, pairs, epochs_csv = protocol(), {}, [], []
    for seed in SEEDS:
        for arm in ARMS:
            out = root / "c1" / "runs" / f"s{seed}__{arm}"
            key = f"s{seed}__{arm}"
            if not (out / "unit_result.json").is_file():
                units[key] = {"seed": seed, "arm": arm, "status": "NOT_COMPLETE"}
                continue
            res, log = json.loads((out / "unit_result.json").read_text()), json.loads((out / "train_log.json").read_text())
            eps = log["epochs"]
            rs = [e["dev"]["rsum"] for e in eps]
            b = max(range(len(rs)), key=lambda i: (rs[i], -i))
            attempts = [json.loads(x) for x in (out / "attempts.jsonl").read_text().splitlines()]
            ends = [a for a in attempts if a["event"] == "end"]
            units[key] = {"seed": seed, "arm": arm, "status": res["status"], "best_epoch": eps[b]["epoch"], "best_rsum": rs[b],
                          "best_mR": rs[b] / 6, **{k: eps[b]["dev"][k] for k in RECALLS}, "last_rsum": rs[-1],
                          "mean_last5_rsum": sum(rs[-5:]) / 5, "severe_collapse": collapse_diagnostic(log)["status"],
                          "gpu": res["environment"]["gpu"], "host": res["environment"]["host"],
                          "attempts": sum(a["event"] == "start" for a in attempts),
                          "attempt_seconds_known": sum(a["elapsed_s"] for a in ends),
                          "attempts_without_end_record": sum(a["event"] == "start" for a in attempts) - len(ends),
                          "initial_model_sha256": res["initial_model_sha256"], "data_order_sha256": res["data_order_sha256"],
                          "weight_sha256": res["handoff_receipt"]["weight_sha256"], "backend_sha256": res["backend_sha256"],
                          "optimizer_sha256": res["optimizer_sha256"]}
            epochs_csv += [{"seed": seed, "arm": arm, "epoch": e["epoch"], **{k: e["dev"][k] for k in RECALLS}, "rsum": e["dev"]["rsum"]} for e in eps]
        v, c = units.get(f"s{seed}__RICH_VCS", {}), units.get(f"s{seed}__RICH_BCE", {})
        if v.get("status") == c.get("status") == "COMPLETE_VERIFIED":
            pairs.append({"seed": seed, "same_initial_model": v["initial_model_sha256"] == c["initial_model_sha256"],
                          "same_data_order": v["data_order_sha256"] == c["data_order_sha256"],
                          "same_backend": v["backend_sha256"] == c["backend_sha256"], "same_gpu": v["gpu"] == c["gpu"],
                          "vcs_minus_bce_best_rsum": v["best_rsum"] - c["best_rsum"],
                          **{f"vcs_minus_bce_{k}": v[k] - c[k] for k in RECALLS},
                          "vcs_minus_bce_last_rsum": v["last_rsum"] - c["last_rsum"],
                          "vcs_minus_bce_mean_last5_rsum": v["mean_last5_rsum"] - c["mean_last5_rsum"]})
    gate = rich_gate(units, pairs)
    summary = {"schema": "v9-n2p-c1-summary-v1", "utc": utc(), "primary": spec["primary"], "units": units, "pairs": pairs,
               "new_rich_gate": gate, "gate_rule": json.loads((PKG / "config" / "ROUND_PROTOCOL.json").read_text())["v9"]["analysis"]["new_rich_gate"],
               "unit_of_analysis": "seed (three paired seeds); DEV is developmental; no TEST read",
               "not_a_significance_test": True}
    write_json(root / "c1" / "C1_SUMMARY.json", summary)
    with open(root / "c1" / "C1_EPOCHS.csv", "w") as f:
        cols = ["seed", "arm", "epoch", *RECALLS, "rsum"]
        f.write(",".join(cols) + "\n")
        for r in epochs_csv:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("accept", "run", "summarize"))
    ap.add_argument("--root", required=True, help="N2P C output root (contains c0/ and c1/)")
    ap.add_argument("--seed", type=int, choices=SEEDS, default=901)
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.command == "summarize":
        print(json.dumps(summarize(root)["new_rich_gate"]))
        return 0
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))   # time limit: keep records
    finish = trace_open.install("c1")
    attempt_id = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}_{int(time.time())}"
    trace_path = root / "c1" / "traces" / f"{a.command}_s{a.seed}_{attempt_id}.json"
    attempt_dir = root / "c1" / "acceptance" / attempt_id
    code, result, report = 0, None, None
    try:
        if a.command == "accept":
            report = accept(root, a.seed, a.steps, attempt_dir, finish.forbidden_hits)
        else:
            result = run_seed(root, a.seed, finish.forbidden_hits)
    except Refused as exc:
        code, result = exc.code, str(exc)
        print(f"REFUSED({exc.code}): {exc}", file=sys.stderr)
    finally:
        trace = finish(trace_path, {"command": a.command, "seed": a.seed, "result": result})
    if trace["status"] != "PASS":
        print(f"FORBIDDEN_ACCESS: {trace['forbidden_hits']}", file=sys.stderr)
        code = 9
    if report is not None:
        report["read_evidence"] = {"trace": str(trace_path), "status": trace["status"]}
        report["status"] = "PASS" if report["checks_pass"] and trace["status"] == "PASS" else "FAIL"
        write_json(attempt_dir / "C1_ACCEPTANCE.json", report, overwrite=False)
        if report["status"] == "PASS" and not (root / "c1" / "C1_ACCEPTANCE.json").exists():
            write_json(root / "c1" / "C1_ACCEPTANCE.json", report, overwrite=False)      # the gate file for `run`
        print(json.dumps({"acceptance": report["status"], "paired": report["paired"], "report": str(attempt_dir)}))
        code = code or (0 if report["status"] == "PASS" else 4)
    elif code == 0:
        print(json.dumps({"seed": a.seed, "units": result}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
