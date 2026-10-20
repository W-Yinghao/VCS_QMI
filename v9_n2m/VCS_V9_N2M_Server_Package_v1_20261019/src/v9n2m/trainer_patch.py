"""Pinned, minimal telemetry changes to the executed V9 N2F trainer.

This module has no Torch dependency. It never edits a baseline in place. The
caller must write the returned source into a separate, identified runtime.
"""
from __future__ import annotations

import hashlib

BASELINE_TRAIN_SHA256 = "cafac01f989b0bbdd88f4d21c51e7475227367d8d4750057897b77832a61302c"
PATCH_VERSION = "V9-N2M-telemetry-v1"

# Kept as literal source so structural and bookkeeping tests do not need Torch.
_PATCH_HELPERS = '''
# V9-N2M telemetry v1: the objective, optimiser, batches and RNG are unchanged.
def _n2m_completed_updates(log):
    total = 0
    for expected_epoch, rec in enumerate(log.get("epochs", [])):
        if rec.get("epoch") != expected_epoch:
            raise ValueError("checkpoint epochs are not contiguous from zero")
        count = rec.get("steps_in_epoch")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("checkpoint lacks a positive integer steps_in_epoch")
        total += count
        if "global_step" in rec and rec["global_step"] != total:
            raise ValueError("epoch global_step disagrees with completed updates")
    return total


def _n2m_atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1), encoding="utf-8")
    temporary.replace(path)


def _n2m_interrupted(out, best, log, target_epochs, global_step):
    _n2m_atomic_json(out / "train_log.json", log)
    return {"status": "INTERRUPTED", "best": best,
            "epochs_run": len(log["epochs"]), "target_epochs": target_epochs,
            "global_step": global_step, "reason": "STOP_REQUESTED at epoch boundary"}


def _n2m_validate_resume_identity(out, checkpoint):
    identity = out / "identity.json"
    manifest = out / "run_manifest.json"
    saved_identity = checkpoint.get("execution_identity_sha256")
    saved_manifest = checkpoint.get("run_manifest_sha256")
    if identity.is_file():
        identity_hash = hashlib.sha256(identity.read_bytes()).hexdigest()
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if saved_identity != identity_hash:
            raise ValueError("checkpoint execution identity is missing or differs from this unit")
        if saved_manifest != manifest_hash:
            raise ValueError("checkpoint run manifest is missing or differs from this unit")
    elif saved_identity is not None:
        raise ValueError("identified checkpoint has lost its execution identity file")
    elif saved_manifest is not None and saved_manifest != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("checkpoint run manifest differs from this diagnostic fixture")
    # A legacy checkpoint without either field is accepted only when there is
    # no identity.json. The production runner always supplies identity.json;
    # this exception is used solely by the synthetic legacy-resume check.


def _n2m_read_completed(out, epochs):
    done = json.loads((out / "done.json").read_text())
    required = ("train_log.json", "best.pt", "last.pt")
    if any(not (out / name).is_file() for name in required):
        raise ValueError("done.json exists without required completion artifacts")
    log = json.loads((out / "train_log.json").read_text())
    count = _n2m_completed_updates(log)
    if done.get("epochs_run") != epochs or len(log["epochs"]) != epochs:
        raise ValueError("existing completion does not match requested epoch count")
    if "global_step" in done and done["global_step"] != count:
        raise ValueError("done.json global_step disagrees with completed updates")
    best = {"rsum": -1.0, "epoch": -1}
    for rec in log["epochs"]:
        value = rec["dev"]["rsum"]
        if not (-float("inf") < value < float("inf")):
            raise ValueError("completed log contains a nonfinite DEV rSum")
        if value > best["rsum"]:
            best = {"rsum": value, "epoch": rec["epoch"]}
    if done.get("best") != best:
        raise ValueError("done.json best does not match full epoch log")
    return done


'''


def _once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Pinned patch hunk occurrence mismatch ({count}): {old[:96]!r}")
    return text.replace(old, new, 1)


def patch_train_source(source_text: str) -> str:
    """Return patched source; refuse any source except the executed frozen file."""
    actual = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if actual != BASELINE_TRAIN_SHA256:
        raise ValueError(
            "Refusing to patch an unrecognised trainer: "
            f"expected {BASELINE_TRAIN_SHA256}, received {actual}"
        )
    text = _once(source_text, "class WeightedStudent:\n", _PATCH_HELPERS + "class WeightedStudent:\n")
    text = _once(text, "if np.any(v < 0) or np.any(v > 1):", "if not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1):")
    text = _once(
        text,
        '        self.manifest = {"n_pool": len(pool_rows)',
        '        if not np.isfinite(self.mean_v_full) or self.mean_v_full <= 0:\n'
        '            raise ValueError("the fixed full-pool normaliser must be finite and positive")\n'
        '        self.manifest = {"n_pool": len(pool_rows)',
    )
    text = _once(
        text,
        '"epoch": epoch, "best": best, "log": log, "rng": _rng_state()}, self.out / f"{name}.tmp")',
        '"epoch": epoch, "best": best, "log": log, "rng": _rng_state(), '
        '"global_step": _n2m_completed_updates(log), "telemetry_patch": "V9-N2M-telemetry-v1", '
        '"execution_identity_sha256": (hashlib.sha256((self.out / "identity.json").read_bytes()).hexdigest() if (self.out / "identity.json").is_file() else None), '
        '"run_manifest_sha256": hashlib.sha256((self.out / "run_manifest.json").read_bytes()).hexdigest()}, self.out / f"{name}.tmp")',
    )
    text = _once(
        text,
        '        return recall_metrics(sims, 5, "stable")',
        '        if not np.isfinite(sims).all():\n'
        '            raise FloatingPointError("nonfinite validation similarity; refusing a misleading rank result")\n'
        '        return recall_metrics(sims, 5, "stable")',
    )
    text = _once(
        text,
        '        if (self.out / "done.json").exists():\n'
        '            return json.loads((self.out / "done.json").read_text())\n'
        '        cfg = self.cfg; epochs = cfg.num_epochs if epochs is None else epochs',
        '        cfg = self.cfg; epochs = cfg.num_epochs if epochs is None else epochs\n'
        '        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:\n'
        '            raise ValueError("epochs must be a positive integer")\n'
        '        if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every <= 0:\n'
        '            raise ValueError("log_every must be a positive integer")\n'
        '        if (self.out / "done.json").exists():\n'
        '            return _n2m_read_completed(self.out, epochs)',
    )
    text = _once(
        text,
        '            ck = torch.load(self.out / "last.pt", map_location="cpu", weights_only=False)\n',
        '            ck = torch.load(self.out / "last.pt", map_location="cpu", weights_only=False)\n'
        '            _n2m_validate_resume_identity(self.out, ck)\n',
    )
    text = _once(
        text,
        '        t0 = time.time(); step = len(log["steps"])',
        '        step = _n2m_completed_updates(log)\n'
        '        if start != len(log["epochs"]) or start > epochs:\n'
        '            raise ValueError("checkpoint epoch does not agree with the requested complete history")\n'
        '        if (self.out / "last.pt").exists() and ck.get("global_step", step) != step:\n'
        '            raise ValueError("checkpoint global_step disagrees with completed epoch updates")\n'
        '        t0 = time.time()\n'
        '        if start < epochs and (self.out / "STOP_REQUESTED").exists():\n'
        '            return _n2m_interrupted(self.out, best, log, epochs, step)',
    )
    text = _once(
        text,
        '                opt.zero_grad(set_to_none=True); loss.backward()',
        '                if not bool(torch.isfinite(loss)):\n'
        '                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}, update {in_epoch + 1}")\n'
        '                opt.zero_grad(set_to_none=True); loss.backward()',
    )
    text = _once(
        text,
        '                opt.step(); step += 1; in_epoch += 1',
        '                if gn_t is not None and not bool(torch.isfinite(gn_t)):\n'
        '                    raise FloatingPointError(f"nonfinite gradient norm at epoch {epoch}, update {in_epoch + 1}")\n'
        '                if gn_t is None and any(q.grad is not None and not bool(torch.isfinite(q.grad).all()) for q in host.parameters()):\n'
        '                    raise FloatingPointError(f"nonfinite gradient at epoch {epoch}, update {in_epoch + 1}")\n'
        '                opt.step(); step += 1; in_epoch += 1',
    )
    text = _once(
        text,
        'vb = self.v[idx]; log["steps"].append({"step": step, "epoch": epoch, "loss": float(loss)',
        'vb = self.v[idx]; log["steps"].append({"step": step, "global_step": step, "epoch": epoch, "step_in_epoch": in_epoch, "loss": float(loss)',
    )
    text = _once(
        text,
        '"steps_in_epoch": in_epoch, "mean_loss": agg["loss"]',
        '"steps_in_epoch": in_epoch, "global_step": step, "mean_loss": agg["loss"]',
    )
    text = _once(
        text,
        '            self._save(host, opt, epoch, best, log, name="last.pt")\n'
        '        done = {"best": best, "epochs_run": epochs,',
        '            self._save(host, opt, epoch, best, log, name="last.pt")\n'
        '            if epoch + 1 < epochs and (self.out / "STOP_REQUESTED").exists():\n'
        '                return _n2m_interrupted(self.out, best, log, epochs, step)\n'
        '        done = {"best": best, "epochs_run": epochs, "global_step": step, "telemetry_patch": "V9-N2M-telemetry-v1",',
    )
    text = _once(
        text,
        '        (self.out / "train_log.json").write_text(json.dumps(log, indent=1)); (self.out / "done.json").write_text(json.dumps(done, indent=1))',
        '        _n2m_atomic_json(self.out / "train_log.json", log); _n2m_atomic_json(self.out / "done.json", done)',
    )
    compile(text, "<V9-N2M-patched-train.py>", "exec")
    return text


def telemetry_patch_identity() -> dict:
    return {
        "version": PATCH_VERSION,
        "baseline_train_sha256": BASELINE_TRAIN_SHA256,
        "scope": [
            "actual completed update counts and epoch-relative sparse logs",
            "atomic train-log and completion JSON writes",
            "epoch-boundary STOP_REQUESTED without premature completion",
            "checkpoint binding to this execution identity and run manifest",
            "nonfinite refusal and existing-completion integrity checks",
        ],
        "scientific_loop_changes": False,
        "gpu_equivalence": "requires the separately recorded verify_runtime result",
    }
