"""N2P read guard: run a Python script under an audit hook that writes every opened file (outside the interpreter's own prefix)
to a JSONL trace and refuses, by raising, to open
  * any file of the dataset directory whose name starts with "test" (Flickr TEST / testall),
  * any EVAL_ONLY label file (path containing "eval_only"),
  * under policy "dev": any dataset file that is not a dev_* file (DEV allowlist for the evaluation process).

    python -I -u n2p_guard.py <trace.jsonl> <train|dev> <real dataset dir> <script.py> [args...]
"""
import json
import os
import runpy
import sys
import time


def main():
    trace, policy, data_dir, script, *args = sys.argv[1:]
    if policy not in ("train", "dev"):
        raise SystemExit("policy must be train or dev")
    real_data = os.path.realpath(data_dir) + os.sep
    fd = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    skip = tuple(sorted({os.path.realpath(sys.prefix) + os.sep, os.path.realpath(sys.base_prefix) + os.sep}))
    busy = [False]

    def hook(event, a):
        if event != "open" or busy[0] or not isinstance(a[0], (str, bytes, os.PathLike)):
            return
        busy[0] = True
        try:
            path = os.fsdecode(a[0])
            real = os.path.realpath(path)
            if real.startswith(skip):
                return
            base = os.path.basename(real)
            in_data = real.startswith(real_data)
            deny = None
            if in_data and base.startswith("test"):
                deny = "TEST split file"
            elif "eval_only" in real:
                deny = "EVAL_ONLY label file"
            elif policy == "dev" and in_data and not base.startswith("dev_"):
                deny = "non-DEV dataset file under the DEV-only policy"
            rec = {"t": round(time.time(), 3), "pid": os.getpid(), "path": path, "real": real, "mode": a[1],
                   "dataset_file": in_data, "decision": "DENY" if deny else "allow"}
            if deny:
                rec["reason"] = deny
            os.write(fd, (json.dumps(rec) + "\n").encode())
            if deny:
                raise PermissionError("N2P read guard refused %s: %s" % (deny, real))
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    if os.environ.get("N2P_CPU_SMOKE") == "1":          # CPU fixture smoke only (tests/cpu_smoke.py): .cuda() becomes a no-op
        import torch
        import torch.utils.data.dataloader as dl
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self
        nxt = dl._BaseDataLoaderIter.__next__            # on GPU, pin_memory turns the collated tuples into lists; mimic it
        dl._BaseDataLoaderIter.__next__ = lambda self: (lambda b: [list(x) if isinstance(x, tuple) else x for x in b]
                                                        if isinstance(b, (tuple, list)) else b)(nxt(self))
    sys.argv = [script, *args]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
