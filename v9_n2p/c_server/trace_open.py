"""Read evidence for N2P C0/C1: run a Python script unchanged and record the files it opens.

    python -I -B trace_open.py --out TRACE.json --stage c0|c1 -- SCRIPT [ARGS...]

The script runs as `python SCRIPT ARGS` would (its own directory first on sys.path), so the
package's sibling imports work under -I.  A Python audit hook records every `open` event
(builtins.open, io.open, os.open).  Files under the interpreter prefixes and system roots are
only counted; every other path is listed.  Stage c0 (extraction, fit, export, handoff, decision)
forbids DEV and TEST split files, eval_only labels, noise manifests/indices and precheck records;
stage c1 (students) allows DEV.  A forbidden path turns the exit status into 9.  Scope:
Python-level opens only.  This server has no mount/permission separation between data roles,
so this trace is the read evidence.
"""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
import time
import traceback

_SPLIT = r"(^|/)({})_(ims\.npy|caps\.txt|ids\.txt|tags\.txt)$"
COMMON_FORBIDDEN = (r"eval[_-]only", r"noise_manifest", r"noise_index", r"/precheck/", r"testall",
                    _SPLIT.format("test"), r"_test_caps_tokens_")
STAGE_FORBIDDEN = {"c0": COMMON_FORBIDDEN + (_SPLIT.format("dev"), r"_dev_caps_tokens_"),
                   "c1": COMMON_FORBIDDEN}


def install(stage):
    """Start recording open events; returns finish(out_path, extra) -> trace dict (also written)."""
    if stage not in STAGE_FORBIDDEN:
        raise ValueError(f"unknown stage {stage}")
    opened, state = {}, {"on": True}

    def hook(event, args):
        if event != "open" or not state["on"] or not args or isinstance(args[0], int):
            return
        try:
            path = os.fsdecode(args[0])
        except TypeError:
            return
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        mode = args[1] if isinstance(args[1], str) else f"flags={args[2]}"
        opened.setdefault(path, set()).add(mode)

    sys.addaudithook(hook)
    started = time.time()

    def finish(out_path, extra=None):
        state["on"] = False
        roots = sorted({os.path.realpath(p).rstrip("/") + "/" for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
                       | {"/usr/", "/lib/", "/lib64/", "/proc/", "/sys/", "/dev/", "/etc/"})
        env_counts, listed, hits = {}, [], []
        for raw in sorted(opened):
            real = os.path.realpath(raw)
            root = next((r for r in roots if real.startswith(r)), None)
            if root is not None:
                env_counts[root] = env_counts.get(root, 0) + 1
                continue
            listed.append({"path": real, "modes": sorted(opened[raw])})
            if any(re.search(p, real) for p in STAGE_FORBIDDEN[stage]):
                hits.append(real)
        trace = {"schema": "n2p-c-open-trace-v1", "stage": stage, "forbidden_patterns": list(STAGE_FORBIDDEN[stage]),
                 "status": "PASS" if not hits else "FORBIDDEN_ACCESS", "forbidden_hits": hits,
                 "opened": listed, "environment_files_opened": env_counts,
                 "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
                 "python": sys.version.split()[0], "seconds": round(time.time() - started, 3),
                 "scope": "Python audit 'open' events, including attempted opens that fail (e.g. absent .pyc caches); native reads outside Python file APIs are not observed",
                 **(extra or {})}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(trace, handle, indent=1)
        return trace

    return finish


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    sep = argv.index("--")
    options, target = argv[:sep], argv[sep + 1:]
    out = options[options.index("--out") + 1]
    stage = options[options.index("--stage") + 1]
    if os.path.exists(out):
        raise SystemExit(f"refusing to replace trace {out}")
    script = os.path.abspath(target[0])
    finish = install(stage)
    sys.argv = [script] + target[1:]
    sys.path.insert(0, os.path.dirname(script))
    code, error = 0, None
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:            # the script's own failure is kept and reported
        code, error = 1, f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    trace = finish(out, {"script": script, "argv": sys.argv[1:], "exit_code": code, "error": error})
    if trace["status"] != "PASS":
        print(f"FORBIDDEN_ACCESS: {trace['forbidden_hits']}", file=sys.stderr)
        code = 9
    raise SystemExit(code)


if __name__ == "__main__":
    main()
