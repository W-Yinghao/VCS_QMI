"""Two CoMIR trainers in one process (CPU): the frozen check refuses the second (v5); comir_check_v6 lets it pass."""
import json, sys, time, types
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts/v10_test_repeat"))
import tr_common as C
import evaluate as E
man = C.read(repo / C.RECORDS / C.MANIFEST)
u = C.unit(man, "CoMIR_native__1001")
ident = u["identity"]
run, source = E.load_source(u)
dg = run.A.DataGuard()
new = lambda: run.new_trainer(ident["family"], ident["candidate"], ident["seed"], dg, "cpu")
out = {}
t0 = time.time(); new(); out["first_trainer"] = "built"
try:
    new(); out["second_trainer_v5"] = "built"
except RuntimeError as e:
    out["second_trainer_v5"] = str(e)
mod = sys.modules[E.COMIR_MODULE]
ns = {n: getattr(m, "__file__", None) for n, m in sys.modules.items() if n == "models" or n.startswith("models.")}
out["models_modules"] = {n: (f is None) for n, f in sorted(ns.items())}
E.comir_check_v6(mod)
new(); out["second_trainer_v6"] = "built"
before = dict(sys.modules)
def probe(name, **attrs):
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]; sys.modules[name] = m
    try:
        mod.verify_official_source(); r = "passed"
    except RuntimeError as e:
        r = str(e)
    restored = {k: v for k, v in sys.modules.items()} == {**before, name: m}
    del sys.modules[name]
    return r, restored
out["namespace_outside_root"] = probe("models.fake_ns", __path__=["/tmp"])
out["namespace_without_path"] = probe("models.fake_ns2")
out["regular_outside_root"] = probe("models.fake_mod", __file__="/tmp/fake_mod.py")
mod.verify_official_source(); out["sys_modules_restored_after_pass"] = dict(sys.modules) == before
out["seconds"] = round(time.time() - t0, 1)
print("RESULT " + json.dumps(out))
