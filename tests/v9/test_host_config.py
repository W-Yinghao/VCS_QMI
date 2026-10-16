"""Strict machine host lock (review item U03): reports/V9/HOST_LOCK_MACHINE.yaml read by src/v9/host_config.py.

* the real machine file loads for f30k_precomp and equals, field by field (value AND type), what the LEGACY loader
  v9.n2.teacher.load_host_lock resolves from the real reports/V9/HOST_LOCK.yaml; every provenance source is found in that file;
* coco_precomp (status not_locked) and absent datasets raise; missing / unknown / wrong-type / out-of-range fields raise; YAML that a typed
  lock must not contain (duplicate keys, yes / True booleans, anchors, merge keys, 2e-4 floats, octal ints, timestamps) raises; no defaults.

The machine file and host_config.py are taken next to this test (<root>/reports/V9, <root>/src/v9).  The legacy loader and the real
HOST_LOCK.yaml are taken from the tree that provides the importable `v9` package: the repo itself once deployed; for a staging copy, the
live repo through PYTHONPATH=<repo>/src (imported read-only, nothing is written there).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
MACHINE = ROOT / "reports/V9/HOST_LOCK_MACHINE.yaml"


def _module_under_test():
    path = (ROOT / "src/v9/host_config.py").resolve()
    try:
        mod = importlib.import_module("v9.host_config")
        if Path(mod.__file__).resolve() == path:
            return mod
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location("v9_host_config_staged", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


HC = _module_under_test()
import v9  # noqa: E402  (the package that provides the legacy loader)

LIVE = Path(v9.__file__).resolve().parents[2]
REAL_LOCK = LIVE / "reports/V9/HOST_LOCK.yaml"
F30K_VOCAB_SIZE = 8481        # len(word2idx) of code_v9/SGRAF_py3/vocab/f30k_precomp_vocab.json: a runtime input, not part of the lock
# "host config:" printed by the running block's jobs (legacy loader on HOST_LOCK.yaml sha256 8be8ced9...), without to_verify / lock_notes
PRINTED_BY_RUNNING_BLOCK = {"img_dim": 2048, "word_dim": 300, "embed_size": 1024, "sim_dim": 256, "num_layers": 1, "bi_gru": True, "num_region": 36,
                            "sgr_step": 3, "margin": 0.2, "learning_rate": 0.0002, "grad_clip": 2.0, "batch_size": 128, "lr_update": 30,
                            "num_epochs": 40, "teacher_epochs": 5, "warmup_epochs": 5}
LOCKED_NAMES = {"module_name": "SGR", "optimizer": "adam", "lr_decay_factor": 0.1, "warmup_loss": "warmup_mean_hinge",
                "main_loss": "max_violation_hinge", "checkpoint_selection": "best_dev_rsum"}
VALUE_FIELDS = [("model", k) for k in HC.MODEL_SPEC] + [("schedule", k) for k in HC.SCHEDULE_SPEC]


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def _doc() -> dict:
    return yaml.safe_load(MACHINE.read_text())


def _dump(tmp_path, doc, name="machine.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000))
    return p


def _edit_text(tmp_path, old, new) -> Path:
    text = MACHINE.read_text()
    assert text.count(old) == 1, old
    p = tmp_path / "machine.yaml"
    p.write_text(text.replace(old, new))
    return p


def _set(doc, path, value):
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return doc


def _del(doc, path):
    node = doc
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    return doc


# ------------------------------------------------------------------------------------------------------------------ the real file
def test_real_machine_file_loads_f30k():
    cfg = HC.load_host_config(MACHINE, "f30k_precomp", source_lock=REAL_LOCK)
    want = {**PRINTED_BY_RUNNING_BLOCK, **LOCKED_NAMES}
    got = cfg.values()
    assert set(got) == set(want) and all(_same(got[k], v) for k, v in want.items()), got
    assert (cfg.dataset, cfg.schema_version, cfg.source_path) == ("f30k_precomp", 1, "reports/V9/HOST_LOCK.yaml")
    assert cfg.source_sha256 == hashlib.sha256(REAL_LOCK.read_bytes()).hexdigest()
    assert {k for k, _ in cfg.provenance} == set(HC.VALUE_KEYS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.num_epochs = 41
    assert all(f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING for f in dataclasses.fields(HC.StrictHostConfig))
    assert HC.read_machine_lock(MACHINE) == _doc()           # the strict YAML layer reads the real file exactly as plain YAML does


def test_equals_the_legacy_loader_on_the_real_host_lock():
    pytest.importorskip("torch")                             # v9.n2.teacher imports torch
    from v9.n2 import teacher as T
    assert Path(T.__file__).resolve().is_relative_to(LIVE / "src")
    legacy = T.load_host_lock(REAL_LOCK, "f30k_precomp", F30K_VOCAB_SIZE)
    cfg = HC.load_host_config(MACHINE, "f30k_precomp", source_lock=REAL_LOCK)
    rep = HC.assert_matches_legacy(cfg, legacy)
    assert rep["equal"] and rep["mismatched"] == [] and rep["legacy_to_verify"] == [] and rep["unmapped_legacy_fields"] == []
    for k in HC.LEGACY_FIELDS:                               # field by field, value and type
        assert _same(getattr(cfg, k), getattr(legacy, k)), (k, getattr(cfg, k), getattr(legacy, k))
    assert {k: getattr(legacy, k) for k in HC.LEGACY_FIELDS} == PRINTED_BY_RUNNING_BLOCK
    built = T.HostConfig(vocab_size=F30K_VOCAB_SIZE, **HC.legacy_kwargs(cfg))
    assert all(_same(getattr(built, k), getattr(legacy, k)) for k in HC.LEGACY_FIELDS) and built.to_verify == []
    assert {f.name for f in dataclasses.fields(T.HostConfig)} == set(HC.LEGACY_FIELDS) | set(HC.LEGACY_BOOKKEEPING)


def test_compare_with_legacy_flags_value_type_pending_and_unmapped_differences():
    cfg = HC.load_host_config(MACHINE, "f30k_precomp")
    base = dict(PRINTED_BY_RUNNING_BLOCK, vocab_size=F30K_VOCAB_SIZE, to_verify=[], lock_notes=[])
    assert HC.compare_with_legacy(cfg, types.SimpleNamespace(**base))["equal"]
    for change, flag in (({"margin": 0.25}, "margin"), ({"bi_gru": 1}, "bi_gru"), ({"batch_size": 128.0}, "batch_size")):
        rep = HC.compare_with_legacy(cfg, types.SimpleNamespace(**{**base, **change}))
        assert not rep["equal"] and rep["mismatched"] == [flag]
    assert HC.compare_with_legacy(cfg, types.SimpleNamespace(**{**base, "to_verify": ["margin"]}))["equal"] is False
    rep = HC.compare_with_legacy(cfg, types.SimpleNamespace(**{**base, "dropout": 0.4}))
    assert not rep["equal"] and rep["unmapped_legacy_fields"] == ["dropout"]
    with pytest.raises(HC.HostConfigError, match="margin: strict 0.2"):
        HC.assert_matches_legacy(cfg, types.SimpleNamespace(**{**base, "margin": 0.25}))


def test_every_value_is_traceable_to_the_real_host_lock():
    rep = HC.verify_provenance(MACHINE, REAL_LOCK)
    assert rep["problems"] == [] and rep["source_sha256"] == rep["recorded_sha256"]
    doc = _doc()
    assert rep["checked_sources"] == sum(len(v) for d in doc["datasets"].values() for v in d["provenance"].values())
    prov = doc["datasets"]["f30k_precomp"]["provenance"]
    for key in HC.VALUE_KEYS:
        assert prov[key], key
        if key.startswith("schedule."):                       # the LOCKED recipe is cited first, the native 'training' prose after it
            assert prov[key][0]["section"][0] in ("locked_recipes", "host_finding_20261015_timing_job_1042033"), key
    for key in ("model.num_layers", "model.bi_gru", "model.num_region"):   # the legacy loader takes these three from code defaults
        assert any(s["section"][0] in ("model (SGR single, SGRAF opts.py defaults)", "features") for s in prov[key]), key


@pytest.mark.parametrize("block,key,value", [("schedule", "num_epochs", 41), ("schedule", "learning_rate", 0.0003), ("model", "bi_gru", False),
                                             ("model", "num_region", 37), ("schedule", "lr_decay_factor", 0.5)])
def test_provenance_catches_a_value_that_is_not_what_host_lock_says(tmp_path, block, key, value):
    d = _set(_doc(), ("datasets", "f30k_precomp", block, key), value)
    rep = HC.verify_provenance(_dump(tmp_path, d), REAL_LOCK)
    assert rep["problems"] and all(f"provenance.{block}.{key}[" in p for p in rep["problems"]), rep["problems"]


def test_provenance_catches_a_wrong_quote_or_section(tmp_path):
    d = _doc()
    prov = d["datasets"]["f30k_precomp"]["provenance"]
    prov["model.sim_dim"][0]["quote"] = "sim_dim: 256 (verified)"
    prov["schedule.margin"][1]["section"] = ["training", "native_margin"]       # the key the legacy dict lookup uses: absent from HOST_LOCK
    problems = HC.verify_provenance(_dump(tmp_path, d), REAL_LOCK)["problems"]
    assert any("model.sim_dim[0]: quote" in p for p in problems) and any("schedule.margin[1]: section" in p for p in problems), problems


def test_a_changed_host_lock_makes_the_machine_lock_stale(tmp_path):
    changed = tmp_path / "HOST_LOCK.yaml"
    changed.write_bytes(REAL_LOCK.read_bytes() + b"# edited after the derivation\n")
    with pytest.raises(HC.StaleMachineLock, match="re-derive"):
        HC.load_host_config(MACHINE, "f30k_precomp", source_lock=changed)
    assert any("derived_from.sha256" in p for p in HC.verify_provenance(MACHINE, changed)["problems"])
    with pytest.raises(HC.HostConfigError, match="cannot read"):
        HC.load_host_config(MACHINE, "f30k_precomp", source_lock=tmp_path / "absent.yaml")


def test_cli_loads_and_verifies(capsys):
    assert HC.main(["--machine", str(MACHINE), "--host-lock", str(REAL_LOCK), "--dataset", "f30k_precomp"]) == 0
    assert '"problems": []' in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------------- datasets not locked
def test_coco_is_not_locked_and_carries_no_values():
    with pytest.raises(HC.DatasetNotLocked, match="status 'not_locked'"):
        HC.load_host_config(MACHINE, "coco_precomp", source_lock=REAL_LOCK)
    assert set(_doc()["datasets"]["coco_precomp"]) == {"status", "reason", "provenance"}


@pytest.mark.parametrize("name", ["cc152k_precomp", "f30k", "coco"])
def test_absent_dataset_raises(name):
    with pytest.raises(HC.HostConfigError, match="is not in the machine lock"):
        HC.load_host_config(MACHINE, name)


def test_a_not_locked_dataset_must_not_carry_values_and_cannot_be_flipped_to_locked(tmp_path):
    d = _doc()
    d["datasets"]["coco_precomp"]["schedule"] = dict(d["datasets"]["f30k_precomp"]["schedule"], num_epochs=20, lr_update=10)
    with pytest.raises(HC.HostConfigError, match=r"coco_precomp: unknown key\(s\) \['schedule'\]"):
        HC.load_host_config(_dump(tmp_path, d), "f30k_precomp")            # the whole file is validated, whatever dataset is asked for
    d = _set(_doc(), ("datasets", "coco_precomp", "status"), "locked")
    with pytest.raises(HC.HostConfigError, match="coco_precomp: missing key"):
        HC.load_host_config(_dump(tmp_path, d, "flip.yaml"), "coco_precomp")


# -------------------------------------------------------------------------------------------------------------------- schema
@pytest.mark.parametrize("block,key", VALUE_FIELDS)
def test_missing_value_field_raises(tmp_path, block, key):
    d = _del(_doc(), ("datasets", "f30k_precomp", block, key))
    with pytest.raises(HC.HostConfigError, match=rf"datasets\.f30k_precomp\.{block}: missing key\(s\) \['{key}'\]"):
        HC.load_host_config(_dump(tmp_path, d), "f30k_precomp")


@pytest.mark.parametrize("path", [("schema_version",), ("derived_from",), ("datasets",), ("derived_from", "sha256"), ("derived_from", "git_commit"),
                                  ("datasets", "f30k_precomp", "status"), ("datasets", "f30k_precomp", "provenance"),
                                  ("datasets", "f30k_precomp", "provenance", "schedule.margin"), ("datasets", "f30k_precomp", "provenance", "model.bi_gru", 0, "token"),
                                  ("datasets", "coco_precomp", "reason")])
def test_missing_structural_key_raises(tmp_path, path):
    with pytest.raises(HC.HostConfigError, match="missing key"):
        HC.load_host_config(_dump(tmp_path, _del(_doc(), path)), "f30k_precomp")


@pytest.mark.parametrize("path,value", [(("notes",), "free text"), (("derived_from", "derived_on"), "2026-10-16"), (("datasets", "f30k_precomp", "notes"), "x"),
                                        (("datasets", "f30k_precomp", "model", "dropout"), 0.4), (("datasets", "f30k_precomp", "schedule", "lr_mult"), 1.0),
                                        (("datasets", "f30k_precomp", "schedule", "seed"), 901),
                                        (("datasets", "f30k_precomp", "provenance", "schedule.lr_mult"), [{"section": ["locked_recipes"], "quote": "x", "token": "x"}]),
                                        (("datasets", "f30k_precomp", "provenance", "model.img_dim", 0, "note"), "x"), (("datasets", "coco_precomp", "num_epochs"), 20)])
def test_unknown_key_raises(tmp_path, path, value):
    with pytest.raises(HC.HostConfigError, match="unknown key"):
        HC.load_host_config(_dump(tmp_path, _set(_doc(), path, value)), "f30k_precomp")


@pytest.mark.parametrize("block,key,value", [("model", "img_dim", "2048"), ("model", "img_dim", 2048.0), ("model", "num_layers", True), ("model", "bi_gru", 1),
                                             ("model", "bi_gru", "true"), ("model", "module_name", "SAF"), ("schedule", "margin", 1), ("schedule", "grad_clip", 2),
                                             ("schedule", "learning_rate", "2e-4"), ("schedule", "batch_size", None), ("schedule", "num_epochs", [40]),
                                             ("schedule", "optimizer", "Adam"), ("schedule", "warmup_loss", "max_violation_hinge")])
def test_wrong_type_or_name_raises(tmp_path, block, key, value):
    d = _set(_doc(), ("datasets", "f30k_precomp", block, key), value)
    with pytest.raises(HC.HostConfigError, match=rf"{block}\.{key}: expected"):
        HC.load_host_config(_dump(tmp_path, d), "f30k_precomp")


@pytest.mark.parametrize("block,key,value", [("model", "img_dim", 0), ("model", "img_dim", 204800), ("model", "num_region", 100000), ("schedule", "batch_size", 1),
                                             ("schedule", "num_epochs", 0), ("schedule", "learning_rate", 0.0), ("schedule", "learning_rate", 2.0),
                                             ("schedule", "margin", -0.2), ("schedule", "grad_clip", -1.0), ("schedule", "lr_decay_factor", 1.5),
                                             ("schedule", "warmup_epochs", -1), ("schedule", "teacher_epochs", 0)])
def test_out_of_range_raises(tmp_path, block, key, value):
    d = _set(_doc(), ("datasets", "f30k_precomp", block, key), value)
    with pytest.raises(HC.HostConfigError, match=rf"{block}\.{key}: .* outside"):
        HC.load_host_config(_dump(tmp_path, d), "f30k_precomp")


def test_the_warm_up_must_leave_a_hard_negative_phase(tmp_path):
    d = _set(_doc(), ("datasets", "f30k_precomp", "schedule", "warmup_epochs"), 40)
    with pytest.raises(HC.HostConfigError, match=r"warmup_epochs \(40\) must be < num_epochs \(40\)"):
        HC.load_host_config(_dump(tmp_path, d), "f30k_precomp")


@pytest.mark.parametrize("path,value,match", [(("schema_version",), 2, "schema_version: expected 1"), (("schema_version",), "1", "schema_version: expected 1"),
                                              (("derived_from", "sha256"), "8be8ced9", "derived_from.sha256"),
                                              (("derived_from", "sha256"), "8BE8CED9F6BC99B4748B4384658E42425D6ADF186047D2246A9A25CFB0BCF8B5", "derived_from.sha256"),
                                              (("derived_from", "git_commit"), "103d8f8", "derived_from.git_commit"), (("derived_from", "path"), "", "derived_from.path"),
                                              (("datasets",), {}, "non-empty mapping"), (("datasets", "f30k_precomp", "status"), "LOCKED", "status: expected one of"),
                                              (("datasets", "Flickr30K"), {"status": "not_locked"}, "does not match")])
def test_header_status_and_names_are_checked(tmp_path, path, value, match):
    with pytest.raises(HC.HostConfigError, match=match):
        HC.load_host_config(_dump(tmp_path, _set(_doc(), path, value)), "f30k_precomp")


@pytest.mark.parametrize("key,index,field,value,match", [("model.img_dim", None, None, [], "non-empty list of sources"),
                                                         ("model.img_dim", 0, "token", "4096", "does not occur in quote"),
                                                         ("model.img_dim", 0, "section", [], "section: expected a non-empty list"),
                                                         ("model.img_dim", 0, "section", ["features", True], "section: expected"),
                                                         ("model.img_dim", 0, "quote", " ", "quote: expected a non-empty string")])
def test_provenance_structure_is_checked(tmp_path, key, index, field, value, match):
    path = ("datasets", "f30k_precomp", "provenance", key) + (() if index is None else (index, field))
    with pytest.raises(HC.HostConfigError, match=match):
        HC.load_host_config(_dump(tmp_path, _set(_doc(), path, value)), "f30k_precomp")


@pytest.mark.parametrize("old,new,match", [("      img_dim: 2048\n", "      img_dim: 2048\n      img_dim: 2048\n", "duplicate key 'img_dim'"),
                                           ("      bi_gru: true\n", "      bi_gru: yes\n", "true or false"),
                                           ("      bi_gru: true\n", "      bi_gru: True\n", "true or false"),
                                           ("      learning_rate: 0.0002\n", "      learning_rate: 2e-4\n", "learning_rate: expected float"),
                                           ("      num_epochs: 40\n", "      num_epochs: 040\n", "plain decimal"),
                                           ("      batch_size: 128\n", "      batch_size: 1_28\n", "plain decimal"),
                                           ("      margin: 0.2\n", "      margin: .inf\n", "must be finite"),
                                           ("      img_dim: 2048\n", "      img_dim: &d 2048\n", "anchors / aliases"),
                                           ("      img_dim: 2048\n", "      <<: {img_dim: 2048}\n", "merge keys"),
                                           ("      img_dim: 2048\n", "      img_dim: 2048\n      1: 2\n", "keys must be strings"),
                                           ("  path: reports/V9/HOST_LOCK.yaml\n", "  path: 2026-10-15\n", "timestamp"),
                                           ("schema_version: 1\n", "schema_version: 1\n---\nschema_version: 1\n", "not a valid machine lock document")])
def test_yaml_that_a_typed_lock_must_not_contain_raises(tmp_path, old, new, match):
    with pytest.raises(HC.HostConfigError, match=match):
        HC.load_host_config(_edit_text(tmp_path, old, new), "f30k_precomp")


# --------------------------------------------------------------------------------------------------------------- code bridge
@pytest.mark.parametrize("key,value,match", [("lr_decay_factor", 0.5, "lr_decay_factor = 0.5"), ("teacher_epochs", 6, r"teacher_epochs \(6\) > warmup_epochs \(5\)")])
def test_legacy_kwargs_refuse_what_the_v9_code_cannot_honour(tmp_path, key, value, match):
    cfg = HC.load_host_config(_dump(tmp_path, _set(_doc(), ("datasets", "f30k_precomp", "schedule", key), value)), "f30k_precomp")   # schema-valid ...
    with pytest.raises(HC.HostConfigError, match=match):
        HC.legacy_kwargs(cfg)                                                                                             # ... but train.py / teacher.py would not do it
