"""V9 strict host configuration (review item U03): the typed, machine-readable twin of reports/V9/HOST_LOCK.yaml.

reports/V9/HOST_LOCK.yaml (fork A) mixes prose and values: key 'model (SGR single, SGRAF opts.py defaults)' holds
opts: "img_dim: 2048; word_dim: 300; ...", key 'training (native SGR, train.py)' holds per-dataset dicts and optimizer prose.  The legacy
loader v9.n2.teacher.load_host_lock therefore reads it with dict lookups that miss those key names, regex fallbacks over the raw text
(first match wins) and code defaults for num_layers / bi_gru / num_region.  This module reads reports/V9/HOST_LOCK_MACHINE.yaml instead:

* strict schema, no defaults: every field is required and has an exact type (an int is never a bool; floats are written with a decimal
  point; booleans are true / false), a finite range, and for names a closed set (only what src/v9/n2 implements).  Unknown, missing or
  duplicate keys, YAML anchors / aliases / merge keys, non-decimal ints, timestamps and multi-document files raise HostConfigError;
* a dataset absent from the file raises HostConfigError; a dataset whose status is not 'locked' raises DatasetNotLocked;
* with source_lock=<HOST_LOCK.yaml>, a machine file derived from another version of HOST_LOCK.yaml raises StaleMachineLock;
* every value carries provenance (section = key path in the parsed HOST_LOCK.yaml, quote = verbatim text there, token = the part of the
  quote that denotes the value); verify_provenance() checks every source against the real HOST_LOCK.yaml.

Pure Python + PyYAML (no torch, no v9.n2 import), so the V9 runner (v9_sgr env) and scripts/v9/n2_official.py (main env) can both import it.
For use from the NEXT block onwards; the running V9-N2 Flickr30K block keeps v9.n2.teacher.load_host_lock.

    strict = load_host_config("reports/V9/HOST_LOCK_MACHINE.yaml", "f30k_precomp", source_lock="reports/V9/HOST_LOCK.yaml")
    cfg = HostConfig(vocab_size=len(w2i), **legacy_kwargs(strict))     # v9.n2.teacher.HostConfig; vocab_size stays a runtime input
    assert_matches_legacy(strict, load_host_lock(...))                 # optional transition check against the legacy resolution
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA_VERSION = 1
MACHINE_LOCK_PATH = "reports/V9/HOST_LOCK_MACHINE.yaml"     # repo-relative
SOURCE_LOCK_PATH = "reports/V9/HOST_LOCK.yaml"


class HostConfigError(ValueError):
    """The machine host lock violates the strict schema (the message names the key path), or cannot be used as asked."""


class DatasetNotLocked(HostConfigError):
    """The requested dataset is in the machine lock but its status is not 'locked'."""


class StaleMachineLock(HostConfigError):
    """The machine lock was derived from a different HOST_LOCK.yaml than the one given as source_lock."""


# ---------------------------------------------------------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class _Spec:
    kind: str                     # 'int' | 'float' | 'bool' | 'enum'
    lo: float | None = None       # lower bound, inclusive (exclusive when lo_open)
    hi: float | None = None       # upper bound, inclusive
    lo_open: bool = False
    choices: tuple = ()


# Ranges are sanity bounds that catch slips (a dropped or extra digit, a wrong unit); name sets are what src/v9/n2 implements.
MODEL_SPEC = {
    "img_dim": _Spec("int", 1, 16384),             # region feature size (BUTD precomp: 2048)
    "word_dim": _Spec("int", 1, 4096),
    "embed_size": _Spec("int", 1, 16384),
    "sim_dim": _Spec("int", 1, 4096),
    "num_layers": _Spec("int", 1, 8),              # GRU layers (SGRAF opts num_layers)
    "bi_gru": _Spec("bool"),
    "num_region": _Spec("int", 1, 1024),           # regions per image (precomp features: 36)
    "sgr_step": _Spec("int", 1, 32),
    "module_name": _Spec("enum", choices=("SGR",)),                       # src/v9/n2/sgr_host.py keeps the SGR path only
}
SCHEDULE_SPEC = {
    "num_epochs": _Spec("int", 1, 1000),           # student epochs 0 .. num_epochs-1, warm-up included
    "lr_update": _Spec("int", 1, 1000),            # lr(epoch) = learning_rate * lr_decay_factor ** (epoch // lr_update)
    "lr_decay_factor": _Spec("float", 0.0, 1.0, lo_open=True),
    "batch_size": _Spec("int", 2, 65536),          # in-batch negatives need >= 2 pairs (singleton batches are dropped)
    "optimizer": _Spec("enum", choices=("adam",)),
    "learning_rate": _Spec("float", 0.0, 1.0, lo_open=True),
    "margin": _Spec("float", 0.0, 10.0, lo_open=True),
    "grad_clip": _Spec("float", 0.0, 1000.0),      # clip_grad_norm_ max norm; 0.0 = no clipping (train.py / teacher.py)
    "warmup_epochs": _Spec("int", 0, 1000),        # epochs 0 .. warmup_epochs-1 use warmup_loss, the remaining epochs main_loss
    "warmup_loss": _Spec("enum", choices=("warmup_mean_hinge",)),         # HOST_LOCK 'sum-violation triplet loss'; mean over in-batch negatives
    "main_loss": _Spec("enum", choices=("max_violation_hinge",)),         # hardest in-batch negative per row and per column
    "teacher_epochs": _Spec("int", 1, 1000),
    "checkpoint_selection": _Spec("enum", choices=("best_dev_rsum",)),
}
STATUSES = ("locked", "not_locked")
TOP_KEYS = frozenset({"schema_version", "derived_from", "datasets"})
DERIVED_FROM_KEYS = frozenset({"path", "sha256", "git_commit"})
LOCKED_KEYS = frozenset({"status", "model", "schedule", "provenance"})
NOT_LOCKED_KEYS = frozenset({"status", "reason", "provenance"})          # a dataset that is not locked carries no values
SOURCE_KEYS = frozenset({"section", "quote", "token"})
VALUE_KEYS = tuple([f"model.{k}" for k in MODEL_SPEC] + [f"schedule.{k}" for k in SCHEDULE_SPEC])
_DATASET_RE = re.compile(r"[a-z0-9]+_precomp\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

# legacy v9.n2.teacher.HostConfig: the fields this config supplies, and the ones it never supplies (runtime input / legacy pending lists)
LEGACY_FIELDS = ("img_dim", "word_dim", "embed_size", "sim_dim", "num_layers", "bi_gru", "num_region", "sgr_step", "margin", "learning_rate",
                 "grad_clip", "batch_size", "lr_update", "num_epochs", "teacher_epochs", "warmup_epochs")
LEGACY_BOOKKEEPING = ("vocab_size", "to_verify", "lock_notes")
# choices src/v9/n2 hard-codes instead of reading them: legacy_kwargs() refuses a schema-valid config that asks for anything else
CODE_FIXED = {"module_name": "SGR", "optimizer": "adam", "lr_decay_factor": 0.1, "warmup_loss": "warmup_mean_hinge", "main_loss": "max_violation_hinge",
              "checkpoint_selection": "best_dev_rsum"}
CODE_FIXED_WHERE = {
    "module_name": "src/v9/n2/sgr_host.py keeps the SGR path only",
    "optimizer": "torch.optim.Adam in src/v9/n2/train.py WeightedStudent.fit and teacher.py train_teacher",
    "lr_decay_factor": "0.1 ** (epoch // lr_update) in src/v9/n2/train.py and teacher.py",
    "warmup_loss": "hinge_per_anchor(hard_negative=False) in src/v9/n2/train.py (epoch < warmup_epochs) and teacher.py (every epoch)",
    "main_loss": "hinge_per_anchor(hard_negative=True) for epoch >= warmup_epochs in src/v9/n2/train.py",
    "checkpoint_selection": "best dev rSum, ties -> earlier epoch, in src/v9/n2/train.py",
}
# words of HOST_LOCK.yaml that denote each name value: verify_provenance() requires a source's token to be one of them
ENUM_EVIDENCE = {
    "SGR": ("SGR",),
    "adam": ("Adam",),
    "warmup_mean_hinge": ("sum-violation", "warm-up loss"),
    "max_violation_hinge": ("hardest in-batch negative", "hard-negative", "max_violation=True"),
    "best_dev_rsum": ("dev-rSum", "best_dev_rsum"),
    "not_locked": ("not locked",),
}


@dataclass(frozen=True)
class StrictHostConfig:
    """One locked dataset of the machine host lock: frozen, every field required (no defaults).
    provenance = ((value key, ((section, quote, token), ...)), ...) for 'model.<field>' / 'schedule.<field>'."""
    dataset: str
    schema_version: int
    source_path: str
    source_sha256: str
    # model: SGR single
    img_dim: int
    word_dim: int
    embed_size: int
    sim_dim: int
    num_layers: int
    bi_gru: bool
    num_region: int
    sgr_step: int
    module_name: str
    # schedule: common student (B table) and teacher
    num_epochs: int
    lr_update: int
    lr_decay_factor: float
    batch_size: int
    optimizer: str
    learning_rate: float
    margin: float
    grad_clip: float
    warmup_epochs: int
    warmup_loss: str
    main_loss: str
    teacher_epochs: int
    checkpoint_selection: str
    provenance: tuple = field(compare=False, repr=False)

    def values(self) -> dict:
        """model + schedule values (plain dict, e.g. for a run manifest)."""
        return {k: getattr(self, k) for k in (*MODEL_SPEC, *SCHEDULE_SPEC)}

    def sources(self, key: str) -> tuple:
        return dict(self.provenance)[key]


_META_FIELDS = ("dataset", "schema_version", "source_path", "source_sha256", "provenance")
if ({f.name for f in dataclasses.fields(StrictHostConfig)} - set(_META_FIELDS) != set(MODEL_SPEC) | set(SCHEDULE_SPEC)
        or not set(LEGACY_FIELDS) <= set(MODEL_SPEC) | set(SCHEDULE_SPEC) or set(CODE_FIXED) != set(CODE_FIXED_WHERE)):
    raise RuntimeError("v9.host_config: StrictHostConfig fields, MODEL_SPEC / SCHEDULE_SPEC, LEGACY_FIELDS and CODE_FIXED disagree")


# ------------------------------------------------------------------------------------------------------------------ strict YAML layer
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses what a typed lock must not contain (anchors, aliases, merge keys, duplicate or non-string keys, loose scalars)."""

    def compose_node(self, parent, index):
        ev = self.peek_event()
        if isinstance(ev, yaml.AliasEvent) or getattr(ev, "anchor", None) is not None:
            raise HostConfigError(f"line {ev.start_mark.line + 1}: YAML anchors / aliases are not allowed in the machine lock")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen = {}
            for key_node, _ in node.value:
                line = key_node.start_mark.line + 1
                if key_node.tag == "tag:yaml.org,2002:merge":
                    raise HostConfigError(f"line {line}: YAML merge keys (<<) are not allowed in the machine lock")
                key = self.construct_object(key_node, deep=True)
                if not isinstance(key, str):
                    raise HostConfigError(f"line {line}: mapping keys must be strings, got {type(key).__name__} {key!r}")
                if key in seen:
                    raise HostConfigError(f"line {line}: duplicate key {key!r} (first at line {seen[key]})")
                seen[key] = line
        return super().construct_mapping(node, deep=deep)


_INT_RE = re.compile(r"[-+]?(?:0|[1-9][0-9]*)\Z")
_FLOAT_RE = re.compile(r"[-+]?[0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?\Z")


def _yaml_int(loader, node):
    if not _INT_RE.match(node.value):
        raise HostConfigError(f"line {node.start_mark.line + 1}: integer {node.value!r} must be plain decimal (no octal / hex / '_' / sexagesimal)")
    return int(node.value)


def _yaml_float(loader, node):
    if not _FLOAT_RE.match(node.value):
        raise HostConfigError(f"line {node.start_mark.line + 1}: float {node.value!r} must be finite and written digits.digits[e-N]")
    return float(node.value)


def _yaml_bool(loader, node):
    if node.value not in ("true", "false"):
        raise HostConfigError(f"line {node.start_mark.line + 1}: boolean {node.value!r} must be written true or false (YAML 1.1 yes / no / on / off refused)")
    return node.value == "true"


def _yaml_refused(kind):
    def construct(loader, node):
        raise HostConfigError(f"line {node.start_mark.line + 1}: {kind} values are not allowed in the machine lock (quote it if it is text)")
    return construct


for _tag, _fn in (("int", _yaml_int), ("float", _yaml_float), ("bool", _yaml_bool), ("timestamp", _yaml_refused("timestamp")),
                  ("binary", _yaml_refused("binary")), ("set", _yaml_refused("set")), ("omap", _yaml_refused("omap")), ("pairs", _yaml_refused("pairs"))):
    _StrictLoader.add_constructor(f"tag:yaml.org,2002:{_tag}", _fn)


def _parse(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise HostConfigError(f"{path}: cannot read the machine lock ({e})") from None
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except HostConfigError as e:                                   # raised by the strict YAML layer (line numbers in the message)
        raise HostConfigError(f"{path}: {e}") from None
    except yaml.YAMLError as e:
        raise HostConfigError(f"{path}: not a valid machine lock document: {e}") from None


# ------------------------------------------------------------------------------------------------------------------------ validation
def _keys(obj, required, where: str) -> None:
    if not isinstance(obj, dict):
        raise HostConfigError(f"{where}: expected a mapping, got {type(obj).__name__}")
    missing, unknown = sorted(set(required) - set(obj)), sorted((str(k) for k in set(obj) - set(required)))
    if missing:
        raise HostConfigError(f"{where}: missing key(s) {missing} (the strict loader has no defaults)")
    if unknown:
        raise HostConfigError(f"{where}: unknown key(s) {unknown}")


def _value(v, spec: _Spec, where: str) -> None:
    if spec.kind == "enum":
        if type(v) is not str or v not in spec.choices:
            raise HostConfigError(f"{where}: expected one of {list(spec.choices)}, got {v!r}")
        return
    want = {"int": int, "float": float, "bool": bool}[spec.kind]
    if type(v) is not want:
        hint = " (write floats with a decimal point, e.g. 0.0002 or 2.0)" if spec.kind == "float" else ""
        raise HostConfigError(f"{where}: expected {spec.kind}, got {type(v).__name__} {v!r}{hint}")
    if spec.kind == "bool":
        return
    if not math.isfinite(v) or not (v > spec.lo if spec.lo_open else v >= spec.lo) or v > spec.hi:
        raise HostConfigError(f"{where}: {v!r} outside {'(' if spec.lo_open else '['}{spec.lo:g}, {spec.hi:g}]")


def _text(v, where: str) -> None:
    if type(v) is not str or not v.strip():
        raise HostConfigError(f"{where}: expected a non-empty string, got {v!r}")


def _provenance(prov, required, where: str) -> None:
    _keys(prov, required, where)
    for key in required:
        sources = prov[key]
        if type(sources) is not list or not sources:
            raise HostConfigError(f"{where}.{key}: expected a non-empty list of sources")
        for i, src in enumerate(sources):
            w = f"{where}.{key}[{i}]"
            _keys(src, SOURCE_KEYS, w)
            sec = src["section"]
            if type(sec) is not list or not sec or not all((type(s) is str and s.strip()) or (type(s) is int and s >= 0) for s in sec):
                raise HostConfigError(f"{w}.section: expected a non-empty list of keys (strings) / list indices (ints >= 0), got {sec!r}")
            _text(src["quote"], f"{w}.quote")
            _text(src["token"], f"{w}.token")
            if src["token"] not in src["quote"]:
                raise HostConfigError(f"{w}: token {src['token']!r} does not occur in quote {src['quote']!r}")


def _dataset(entry, where: str) -> None:
    if not isinstance(entry, dict):
        raise HostConfigError(f"{where}: expected a mapping, got {type(entry).__name__}")
    if "status" not in entry:
        raise HostConfigError(f"{where}: missing key(s) ['status'] (the strict loader has no defaults)")
    status = entry["status"]
    if type(status) is not str or status not in STATUSES:
        raise HostConfigError(f"{where}.status: expected one of {list(STATUSES)}, got {status!r}")
    if status == "not_locked":
        _keys(entry, NOT_LOCKED_KEYS, where)
        _text(entry["reason"], f"{where}.reason")
        _provenance(entry["provenance"], ("status",), f"{where}.provenance")
        return
    _keys(entry, LOCKED_KEYS, where)
    for block, specs in (("model", MODEL_SPEC), ("schedule", SCHEDULE_SPEC)):
        _keys(entry[block], specs, f"{where}.{block}")
        for k, spec in specs.items():
            _value(entry[block][k], spec, f"{where}.{block}.{k}")
    s = entry["schedule"]
    if s["warmup_epochs"] >= s["num_epochs"]:
        raise HostConfigError(f"{where}.schedule: warmup_epochs ({s['warmup_epochs']}) must be < num_epochs ({s['num_epochs']}): "
                              "the hard-negative phase follows the warm-up inside num_epochs")
    _provenance(entry["provenance"], VALUE_KEYS, f"{where}.provenance")


def _document(doc) -> None:
    _keys(doc, TOP_KEYS, "(top level)")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != SCHEMA_VERSION:
        raise HostConfigError(f"schema_version: expected {SCHEMA_VERSION} (the only version this loader reads), got {doc['schema_version']!r}")
    df = doc["derived_from"]
    _keys(df, DERIVED_FROM_KEYS, "derived_from")
    _text(df["path"], "derived_from.path")
    for k, rx in (("sha256", _SHA256_RE), ("git_commit", _COMMIT_RE)):
        if type(df[k]) is not str or not rx.match(df[k]):
            raise HostConfigError(f"derived_from.{k}: expected lower-case hex matching {rx.pattern}, got {df[k]!r}")
    ds = doc["datasets"]
    if not isinstance(ds, dict) or not ds:
        raise HostConfigError("datasets: expected a non-empty mapping of dataset names")
    for name, entry in ds.items():
        if type(name) is not str or not _DATASET_RE.match(name):
            raise HostConfigError(f"datasets: dataset name {name!r} does not match {_DATASET_RE.pattern}")
        _dataset(entry, f"datasets.{name}")


def read_machine_lock(path) -> dict:
    """The whole machine lock, parsed by the strict YAML layer and validated (every dataset, locked or not)."""
    path = Path(path)
    doc = _parse(path)
    try:
        _document(doc)
    except HostConfigError as e:
        raise type(e)(f"{path}: {e}") from None
    return doc


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_source(doc: dict, source_lock) -> None:
    try:
        actual = sha256_of(source_lock)
    except OSError as e:
        raise HostConfigError(f"source lock {source_lock}: cannot read it ({e})") from None
    recorded = doc["derived_from"]["sha256"]
    if actual != recorded:
        raise StaleMachineLock(f"{source_lock}: sha256 {actual} != derived_from.sha256 {recorded}: HOST_LOCK.yaml changed after the machine lock was "
                               f"derived; re-derive {MACHINE_LOCK_PATH} and run tests/v9/test_host_config.py before loading it")


def load_host_config(path, dataset: str, *, source_lock=None) -> StrictHostConfig:
    """The locked host configuration of `dataset` (e.g. 'f30k_precomp').  Validates the whole file first; raises HostConfigError on any schema
    violation or an absent dataset, DatasetNotLocked when its status is not 'locked', StaleMachineLock when `source_lock` (the prose
    HOST_LOCK.yaml; production entry points pass it) is not the file the machine lock was derived from.  Nothing is ever defaulted."""
    doc = read_machine_lock(path)
    if source_lock is not None:
        _check_source(doc, source_lock)
    if type(dataset) is not str or dataset not in doc["datasets"]:
        raise HostConfigError(f"{path}: dataset {dataset!r} is not in the machine lock (present: {sorted(doc['datasets'])})")
    entry = doc["datasets"][dataset]
    if entry["status"] != "locked":
        raise DatasetNotLocked(f"{path}: dataset {dataset!r} has status {entry['status']!r} ({entry['reason']}); the strict loader refuses it: "
                               f"lock it in {doc['derived_from']['path']} first, then re-derive the machine lock")
    prov = tuple((k, tuple((tuple(s["section"]), s["quote"], s["token"]) for s in entry["provenance"][k])) for k in VALUE_KEYS)
    return StrictHostConfig(dataset=dataset, schema_version=doc["schema_version"], source_path=doc["derived_from"]["path"],
                            source_sha256=doc["derived_from"]["sha256"], provenance=prov, **entry["model"], **entry["schedule"])


# ------------------------------------------------------------------------------------------------------------------- legacy bridge
def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def check_code_compatibility(cfg: StrictHostConfig) -> None:
    """Refuse a schema-valid config that the V9 SGR code would not honour (it hard-codes these choices instead of reading them)."""
    problems = [f"{k} = {getattr(cfg, k)!r} but the V9 code implements only {v!r} ({CODE_FIXED_WHERE[k]})"
                for k, v in CODE_FIXED.items() if not _same(getattr(cfg, k), v)]
    if cfg.teacher_epochs > cfg.warmup_epochs:
        problems.append(f"teacher_epochs ({cfg.teacher_epochs}) > warmup_epochs ({cfg.warmup_epochs}): train_teacher (src/v9/n2/teacher.py) uses the "
                        "warm-up loss for every teacher epoch, so a teacher running past the warm-up would not follow the same recipe")
    if problems:
        raise HostConfigError(f"{cfg.dataset}: " + "; ".join(problems))


def legacy_kwargs(cfg: StrictHostConfig) -> dict:
    """Keyword arguments for v9.n2.teacher.HostConfig: all its fields except vocab_size (runtime: len of the vocab file) and the legacy
    to_verify / lock_notes lists (nothing is pending in a locked dataset).  Runs check_code_compatibility first."""
    check_code_compatibility(cfg)
    return {k: getattr(cfg, k) for k in LEGACY_FIELDS}


_ABSENT = object()


def compare_with_legacy(cfg: StrictHostConfig, legacy) -> dict:
    """Field-by-field comparison (value AND type) with a legacy v9.n2.teacher.HostConfig, e.g. load_host_lock(HOST_LOCK.yaml, dataset, vocab_size).
    equal = every LEGACY_FIELDS entry matches, the legacy to_verify list is empty and the legacy HostConfig has no field this schema does not cover."""
    names = [f.name for f in dataclasses.fields(legacy)] if dataclasses.is_dataclass(legacy) else sorted(vars(legacy))
    rows = {}
    for k in LEGACY_FIELDS:
        sv, lv = getattr(cfg, k), getattr(legacy, k, _ABSENT)
        rows[k] = {"strict": sv, "legacy": None if lv is _ABSENT else lv, "strict_type": type(sv).__name__,
                   "legacy_type": "absent" if lv is _ABSENT else type(lv).__name__, "equal": lv is not _ABSENT and _same(sv, lv)}
    rep = {"dataset": cfg.dataset, "fields": rows, "mismatched": [k for k, r in rows.items() if not r["equal"]],
           "legacy_to_verify": list(getattr(legacy, "to_verify", None) or []), "legacy_lock_notes": list(getattr(legacy, "lock_notes", None) or []),
           "unmapped_legacy_fields": sorted(set(names) - set(LEGACY_FIELDS) - set(LEGACY_BOOKKEEPING)), "legacy_vocab_size": getattr(legacy, "vocab_size", None)}
    rep["equal"] = not rep["mismatched"] and not rep["legacy_to_verify"] and not rep["unmapped_legacy_fields"]
    return rep


def assert_matches_legacy(cfg: StrictHostConfig, legacy) -> dict:
    rep = compare_with_legacy(cfg, legacy)
    if not rep["equal"]:
        parts = [f"{k}: strict {r['strict']!r} ({r['strict_type']}) vs legacy {r['legacy']!r} ({r['legacy_type']})" for k, r in rep["fields"].items() if not r["equal"]]
        if rep["legacy_to_verify"]:
            parts.append(f"legacy to_verify {rep['legacy_to_verify']}")
        if rep["unmapped_legacy_fields"]:
            parts.append(f"legacy HostConfig fields this schema does not cover: {rep['unmapped_legacy_fields']}")
        raise HostConfigError(f"{cfg.dataset}: strict and legacy host configs differ: " + "; ".join(parts))
    return rep


# --------------------------------------------------------------------------------------------------------------- provenance check
def _walk(node, section):
    for key in section:
        if isinstance(node, dict) and type(key) is str and key in node:
            node = node[key]
        elif isinstance(node, list) and type(key) is int and key < len(node):
            node = node[key]
        else:
            return False, None
    return True, node


def _top_level_blocks(text: str) -> dict:
    """Raw text (comments included) of each top-level key of a YAML file, from its key line to the next top-level key."""
    blocks, cur = {}, None
    for line in text.splitlines(keepends=True):
        m = re.match(r"([^\s#][^:]*):(?:\s|$)", line)
        if m:
            cur = m.group(1)
            blocks[cur] = ""
        if cur is not None:
            blocks[cur] += line
    return blocks


def _token_problem(token: str, value):
    if type(value) is bool:
        ok = token.lower() == ("true" if value else "false")
    elif type(value) is int:
        ok = bool(re.fullmatch(r"[-+]?[0-9]+", token)) and int(token) == value
    elif type(value) is float:
        try:
            ok = float(token) == value
        except ValueError:
            ok = False
    else:
        ok = token in ENUM_EVIDENCE.get(value, ())
    return None if ok else f"token {token!r} does not denote {value!r}"


def verify_provenance(path, source_lock) -> dict:
    """Check every provenance source of the machine lock against the prose HOST_LOCK.yaml: the section exists in the parsed file; the quote occurs
    verbatim in it (in its text when the section is a string, otherwise in the raw text of its top-level block, comments included); the token
    denotes the value (ints / floats / booleans parsed by the field's type, names through ENUM_EVIDENCE); a scalar section equals the value in value
    and type; the file's sha256 equals derived_from.sha256.  Returns {'checked_sources', 'problems', 'source_sha256', 'recorded_sha256'}."""
    doc = read_machine_lock(path)
    try:
        raw = Path(source_lock).read_bytes()
    except OSError as e:
        raise HostConfigError(f"source lock {source_lock}: cannot read it ({e})") from None
    text = raw.decode("utf-8")
    src, blocks, sha = yaml.safe_load(text), _top_level_blocks(text), hashlib.sha256(raw).hexdigest()
    problems, checked = [], 0
    if sha != doc["derived_from"]["sha256"]:
        problems.append(f"derived_from.sha256 {doc['derived_from']['sha256']} != sha256 of {source_lock} ({sha})")
    for name, entry in doc["datasets"].items():
        if entry["status"] == "locked":
            values = {**{f"model.{k}": v for k, v in entry["model"].items()}, **{f"schedule.{k}": v for k, v in entry["schedule"].items()}}
        else:
            values = {"status": entry["status"]}
        for key, sources in entry["provenance"].items():
            for i, s in enumerate(sources):
                checked += 1
                w = f"datasets.{name}.provenance.{key}[{i}]"
                found, node = _walk(src, s["section"])
                if not found:
                    problems.append(f"{w}: section {s['section']} not found in {source_lock}")
                    continue
                if s["quote"] not in (node if isinstance(node, str) else blocks.get(s["section"][0], "")):
                    problems.append(f"{w}: quote {s['quote']!r} not found in section {s['section']}")
                if isinstance(node, (bool, int, float)) and not _same(node, values[key]):
                    problems.append(f"{w}: section value {node!r} != machine value {values[key]!r}")
                tp = _token_problem(s["token"], values[key])
                if tp:
                    problems.append(f"{w}: {tp}")
    return {"checked_sources": checked, "problems": problems, "source_sha256": sha, "recorded_sha256": doc["derived_from"]["sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load one dataset of the machine host lock strictly and verify its provenance against HOST_LOCK.yaml.")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--host-lock", required=True)
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args(argv)
    cfg = load_host_config(a.machine, a.dataset, source_lock=a.host_lock)
    prov = verify_provenance(a.machine, a.host_lock)
    print(json.dumps({"dataset": cfg.dataset, "source_sha256": cfg.source_sha256, "values": cfg.values(), "host_config_kwargs": legacy_kwargs(cfg),
                      "provenance_check": prov}, indent=1))
    return 0 if not prov["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
