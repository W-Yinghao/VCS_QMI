"""V10 EPFL data gate (decision 2026-10-17 section 6.2 item 5: "pass the V10 data gate").  Standard library only; reads no image.

The phase-1 V10 DataGuard (src/v10/adapter.py) admits BF_SHG only and is left unchanged.  This gate admits EPFL_RGB_NIR windows only under
an explicit activation record whose sha256 the caller pins (job environment / release), with the roles of that record's stage:

  stage GATE_CHECKS  FIT and CAL windows: the item-5 numerical checks and the full-cost timing; nothing is selected or reported from them
  stage ACTIVATED    FIT, CAL and DEV windows: EPFL experiments under a frozen EPFL protocol; requires the owner's decision reference

Always refused: any record that is not an EPFL row of the pinned V10 manifest (in particular every BF_SHG record and the 134 held BF TEST
cores), the 57 excluded EPFL pairs (33 alignment, 24 size; they have no window and no role), a window whose REG-P3 role differs from its
V10 role, roles outside the stage, and any path under a held location.  EPFL has no TEST / held role in the frozen split; if a later
manifest version adds one it stays refused here (only FIT / CAL / DEV can ever be admitted).

Window ids are REG-P3's ("<category>_<id>", one centred 512 window per kept pair); record ids are the V10 manifest's ("<category>/<id>").
Every granted read is logged in .reads (window, record, role, purpose, path, kind, utc), as the BF gate does.

Group record (main-session decision 2026-10-19 no. 1, option A): a frozen group record v2 removes windows from FIT (water_0006 and
street_0020, the same scenes as DEV water_0021 / street_0021).  At stage ACTIVATED the record must name it (path + sha256) and the gate
applies it: removed windows leave included('FIT') and are refused by check().  A caller may also pass a pinned group record explicitly
(production smoke / tests at stage GATE_CHECKS).
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[3]
V10_MANIFEST = REPO / "v10" / "T00" / "V10_DATA_MANIFEST.csv"
V10_MANIFEST_SHA256 = "cc309cece34b574b2a518b5775a1e30987ebd9015693cca1543d5610f13564f6"
P3_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3")
P3_MANIFEST_SHA256 = "3ecf9fe8e76524135edcef5d1641e866e47fc1f5ae0de552839ad7736a476e78"
P3_PERTURBATIONS_SHA256 = "131fc797904e9cb3b0e7f62df57dd14523d914852f2dd1064f864b2e02055dcc"
ACTIVATION_RECORD = REPO / "v10" / "epfl" / "EPFL_ACTIVATION_RECORD.json"
GROUP_RECORD_V2 = REPO / "v10" / "epfl" / "EPFL_GROUP_RECORD_V2_FROZEN.json"
GROUP_SCHEMA = "v10.epfl.group_record.v2"
RECORD_SCHEMA = "v10.epfl.activation_record.v1"
STAGES = {"GATE_CHECKS": ("FIT", "CAL"), "ACTIVATED": ("CAL", "DEV", "FIT")}
# item 4 found same-scene FIT / DEV relations (ITEM4_DECISIONS.json): the numerical checks on FIT / CAL do not depend on the DEV grouping,
# so GATE_CHECKS may run while the group record awaits the owner; ACTIVATED needs the frozen group record (decision 6.2 item 4)
ITEM4_OK = {"GATE_CHECKS": {"PASS", "REVIEWED_GROUP_RECORD_PENDING_OWNER"}, "ACTIVATED": {"PASS"}}
ROLES = ("FIT", "CAL", "DEV")
FORBIDDEN_PATH_PARTS = ("/closed/", "outputs_v9/reg_test", "test.zip", "/test/")
EXPECTED = {"pairs": 477, "excluded": 57, "windows": 420, "FIT": 336, "CAL": 10, "DEV": 74}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def window_id(record_id: str) -> str:
    return record_id.replace("/", "_")


def record_id(window: str) -> str:
    cat, sid = window.split("_", 1)
    return f"{cat}/{sid}"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_record(rec: dict) -> None:
    """Structural rules of an activation record (no file access)."""
    if rec.get("schema") != RECORD_SCHEMA:
        raise PermissionError(f"activation record schema {rec.get('schema')!r} != {RECORD_SCHEMA}")
    if rec.get("dataset") != "EPFL_RGB_NIR":
        raise PermissionError("activation record is not for EPFL_RGB_NIR")
    stage = rec.get("stage")
    if stage not in STAGES:
        raise PermissionError(f"unknown activation stage {stage!r}")
    if sorted(rec.get("admitted_roles", [])) != sorted(STAGES[stage]):
        raise PermissionError(f"stage {stage} admits exactly {sorted(STAGES[stage])}, record says {rec.get('admitted_roles')}")
    if rec.get("manifest_sha256") != V10_MANIFEST_SHA256 or rec.get("p3_manifest_sha256") != P3_MANIFEST_SHA256:
        raise PermissionError("activation record names a different V10 / REG-P3 manifest")
    gates = rec.get("gates", {})
    for item in ("item1", "item2", "item3"):
        if gates.get(item, {}).get("status") != "PASS":
            raise PermissionError(f"activation record: gate {item} is not PASS")
    item4 = gates.get("item4", {}).get("status")
    if item4 not in ITEM4_OK[stage]:
        raise PermissionError(f"activation record: gate item4 is {item4!r}; stage {stage} needs one of {sorted(ITEM4_OK[stage])}")
    if stage == "ACTIVATED":
        if gates.get("item5", {}).get("status") != "PASS" or gates.get("item5_timing", {}).get("status") != "MEASURED":
            raise PermissionError("ACTIVATED needs the item-5 GPU gate PASS and the measured full cost")
        if not rec.get("owner_decision"):
            raise PermissionError("ACTIVATED needs the owner's decision reference")
        if not rec.get("frozen_epfl_protocol_sha256"):
            raise PermissionError("ACTIVATED needs a frozen EPFL protocol")
        if not rec.get("frozen_group_record_sha256"):
            raise PermissionError("ACTIVATED needs the frozen EPFL group record (decision 6.2 item 4)")
        ref = rec.get("frozen_group_record") or {}
        if not ref.get("path") or ref.get("sha256") != rec.get("frozen_group_record_sha256"):
            raise PermissionError("ACTIVATED must name the frozen group record file and its sha256")


class EPFLDataGuard:
    """EPFL data gate; interface of adapter.DataGuard (included / held / check / log_rule / reads / r6_root), keyed by REG-P3 window id."""

    dataset = "EPFL_RGB_NIR"

    def __init__(self, record=ACTIVATION_RECORD, expect_record_sha256: str | None = None, manifest=V10_MANIFEST,
                 expect_manifest_sha256: str = V10_MANIFEST_SHA256, p3_root=P3_ROOT, data_lock: dict | None = None,
                 group_record=None, expect_group_sha256: str | None = None):
        if not expect_record_sha256:
            raise PermissionError("the activation record sha256 must be pinned by the caller (no implicit activation)")
        raw_rec = Path(record).read_bytes()
        self.record_path, self.record_sha256 = str(record), hashlib.sha256(raw_rec).hexdigest()
        if self.record_sha256 != expect_record_sha256:
            raise PermissionError(f"activation record sha256 {self.record_sha256} != pinned {expect_record_sha256}")
        self.record = json.loads(raw_rec)
        validate_record(self.record)
        self.stage, self.admitted = self.record["stage"], frozenset(self.record["admitted_roles"])
        raw = Path(manifest).read_bytes()
        self.manifest_path, self.manifest_sha256 = str(manifest), hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != expect_manifest_sha256:
            raise RuntimeError(f"V10 manifest sha256 {self.manifest_sha256} != pinned {expect_manifest_sha256}")
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        self.rows = {r["record_id"]: r for r in rows if r["dataset"] == "EPFL_RGB_NIR"}
        self.foreign = {r["record_id"] for r in rows if r["dataset"] != "EPFL_RGB_NIR"}
        self.r6_root = Path(p3_root)
        p3raw = (self.r6_root / "manifest.json").read_bytes()
        if hashlib.sha256(p3raw).hexdigest() != P3_MANIFEST_SHA256:
            raise RuntimeError("REG-P3 manifest changed")
        p3 = json.loads(p3raw)
        self.p3_roles, self.p3_windows = p3["roles"], p3["windows"]
        self.data_lock = {w["window"]: w for w in (data_lock or {}).get("windows", [])}
        self.reads: list[dict] = []
        inc = [r for r in self.rows.values() if r["inclusion"] == "include"]
        if (len(self.rows), len(self.rows) - len(inc), len(inc)) != (EXPECTED["pairs"], EXPECTED["excluded"], EXPECTED["windows"]):
            raise RuntimeError("EPFL manifest counts differ from the frozen 477 / 57 / 420")
        for r in inc:
            w = window_id(r["record_id"])
            if self.p3_roles.get(w) != r["role"] or self.p3_windows.get(w, {}).get("pair") != r["record_id"]:
                raise RuntimeError(f"{r['record_id']}: V10 role {r['role']} / REG-P3 role {self.p3_roles.get(w)} mismatch")
        if sorted(self.p3_roles) != sorted(window_id(r["record_id"]) for r in inc):
            raise RuntimeError("REG-P3 window set differs from the V10 included EPFL rows")
        self.removed: dict = {}
        self.group_record, self.group_sha256 = None, None
        if self.stage == "ACTIVATED" or group_record is not None:
            if group_record is None:
                ref = self.record["frozen_group_record"]
                group_record = Path(ref["path"]) if Path(ref["path"]).is_absolute() else Path(self.record_path).parent / ref["path"]
                expect_group_sha256 = self.record["frozen_group_record_sha256"]
            self._apply_group_record(group_record, expect_group_sha256)

    def _apply_group_record(self, path, expect_sha):
        if not expect_sha:
            raise PermissionError("the group record sha256 must be pinned")
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expect_sha:
            raise PermissionError(f"group record sha256 differs from the pin {expect_sha}")
        g = json.loads(raw)
        if g.get("schema") != GROUP_SCHEMA or g.get("status") != "FROZEN" or g.get("base", {}).get("v10_manifest_sha256") != self.manifest_sha256:
            raise PermissionError("not a frozen EPFL group record on this manifest")
        for r in g["removed_from_FIT"]:
            w = r["window"]
            if self.p3_roles.get(w) != "FIT" or r["from_role"] != "FIT":
                raise PermissionError(f"{w}: the group record removes a window that is not FIT")
            self.removed[w] = r
        for role in ROLES:
            if sorted(g["roles"][role]) != self.included(role):
                raise PermissionError(f"group record role set {role} differs from the gate's")
        self.group_record, self.group_sha256 = g, expect_sha

    # listing ids is metadata (no read is granted or logged)
    def included(self, role: str) -> list[str]:
        if role not in ROLES:
            raise ValueError(role)
        removed = getattr(self, "removed", {})
        return sorted(window_id(k) for k, r in self.rows.items() if r["inclusion"] == "include" and r["role"] == role and window_id(k) not in removed)

    def expected_fit_windows(self) -> list[str]:
        """REG-P3's FIT windows minus the windows a frozen group record removed."""
        return sorted(w for w, r in self.p3_roles.items() if r == "FIT" and w not in self.removed)

    def held(self) -> list[str]:
        """Records that this gate always refuses: the 57 excluded EPFL pairs (as window-style ids; they have no window) and the windows a
        frozen group record removed from FIT."""
        return sorted([window_id(k) for k, r in self.rows.items() if r["inclusion"] != "include"] + list(getattr(self, "removed", {})))

    def unit_of(self, window: str) -> str:
        return self.rows[record_id(window)]["group_id"]

    def check(self, window: str, allowed_roles: Iterable[str], purpose: str) -> Path:
        allowed = set(allowed_roles)
        if not allowed or not allowed <= set(ROLES):
            raise PermissionError(f"allowed roles must be a non-empty subset of {ROLES}, got {sorted(allowed)}")
        if not allowed <= self.admitted:
            raise PermissionError(f"roles {sorted(allowed - self.admitted)} are not admitted at stage {self.stage} ({purpose})")
        if not isinstance(window, str) or "_" not in window:
            raise PermissionError(f"{window!r}: not an EPFL window id")
        rid = record_id(window)
        if rid in self.foreign or window in self.foreign:
            raise PermissionError(f"{window}: not an EPFL record (the EPFL gate never opens BF_SHG records)")
        r = self.rows.get(rid)
        if r is None:
            raise PermissionError(f"{window}: not an EPFL record of the V10 manifest")
        if r["inclusion"] != "include":
            raise PermissionError(f"{rid}: excluded ({r['exclusion_reason'][:90]})")
        if window in self.removed:
            raise PermissionError(f"{rid}: removed from FIT by the frozen group record v2 (same scene as DEV {self.removed[window]['same_scene_as']['window']})")
        if r["role"] not in allowed:
            raise PermissionError(f"{rid}: role {r['role']} not in {sorted(allowed)} for {purpose}")
        if self.p3_roles.get(window) != r["role"]:
            raise PermissionError(f"{window}: REG-P3 role {self.p3_roles.get(window)} != V10 role {r['role']}")
        path = self.r6_root / "cache" / f"{window}.npz"
        for text in (str(path), r["fixed_path"], r["moving_path"]):
            if any(bad in text for bad in FORBIDDEN_PATH_PARTS):
                raise PermissionError(f"{window}: path {text} is in a forbidden location")
        self.reads.append({"window": window, "record_id": rid, "role": r["role"], "purpose": purpose, "path": str(path), "kind": "read",
                           "stage": self.stage, "utc": utc()})
        return path

    def log_rule(self, window: str, allowed_roles: Iterable[str], purpose: str) -> None:
        """A truth-related quantity from a frozen rule without a file read (e.g. Q.perturbations_p2(window)): same gate, kind 'rule'."""
        self.check(window, allowed_roles, purpose)
        self.reads[-1].update({"kind": "rule", "path": None})

    def verify_data_lock(self, roles: Iterable[str]) -> dict:
        """Every cache file of the given (admitted) roles against the data lock's sha256 (EPFL_DATA_LOCK.json of gate 2)."""
        roles = set(roles)
        if not roles <= self.admitted:
            raise PermissionError(f"roles {sorted(roles - self.admitted)} not admitted at stage {self.stage}")
        if not self.data_lock:
            raise RuntimeError("no data lock given")
        checked = 0
        for role in sorted(roles):
            for w in self.included(role):
                entry = self.data_lock.get(w)
                if entry is None or entry["role"] != role:
                    raise RuntimeError(f"{w}: missing from the data lock or role differs")
                path = self.check(w, {role}, "data-lock integrity")
                if sha256_file(path) != entry["sha256"]:
                    raise RuntimeError(f"{w}: cache file changed since the data lock")
                checked += 1
        return {"checked": checked, "roles": sorted(roles)}
