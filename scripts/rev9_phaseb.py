"""REV9 — paired phase-B interventions on the frozen D1-20 selector units (reports/REV9/REV9_PROTOCOL_FROZEN_20261014.yaml).
Each new unit starts from the frozen unit's end-of-phase-A state (<tag>.resume.pt: estimator, optimiser, best, log, phase_a_done, RNG state), so
phase B sees the same estimator, question batches, budgets and Gumbel draws as the frozen unit; only the intervention differs:
  I0  unmodified rerun (guard: reproduces the frozen gate within tolerance)
  I1  the critic in eval mode during phase B (dropout off; REV8 R8-01)
  I2  I1 + straight-through hard top-k in the phase-B loss: forward = the deployed hard top-k of the gate logits, gradient = the frozen relaxed
      top-k's (REV8 R8-02)
The frozen modules (r7_d1_selectors, r7_d1_train) are imported unchanged; the interventions wrap SEL.gate_loss for phase B only.
    run --tag T --interventions I0 I1 I2 --d1 D --out O"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_selectors as SEL  # noqa: E402
import r7_d1_train as TR  # noqa: E402

ORIG_GATE_LOSS = SEL.gate_loss; ORIG_RELAXED = SEL.relaxed_topk


def gate_loss_I1(model, batch, k, tau, gen=None):
    was = model.training; model.eval()
    try:
        return ORIG_GATE_LOSS(model, batch, k, tau, gen)
    finally:
        model.train(was)


def gate_loss_I2(model, batch, k, tau, gen=None):
    def st_topk(g, kk, tt, gg=None):
        soft = ORIG_RELAXED(g, kk, tt, gg); hard = SEL.hard_topk(g, kk).to(soft.dtype)
        return hard + soft - soft.detach()
    SEL.relaxed_topk = st_topk
    try:
        return gate_loss_I1(model, batch, k, tau, gen)
    finally:
        SEL.relaxed_topk = ORIG_RELAXED


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--tag", required=True); r.add_argument("--interventions", nargs="+", choices=["I0", "I1", "I2"], required=True)
    r.add_argument("--d1", required=True); r.add_argument("--out", required=True); a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; D1 = Path(a.d1); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sel = D1 / "selectors"
    frozen = json.loads((sel / f"{a.tag}.json").read_text()); cfg = SEL.Cfg(**{k: v for k, v in frozen["cfg"].items() if k in SEL.Cfg.__dataclass_fields__})
    md, td, cache = D1 / "data", D1 / "teacher", D1 / "cache"
    meta = json.loads((md / "d1_manifest_meta.json").read_text()); wts = (meta["class_weights_from_FIT"]["w_yes"], meta["class_weights_from_FIT"]["w_no"])
    fit_recs, _ = TR.role_records(td, md, "FIT"); cal_recs, _ = TR.role_records(td, md, "CAL")
    fit, cal = TR.tuple_data(fit_recs, cache, "FIT", dev), TR.tuple_data(cal_recs, cache, "CAL", dev)
    base = torch.load(sel / f"{a.tag}.resume.pt", map_location="cpu", weights_only=False)
    if not base.get("phase_a_done"):
        raise SystemExit(f"{a.tag}: resume state is not at the end of phase A")
    for iv in a.interventions:
        name = f"{a.tag}_{iv}"
        if (out / f"{name}.pt").exists():
            print(f"[rev9] {name} exists", flush=True); continue
        shutil.copyfile(sel / f"{a.tag}.resume.pt", out / f"{name}.resume.pt")
        SEL.gate_loss = {"I0": ORIG_GATE_LOSS, "I1": gate_loss_I1, "I2": gate_loss_I2}[iv]
        try:
            model, log = SEL.train(cfg, frozen["objective"], fit, cal, wts, out / f"{name}.json", out / f"{name}.pt", resume_path=out / f"{name}.resume.pt")
        finally:
            SEL.gate_loss = ORIG_GATE_LOSS
        log["dynamic"] = {str(f): SEL.dynamic_tau(model, cal, cfg, f) for f in (0.25, 0.5, 0.75)}
        ref = torch.load(sel / f"{a.tag}.pt", map_location="cpu", weights_only=False)["state_dict"]; new = model.state_dict()
        est = max(float((new[k].cpu() - ref[k]).abs().max()) for k in ref if not k.startswith("gate."))
        gate = max(float((new[k].cpu() - ref[k]).abs().max()) for k in ref if k.startswith("gate."))
        log["rev9"] = {"intervention": iv, "frozen_unit": a.tag, "max_abs_diff_estimator_vs_frozen": est, "max_abs_diff_gate_vs_frozen": gate,
                       "frozen_soft_hard_gap": frozen.get("soft_hard_gap")}
        (out / f"{name}.json").write_text(json.dumps(log, indent=1, default=float))
        print(f"[rev9] {name}: estimator diff {est:.2e}, gate diff vs frozen {gate:.3e}, gap {json.dumps(log['soft_hard_gap'])[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
