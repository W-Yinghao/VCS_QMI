"""P107 launch condition (frozen with the prereg): exit 0 only if the GPU smoke summary exists and every A cell COMPLETED 100 steps with finite
gradients, its stop/resume run COMPLETED, and resumed-vs-uninterrupted |ΔJ| ≤ 0.05 (P104 measured GPU run-to-run noise up to 0.017)."""
import glob, json, sys
fs = sorted(glob.glob("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/P107_GATE_SMOKE_*/summary.json"))
if not fs:
    sys.exit(1)
s = json.load(open(fs[-1])); cells = s.get("cells", s)
for v in ("AL1", "AL2", "AP1", "AP2", "AP3"):
    r = cells.get(v)
    if not r or r.get("status") != "COMPLETED" or not r.get("grads_finite"):
        print(f"{v}: not OK {r and r.get('status')}"); sys.exit(2)
    res = r.get("resume") or {}
    if res.get("status") != "COMPLETED" or max((res.get("J_absdiff") or {"x": 9}).values()) > 0.05:
        print(f"{v}: resume not OK {res}"); sys.exit(3)
print("P107 smoke OK:", fs[-1]); sys.exit(0)
