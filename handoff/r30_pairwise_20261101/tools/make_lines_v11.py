"""LINES_R30.txt (the deduplicated v1.1 + A1 execution list) from plan_A1/EXECUTION_PLAN_A1.csv: every RUN40 / RUN10 /
RUN10_CONTROL row once. Order (queue order only, not a result gate): stage-view FULL40 (H0/L0 and the A1-extended s1001 HN/HB),
the 10-epoch branches and controls, the FULL40 s1001 Ls/PN/PV/PB, then the A1 block of the f60_s1002/f60_s1003 L40S cells
(the 2 HH controls extended to 40 and the 6 seed-confirmation PN/PV/PB). Partition per hardware block (CELLS_R30.json);
normal QOS, nice 0, exclude node51/53/60."""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = json.load(open(os.path.join(ROOT, 'CELLS_R30.json')))
plan = list(csv.DictReader(open(os.path.join(ROOT, 'plan_A1', 'EXECUTION_PLAN_A1.csv'))))
run = [r for r in plan if r['action'] in ('RUN40', 'RUN10', 'RUN10_CONTROL')]
a1_block = [r for r in run if r['unit_id'] in ('r30v11_f60_s1002_HH', 'r30v11_f60_s1003_HH') or r['execution_class'] == 'FULL40_SEEDCONF_A1']
main = [r for r in run if r not in a1_block]
g1 = [r for r in main if r['action'] == 'RUN40' and r['stage_view'] == 'yes']
g2 = [r for r in main if r['action'] != 'RUN40']
g3 = [r for r in main if r['action'] == 'RUN40' and r['stage_view'] != 'yes']
lines = []
for r in g1 + g2 + g3 + a1_block:
    c = cells[f"f{int(r['noise_percent']):02d}_s{int(r['seed'])}"]
    t = '10:00:00' if r['action'] == 'RUN40' else ('03:00:00' if c['partition'] == 'RTX6000PRO' else '04:00:00')
    lines.append(f"sbatch --job-name={r['unit_id']} --partition={c['partition']} --exclude=node51,node53,node60 --qos=normal --nice=0 --time={t} "
                 f"{ROOT}/sbatch/r30v11_unit.sbatch {r['unit_id']}")
assert len(lines) == len({l.split()[1] for l in lines}) == 36 and len(g1 + g2 + g3) == 28 and len(a1_block) == 8
open(os.path.join(ROOT, 'LINES_R30.txt'), 'w').write('\n'.join(lines) + '\n')
print(len(lines), len(g1), len(g2), len(g3), len(a1_block))
