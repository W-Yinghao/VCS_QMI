"""Informational (not gating): repeat 250 RECT steps of Flickr60 OBS in a new process and compare with the
acceptance run (1088291, node59) and the historical CR1 checkpoint."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import cr3_student as S
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); O = Path(sys.argv[1])
G = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/graphs/Flickr60/OBS.npz')
res = {'gpu': torch.cuda.get_device_name()}
for k in ('a', 'b'):
    S.train(CR1/'banks'/'Flickr60'/'fit_bank.npz', CR1/'Flickr60'/'prepared', G, O/k, 'PROBE', 'RECT', 'cuda', steps=250, check_frozen=False, saves=(0, 250))
ld = lambda p: torch.load(p, weights_only=False, map_location='cpu')['model']
A, B = ld(O/'a'/'step_000250.pt'), ld(O/'b'/'step_000250.pt')
C = ld(Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/acceptance/acceptance_1088291/reuse_Flickr60_OBS/step_000250.pt'))
D = ld(CR1/'Flickr60'/'students'/'OBS'/'step_000250.pt')
mx = lambda x, y: max(float((x[k]-y[k]).abs().max()) for k in x)
res.update(same_job_a_vs_b=mx(A, B), this_vs_acceptance_job=mx(A, C), this_vs_historical_CR1_L40S=mx(A, D))
(O/'PROBE.json').write_text(json.dumps(res, indent=1)); print(res)
