"""Numerical failure diagnosis on frozen CAL input; no scientific release."""
import copy
import gc
import os
from pathlib import Path
import sys

SOURCE = Path('/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019')
sys.path[:0] = [str(SOURCE/'src'), str(SOURCE/'scripts/v10_supplement')]
from validate import verify_manifest
from v10.supplement.contract import candidates, read, sha
from v10.supplement.records import write, utc


def main():
    pin, manifest = verify_manifest()
    assert sha(__file__) == os.environ['V10_GEOMETRY_DIAGNOSTIC_SHA256']
    import torch
    import run as R
    from v10.supplement import checkpoints as CP
    from v10.supplement.registration import RegistrationSession
    from v10.supplement.training import new_trainer
    R.initialize_science()
    environment = R.runtime_environment()
    reports = []
    for family, job in [('MI_native', '1047324'), ('CoMIR_native', '1047326')]:
        guard = R.A.DataGuard()
        candidate = candidates(family)[0]
        trainer = new_trainer(family, candidate, 1117, guard, 'cuda')
        ident = R.identity(os.environ['V10_VALIDATION_SHA256'], family, candidate, 1117,
                           'VALIDATION_GATE', manifest['data_lock']['sha256'])
        CP.load(trainer, pin.parent/'runs'/job/family/'resume.pt', ident)
        if family == 'CoMIR_native': trainer.step()
        with CP.evaluation_state(trainer):
            core = guard.included('CAL')[0]
            session = RegistrationSession(family, trainer, guard, 'CAL', [core], 'cuda', candidate)
            energy, prep = session.energy(core, 0)
            p32 = torch.tensor([.37, .23, -.41], device='cuda', requires_grad=True)
            value32 = energy(p32); grad32 = torch.autograd.grad(value32, p32)[0]
            raw64 = copy.copy(energy.energy)
            for name in ('x', 'u', 'mov'):
                setattr(raw64, name, getattr(raw64, name).detach().double())
            p64 = p32.detach().double().requires_grad_()
            value64 = raw64(p64); grad64 = torch.autograd.grad(value64, p64)[0]
            sweep = []
            for dtype, fun, p, grad in [('fp32', energy, p32, grad32), ('fp64', raw64, p64, grad64)]:
                for h in [.01, .001, .0001, .00001, .000001, .0000005]:
                    eye = torch.eye(3, device=p.device, dtype=p.dtype)*h
                    with torch.no_grad():
                        fd = torch.stack([(fun(p.detach()+e)-fun(p.detach()-e))/(2*h) for e in eye])
                    sweep.append({'dtype': dtype, 'h': h, 'fd': fd.cpu().tolist(),
                                  'absolute_error': (fd-grad).abs().cpu().tolist()})
            coords = R.A.Q.apply_T(torch, p64[0:1]/R.A.Q.RHO, p64[1:3][None], raw64.x)
            boundary_distance = (coords-coords.round()).abs()
            report = {'family': family, 'original_gate_job': job, 'status': 'DIAGNOSTIC_ONLY',
                      'value_fp32': float(value32.detach()), 'value_fp64': float(value64.detach()),
                      'gradient_fp32': grad32.cpu().tolist(), 'gradient_fp64': grad64.cpu().tolist(),
                      'fp32_fp64_gradient_absolute_difference': (grad32.double()-grad64).abs().cpu().tolist(),
                      'minimum_distance_to_interpolation_boundary': float(boundary_distance.min().detach()),
                      'coordinates_within_001_of_boundary': int((boundary_distance < .01).sum()),
                      'finite_difference_sweep': sweep, 'data_roles': sorted({r['role'] for r in guard.reads})}
            reports.append(report)
            write(Path(__file__).parent/(family+'.json'), report)
            print(family, report['gradient_fp32'], report['gradient_fp64'], flush=True)
            del energy, raw64, session, coords, boundary_distance
        del trainer
        gc.collect(); torch.cuda.empty_cache()
    write(Path(__file__).parent/'DIAGNOSTIC.json', {'utc': utc(), 'job': os.environ['SLURM_JOB_ID'],
          'status': 'DIAGNOSTIC_ONLY', 'code_sha256': manifest['code_sha256'],
          'source_manifest_sha256': sha(pin), 'script_sha256': sha(__file__), 'environment': environment,
          'scientific_release': False, 'reports': reports})


if __name__ == '__main__': main()
