"""Run original frozen unit with only ESS diagnostic serialization canonicalized."""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import traceback
from recovery_common import *

def main():
    cfg=verify_bundle()
    require(bool(os.environ.get('SLURM_JOB_ID')),'Recovery requires Slurm')
    sys.path.insert(0,str(SPLIT));sys.path.insert(0,str(FULL/'source/src'))
    import split_common as S
    S.verify_control();S.check_old_absent(cfg['old_serial_job']);S.check_old_absent(cfg['failed_job'])
    unit=Path(cfg['unit'])
    with (unit/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        diagnosis=preflight(cfg)
        original_bytes=(Path(cfg['archive'])/'run_manifest.json').read_bytes()
        original=read(Path(cfg['archive'])/'run_manifest.json')
        write(BUNDLE/'PREFLIGHT_GPU.json',{'utc':utc(),'job':os.environ['SLURM_JOB_ID'],**diagnosis})
        # Failed constructor rewrote only this diagnostic field. Preserve that
        # evidence in the frozen bundle and use the original checkpoint metadata.
        unit.joinpath('run_manifest.json').write_bytes(original_bytes)
        from v9.n2.train import WeightedStudent
        original_init=WeightedStudent.__init__
        def guarded_init(self,*args,**kwargs):
            original_init(self,*args,**kwargs)
            require(self.out.resolve()==unit.resolve(),'Recovery escaped its single unit')
            recomputed=read(unit/'run_manifest.json')
            comparison=compare_manifest(original,recomputed)
            job=os.environ['SLURM_JOB_ID']
            write(BUNDLE/('RECOMPUTED_MANIFEST_'+job+'.json'),recomputed)
            write(BUNDLE/('CONSTRUCTOR_'+job+'.json'),{'utc':utc(),'job':job,**comparison,
                  'policy':'only ESS diagnostic canonicalized; every training field exactly equal',
                  'science_source_changed':False,'checkpoint_sha256':cfg['checkpoint_hashes']})
            unit.joinpath('run_manifest.json').write_bytes(original_bytes)
            self.manifest=original
        WeightedStudent.__init__=guarded_init
        try:
            spec=importlib.util.spec_from_file_location('frozen_split_unit',SPLIT/'run_unit.py')
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
            runner.main(cfg['group'],cfg['index'])
            path=unit/'full_result.json';result=read(path)
            result['metadata_resume_amendment']={'bundle':str(BUNDLE),'freeze_sha256':sha(BUNDLE/'FREEZE.json'),
                    'constructor':str(BUNDLE/('CONSTRUCTOR_'+os.environ['SLURM_JOB_ID']+'.json'))}
            write(path,result)
        finally:WeightedStudent.__init__=original_init

if __name__=='__main__':
    try:main()
    except BaseException:
        write(SPLIT/'units/903_1/0'/('FAILED_'+os.environ.get('SLURM_JOB_ID','unknown')+'.json'),
              {'utc':utc(),'job_id':os.environ.get('SLURM_JOB_ID'),'group':'903_1','unit_index':0,
               'traceback':traceback.format_exc()})
        raise
