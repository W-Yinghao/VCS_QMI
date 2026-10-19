"""Control-flow checks only: no training, Torch, or scheduler mutation."""
import ast
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import split_common as sc

class ControlTests(unittest.TestCase):
    def test_original_training_invocation_unchanged(self):
        def calls(path,name):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id==name)
                      or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
        for name in ('WeightedStudent','fit','load_host_lock','load_precomp','verify_input'):
            self.assertEqual(calls(sc.ROOT/'run_full.py',name),calls(sc.BUNDLE/'run_unit.py',name))
        s=(sc.BUNDLE/'run_unit.py').read_text()
        self.assertNotIn("write(out / 'DONE.json'",s)
        self.assertIn("group = {**group, 'units': [group['units'][unit_index]]}",s)
        self.assertIn('fcntl.LOCK_EX | fcntl.LOCK_NB',s)

    def test_no_old_writer_overlap(self):
        with patch.object(sc,'queue',return_value=[{'id':'1045063'}]):
            with self.assertRaises(RuntimeError): sc.check_old_absent('1045063')

    def test_live_job_is_not_complete(self):
        with patch.object(sc,'verify_full',side_effect=AssertionError('must not read live artifacts')):
            self.assertIsNone(sc.verify_unit_completion('903_1',0,{'job':'55'},{'55'}))

    def test_missing_or_nonzero_exit_blocks_complete(self):
        for ex in ({},{'job_id':'55','exit_code':143},{'job_id':'99','exit_code':0}):
            with patch.object(sc,'unit_exit',return_value=ex):
                with self.assertRaises(RuntimeError): sc.verify_unit_completion('903_1',0,{'job':'55'},set())

    def test_partial_group_never_finalizes(self):
        with patch.object(sc,'write',side_effect=AssertionError('no writes allowed')):
            with self.assertRaises(RuntimeError):
                sc.finalize_group('903_1',{'0':{'status':'COMPLETE_VERIFIED'},'1':{'status':'SUBMITTED'}},{})

    def test_group_preserves_real_worker_exit_and_proofs(self):
        writes={}
        units={str(i):{'status':'COMPLETE_VERIFIED','job':str(55+i),
               'completion':{'exit':{'job_id':str(55+i),'exit_code':0,'utc':f'2026-10-19T0{i}:00:00Z'}}} for i in (0,1)}
        launch={'903_1':{'job':'1045063'}}
        with patch.object(sc,'write',side_effect=lambda p,o:writes.update({str(p):copy.deepcopy(o)})):
            sc.finalize_group('903_1',units,launch)
        ex=writes[str(sc.ROOT/'results/903_1/SCHED_EXIT.json')]
        done=writes[str(sc.ROOT/'results/903_1/DONE.json')]
        self.assertEqual(ex['job_id'],'56'); self.assertEqual(ex['utc'],'2026-10-19T01:00:00Z')
        self.assertEqual(len(ex['unit_exits']),2); self.assertEqual(len(done['units']),2)
        self.assertEqual(launch['903_1']['job'],'56')

    def test_capacity_counts_cpu_and_expanded_tasks_but_not_cpu_as_gpu(self):
        cfg={'max_normal_jobs':29,'max_own_live':16,'target_pending':8}
        row={'name':'other','qos':'normal','tres':'gres/gpu:0','state':'RUNNING'}
        self.assertTrue(sc.capacity([dict(row) for _ in range(28)],cfg))
        self.assertFalse(sc.capacity([dict(row) for _ in range(29)],cfg))
        pend={'name':'v9f_x','qos':'normal','tres':'gres/gpu:1','state':'PENDING'}
        self.assertFalse(sc.capacity([dict(pend) for _ in range(8)],cfg))

    def test_existing_closeout_waits_for_split_jobs(self):
        spec=importlib.util.spec_from_file_location('audit_check',sc.BUNDLE.parent/'v9_overlap_v10_20261018/bridge.py')
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        groups=sc.read(sc.ROOT/'GROUPS.json'); launch={g['id']:{'job':'1'} for g in groups}
        args=(groups,launch,[g['id'] for g in groups],{'full_complete':33,'failed_groups':[]})
        self.assertTrue(m.readiness([],*args)['ready'])
        self.assertFalse(m.readiness([{'id':'99','name':'v9f_903_1_u1'}],*args)['ready'])
        self.assertFalse(m.readiness([{'id':'99','name':'v9_split_feed'}],*args)['ready'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests))
    sc.write(sc.BUNDLE/'CONTROL_TESTS.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
             'utc':sc.utc(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
             'scientific_training_executed':False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
