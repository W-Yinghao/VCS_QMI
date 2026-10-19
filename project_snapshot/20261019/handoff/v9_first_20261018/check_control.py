import unittest
from unittest.mock import patch
import bridge as b

class Tests(unittest.TestCase):
 def test_pending_only_running_and_other_projects_preserved(self):
  rows=[{'id':'1','name':'v10p8_c2','state':'RUNNING'},{'id':'2','name':'v10p8_f3_1003','state':'PENDING'},{'id':'3','name':'v9f_903_1_u1','state':'PENDING'},{'id':'4','name':'c3_job','state':'PENDING'}]
  self.assertEqual([r['id'] for r in b.v10_pending(rows)],['2'])
 def test_complete_needs_both33_and_audit_pass(self):
  ready=lambda f,s,a:b.closeout_ready([],f,s,a)[0]
  feed={'full_complete':33,'failed_groups':[]}; status={'state':'PASS_V9_CLOSEOUT_INDEPENDENT'}; audit={'status':'PASS','units':[{}]*33}
  self.assertTrue(ready(feed,status,audit))
  self.assertFalse(ready({'full_complete':32,'failed_groups':[]},status,audit))
  self.assertFalse(ready({'full_complete':33,'failed_groups':['903_1']},status,audit))
  self.assertFalse(ready(feed,{'state':'AUDITING'},audit))
  self.assertFalse(ready(feed,status,{'status':'FAIL','units':[{}]*33}))
 def test_live_v9_or_auditor_blocks_release_but_bridge_does_not(self):
  args=({'full_complete':33,'failed_groups':[]},{'state':'PASS_V9_CLOSEOUT_INDEPENDENT'},{'status':'PASS','units':[{}]*33})
  for row in ({'id':'1','name':'v9f_903_1_u1'},{'id':'2','name':'v9_split_feed'},{'id':'1046165','name':'audit_v9_end'}):
   self.assertFalse(b.closeout_ready([row],*args)[0])
  self.assertTrue(b.closeout_ready([{'id':'3','name':'stage_bridge'}],*args)[0])
 def test_running_jobs_never_sent_to_hold(self):
  with patch.object(b,'read',return_value={'units':[{'job':'1'}]}),patch.object(b.subprocess,'run',side_effect=AssertionError('running job mutation')):
   b.hold_pending([{'id':'1','name':'v10p8_c2','state':'RUNNING'}],{})
 def test_start_race_preserved(self):
  row={'id':'1','name':'v10p8_c2','state':'PENDING','reason':'(Resources)'}
  ledger={'held_by_this_policy':{},'preexisting_holds':{},'raced_running_preserved':{}}
  with patch.object(b,'read',return_value={'units':[{'job':'1'}]}),patch.object(b,'queue',return_value=[{**row,'state':'RUNNING'}]),patch.object(b.subprocess,'run',side_effect=AssertionError('must preserve raced run')):
   b.hold_pending([row],ledger)
 def test_superseded_policy_never_releases(self):
  with patch.object(b,'verify_freeze'),patch.object(b,'validate_closeout'),patch.object(b,'policy_current',return_value=False),patch.object(b,'command',side_effect=AssertionError('no scheduler mutation')):
   with self.assertRaises(RuntimeError): b.release_after_closeout({}, {}, None)

if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 b.write(b.BUNDLE/'CONTROL_TESTS.json',{'utc':b.utc(),'status':'PASS' if r.wasSuccessful() else 'FAIL','test_methods':r.testsRun,'failure_count':len(r.failures),'error_count':len(r.errors),'scientific_training_executed':False})
 raise SystemExit(0 if r.wasSuccessful() else 1)
