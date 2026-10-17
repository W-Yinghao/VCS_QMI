"""Lightweight regression checks for scientific identity, selection and accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C


class Checks(unittest.TestCase):
    def test_complete_rejects_changed_identity_and_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);C.write(p/'checkpoint.json',{'weights':'a'})
            ident={'lr':1e-4}
            C.write(p/'COMPLETE.json',{'status':'COMPLETE','identity':ident,'steps':3000,
                'checkpoints':[{'step':250*(i+1)} for i in range(12)],
                'artifacts':{'checkpoint.json':C.sha(p/'checkpoint.json')}})
            self.assertIsNotNone(C.verify_complete(p,ident))
            with self.assertRaises(RuntimeError):C.verify_complete(p,{'lr':3e-4})
            C.write(p/'checkpoint.json',{'weights':'b'})
            with self.assertRaises(RuntimeError):C.verify_complete(p,ident)

    def test_partial_full_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            C.write(p/'COMPLETE.json',{'status':'COMPLETE','identity':{},'steps':2750,'checkpoints':[], 'artifacts':{}})
            with self.assertRaises(RuntimeError):C.verify_complete(p,{})

    def test_selection_priority_and_ties(self):
        a={'metrics':{'success42':.9,'success9':.8,'median_err':4},'step':500,'order':1}
        b=copy.deepcopy(a);b['metrics']['success42']=.89;b['metrics']['success9']=1
        self.assertLess(C.selection_key(a),C.selection_key(b))
        b=copy.deepcopy(a);b['metrics']['success9']=.79
        self.assertLess(C.selection_key(a),C.selection_key(b))
        b=copy.deepcopy(a);b['metrics']['median_err']=5
        self.assertLess(C.selection_key(a),C.selection_key(b))
        b=copy.deepcopy(a);b['step']=750
        self.assertLess(C.selection_key(a),C.selection_key(b))
        b=copy.deepcopy(a);b['order']=2
        self.assertLess(C.selection_key(a),C.selection_key(b))

    def test_failed_allocations_count_once_at_their_max_elapsed(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'ROOT',Path(d)):
            p=Path(d)/'COST_LEDGER.jsonl'
            for job,seconds in [('a',10),('a',20),('a',21),('b',100)]:
                C.append(p,{'family':'MF_VCS_M1','phase':'CAL','job_id':job,'allocated_seconds':seconds})
            C.append(p,{'family':'MF_VCS_M1','phase':'FINAL_DEV','job_id':'c','allocated_seconds':1000})
            self.assertEqual(C.spent_seconds('MF_VCS_M1'),121)

    def test_grid_and_strict_thresholds(self):
        self.assertEqual([len(C.candidates(f)) for f in C.FAMILIES],[3,3,9,3])
        self.assertEqual([(x['lr'],x['cap']) for x in C.candidates('MF_VCS_M1')],
            [(lr,cap) for lr in (1e-4,3e-4,1e-3) for cap in (10,30,100)])
        m=C.metrics([{'err_full':8},{'err_full':9},{'err_full':42}])
        self.assertEqual(m['success42'],2/3);self.assertEqual(m['success9'],1/3)
        with self.assertRaises(ValueError):C.metrics([{'err_full':float('nan')}])


if __name__=='__main__':unittest.main()
