import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from audit_p134 import extract_selected,paired_changes,summaries

class ReaderChecks(unittest.TestCase):
    def test_selected_and_common_values(self):
        d={'cell':{'base':{'seed':0},'contamination':{'type':'clean','eps':0}},'truth':{'S':.8},'rows':[{'kind':'js','negative_construction':'product','selected':True,'posterior':{'J_eval':.7,'R_eval':.3,'posterior_mse':.1},'native':{'value':1.2}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fixture.json';p.write_text(json.dumps(d));r=extract_selected(p)[0]
            self.assertEqual(r['common_J'],.7);self.assertEqual(r['native_value'],1.2)
    def test_seed_pairing_and_target(self):
        rows=[]
        for seed in range(3):
            for method,base,delta in [('vcs',.1,.1),('js',.08,.15)]:
                for typ,eps,err in [('clean',0,base),('independent',.2,base+delta+seed*.001)]:
                    rows.append(dict(seed=seed,method=method,contamination=typ,epsilon=eps,posterior_mse=err,negative_construction='product'))
        p=paired_changes(rows);s=summaries(p)[0]
        self.assertEqual(len(p),3);self.assertAlmostEqual(s['degradation_difference_mean'],-.05)
        self.assertIn('contaminated_truth',p[0]['target'])
    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):paired_changes([dict(seed=0,method='vcs',contamination='independent',epsilon=.2,posterior_mse=.1,negative_construction='product')])
if __name__=='__main__':unittest.main()
