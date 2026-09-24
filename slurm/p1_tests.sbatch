#!/usr/bin/env bash
#SBATCH --job-name=vcs_p1_tests
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p1_tests_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P1 tests: 29 reference tests + server integration tests (CUDA visible)"
"$PYTHON" -m pytest -q tests/test_ssl_core.py --junitxml=reports/junit_P1_reference.xml 2>&1 | tee reports/P1_reference_tests.txt
rc1=${PIPESTATUS[0]}
"$PYTHON" -m pytest -q tests/test_integration.py --junitxml=reports/junit_P1_integration.xml -p no:cacheprovider 2>&1 | tee reports/P1_integration_tests.txt
rc2=${PIPESTATUS[0]}
"$PYTHON" - <<PY
import json, xml.etree.ElementTree as ET
out={}
for name in ("P1_reference","P1_integration"):
    r=ET.parse(f"reports/junit_{name}.xml").getroot()
    s=r if r.tag=="testsuite" else r.find("testsuite")
    out[name]={k:int(s.get(k,0)) for k in ("tests","failures","errors","skipped")}
    out[name]["cases"]={c.get("name"):("skipped" if c.find("skipped") is not None else "failed" if (c.find("failure") is not None or c.find("error") is not None) else "passed") for c in s.iter("testcase")}
json.dump(out,open("reports/P1_tests_summary.json","w"),indent=2); print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!="cases"} for k,v in out.items()}))
PY
echo "reference_rc=$rc1 integration_rc=$rc2 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
