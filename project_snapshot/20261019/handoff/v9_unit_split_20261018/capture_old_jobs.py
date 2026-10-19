"""Read-only controller evidence for intentional checkpoint migrations."""
import json,os,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parent
pending={'1045063':'903_1','1045064':'903_2'}
start=time.monotonic()
while pending and time.monotonic()-start<1800:
 for job,g in list(pending.items()):
  r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=45,env={**os.environ,'TZ':'UTC'})
  rec={'utc':time.strftime('%FT%TZ',time.gmtime()),'query_timezone':'UTC','command':['scontrol','show','job','-o',job],
       'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'collector_job':os.environ['SLURM_JOB_ID']}
  p=b/'transitions'/g/'CONTROLLER_ORIGINAL.json'; p.parent.mkdir(parents=True,exist_ok=True)
  if r.returncode==0:
   p.write_text(json.dumps(rec,indent=2)+'\n')
   if any('JobState='+s in r.stdout for s in ('CANCELLED','COMPLETED','FAILED','TIMEOUT','NODE_FAIL')):
    pending.pop(job)
  else:
   (p.parent/'CONTROLLER_QUERY_ERROR.json').write_text(json.dumps(rec,indent=2)+'\n')
 if pending: time.sleep(15)
if pending: raise SystemExit('Original controller record not finalized: '+str(pending))
