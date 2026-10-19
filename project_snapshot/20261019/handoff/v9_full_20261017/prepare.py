"""One-time source/input freeze. Metadata/file hashing only; no training or Torch here."""
import ast
import datetime
import shutil
import subprocess
from pathlib import Path
from common import ROOT, sha, write


def prepare(supersede_unsubmitted=False):
    if (ROOT/'FROZEN.json').exists():
        if not supersede_unsubmitted or (ROOT/'LAUNCH.json').exists() or (ROOT/'VALIDATION.json').exists():
            raise RuntimeError('Already frozen; a submitted or evaluated protocol cannot be overwritten')
        shutil.copy2(ROOT/'FROZEN.json',ROOT/'FROZEN_UNSUBMITTED_v1.json')
        write(ROOT/'PRE_SUBMISSION_AMENDMENT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'prior_freeze_sha256':sha(ROOT/'FROZEN.json'),'prior_jobs_submitted':0,
              'reason':'Include strongest existing tuning comparator RuLSIF at SUM, 3 full seeds; 30 -> 33 full units. No new outcomes read.'})
    repo=ROOT.parents[1]/'ssl_pilot_v9'
    old=Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k')
    dest=ROOT/'source'
    shutil.copytree(repo/'src/v9',dest/'src/v9',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    shutil.copytree(repo/'v9/package_v9_20261016/reference',dest/'v9/package_v9_20261016/reference',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    shutil.copy2(repo/'reports/V9/HOST_LOCK.yaml',dest/'HOST_LOCK.yaml')
    cache=Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/token_cache')
    cache_names=['f30k_precomp_'+s+'_caps_tokens_f99364c3057a314f.json' for s in ('train','dev')]
    (dest/'token_cache').mkdir(exist_ok=True)
    for name in cache_names:shutil.copy2(cache/name,dest/'token_cache'/name)
    methods=['VCS_Weight','BCE_Weight','RankMass','Unweighted_A']
    groups=[]
    for seed in (901,902,903):
        for i,method in enumerate(methods):
            scales=['mean','sum'] if (seed+i)%2 else ['sum','mean']
            groups.append({'id':f'{seed}_{i+1}', 'seed':seed,'units':[{'method':method,'scale':s} for s in scales]})
        groups.append({'id':f'{seed}_or','seed':seed,'units':[{'method':'OracleMask','scale':'sum'},{'method':'OracleDrop','scale':'sum'}]})
        groups.append({'id':f'{seed}_ru','seed':seed,'units':[{'method':'RuLSIF_Weight','scale':'sum'}]})
    write(ROOT/'GROUPS.json',groups)
    inputs={}
    for seed in (901,902,903):
        base=old/'rho0.4'/f's{seed}'
        paths=[base/'pairs.observed_pairs.json',base/'eval_only/pairs.eval_only.json',base/'pairs.noise_manifest.json']
        for method in methods+['RuLSIF_Weight']:
            p=base/'weights'/f'{method}.csv'; paths.extend([p,Path(str(p)+'.lock.json')])
        for p in paths:inputs[str(p)]=sha(p)
    data=Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp')
    vocab=Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json')
    inputs[str(vocab)]=sha(vocab)
    data_files={}
    for split in ('train','dev'):
        for suffix in ('ims.npy','caps.txt'):
            p=data/f'{split}_{suffix}'; st=p.stat();data_files[str(p)]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    # AST-only syntax validation on the orchestration host.
    code={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='FROZEN.json':continue
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if p.suffix in ('.py','.sbatch','.md','.json','.yaml'):
            code[str(p.relative_to(ROOT))]=sha(p)
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_git':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
              'code_sha256':code,'input_sha256':inputs,'old_output':str(old),'data':str(data),'data_files':data_files,
              'data_identity_note':'Existing train/dev feature files checked by path/size/mtime; no new full feature checksum claim.',
              'vocab':str(vocab),'cache_names':cache_names,'full_units':33,'paired_groups':15,'single_groups':3,'max_normal_gpu':8}
    write(ROOT/'FROZEN.json',manifest)
    print('FROZEN',sha(ROOT/'FROZEN.json'),'files',len(code),'inputs',len(inputs),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--supersede-unsubmitted',action='store_true')
    prepare(ap.parse_args().supersede_unsubmitted)
