"""Check the handoff structure, task DAG, six-run scope and text hygiene."""
from pathlib import Path
import json


def main():
    root=Path(__file__).resolve().parents[1]
    tasks=json.loads((root/'machine/tasks.json').read_text())
    runs=json.loads((root/'machine/proposed_training_runs.json').read_text())
    ids={t['id'] for t in tasks['tasks']}
    assert len(ids)==len(tasks['tasks'])
    assert sum(t['new_training_runs'] for t in tasks['tasks'])==6
    assert len(runs)==6 and len({r['logical_id'] for r in runs})==6
    assert not tasks['official_evaluation_authorized']
    by={t['id']:t for t in tasks['tasks']}
    visited,active=set(),set()
    def visit(i):
        assert i not in active, 'cyclic dependency'
        if i in visited:return
        active.add(i)
        for j in by[i]['depends_on']:
            assert j in ids
            visit(j)
        active.remove(i);visited.add(i)
    for i in ids:visit(i)
    for ds in ('refcocog','refcocoplus'):
        assert sorted(r['seed'] for r in runs if r['dataset']==ds)==[0,1,2]
    assert all(r['initialize_from']=='original_pretrained' and r['both_towers_trainable'] for r in runs)
    assert by['R5-COND-SEM']['mode']=='design_only_not_authorized'
    for name in ('00_README_CN.md','01_SERVER_AGENT_PROMPT_CN.md','08_R4_TO_R5_DELTA_CN.md',
                 'docs/02_SERVER_PLAN_CN.md','docs/03_TARGETS_AND_LOSSES_CN.md',
                 'docs/04_DIAGNOSTICS_AND_FUSION_CN.md','docs/05_EXECUTION_STATS_AND_DECISIONS_CN.md',
                 'docs/06_EVIDENCE_AND_SOURCES_CN.md','sources/repo_source_manifest.json'):
        assert (root/name).is_file(),name
    bad=[]
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.md','.py','.json','.yaml','.tex','.sh'):
            text=path.read_text(encoding='utf-8')
            controls=[ord(c) for c in text if (ord(c)<32 and c not in '\n\r\t') or ord(c)==127]
            if controls:bad.append((str(path.relative_to(root)),controls))
    assert not bad,bad
    print('Package structure, task DAG, six-run scope and text hygiene: PASS')


if __name__=='__main__':main()
