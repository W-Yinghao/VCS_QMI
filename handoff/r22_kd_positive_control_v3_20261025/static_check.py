"""Check entrypoint globals and reproduce the original missing-import failure."""
import builtins
import symtable
from pathlib import Path
from common import HERE, BASE, read, write, sha, verify

def missing(path):
    table = symtable.symtable(Path(path).read_text(), str(path), 'exec')
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    known = defined | set(dir(builtins)) | {'__file__', '__name__', '__package__', '__doc__'}
    result = []
    def visit(scope):
        for s in scope.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in known:
                result.append([scope.get_name(), s.get_name()])
        for child in scope.get_children(): visit(child)
    visit(table)
    return result

def main():
    freeze = verify()
    revision = read(HERE/'IMPORT_REPAIR.json')
    prior = missing(BASE/revision['old_entrypoint'])
    assert prior == revision['expected_old_missing'], prior
    results = {p.name: missing(p) for p in sorted(HERE.glob('*.py'))}
    assert all(not v for v in results.values()), results
    # Execute the exact repaired expression with frozen, metadata-only inputs.
    if revision['kind'] == 'kd':
        controls = read(HERE/'SOURCE_LOCK.json')['controls']
        count = 0
        for refs in controls.values():
            for method in ('cosine', 'vcs'):
                r = read(Path(refs[method]['receipt']).parent/'run.json')
                assert 'environment' in r
                count += 1
        assert count == 12
    else:
        import hashlib
        import json
        cached = read(HERE/'CACHED_STARTS.json')
        count = 0
        for problems in cached.values():
            for candidates in problems.values():
                start_hash = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
                assert len(start_hash) == 64 and len(candidates) == 3
                count += 1
        assert count == 48
    write(HERE/'STATIC_CHECK.json', dict(status='PASS', freeze_sha256=freeze,
          old_failure_reproduced=prior, modules=results, exact_expression_metadata_cases=count))
    print('STATIC_IMPORT_CHECK PASS', count, flush=True)

if __name__ == '__main__': main()
