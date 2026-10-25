"""Reject undefined entrypoint globals before numerical/gpu work."""
import builtins
import symtable
from common import HERE,verify,write

def missing(p):
    t=symtable.symtable(p.read_text(),str(p),'exec')
    known={s.get_name() for s in t.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}|set(dir(builtins))|{'__name__','__file__','__doc__','__package__'}
    errors=[]
    def visit(scope):
        for s in scope.get_symbols():
            if s.is_referenced() and s.is_global() and s.get_name() not in known:errors.append([scope.get_name(),s.get_name()])
        for c in scope.get_children():visit(c)
    visit(t);return errors
def main():
    f=verify();rows={p.name:missing(p) for p in HERE.glob('*.py')};assert not any(rows.values()),rows
    write(HERE/'STATIC_CHECK.json',dict(status='PASS',freeze_sha256=f,modules=rows));print('STATIC PASS',flush=True)
if __name__=='__main__':main()
