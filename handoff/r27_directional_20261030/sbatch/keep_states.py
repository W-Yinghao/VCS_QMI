"""Side watcher for one R27-D unit (protocol §6: keep epoch10/20 full states for later diagnostics, no extra forward/RNG).
state_last.pt is replaced atomically by the driver (tmp + os.replace). On every new inode we hardlink it (the link pins
that exact file), read its 'epoch' on CPU (mmap), keep it as state_epoch{E}.pt if E in KEEP, else drop the link.
Runs in the job shell, never inside the training process. Usage: keep_states.py OUT_DIR  (exits when DONE.json appears)"""
import os, sys, time, torch
KEEP = (10, 20)
out = sys.argv[1]; last = os.path.join(out, 'state_last.pt'); link = os.path.join(out, '.keep_link')
seen = None
while True:
    try:
        ino = os.stat(last).st_ino
    except FileNotFoundError:
        ino = None
    if ino is not None and ino != seen:
        seen = ino
        try:
            if os.path.exists(link): os.unlink(link)
            os.link(last, link)
            if os.stat(link).st_ino == ino:
                e = int(torch.load(link, map_location='cpu', mmap=True, weights_only=False)['epoch'])
                dst = os.path.join(out, f'state_epoch{e}.pt')
                if e in KEEP and not os.path.exists(dst):
                    os.rename(link, dst); print('kept', dst, flush=True)
            if os.path.exists(link): os.unlink(link)
        except Exception as ex:  # diagnostics only; never disturb the run
            print('keep_states warning', repr(ex), flush=True)
    if os.path.exists(os.path.join(out, 'DONE.json')) or all(os.path.exists(os.path.join(out, f'state_epoch{k}.pt')) for k in KEEP):
        break
    time.sleep(20)
