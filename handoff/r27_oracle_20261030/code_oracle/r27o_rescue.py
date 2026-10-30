"""R27-O (ORACLE diagnostic, never a SOTA table entry): correct-candidate rescue of the t2i noisy-branch candidates.

This is the ONLY module of the snapshot that reads correspondence truth. Flickr30K precomp truth (pre-injection
pairing): training caption with global index j belongs to training image j // 5 (5 captions per image); the noise
lock only replaces the OBSERVED image of a caption. Candidate image IDs in the R27 queue (ids['img']) are global
training image indices, the same ID space as j // 5 and as Ctx.gpu_images() rows.

Rule (t2i only, rows of the native noisy/unlabeled split only): if the query caption's true image is already among
its K = 5 candidate image IDs the row is unchanged; otherwise the K-th (5th-ranked, lowest similarity) slot's image ID
is replaced by the true image ID. Candidate count stays K, and cols / inv / A are rebuilt with the production weight
rule of r27_interface.search (each slot 1/K, duplicates merged by scatter_add), so P/Q construction, re-encoding,
GraphRefiner, SCE and the optimizer step are unchanged. The replaced slot carries slots = -1, cap = -1 (no queue
slot) and wepoch = current epoch (diagnostic field only)."""
import torch

O_ARMS = {
    'o1_rescue_native_t2i': {'i2t': 'identity', 't2i': 'identity'},
    'o2_rescue_vcs_t2i': {'i2t': 'identity', 't2i': 'cap_vcs'},
}
IM_DIV = 5          # Flickr30K precomp: 5 training captions per training image
ENABLED = True      # gate-only switch (rescue disabled -> O1 == A1, O2 == D2); production never changes it
_checked = [False]


def true_image(cap_gidx):
    """Truth: global training caption index -> its pre-noise training image index."""
    return torch.div(cap_gidx, IM_DIV, rounding_mode='floor')


def _check_id_space():
    if _checked[0]:
        return
    from r27_interface import Ctx
    assert Ctx.images is not None and Ctx.captions is not None
    assert len(Ctx.captions) == IM_DIV * Ctx.images.shape[0], (len(Ctx.captions), Ctx.images.shape)
    _checked[0] = True


@torch.no_grad()
def rescue_t2i(S, cap_gidx, k, epoch, stats=None):
    """S: dict from r27_interface.search(branch='t2i'); cap_gidx [B] global caption IDs of the noisy query batch.
    Returns (S_new, info). S_new is S itself when no row needs a rescue."""
    _check_id_space()
    img = S['img']
    B, K = img.shape
    assert K == k and cap_gidx.shape == (B,)
    tru = true_image(cap_gidx.to(img.device).long())
    present = (img == tru.unsqueeze(1)).any(1)
    need = ~present
    nres = int(need.sum())
    info = dict(rows=B, present=int(present.sum()), rescued=nres, need=need, truth=tru)
    if stats is not None:
        stats['cnt_t2i_rescue_rows'] += B
        stats['cnt_t2i_truth_present_rows'] += info['present']
        stats['cnt_t2i_rescued_rows'] += nres
    if nres == 0:
        return S, info
    img = img.clone()
    img[need, K - 1] = tru[need]
    slots = S['slots'].clone(); slots[need, K - 1] = -1
    cap = S['cap'].clone(); cap[need, K - 1] = -1
    wep = S['wepoch'].clone(); wep[need, K - 1] = int(epoch)
    cols, inv = torch.unique(img.flatten(), return_inverse=True)        # same rule as r27_interface.search
    inv = inv.view_as(img)
    A = torch.zeros(B, cols.numel(), device=S['A'].device, dtype=torch.float32)
    A.scatter_add_(1, inv, torch.full_like(inv, 1.0 / k, dtype=torch.float32))
    S_new = dict(S)
    S_new.update(slots=slots, cols=cols, inv=inv, A=A, wepoch=wep, cap=cap, img=img)
    return S_new, info
