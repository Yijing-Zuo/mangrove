import torch

G16 = 32767.0
H16 = 65535.0
G28 = float(2 ** 27 - 1)
H28 = float(2 ** 28 - 1)
KRTEPS = 1e-6
NEG_INF = float("-inf")
_U32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


def mix32(x):
    x = x & _U32
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & _U32
    x = ((x ^ (x >> 15)) * 0x846CA68B) & _U32
    return x ^ (x >> 16)


ROW_STREAM = 1
COL_STREAM = 2


def hash_u01(*keys):
    h = torch.zeros((), dtype=torch.int64, device=keys[0].device)
    for k in keys:
        h = mix32(h * 0x9E3779B1 + mix32(k.to(torch.int64) * 0x85EBCA77 + 0xC2B2AE3D))
    return h.to(torch.float64) / 4294967296.0


def grad_hess(loss, K, pred, y, mrow_base, mrow_len):
    if loss == "squared":
        g = pred - y
        return g, torch.ones_like(g)
    if loss == "logistic":
        p = (1.0 / (1.0 + torch.exp(-pred.to(torch.float64)))).to(torch.float32)
        g = p - y
        h = torch.clamp(p * (1.0 - p), min=1e-16)
        return g, h
    idx = mrow_base[None, :] + torch.arange(K, device=pred.device, dtype=torch.int64)[:, None] * mrow_len[None, :]
    Z = pred[idx].to(torch.float64)
    mx = Z[0].clone()
    for k in range(1, K):
        mx = torch.maximum(mx, Z[k])
    E = torch.exp(Z - mx[None, :])
    s = E[0].clone()
    for k in range(1, K):
        s = s + E[k]
    P = (E / s[None, :]).to(torch.float32)
    yk = y[mrow_base]
    onehot = (torch.arange(K, device=pred.device)[:, None] == yk[None, :]).to(torch.float32)
    g = torch.empty_like(pred)
    h = torch.empty_like(pred)
    g[idx.reshape(-1)] = (P - onehot).reshape(-1)
    h[idx.reshape(-1)] = torch.clamp(2.0 * P * (1.0 - P), min=1e-16).reshape(-1)
    return g, h


def slot_amax(x, slot_of_rt, S, rt_off=None):
    if rt_off is None or S == 0 or x.numel() == 0:
        return torch.zeros(S, dtype=x.dtype, device=x.device).scatter_reduce(0, slot_of_rt.to(torch.int64), x, "amax")
    return torch.segment_reduce(x, "max", offsets=rt_off, initial=0.0, unsafe=True)


def quantize(g, h, slot_of_rt, work, bits, S, has_hess=True, rt_off=None):
    dev = g.device
    slot64 = slot_of_rt.to(torch.int64)
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    gabs = torch.where(work, g.abs(), zero)
    gmax = slot_amax(gabs, slot_of_rt, S, rt_off)
    del gabs
    if has_hess:
        hw = torch.where(work, h, zero)
        hmax = slot_amax(hw, slot_of_rt, S, rt_off)
        del hw
    one32 = torch.ones_like(gmax)
    sg32 = torch.where(gmax > 0, torch.tensor(G16, dtype=torch.float32, device=dev) / gmax, one32)
    gq16 = torch.clamp(torch.round(g * sg32[slot64]), -G16, G16).to(torch.int64)
    g64 = gmax.to(torch.float64)
    sg28 = torch.where(g64 > 0, G28 / g64, torch.ones_like(g64))
    gs = g.to(torch.float64) * sg28[slot64]
    gq28 = torch.clamp((gs + torch.where(gs >= 0, 0.5, -0.5)).trunc().to(torch.int64), -(2 ** 27 - 1), 2 ** 27 - 1)
    is28 = (bits == 28)
    sg = torch.where(is28, sg28, sg32.to(torch.float64))
    gq = torch.where(is28[slot64], gq28, gq16)
    if has_hess:
        sh32 = torch.where(hmax > 0, torch.tensor(H16, dtype=torch.float32, device=dev) / hmax, one32)
        hq16 = torch.clamp(torch.round(h * sh32[slot64]), 0.0, H16).to(torch.int64)
        h64 = hmax.to(torch.float64)
        sh28 = torch.where(h64 > 0, H28 / h64, torch.ones_like(h64))
        hq28 = torch.clamp((h.to(torch.float64) * sh28[slot64] + 0.5).trunc().to(torch.int64), 0, 2 ** 28 - 1)
        sh = torch.where(is28, sh28, sh32.to(torch.float64))
        hq = torch.where(is28[slot64], hq28, hq16)
    else:
        sh = torch.ones(S, dtype=torch.float64, device=dev)
        hq = torch.ones_like(gq)
    gh = torch.where(work, gq * 4294967296 + hq, torch.zeros((), dtype=torch.int64, device=dev))
    return gh, sg, sh


def split_gh(gh):
    H = gh & _U32
    G = (gh - H) >> 32
    return G, H


def _expand_segments(beg, end):
    cnt = (end - beg).to(torch.int64)
    total = int(cnt.sum().item())
    seg_of = torch.repeat_interleave(torch.arange(len(cnt), device=beg.device), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(total, device=beg.device, dtype=torch.int64) - start[seg_of] + beg.to(torch.int64)[seg_of]
    return seg_of, pos, cnt


def build_hist(X_all, xrow, gh, wl, beg, end, dest, fmask, out, wide):
    if len(beg) == 0:
        return
    seg_of, pos, _ = _expand_segments(beg, end)
    q = wl[pos].to(torch.int64)
    xb = X_all[xrow[q].to(torch.int64)].to(torch.int64)
    d = X_all.shape[1]
    B = out.shape[-1]
    key = (dest.to(torch.int64)[seg_of][:, None] * d + torch.arange(d, device=X_all.device)[None, :]) * B + xb
    m = fmask[seg_of]
    key = key[m]
    v = gh[q][:, None].expand(-1, d)[m]
    if wide:
        G, H = split_gh(v)
        plane = out.shape[1] * d * B
        flat = out.view(-1)
        flat.index_add_(0, key, G)
        flat.index_add_(0, key + plane, H)
    else:
        out.view(-1).index_add_(0, key, v)


def node_sums(gh, wl, beg, end):
    if len(beg) == 0:
        z = torch.zeros(0, dtype=torch.int64, device=gh.device)
        return z, z.clone()
    seg_of, pos, cnt = _expand_segments(beg, end)
    v = gh[wl[pos].to(torch.int64)]
    G, H = split_gh(v)
    nb = len(beg)
    Gs = torch.zeros(nb, dtype=torch.int64, device=gh.device).index_add_(0, seg_of, G)
    Hs = torch.zeros(nb, dtype=torch.int64, device=gh.device).index_add_(0, seg_of, H)
    return Gs, Hs


def subtract_into(out, out_idx, parent, parent_idx, small, small_idx, wide):
    if len(out_idx) == 0:
        return
    if wide:
        out[:, out_idx] = parent[:, parent_idx] - small[:, small_idx]
    else:
        out[out_idx] = parent[parent_idx] - small[small_idx]


def decode_hist(hist, wide):
    if wide:
        return hist[0], hist[1]
    return split_gh(hist)


def split_search(G, H, Gt, Ht, isg, ish, lam, mcw, fmask):
    C, d, B = G.shape
    GL = torch.cumsum(G, dim=2)
    HL = torch.cumsum(H, dim=2)
    GR = Gt[:, None, None] - GL
    HR = Ht[:, None, None] - HL
    isg_ = isg[:, None, None]
    ish_ = ish[:, None, None]
    lam_ = lam[:, None, None]
    gl = GL.to(torch.float32) * isg_
    hl = HL.to(torch.float32) * ish_
    gr = GR.to(torch.float32) * isg_
    hr = HR.to(torch.float32) * ish_
    gt = Gt.to(torch.float32) * isg
    ht = Ht.to(torch.float32) * ish
    parent_term = (gt * gt / (ht + lam))[:, None, None]
    gain = gl * gl / (hl + lam_) + gr * gr / (hr + lam_) - parent_term
    ok = (hl >= mcw[:, None, None]) & (hr >= mcw[:, None, None]) & (HL > 0) & (HR > 0) & fmask[:, :, None]
    gain = torch.where(ok, gain, torch.tensor(NEG_INF, dtype=torch.float32, device=G.device))
    bbin = gain.argmax(dim=2)
    bgain = gain.gather(2, bbin[:, :, None]).squeeze(2)
    bfeat = bgain.argmax(dim=1)
    best_gain = bgain.gather(1, bfeat[:, None]).squeeze(1)
    best_bin = bbin.gather(1, bfeat[:, None]).squeeze(1)
    ar = torch.arange(C, device=G.device)
    GLb = GL[ar, bfeat, best_bin]
    HLb = HL[ar, bfeat, best_bin]
    return bfeat.to(torch.int32), best_bin.to(torch.int32), best_gain, GLb, HLb


def partition(X_all, xrow, wl, beg, end, feat, thr):
    P = len(beg)
    if P == 0:
        return beg.clone()
    seg_of, pos, cnt = _expand_segments(beg, end)
    q = wl[pos]
    x = X_all[xrow[q.to(torch.int64)].to(torch.int64), feat.to(torch.int64)[seg_of]].to(torch.int32)
    left = (x <= thr[seg_of]).to(torch.int64)
    lc = torch.cumsum(left, 0)
    start = torch.cumsum(cnt, 0) - cnt
    lc_excl = lc - left
    lc_start = lc_excl[start.clamp(max=max(len(lc_excl) - 1, 0))]
    nleft_before = lc_excl - lc_start[seg_of]
    nleft_seg = torch.zeros(P, dtype=torch.int64, device=wl.device).index_add_(0, seg_of, left)
    i_in_seg = torch.arange(len(pos), device=wl.device, dtype=torch.int64) - start[seg_of]
    beg64 = beg.to(torch.int64)[seg_of]
    newpos = torch.where(left > 0, beg64 + nleft_before, beg64 + nleft_seg[seg_of] + (i_in_seg - nleft_before))
    wl[newpos] = q
    return (beg.to(torch.int64) + nleft_seg).to(torch.int32)


def leaf_values(G, H, sg, sh, eta, lam):
    return (-eta * (G.to(torch.float64) / sg) / (H.to(torch.float64) / sh + lam)).to(torch.float32)


def leaf_update(pred, wl, beg, end, lv):
    if len(beg) == 0:
        return
    seg_of, pos, _ = _expand_segments(beg, end)
    q = wl[pos].to(torch.int64)
    pred[q] = pred[q] + lv[seg_of]


def route_update(X_all, xrow, pred, q, slot_of_q, levels, leaf_key, leaf_val, node_shift):
    if len(q) == 0:
        return
    nid = torch.zeros(len(q), dtype=torch.int64, device=q.device)
    xr = xrow[q].to(torch.int64)
    for key_sorted, feat, thr in levels:
        if len(key_sorted) == 0:
            continue
        key = (slot_of_q << node_shift) | nid
        j = torch.searchsorted(key_sorted, key).clamp_(max=len(key_sorted) - 1)
        hit = key_sorted[j] == key
        f = feat[j].to(torch.int64)
        x = X_all[xr, torch.where(hit, f, torch.zeros_like(f))].to(torch.int32)
        go_right = (x > thr[j]).to(torch.int64)
        nid = torch.where(hit, 2 * nid + 1 + go_right, nid)
    key = (slot_of_q << node_shift) | nid
    j = torch.searchsorted(leaf_key, key).clamp_(max=len(leaf_key) - 1)
    assert bool((leaf_key[j] == key).all()), "every routed row must land on a leaf"
    pred[q] = pred[q] + leaf_val[j]


def metric_fixed_point(loss, K, pred, y, is_eval, slot_of_rt, S, mrow_base, mrow_len):
    dev = pred.device
    slot64 = slot_of_rt.to(torch.int64)
    if loss == "softmax":
        idx = mrow_base[None, :] + torch.arange(K, device=dev, dtype=torch.int64)[:, None] * mrow_len[None, :]
        Z = pred[idx].to(torch.float64)
        mx = Z[0].clone()
        for k in range(1, K):
            mx = torch.maximum(mx, Z[k])
        E = torch.exp(Z - mx[None, :])
        s = E[0].clone()
        for k in range(1, K):
            s = s + E[k]
        yk = y[mrow_base].to(torch.int64)
        pk = E[yk, torch.arange(len(yk), device=dev)] / s
        row_loss = -torch.log(torch.clamp(pk, min=1e-15))
        ev = is_eval[mrow_base]
        slot_r = slot64[mrow_base]
    else:
        pf = pred.to(torch.float64)
        yf = y.to(torch.float64)
        if loss == "logistic":
            p = torch.clamp(1.0 / (1.0 + torch.exp(-pf)), 1e-15, 1 - 1e-15)
            row_loss = -(yf * torch.log(p) + (1 - yf) * torch.log(1 - p))
        else:
            row_loss = (pf - yf) ** 2
        ev = is_eval
        slot_r = slot64
    row_loss = torch.where(ev, row_loss, torch.zeros((), dtype=torch.float64, device=dev))
    count = torch.zeros(S, dtype=torch.int64, device=dev).index_add_(0, slot_r, ev.to(torch.int64))
    mx = torch.zeros(S, dtype=torch.float64, device=dev).scatter_reduce(0, slot_r, row_loss, "amax")
    denom = torch.clamp(mx * count.to(torch.float64), min=1.0)
    _, e = torch.frexp(2.0 ** 62 / denom)
    scale = ((e.to(torch.int64) - 1 + 1023) << 52).view(torch.float64)
    rq = torch.round(row_loss * scale[slot_r]).to(torch.int64)
    num = torch.zeros(S, dtype=torch.int64, device=dev).index_add_(0, slot_r, rq)
    return num, scale, count
