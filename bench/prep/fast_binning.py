from concurrent.futures import ThreadPoolExecutor

import numpy as np

import binning


def _lerp(a, b, t):
    diff = b - a
    out = a + diff * t
    hi = t >= 0.5
    out[hi] = (b - diff * (1 - t))[hi]
    return out


def _pack(fit_masks):
    F = len(fit_masks)
    assert F <= 16
    packed = np.zeros(len(fit_masks[0]), dtype=np.uint16)
    for k in range(F):
        packed |= fit_masks[k].astype(np.uint16) << k
    return packed


def _mixed_zero(s):
    lo, hi = np.searchsorted(s, 0.0, side="left"), np.searchsorted(s, 0.0, side="right")
    if hi - lo < 2:
        return False
    z = np.signbit(s[lo:hi])
    return bool(z.any() and not z.all())


def cuts_from_sorted(s, max_bin, qs):
    m = len(s)
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    ne = s[1:] != s[:-1]
    if 1 + int(np.count_nonzero(ne)) <= max_bin - 1:
        return s[np.concatenate(([True], ne))][:-1]
    vi = (m - 1) * qs
    lo = np.floor(vi).astype(np.intp)
    return np.unique(_lerp(s[lo], s[lo + 1], vi - lo))


def _categorical(j, codes, labels, fit_masks, max_bin, cols, nbf, cuts):
    for k, mask in enumerate(fit_masks):
        x = binning.cat_codes_by_freq(codes, labels, mask)
        has_nan = bool(np.isnan(x).any())
        c = binning.fit_cuts(x[mask], max_bin)
        b = binning.apply_cuts(x, c, has_nan)
        nb = len(c) + 1 + (1 if has_nan else 0)
        assert nb <= 256 and (len(b) == 0 or b.max() < nb)
        cols[k][j], nbf[k, j], cuts[k][j] = b.astype(np.uint8), nb, c


def _assemble(cols, F, n, d, nbf, cuts, workers):
    Xbf = np.empty((F, n, d), dtype=np.uint8)

    def one(k):
        for j in range(d):
            Xbf[k, :, j] = cols[k][j]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, F))) as ex:
        list(ex.map(one, range(F)))
    ptr, val = binning.cuts_to_csr(cuts)
    return Xbf, nbf, ptr, val


def fold_local_mask_sortonce(Xnum, fit_masks, cat_cols=None, max_bin=binning.MAX_BIN, workers=12):
    cat_cols = cat_cols or {}
    n, d = Xnum.shape
    F = len(fit_masks)
    qs = np.linspace(0, 1, max_bin)[1:-1]
    packed = _pack(fit_masks)
    cols = [[None] * d for _ in range(F)]
    cuts = [[None] * d for _ in range(F)]
    nbf = np.zeros((F, d), dtype=np.int16)

    def numeric(j):
        x = Xnum[:, j]
        if x.dtype != np.float64:
            x = x.astype(np.float64)
        isnan = np.isnan(x)
        nn = int(np.count_nonzero(isnan))
        has_nan = nn > 0
        order = np.argsort(x, kind="quicksort")[:n - nn]
        xs = x[order]
        ps = packed[order]
        mixed = _mixed_zero(xs)
        for k in range(F):
            c = binning.fit_cuts(x[fit_masks[k]], max_bin) if mixed else cuts_from_sorted(xs[((ps >> k) & 1).astype(bool)], max_bin, qs)
            pos = np.searchsorted(xs, c, side="right")
            bs = np.repeat(np.arange(len(c) + 1, dtype=np.uint8), np.diff(np.concatenate(([0], pos, [len(xs)]))))
            b = np.empty(n, dtype=np.uint8)
            b[order] = bs
            if has_nan:
                b[isnan] = len(c) + 1
            nb = len(c) + 1 + (1 if has_nan else 0)
            assert nb <= 256
            cols[k][j], nbf[k, j], cuts[k][j] = b, nb, c

    def task(j):
        if j in cat_cols:
            _categorical(j, *cat_cols[j], fit_masks, max_bin, cols, nbf, cuts)
        else:
            numeric(j)

    if workers == 1:
        for j in range(d):
            task(j)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(task, range(d)))
    return _assemble(cols, F, n, d, nbf, cuts, workers)


def fold_local_mask_gpu(Xnum, fit_masks, cat_cols=None, max_bin=binning.MAX_BIN, device="cuda", workers=12):
    import torch
    cat_cols = cat_cols or {}
    n, d = Xnum.shape
    F = len(fit_masks)
    dev = torch.device(device)
    qs = np.linspace(0, 1, max_bin)[1:-1]
    packed = torch.from_numpy(_pack(fit_masks).astype(np.int32)).to(dev)
    cols = [[None] * d for _ in range(F)]
    cuts = [[None] * d for _ in range(F)]
    nbf = np.zeros((F, d), dtype=np.int16)
    first = torch.ones(1, dtype=torch.bool, device=dev)
    zero = torch.zeros(1, dtype=torch.int64, device=dev)

    def numeric(j):
        xn = Xnum[:, j]
        if xn.dtype != np.float64:
            xn = xn.astype(np.float64)
        elif not xn.flags.c_contiguous:
            xn = xn.copy()
        x = torch.from_numpy(xn).to(dev)
        isnan = torch.isnan(x)
        has_nan = bool(isnan.any())
        if has_nan:
            idx = torch.nonzero(~isnan, as_tuple=False).squeeze(1)
            xs, perm = torch.sort(x[idx])
            order = idx[perm]
        else:
            xs, order = torch.sort(x)
        z = xs == 0
        mixed = False
        if bool(z.any()):
            sb = torch.signbit(xs[z])
            mixed = bool(sb.any()) and not bool(sb.all())
        ps = packed[order]
        nv = xs.numel()
        last = torch.tensor([nv], dtype=torch.int64, device=dev)
        B = torch.empty((F, n), dtype=torch.uint8, device=dev)
        for k in range(F):
            if mixed:
                c = binning.fit_cuts(xn[fit_masks[k]], max_bin)
            else:
                s = xs[((ps >> k) & 1).bool()]
                m = s.numel()
                if m == 0:
                    c = np.zeros(0, dtype=np.float64)
                else:
                    ne = s[1:] != s[:-1]
                    if 1 + int(ne.sum()) <= max_bin - 1:
                        c = s[torch.cat((first, ne))][:-1].cpu().numpy()
                    else:
                        vi = (m - 1) * qs
                        lo = np.floor(vi).astype(np.int64)
                        got = s[torch.from_numpy(np.concatenate((lo, lo + 1))).to(dev)].cpu().numpy()
                        c = np.unique(_lerp(got[:len(lo)], got[len(lo):], vi - lo))
            ct = torch.from_numpy(c).to(dev)
            pos = torch.searchsorted(xs, ct, right=True)
            edges = torch.cat((zero, pos, last))
            bs = torch.repeat_interleave(torch.arange(len(c) + 1, dtype=torch.uint8, device=dev), edges[1:] - edges[:-1])
            Bk = B[k]
            Bk[order] = bs
            if has_nan:
                Bk[isnan] = len(c) + 1
            nb = len(c) + 1 + (1 if has_nan else 0)
            assert nb <= 256
            nbf[k, j], cuts[k][j] = nb, c
        Bn = B.cpu().numpy()
        for k in range(F):
            cols[k][j] = Bn[k]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_categorical, j, *cat_cols[j], fit_masks, max_bin, cols, nbf, cuts) for j in cat_cols]
        for j in range(d):
            if j not in cat_cols:
                numeric(j)
        for f in futs:
            f.result()
    return _assemble(cols, F, n, d, nbf, cuts, workers)
