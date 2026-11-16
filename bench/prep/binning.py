import numpy as np

MAX_BIN = 256


def cat_codes_by_freq(codes, labels, fit_mask):
    codes = np.asarray(codes, dtype=np.int64)
    K = len(labels)
    slab = np.array([str(x) for x in labels], dtype=object).astype(str) if K else np.zeros(0, dtype=str)
    miss = codes < 0
    cnt = np.bincount(codes[fit_mask & ~miss], minlength=K)
    present = np.nonzero(cnt > 0)[0]
    order = np.lexsort((slab[present], -cnt[present]))
    rank = np.full(K + 1, np.nan)
    rank[present[order]] = np.arange(len(present), dtype=np.float64)
    out = rank[np.where(miss, K, codes)]
    return out


def fit_cuts(x_fit, max_bin=MAX_BIN):
    x = x_fit[~np.isnan(x_fit)]
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    u = np.unique(x)
    if len(u) <= max_bin - 1:
        return u[:-1].astype(np.float64)
    qs = np.quantile(x, np.linspace(0, 1, max_bin)[1:-1], method="linear")
    return np.unique(qs)


def apply_cuts(x, cuts, has_missing_bin):
    b = np.searchsorted(cuts, x, side="left").astype(np.int64)
    miss = np.isnan(x)
    if miss.any():
        assert has_missing_bin
        b[miss] = len(cuts) + 1
    return b


def bin_matrix(Xnum, fit_mask, cat_cols=None, max_bin=MAX_BIN):
    cat_cols = cat_cols or {}
    n, d = Xnum.shape
    Xb = np.zeros((n, d), dtype=np.uint8)
    nbins = np.zeros(d, dtype=np.int16)
    cuts_all = []
    for j in range(d):
        if j in cat_cols:
            codes, labels = cat_cols[j]
            x = cat_codes_by_freq(codes, labels, fit_mask)
        else:
            x = Xnum[:, j]
            if x.dtype != np.float64:
                x = x.astype(np.float64)
        has_nan = bool(np.isnan(x).any())
        cuts = fit_cuts(x[fit_mask], max_bin)
        b = apply_cuts(x, cuts, has_nan)
        nb = len(cuts) + 1 + (1 if has_nan else 0)
        assert nb <= 256 and (len(b) == 0 or b.max() < nb)
        Xb[:, j] = b.astype(np.uint8)
        nbins[j] = nb
        cuts_all.append(cuts)
    return Xb, nbins, cuts_all


def cuts_to_csr(cuts_per_fold):
    F = len(cuts_per_fold)
    d = len(cuts_per_fold[0])
    ptr = np.zeros((F, d + 1), dtype=np.int64)
    for k in range(F):
        ptr[k, 1:] = np.cumsum([len(c) for c in cuts_per_fold[k]])
    T = int(ptr[:, -1].max())
    val = np.full((F, T), np.nan, dtype=np.float32)
    for k in range(F):
        if T:
            cat = np.concatenate(cuts_per_fold[k]).astype(np.float32)
            val[k, :len(cat)] = cat
    return ptr, val


def global_quantize(Xf, max_bins=256):
    n, d = Xf.shape
    Xb = np.zeros((n, d), dtype=np.uint8)
    nbins = []
    for j in range(d):
        x = Xf[:, j].copy()
        miss = np.isnan(x)
        if miss.any():
            lo = np.nanmin(x) if (~miss).any() else 0.0
            x[miss] = lo - 1.0
        u = np.unique(x)
        if len(u) <= max_bins:
            b = np.searchsorted(u, x, side="left")
        else:
            qs = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
            cuts = np.unique(qs)
            b = np.searchsorted(cuts, x, side="right")
        Xb[:, j] = b.astype(np.uint8)
        nbins.append(int(b.max()) + 1)
    return Xb, nbins


def fold_local(Xnum, folds, n_folds, cat_cols=None, max_bin=MAX_BIN):
    return fold_local_mask(Xnum, [folds != k for k in range(n_folds)], cat_cols, max_bin)


def fold_local_mask(Xnum, fit_masks, cat_cols=None, max_bin=MAX_BIN):
    n, d = Xnum.shape
    n_folds = len(fit_masks)
    Xbf = np.zeros((n_folds, n, d), dtype=np.uint8)
    nbf = np.zeros((n_folds, d), dtype=np.int16)
    cuts = []
    for k in range(n_folds):
        Xb, nb, c = bin_matrix(Xnum, fit_masks[k], cat_cols, max_bin)
        Xbf[k] = Xb
        nbf[k] = nb
        cuts.append(c)
    ptr, val = cuts_to_csr(cuts)
    return Xbf, nbf, ptr, val
