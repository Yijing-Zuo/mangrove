import numpy as np

import binning
import io_utils as U
import sources

MAIN_SPECS = [
    ("adult", 1590, "binary", 50000, "occupation"),
    ("bank-marketing", 1461, "binary", 50000, "V2"),
    ("higgs-50k", 23512, "binary", 50000, None),
    ("california", 44025, "regression", 50000, None),
    ("letter", 6, "multiclass", 50000, None),
]
SPEC = {s[0]: s for s in MAIN_SPECS}


def prepare(name, out_dir, cache):
    _, did, task, nmax, gcol = SPEC[name]
    X, y, _, info = sources.fetch_sklearn(did, cache)
    enc = U.encode_frame(X)
    yv, keep, classes = U.encode_target(y, task, class_order="declared")
    idx = np.nonzero(keep)[0]
    n0 = len(idx)
    subsampled = n0 > nmax
    if subsampled:
        rng = np.random.default_rng(0)
        idx = idx[np.sort(rng.choice(n0, nmax, replace=False))]
    gcodes = None
    if gcol is not None:
        jg = enc["feature_names"].index(gcol)
        gcodes, glabels = enc["cat_cols"][jg]
        gcodes = gcodes[idx]
    enc = U.subset_rows(enc, idx)
    yv = yv[idx]
    enc, dropped = U.drop_constant_columns(enc)
    Xb_g, nb_g = binning.global_quantize(enc["Xnum"])
    folds = U.make_folds(yv, task)
    group = gnames = gcounts = None
    if gcodes is not None:
        group, gnames, gcounts = U.group_ids(gcodes, glabels)
    meta = U.write_dataset(
        out_dir, name, enc, yv, task, classes, info, folds=folds, group=group, group_names=gnames,
        group_counts=gcounts, group_source=(f"feature '{gcol}'" if gcol else None),
        xb_global=Xb_g, xb_global_rule="global_quantize", nbins_global=nb_g,
        extra_meta={"group_set": "main", "n_raw": int(n0), "subsampled": bool(subsampled),
                    "dropped_constant_columns": dropped, "class_order": "declared (pd.Categorical)"})
    return meta
