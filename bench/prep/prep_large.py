import numpy as np

import io_utils as U
import sources

LARGE_SPECS = {
    "higgs-100k": dict(did=42769, task="binary", nrows=100_000),
    "higgs-250k": dict(did=42769, task="binary", nrows=250_000),
    "higgs-500k": dict(did=42769, task="binary", nrows=500_000),
    "higgs-1m": dict(did=42769, task="binary", nrows=1_000_000),
    "higgs45-250k": dict(did=45570, task="binary", nrows=250_000, big=True),
    "higgs45-500k": dict(did=45570, task="binary", nrows=500_000, big=True),
    "higgs45-1m": dict(did=45570, task="binary", nrows=1_000_000, big=True),
    "higgs-2m": dict(did=45570, task="binary", nrows=2_000_000, big=True),
    "higgs-4m": dict(did=45570, task="binary", nrows=4_000_000, big=True),
    "higgs-11m": dict(did=45570, task="binary", nrows=11_000_000, big=True),
    "covertype": dict(did=1596, task="multiclass", group="wilderness"),
    "year": dict(did=44027, task="regression"),
    "airlines": dict(did=1169, task="binary", group="Airline"),
}


NESTED_WITH_42769 = False


def higgs_big_rows(N, n_src):
    if N >= n_src:
        return np.arange(n_src)
    return np.sort(np.random.default_rng(0).permutation(n_src)[:N])


def prepare_higgs_big(name, out_dir, cache):
    sp = LARGE_SPECS[name]
    H = sources.Higgs11M(cache)
    idx = higgs_big_rows(sp["nrows"], H.n)
    full = len(idx) == H.n
    yraw = H.target_values()
    yraw = yraw if full else yraw[idx]
    yv, keep, classes = U.encode_target(yraw, "binary")
    assert keep.all() and len(classes) == 2, classes
    del yraw
    d = len(H.feature_names)
    X = np.empty((len(idx), d), dtype=np.float64, order="F")
    for j, c in enumerate(H.feature_names):
        v = np.asarray(H.column(c), dtype=np.float64)
        X[:, j] = v if full else v[idx]
        del v
    enc = {"Xnum": X, "cat_cols": {}, "feature_names": list(H.feature_names), "categorical_features": []}
    enc, dropped = U.drop_constant_columns(enc)
    extra = {"group_set": "scaling", "n_source_rows": int(H.n), "dropped_constant_columns": dropped,
             "class_order": "natural (numeric if labels parse as numbers, else str)",
             "subset_rule": "all rows" if full else "sorted(default_rng(0).permutation(n_source)[:N])",
             "nested": "higgs45-250k c higgs45-500k c higgs45-1m c higgs-2m c higgs-4m c higgs-11m (row_id); "
                       "not nested with higgs-100k..1m (OpenML 42769)",
             "nested_with_42769": NESTED_WITH_42769, "row_id_note": "row_id omitted: identity" if full else None}
    return U.write_dataset(out_dir, name, enc, yv, "binary", classes, H.info, row_id=None if full else idx,
                           extra_meta=extra, outer="mask")


def prepare(name, out_dir, cache):
    sp = LARGE_SPECS[name]
    if sp.get("big"):
        return prepare_higgs_big(name, out_dir, cache)
    X, y, cat_ind, info = sources.fetch(sp["did"], cache)
    task = sp["task"]
    enc = U.encode_frame(X, cat_ind)
    yv, keep, classes = U.encode_target(y, task)
    idx = np.nonzero(keep)[0]
    n_full = int(len(X))
    extra = {"group_set": "scaling", "n_source_rows": n_full}
    row_id = None
    if name.startswith("higgs") and sp.get("nrows"):
        perm = np.random.default_rng(0).permutation(len(idx))
        sel = np.sort(perm[: sp["nrows"]])
        idx = idx[sel]
        row_id = idx
        extra.update(subset_rule="sorted(default_rng(0).permutation(n_source)[:N]) -- nested across sizes",
                     full_higgs_rows=11_000_000, full_higgs_openml_id=45570)
    group = gnames = gcounts = gsrc = None
    if sp.get("group") == "wilderness":
        cols = [j for j, f in enumerate(enc["feature_names"]) if "ilderness" in f]
        assert len(cols) == 4, enc["feature_names"]
        W = np.stack([X.iloc[:, j].astype(str).astype(float).to_numpy() for j in cols], axis=1)
        wc = np.argmax(W, axis=1).astype(np.int64)
        wc[W.max(1) <= 0] = -1
        group, gnames, gcounts = U.group_ids(wc[idx], [enc["feature_names"][j] for j in cols])
        gsrc = "argmax of the 4 Wilderness_Area one-hot columns"
    elif sp.get("group"):
        jg = enc["feature_names"].index(sp["group"])
        codes, labels = enc["cat_cols"][jg]
        group, gnames, gcounts = U.group_ids(codes[idx], labels)
        gsrc = f"feature '{sp['group']}'"
    enc = U.subset_rows(enc, idx)
    yv = yv[idx]
    enc, dropped = U.drop_constant_columns(enc)
    extra["dropped_constant_columns"] = dropped
    extra["class_order"] = "natural (numeric if labels parse as numbers, else str)"
    return U.write_dataset(out_dir, name, enc, yv, task, classes, info, group=group, group_names=gnames,
                           group_counts=gcounts, group_source=gsrc, row_id=row_id, extra_meta=extra,
                           outer="mask")
