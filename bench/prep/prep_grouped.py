import numpy as np

import io_utils as U
import sources

GROUPED_SPECS = {
    "airlines-airport": dict(did=1169, task="binary", group="AirportFrom"),
    "nyc-taxi-dolocation": dict(did=42729, task="regression", group="DOLocationID"),
    "house-sales-zipcode": dict(did=42731, task="regression", group="zipcode"),
}


def _drop_column(enc, j):
    keep = [i for i in range(enc["Xnum"].shape[1]) if i != j]
    remap = {o: i for i, o in enumerate(keep)}
    out = {"Xnum": enc["Xnum"][:, keep],
           "cat_cols": {remap[c]: v for c, v in enc["cat_cols"].items() if c != j},
           "feature_names": [enc["feature_names"][i] for i in keep]}
    out["categorical_features"] = sorted(out["cat_cols"])
    return out


def prepare(name, out_dir, cache):
    sp = GROUPED_SPECS[name]
    X, y, cat_ind, info = sources.fetch(sp["did"], cache)
    task = sp["task"]
    enc = U.encode_frame(X, cat_ind)
    yv, keep, classes = U.encode_target(y, task)
    idx = np.nonzero(keep)[0]
    jg = enc["feature_names"].index(sp["group"])
    if jg in enc["cat_cols"]:
        codes, labels = enc["cat_cols"][jg]
    else:
        v = enc["Xnum"][:, jg]
        labels, codes = np.unique(np.nan_to_num(v, nan=-1), return_inverse=True)
        codes = np.where(np.isnan(v), -1, codes)
        labels = list(labels)
    group, gnames, gcounts = U.group_ids(np.asarray(codes)[idx], labels)
    enc = _drop_column(enc, jg)
    enc = U.subset_rows(enc, idx)
    yv = yv[idx]
    enc, dropped = U.drop_constant_columns(enc)
    extra = {"group_set": "grouped", "n_source_rows": int(len(X)), "dropped_constant_columns": dropped,
             "group_column_in_features": False, "group_column": sp["group"],
             "group_rule": "group column removed from features; ids by descending size over all rows",
             "class_order": "natural (numeric if labels parse as numbers, else str)"}
    return U.write_dataset(out_dir, name, enc, yv, task, classes, info, group=group, group_names=gnames,
                           group_counts=gcounts, group_source=f"feature '{sp['group']}' (removed from features)",
                           extra_meta=extra, outer="full")
