import hashlib, io, json, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

import binning

PREP_DIR = Path(__file__).resolve().parent
WORK_DIR = PREP_DIR.parent
N_FOLDS = 5
FOLD_SEED = 0
RAW_MAX_CELLS = 2_500_000
DEFLATE_MAX_BYTES = 23 * 1024 * 1024
STORED_DIRECT_BYTES = 400 * 1024 * 1024
VARIANT = {"n_folds": None, "suffix": ""}


def is_categorical_series(s):
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s)
            or not pd.api.types.is_numeric_dtype(s))


def encode_frame(X, cat_indicator=None):
    n, d = X.shape
    Xnum = np.empty((n, d), dtype=np.float64)
    cat_cols = {}
    for j, c in enumerate(X.columns):
        s = X[c]
        is_cat = is_categorical_series(s) or (cat_indicator is not None and bool(cat_indicator[j]))
        if is_cat:
            cat = pd.Categorical(s)
            codes = np.asarray(cat.codes, dtype=np.int64)
            labels = [x.item() if hasattr(x, "item") else x for x in cat.categories]
            cat_cols[j] = (codes, labels)
            v = codes.astype(np.float64)
            v[codes < 0] = np.nan
            Xnum[:, j] = v
        else:
            Xnum[:, j] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return {"Xnum": Xnum, "cat_cols": cat_cols, "feature_names": [str(c) for c in X.columns],
            "categorical_features": sorted(cat_cols)}


def subset_rows(enc, idx):
    return {"Xnum": enc["Xnum"][idx], "cat_cols": {j: (c[idx], l) for j, (c, l) in enc["cat_cols"].items()},
            "feature_names": enc["feature_names"], "categorical_features": enc["categorical_features"]}


def drop_constant_columns(enc):
    X = enc["Xnum"]
    keep = []
    for j in range(X.shape[1]):
        x = X[:, j]
        m = np.isnan(x)
        v = x[~m]
        n_distinct = 0 if len(v) == 0 else (1 if v.min() == v.max() else 2)
        keep.append(n_distinct + int(m.any()) > 1)
    keep = np.array(keep, dtype=bool)
    if keep.all():
        return dict(enc), []
    old = np.nonzero(keep)[0]
    remap = {int(o): i for i, o in enumerate(old)}
    out = {"Xnum": X[:, keep],
           "cat_cols": {remap[j]: v for j, v in enc["cat_cols"].items() if keep[j]},
           "feature_names": [f for f, k in zip(enc["feature_names"], keep) if k]}
    out["categorical_features"] = sorted(out["cat_cols"])
    dropped = [f for f, k in zip(enc["feature_names"], keep) if not k]
    return out, dropped


def _natural_sorted(labels):
    labs = list(labels)
    try:
        vals = [float(str(x)) for x in labs]
        return [l for _, l in sorted(zip(vals, labs), key=lambda t: t[0])]
    except ValueError:
        return sorted(labs, key=lambda x: str(x))


def encode_target(y, task, class_order="natural"):
    if task == "regression":
        v = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        return v, ~np.isnan(v), None
    cat = pd.Categorical(y)
    if class_order == "natural":
        c0 = np.asarray(cat.codes)
        cnt = np.bincount(c0[c0 >= 0], minlength=len(cat.categories))
        present = [c for c, k in zip(cat.categories, cnt) if k > 0]
        cat = pd.Categorical(y, categories=_natural_sorted(present))
    codes = np.asarray(cat.codes, dtype=np.int64)
    return codes, codes >= 0, [str(c) for c in cat.categories]


def make_folds(y, task, n_folds=N_FOLDS, seed=FOLD_SEED):
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(y)
    folds = np.zeros(n, dtype=np.int8)
    dummy = np.zeros((n, 1), dtype=np.uint8)
    if task == "regression":
        it = KFold(n_folds, shuffle=True, random_state=seed).split(dummy)
    else:
        it = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(dummy, y)
    for k, (_, te) in enumerate(it):
        folds[te] = k
    return folds


def group_ids(codes, labels):
    codes = np.asarray(codes, dtype=np.int64)
    K = len(labels)
    cnt = np.bincount(np.where(codes < 0, K, codes), minlength=K + 1)
    names = [str(l) for l in labels] + ["<missing>"]
    present = np.nonzero(cnt > 0)[0]
    order = sorted(present, key=lambda i: (-cnt[i], names[i]))
    rank = np.full(K + 1, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    g = rank[np.where(codes < 0, K, codes)].astype(np.int32)
    return g, [names[i] for i in order], [int(cnt[i]) for i in order]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def content_sha256(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        h.update(k.encode()); h.update(a.dtype.str.encode()); h.update(repr(a.shape).encode())
        h.update(memoryview(a).cast("B") if a.size else b"")
    return h.hexdigest()


def _npy_header(a):
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, np.lib.format.header_data_from_array_1_0(a))
    return buf.getvalue()


def _write_stored_zip(path, arrays):
    import struct, zlib
    DOS_DATE, DOS_TIME, V = 0x21, 0, 45
    cent = []
    with open(path, "wb") as f:
        for k in sorted(arrays):
            a = np.ascontiguousarray(arrays[k])
            if a.dtype.byteorder == ">":
                a = a.astype(a.dtype.newbyteorder("<"))
            hdr = _npy_header(a)
            data = memoryview(a).cast("B") if a.size else memoryview(b"")
            crc = zlib.crc32(data, zlib.crc32(hdr)) & 0xFFFFFFFF
            size = len(hdr) + data.nbytes
            name = (k + ".npy").encode("ascii")
            off = f.tell()
            f.write(struct.pack("<IHHHHHIIIHH", 0x04034B50, V, 0, 0, DOS_TIME, DOS_DATE, crc,
                                0xFFFFFFFF, 0xFFFFFFFF, len(name), 20))
            f.write(name)
            f.write(struct.pack("<HHQQ", 1, 16, size, size))
            f.write(hdr)
            f.write(data)
            cent.append((name, crc, size, off))
        cd_off = f.tell()
        for name, crc, size, off in cent:
            f.write(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, (3 << 8) | V, V, 0, 0, DOS_TIME, DOS_DATE, crc,
                                0xFFFFFFFF, 0xFFFFFFFF, len(name), 28, 0, 0, 0, 0o644 << 16, 0xFFFFFFFF))
            f.write(name)
            f.write(struct.pack("<HHQQQ", 1, 24, size, size, off))
        cd_end = f.tell()
        n = len(cent)
        f.write(struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, (3 << 8) | V, V, 0, 0, n, n, cd_end - cd_off, cd_off))
        f.write(struct.pack("<IIQI", 0x07064B50, 0, cd_end, 1))
        f.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, min(n, 0xFFFF), min(n, 0xFFFF), 0xFFFFFFFF,
                            0xFFFFFFFF, 0))


def save_npz_deterministic(path, arrays, compression="deflate"):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if compression == "stored":
        _write_stored_zip(tmp, arrays)
        tmp.replace(path)
        return
    ct = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(tmp, "w", compression=ct, allowZip64=True) as zf:
        for k in sorted(arrays):
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = ct
            zi.create_system = 3
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arrays[k]), allow_pickle=False)
    tmp.replace(path)


def code_hash():
    h = hashlib.sha256()
    for p in sorted(list(PREP_DIR.glob("*.py")) + list(PREP_DIR.glob("*.json"))):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def versions():
    import sklearn
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
         "sklearn": sklearn.__version__, "platform": platform.platform()}
    try:
        import openml
        v["openml"] = openml.__version__
    except Exception:
        pass
    return v


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")


TEST_FRAC = 0.2
OUTER_SEED = 0


def outer_test_mask(y, task, frac=TEST_FRAC, seed=OUTER_SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    _, te = train_test_split(idx, test_size=frac, random_state=seed, stratify=None if task == "regression" else y)
    m = np.zeros(len(y), dtype=bool)
    m[te] = True
    return m


def outer_split(Xnum, y, task, cat_cols, n_folds, full=True):
    test = outer_test_mask(y, task)
    fo = np.full(len(y), -1, dtype=np.int8)
    fo[~test] = make_folds(np.asarray(y)[~test], task, n_folds, FOLD_SEED)
    arrays = {"test_mask": test, "folds_outer": fo}
    if full:
        Xbf, nbf, ptr, val = binning.fold_local_mask(Xnum, [(fo != k) & ~test for k in range(n_folds)], cat_cols)
        Xbo, nbo, _ = binning.bin_matrix(Xnum, ~test, cat_cols)
        arrays.update(Xb_fold_outer=Xbf, nbins_fold_outer=nbf, cut_ptr_fold_outer=ptr, cut_val_fold_outer=val,
                      Xb_outer=Xbo, nbins_outer=nbo)
    meta = {"test_frac": TEST_FRAC, "outer_seed": OUTER_SEED, "n_test": int(test.sum()),
            "outer_arrays": sorted(arrays), "outer_level": "full" if full else "mask_only",
            "outer_rule": "test_mask = train_test_split(arange(n), test_size=0.2, random_state=0, stratify=y "
                          "(classification) / None (regression)); folds_outer = StratifiedKFold/KFold(5, shuffle, 0) "
                          "on non-test rows, -1 on test rows"}
    return arrays, meta


def _raw_nbytes(arrays):
    return int(sum(np.asarray(a).nbytes for a in arrays.values()))


def check_fold_structure(enc, y, task, folds, include_outer=True):
    Xnum, cat_cols = enc['Xnum'], enc['cat_cols']
    folds = np.asarray(folds)
    F = int(folds.max()) + 1
    fit_sets = [(f'cv_{k}', folds != k) for k in range(F)]
    if include_outer:
        test = outer_test_mask(y, task)
        outer_folds = np.full(len(y), -1, dtype=np.int8)
        outer_folds[~test] = make_folds(np.asarray(y)[~test], task, F, FOLD_SEED)
        fit_sets += [('outer_train', ~test)]
        fit_sets += [(f'outer_cv_{k}', (outer_folds != k) & ~test) for k in range(F)]
    examples, violations = [], 0

    def fail(mode, j, reason):
        nonlocal violations
        violations += 1
        if len(examples) < 20:
            examples.append({'fitting_set': mode, 'feature': enc['feature_names'][j], 'reason': reason})

    def n_states(x):
        missing = np.isnan(x)
        valid = x[~missing]
        n = 0 if not len(valid) else (1 if valid.min() == valid.max() else 2)
        return n + int(missing.any())

    for j in range(Xnum.shape[1]):
        x = Xnum[:, j]
        global_states = n_states(x)
        global_missing = bool(np.isnan(x).any())
        if j in cat_cols:
            codes, labels = cat_cols[j]
            codes = np.asarray(codes, dtype=np.int64)
            present_global = np.bincount(codes[codes >= 0], minlength=len(labels)) > 0
        for mode, fit in fit_sets:
            train = x[fit]
            if not len(train):
                fail(mode, j, 'empty_fitting_set')
                continue
            if n_states(train) <= 1 < global_states:
                fail(mode, j, 'training_constant_but_global_nonconstant')
            train_missing = bool(np.isnan(train).any())
            unseen = False
            if j in cat_cols:
                present_train = np.bincount(codes[fit & (codes >= 0)], minlength=len(labels)) > 0
                unseen = bool((present_global & ~present_train).any())
            if not train_missing and (global_missing or unseen):
                fail(mode, j, 'missing_bin_exists_only_due_to_nonfitting_rows')
    return {'status': 'PASS' if violations == 0 else 'FAIL',
            'fitting_sets': [name for name, _ in fit_sets], 'rows': int(Xnum.shape[0]),
            'features': int(Xnum.shape[1]), 'violations': violations, 'examples': examples,
            'scope': 'Actual CV/outer fitting sets; checks structure before binning, does not change stored arrays.'}


def write_dataset(out_dir, name, enc, y, task, classes, source, *, folds=None, group=None, group_names=None,
                  group_counts=None, group_source=None, row_id=None, xb_global=None, xb_global_rule="schema",
                  nbins_global=None, extra_arrays=None, extra_meta=None, raw_max_cells=RAW_MAX_CELLS,
                  outer="full", compression=None):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Xnum, cat_cols = enc["Xnum"], enc["cat_cols"]
    n, d = Xnum.shape
    variant_meta = {}
    if VARIANT["n_folds"]:
        if folds is not None:
            raise ValueError(f"{name}: --n-folds variants need derived folds (not available for the main set)")
        base = name
        name = name + VARIANT["suffix"]
        folds = make_folds(y, task, VARIANT["n_folds"])
        variant_meta = {"variant_of": base, "group_set": "campaign", "variant": f"{VARIANT['n_folds']}-fold"}
    if folds is None:
        folds = make_folds(y, task)
    folds = np.asarray(folds, dtype=np.int8)
    F = int(folds.max()) + 1
    if task == "regression":
        yout = np.asarray(y, dtype=np.float32)
    elif task == "binary":
        yout = np.asarray(y, dtype=np.float32)
    else:
        yout = np.asarray(y, dtype=np.int32)
    if os.environ.get('MG_DATA_STRICT_FOLD_STRUCTURE') == '1':
        guard = check_fold_structure(enc, yout, task, folds, include_outer=bool(outer))
        guard_path = out_dir / 'fold_structure_guards' / f'{name}.json'
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(guard_path, guard)
        print('FOLD_STRUCTURE_GUARD ' + json.dumps({'dataset': name, **guard}), flush=True)
        if guard['status'] != 'PASS':
            raise ValueError(f'{name}: fold structure depends on non-fitting rows; see {guard_path}')
    Xbf, nbf, ptr, val = binning.fold_local(Xnum, folds, F, cat_cols)
    t_fold = time.time() - t0
    Xbg_s, nbg_s, _ = binning.bin_matrix(Xnum, np.ones(n, dtype=bool), cat_cols)
    arrays = {"y": yout, "folds": folds, "Xb_fold": Xbf, "nbins_fold": nbf, "cut_ptr_fold": ptr, "cut_val_fold": val}
    v1_keys = None
    if xb_global is None:
        arrays["Xb_global"] = Xbg_s
        arrays["nbins_global"] = nbg_s
        xb_global_rule = "schema"
    else:
        arrays["Xb_global"] = np.asarray(xb_global, dtype=np.uint8)
        arrays["nbins_global"] = np.asarray(nbins_global, dtype=np.int16)
        arrays["Xb_global_schema"] = Xbg_s
        arrays["nbins_global_schema"] = nbg_s
    has_raw = n * d <= raw_max_cells
    if has_raw:
        arrays["X_raw"] = Xnum.astype(np.float32)
    if group is not None:
        arrays["group"] = np.asarray(group, dtype=np.int32)
    if row_id is not None:
        arrays["row_id"] = np.asarray(row_id, dtype=np.int64)
    if extra_arrays:
        arrays.update(extra_arrays)
    v1_keys = sorted(arrays)
    outer_meta = {}
    if outer:
        outer_arrays, outer_meta = outer_split(Xnum, yout, task, cat_cols, F, full=(outer == "full"))
        arrays.update(outer_arrays)
    path = out_dir / f"{name}.npz"
    comp = compression
    old_json = out_dir / f"{name}.json"
    if comp is None and old_json.exists():
        try:
            comp = json.loads(old_json.read_text(encoding="utf-8")).get("npz_compression")
        except Exception:
            comp = None
    if comp is None and _raw_nbytes(arrays) > STORED_DIRECT_BYTES:
        comp = "stored"
    if comp is None:
        save_npz_deterministic(path, arrays, "deflate")
        comp = "deflate" if path.stat().st_size <= DEFLATE_MAX_BYTES else "stored"
        if comp == "stored":
            save_npz_deterministic(path, arrays, comp)
    else:
        save_npz_deterministic(path, arrays, comp)
    miss_cells = int(np.isnan(Xnum).sum())
    meta = {
        "name": name, "source": source, "n": int(n), "d": int(d), "task": task,
        "n_classes": (len(classes) if classes is not None else None), "classes": classes,
        "n_folds": F, "fold_rule": ("KFold(shuffle=True, random_state=0)" if task == "regression"
                                    else "StratifiedKFold(shuffle=True, random_state=0)"),
        "feature_names": enc["feature_names"], "categorical_features": enc["categorical_features"],
        "n_categorical": len(enc["categorical_features"]),
        "category_labels": {str(j): [str(x) for x in l] for j, (c, l) in cat_cols.items()},
        "missing_rate": miss_cells / float(max(n * d, 1)), "missing_cells": miss_cells,
        "max_bin": binning.MAX_BIN, "xb_global_rule": xb_global_rule,
        "max_nbins_fold": int(nbf.max()), "mean_nbins_fold": float(nbf.mean()),
        "max_nbins_global": int(np.max(arrays["nbins_global"])),
        "pos_rate": (float(np.mean(yout)) if task == "binary" else None),
        "y_mean": float(np.mean(yout)) if task == "regression" else None,
        "y_std": float(np.std(yout)) if task == "regression" else None,
        "class_counts": (np.bincount(np.asarray(y, dtype=np.int64)).tolist() if task != "regression" else None),
        "fold_sizes": np.bincount(folds.astype(np.int64)).tolist(),
        "has_X_raw": bool(has_raw), "X_raw_note": ("categorical columns hold global category ids "
                                                   "(index into category_labels); NaN = missing") if has_raw else None,
        "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()},
        "npz": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "content_sha256": content_sha256(arrays),
        "content_sha256_v1": content_sha256({k: arrays[k] for k in v1_keys}),
        "v1_arrays": v1_keys, "npz_compression": comp,
        "prep_code_sha256": code_hash(), "prep_versions": versions(),
        "prep_seconds": round(time.time() - t0, 2), "fold_binning_seconds": round(t_fold, 2),
    }
    meta.update(outer_meta)
    if group is not None:
        meta.update(group_source=group_source, group_names=group_names, group_counts=group_counts,
                    n_groups=len(group_names))
    if extra_meta:
        meta.update(extra_meta)
    meta.update(variant_meta)
    write_json(out_dir / f"{name}.json", meta)
    return meta
