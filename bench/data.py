import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import paths


class Dataset:
    def __init__(self, name, task, y, folds, Xb_fold=None, Xb_global=None, group=None, meta=None, path=None,
                 fmt="schema_v1", X_raw=None):
        self.name = name
        self.task = task
        self.y = y
        self.folds = np.asarray(folds).astype(np.int64)
        self.Xb_fold = Xb_fold
        self.Xb_global = Xb_global
        self.group = None if group is None else np.asarray(group).astype(np.int64)
        self.meta = meta or {}
        self.path = path
        self.fmt = fmt
        self.X_raw = X_raw
        self.n_folds = int(self.folds.max()) + 1
        self.K = int(self.meta.get("n_classes") or (int(np.max(y)) + 1)) if task == "multiclass" else 1
        Xany = Xb_global if Xb_global is not None else Xb_fold[0]
        self.n, self.d = Xany.shape
        self.transforms = []
        self.test_mask = None

    def X(self, cohort):
        if cohort is None or cohort < 0:
            if self.Xb_global is None:
                raise ValueError(f"{self.name}: no Xb_global")
            return self.Xb_global
        if self.Xb_fold is None:
            raise ValueError(f"{self.name}: no fold-local cuts (Xb_fold) in this file ({self.fmt}); "
                             "use cuts='global' or the SCHEMA v1 file")
        return self.Xb_fold[cohort]

    def nbins(self, cohort):
        nb = getattr(self, "_nbins", {}).get(-1 if cohort is None or cohort < 0 else int(cohort))
        return nb

    def base_margin(self, train_mask):
        if self.task == "regression":
            return float(np.mean(self.y[train_mask]))
        return 0.0

    def describe(self):
        return {"name": self.name, "task": self.task, "n": int(self.n), "d": int(self.d), "n_classes": self.K,
                "n_folds": self.n_folds, "format": self.fmt, "file": None if self.path is None else paths.rel_to_bench(self.path),
                "sha256": self.meta.get("sha256"), "has_group": self.group is not None,
                "n_test": None if self.test_mask is None else int(self.test_mask.sum()),
                "transforms": self.transforms}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_dataset_file(name, data_dir=None):
    data_dir = Path(data_dir) if data_dir else paths.DATA_DIR
    p = data_dir / f"{name}.npz"
    if p.exists():
        return p, "schema_v1"
    return None, None


def inproc_bins(name, outer=None):
    raw_dir = os.environ.get("MANGROVE_BENCH_INPROC_BIN")
    if not raw_dir:
        return None, None, None
    t0 = time.perf_counter()
    d = str(paths.PREP_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    import fast_binning
    with np.load(Path(raw_dir) / f"{name}.npz", allow_pickle=False) as rz:
        Xnum, folds, F, max_bin = rz["Xnum"], rz["folds"], int(rz["n_folds"]), int(rz["max_bin"])
        cat = {int(j): (rz[f"codes_{j}"], rz[f"labels_{j}"]) for j in rz["cat_columns"]}
    workers = int(os.environ.get("MANGROVE_BENCH_INPROC_BIN_WORKERS", "12"))
    masks = [folds != k for k in range(F)] if outer is None else [(outer[0] != k) & ~outer[1] for k in range(F)]
    Xbf, nbf, _, _ = fast_binning.fold_local_mask_sortonce(Xnum, masks, cat, max_bin, workers=workers)
    return Xbf, nbf, time.perf_counter() - t0


def load_dataset(name, data_dir=None, hash_file=False):
    p, fmt = find_dataset_file(name, data_dir)
    if p is None:
        raise FileNotFoundError(f"dataset {name!r} not found in {data_dir or paths.DATA_DIR}")
    z = np.load(p, allow_pickle=False)
    if fmt == "schema_v1":
        mp = p.with_suffix(".json")
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        task = meta.get("task") or ("multiclass" if z["y"].dtype.kind in "iu" else "binary")
        outer_mode = os.environ.get("MANGROVE_BENCH_INPROC_BIN_MODE") == "outer"
        Xbf_in, nbf_in, bin_s = inproc_bins(name, (z["folds_outer"], z["test_mask"].astype(bool)) if outer_mode else None)
        Xbf_stored = Xbf_in if Xbf_in is not None and not outer_mode else z["Xb_fold"] if "Xb_fold" in z else None
        ds = Dataset(name, task, z["y"], z["folds"], Xb_fold=Xbf_stored,
                     Xb_global=z["Xb_global"] if "Xb_global" in z else None,
                     group=z["group"] if "group" in z else None, meta=meta, path=p, fmt=fmt,
                     X_raw=z["X_raw"] if "X_raw" in z else None)
        ds._nbins = {}
        if "nbins_fold" in z and ds.Xb_fold is not None:
            nbf = nbf_in if nbf_in is not None and not outer_mode else z["nbins_fold"]
            for k in range(nbf.shape[0]):
                ds._nbins[k] = nbf[k].astype(np.int64)
        if bin_s is not None:
            ds.transforms.append(f"inproc_binning: sort-once CPU, {bin_s:.3f} s")
        if "nbins_global" in z and ds.Xb_global is not None:
            ds._nbins[-1] = z["nbins_global"].astype(np.int64)
        if "test_mask" in z:
            ds.stored_test_mask = z["test_mask"].astype(bool)
            if "folds_outer" in z:
                ds.stored_folds_outer = z["folds_outer"].astype(np.int64)
            if "Xb_fold_outer" in z and "folds_outer" in z:
                ds.stored_outer = (z["folds_outer"].astype(np.int64), Xbf_in if Xbf_in is not None and outer_mode else z["Xb_fold_outer"],
                                   z["Xb_outer"] if "Xb_outer" in z else None)
    if hash_file or not ds.meta.get("sha256"):
        ds.meta["sha256"] = _sha256(p)
    return ds


def make_synthetic(n=600, d=5, task="binary", n_folds=5, n_classes=3, seed=0, n_groups=0, max_bin=32):
    rng = np.random.default_rng(seed)
    Xr = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    s = Xr @ w + 0.5 * np.sin(3 * Xr[:, 0]) + rng.normal(0, 0.5, n)
    if task == "binary":
        y = (s > 0).astype(np.float32)
    elif task == "regression":
        y = s.astype(np.float32)
    else:
        y = np.digitize(s, np.quantile(s, np.linspace(0, 1, n_classes + 1)[1:-1])).astype(np.int32)
    folds = (rng.permutation(n) % n_folds).astype(np.int8)
    Xbf = np.zeros((n_folds, n, d), dtype=np.uint8)
    for k in range(n_folds):
        for j in range(d):
            cuts = np.unique(np.quantile(Xr[folds != k, j], np.linspace(0, 1, max_bin)[1:-1]))
            Xbf[k, :, j] = np.searchsorted(cuts, Xr[:, j], side="left")
    Xbg = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        cuts = np.unique(np.quantile(Xr[:, j], np.linspace(0, 1, max_bin)[1:-1]))
        Xbg[:, j] = np.searchsorted(cuts, Xr[:, j], side="left")
    group = None
    if n_groups:
        pz = 1.0 / np.arange(1, n_groups + 1)
        group = rng.choice(n_groups, size=n, p=pz / pz.sum()).astype(np.int32)
    meta = {"name": f"synth-{task}-{n}x{d}", "task": task, "n": n, "d": d,
            "n_classes": n_classes if task == "multiclass" else (2 if task == "binary" else None), "synthetic_seed": seed}
    return Dataset(meta["name"], task, y, folds, Xb_fold=Xbf, Xb_global=Xbg, group=group, meta=meta,
                   fmt="synthetic", X_raw=Xr)


def save_schema_npz(ds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"y": ds.y, "folds": ds.folds.astype(np.int8), "Xb_fold": ds.Xb_fold, "Xb_global": ds.Xb_global,
              "nbins_fold": (ds.Xb_fold.max(axis=1).astype(np.int16) + 1)}
    if ds.group is not None:
        arrays["group"] = ds.group.astype(np.int32)
    if ds.X_raw is not None:
        arrays["X_raw"] = ds.X_raw
    p = out_dir / f"{ds.name}.npz"
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(p)
    meta = dict(ds.meta)
    meta.update(name=ds.name, n_folds=ds.n_folds, sha256=_sha256(p), npz=p.name)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return p


def apply_row_limit(ds, N, seed=0):
    if not N or N >= ds.n:
        return ds
    idx = np.sort(np.random.default_rng(seed).permutation(ds.n)[:N])
    sub = Dataset(ds.name, ds.task, ds.y[idx], ds.folds[idx],
                  Xb_fold=None if ds.Xb_fold is None else ds.Xb_fold[:, idx],
                  Xb_global=None if ds.Xb_global is None else ds.Xb_global[idx],
                  group=None if ds.group is None else ds.group[idx], meta=dict(ds.meta), path=ds.path, fmt=ds.fmt,
                  X_raw=None if ds.X_raw is None else ds.X_raw[idx])
    sub.transforms = ds.transforms + [f"row_limit={N} (sorted(default_rng({seed}).permutation(n)[:N]))"]
    return sub


def _load_prep_binning():
    p = paths.PREP_DIR / "binning.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("bench_prep_binning", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_folds(y, task, n_folds, seed):
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(y)
    folds = np.zeros(n, dtype=np.int64)
    dummy = np.zeros((n, 1), dtype=np.uint8)
    it = (KFold(n_folds, shuffle=True, random_state=seed).split(dummy) if task == "regression"
          else StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(dummy, y))
    for k, (_, te) in enumerate(it):
        folds[te] = k
    return folds


def outer_test_mask(y, task, frac, seed=0):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    _, te = train_test_split(idx, test_size=frac, random_state=seed, stratify=None if task == "regression" else y)
    m = np.zeros(len(y), dtype=bool)
    m[te] = True
    return m


def _cat_cols(ds, Xnum):
    cat_cols = {}
    for j_str, labels in (ds.meta.get("category_labels") or {}).items():
        j = int(j_str)
        codes = np.where(np.isnan(Xnum[:, j]), -1, Xnum[:, j]).astype(np.int64)
        cat_cols[j] = (codes, labels)
    return cat_cols


def rebin(ds, folds, test):
    if ds.X_raw is None:
        raise ValueError(f"{ds.name}: re-binning (fold re-draw / outer split) needs X_raw (not stored for this dataset)")
    Xnum = ds.X_raw.astype(np.float64)
    cat_cols = _cat_cols(ds, Xnum)
    B = _load_prep_binning()
    F = ds.n_folds
    Xbf = np.zeros((F,) + Xnum.shape, dtype=np.uint8)

    def fallback(fit):
        out = np.zeros(Xnum.shape, dtype=np.uint8)
        for j in range(Xnum.shape[1]):
            x = Xnum[fit, j]
            cuts = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0, 1, 256)[1:-1]))
            out[:, j] = np.minimum(np.searchsorted(cuts, Xnum[:, j], side="left"), 255)
        return out
    for k in range(F):
        fit = (folds != k) & ~test
        Xbf[k] = B.bin_matrix(Xnum, fit, cat_cols)[0] if B is not None else fallback(fit)
    Xbg = B.bin_matrix(Xnum, ~test, cat_cols)[0] if B is not None else fallback(~test)
    rule = "prep/binning.py bin_matrix" if B is not None else "fallback per-fold quantiles (prep/binning.py not found)"
    out = Dataset(ds.name, ds.task, ds.y, folds, Xb_fold=Xbf, Xb_global=Xbg, group=ds.group,
                  meta=dict(ds.meta), path=ds.path, fmt=ds.fmt, X_raw=ds.X_raw)
    out.transforms = list(ds.transforms) + [f"rebin ({rule}; fit rows exclude {int(test.sum())} test rows)"]
    return out


def refold(ds, seed):
    if seed == 0:
        return ds
    folds = make_folds(ds.y, ds.task, ds.n_folds, seed)
    out = rebin(ds, folds, np.zeros(ds.n, dtype=bool) if ds.test_mask is None else ds.test_mask)
    out.test_mask = ds.test_mask
    out.transforms.append(f"refold seed={seed}")
    return out


def prepare_dataset(spec, data_dir=None, ds=None):
    ds = ds if ds is not None else load_dataset(spec.dataset, data_dir)
    ds = apply_row_limit(ds, spec.row_limit)
    frac = float(getattr(spec, "test_frac", 0.0) or 0.0)
    test = None
    usable = False
    stored = getattr(ds, "stored_test_mask", None)
    if frac:
        usable = stored is not None and abs(stored.mean() - frac) < 0.01 and not spec.row_limit \
            and getattr(spec, "outer_seed", 0) == int(ds.meta.get("outer_seed", 0))
        so = getattr(ds, "stored_outer", None)
        if usable and so is not None and not spec.fold_seed:
            folds_o, Xbf_o, Xbg_o = so
            out = Dataset(ds.name, ds.task, ds.y, np.where(stored, 0, folds_o), Xb_fold=Xbf_o,
                          Xb_global=Xbg_o if Xbg_o is not None else ds.Xb_global, group=ds.group, meta=dict(ds.meta),
                          path=ds.path, fmt=ds.fmt, X_raw=ds.X_raw)
            out.test_mask = stored
            out.transforms = ds.transforms + ["outer split: stored test_mask / folds_outer / Xb_fold_outer"]
            return out
        test = stored if usable else outer_test_mask(ds.y, ds.task, frac, getattr(spec, "outer_seed", 0))
    folds = ds.folds if not spec.fold_seed else make_folds(ds.y, ds.task, ds.n_folds, spec.fold_seed)
    if test is None and not spec.fold_seed:
        return ds
    if test is not None and not spec.fold_seed:
        sfo = getattr(ds, "stored_folds_outer", None)
        if usable and sfo is not None:
            fo = sfo
        else:
            fo = np.full(ds.n, -1, dtype=np.int64)
            fo[~test] = make_folds(ds.y[~test], ds.task, ds.n_folds, 0)
        folds = np.where(test, 0, fo)
    out = rebin(ds, folds, np.zeros(ds.n, dtype=bool) if test is None else test)
    out.test_mask = test
    if spec.fold_seed:
        out.transforms.append(f"refold seed={spec.fold_seed}")
    if test is not None:
        out.transforms.append(f"outer split derived: test_frac={frac} outer_seed={getattr(spec, 'outer_seed', 0)} "
                              "(stratified train_test_split)")
    return out
