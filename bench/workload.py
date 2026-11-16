import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np

DEFAULT_RANGES = {"eta": [0.02, 0.3], "lam": [0.1, 10.0], "mcw": [0.5, 20.0]}
KINDS = ("hpo_cv", "grouped", "dynamic")


@dataclass
class WorkloadSpec:
    kind: str = "hpo_cv"
    dataset: str = "adult"
    cuts: str = "fold"
    n_configs: int = 64
    config_seed: int = 0
    ranges: Dict[str, List[float]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RANGES))
    depth: Dict[str, Any] = field(default_factory=lambda: {"mode": "fixed", "value": 6})
    subsample: List[float] = field(default_factory=lambda: [1.0, 1.0])
    colsample: List[float] = field(default_factory=lambda: [1.0, 1.0])
    folds: Optional[List[int]] = None
    rounds: int = 100
    config_stride: int = 1
    fold_seed: int = 0
    row_limit: Optional[int] = None
    test_frac: float = 0.0
    outer_seed: int = 0
    seed_base: int = 1000
    group_val_fold: int = 0
    max_groups: Optional[int] = None
    min_group_rows: int = 200
    min_group_size: int = 0
    dynamic: Optional[Dict[str, Any]] = None
    tag: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown workload fields: {sorted(bad)}")
        return cls(**d)

    @classmethod
    def from_json(cls, s_or_path):
        from pathlib import Path
        p = Path(str(s_or_path))
        if not str(s_or_path).lstrip().startswith("{") and p.exists():
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return cls.from_dict(json.loads(s_or_path))

    def validate(self):
        errs = []
        if self.kind not in KINDS:
            errs.append(f"kind must be one of {KINDS}")
        if self.cuts not in ("fold", "global"):
            errs.append("cuts must be 'fold' or 'global'")
        if self.n_configs < 1 or self.rounds < 1 or self.config_stride < 1:
            errs.append("n_configs, rounds, config_stride must be >= 1")
        if self.depth.get("mode") == "fixed":
            if not 1 <= int(self.depth.get("value", 0)) <= 16:
                errs.append("fixed depth must be in 1..16")
        elif self.depth.get("mode") == "mixed":
            if not 1 <= int(self.depth["lo"]) <= int(self.depth["hi"]) <= 16:
                errs.append("mixed depth needs 1 <= lo <= hi <= 16")
        else:
            errs.append("depth.mode must be fixed|mixed")
        for k in ("eta", "lam", "mcw"):
            lo, hi = self.ranges.get(k, DEFAULT_RANGES[k])
            if not 0 < lo <= hi:
                errs.append(f"range {k} must satisfy 0 < lo <= hi")
        for name in ("subsample", "colsample"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi <= 1:
                errs.append(f"{name} must satisfy 0 < lo <= hi <= 1")
        if not 0.0 <= float(self.test_frac) < 0.9:
            errs.append("test_frac must be in [0, 0.9)")
        if self.kind == "dynamic":
            dy = self.dynamic or {}
            if dy.get("algo") not in ("sha", "asha", "hyperband", "hbpar", "grid"):
                errs.append("dynamic.algo must be sha|asha|hyperband|hbpar|grid")
        return errs

    def workload_id(self):
        d = self.to_dict()
        d.pop("config_stride", None)
        d.pop("tag", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def label(self):
        dep = (f"d{self.depth['value']}" if self.depth["mode"] == "fixed"
               else f"d{self.depth['lo']}-{self.depth['hi']}")
        s = f"{self.dataset}_{self.kind}_{self.n_configs}x{self.rounds}_{dep}_{self.cuts}"
        if self.subsample != [1.0, 1.0] or self.colsample != [1.0, 1.0]:
            s += f"_ss{self.subsample[0]:g}-{self.subsample[1]:g}_cs{self.colsample[0]:g}-{self.colsample[1]:g}"
        if self.fold_seed:
            s += f"_fs{self.fold_seed}"
        if self.row_limit:
            s += f"_n{self.row_limit}"
        if self.test_frac:
            s += f"_test{self.test_frac:g}"
        return s


def sample_configs(spec: WorkloadSpec):
    n = spec.n_configs
    rng = np.random.default_rng(spec.config_seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    r = {**DEFAULT_RANGES, **spec.ranges}
    eta = lu(*r["eta"])
    lam = lu(*r["lam"])
    mcw = lu(*r["mcw"])
    if spec.depth["mode"] == "fixed":
        md = np.full(n, int(spec.depth["value"]))
    else:
        md = rng.integers(int(spec.depth["lo"]), int(spec.depth["hi"]) + 1, n)

    def unif(lo_hi):
        lo, hi = lo_hi
        return np.full(n, float(lo)) if lo == hi else rng.uniform(lo, hi, n)
    ss = unif(spec.subsample)
    cs = unif(spec.colsample)
    return [dict(eta=float(eta[i]), lam=float(lam[i]), mcw=float(mcw[i]), max_depth=int(md[i]),
                 subsample=float(ss[i]), colsample=float(cs[i])) for i in range(n)]


@dataclass
class ModelJob:
    mid: int
    cfg: int
    part: int
    cohort: int
    params: Dict[str, Any]
    seed: int
    train_mask: Any = None
    val_mask: Any = None
    test_mask: Any = None

    def key(self):
        return (self.cfg, self.part)

    def public(self):
        return {"mid": self.mid, "cfg": self.cfg, "part": self.part, "cohort": self.cohort,
                "seed": self.seed, "params": self.params}


def fold_list(spec, ds):
    return list(range(ds.n_folds)) if spec.folds is None else [int(k) for k in spec.folds]


def selected_groups(spec, ds):
    if ds.group is None:
        raise ValueError(f"dataset {ds.name} has no `group` array (grouped workload)")
    v = spec.group_val_fold
    tr = ds.folds != v
    gids, cnt = np.unique(ds.group[tr & (ds.group >= 0)], return_counts=True)
    tot = dict(zip(*np.unique(ds.group[ds.group >= 0], return_counts=True)))
    keep = [int(g) for g, c in zip(gids, cnt) if c >= spec.min_group_rows and tot.get(g, 0) >= spec.min_group_size]
    keep.sort()
    if spec.max_groups:
        keep = keep[: spec.max_groups]
    return keep


def build_jobs(spec: WorkloadSpec, ds, all_configs=False):
    cfgs = sample_configs(spec)
    stride = 1 if all_configs else spec.config_stride
    jobs = []
    if spec.kind in ("hpo_cv", "dynamic"):
        folds = fold_list(spec, ds)
        F = ds.n_folds
        for c, cf in enumerate(cfgs):
            for k in folds:
                mid = c * F + k
                if c % stride:
                    continue
                jobs.append(ModelJob(mid, c, k, k if spec.cuts == "fold" else -1, cf, spec.seed_base + mid,
                                     ds.folds != k, ds.folds == k))
    elif spec.kind == "grouped":
        groups = selected_groups(spec, ds)
        v = spec.group_val_fold
        G = len(groups)
        for c, cf in enumerate(cfgs):
            for gi, g in enumerate(groups):
                mid = c * G + gi
                if c % stride:
                    continue
                ing = ds.group == g
                jobs.append(ModelJob(mid, c, g, v if spec.cuts == "fold" else -1, cf, spec.seed_base + mid,
                                     ing & (ds.folds != v), ing & (ds.folds == v)))
    else:
        raise ValueError(spec.kind)
    test = getattr(ds, "test_mask", None)
    if test is not None:
        for j in jobs:
            j.train_mask = j.train_mask & ~test
            j.val_mask = j.val_mask & ~test
            j.test_mask = test & (ds.group == j.part) if spec.kind == "grouped" else test
    return jobs


def config_list_hash(jobs):
    rows = [[j.cfg, j.part, j.cohort, j.seed, [j.params[k] for k in sorted(j.params)]] for j in jobs]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def by_cohort(jobs):
    out = {}
    for j in jobs:
        out.setdefault(j.cohort, []).append(j)
    return out
