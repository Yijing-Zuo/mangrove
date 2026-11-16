from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class ModelSpec:
    data_key: str
    train_rows: np.ndarray
    eval_rows: np.ndarray = None
    eta: float = 0.1
    lam: float = 1.0
    mcw: float = 1.0
    max_depth: int = 6
    n_rounds: int = 100
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    seed: int = 0
    base_margin: float = 0.0
    quant_bits: int = 16
    acc_mode: str = "auto"
    early_stopping_rounds: int = 0
    tag: Any = None

    def __post_init__(self):
        self.train_rows = np.asarray(self.train_rows, dtype=np.int64)
        self.eval_rows = np.asarray([] if self.eval_rows is None else self.eval_rows, dtype=np.int64)
        assert 1 <= self.max_depth <= 12, "max_depth in 1..12"
        assert self.quant_bits in (16, 28)
        assert self.acc_mode in ("auto", "packed", "wide")
        assert len(self.train_rows) >= 1

    def packed_ok(self):
        n = len(self.train_rows)
        hmax = 2 ** 16 - 1 if self.quant_bits == 16 else 2 ** 28 - 1
        gmax = 2 ** 15 - 1 if self.quant_bits == 16 else 2 ** 27 - 1
        return n * hmax < 2 ** 32 and n * gmax < 2 ** 31

    def resolve_mode(self):
        if self.acc_mode == "wide":
            return "wide"
        if self.acc_mode == "packed":
            assert self.packed_ok(), "packed accumulator requested but the model's integer bound does not fit"
            return "packed"
        return "packed" if self.packed_ok() else "wide"


@dataclass
class Tree:
    split_node: np.ndarray
    split_feat: np.ndarray
    split_thr: np.ndarray
    leaf_node: np.ndarray
    leaf_val: np.ndarray

    def route(self, Xb, max_depth):
        n = Xb.shape[0]
        nid = np.zeros(n, dtype=np.int64)
        order = np.argsort(self.split_node, kind="stable")
        sk, sf, st = self.split_node[order].astype(np.int64), self.split_feat[order], self.split_thr[order]
        for _ in range(max_depth):
            if len(sk) == 0:
                break
            j = np.minimum(np.searchsorted(sk, nid), len(sk) - 1)
            hit = sk[j] == nid
            if not hit.any():
                break
            f = np.where(hit, sf[j], 0).astype(np.int64)
            x = Xb[np.arange(n), f].astype(np.int32)
            nid = np.where(hit, 2 * nid + 1 + (x > st[j]), nid)
        lo = np.argsort(self.leaf_node, kind="stable")
        lk, lv = self.leaf_node[lo].astype(np.int64), self.leaf_val[lo]
        j = np.minimum(np.searchsorted(lk, nid), len(lk) - 1)
        assert (lk[j] == nid).all()
        return lv[j]


class RoundArchive:
    __slots__ = ("n32", "f16", "t32", "ln32", "lvv", "so", "lo")

    def __init__(self, n32, f16, t32, ln32, lvv, so, lo):
        self.n32, self.f16, self.t32, self.ln32, self.lvv, self.so, self.lo = n32, f16, t32, ln32, lvv, so, lo

    def tree(self, s):
        a, b, c, e = self.so[s], self.so[s + 1], self.lo[s], self.lo[s + 1]
        return Tree(self.n32[a:b].copy(), self.f16[a:b].copy(), self.t32[a:b].copy(), self.ln32[c:e].copy(),
                    self.lvv[c:e])


class LazyTrees:
    __slots__ = ("_items",)

    def __init__(self, items=()):
        self._items = list(items)

    def _mat(self, i):
        it = self._items[i]
        if type(it) is tuple:
            arc, s0, K = it
            it = [arc.tree(s0 + k) for k in range(K)]
            self._items[i] = it
        return it

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._mat(j) for j in range(*i.indices(len(self._items)))]
        n = len(self._items)
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("tree round index out of range")
        return self._mat(i)

    def __iter__(self):
        for i in range(len(self._items)):
            yield self._mat(i)

    def append(self, trees_of_round):
        self._items.append(trees_of_round)

    def append_lazy(self, archive, s0, K):
        self._items.append((archive, s0, K))

    def __eq__(self, other):
        if isinstance(other, LazyTrees):
            other = list(other)
        return list(self) == other

    def __repr__(self):
        return repr(list(self))


@dataclass
class ModelState:
    model_id: int
    spec: ModelSpec
    K: int
    status: str = "pending"
    local_round: int = 0
    slot: Optional[int] = None
    trees: List[List[Tree]] = field(default_factory=LazyTrees)
    metric_history: List[float] = field(default_factory=list)
    metric_q_history: List[int] = field(default_factory=list)
    best_round: int = -1
    best_metric_q: Optional[int] = None
    best_metric: Optional[float] = None
    stop_reason: Optional[str] = None
    eval_margin: Optional[np.ndarray] = None
    admitted_at: Optional[int] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def predict_margin(self, Xb, n_rounds=None):
        T = len(self.trees) if n_rounds is None else min(n_rounds, len(self.trees))
        out = np.full((Xb.shape[0], self.K), np.float32(self.spec.base_margin), dtype=np.float32)
        for t in range(T):
            for k in range(self.K):
                out[:, k] += self.trees[t][k].route(Xb, self.spec.max_depth)
        return out[:, 0] if self.K == 1 else out


class Registry:
    def __init__(self):
        self.models: Dict[int, ModelState] = {}
        self._next = 0

    def add(self, spec: ModelSpec, K: int) -> ModelState:
        st = ModelState(model_id=self._next, spec=spec, K=K)
        self._next += 1
        self.models[st.model_id] = st
        return st

    def __getitem__(self, mid) -> ModelState:
        return self.models[mid]

    def ids(self, status=None):
        return [m for m, s in self.models.items() if status is None or s.status == status]
