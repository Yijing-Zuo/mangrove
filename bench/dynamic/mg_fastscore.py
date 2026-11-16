import contextvars
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .. import metrics as M

_TWO31 = 1 << 31
_EXACT_LIMIT = 1 << 52
DEFAULT_CHUNK = 4_000_000
WARMUP_N = 8192


def rank_sum2(chunk, device):
    sizes = [len(m) for m, _ in chunk]
    N = sum(sizes)
    if len(chunk) == 1:
        v = torch.from_numpy(np.array(chunk[0][0], dtype=np.float32, order="C")).to(device)
        pos = chunk[0][1]
    else:
        v = torch.from_numpy(np.concatenate([m for m, _ in chunk], dtype=np.float32)).to(device)
        pos = torch.cat([p for _, p in chunk])
    sizes_t = torch.tensor(sizes, dtype=torch.int64, device=device)
    off = torch.cumsum(sizes_t, 0) - sizes_t
    seg = torch.repeat_interleave(torch.arange(len(chunk), device=device), sizes_t, output_size=N)
    b = v.view(torch.int32).to(torch.int64)
    key = (seg << 32) | torch.where(b >= 0, b + _TWO31, -1 - b)
    del b, seg
    key, order = torch.sort(key)
    seg_s = key >> 32
    del key
    v_s, pos_s = v[order], pos[order]
    del order
    new = torch.ones(N, dtype=torch.bool, device=device)
    new[1:] = (seg_s[1:] != seg_s[:-1]) | (v_s[1:] != v_s[:-1])
    del v_s
    gid = torch.cumsum(new, 0) - 1
    starts = torch.nonzero(new).squeeze(1)
    counts = torch.diff(starts, append=torch.tensor([N], dtype=torch.int64, device=device))
    rank2 = 2 * (starts[gid] - off[seg_s]) + counts[gid] + 1
    s2 = torch.zeros(len(chunk), dtype=torch.int64, device=device).index_add_(0, seg_s, rank2 * pos_s)
    return s2.cpu().tolist()


def _logloss(y, margin):
    margin = np.asarray(margin, dtype=np.float64)
    y = np.asarray(y)
    p = np.clip(1.0 / (1.0 + np.exp(-margin)), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


_tls = threading.local()


def _scratch(n):
    b = getattr(_tls, "buf", None)
    if b is None or len(b[0]) < n:
        b = _tls.buf = [np.empty(n) for _ in range(6)]
    return [x[:n] for x in b]


def _logloss_buffered(lab, margin):
    if lab.omy is None:
        return _logloss(lab.y, margin)
    m, t, p, q, r, s = _scratch(len(margin))
    np.copyto(m, margin, casting="unsafe")
    np.negative(m, out=t)
    np.exp(t, out=r)
    np.add(1.0, r, out=t)
    np.true_divide(1.0, t, out=p)
    np.clip(p, 1e-15, 1 - 1e-15, out=q)
    np.log(q, out=r)
    np.multiply(lab.y, r, out=s)
    np.subtract(1, q, out=t)
    np.log(t, out=p)
    np.multiply(lab.omy, p, out=t)
    np.add(s, t, out=r)
    return float(-np.mean(r))


def same_metrics(a, b):
    if list(a) != list(b):
        return False
    return all(x == y or (x != x and y != y) for x, y in ((a[k], b[k]) for k in a))


class _ContentCache:
    def __init__(self):
        self._ident = {}
        self._groups = {}

    @staticmethod
    def _eq(a, b):
        if a is None or b is None:
            return a is None and b is None
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)

    def get(self, group, masks, build):
        masks = tuple(masks)
        ik = (group,) + tuple(map(id, masks))
        hit = self._ident.get(ik)
        if hit is not None and all(a is b for a, b in zip(hit[0], masks)):
            return hit[1]
        for cand, val in self._groups.get(group, ()):
            if all(self._eq(a, b) for a, b in zip(cand, masks)):
                break
        else:
            val = build()
            self._groups.setdefault(group, []).append((masks, val))
        self._ident[ik] = (masks, val)
        return val

    def __len__(self):
        return sum(len(v) for v in self._groups.values())


class _Labels:
    __slots__ = ("y", "pos", "npos", "nneg", "omy", "_dev")

    def __init__(self, y, binary):
        self.y = y
        self._dev = None
        self.pos = self.npos = self.nneg = self.omy = None
        if binary:
            self.pos = (y == 1)
            self.npos = float(y.sum())
            self.nneg = len(y) - self.npos
            try:
                self.omy = 1 - y
            except Exception:
                self.omy = None

    def pos_dev(self, device):
        if self._dev is None:
            self._dev = torch.from_numpy(self.pos).to(device)
        return self._dev


class FastScorer:
    def __init__(self, ds, auc_device=None, threads=None, chunk_elems=None, check=None):
        self.ds, self.task = ds, ds.task
        if threads is None:
            threads = os.environ.get("MANGROVE_BENCH_MG_SCORE_THREADS") or min(8, os.cpu_count() or 1)
        if chunk_elems is None:
            chunk_elems = os.environ.get("MANGROVE_BENCH_MG_FASTSCORE_CHUNK") or DEFAULT_CHUNK
        if check is None:
            check = os.environ.get("MANGROVE_BENCH_MG_FASTSCORE_CHECK", "0") == "1"
        self.threads, self.chunk_elems, self.check = max(1, int(threads)), max(1, int(chunk_elems)), bool(check)
        self.auc_device = None
        if auc_device is not None and self.task == "binary":
            dev = torch.device(auc_device)
            if dev.type != "cuda" or torch.cuda.is_available():
                self.auc_device = dev
        self.pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="mgscore")
        self._labels = _ContentCache()
        self._rows = _ContentCache()
        self.counters = dict(batches=0, models=0, gpu_models=0, host_models=0, chunks=0, checked=0, score_s=0.0,
                             check_s=0.0)

    def labels(self, j):
        return self._labels.get(int(j.part), (j.val_mask,),
                                lambda: _Labels(self.ds.y[j.val_mask], self.task == "binary"))

    def spec_rows(self, j):
        def build():
            ev = j.val_mask if j.test_mask is None else (j.val_mask | j.test_mask)
            return np.nonzero(j.train_mask)[0], np.nonzero(ev)[0], self.ds.base_margin(j.train_mask)
        return self._rows.get((int(j.part), int(j.cohort)), (j.train_mask, j.val_mask, j.test_mask), build)

    def metrics_batch(self, items):
        t0 = time.perf_counter()
        entries = [(self.labels(j), np.asarray(m)) for j, m in items]
        out = self._score(entries)
        self.counters["batches"] += 1
        self.counters["models"] += len(items)
        t1 = time.perf_counter()
        self.counters["score_s"] += t1 - t0
        if self.check:
            for (j, m), mt in zip(items, out):
                ref = M.metrics(self.task, self.ds.y[j.val_mask], np.asarray(m))
                if not same_metrics(mt, ref):
                    raise RuntimeError(f"MANGROVE_BENCH_MG_FASTSCORE_CHECK: model cfg={j.cfg} part={j.part}: "
                                       f"fast {mt} != M.metrics {ref}")
            self.counters["checked"] += len(items)
            self.counters["check_s"] += time.perf_counter() - t1
        return out

    def _submit(self, fn, *args):
        return self.pool.submit(contextvars.copy_context().run, fn, *args)

    def _score(self, entries):
        task = self.task
        if task != "binary":
            futs = [self._submit(M.metrics, task, lab.y, m) for lab, m in entries]
            self.counters["host_models"] += len(entries)
            return [f.result() for f in futs]
        auc, todo, futs = {}, [], []
        for i, (lab, m) in enumerate(entries):
            n = len(m)
            fast = (self.auc_device is not None and m.dtype == np.float32 and m.ndim == 1 and n == len(lab.y)
                    and n > 0 and n * (n + 1) // 2 < _EXACT_LIMIT and bool(np.isfinite(m).all()))
            if not fast:
                futs.append(self._submit(M.metrics, task, lab.y, m))
                continue
            futs.append(self._submit(_logloss_buffered, lab, m))
            if lab.npos == 0 or lab.nneg == 0:
                auc[i] = float("nan")
            else:
                todo.append((i, m, lab))
        for (i, _, lab), s2 in zip(todo, self._rank_sums([(m, lab.pos_dev(self.auc_device)) for _, m, lab in todo])):
            auc[i] = float((np.float64(s2) / 2.0 - lab.npos * (lab.npos + 1) / 2) / (lab.npos * lab.nneg))
        out = []
        for i, f in enumerate(futs):
            r = f.result()
            out.append(r if isinstance(r, dict) else {"auc": auc[i], "logloss": r})
        self.counters["gpu_models"] += len(todo)
        self.counters["host_models"] += len(entries) - len(todo)
        return out

    def _rank_sums(self, pairs):
        out, chunk, n = [], [], 0
        for m, p in pairs:
            if chunk and n + len(m) > self.chunk_elems:
                out += rank_sum2(chunk, self.auc_device)
                self.counters["chunks"] += 1
                chunk, n = [], 0
            chunk.append((m, p))
            n += len(m)
        if chunk:
            out += rank_sum2(chunk, self.auc_device)
            self.counters["chunks"] += 1
        return out

    def warmup(self):
        y = (np.arange(WARMUP_N) % 2).astype(self.ds.y.dtype)
        m = np.linspace(-1.0, 1.0, WARMUP_N, dtype=np.float32)
        m[::3] = 0.0
        small = _Labels(y[:40], True)
        barrier = threading.Barrier(self.threads)

        def arrive():
            try:
                barrier.wait(timeout=30.0)
            except threading.BrokenBarrierError:
                pass
            return _logloss_buffered(small, m[:40])
        for f in [self._submit(arrive) for _ in range(self.threads)]:
            f.result()
        if self.auc_device is not None:
            big = _Labels(y, True)
            self._score([(big, m)])
            self._score([(big, m), (small, m[:40])])
        for k in self.counters:
            self.counters[k] = type(self.counters[k])(0)

    def describe(self):
        return (f"threads={self.threads} auc_device={self.auc_device} chunk_elems={self.chunk_elems} "
                f"check={self.check}")

    def stats(self):
        return dict(threads=self.threads, gpu_auc=self.auc_device is not None and self.auc_device.type == "cuda",
                    auc_device=None if self.auc_device is None else str(self.auc_device), check=self.check,
                    chunk_elems=self.chunk_elems, label_sets=len(self._labels), row_sets=len(self._rows),
                    **self.counters)

    def close(self):
        self.pool.shutdown(wait=True)
