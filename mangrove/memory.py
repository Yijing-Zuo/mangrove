import torch


class Accounting:
    def __init__(self):
        self.shared = 0
        self.persistent = 0
        self.transient = 0
        self.peak_transient = 0
        self.peak_total = 0
        self.levels = []

    def set_transient(self, b):
        self.transient = b
        self.peak_transient = max(self.peak_transient, b)
        self.peak_total = max(self.peak_total, self.total)

    @property
    def total(self):
        return self.shared + self.persistent + self.transient

    def snapshot(self):
        return dict(shared=self.shared, persistent=self.persistent, transient=self.transient,
                    peak_transient=self.peak_transient, peak_total=self.peak_total)


class Workspace:
    def __init__(self, device, poison=False, pooled=True):
        self.device = device
        self.poison = poison
        self.pooled = pooled
        self.buf = {}
        self.hwm = {}
        self.grow_events = 0
        self.requests = 0

    def get(self, name, shape, dtype, zero=False):
        n = 1
        for s in shape:
            n *= int(s)
        self.requests += 1
        b = None if not self.pooled else self.buf.get(name)
        if b is None or b.numel() < n or b.dtype != dtype:
            self.buf[name] = b = torch.empty(max(n, 1), dtype=dtype, device=self.device)
            self.grow_events += 1
            self.hwm[name] = b.numel() * b.element_size()
        v = b[:n].view(*shape) if n > 0 else b[:0].view(*shape)
        if self.poison and n > 0:
            v.fill_(0x7F7F7F7F7F7F7F7F if dtype == torch.int64 else 0)
        if zero and n > 0:
            v.zero_()
        return v

    def reserve(self, name, n, dtype):
        b = self.buf.get(name)
        n = max(int(n), 1)
        if self.pooled and (b is None or b.numel() < n or b.dtype != dtype):
            self.buf[name] = torch.empty(n, dtype=dtype, device=self.device)
            self.grow_events += 1
            self.hwm[name] = n * torch.empty(0, dtype=dtype).element_size()

    def swap(self, a, b):
        old_b = self.buf.get(b)
        self.buf[b] = self.buf.pop(a)
        if old_b is not None:
            self.buf[a] = old_b

    def bytes(self):
        return sum(b.numel() * b.element_size() for b in self.buf.values())

    def projected_bytes(self, requests):
        sizes = {k: b.numel() * b.element_size() for k, b in self.buf.items()}
        for name, n in requests.items():
            need = max(1, int(n)) * 8
            sizes[name] = max(sizes.get(name, 0), need) if self.pooled else need
        return sum(sizes.values())

    def release(self, name):
        self.buf.pop(name, None)


ROW_COEF = {
    "gpu": dict(chunk=dict(base=32, sample=76, metric=60, softmax=56), touch_base=12, touch_per_feature=0),
    "ref": dict(chunk=dict(base=88, sample=88, metric=88, softmax=88), touch_base=60, touch_per_feature=52),
}
SPLIT_COEF = {
    "gpu": {False: 28, True: 20},
    "ref": {False: 128, True: 128},
}


class LevelInfeasible(MemoryError):
    def __init__(self, msg, needed, available):
        super().__init__(msg)
        self.needed, self.available = int(needed), int(available)


class Planner:
    def __init__(self, budget_bytes, d, B, row_temp_bytes=None, reserve_frac=0.05, prior_rows_per_parent=32,
                 backend="ref", headroom=1.25, min_chunk_frac=0.25):
        self.budget = int(budget_bytes)
        self.d, self.B = d, B
        self.backend = backend if backend in ROW_COEF else "ref"
        c = ROW_COEF[self.backend]
        self.row_temp_bytes = (c["touch_base"] + c["touch_per_feature"] * d) if row_temp_bytes is None else row_temp_bytes
        self.chunk_coef = dict(c["chunk"])
        self.chunk_row_bytes = max(self.chunk_coef.values())
        self.reserve = int(reserve_frac * budget_bytes)
        self.prior_rows_per_parent = prior_rows_per_parent
        self.headroom = headroom
        self.min_chunk_frac = min_chunk_frac
        self.admit_ceiling = None
        self.log = []

    @property
    def row_chunk_rows(self):
        return int(max(1 << 12, min(1 << 22, self.budget // (32 * max(self.chunk_row_bytes, 1)))))

    @property
    def split_cap(self):
        return int(max(1, min(256 << 20, self.budget // 64)))

    def chunk_bytes(self, sample=True, metric=True, softmax=True):
        c = self.chunk_coef
        return max(c["base"], c["sample"] if sample else 0, c["metric"] if metric else 0, c["softmax"] if softmax else 0)

    def split_cand_bytes(self, wide):
        return SPLIT_COEF[self.backend][bool(wide)] * self.d * self.B

    def split_batch(self, wide):
        return max(1, self.split_cap // self.split_cand_bytes(wide))

    def split_bytes(self, n_cand_p, n_cand_w):
        b = 0
        for n, w in ((n_cand_p, False), (n_cand_w, True)):
            if n:
                b = max(b, min(n, self.split_batch(w)) * self.split_cand_bytes(w))
        return b

    @staticmethod
    def persistent_bytes(n_rt, K=1, n_slots=None, label_bytes=4):
        n_slots = K if n_slots is None else n_slots
        return (17 + label_bytes) * n_rt * K + 103 * n_slots + 8

    def cell_bytes(self, wide):
        return (16 if wide else 8) * self.d * self.B

    def hist_bytes(self, n_packed, n_wide):
        return n_packed * self.cell_bytes(False) + n_wide * self.cell_bytes(True)

    def parent_prior(self, max_depth, n_train):
        return int(min(2 ** max(max_depth - 1, 0), max(1, -(-n_train // self.prior_rows_per_parent))))

    def nph_prior(self, max_depth, n_train):
        if max_depth <= 1:
            return 0
        return int(min(2 ** (max_depth - 2), max(1, -(-n_train // self.prior_rows_per_parent))))

    def model_hist_bytes(self, k_nph, wide, K=1, buf_frac=None):
        cell = self.cell_bytes(wide)
        f = self.min_chunk_frac if buf_frac is None else buf_frac
        return int(K * cell * (2 * k_nph + f * 2 * max(k_nph, 1)))

    def round_state_bytes(self, RT, W):
        return (9 + getattr(self, "nm_bytes_per_rt", 0)) * RT + 4 * W

    def row_phase_bytes(self, RT, W, max_group_rows):
        return self.round_state_bytes(RT, W) + max(self.chunk_row_bytes * max_group_rows, 4 * W)

    def level_row_bytes(self, RT, W, rows_touched):
        return self.round_state_bytes(RT, W) + self.row_temp_bytes * rows_touched

    def round_need(self, RT, W, max_group_rows, hist_bytes, split_bytes, touched_rows=None):
        T = W if touched_rows is None else touched_rows
        return max(self.row_phase_bytes(RT, W, max_group_rows) + hist_bytes,
                   self.level_row_bytes(RT, W, T) + hist_bytes + split_bytes)

    def admission_need(self, acc: Accounting, persistent_total, transient_need):
        return acc.shared + persistent_total + transient_need + self.reserve

    def can_admit(self, acc: Accounting, persistent_total, transient_need):
        need = self.admission_need(acc, persistent_total, transient_need)
        if self.admit_ceiling is not None and need >= self.admit_ceiling:
            return False
        return need <= self.budget

    def chunk_parents(self, acc: Accounting, n_parent, parent_wide, cand_per_parent_wide, cand_per_parent_packed,
                      rows_touched):
        avail = self.budget - acc.shared - acc.persistent - self.reserve - rows_touched * self.row_temp_bytes
        per_parent_buf = 2 * self.cell_bytes(True) if (parent_wide or cand_per_parent_wide) else 2 * self.cell_bytes(False)
        parent_bytes = n_parent * self.cell_bytes(parent_wide)
        avail -= parent_bytes
        if n_parent == 0:
            return 1
        k = max(1, int(avail // max(per_parent_buf, 1)))
        return min(k, n_parent)


class PhaseTimer:
    def __init__(self, enabled, device):
        self.enabled = enabled
        self.cuda = enabled and torch.device(device).type == "cuda"
        self.t = {}
        self.pending = []

    class _Ctx:
        def __init__(self, tm, name):
            self.tm, self.name = tm, name

        def __enter__(self):
            tm = self.tm
            if tm.cuda:
                self.e0 = torch.cuda.Event(enable_timing=True); self.e0.record()
            else:
                import time; self.t0 = time.perf_counter()

        def __exit__(self, *a):
            tm = self.tm
            if tm.cuda:
                e1 = torch.cuda.Event(enable_timing=True); e1.record(); tm.pending.append((self.name, self.e0, e1))
            else:
                import time; tm.t[self.name] = tm.t.get(self.name, 0.0) + time.perf_counter() - self.t0

    class _Null:
        def __enter__(self): pass
        def __exit__(self, *a): pass

    def __call__(self, name):
        return self._Ctx(self, name) if self.enabled else self._Null()

    def flush(self):
        if self.cuda and self.pending:
            torch.cuda.synchronize()
            for name, a, b in self.pending:
                self.t[name] = self.t.get(name, 0.0) + a.elapsed_time(b) / 1000.0
            self.pending = []

    def report(self):
        self.flush()
        r = dict(self.t)
        if "round_total" in r:
            r["host_other"] = r["round_total"] - sum(r.get(k, 0.0) for k in ("hist_build", "subtract", "split_search", "partition", "leaf_route"))
        return r
