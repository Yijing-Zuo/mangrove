import math
import numpy as np
import torch

from . import kernels_ref as KR
from .registry import Registry, ModelSpec, ModelState, Tree, RoundArchive, LazyTrees
from .memory import Accounting, Workspace, Planner, PhaseTimer, LevelInfeasible

NODE_SHIFT = 13
import os as _os
HOST_LOOP = _os.environ.get("MANGROVE_HOST_LOOP", "1") != "0"
NODEMAP_DEFAULT = _os.environ.get("MANGROVE_NODEMAP", "1") != "0"
NM_DEEP_MN = int(_os.environ.get("MGNM_DEEP_MN", "16"))
NM_MAX_DEPTH = 7
NM_SYNC_CHECK = _os.environ.get("MGNM_SYNC_CHECK", "0") == "1"
NM_DEEP_FG1 = _os.environ.get("MGNM_DEEP_FG1", "1") != "0"
NM_RING_POISON = _os.environ.get("MGNM_RING_POISON", "0") == "1"
NM_GC = _os.environ.get("MANGROVE_GC", "")
if NM_GC not in ("", "disable", "freeze"):
    raise ValueError(f"MANGROVE_GC={NM_GC!r}: expected disable | freeze (or unset)")
import gc as _gc
CHECK_MIRRORS = _os.environ.get("MANGROVE_CHECK_MIRRORS", "0") == "1"
PINNED_LIFECYCLE = _os.environ.get("MANGROVE_PINNED_LIFECYCLE", "1") != "0"
LAY_RECORDS = 64
FUSED_GRADIENTS = _os.environ.get("MANGROVE_FUSED_GRADIENTS", "1") != "0"
FUSED_SAMPLING = _os.environ.get("MANGROVE_FUSED_SAMPLING", "1") != "0"
import weakref as _weakref


def _phys_bytes():
    try:
        phys = int(_os.sysconf("SC_PAGE_SIZE")) * int(_os.sysconf("SC_PHYS_PAGES"))
        for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                with open(f) as fh:
                    v = fh.read().strip()
                if v.isdigit() and 0 < int(v) < phys:
                    phys = int(v)
            except OSError:
                pass
        return phys
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullTotalPhys)
    except Exception:
        pass
    return 16 << 30


def _pin_cap_bytes():
    v = _os.environ.get("MANGROVE_PIN_PAUSE_GB", "")
    if v:
        return int(float(v) * (1 << 30))
    return max(1 << 30, _phys_bytes() // 4)


def _pin_release(live, cls, nbytes):
    live[0] -= nbytes
    cls[0] -= nbytes
PHASE_TIMING_ENV = _os.environ.get("MANGROVE_PHASE_TIMING", "")
DEBUG_CHECKS = _os.environ.get("MANGROVE_DEBUG_CHECKS", "0") == "1"
HAS_NONZERO_STATIC = hasattr(torch, "nonzero_static")


class Cohort:
    def __init__(self, key, Xb, y, nbins=None):
        self.key = key
        self.Xb = np.ascontiguousarray(Xb, dtype=np.uint8)
        self.y = np.asarray(y)
        self.n, self.d = self.Xb.shape
        self.nbins = np.asarray(nbins if nbins is not None else self.Xb.max(0).astype(np.int64) + 1)


class Engine:
    def __init__(self, cohorts, loss, n_classes=1, budget_bytes=2 ** 40, device="cpu", backend="ref",
                 worklist=True, frontier=True, chunk_parents=None, max_model_slots=None, log=None,
                 debug_shuffle_wl=None, debug_poison=False, workspace_pool=True, phase_timing=False,
                 collect_metrics=True, nodemap=None):
        assert loss in ("logistic", "squared", "softmax")
        self.loss = loss
        self.K = n_classes if loss == "softmax" else 1
        self.dev = torch.device(device)
        self.backend = backend
        self.collect_metrics = bool(collect_metrics)
        if backend == "gpu":
            try:
                from . import kernels_gpu as KG
                if self.dev.type != "cuda" and not getattr(KG, "EMU", False):
                    raise RuntimeError("device is CPU and MANGROVE_EMU=1 is not set (kernels_gpu needs CUDA or the emulation build)")
                KG._lib(self.dev)
            except Exception as e:
                raise RuntimeError(f"backend='gpu' requested but mangrove.kernels_gpu is not usable: {e!r}") from e
            self.KR = KG
        elif backend == "ref":
            self.KR = KR
        else:
            raise ValueError(f"unknown backend {backend!r} (ref | gpu)")
        self.use_nodemap = NODEMAP_DEFAULT if nodemap is None else bool(nodemap)
        self._nm_mod = None
        if backend == "gpu" and self.dev.type == "cuda":
            from . import kernels_nm as _KNM
            if _KNM.available(self.dev):
                self._nm_mod = _KNM
        self.nm_require = _os.environ.get("MANGROVE_NODEMAP_REQUIRE", "0") == "1"
        if self.nm_require and not (backend == "gpu" and self.dev.type == "cuda"):
            raise RuntimeError("MANGROVE_NODEMAP_REQUIRE=1 needs backend='gpu' on a CUDA device")
        if (nodemap or self.nm_require) and self._nm_mod is None and backend == "gpu" and self.dev.type == "cuda":
            from . import kernels_nm as _KNM
            _KNM.lib(self.dev)
        self.nm_stats = dict(rounds=0, worklist_rounds=0, hist_calls=0, deep_calls=0, update_calls=0, rows_scanned=0,
                             sp4_device=0, ring_overflow=0, ring_bytes=0, multi_group_rounds=0,
                             fused_rowmask=0, fused_colmask=0)
        self.cohorts = {c.key: c for c in cohorts}
        d = {c.d for c in cohorts}
        assert len(d) == 1, "all cohorts of an engine share d"
        self.d = d.pop()
        self.cohort_off = {}
        off = 0
        for c in cohorts:
            self.cohort_off[c.key] = off
            off += c.n
        self.N = off
        self.X_all = torch.from_numpy(np.concatenate([c.Xb for c in cohorts], 0)).to(self.dev)
        nb = max(int(c.nbins.max()) for c in cohorts)
        self.B = max(2, 1 << (nb - 1).bit_length())
        self.use_worklist = worklist
        self.use_frontier = frontier
        self.chunk_parents = chunk_parents
        self.max_model_slots = max_model_slots
        self.reg = Registry()
        self.mslots = []
        self.pending = []
        self._paused = {}
        self._paused_dev_bytes = 0
        self._pause_dev_ok = False
        self._ev_cache = {}
        self._ev_cache_bytes = 0
        self._pin_live = [0]
        self._pin_cls = {}
        self._pin_cap = _pin_cap_bytes()
        self._pin_failed = False
        self._lay_host = {}
        self.pin_stats = dict(pin_blocks=0, pin_models=0, pin_bytes=0, pin_cap_fallbacks=0, pin_fail=0,
                             resume_pinned=0, layout_records=0)
        self.step_index = 0
        self.events = []
        self.acc = Accounting()
        self.acc.shared = self.X_all.numel() + sum(c.y.nbytes for c in cohorts)
        self.planner = Planner(budget_bytes, self.d, self.B, backend=backend)
        self.planner.nm_bytes_per_rt = self._nm_charge()
        if not worklist:
            self.planner.row_temp_bytes = max(self.planner.row_temp_bytes, 60 + 52 * self.d)
            self.planner.dense_rows = True
        self._admit_seq = 0
        self._admit_order = {}
        self._evicted = set()
        self._groups = None
        self.ws = Workspace(self.dev, poison=debug_poison, pooled=workspace_pool)
        if PHASE_TIMING_ENV in ("0", "1"):
            phase_timing = PHASE_TIMING_ENV == "1"
        self.phase_timing = bool(phase_timing)
        self.tm = PhaseTimer(self.phase_timing, self.dev)
        self.debug_shuffle_wl = debug_shuffle_wl
        self.counters = []
        self.log = log
        self._alloc_flat(0)

    def _alloc_flat(self, S):
        dev = self.dev
        self.S = S
        self._layout_version = getattr(self, "_layout_version", 0) + 1
        f64 = lambda: torch.zeros(S, dtype=torch.float64, device=dev)
        self.eta, self.lam, self.mcw = f64(), f64(), f64()
        self.subsample, self.cs_tree, self.cs_level = f64(), f64(), f64()
        self.maxd = torch.zeros(S, dtype=torch.int32, device=dev)
        self.seed = torch.zeros(S, dtype=torch.int64, device=dev)
        self.local_round = torch.zeros(S, dtype=torch.int64, device=dev)
        self.bits = torch.zeros(S, dtype=torch.int8, device=dev)
        self.wide = torch.zeros(S, dtype=torch.bool, device=dev)
        self.active = torch.zeros(S, dtype=torch.bool, device=dev)
        self.n_cohort = torch.zeros(S, dtype=torch.int64, device=dev)
        self.par_hwm = torch.zeros(S, dtype=torch.int64, device=dev)
        self.nph_hwm = torch.zeros(S, dtype=torch.int64, device=dev)
        self.rt_off = torch.zeros(S + 1, dtype=torch.int64, device=dev)
        self.xrow = torch.zeros(0, dtype=torch.int32, device=dev)
        self.crow = torch.zeros(0, dtype=torch.int32, device=dev)
        self.pred = torch.zeros(0, dtype=torch.float32, device=dev)
        self.y_rt = torch.zeros(0, dtype=torch.int64 if self.loss == "softmax" else torch.float32, device=dev)
        self.is_train = torch.zeros(0, dtype=torch.bool, device=dev)
        self.slot_of_rt = torch.zeros(0, dtype=torch.int32, device=dev)
        self.RT = 0
        self.h = dict(rt_off=np.zeros(S + 1, dtype=np.int64), local_round=np.zeros(S, dtype=np.int64),
                      active=np.zeros(S, dtype=bool), wide=np.zeros(S, dtype=bool), maxd=np.zeros(S, dtype=np.int64),
                      subsample=np.zeros(S), cs_tree=np.zeros(S), cs_level=np.zeros(S), n_cohort=np.zeros(S, dtype=np.int64),
                      par_hwm=np.zeros(S, dtype=np.int64), bits=np.zeros(S, dtype=np.int8),
                      nph_hwm=np.zeros(S, dtype=np.int64),
                      lam=np.zeros(S), mcw=np.zeros(S),
                      seed=np.zeros(S, dtype=np.int64))
        self.wl_ub_slot = np.zeros(S, dtype=np.int64)
        self._groups = None
        self._hwm_dev_stale = False

    def _slot_params(self, st: ModelState):
        sp = st.spec
        return dict(eta=sp.eta, lam=sp.lam, mcw=sp.mcw, subsample=sp.subsample, cs_tree=sp.colsample_bytree,
                    cs_level=sp.colsample_bylevel, maxd=sp.max_depth, seed=sp.seed, bits=sp.quant_bits,
                    wide=(sp.resolve_mode() == "wide"), n_cohort=self.cohorts[sp.data_key].n)

    @staticmethod
    def _layout_sorted(train_rows, eval_rows):
        tr = np.unique(train_rows)
        ev = np.setdiff1d(eval_rows, tr)
        rows = np.concatenate([tr, ev]).astype(np.int64)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        is_tr = np.concatenate([np.ones(len(tr), bool), np.zeros(len(ev), bool)])[order]
        return rows, is_tr

    def _layout_of_spec(self, sp):
        tr, ev = sp.train_rows, sp.eval_rows
        n = self.cohorts[sp.data_key].n
        k = len(tr) + len(ev)
        ok = (tr.dtype.kind in "iu" and ev.dtype.kind in "iu" and tr.ndim == 1 and ev.ndim == 1 and 16 * k >= n
              and (len(tr) == 0 or (int(tr.min()) >= 0 and int(tr.max()) < n))
              and (len(ev) == 0 or (int(ev.min()) >= 0 and int(ev.max()) < n)))
        if not ok:
            return self._layout_sorted(tr, ev)
        flag = np.zeros(n, dtype=np.uint8)
        flag[ev] = 1
        flag[tr] = 2
        rows = np.flatnonzero(flag).astype(np.int64, copy=False)
        return rows, flag[rows] == 2

    class _Layout:
        __slots__ = ("key", "train_rows", "eval_rows", "rows", "is_tr", "dev", "ev_pos", "n_tr", "host")

        def __init__(self, key, train_rows, eval_rows, rows, is_tr):
            self.key, self.train_rows, self.eval_rows = key, train_rows, eval_rows
            self.rows, self.is_tr, self.dev = rows, is_tr, None
            self.ev_pos = None
            self.n_tr = None
            self.host = None

        @property
        def n_rt(self):
            return len(self.rows)

    def _layout(self, sp, cache):
        tr, ev = sp.train_rows, sp.eval_rows
        if cache is None:
            return self._layout_new(sp)
        ik = ("id", sp.data_key, id(tr), id(ev))
        lay = cache.get(ik)
        if lay is not None and lay.train_rows is tr and lay.eval_rows is ev:
            return lay
        ck = ("content", sp.data_key, tr.dtype.str, tr.shape, ev.dtype.str, ev.shape,
              tr[:1].tobytes(), tr[-1:].tobytes(), ev[:1].tobytes(), ev[-1:].tobytes())
        for lay in cache.get(ck, ()):
            if np.array_equal(lay.train_rows, tr) and np.array_equal(lay.eval_rows, ev):
                cache[ik] = lay
                return lay
        lay = self._layout_new(sp)
        cache.setdefault(ck, []).append(lay)
        cache[ik] = lay
        return lay

    def _layout_new(self, sp):
        tr, ev = sp.train_rows, sp.eval_rows
        if not PINNED_LIFECYCLE:
            return Engine._Layout(sp.data_key, tr, ev, *self._layout_of_spec(sp))
        ck = (sp.data_key, tr.dtype.str, tr.shape, ev.dtype.str, ev.shape, tr[:1].tobytes(), tr[-1:].tobytes(),
              ev[:1].tobytes(), ev[-1:].tobytes())
        for t0, e0, h in self._lay_host.get(ck, ()):
            if np.array_equal(t0, tr) and np.array_equal(e0, ev):
                break
        else:
            if sum(len(v) for v in self._lay_host.values()) >= LAY_RECORDS:
                self._lay_host.clear()
            rows, is_tr = self._layout_of_spec(sp)
            h = dict(rows=rows, is_tr=is_tr, n_tr=None, ev_pos=None, src=None)
            self._lay_host.setdefault(ck, []).append((np.array(tr, copy=True), np.array(ev, copy=True), h))
            self.pin_stats["layout_records"] += 1
        lay = Engine._Layout(sp.data_key, tr, ev, h["rows"], h["is_tr"])
        lay.n_tr, lay.ev_pos, lay.host = h["n_tr"], (None if h["ev_pos"] is None else h["ev_pos"].copy()), h
        return lay

    def _layout_src(self, lay):
        h = lay.host
        if h["src"] is None and not self._pin_failed:
            co = self.cohorts[lay.key]
            try:
                h["src"] = (torch.from_numpy(lay.rows.astype(np.int32)).pin_memory(),
                            torch.from_numpy(co.y[lay.rows].astype(np.int64 if self.loss == "softmax" else np.float32))
                            .pin_memory(),
                            torch.from_numpy(lay.is_tr).pin_memory())
            except RuntimeError:
                self._pin_failed = True
                return None
        return h["src"]

    def _layout_device(self, lay):
        if lay.dev is None and PINNED_LIFECYCLE and lay.host is not None and self.dev.type == "cuda":
            src = self._layout_src(lay)
            if src is not None:
                crow = src[0].to(self.dev, non_blocking=True)
                lay.dev = (crow, crow + self.cohort_off[lay.key], src[1].to(self.dev, non_blocking=True),
                           src[2].to(self.dev, non_blocking=True))
        if lay.dev is None:
            co = self.cohorts[lay.key]
            dev = self.dev
            crow = torch.from_numpy(lay.rows.astype(np.int32)).to(dev)
            xrow = crow + self.cohort_off[lay.key]
            y = torch.from_numpy(co.y[lay.rows].astype(np.int64 if self.loss == "softmax" else np.float32)).to(dev)
            lay.dev = (crow, xrow, y, torch.from_numpy(lay.is_tr).to(dev))
        return lay.dev

    def _segments_of_model(self, st: ModelState, cache=None):
        sp = st.spec
        lay = self._layout(sp, cache)
        crow, xrow, y, is_tr = self._layout_device(lay)
        if lay.ev_pos is None:
            lay.ev_pos = np.searchsorted(lay.rows, lay.eval_rows)
            if lay.host is not None:
                lay.host["ev_pos"] = lay.ev_pos
        st._ev_pos = lay.ev_pos
        if lay.n_tr is None:
            lay.n_tr = int(lay.is_tr.sum())
            if lay.host is not None:
                lay.host["n_tr"] = lay.n_tr
        st._n_train = lay.n_tr
        dev = self.dev
        n = len(crow)
        saved = self._paused.get(st.model_id)
        if saved is not None:
            if saved["pred_dev"] is not None:
                pred = saved["pred_dev"]
            elif saved.get("pred_t") is not None:
                pred = saved["pred_t"].to(dev, non_blocking=True)
                self.pin_stats["resume_pinned"] += 1
            else:
                pred = torch.from_numpy(saved["pred"]).to(dev)
        else:
            pred = torch.full((1, 1), float(sp.base_margin), dtype=torch.float32, device=dev).expand(self.K, n)
        return crow, xrow, pred, y, is_tr

    def _slot_ids_int32(self, S):
        RT = int(self.RT)
        ids = torch.zeros(RT, dtype=torch.int32, device=self.dev)
        if S > 1 and RT > 0:
            st = self.h["rt_off"][1:S]
            st = st[st < RT]
            if len(st):
                idx = torch.from_numpy(np.ascontiguousarray(st, dtype=np.int64)).to(self.dev)
                ids.index_add_(0, idx, torch.ones(len(st), dtype=torch.int32, device=self.dev))
            ids.cumsum_(0)
        return ids

    def _rebuild_add(self, st, layouts, seen):
        lb = 8 if self.loss == "softmax" else 4
        lay = self._layout(st.spec, layouts)
        b, ids = 0, set()
        if id(lay) not in seen:
            b += len(lay.rows) * (9 + lb)
            ids.add(id(lay))
        saved = self._paused.get(st.model_id)
        if saved is not None:
            b += self._paused_bytes(saved)
        return b, ids

    def _rebuild_extra(self, tot, extra_bytes):
        lb = 8 if self.loss == "softmax" else 4
        old = int(self.RT) if self.S else 0
        new = int(tot["rt"])
        peak = done = 0
        rest = (13 + lb) * old
        for w in (4, 4, 4, lb, 1):
            peak = max(peak, done + w * new + rest)
            done += w * new
            rest -= w * old
        peak = max(peak, done + 4 * new)
        return max(0, peak + int(extra_bytes) - self._persistent_of(tot))

    def _rebuild(self, order, new_states, layouts=None):
        K = self.K
        segs = []
        params = []
        h = self.h
        hwm = h["par_hwm"]
        nph = h["nph_hwm"]
        lens = []
        n_tr = []
        runs = []
        for j in order:
            mid = self.mslots[j]
            st = self.reg[mid]
            for k in range(K):
                s = j * K + k
                a, b = int(h["rt_off"][s]), int(h["rt_off"][s + 1])
                if runs and runs[-1][1] == a:
                    runs[-1][1] = b
                else:
                    runs.append([a, b])
                lens.append(b - a)
                n_tr.append(int(self.wl_ub_slot[s]))
                p = self._slot_params(st)
                p["local_round"] = int(h["local_round"][s])
                p["active"] = bool(h["active"][s])
                p["par_hwm"] = int(hwm[s])
                p["nph_hwm"] = int(nph[s])
                params.append(p)
        for a, b in runs:
            segs.append((self.crow[a:b], self.xrow[a:b], self.pred[a:b], self.y_rt[a:b], self.is_train[a:b]))
        cache = layouts if layouts is not None else {}
        for st in new_states:
            crow, xrow, pred, y, is_tr = self._segments_of_model(st, cache)
            ntr = int(st._n_train)
            for k in range(K):
                segs.append((crow, xrow, pred[k], y, is_tr))
                lens.append(len(crow))
                n_tr.append(ntr)
                p = self._slot_params(st)
                p["local_round"] = st.local_round
                p["active"] = True
                saved = self._paused.get(st.model_id)
                p["par_hwm"] = int(saved["par_hwm"][k]) if saved is not None else 0
                p["nph_hwm"] = int(saved["nph_hwm"][k]) if saved is not None else 0
                params.append(p)
            self._paused_pop(st.model_id)
            st.eval_margin = None
        new_mslots = [self.mslots[j] for j in order] + [st.model_id for st in new_states]
        S = len(new_mslots) * K
        self._alloc_flat(S)
        self.wl_ub_slot = np.asarray(n_tr, dtype=np.int64)
        self.mslots = new_mslots
        for j, mid in enumerate(self.mslots):
            self.reg[mid].slot = j
        if S == 0:
            self.acc.persistent = 0
            self.acc.set_transient(self.ws.bytes())
            return
        for name in ("eta", "lam", "mcw", "subsample", "cs_tree", "cs_level", "maxd", "seed", "local_round", "bits",
                     "wide", "active", "n_cohort", "par_hwm", "nph_hwm"):
            vals = [p[name] for p in params]
            getattr(self, name).copy_(torch.tensor(vals, dtype=getattr(self, name).dtype))
            if name in self.h:
                self.h[name] = np.asarray(vals, dtype=self.h[name].dtype)
        lens = torch.tensor(lens, dtype=torch.int64)
        self.h["rt_off"] = np.concatenate([[0], np.cumsum(lens.numpy())]).astype(np.int64)
        self.rt_off = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)]).to(self.dev)
        cols = [list(c) for c in zip(*segs)]
        segs = None
        self.slot_of_rt = None
        for i, name in enumerate(("crow", "xrow", "pred", "y_rt", "is_train")):
            setattr(self, name, torch.cat(cols[i]))
            cols[i] = None
        self.RT = len(self.crow)
        self.slot_of_rt = self._slot_ids_int32(S)
        self.acc.persistent = self.planner.persistent_bytes(self.RT, n_slots=self.S,
                                                           label_bytes=8 if self.loss == "softmax" else 4)
        self.acc.set_transient(self.ws.bytes())

    def add_models(self, specs, admit=True):
        ids = []
        for sp in specs:
            assert sp.data_key in self.cohorts
            if not self.collect_metrics and sp.early_stopping_rounds:
                raise ValueError("early stopping requires collect_metrics=True")
            st = self.reg.add(sp, self.K)
            self.pending.append(st.model_id)
            ids.append(st.model_id)
        if admit:
            self.admit_pending()
        return ids

    def admit_pending(self):
        if not self.pending:
            return []
        admitted = []
        pl = self.planner
        live = [j for j, m in enumerate(self.mslots) if m is not None]
        tot = self._terms_total([(self.reg[self.mslots[j]], j) for j in live])
        self.acc.persistent = self._persistent_of(tot)
        new_states = []
        layouts = {}
        rmemo = dict(seen=set(), bytes=0)
        for mid in list(self.pending):
            st = self.reg[mid]
            if self.max_model_slots is not None and len(live) + len(new_states) + 1 > self.max_model_slots:
                break
            cand = self._terms_add(tot, self._model_terms(st, None, layouts))
            radd = self._rebuild_add(st, layouts, rmemo["seen"])
            rx = self._rebuild_extra(cand, rmemo["bytes"] + radd[0])
            ok = pl.can_admit(self.acc, self._persistent_of(cand), max(self._transient_of(cand), rx))
            if not ok and tot["n"] == 0:
                need = pl.admission_need(self.acc, self._persistent_of(cand),
                                         max(self._transient_of(cand, minimal=True), rx))
                ok = need <= pl.budget
            if not ok:
                break
            tot = cand
            self.acc.persistent = self._persistent_of(tot)
            new_states.append(st)
            rmemo["bytes"] += radd[0]
            rmemo["seen"] |= radd[1]
            self.pending.remove(mid)
            st.status = "active"
            st.admitted_at = self.step_index
            self._admit_seq += 1
            self._admit_order[mid] = self._admit_seq
            ev = "readmit" if mid in self._evicted else ("resume" if mid in self._paused else "admit")
            self._evicted.discard(mid)
            admitted.append(mid)
            self.events.append(dict(step=self.step_index, event=ev, model=mid))
        if new_states or len(live) < len(self.mslots):
            if (self.ws.bytes() > tot["hist"] + 2 * tot["cell"] or
                    self.acc.shared + self.acc.persistent + self.ws.bytes() + pl.reserve
                    + self._rebuild_extra(tot, rmemo["bytes"]) > pl.budget):
                self.ws.buf.clear()
            self._spill_paused(keep=[st.model_id for st in new_states])
            self._pause_dev_ok = False
            self._rebuild(live, new_states, layouts)
        return admitted

    def _model_terms(self, st, j=None, layouts=None):
        sp, K, pl, h = st.spec, self.K, self.planner, self.h
        n_rt = int(h["rt_off"][j * K + 1] - h["rt_off"][j * K]) if j is not None else self._layout(sp, layouts).n_rt
        n_tr = len(sp.train_rows)
        wide = sp.resolve_mode() == "wide"
        k = pl.nph_prior(sp.max_depth, n_tr)
        obs = None
        if j is not None and int(h["local_round"][j * K]) > 0:
            obs = int(h["nph_hwm"][j * K:(j + 1) * K].max())
        elif st.model_id in self._paused and st.local_round > 0:
            obs = int(np.max(self._paused[st.model_id]["nph_hwm"]))
        if obs is not None:
            k = min(k, int(math.ceil(obs * pl.headroom)))
        kc = 2 * K * max(k, 1)
        return dict(n=1, sample=sp.subsample < 1.0, rt=K * n_rt, w=K * n_tr, maxw=K * n_tr, hist=pl.model_hist_bytes(k, wide, K),
                    hist_min=pl.model_hist_bytes(k, wide, K, buf_frac=0.0),
                    cand_p=0 if wide else kc, cand_w=kc if wide else 0, maxrows=K * n_rt, cell=pl.cell_bytes(wide),
                    nm_ok=bool(sp.max_depth <= NM_MAX_DEPTH and sp.quant_bits == 16))

    @staticmethod
    def _terms_add(a, b):
        return dict(n=a["n"] + b["n"], sample=a["sample"] or b["sample"], rt=a["rt"] + b["rt"], w=a["w"] + b["w"], hist=a["hist"] + b["hist"],
                    hist_min=a["hist_min"] + b["hist_min"],
                    cand_p=a["cand_p"] + b["cand_p"], cand_w=a["cand_w"] + b["cand_w"],
                    maxrows=max(a["maxrows"], b["maxrows"]), maxw=max(a["maxw"], b["maxw"]),
                    cell=max(a["cell"], b["cell"]), nm_ok=a.get("nm_ok", True) and b.get("nm_ok", True))

    def _terms_total(self, items):
        tot = dict(n=0, sample=False, rt=0, w=0, maxw=0, hist=0, hist_min=0, cand_p=0, cand_w=0, maxrows=0, cell=0,
                   nm_ok=True)
        for st, j in items:
            tot = self._terms_add(tot, self._model_terms(st, j))
        return tot

    def _persistent_of(self, tot):
        return self.planner.persistent_bytes(tot["rt"], n_slots=tot["n"] * self.K,
                                             label_bytes=8 if self.loss == "softmax" else 4) if tot["n"] else 0

    def _transient_of(self, tot, minimal=False):
        pl = self.planner
        pl.nm_bytes_per_rt = self._nm_charge(tot.get("nm_ok", True))
        self._set_chunk_coef(tot["sample"])
        g = max(min(pl.row_chunk_rows, tot["rt"]), tot["maxrows"])
        dense = getattr(pl, "dense_rows", False)
        if minimal:
            return pl.round_need(tot["rt"], tot["w"], g, 2 * tot["cell"],
                                 pl.split_bytes(min(tot["cand_p"], 1), min(tot["cand_w"], 1)),
                                 tot["w"] if dense else tot["maxw"])
        T = tot["w"] if dense else max(int(pl.min_chunk_frac * tot["w"]), tot["maxw"])
        return pl.round_need(tot["rt"], tot["w"], g, tot["hist"] + 2 * tot["cell"],
                             pl.split_bytes(tot["cand_p"], tot["cand_w"]), min(T, tot["w"]))

    def _resident_need(self):
        live = [j for j, m in enumerate(self.mslots) if m is not None]
        tot = self._terms_total([(self.reg[self.mslots[j]], j) for j in live])
        return self.planner.admission_need(self.acc, self._persistent_of(tot), self._transient_of(tot))

    def _parent_estimate(self, j, host=None):
        s = j * self.K
        st = self.reg[self.mslots[j]]
        lr, hwm = (self.h["local_round"], self.h["par_hwm"]) if host is None else (host["lr"], host["hwm"])
        if int(lr[s]) > 0:
            return int(max(hwm[s:s + self.K]))
        return self.planner.parent_prior(st.spec.max_depth, len(st.spec.train_rows))

    def retire(self, ids, reason="retired"):
        self._sync_hwm_dev()
        for mid in ids:
            st = self.reg[mid]
            if st.status in ("paused", "pending"):
                if mid in self.pending:
                    self.pending.remove(mid)
                self._paused_pop(mid)
                st.status, st.stop_reason = "retired", reason
                self.events.append(dict(step=self.step_index, event="finish", model=mid,
                                        reason=reason, local_round=st.local_round))
                continue
            if st.status != "active":
                continue
            self._finish(st, reason)

    def pause(self, ids):
        self._sync_hwm_dev()
        K = self.K
        items, seen = [], set()
        for mid in ids:
            st = self.reg[mid]
            if st.status != "active" or mid in seen:
                continue
            seen.add(mid)
            items.append((st, st.slot))
        if not items:
            return
        snaps = self._pause_batch(items)
        for (st, j), (entry, em) in zip(items, snaps):
            self._paused[st.model_id] = entry
            if entry["pred_dev"] is not None:
                self._paused_dev_bytes += 4 * entry["pred_dev"].numel()
            st.eval_margin = em
        for st, j in items:
            st.status, st.slot = "paused", None
            self.h["active"][j * K:(j + 1) * K] = False
            self.mslots[j] = None
            self.events.append(dict(step=self.step_index, event="pause", model=st.model_id,
                                    local_round=st.local_round))
        self.compact()

    EV_CACHE_BYTES = 16 << 20
    DEV_CHUNK_BYTES = 64 << 20

    def _ev_cache_drop(self):
        self._ev_cache.clear()
        self._ev_cache_bytes = 0

    def _ev_dev(self, pos):
        c = self._ev_cache
        hit = c.get(id(pos))
        if hit is not None and hit[0] is pos:
            return hit[1]
        nb = 8 * len(pos)
        if nb > self.EV_CACHE_BYTES:
            return self._h2d(pos)
        if len(c) >= 256 or self._ev_cache_bytes + nb > self.EV_CACHE_BYTES:
            self._ev_cache_drop()
        t = self._h2d(pos)
        c[id(pos)] = (pos, t)
        self._ev_cache_bytes += nb
        return t

    def _ev_host(self, st, j):
        K, h = self.K, self.h
        a, b = int(h["rt_off"][j * K]), int(h["rt_off"][j * K + K])
        P = self.pred[a:b].cpu().numpy().reshape(K, -1)
        out = [P[k][st._ev_pos] for k in range(K)]
        return out[0] if K == 1 else np.stack(out, 1)

    def _ev_gather(self, items):
        K, h = self.K, self.h
        out = [None] * len(items)
        chunks, cur, cb = [], [], 0
        for i, (st, j) in enumerate(items):
            nb = 4 * K * len(st._ev_pos)
            if nb == 0:
                out[i] = np.zeros(0 if K == 1 else (0, K), dtype=np.float32)
            elif nb > self.DEV_CHUNK_BYTES:
                out[i] = self._ev_host(st, j)
            else:
                if cur and cb + nb > self.DEV_CHUNK_BYTES:
                    chunks.append(cur)
                    cur, cb = [], 0
                cur.append(i)
                cb += nb
        if cur:
            chunks.append(cur)
        for ch in chunks:
            try:
                parts = []
                for i in ch:
                    st, j = items[i]
                    a, b = int(h["rt_off"][j * K]), int(h["rt_off"][j * K + K])
                    parts.append(self.pred[a:b].view(K, (b - a) // K).index_select(1, self._ev_dev(st._ev_pos))
                                 .reshape(-1))
                buf = self._d2h_pinned(parts[0] if len(parts) == 1 else torch.cat(parts))
            except torch.OutOfMemoryError:
                buf = None
            parts = None
            if buf is None:
                for i in ch:
                    out[i] = self._ev_host(*items[i])
                continue
            o = 0
            for i in ch:
                n_ev = len(items[i][0]._ev_pos)
                m = buf[o:o + K * n_ev].reshape(K, n_ev)
                out[i] = m[0].copy() if K == 1 else m.T.copy()
                o += K * n_ev
        return out

    def _paused_bytes(self, saved):
        P = saved["pred_dev"]
        return 4 * P.numel() if P is not None else int(saved["pred"].nbytes)

    def _paused_pop(self, mid):
        s = self._paused.pop(mid, None)
        if s is not None and s["pred_dev"] is not None:
            self._paused_dev_bytes -= 4 * s["pred_dev"].numel()
        return s

    def _pause_batch(self, items):
        K, h, pl = self.K, self.h, self.planner
        dev_ok = self._pause_dev_ok
        if dev_ok:
            need, seen = 0, set()
            for st, j in items:
                pos = st._ev_pos
                need += 4 * int(h["rt_off"][j * K + K] - h["rt_off"][j * K]) + 4 * K * len(pos)
                if id(pos) not in self._ev_cache and id(pos) not in seen:
                    seen.add(id(pos))
                    need += 8 * len(pos)
            avail = pl.budget - (self.acc.shared + self.acc.persistent + self.ws.bytes() + pl.reserve
                                 + self._paused_dev_bytes + self._ev_cache_bytes)
            dev_ok = need <= avail
        snaps = None
        if dev_ok:
            try:
                margins = self._ev_gather(items)
                snaps = []
                for (st, j), em in zip(items, margins):
                    a, b = int(h["rt_off"][j * K]), int(h["rt_off"][j * K + K])
                    P = self.pred[a:b].clone().view(K, (b - a) // K)
                    snaps.append((dict(pred=None, pred_dev=P, par_hwm=h["par_hwm"][j * K:(j + 1) * K].copy(),
                                       nph_hwm=h["nph_hwm"][j * K:(j + 1) * K].copy()), em))
            except torch.OutOfMemoryError:
                snaps = P = None
            margins = None
        if snaps is None:
            snaps = self._snapshot_batch(items)
        return snaps

    def _nm_grad(self, K, pred, y, mrow_base, mrow_len):
        if not (PINNED_LIFECYCLE and FUSED_GRADIENTS and self.dev.type == "cuda"):
            return None
        f = getattr(self._nm_mod, "grad_hess_nm", None)
        return None if f is None else f(self.loss, K, pred, y, mrow_base, mrow_len)

    def _pin_alloc(self, n):
        if self._pin_failed:
            return None
        nb = 1 << max(0, (4 * int(n) - 1).bit_length())
        cls = self._pin_cls.setdefault(nb, [0, 0])
        if cls[0] + nb > cls[1] and sum(c[1] for c in self._pin_cls.values()) + nb > self._pin_cap:
            self.pin_stats["pin_cap_fallbacks"] += 1
            return None
        try:
            buf = torch.empty(int(n), dtype=torch.float32, pin_memory=True)
        except RuntimeError:
            self._pin_failed = True
            self.pin_stats["pin_fail"] += 1
            return None
        self._pin_live[0] += nb
        cls[0] += nb
        cls[1] = max(cls[1], cls[0])
        _weakref.finalize(buf, _pin_release, self._pin_live, cls, nb)
        self.pin_stats["pin_blocks"] += 1
        self.pin_stats["pin_bytes"] += nb
        return buf

    def _d2h_pinned(self, src):
        if not (PINNED_LIFECYCLE and src.is_cuda) or self._pin_failed:
            return src.cpu().numpy()
        try:
            hb = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        except RuntimeError:
            self._pin_failed = True
            return src.cpu().numpy()
        hb.copy_(src)
        return hb.numpy()

    def _snapshot_batch(self, items):
        if not (PINNED_LIFECYCLE and self.dev.type == "cuda"):
            return [self._snapshot_of(st, j) for st, j in items]
        K, h = self.K, self.h
        spans = [(int(h["rt_off"][j * K]), int(h["rt_off"][j * K + K])) for st, j in items]
        out = [None] * len(items)
        pend = []
        for i, (a, b) in enumerate(spans):
            buf = self._pin_alloc(b - a)
            if buf is None:
                out[i] = self._snapshot_of(*items[i])
                continue
            buf.copy_(self.pred[a:b], non_blocking=True)
            pend.append((i, buf, buf))
        if pend:
            torch.cuda.current_stream(self.dev).synchronize()
        for i, v, buf in pend:
            st, j = items[i]
            pt = v.view(K, -1)
            P = pt.numpy()
            entry = dict(pred=P, pred_dev=None, par_hwm=h["par_hwm"][j * K:(j + 1) * K].copy(),
                         nph_hwm=h["nph_hwm"][j * K:(j + 1) * K].copy(), pred_t=pt, pin_chunk=buf)
            ev = [P[k][st._ev_pos] for k in range(K)]
            out[i] = (entry, (ev[0] if K == 1 else np.stack(ev, 1)).copy())
            self.pin_stats["pin_models"] += 1
        return out

    def _spill_paused(self, keep=()):
        if not self._paused_dev_bytes:
            return
        if not (PINNED_LIFECYCLE and self.dev.type == "cuda"):
            return self._spill_paused_unpinned(keep)
        keep = set(keep)
        mids = [mid for mid, s in self._paused.items() if s["pred_dev"] is not None and mid not in keep]
        if not mids:
            return
        pend, rest = [], []
        for m in mids:
            P = self._paused[m]["pred_dev"]
            n = P.numel()
            buf = self._pin_alloc(n)
            if buf is None:
                rest.append(m)
                continue
            buf.copy_(P.reshape(-1), non_blocking=True)
            pend.append((m, buf.view(P.shape), buf, n))
        if pend:
            torch.cuda.current_stream(self.dev).synchronize()
            for m, pt, buf, n in pend:
                s = self._paused[m]
                s["pred_t"], s["pin_chunk"], s["pred"] = pt, buf, pt.numpy()
                s["pred_dev"] = None
                self._paused_dev_bytes -= 4 * n
                self.pin_stats["pin_models"] += 1
        if rest:
            rs = set(rest)
            self._spill_paused_unpinned(keep=[m for m in self._paused if m not in rs])

    def _spill_paused_unpinned(self, keep=()):
        if not self._paused_dev_bytes:
            return
        keep = set(keep)
        mids = [mid for mid, s in self._paused.items() if s["pred_dev"] is not None and mid not in keep]
        chunks, cur, cb = [], [], 0
        for mid in mids:
            nb = 4 * self._paused[mid]["pred_dev"].numel()
            if nb > self.DEV_CHUNK_BYTES:
                chunks.append([mid])
            else:
                if cur and cb + nb > self.DEV_CHUNK_BYTES:
                    chunks.append(cur)
                    cur, cb = [], 0
                cur.append(mid)
                cb += nb
        if cur:
            chunks.append(cur)
        for ch in chunks:
            buf = None
            if len(ch) > 1:
                try:
                    buf = torch.cat([self._paused[mid]["pred_dev"].reshape(-1) for mid in ch]).cpu().numpy()
                except torch.OutOfMemoryError:
                    buf = None
            o = 0
            for mid in ch:
                s = self._paused[mid]
                P = s["pred_dev"]
                n = P.numel()
                if buf is None:
                    s["pred"] = P.cpu().numpy()
                else:
                    s["pred"] = buf[o:o + n].copy().reshape(P.shape)
                    o += n
                s["pred_dev"] = None
                self._paused_dev_bytes -= 4 * n

    def _snapshot(self, mid):
        st = self.reg[mid]
        self._paused[mid], st.eval_margin = self._snapshot_batch([(st, st.slot)])[0]

    def _snapshot_of(self, st, j):
        K, h = self.K, self.h
        a, b = int(h["rt_off"][j * K]), int(h["rt_off"][j * K + K])
        P = self.pred[a:b].cpu().numpy().reshape(K, -1)
        entry = dict(pred=P.copy(), pred_dev=None, par_hwm=h["par_hwm"][j * K:(j + 1) * K].copy(),
                     nph_hwm=h["nph_hwm"][j * K:(j + 1) * K].copy())
        out = [P[k][st._ev_pos] for k in range(K)]
        return entry, (out[0] if K == 1 else np.stack(out, 1)).copy()

    def _degrade(self, msg, needed, available):
        self.ws.buf.clear()
        live = [m for m in self.mslots if m is not None]
        if len(live) <= 1:
            raise MemoryError(f"{msg}; a single resident model does not fit the budget")
        pl = self.planner
        est = self._resident_need()
        pl.admit_ceiling = est if pl.admit_ceiling is None else min(pl.admit_ceiling, est)
        short = max(1, needed - available)
        order = sorted(live, key=lambda m: self._admit_order.get(m, 0), reverse=True)
        evict, freed = [], 0
        for mid in order[:-1]:
            t = self._model_terms(self.reg[mid], self.reg[mid].slot)
            evict.append(mid)
            freed += self._persistent_of(t) + pl.level_row_bytes(t["rt"], t["w"], t["w"]) + t["hist"]
            if freed >= short:
                break
        K = self.K
        for mid in evict:
            st = self.reg[mid]
            j = st.slot
            self._snapshot(mid)
            st.status, st.slot = "pending", None
            self.active[j * K:(j + 1) * K] = False
            self.h["active"][j * K:(j + 1) * K] = False
            self.mslots[j] = None
            self._evicted.add(mid)
            self.events.append(dict(step=self.step_index, event="evict", model=mid, local_round=st.local_round,
                                    needed=int(needed), available=int(available)))
        self.pending[:0] = sorted(evict, key=lambda m: self._admit_order.get(m, 0))
        self.n_evictions = getattr(self, "n_evictions", 0) + len(evict)
        self._rebuild([j for j, m in enumerate(self.mslots) if m is not None], [])

    def resume(self, ids, admit=True):
        self._sync_hwm_dev()
        for mid in ids:
            st = self.reg[mid]
            if st.status != "paused":
                raise ValueError(f"model {mid} is {st.status}, expected paused")
            st.status = "pending"
            self.pending.append(mid)
        return self.admit_pending() if admit else []

    def _finish(self, st: ModelState, reason):
        self._finish_batch([(st, reason)])

    def _finish_batch(self, finished):
        K = self.K
        withev = [(st, st.slot) for st, _ in finished if len(st.spec.eval_rows)]
        if withev:
            for (st, _), em in zip(withev, self._ev_gather(withev)):
                st.eval_margin = em
        for st, reason in finished:
            j = st.slot
            st.status = "done" if reason in ("n_rounds", "early_stop") else "retired"
            st.stop_reason = reason
            self.active[j * K:(j + 1) * K] = False
            self.h["active"][j * K:(j + 1) * K] = False
            self.mslots[j] = None
            st.slot = None
            self.events.append(dict(step=self.step_index, event="finish", model=st.model_id, reason=reason,
                                    local_round=st.local_round))

    def compact(self, order=None):
        self._sync_hwm_dev()
        live = [j for j, m in enumerate(self.mslots) if m is not None]
        if order is not None:
            pos = {self.mslots[j]: j for j in live}
            live = [pos[mid] for mid in order]
            assert len(live) == len(pos)
        self.events.append(dict(step=self.step_index, event="compact", order=[self.mslots[j] for j in live]))
        self._rebuild(live, [])

    def run(self, max_steps=None, callback=None):
        n = 0
        while (self.pending or any(m is not None for m in self.mslots)) and (max_steps is None or n < max_steps):
            if self.pending and not self.active_ids() and not self.admit_pending():
                st = self.reg[self.pending[0]]
                t = self._model_terms(st, None, {})
                need = self.planner.admission_need(self.acc, self._persistent_of(t), self._transient_of(t, minimal=True))
                raise MemoryError(f"budget {self.planner.budget} B admits none of the {len(self.pending)} pending models "
                                  f"(model {st.model_id} alone needs >= {need} B: persistent state, round row state and "
                                  "histogram workspace for one parent group per chunk)")
            self.step()
            n += 1
            if callback:
                callback(self)
        return n

    def active_ids(self):
        return [m for m in self.mslots if m is not None]

    def step(self):
        with self.tm("admission"):
            self.admit_pending()
        self._spill_paused()
        self._ev_cache_drop()
        if not any(m is not None for m in self.mslots):
            self.step_index += 1
            return
        gc_ctl = NM_GC != "" and _gc.isenabled()
        if gc_ctl:
            if NM_GC == "freeze":
                _gc.freeze()
            else:
                _gc.disable()
        try:
            while True:
                fail = None
                try:
                    self._round()
                except LevelInfeasible as e:
                    fail = (str(e), e.needed, e.available)
                if fail is None:
                    break
                self._degrade(*fail)
        finally:
            if gc_ctl and NM_GC == "disable":
                _gc.enable()
        self._pause_dev_ok = True
        self.step_index += 1
        self.events.append(dict(step=self.step_index, event="step"))

    def _mrow(self, s0=0, s1=None, a=0, b=None):
        dev, K = self.dev, self.K
        s1 = self.S if s1 is None else s1
        b = self.RT if b is None else b
        if self.loss != "softmax":
            return None, None
        if K == 1:
            return torch.arange(b - a, device=dev), torch.zeros(b - a, dtype=torch.int64, device=dev)
        idx = torch.arange(s0 // K, s1 // K, device=dev) * K
        starts = self.rt_off[idx] - a
        lens = self.rt_off[idx + 1] - self.rt_off[idx]
        excl = torch.cumsum(lens, 0) - lens
        Nm = (b - a) // K
        base = torch.repeat_interleave(starts - excl, lens, output_size=Nm) + torch.arange(Nm, device=dev)
        return base, torch.repeat_interleave(lens, lens, output_size=Nm)

    def _row_groups(self, cap):
        K = self.K
        key = (int(cap), self.S, self.RT)
        if self._groups is not None and self._groups[0] == key:
            return self._groups[1]
        off = self.h["rt_off"][::K]
        nmod = self.S // K
        groups, m0 = [], 0
        while m0 < nmod:
            m1 = int(np.searchsorted(off, off[m0] + cap, side="right")) - 1
            m1 = min(max(m1, m0 + 1), nmod)
            groups.append((m0 * K, m1 * K, int(off[m0]), int(off[m1])))
            m0 = m1
        self._groups = (key, groups)
        return groups

    def _set_chunk_coef(self, sample):
        pl = self.planner
        pl.chunk_row_bytes = pl.chunk_bytes(sample=bool(sample), metric=self.collect_metrics, softmax=self.loss == "softmax")

    def _round_groups(self, w_ub):
        pl, acc = self.planner, self.acc
        self._set_chunk_coef(bool(((self.h["subsample"] < 1.0) & self.h["active"]).any()))
        free = (pl.budget - acc.shared - acc.persistent - pl.reserve - self.ws.bytes()
                - pl.round_state_bytes(self.RT, w_ub) - 4 * w_ub)
        cap = max(pl.row_chunk_rows, free // max(pl.chunk_row_bytes, 1))
        groups = self._row_groups(cap)
        return groups, max((b - a for _, _, a, b in groups), default=0)

    def _plan_leaf_row_groups(self, W):
        groups, gmax = self._round_groups(W)
        pl, acc = self.planner, self.acc
        need = self.ws.bytes() + pl.row_phase_bytes(self.RT, W, gmax)
        available = pl.budget - acc.shared - acc.persistent - pl.reserve
        if need > available:
            raise LevelInfeasible(f"leaf row phase needs {need} B but only {max(available, 0)} B remain after "
                                  "shared/persistent state and reserve; reduce the resident batch", need, available)
        return groups, gmax

    def _colsample_mask(self, level_key, base_mask, frac):
        S, d = self.S, self.d
        dev = self.dev
        f = torch.arange(d, device=dev, dtype=torch.int64)
        tag = torch.full((), self.KR.COL_STREAM, device=dev, dtype=torch.int64)
        u = self.KR.hash_u01(tag, self.seed[:, None], self.local_round[:, None], torch.full((S, 1), level_key, device=dev, dtype=torch.int64), f[None, :])
        u = torch.where(base_mask, u, torch.full_like(u, 2.0))
        nbase = base_mask.sum(1)
        k = torch.clamp(torch.floor(frac.to(torch.float32) * nbase.to(torch.float32)).to(torch.int64), min=1)
        rank = torch.empty_like(u, dtype=torch.int64)
        rank.scatter_(1, torch.argsort(u, dim=1, stable=True), torch.arange(d, device=dev).expand(S, d))
        return (rank < k[:, None]) & base_mask

    def _round(self):
        dev, S, K, d, B = self.dev, self.S, self.K, self.d, self.B
        pl, acc = self.planner, self.acc
        hm = self.h
        if CHECK_MIRRORS:
            self._check_mirrors()
        h_act = hm["active"]
        act = self.active
        RT = self.RT
        w_ub = int(self.wl_ub_slot[h_act].sum()) if len(h_act) else 0
        nm_on = HOST_LOOP and self.use_worklist and self.use_frontier and self._nodemap_ok()
        self._round_pack = None
        nm_saved = dict(self.nm_stats)
        fsamp = nm_on and self._fused_sampling_on()
        if getattr(self, "_lr_dev_stale", False) and (not nm_on or (not fsamp and bool((((hm["subsample"] < 1.0)
                                                                          | (hm["cs_tree"] < 1.0) | (hm["cs_level"] < 1.0)) & h_act).any()))):
            self._sync_lr_dev()
        if self.nm_require and not nm_on and bool(self.h["active"].any()):
            raise RuntimeError("MANGROVE_NODEMAP_REQUIRE=1 but this round cannot use the node-map path "
                               "(max_depth > 7, 28-bit words, nodemap=False or an ablation switch)")
        pl.nm_bytes_per_rt = 1 if nm_on else 0
        groups, gmax = self._round_groups(w_ub)
        r_need = pl.row_phase_bytes(RT, w_ub, gmax)
        base = acc.shared + acc.persistent + pl.reserve
        if base + self.ws.bytes() + r_need > pl.budget:
            self.ws.buf.clear()
            groups, gmax = self._round_groups(w_ub)
            r_need = pl.row_phase_bytes(RT, w_ub, gmax)
        if base + r_need > pl.budget:
            raise LevelInfeasible(f"round row state needs {r_need} B but only {max(pl.budget - base, 0)} B remain after "
                                  "shared/persistent state (no histogram workspace allocated); reduce the resident batch",
                                  r_need, pl.budget - base)
        sub_h = (hm["subsample"] < 1.0) & h_act
        row_tab = self._nm_rowmask_tab() if fsamp and bool(sub_h.any()) else None
        n_w = np.where(h_act, self.wl_ub_slot, 0).astype(np.int64)
        static_all = HAS_NONZERO_STATIC and not bool(sub_h.any())
        has_hess = self.loss != "squared"
        one = len(groups) == 1
        if not one:
            work = torch.zeros(RT, dtype=torch.bool, device=dev)
            gh = torch.zeros(RT, dtype=torch.int64, device=dev)
            sg = torch.ones(S, dtype=torch.float64, device=dev)
            sh = torch.ones(S, dtype=torch.float64, device=dev)
        wl_parts = []
        for s0, s1, a, b in groups:
            if not h_act[s0:s1].any():
                continue
            whole = s0 == 0 and s1 == S
            sl = self.slot_of_rt if whole else self.slot_of_rt[a:b] - s0
            pv = (lambda t: t) if whole else (lambda t: t[s0:s1])
            rv = (lambda t: t) if whole else (lambda t: t[a:b])
            samp = bool(sub_h[s0:s1].any())
            if row_tab is not None and samp:
                w = self._nm_rowmask(a, b, row_tab)
            else:
                act_rt = pv(act).index_select(0, sl)
                w = rv(self.is_train) & act_rt
                del act_rt
                samp = bool(sub_h[s0:s1].any())
                if samp:
                    sl64 = sl.to(torch.int64)
                    sub = pv(self.subsample)[sl64]
                    tag = torch.full((), self.KR.ROW_STREAM, device=dev, dtype=torch.int64)
                    u = self.KR.hash_u01(tag, pv(self.seed)[sl64], pv(self.local_round)[sl64], rv(self.crow).to(torch.int64))
                    del sl64
                    w = w & (u < sub)
                    del u, sub
            mrow_base, mrow_len = self._mrow(s0, s1, a, b)
            with self.tm("grad_quantize"):
                gh2 = self._nm_grad(K, rv(self.pred), rv(self.y_rt), mrow_base, mrow_len) if nm_on else None
                g, h = gh2 if gh2 is not None else self.KR.grad_hess(self.loss, K, rv(self.pred), rv(self.y_rt),
                                                                     mrow_base, mrow_len)
                del gh2
                if nm_on:
                    ghg, sgg, shg = self._nm_quantize(g, h, sl, w, pv(self.bits), s0, s1, has_hess,
                                                      self.rt_off if whole else self.rt_off[s0:s1 + 1] - a)
                else:
                    ghg, sgg, shg = self.KR.quantize(g, h, sl, w, pv(self.bits), s1 - s0, has_hess=has_hess,
                                                     rt_off=self.rt_off if whole else self.rt_off[s0:s1 + 1] - a)
            del g, h, mrow_base, mrow_len
            if one:
                work, gh, sg, sh = w, ghg, sgg, shg
            else:
                work[a:b] = w
                gh[a:b] = ghg
                sg[s0:s1] = sgg
                sh[s0:s1] = shg
            del ghg
            if nm_on:
                del w, sl
            else:
                if HAS_NONZERO_STATIC and not samp:
                    wi = torch.nonzero_static(w, size=int(n_w[s0:s1].sum())).squeeze(1)
                    if DEBUG_CHECKS:
                        assert torch.equal(wi, torch.nonzero(w).squeeze(1)), "host work-list length"
                else:
                    wi = torch.nonzero(w).squeeze(1)
                wl_parts.append((wi if a == 0 else wi + a).to(torch.int32))
                del wi, w, sl
        if nm_on:
            wl = wl_off = None
            del wl_parts
            nm, W, rb0, cnt_h, sg, sh = self._nm_prologue(gh, work, sg, sh, one, n_w, not bool(sub_h.any()))
        else:
            wl = wl_parts[0] if len(wl_parts) == 1 else torch.cat(wl_parts)
            del wl_parts
            wl_off_h = None
            if static_all:
                wl_off_h = np.zeros(S + 1, dtype=np.int64)
                np.cumsum(n_w, out=wl_off_h[1:])
                wl_off = None
                if DEBUG_CHECKS:
                    assert np.array_equal(wl_off_h, torch.searchsorted(wl, self.rt_off.to(torch.int32)).cpu().numpy())
            else:
                wl_off = torch.searchsorted(wl, self.rt_off.to(torch.int32)).to(torch.int64)
            if self.debug_shuffle_wl is not None:
                gen = torch.Generator().manual_seed(int(self.debug_shuffle_wl) + self.step_index)
                key = self.slot_of_rt[wl.to(torch.int64)].to(torch.float64) + torch.rand(len(wl), generator=gen, dtype=torch.float64)
                wl = wl[torch.argsort(key)]
            W = len(wl)
        acc.set_transient(self.ws.bytes() + pl.row_phase_bytes(RT, W, gmax))
        if nm_on:
            allf = self._nm_allf()
        else:
            allf = torch.ones((S, d), dtype=torch.bool, device=dev)
        if fsamp and bool((hm["cs_tree"][h_act] < 1.0).any()):
            fm_tree = self._nm_colsample(0, None)
        else:
            fm_tree = self._colsample_mask(0, allf, self.cs_tree) if bool((hm["cs_tree"][h_act] < 1.0).any()) else allf
        fast = HOST_LOOP and self.use_worklist and self.use_frontier
        if not fast:
            self._sync_hwm_dev()
        saved = (self.par_hwm, self.nph_hwm, hm["par_hwm"].copy(), hm["nph_hwm"].copy(), len(self.counters),
                 self._hwm_dev_stale)
        try:
            with self.tm("round_total"):
                if nm_on:
                    self._grow_trees_nm(nm, W, rb0, cnt_h, gh, sg, sh, fm_tree, work, groups, gmax)
                    nm = rb0 = None
                elif fast:
                    self.nm_stats["worklist_rounds"] += 1
                    self._grow_trees_fast(wl, wl_off, gh, sg, sh, fm_tree, work, groups, gmax, wl_off_h=wl_off_h)
                else:
                    if wl_off is None:
                        wl_off = self._h2d(wl_off_h)
                    self._grow_trees(wl, wl_off, gh, sg, sh, fm_tree, work, groups, gmax)
        except LevelInfeasible:
            self.par_hwm, self.nph_hwm, hm["par_hwm"], hm["nph_hwm"] = saved[:4]
            del self.counters[saved[4]:]
            self._hwm_dev_stale = saved[5]
            self.nm_stats.update(nm_saved)
            self._nm_ring_abort()
            raise
        del wl, wl_off, gh, work
        if nm_on:
            self._lr_dev_stale = True
        else:
            self.local_round += act.to(torch.int64)
        hm["local_round"] += h_act
        self._after_round()

    def _sync_hwm_dev(self):
        if getattr(self, "_hwm_dev_stale", False):
            for name in ("par_hwm", "nph_hwm"):
                hv = torch.from_numpy(self.h[name].copy())
                setattr(self, name, hv.pin_memory().to(self.dev, non_blocking=True) if self.dev.type == "cuda" else hv.to(self.dev))
            self._hwm_dev_stale = False
        self._sync_lr_dev()

    def _check_mirrors(self):
        self._sync_hwm_dev()
        h = self.h
        for name in ("local_round", "active", "wide", "maxd", "subsample", "cs_tree", "cs_level", "n_cohort", "rt_off",
                     "par_hwm", "bits", "nph_hwm", "seed"):
            dv = getattr(self, name).cpu().numpy()
            assert dv.shape == h[name].shape and (dv == h[name]).all(), f"host mirror {name} diverged"

    def _h2d(self, a):
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))
        if self.dev.type == "cuda":
            return t.pin_memory().to(self.dev, non_blocking=True)
        return t

    def _h2d_mixed(self, pieces):
        words, lay, off = [], [], 0
        for a, kind in pieces:
            if kind == "i32":
                v = np.ascontiguousarray(a, dtype=np.int32)
                n = len(v)
                if n % 2:
                    v = np.concatenate([v, np.zeros(1, dtype=np.int32)])
                w = v.view(np.int64)
            else:
                w = np.ascontiguousarray(a, dtype=np.int64)
                n = len(w)
            lay.append((off, n, kind))
            words.append(w)
            off += len(w)
        T = self._h2d(np.concatenate(words))
        return [T[o:o + n] if kind == "i64" else T[o:o + (n + 1) // 2].view(torch.int32)[:n] for o, n, kind in lay]

    def _plan_hist_workspace(self, chunk, grp, buffered, wide, ngrp, ncand_p, ncand_w,
                             act_slots, level, has_parent, fixed_bytes, nrows=None):
        gp = torch.bincount(grp[buffered & ~wide], minlength=ngrp).cpu().numpy()
        gw = torch.bincount(grp[buffered & wide], minlength=ngrp).cpu().numpy()
        gr = (torch.bincount(grp[buffered], weights=nrows[buffered].to(torch.float64), minlength=ngrp).cpu().numpy()
              if nrows is not None else np.zeros(ngrp))
        if self.use_frontier:
            request, rows = self._plan_windows(ngrp, ncand_p, ncand_w, lambda: (gp, gw, gr))
        else:
            cell = self.d * self.B
            dense = len(act_slots) * (1 << level)
            p = dense if bool((~self.wide[act_slots]).any()) else 0
            w = dense if bool(self.wide[act_slots].any()) else 0
            _, rows = self._plan_windows(ngrp, ncand_p, ncand_w, lambda: (gp, gw, gr))
            request = lambda k: dict(NPH_p=ncand_p * cell, NPH_w=2 * ncand_w * cell, BUF_p=p * cell, BUF_w=2 * w * cell)
        return self._plan_core(chunk, ngrp, level, has_parent, fixed_bytes, request, rows)

    def _plan_hist_workspace_h(self, chunk, grp, buffered, wide, ngrp, ncand_p, ncand_w, level, has_parent,
                               fixed_bytes, nrows):
        cache = {}

        def counts():
            if not cache:
                cache["c"] = (np.bincount(grp[buffered & ~wide], minlength=ngrp),
                              np.bincount(grp[buffered & wide], minlength=ngrp),
                              np.bincount(grp[buffered], weights=nrows[buffered], minlength=ngrp))
            return cache["c"]
        tw = int((buffered & wide).sum())
        totals = (int(buffered.sum()) - tw, tw, int(nrows[buffered].sum()))
        request, rows = self._plan_windows(ngrp, ncand_p, ncand_w, counts, totals)
        return self._plan_core(chunk, ngrp, level, has_parent, fixed_bytes, request, rows)

    def _plan_windows(self, ngrp, ncand_p, ncand_w, counts, totals=None):
        cell = self.d * self.B
        dense_rows = getattr(self.planner, "dense_rows", False)

        def wmax(a, k):
            return int(np.add.reduceat(a, np.arange(0, ngrp, k)).max()) if ngrp else 0

        def win(i, k):
            if k >= ngrp and totals is not None:
                return totals[i]
            return wmax(counts()[i], k)

        def request(k):
            return dict(NPH_p=ncand_p * cell, NPH_w=2 * ncand_w * cell,
                        BUF_p=win(0, k) * cell, BUF_w=2 * win(1, k) * cell)

        def rows(k):
            return self._W_level if dense_rows else win(2, k)
        return request, rows

    def _plan_core(self, chunk, ngrp, level, has_parent, fixed_bytes, request, rows):
        avail = self.planner.budget - self.acc.shared - self.acc.persistent - self.planner.reserve
        touch = self.planner.row_temp_bytes

        def need(k, req=None):
            return self.ws.projected_bytes(request(k) if req is None else req) + touch * rows(k)

        chunk = max(1, min(int(chunk), ngrp))
        cap = avail - fixed_bytes
        self._nph_sym = None
        if self.ws.pooled:
            r = request(chunk)
            sym = {k: max(r[k], self.ws.buf[p].numel() if p in self.ws.buf else 0)
                   for k, p in (("NPH_p", "PH_p"), ("NPH_w", "PH_w"))}
            ns = need(chunk, dict(r, **sym))
            if ns <= cap:
                self._nph_sym = sym
                self._level_bytes = fixed_bytes + touch * rows(chunk)
                self._level_slack = cap - ns
                return chunk
        if need(chunk) > cap:
            for name in ("NPH_p", "NPH_w", "BUF_p", "BUF_w"):
                self.ws.release(name)
            if not has_parent:
                self.ws.release("PH_p"); self.ws.release("PH_w")
        while need(chunk) > cap and chunk > 1:
            chunk = max(1, chunk // 2)
        needed = need(chunk)
        if needed > cap:
            raise LevelInfeasible(f"level {level}: histogram workspace and chunk row temporaries need {needed} B but "
                                  f"only {max(cap, 0)} B remain after shared/persistent state and round row state "
                                  f"({fixed_bytes} B); reduce the resident batch", needed + fixed_bytes, avail)
        self._level_bytes = fixed_bytes + touch * rows(chunk)
        self._level_slack = cap - needed
        return chunk

    def _split_batches(self, split_bytes, n_cand_p, n_cand_w):
        pl = self.planner
        full = max(n_cand_p * pl.split_cand_bytes(False), n_cand_w * pl.split_cand_bytes(True))
        eff = min(full, split_bytes + max(0, int(getattr(self, "_level_slack", 0))))
        nb = {w: max(pl.split_batch(w), eff // pl.split_cand_bytes(w)) for w in (False, True)}
        return nb, max(eff, split_bytes)

    def _route_rows(self, work, groups, levels, lk, lvs):
        self._route_err = []
        h_act = self.h["active"]
        samp_h = (self.h["subsample"] < 1.0) & h_act
        n_r = None
        for s0, s1, a, b in groups:
            if not h_act[s0:s1].any():
                continue
            whole = s0 == 0 and s1 == self.S
            sl = self.slot_of_rt if whole else self.slot_of_rt[a:b] - s0
            act_rt = (self.active if whole else self.active[s0:s1]).index_select(0, sl)
            mask = act_rt & ~(work if whole else work[a:b])
            if HAS_NONZERO_STATIC and not samp_h[s0:s1].any():
                if n_r is None:
                    n_r = np.where(h_act, np.diff(self.h["rt_off"]) - self.wl_ub_slot, 0)
                q = torch.nonzero_static(mask, size=int(n_r[s0:s1].sum())).squeeze(1)
                if DEBUG_CHECKS:
                    assert torch.equal(q, torch.nonzero(mask).squeeze(1)), "host routed-row count"
            else:
                q = torch.nonzero(mask).squeeze(1)
            del act_rt, sl, mask
            if a:
                q = q + a
            kw = dict(err_out=self._route_err) if getattr(self.KR, "ROUTE_ERR_DEFER", False) else {}
            self.KR.route_update(self.X_all, self.xrow, self.pred, q, self.slot_of_rt[q].to(torch.int64), levels, lk, lvs,
                                 NODE_SHIFT, **kw)
            del q

    def _grow_trees_fast(self, wl, wl_off, gh, sg, sh, fm_tree, work, groups=None, gmax=0, wl_off_h=None):
        dev, S, K, d, B = self.dev, self.S, self.K, self.d, self.B
        KR = self.KR
        h = self.h
        pl = self.planner
        RT, W = self.RT, len(wl)
        if groups is None:
            groups, gmax = self._round_groups(W)
        nphw = h["nph_hwm"]
        act_np = np.nonzero(h["active"])[0].astype(np.int64)
        ns_kw, root = {}, None
        if wl_off_h is not None:
            rb, re_ = wl_off_h[act_np], wl_off_h[act_np + 1]
            pieces = [(act_np, "i64"), (rb, "i32"), (re_, "i32")]
            if hasattr(KR, "segment_grid"):
                cum_h, n_ch = KR.segment_grid(re_ - rb)
                pieces.append((cum_h, "i64"))
            tt = self._h2d_mixed(pieces)
            act_slots, root = tt[0], (tt[1], tt[2])
            if len(tt) > 3:
                ns_kw = dict(cum=tt[3], n_chunks=n_ch)
        else:
            act_slots = self._h2d(act_np)
        Dmax = int(h["maxd"][act_np].max())
        wide_h, maxd_h = h["wide"], h["maxd"]
        need_cs_level = bool((h["cs_level"][act_np] < 1.0).any())
        has_sizes = hasattr(KR, "build_hist_sizes")
        has_grid = hasattr(KR, "build_hist_grid")
        bh_kw = {}
        if getattr(KR, "BUILD_HIST_NARROW", False):
            bh_kw = dict(narrow=bool((h["bits"][act_np] == 16).all()))
        fr_slot = act_slots
        fr_node = torch.zeros(len(act_np), dtype=torch.int64, device=dev)
        if root is not None:
            fr_beg, fr_end = root
        else:
            fr_beg = wl_off[fr_slot].to(torch.int32)
            fr_end = wl_off[fr_slot + 1].to(torch.int32)
        with self.tm("hist_build"):
            fr_G, fr_H = KR.node_sums(gh, wl, fr_beg, fr_end, **ns_kw)
        fr_small = torch.ones(len(act_np), dtype=torch.bool, device=dev)
        hf_slot = act_np
        hf_par = None
        PH = {"packed": None, "wide": None}
        hph_idx = None
        n_prev_par = n_prev_w = 0
        splits = []
        leaf_ev = []
        hwm = h["par_hwm"]
        n_cohort_act = int(h["n_cohort"][act_np].sum())
        SP4 = torch.stack([(1.0 / sg).to(torch.float32), (1.0 / sh).to(torch.float32),
                           self.lam.to(torch.float32), self.mcw.to(torch.float32)])
        for l in range(Dmax):
            F = len(hf_slot)
            meta = torch.stack([fr_end - fr_beg, (fr_H > 0).to(torch.int32), fr_small.to(torch.int32)]).cpu().numpy()
            nrows = meta[0].astype(np.int64)
            small = meta[2] != 0
            fw = wide_h[hf_slot]
            cand = (l < maxd_h[hf_slot]) & (nrows >= 2) & (meta[1] != 0)
            ncand = int(cand.sum())
            frt = (fr_slot, fr_node, fr_beg, fr_end, fr_G, fr_H)
            if ncand < F:
                leaf_ev.append((frt, np.nonzero(~cand)[0]))
            if ncand == 0:
                break
            if l > 0:
                grp, ngrp = hf_par, n_prev_par
                sib = np.arange(F) ^ 1
                e = cand & ~small & ~cand[sib]
                extra = e[sib]
            else:
                grp, ngrp = np.arange(F), F
                sib = None
                extra = np.zeros(F, dtype=bool)
            coe = cand | extra
            rows_touch = int(nrows[coe].sum())
            n_cand_p = int((cand & ~fw).sum())
            n_cand_w = ncand - n_cand_p
            if self.chunk_parents is not None:
                chunk = max(1, int(self.chunk_parents))
            else:
                chunk = self.planner.chunk_parents(self.acc, ngrp, n_prev_w > 0, n_cand_w > 0, n_cand_p > 0, rows_touch)
            BUF = Gp = Hp = None
            keep = cand & ((l + 1) < maxd_h[hf_slot])
            n_keep_w = int((keep & fw).sum())
            n_keep_p = int(keep.sum()) - n_keep_w
            split_bytes = pl.split_bytes(n_cand_p, n_cand_w)
            self._W_level = W
            chunk = self._plan_hist_workspace_h(chunk, grp, coe, fw, ngrp, n_keep_p, n_keep_w, l, l > 0,
                                                pl.round_state_bytes(RT, W) + split_bytes, nrows)
            row_bytes = self._level_bytes - split_bytes
            split_nb, split_bytes = self._split_batches(split_bytes, n_cand_p, n_cand_w)
            if self._nph_sym is not None:
                for name, n in self._nph_sym.items():
                    self.ws.reserve(name, n, torch.int64)
            NPH = {"packed": self.ws.get("NPH_p", (n_keep_p, d, B), torch.int64),
                   "wide": self.ws.get("NPH_w", (2, n_keep_w, d, B), torch.int64)}
            if n_keep_p + n_keep_w:
                nphw = np.maximum(nphw, np.bincount(hf_slot[keep], minlength=S))
            n_new_par = n_new_p = n_new_w = n_split_w = 0
            new_idx, hk_slot = [], []
            lv_slot, lv_node, lv_feat, lv_thr = [], [], [], []
            kids = []
            level_rows_hist = level_rows_part = level_cells_buf = level_cells_sub = peak_hist_bytes = 0
            fm_level = fm_tree if not need_cs_level else self._colsample_mask(l + 1, fm_tree, self.cs_level)
            for g0 in range(0, ngrp, chunk):
                g1 = min(ngrp, g0 + chunk)
                if chunk >= ngrp:
                    sel, bld = cand, coe
                else:
                    inc = (grp >= g0) & (grp < g1)
                    sel, bld = cand & inc, coe & inc
                if not sel.any():
                    continue
                pos = np.full(F, -1, dtype=np.int64)
                nmode = {}
                for wflag in (False, True):
                    b_ = np.nonzero(bld & (fw == wflag))[0]
                    pos[b_] = np.arange(len(b_))
                    nmode[wflag] = len(b_)
                n_p, n_w = nmode[False], nmode[True]
                BUF = {"packed": self.ws.get("BUF_p", (n_p, d, B), torch.int64, zero=True),
                       "wide": self.ws.get("BUF_w", (2, n_w, d, B), torch.int64, zero=True)}
                level_cells_buf += (n_p + 2 * n_w) * d * B
                direct = bld & (small | extra)
                big = sel & ~small if l > 0 else np.zeros(F, dtype=bool)
                pieces, plan = [], {}
                off = 0
                for wflag in (False, True):
                    mm = fw == wflag
                    di = np.nonzero(direct & mm)[0]
                    bm = np.nonzero(big & mm)[0]
                    ci = np.nonzero(sel & mm)[0]
                    ident = len(ci) == nmode[wflag] and bool((pos[ci] == np.arange(len(ci))).all())
                    arrs = [di, pos[di], ci] + ([pos[bm], hph_idx[grp[bm]], pos[sib[bm]]] if len(bm) else []) \
                        + ([] if ident else [pos[ci]])
                    pci = None if ident else len(arrs) - 1
                    grid = None
                    if has_grid and len(di):
                        gsz, g_tiny, g_ch = KR.build_hist_grid(self.X_all, nrows[di])
                        grid = (gsz, len(arrs))
                        arrs = arrs + [g_tiny, g_ch]
                    sl = []
                    for a in arrs:
                        sl.append((off, off + len(a)))
                        off += len(a)
                    pieces += arrs
                    plan[wflag] = (di, bm, ci, ident, sl, pci, grid)
                IX = self._h2d(np.concatenate(pieces))
                cut = lambda se: IX[se[0]:se[1]]
                for mode, wflag in (("packed", False), ("wide", True)):
                    di, bm, ci, ident, sl, pci, grid = plan[wflag]
                    if len(di) == 0:
                        continue
                    t_di = cut(sl[0])
                    if grid is not None:
                        kw = dict(bh_kw, sizes=grid[0], cums=(cut(sl[grid[1]]), cut(sl[grid[1] + 1])))
                    else:
                        kw = dict(bh_kw, sizes=KR.build_hist_sizes(self.X_all, nrows[di])) if has_sizes else dict(bh_kw)
                    with self.tm("hist_build"):
                        KR.build_hist(self.X_all, self.xrow, gh, wl, fr_beg[t_di], fr_end[t_di], cut(sl[1]).to(torch.int32),
                                      fm_tree[fr_slot[t_di]], BUF[mode], wflag, **kw)
                    level_rows_hist += int(nrows[di].sum())
                if l > 0:
                    for mode, wflag in (("packed", False), ("wide", True)):
                        di, bm, ci, ident, sl, pci, grid = plan[wflag]
                        if len(bm) == 0:
                            continue
                        nb = split_nb[wflag]
                        t_o, t_p, t_s = cut(sl[3]), cut(sl[4]), cut(sl[5])
                        with self.tm("subtract"):
                            for c0 in range(0, len(bm), nb):
                                c1 = min(len(bm), c0 + nb)
                                if c0 == 0 and c1 == len(bm):
                                    KR.subtract_into(BUF[mode], t_o, PH[mode], t_p, BUF[mode], t_s, wflag)
                                else:
                                    KR.subtract_into(BUF[mode], t_o[c0:c1], PH[mode], t_p[c0:c1], BUF[mode], t_s[c0:c1], wflag)
                        level_cells_sub += len(bm) * d * B * (2 if wflag else 1)
                outs, rank, roff = [], np.full(F, -1, dtype=np.int64), 0
                for mode, wflag in (("packed", False), ("wide", True)):
                    di, bm, ci, ident, sl, pci, grid = plan[wflag]
                    if len(ci) == 0:
                        continue
                    t_ci = cut(sl[2])
                    nb, n_ci = split_nb[wflag], len(ci)
                    with self.tm("split_search"):
                        s_ = fr_slot[t_ci]
                        P4 = SP4[:, s_]
                        fl_ = fm_level[s_]
                        p_all = None if ident else cut(sl[pci])
                        for c0 in range(0, n_ci, nb):
                            c1 = min(n_ci, c0 + nb)
                            full = c0 == 0 and c1 == n_ci
                            cs = slice(None) if full else slice(c0, c1)
                            if wflag:
                                Gv, Hv = KR.decode_hist(BUF[mode], True)
                                if ident:
                                    Gp, Hp = Gv[cs], Hv[cs]
                                else:
                                    pp = p_all[cs]
                                    Gp, Hp = Gv[pp], Hv[pp]
                                del Gv, Hv
                            else:
                                Gp, Hp = KR.decode_hist(BUF[mode][cs] if ident else BUF[mode][p_all[cs]], False)
                            if full:
                                outs.append(KR.split_search(Gp, Hp, fr_G[t_ci], fr_H[t_ci], P4[0], P4[1], P4[2], P4[3], fl_))
                            else:
                                tc = t_ci[cs]
                                outs.append(KR.split_search(Gp, Hp, fr_G[tc], fr_H[tc], P4[0, cs], P4[1, cs], P4[2, cs],
                                                            P4[3, cs], fl_[cs]))
                            Gp = Hp = None
                    rank[ci] = roff + np.arange(len(ci))
                    roff += len(ci)
                if len(outs) == 1:
                    bf, bb, bg, GLc, HLc = outs[0]
                else:
                    bf, bb, bg, GLc, HLc = (torch.cat([o[i] for o in outs]) for i in range(5))
                dsp = (bg > KR.KRTEPS).cpu().numpy()
                do_split = np.zeros(F, dtype=bool)
                do_split[rank >= 0] = dsp[rank[rank >= 0]]
                nl = sel & ~do_split
                if nl.any():
                    leaf_ev.append((frt, np.nonzero(nl)[0]))
                peak_hist_bytes = max(peak_hist_bytes, self.ws.bytes())
                si = np.nonzero(do_split)[0]
                nsi = len(si)
                if nsi == 0:
                    continue
                sw = fw[si]
                kp = keep[si]
                n_split_w += int(sw.sum())
                mode_idx = np.full(nsi, -1, dtype=np.int64)
                src = {}
                for wflag in (False, True):
                    m = (sw == wflag) & kp
                    k = int(m.sum())
                    if k == 0:
                        continue
                    prev = n_new_w if wflag else n_new_p
                    mode_idx[m] = np.arange(prev, prev + k)
                    src[wflag] = (prev, k, pos[si[m]])
                    if wflag:
                        n_new_w += k
                    else:
                        n_new_p += k
                new_idx.append(mode_idx)
                hk_slot.append(hf_slot[si])
                pieces2 = [si, rank[si]] + [src[w][2] for w in (False, True) if w in src]
                ln_si = nrows[si]
                pgrid = None
                if has_grid:
                    pgrid, p_ch, p_len = KR.partition_grid(ln_si)
                    pieces2 += [p_ch, p_len]
                IX2 = self._h2d(np.concatenate(pieces2))
                t_si, q = IX2[:nsi], IX2[nsi:2 * nsi]
                o2 = 2 * nsi
                for wflag in (False, True):
                    if wflag not in src:
                        continue
                    prev, k, _ = src[wflag]
                    t_src = IX2[o2:o2 + k]
                    o2 += k
                    if wflag:
                        for pl_ in (0, 1):
                            torch.index_select(BUF["wide"][pl_], 0, t_src, out=NPH["wide"][pl_, prev:prev + k])
                    else:
                        torch.index_select(BUF["packed"], 0, t_src, out=NPH["packed"][prev:prev + k])
                n_new_par += nsi
                s_si, n_si = fr_slot[t_si], fr_node[t_si]
                b_si, e_si = fr_beg[t_si], fr_end[t_si]
                f_si, t_thr = bf[q], bb[q]
                lv_slot.append(s_si); lv_node.append(n_si); lv_feat.append(f_si); lv_thr.append(t_thr)
                if pgrid is not None:
                    kw = {"sizes": pgrid, "cums": (IX2[o2:o2 + nsi], IX2[o2 + nsi:o2 + 2 * nsi])}
                else:
                    kw = {"sizes": KR.partition_sizes(ln_si)} if has_sizes else {}
                with self.tm("partition"):
                    mid = KR.partition(self.X_all, self.xrow, wl, b_si, e_si, f_si, t_thr, **kw)
                level_rows_part += int(ln_si.sum())
                GLs, HLs = GLc[q], HLc[q]
                GRs, HRs = fr_G[t_si] - GLs, fr_H[t_si] - HLs
                kids.append(dict(slot=s_si, node=n_si, beg=b_si, mid=mid, end=e_si,
                                 GL=GLs, HL=HLs, GR=GRs, HR=HRs, left_small=HLs <= HRs))
            self.counters.append(dict(step=self.step_index, level=l, n_active_slots=len(act_np), n_frontier=F, n_cand=ncand,
                                      n_split=n_new_par, rows_hist=level_rows_hist, rows_partition=level_rows_part,
                                      cells_buf=level_cells_buf, cells_sub=level_cells_sub, peak_hist_bytes=peak_hist_bytes,
                                      chunk=chunk, n_groups=ngrp, n_keep=n_keep_p + n_keep_w, row_bytes=row_bytes,
                                      split_bytes=split_bytes,
                                      v1_rows=n_cohort_act,
                                      v1_hist_bytes=int(len(act_np) * d * B * 8 * (1 if l == 0 else 2 * (1 << l)))))
            self.acc.set_transient(peak_hist_bytes + row_bytes + split_bytes)
            if lv_slot:
                one = len(lv_slot) == 1
                splits.append(tuple(x[0] if one else torch.cat(x) for x in (lv_slot, lv_node, lv_feat, lv_thr)))
                hwm = np.maximum(hwm, np.bincount(np.concatenate(hk_slot), minlength=S))
            if n_new_par == 0:
                break
            kk = kids[0] if len(kids) == 1 else {key: torch.cat([k[key] for k in kids]) for key in kids[0]}
            ilv = lambda a, b: torch.stack([a, b], 1).reshape(-1)
            fr_slot = ilv(kk["slot"], kk["slot"])
            fr_node = ilv(2 * kk["node"] + 1, 2 * kk["node"] + 2)
            fr_beg = ilv(kk["beg"], kk["mid"])
            fr_end = ilv(kk["mid"], kk["end"])
            fr_G = ilv(kk["GL"], kk["GR"])
            fr_H = ilv(kk["HL"], kk["HR"])
            fr_small = ilv(kk["left_small"], ~kk["left_small"])
            hf_slot = np.repeat(np.concatenate(hk_slot), 2)
            hf_par = np.repeat(np.arange(n_new_par), 2)
            n_prev_par, n_prev_w = n_new_par, n_split_w
            PH = NPH
            self.ws.swap("NPH_p", "PH_p")
            self.ws.swap("NPH_w", "PH_w")
            hph_idx = np.concatenate(new_idx)
        else:
            if len(hf_slot):
                leaf_ev.append(((fr_slot, fr_node, fr_beg, fr_end, fr_G, fr_H), None))
        h["par_hwm"] = np.asarray(hwm, dtype=np.int64)
        h["nph_hwm"] = np.asarray(nphw, dtype=np.int64)
        self._hwm_dev_stale = True
        if len(leaf_ev) == 1 and leaf_ev[0][1] is None:
            L_slot, L_node, L_beg, L_end, L_G, L_H = leaf_ev[0][0]
        else:
            idx, off = [], 0
            for frt, ix in leaf_ev:
                n = len(frt[0])
                idx.append(np.arange(off, off + n) if ix is None else ix + off)
                off += n
            I = self._h2d(np.concatenate(idx))
            if len(leaf_ev) == 1:
                L_slot, L_node, L_beg, L_end, L_G, L_H = (t[I] for t in leaf_ev[0][0])
            else:
                L_slot, L_node, L_beg, L_end, L_G, L_H = (torch.cat([e[0][i] for e in leaf_ev])[I] for i in range(6))
        groups, gmax = self._plan_leaf_row_groups(W)
        self.tm_leaf = self.tm("leaf_route"); self.tm_leaf.__enter__()
        lv = KR.leaf_values(L_G, L_H, sg[L_slot], sh[L_slot], self.eta[L_slot], self.lam[L_slot])
        KR.leaf_update(self.pred, wl, L_beg, L_end, lv)
        lkey = (L_slot << NODE_SHIFT) | L_node
        lo = torch.argsort(lkey)
        lkey_s, lv_s = lkey[lo], lv[lo]
        levels = []
        for (s_, n_, f_, t_) in splits:
            key = (s_ << NODE_SHIFT) | n_
            if DEBUG_CHECKS:
                assert bool((key[1:] > key[:-1]).all()), "routing split keys are not strictly ascending"
            levels.append((key, f_, t_))
        self._route_rows(work, groups, levels, lkey_s, lv_s)
        self.acc.set_transient(self.ws.bytes() + pl.row_phase_bytes(RT, W, gmax))
        self.tm_leaf.__exit__(None, None, None)
        self._round_trees = (splits, lkey_s, lv_s)

    def _grow_trees_nm(self, nm, W, rb0, cnt_h, gh, sg, sh, fm_tree, work, groups=None, gmax=0):
        dev, S, K, d, B = self.dev, self.S, self.K, self.d, self.B
        KR = self.KR
        h = self.h
        pl = self.planner
        RT = self.RT
        if groups is None:
            groups, gmax = self._round_groups(W)
        nphw = h["nph_hwm"]
        act_np = np.nonzero(h["active"])[0].astype(np.int64)
        self._nm_ring_begin()
        Dmax = int(h["maxd"][act_np].max())
        wide_h, maxd_h = h["wide"], h["maxd"]
        need_cs_level = bool((h["cs_level"][act_np] < 1.0).any())
        need_fl_gather = need_cs_level or bool((h["cs_tree"][act_np] < 1.0).any())
        bh_kw = {}
        if getattr(KR, "BUILD_HIST_NARROW", False):
            bh_kw = dict(narrow=bool((h["bits"][act_np] == 16).all()))
        meta_nm = self._nm_meta()
        rt_off_h = h["rt_off"]
        fmask_u8 = self._nm_fmask_u8(fm_tree)
        hf_slot = act_np
        hf_node = np.zeros(len(act_np), dtype=np.int64)
        hf_par = None
        PH = {"packed": None, "wide": None}
        hph_idx = None
        n_prev_par = n_prev_w = 0
        levels_h = []
        leaf_h = []
        hwm = h["par_hwm"]
        n_cohort_act = int(h["n_cohort"][act_np].sum())
        for l in range(Dmax):
            F = len(hf_slot)
            if l == 0:
                hf_G, hf_H, hf_cnt, sg_h, sh_h = self._nm_root_host(rb0, act_np, cnt_h)
                hf_small = np.ones(F, dtype=bool)
                SP4_h = self._nm_sp4_host(sg_h, sh_h, sg, sh)
            else:
                cl = self._nm_lcount_host(lcount_dev)
                hf_G, hf_H, hf_cnt, hf_small = self._nm_children_host(hk_G, hk_H, hk_cnt, hk_GL, hk_HL, cl)
            nrows = hf_cnt
            small = hf_small
            fw = wide_h[hf_slot]
            cand = (l < maxd_h[hf_slot]) & (nrows >= 2) & (hf_H > 0)
            ncand = int(cand.sum())
            if ncand < F:
                leaf_h.append((hf_slot[~cand], hf_node[~cand], hf_G[~cand], hf_H[~cand]))
            if ncand == 0:
                break
            if l > 0:
                grp, ngrp = hf_par, n_prev_par
                sib = np.arange(F) ^ 1
                e = cand & ~small & ~cand[sib]
                extra = e[sib]
            else:
                grp, ngrp = np.arange(F), F
                sib = None
                extra = np.zeros(F, dtype=bool)
            coe = cand | extra
            dl = coe & (small | extra)
            self._nm_mn_level = int(np.bincount(hf_slot[dl]).max()) if dl.any() else 0
            rows_touch = int(nrows[coe].sum())
            n_cand_p = int((cand & ~fw).sum())
            n_cand_w = ncand - n_cand_p
            if self.chunk_parents is not None:
                chunk = max(1, int(self.chunk_parents))
            else:
                chunk = self.planner.chunk_parents(self.acc, ngrp, n_prev_w > 0, n_cand_w > 0, n_cand_p > 0, rows_touch)
            BUF = Gp = Hp = None
            keep = cand & ((l + 1) < maxd_h[hf_slot])
            n_keep_w = int((keep & fw).sum())
            n_keep_p = int(keep.sum()) - n_keep_w
            split_bytes = pl.split_bytes(n_cand_p, n_cand_w)
            self._W_level = W
            chunk = self._plan_hist_workspace_h(chunk, grp, coe, fw, ngrp, n_keep_p, n_keep_w, l, l > 0,
                                                pl.round_state_bytes(RT, W) + split_bytes, nrows)
            row_bytes = self._level_bytes - split_bytes
            split_nb, split_bytes = self._split_batches(split_bytes, n_cand_p, n_cand_w)
            if self._nph_sym is not None:
                for name, n in self._nph_sym.items():
                    self.ws.reserve(name, n, torch.int64)
            NPH = {"packed": self.ws.get("NPH_p", (n_keep_p, d, B), torch.int64),
                   "wide": self.ws.get("NPH_w", (2, n_keep_w, d, B), torch.int64)}
            if n_keep_p + n_keep_w:
                nphw = np.maximum(nphw, np.bincount(hf_slot[keep], minlength=S))
            n_new_par = n_new_p = n_new_w = n_split_w = 0
            new_idx, hk_slot, hk_node = [], [], []
            hk_feat, hk_thr, hk_G, hk_H, hk_cnt, hk_GL, hk_HL = ([] for _ in range(7))
            upd = []
            level_rows_hist = level_rows_part = level_cells_buf = level_cells_sub = peak_hist_bytes = 0
            fm_level = fm_tree if not need_cs_level else (self._nm_colsample(l + 1, fm_tree) if self._fused_sampling_on() else
                                                          self._colsample_mask(l + 1, fm_tree, self.cs_level))
            for g0 in range(0, ngrp, chunk):
                g1 = min(ngrp, g0 + chunk)
                if chunk >= ngrp:
                    sel, bld = cand, coe
                else:
                    inc = (grp >= g0) & (grp < g1)
                    sel, bld = cand & inc, coe & inc
                if not sel.any():
                    continue
                pos = np.full(F, -1, dtype=np.int64)
                nmode = {}
                for wflag in (False, True):
                    b_ = np.nonzero(bld & (fw == wflag))[0]
                    pos[b_] = np.arange(len(b_))
                    nmode[wflag] = len(b_)
                n_p, n_w = nmode[False], nmode[True]
                BUF = {"packed": self.ws.get("BUF_p", (n_p, d, B), torch.int64, zero=True),
                       "wide": self.ws.get("BUF_w", (2, n_w, d, B), torch.int64, zero=True)}
                level_cells_buf += (n_p + 2 * n_w) * d * B
                direct = bld & (small | extra)
                big = sel & ~small if l > 0 else np.zeros(F, dtype=bool)
                pieces, plan = [], {}
                for wflag in (False, True):
                    mm = fw == wflag
                    di = np.nonzero(direct & mm)[0]
                    bm = np.nonzero(big & mm)[0]
                    ci = np.nonzero(sel & mm)[0]
                    ident = len(ci) == nmode[wflag] and bool((pos[ci] == np.arange(len(ci))).all())
                    tab = []
                    if len(bm):
                        tab += [("sub_o", pos[bm], "i64"), ("sub_p", hph_idx[grp[bm]], "i64"), ("sub_s", pos[sib[bm]], "i64")]
                    if not ident:
                        tab.append(("pall", pos[ci], "i64"))
                    if len(ci):
                        s_ci = hf_slot[ci]
                        tab += [("Gt", hf_G[ci], "i64"), ("Ht", hf_H[ci], "i64"),
                                ("P4", np.concatenate([SP4_h[0][s_ci], SP4_h[1][s_ci], SP4_h[2][s_ci],
                                                       SP4_h[3][s_ci]]).view(np.int32), "i32")]
                        if need_fl_gather:
                            tab.append(("sl", hf_slot[ci], "i64"))
                    sl = {name: len(pieces) + i for i, (name, _, _) in enumerate(tab)}
                    pieces += [(a, kind) for _, a, kind in tab]
                    plan[wflag] = (di, bm, ci, ident, sl)
                IX = self._nm_h2d_mixed(pieces)
                cut = lambda k: IX[k]
                for mode, wflag in (("packed", False), ("wide", True)):
                    di, bm, ci, ident, sl = plan[wflag]
                    if len(di) == 0:
                        continue
                    with self.tm("hist_build"):
                        self._nm_build(nm, gh, BUF[mode], wflag, di, pos, hf_slot, hf_node, nrows, fm_tree, fmask_u8,
                                       meta_nm, bh_kw, rt_off_h)
                    level_rows_hist += int(nrows[di].sum())
                if l > 0:
                    for mode, wflag in (("packed", False), ("wide", True)):
                        di, bm, ci, ident, sl = plan[wflag]
                        if len(bm) == 0:
                            continue
                        nb = split_nb[wflag]
                        t_o, t_p, t_s = cut(sl["sub_o"]), cut(sl["sub_p"]), cut(sl["sub_s"])
                        with self.tm("subtract"):
                            for c0 in range(0, len(bm), nb):
                                c1 = min(len(bm), c0 + nb)
                                if c0 == 0 and c1 == len(bm):
                                    KR.subtract_into(BUF[mode], t_o, PH[mode], t_p, BUF[mode], t_s, wflag)
                                else:
                                    KR.subtract_into(BUF[mode], t_o[c0:c1], PH[mode], t_p[c0:c1], BUF[mode], t_s[c0:c1], wflag)
                        level_cells_sub += len(bm) * d * B * (2 if wflag else 1)
                C_chunk = sum(len(plan[w][2]) for w in (False, True))
                RB = self._nm_scratch(C_chunk)
                rank, roff = np.full(F, -1, dtype=np.int64), 0
                for mode, wflag in (("packed", False), ("wide", True)):
                    di, bm, ci, ident, sl = plan[wflag]
                    if len(ci) == 0:
                        continue
                    nb, n_ci = split_nb[wflag], len(ci)
                    with self.tm("split_search"):
                        Gt_, Ht_ = cut(sl["Gt"]), cut(sl["Ht"])
                        P4 = cut(sl["P4"]).view(torch.float32).view(4, n_ci)
                        fl_ = fm_level[cut(sl["sl"])] if need_fl_gather else self._nm_ones_bool(n_ci)
                        p_all = None if ident else cut(sl["pall"])
                        for c0 in range(0, n_ci, nb):
                            c1 = min(n_ci, c0 + nb)
                            full = c0 == 0 and c1 == n_ci
                            cs = slice(None) if full else slice(c0, c1)
                            if wflag:
                                Gv, Hv = KR.decode_hist(BUF[mode], True)
                                if ident:
                                    Gp, Hp = Gv[cs], Hv[cs]
                                else:
                                    pp = p_all[cs]
                                    Gp, Hp = Gv[pp], Hv[pp]
                                del Gv, Hv
                            else:
                                Gp, Hp = KR.decode_hist(BUF[mode][cs] if ident else BUF[mode][p_all[cs]], False)
                            if full:
                                self._nm_split(Gp, Hp, Gt_, Ht_, P4[0], P4[1], P4[2], P4[3], fl_, RB, C_chunk, roff)
                            else:
                                self._nm_split(Gp, Hp, Gt_[cs], Ht_[cs], P4[0, cs], P4[1, cs], P4[2, cs], P4[3, cs],
                                               fl_[cs], RB, C_chunk, roff + c0)
                            Gp = Hp = None
                    rank[ci] = roff + np.arange(len(ci))
                    roff += len(ci)
                dsp, GL_h, HL_h, bf_h, bb_h = self._nm_readback_split(RB, C_chunk)
                do_split = np.zeros(F, dtype=bool)
                do_split[rank >= 0] = dsp[rank[rank >= 0]]
                nl = sel & ~do_split
                if nl.any():
                    leaf_h.append((hf_slot[nl], hf_node[nl], hf_G[nl], hf_H[nl]))
                peak_hist_bytes = max(peak_hist_bytes, self.ws.bytes())
                si = np.nonzero(do_split)[0]
                nsi = len(si)
                if nsi == 0:
                    continue
                sw = fw[si]
                kp = keep[si]
                n_split_w += int(sw.sum())
                mode_idx = np.full(nsi, -1, dtype=np.int64)
                src = {}
                for wflag in (False, True):
                    m = (sw == wflag) & kp
                    k = int(m.sum())
                    if k == 0:
                        continue
                    prev = n_new_w if wflag else n_new_p
                    mode_idx[m] = np.arange(prev, prev + k)
                    src[wflag] = (prev, k, pos[si[m]])
                    if wflag:
                        n_new_w += k
                    else:
                        n_new_p += k
                new_idx.append(mode_idx)
                hk_slot.append(hf_slot[si])
                hk_node.append(hf_node[si])
                pieces2 = [src[w][2] for w in (False, True) if w in src]
                ln_si = nrows[si]
                IX2 = self._nm_h2d(np.concatenate(pieces2)) if pieces2 else None
                o2 = 0
                for wflag in (False, True):
                    if wflag not in src:
                        continue
                    prev, k, _ = src[wflag]
                    t_src = IX2[o2:o2 + k]
                    o2 += k
                    if wflag:
                        for pl_ in (0, 1):
                            torch.index_select(BUF["wide"][pl_], 0, t_src, out=NPH["wide"][pl_, prev:prev + k])
                    else:
                        torch.index_select(BUF["packed"], 0, t_src, out=NPH["packed"][prev:prev + k])
                n_new_par += nsi
                q = rank[si]
                hk_feat.append(bf_h[q]); hk_thr.append(bb_h[q])
                hk_G.append(hf_G[si]); hk_H.append(hf_H[si]); hk_cnt.append(nrows[si])
                hk_GL.append(GL_h[q]); hk_HL.append(HL_h[q])
                upd.append(si)
                level_rows_part += int(ln_si.sum())
            if upd:
                with self.tm("partition"):
                    si_all = upd[0] if len(upd) == 1 else np.concatenate(upd)
                    feat_h, thr_h = np.concatenate(hk_feat), np.concatenate(hk_thr)
                    lcount_dev = self._nm_update(nm, si_all, hf_slot, hf_node, feat_h, thr_h, rt_off_h)
                hk_G, hk_H, hk_cnt, hk_GL, hk_HL = (np.concatenate(x) for x in (hk_G, hk_H, hk_cnt, hk_GL, hk_HL))
            self.counters.append(dict(step=self.step_index, level=l, n_active_slots=len(act_np), n_frontier=F, n_cand=ncand,
                                      n_split=n_new_par, rows_hist=level_rows_hist, rows_partition=level_rows_part,
                                      cells_buf=level_cells_buf, cells_sub=level_cells_sub, peak_hist_bytes=peak_hist_bytes,
                                      chunk=chunk, n_groups=ngrp, n_keep=n_keep_p + n_keep_w, row_bytes=row_bytes,
                                      split_bytes=split_bytes,
                                      v1_rows=n_cohort_act,
                                      v1_hist_bytes=int(len(act_np) * d * B * 8 * (1 if l == 0 else 2 * (1 << l)))))
            self.acc.set_transient(peak_hist_bytes + row_bytes + split_bytes)
            if upd:
                levels_h.append((np.concatenate(hk_slot), np.concatenate(hk_node), feat_h, thr_h))
                hwm = np.maximum(hwm, np.bincount(np.concatenate(hk_slot), minlength=S))
            if n_new_par == 0:
                break
            hf_slot = np.repeat(np.concatenate(hk_slot), 2)
            hn = np.concatenate(hk_node)
            hf_node = np.stack([2 * hn + 1, 2 * hn + 2], 1).reshape(-1)
            hf_par = np.repeat(np.arange(n_new_par), 2)
            n_prev_par, n_prev_w = n_new_par, n_split_w
            PH = NPH
            self.ws.swap("NPH_p", "PH_p")
            self.ws.swap("NPH_w", "PH_w")
            hph_idx = np.concatenate(new_idx)
        else:
            if len(hf_slot):
                hf_G, hf_H = self._nm_children_host(hk_G, hk_H, hk_cnt, hk_GL, hk_HL, None)
                leaf_h.append((hf_slot, hf_node, hf_G, hf_H))
        h["par_hwm"] = np.asarray(hwm, dtype=np.int64)
        h["nph_hwm"] = np.asarray(nphw, dtype=np.int64)
        self._hwm_dev_stale = True
        groups, gmax = self._plan_leaf_row_groups(W)
        self.tm_leaf = self.tm("leaf_route"); self.tm_leaf.__enter__()
        self._round_pack = self._nm_tail(nm, work, groups, leaf_h, levels_h, sg, sh, rt_off_h)
        self._nm_ring_end()
        del nm
        self.acc.set_transient(self.ws.bytes() + pl.row_phase_bytes(RT, W, gmax))
        self.tm_leaf.__exit__(None, None, None)
        self._round_trees = None

    def _nodemap_ok(self):
        if not self.use_nodemap or self._nm_mod is None:
            return False
        a = self.h["active"]
        if not a.any():
            return False
        return bool((self.h["maxd"][a] <= NM_MAX_DEPTH).all() and (self.h["bits"][a] == 16).all())

    def _nm_meta(self):
        c = getattr(self, "_nm_meta_cache", None)
        if c is None or c[0] is not self.X_all:
            c = (self.X_all, self._nm_mod.Meta(self.X_all))
            self._nm_meta_cache = c
        return c[1]

    def _nm_charge(self, eligible=True):
        return 1 if (eligible and self.use_nodemap and self._nm_mod is not None and HOST_LOOP and self.use_worklist
                     and self.use_frontier) else 0

    def _nm_xbase(self):
        key = self._layout_version
        c = getattr(self, "_nm_xbase_cache", None)
        if c is not None and c[0] == key:
            return c[1]
        S, K = self.S, self.K
        xb = np.full(S, -1, dtype=np.int64)
        ro = self.h["rt_off"]
        cand, n_c, off_c = [], [], []
        for s in range(S):
            mid = self.mslots[s // K] if s // K < len(self.mslots) else None
            if mid is None:
                continue
            key_ = self.reg[mid].spec.data_key
            n = int(self.cohorts[key_].n)
            if n > 0 and int(ro[s + 1] - ro[s]) == n:
                cand.append(s)
                n_c.append(n)
                off_c.append(int(self.cohort_off[key_]))
        if cand:
            cand = np.asarray(cand, dtype=np.int64)
            k = len(cand)
            ends = self.crow[self._nm_h2d(np.concatenate([ro[cand], ro[cand + 1] - 1]))].cpu().numpy().astype(np.int64)
            ok = (ends[:k] == 0) & (ends[k:] == np.asarray(n_c, dtype=np.int64) - 1)
            xb[cand[ok]] = np.asarray(off_c, dtype=np.int64)[ok]
        self._nm_xbase_cache = (key, xb)
        return xb

    def _h2d_raw(self, a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        if self.dev.type == "cuda":
            return t.pin_memory().to(self.dev, non_blocking=True)
        return t.to(self.dev)

    def _nm_err_buf(self):
        e = getattr(self, "_nm_err", None)
        if e is None:
            e = self._nm_err = torch.zeros(1, dtype=torch.int32, device=self.dev)
            self._nm_err_host = torch.zeros(1, dtype=torch.int32).pin_memory()
            self._nm_err_evt = None
        return e

    def _nm_err_check(self, sync=False):
        e = self._nm_err_buf()
        n = 0
        if sync or NM_SYNC_CHECK:
            n = int(e.item())
            self._nm_err_evt = None
        else:
            if self._nm_err_evt is not None:
                self._nm_err_evt.synchronize()
                n = int(self._nm_err_host[0])
            self._nm_err_host.copy_(e, non_blocking=True)
            self._nm_err_evt = torch.cuda.Event()
            self._nm_err_evt.record()
        if n:
            raise RuntimeError(f"node-map path: {n} anomalies (work rows that reached no leaf, or compaction segments "
                               "whose row count differs from the host mirror)")

    def nm_check(self):
        if getattr(self, "_nm_err", None) is not None:
            self._nm_err_check(sync=True)

    def _nm_build(self, nm, gh, out, wide, di, pos, hf_slot, hf_node, nrows, fm_tree, fmask_u8, meta_nm, bh_kw, rt_off_h):
        NM = self._nm_mod
        dev = self.dev
        P = NM.Plan(rt_off_h, hf_slot[di], hf_node[di], pos[di])
        self.nm_stats["hist_calls"] += 1
        self.nm_stats["rows_scanned"] += P.rows
        mn = max(P.MN, getattr(self, "_nm_mn_level", 0))
        shape = NM.hist_shape(dev, meta_nm, P.MN, P.rows) if mn < NM_DEEP_MN else None
        if shape is None or (NM_DEEP_FG1 and ((shape["FG"] == 1 and shape["ng"] > 1) or shape["nnb"] > 1)):
            self.nm_stats["deep_calls"] += 1
            ln = np.asarray(nrows[di], dtype=np.int64)[P.order]
            dest = P.aux
            P.aux, P.aux2 = np.cumsum(ln) - ln, ln
            P.upload(self._nm_h2d, NM.light_rch(dev, P.rows))
            W = int(ln.sum())
            wl_c = NM.compact(nm, P, W, self._nm_err_buf())
            if NM_SYNC_CHECK:
                self._nm_err_check(sync=True)
            E = P.E
            seg = np.concatenate([P.aux, P.aux + ln, dest])
            kw = dict(bh_kw)
            if hasattr(self.KR, "build_hist_grid"):
                gsz, g_tiny, g_ch = self.KR.build_hist_grid(self.X_all, ln)
                tt = self._nm_h2d_mixed([(seg, "i32"), (g_tiny, "i64"), (g_ch, "i64")])
                i32 = tt[0]
                kw.update(sizes=gsz, cums=(tt[1], tt[2]))
            else:
                i32 = self._h2d_raw(seg.astype(np.int32))
                if hasattr(self.KR, "build_hist_sizes"):
                    kw["sizes"] = self.KR.build_hist_sizes(self.X_all, ln)
            self.KR.build_hist(self.X_all, self.xrow, gh, wl_c, i32[:E], i32[E:2 * E], i32[2 * E:],
                               fm_tree[P.ent_slot_dev()], out, wide, **kw)
            return
        P.upload(self._nm_h2d, shape["rch"], self._nm_xbase())
        NM.hist(self.X_all, self.xrow, gh, nm, out, wide, P, shape, fmask_u8, meta_nm)

    def _nm_update(self, nm, si, hf_slot, hf_node, feat_h, thr_h, rt_off_h):
        NM = self._nm_mod
        P = NM.Plan(rt_off_h, hf_slot[si], hf_node[si], np.arange(len(si)))
        sf, st = P.upload(self._nm_h2d_mixed, NM.light_rch(self.dev, P.rows), self._nm_xbase(),
                          extra=[(feat_h, "i32"), (thr_h, "i32")])
        cl = NM.update(self.X_all, self.xrow, nm, P, sf, st)
        self.nm_stats["update_calls"] += 1
        self.nm_stats["rows_scanned"] += P.rows
        return cl

    def _nm_root_host(self, rb0, act_np, cnt_h):
        if rb0 is None:
            rb0 = self._nm_scal().readback()
        G, H, cnt, sg_h, sh_h = rb0
        if cnt_h is not None:
            if DEBUG_CHECKS:
                assert np.array_equal(cnt.astype(np.int64), cnt_h), "mgnm_rowsum work counts differ from the host counts"
            cnt = cnt_h
        return G[act_np], H[act_np], cnt[act_np].astype(np.int64, copy=False), sg_h, sh_h

    def _nm_sp4_host(self, sg_h, sh_h, sg=None, sh=None):
        h = self.h
        if np.float32(np.float64(1.0) / np.float64(3e38)).view(np.int32) != 2378746:
            if self.nm_require:
                raise RuntimeError("MANGROVE_NODEMAP_REQUIRE=1 but the host flushes subnormals (FTZ/DAZ is on)")
            self.nm_stats["sp4_device"] = self.nm_stats.get("sp4_device", 0) + 1
            return torch.stack([(1.0 / sg).to(torch.float32), (1.0 / sh).to(torch.float32),
                                self.lam.to(torch.float32), self.mcw.to(torch.float32)]).cpu().numpy()
        return np.stack([(np.float64(1.0) / sg_h).astype(np.float32), (np.float64(1.0) / sh_h).astype(np.float32),
                         h["lam"].astype(np.float32), h["mcw"].astype(np.float32)])

    def _nm_lcount_host(self, lcount_dev):
        return lcount_dev.cpu().numpy().astype(np.int64)

    def _nm_children_host(self, Gp, Hp, cnt_p, GL, HL, cl):
        ilv = lambda a, b: np.stack([a, b], 1).reshape(-1)
        GR, HR = Gp - GL, Hp - HL
        G, H = ilv(GL, GR), ilv(HL, HR)
        if cl is None:
            return G, H
        ls = HL <= HR
        return G, H, ilv(cl, cnt_p - cl), ilv(ls, ~ls)

    def _nm_ones_bool(self, n):
        key = self._layout_version
        c = getattr(self, "_nm_ones_cache", None)
        if c is None or c[0] != key or c[1].shape[0] < n:
            self._nm_ones_cache = None
            rows = 1 << max(0, int(n) - 1).bit_length()
            c = (key, torch.ones((rows, self.d), dtype=torch.bool, device=self.dev))
            self._nm_ones_cache = c
        return c[1][:n]

    def _nm_scratch(self, C_chunk):
        key = self._layout_version
        need = 7 * max(int(C_chunk), 1)
        c = getattr(self, "_nm_scratch_cache", None)
        if c is None or c[0] != key or c[1].numel() < need:
            self._nm_scratch_cache = None
            n = 1 << (need - 1).bit_length()
            c = (key, torch.empty(n, dtype=torch.int32, device=self.dev))
            self._nm_scratch_cache = c
        return c[1]

    def _nm_split(self, Gp, Hp, Gt, Ht, isg, ish, lam, mcw, fl_, RB, C_chunk, pos0):
        self._nm_mod.split_search_nm(Gp, Hp, Gt, Ht, isg, ish, lam, mcw, fl_, RB, C_chunk, pos0)

    def _nm_readback_split(self, RB, C):
        pk = RB[:7 * C].cpu().numpy()
        GL = pk[:2 * C].view(np.int64)
        HL = pk[2 * C:4 * C].view(np.int64)
        bg = pk[4 * C:5 * C].view(np.float32)
        bf, bb = pk[5 * C:6 * C], pk[6 * C:7 * C]
        return bg > np.float32(self.KR.KRTEPS), GL, HL, bf, bb

    def _nm_tail(self, nm, work, groups, leaf_h, levels_h, sg, sh, rt_off_h):
        NM, KR = self._nm_mod, self.KR
        i64 = lambda xs: np.concatenate(xs).astype(np.int64, copy=False)
        Ls, Ln, LG, LH = (i64([e[i] for e in leaf_h]) for i in range(4))
        nlf = len(Ls)
        keys = (Ls << NODE_SHIFT) | Ln
        lo = np.argsort(keys, kind="stable")
        lkey_s = keys[lo]
        ks, fs, ts, lev_off = [], [], [], [0]
        for s_l, n_l, f_l, t_l in levels_h:
            key_l = (s_l << NODE_SHIFT) | n_l
            if DEBUG_CHECKS:
                assert bool((key_l[1:] > key_l[:-1]).all()), "routing split keys are not strictly ascending"
            ks.append(key_l); fs.append(f_l); ts.append(t_l); lev_off.append(lev_off[-1] + len(key_l))
        L = len(levels_h)
        z64, z32 = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
        keys_all = np.concatenate(ks) if L else z64
        feat_all = np.concatenate(fs).astype(np.int32, copy=False) if L else z32
        thr_all = np.concatenate(ts).astype(np.int32, copy=False) if L else z32
        s_all = i64([x[0] for x in levels_h]) if L else z64
        n_all = i64([x[1] for x in levels_h]) if L else z64
        lev_off = np.asarray(lev_off, dtype=np.int32)
        P = NM.Plan(rt_off_h, Ls, Ln, np.arange(nlf))
        LG_d, LH_d, Ls_d, lo_d, keys_d, lkey_s_d, feat_d, thr_d, lev_off_d = P.upload(
            self._nm_h2d_mixed, NM.light_rch(self.dev, P.rows),
            extra=[(LG, "i64"), (LH, "i64"), (Ls, "i64"), (lo, "i64"), (keys_all, "i64"), (lkey_s, "i64"),
                   (feat_all, "i32"), (thr_all, "i32"), (lev_off, "i32")])
        lv = KR.leaf_values(LG_d, LH_d, sg[Ls_d], sh[Ls_d], self.eta[Ls_d], self.lam[Ls_d])
        NM.leaf(self.pred, nm, P, lv, self._nm_err_buf())
        self._nm_err_check()
        self.nm_stats["rounds"] += 1
        self.nm_stats["rows_scanned"] += P.rows
        lv_s = lv[lo_d]
        self._nm_route_rows(work, groups, keys_d, feat_d, thr_d, lev_off_d, L, lkey_s_d, lv_s)
        return (lv_s, lkey_s, s_all, n_all, feat_all, thr_all, len(s_all), nlf)

    def _nm_route_rows(self, work, groups, keys, feat, thr, lev_off, L, lkey_s, lv_s):
        NM = self._nm_mod
        self._route_err = []
        h_act = self.h["active"]
        samp_h = (self.h["subsample"] < 1.0) & h_act
        n_r = n_r8 = None
        cnt8 = getattr(self, "_nm_cnt_round", None) if self._fused_sampling_on() else None
        for s0, s1, a, b in groups:
            if not h_act[s0:s1].any():
                continue
            whole = s0 == 0 and s1 == self.S
            sl = self.slot_of_rt if whole else self.slot_of_rt[a:b] - s0
            act_rt = (self.active if whole else self.active[s0:s1]).index_select(0, sl)
            mask = act_rt & ~(work if whole else work[a:b])
            if HAS_NONZERO_STATIC and not samp_h[s0:s1].any():
                if n_r is None:
                    n_r = np.where(h_act, np.diff(self.h["rt_off"]) - self.wl_ub_slot, 0)
                q = torch.nonzero_static(mask, size=int(n_r[s0:s1].sum())).squeeze(1)
                if DEBUG_CHECKS:
                    assert torch.equal(q, torch.nonzero(mask).squeeze(1)), "host routed-row count"
            elif HAS_NONZERO_STATIC and cnt8 is not None:
                if n_r8 is None:
                    n_r8 = np.where(h_act, np.diff(self.h["rt_off"]) - cnt8.astype(np.int64), 0)
                q = torch.nonzero_static(mask, size=int(n_r8[s0:s1].sum())).squeeze(1)
                if DEBUG_CHECKS:
                    assert torch.equal(q, torch.nonzero(mask).squeeze(1)), "host routed-row count (sampled)"
            else:
                q = torch.nonzero(mask).squeeze(1)
            del act_rt, sl, mask
            if a:
                q = q + a
            if len(q):
                err = torch.zeros(1, dtype=torch.int32, device=self.dev)
                NM.route_update_nm(self.X_all, self.xrow, self.pred, q, self.slot_of_rt[q].to(torch.int64), lev_off, keys,
                                   feat, thr, L, lkey_s, lv_s, NODE_SHIFT, err)
                self._route_err.append(err)
            del q

    def _nm_archive_unpack(self):
        lv_s, lkey_s, s_, n_, f_, t_, ns, nlf = self._round_pack
        self._round_pack = None
        errs = getattr(self, "_route_err", None) or []
        self._route_err = []
        parts = [lv_s.contiguous().view(torch.int32)] + [e.view(torch.int32) for e in errs]
        pack = (parts[0] if len(parts) == 1 else torch.cat(parts)).cpu().numpy()
        if errs:
            n_err = int(pack[nlf:].astype(np.int64).sum())
            assert n_err == 0, f"route_update: {n_err} routed rows did not land on a leaf"
        lvv = pack[:nlf].copy().view(np.float32)
        return s_, n_, f_, t_, lkey_s, lvv, ns, nlf

    def _sync_lr_dev(self):
        if getattr(self, "_lr_dev_stale", False):
            hv = torch.from_numpy(self.h["local_round"].copy())
            self.local_round = hv.pin_memory().to(self.dev, non_blocking=True) if self.dev.type == "cuda" else hv.to(self.dev)
            self._lr_dev_stale = False

    def _nm_scal(self):
        key = (self._layout_version, self.S)
        c = getattr(self, "_nm_scal_cache", None)
        if c is None or c[0] != key:
            self._nm_scal_cache = None
            c = (key, self._nm_mod.Scal(self.S, self.dev))
            self._nm_scal_cache = c
        return c[1]

    def _nm_quantize(self, g, h, sl, w, bits, s0, s1, has_hess, rt_off):
        return self._nm_mod.quantize_nm(self.KR, g, h, sl, w, bits, s1 - s0, has_hess, rt_off, self._nm_scal(), s0)

    def _nm_rowsum_plan(self):
        key = (self._layout_version, self.S)
        c = getattr(self, "_nm_rowsum_cache", None)
        if c is None or c[0] != key:
            self._nm_rowsum_cache = None
            NM = self._nm_mod
            P = NM.Plan(self.h["rt_off"], np.arange(self.S, dtype=np.int64))
            P.upload(self._h2d_raw, NM.light_rch(self.dev, max(P.rows, 1)))
            c = (key, P)
            self._nm_rowsum_cache = c
        return c[1]

    def _nm_prologue(self, gh, work, sg, sh, one, n_w, no_samp):
        NM = self._nm_mod
        scal = self._nm_scal()
        if not one:
            scal.sg.copy_(sg)
            scal.sh.copy_(sh)
            sg, sh = scal.sg, scal.sh
            self.nm_stats["multi_group_rounds"] += 1
        nm = torch.empty(self.RT, dtype=torch.uint8, device=self.dev)
        with self.tm("round_total"):
            with self.tm("hist_build"):
                P = self._nm_rowsum_plan()
                scal.zero_sums()
                NM.rowsum(gh, work, nm, P, scal)
                self.nm_stats["rows_scanned"] += P.rows
            if no_samp:
                cnt_h, W, rb0 = n_w, int(n_w.sum()), None
                self._nm_cnt_round = None
            else:
                rb0 = scal.readback()
                cnt_h, W = None, int(rb0[2].astype(np.int64).sum())
                self._nm_cnt_round = rb0[2]
        return nm, W, rb0, cnt_h, sg, sh

    def _nm_allf(self):
        key = (self._layout_version, self.S)
        c = getattr(self, "_nm_allf_cache", None)
        if c is None or c[0] != key:
            self._nm_allf_cache = None
            c = [key, torch.ones((self.S, self.d), dtype=torch.bool, device=self.dev), None]
            self._nm_allf_cache = c
        return c[1]

    def _nm_fmask_u8(self, fm_tree):
        c = getattr(self, "_nm_allf_cache", None)
        if c is not None and fm_tree is c[1]:
            if c[2] is None:
                c[2] = fm_tree.to(torch.uint8).contiguous()
            return c[2]
        t = getattr(self, "_nm_fm_tree", None)
        if t is not None and fm_tree is t[0]:
            return t[2]
        return fm_tree.to(torch.uint8).contiguous()

    def _fused_sampling_on(self):
        return FUSED_SAMPLING and self._nm_mod is not None and self.dev.type == "cuda"

    def _nm_rowmask_tab(self):
        h = self.h
        self.nm_stats["fused_rowmask"] += 1
        return self._h2d_raw(self._nm_mod.rowmask_tab(h["seed"], h["local_round"], h["subsample"], h["active"]))

    def _nm_rowmask(self, a, b, tab):
        w = torch.empty(b - a, dtype=torch.bool, device=self.dev)
        self._nm_mod.rowmask(self.is_train, self.crow, self.slot_of_rt, a, b - a, tab, self.S, w)
        return w

    def _nm_fm_host(self, fm):
        c = getattr(self, "_nm_allf_cache", None)
        if c is not None and fm is c[1]:
            return None, True
        t = getattr(self, "_nm_fm_tree", None)
        if t is not None and fm is t[0]:
            return t[1], True
        return None, False

    def _nm_colsample(self, level_key, fm_tree):
        S, d, h = self.S, self.d, self.h
        if level_key == 0:
            base, frac = None, h["cs_tree"]
        else:
            base, known = self._nm_fm_host(fm_tree)
            if not known:
                self._sync_lr_dev()
                return self._colsample_mask(level_key, fm_tree, self.cs_level)
            frac = h["cs_level"]
        m = self._nm_mod.colsample_host(h["seed"], h["local_round"], level_key, base, frac, d=d)
        n = S * d
        buf = np.zeros(-(-n // 8) * 8, dtype=np.uint8)
        buf[:n] = m.reshape(-1)
        words = buf.view(np.int64)
        dv = self._h2d_raw(words) if level_key == 0 else self._nm_h2d(words)
        u8 = dv.view(torch.uint8)[:n].view(S, d)
        fm = u8.view(torch.bool)
        self.nm_stats["fused_colmask"] += 1
        if level_key == 0:
            self._nm_fm_tree = (fm, m.astype(bool), u8)
        return fm

    NM_RING_BYTES = 8 << 20
    NM_RING_MAX_BYTES = 64 << 20

    def _nm_ring_begin(self):
        if self.dev.type != "cuda":
            return
        r = getattr(self, "_nm_ring", None)
        if r is None or r["dev"] != self.dev:
            r = self._nm_ring = dict(dev=self.dev, cap=0, h=None, h64=None, d=None, d64=None, off=0, evt=None,
                                     dirty=False, open=False, want=int(self.NM_RING_BYTES), views={})
        if r["dirty"]:
            torch.cuda.current_stream(self.dev).synchronize()
        elif r["evt"] is not None:
            r["evt"].synchronize()
        r["evt"], r["dirty"], r["off"] = None, False, 0
        if r["cap"] < r["want"]:
            r["h"] = r["h64"] = r["d"] = r["d64"] = None
            r["h"] = torch.empty(r["want"], dtype=torch.uint8, pin_memory=True)
            r["h64"] = r["h"].numpy().view(np.int64)
            r["d"] = torch.empty(r["want"], dtype=torch.uint8, device=self.dev)
            r["d64"] = r["d"].view(torch.int64)
            r["cap"] = r["want"]
            r["views"] = {}
        self.nm_stats["ring_bytes"] = r["cap"]
        if NM_RING_POISON:
            r["h64"][:] = np.int64(-6510615555426900571)
            r["d64"].fill_(-6510615555426900571)
        r["open"] = True

    def _nm_ring_end(self):
        r = getattr(self, "_nm_ring", None)
        if r is None:
            return
        r["evt"] = torch.cuda.Event()
        r["evt"].record()
        r["dirty"], r["open"] = False, False

    def _nm_ring_abort(self):
        r = getattr(self, "_nm_ring", None)
        if r is not None:
            r["open"] = False

    def _nm_h2d_mixed(self, pieces):
        r = getattr(self, "_nm_ring", None)
        if r is None or not r["open"]:
            return self._h2d_mixed(pieces)
        words, lay, off = [], [], 0
        for a, kind in pieces:
            if kind == "i32":
                v = np.ascontiguousarray(a, dtype=np.int32)
                n, w = len(v), (len(v) + 1) // 2
            else:
                v = np.ascontiguousarray(a, dtype=np.int64)
                n = w = len(v)
            lay.append((off, n, w, kind))
            words.append(v)
            off += w
        if off == 0:
            return self._h2d_mixed(pieces)
        o = r["off"]
        if (o + off) * 8 > r["cap"]:
            self.nm_stats["ring_overflow"] = self.nm_stats.get("ring_overflow", 0) + 1
            r["want"] = min(int(self.NM_RING_MAX_BYTES), max(r["want"], 2 * r["cap"], 2 * (o + off) * 8))
            return self._h2d_mixed(pieces)
        h64 = r["h64"]
        for (po, n, w, kind), v in zip(lay, words):
            if kind == "i32":
                seg = h64[o + po:o + po + w].view(np.int32)
                seg[:n] = v
                if n % 2:
                    seg[n] = 0
            else:
                h64[o + po:o + po + n] = v
        b0, b1 = o * 8, (o + off) * 8
        r["d"][b0:b1].copy_(r["h"][b0:b1], non_blocking=True)
        r["off"] = o + ((off + 15) // 16) * 16
        r["dirty"] = True
        d64, vc = r["d64"], r["views"]
        out = []
        for po, n, w, kind in lay:
            key = (o + po, n, kind)
            v = vc.get(key)
            if v is None:
                v = d64[o + po:o + po + n] if kind == "i64" else d64[o + po:o + po + w].view(torch.int32)[:n]
                if len(vc) >= 16384:
                    vc.clear()
                vc[key] = v
            out.append(v)
        return out

    def _nm_h2d(self, a):
        r = getattr(self, "_nm_ring", None)
        if r is None or not r["open"]:
            return self._h2d(a)
        return self._nm_h2d_mixed([(a, "i64")])[0]

    def _grow_trees(self, wl, wl_off, gh, sg, sh, fm_tree, work, groups=None, gmax=0):
        dev, S, K, d, B = self.dev, self.S, self.K, self.d, self.B
        pl = self.planner
        RT, W = self.RT, len(wl)
        if groups is None:
            groups, gmax = self._round_groups(W)
        self._sync_hwm_dev()
        act_slots = torch.nonzero(self.active).squeeze(1)
        Dmax = int(self.maxd[act_slots].max())
        i64 = lambda n: torch.zeros(n, dtype=torch.int64, device=dev)
        fr_slot = act_slots.clone()
        fr_node = i64(len(fr_slot))
        fr_beg = wl_off[fr_slot].to(torch.int32)
        fr_end = wl_off[fr_slot + 1].to(torch.int32)
        with self.tm("hist_build"):
            fr_G, fr_H = self.KR.node_sums(gh, wl, fr_beg, fr_end)
        fr_par = i64(len(fr_slot)) - 1
        fr_small = torch.ones(len(fr_slot), dtype=torch.bool, device=dev)
        PH = {"packed": None, "wide": None}
        ph_bytes = 0
        ph_mode_idx = None
        ph_wide = None
        splits = []
        leaves = []
        wide_rt = self.wide
        rows_hist_total = 0
        bh_kw = {}
        if getattr(self.KR, "BUILD_HIST_NARROW", False):
            bh_kw = dict(narrow=bool((self.bits[act_slots] == 16).all()))
        if not self.use_worklist:
            nid_dense = self._dense_init(wl, wl_off)
        for l in range(Dmax):
            depth_ok = l < self.maxd[fr_slot]
            nrows = (fr_end - fr_beg).to(torch.int64)
            cand = depth_ok & (nrows >= 2) & (fr_H > 0)
            notc = ~cand
            if bool(notc.any()):
                leaves.append((fr_slot[notc], fr_node[notc], fr_beg[notc], fr_end[notc], fr_G[notc], fr_H[notc]))
            ncand = int(cand.sum())
            if ncand == 0:
                break
            F = len(fr_slot)
            fr_wide = wide_rt[fr_slot]
            grp = fr_par if l > 0 else torch.arange(F, device=dev)
            ngrp = int(grp.max()) + 1
            if l > 0:
                sib = torch.arange(F, device=dev) ^ 1
                extra = cand & ~fr_small & ~cand[sib]
                extra = torch.zeros(F, dtype=torch.bool, device=dev).index_fill_(0, sib[extra], True)
            else:
                extra = torch.zeros(F, dtype=torch.bool, device=dev)
                sib = None
            cand_or_extra = cand | extra
            if self.chunk_parents is not None:
                chunk = max(1, int(self.chunk_parents))
            else:
                rows_touch = int(nrows[cand_or_extra].sum())
                chunk = self.planner.chunk_parents(self.acc, ngrp, bool(ph_wide.any()) if ph_wide is not None and len(ph_wide) else False,
                                                   bool(fr_wide[cand].any()), bool((~fr_wide[cand]).any()), rows_touch)
            keep = cand & ((l + 1) < self.maxd[fr_slot])
            n_cand_p, n_cand_w = int((cand & ~fr_wide).sum()), int((cand & fr_wide).sum())
            n_keep_p, n_keep_w = int((keep & ~fr_wide).sum()), int((keep & fr_wide).sum())
            BUF = Gp = Hp = None
            split_bytes = pl.split_bytes(n_cand_p, n_cand_w)
            self._W_level = W
            chunk = self._plan_hist_workspace(chunk, grp, cand_or_extra, fr_wide, ngrp, n_keep_p, n_keep_w,
                                              act_slots, l, l > 0, pl.round_state_bytes(RT, W) + split_bytes, nrows)
            row_bytes = self._level_bytes - split_bytes
            split_nb, split_bytes = self._split_batches(split_bytes, n_cand_p, n_cand_w)
            if self._nph_sym is not None:
                for name, n in self._nph_sym.items():
                    self.ws.reserve(name, n, torch.int64)
            NPH = {"packed": self.ws.get("NPH_p", (n_keep_p, d, B), torch.int64),
                   "wide": self.ws.get("NPH_w", (2, n_keep_w, d, B), torch.int64)}
            if n_keep_p + n_keep_w:
                self.nph_hwm = torch.maximum(self.nph_hwm, torch.bincount(fr_slot[keep], minlength=S))
            n_new_par, n_new_p, n_new_w = 0, 0, 0
            new_par_mode_idx, ph_new_wide = [], []
            lv_slot, lv_node, lv_feat, lv_thr = [], [], [], []
            kids = []
            level_rows_hist = level_rows_part = level_cells_buf = level_cells_sub = peak_hist_bytes = 0
            fm_level = fm_tree if not bool((self.cs_level[self.active] < 1.0).any()) else self._colsample_mask(l + 1, fm_tree, self.cs_level)
            any_p, any_w = bool((~wide_rt[act_slots]).any()), bool(wide_rt[act_slots].any())
            for g0 in range(0, ngrp, chunk):
                g1 = min(ngrp, g0 + chunk)
                in_chunk = (grp >= g0) & (grp < g1)
                sel = cand & in_chunk
                bld = cand_or_extra & in_chunk
                sel_idx = torch.nonzero(sel).squeeze(1)
                bld_idx = torch.nonzero(bld).squeeze(1)
                if len(sel_idx) == 0:
                    continue
                bw = fr_wide[bld_idx]
                pos_in_buf = i64(F) - 1
                if self.use_frontier:
                    pos_in_buf[bld_idx[~bw]] = torch.arange(int((~bw).sum()), device=dev)
                    pos_in_buf[bld_idx[bw]] = torch.arange(int(bw.sum()), device=dev)
                    n_p, n_w = int((~bw).sum()), int(bw.sum())
                else:
                    L = 1 << l
                    dense = self._dense_index(fr_slot, fr_node, act_slots, l)
                    pos_in_buf[bld_idx] = dense[bld_idx]
                    n_p = len(act_slots) * L if any_p else 0
                    n_w = len(act_slots) * L if any_w else 0
                BUF = {"packed": self.ws.get("BUF_p", (n_p, d, B), torch.int64, zero=True),
                       "wide": self.ws.get("BUF_w", (2, n_w, d, B), torch.int64, zero=True)}
                level_cells_buf += (n_p + 2 * n_w) * d * B
                direct = bld & (fr_small | extra)
                dir_idx = torch.nonzero(direct).squeeze(1)
                for mode, wflag in (("packed", False), ("wide", True)):
                    di = dir_idx[fr_wide[dir_idx] == wflag]
                    if len(di) == 0:
                        continue
                    if self.use_worklist:
                        with self.tm("hist_build"):
                            self.KR.build_hist(self.X_all, self.xrow, gh, wl, fr_beg[di], fr_end[di], pos_in_buf[di].to(torch.int32),
                                               fm_tree[fr_slot[di]], BUF[mode], wflag, **bh_kw)
                        level_rows_hist += int(nrows[di].sum())
                    else:
                        level_rows_hist += self._dense_build(nid_dense, wl, wl_off, gh, fr_slot[di], fr_node[di], pos_in_buf[di], fm_tree, BUF[mode], wflag)
                if l > 0:
                    big = sel & ~fr_small
                    bi = torch.nonzero(big).squeeze(1)
                    for mode, wflag in (("packed", False), ("wide", True)):
                        bm = bi[fr_wide[bi] == wflag]
                        if len(bm) == 0:
                            continue
                        nb = split_nb[wflag]
                        with self.tm("subtract"):
                            for c0 in range(0, len(bm), nb):
                                bb = bm[c0:c0 + nb]
                                self.KR.subtract_into(BUF[mode], pos_in_buf[bb], PH[mode], ph_mode_idx[fr_par[bb]], BUF[mode], pos_in_buf[sib[bb]], wflag)
                        level_cells_sub += len(bm) * d * B * (2 if wflag else 1)
                feat = torch.zeros(F, dtype=torch.int32, device=dev)
                thr = torch.zeros(F, dtype=torch.int32, device=dev)
                gain = torch.full((F,), self.KR.NEG_INF, dtype=torch.float32, device=dev)
                GL = i64(F)
                HL = i64(F)
                for mode, wflag in (("packed", False), ("wide", True)):
                    ci = sel_idx[fr_wide[sel_idx] == wflag]
                    if len(ci) == 0:
                        continue
                    nb = split_nb[wflag]
                    with self.tm("split_search"):
                        for c0 in range(0, len(ci), nb):
                            cc = ci[c0:c0 + nb]
                            p = pos_in_buf[cc]
                            if wflag:
                                Gv, Hv = self.KR.decode_hist(BUF[mode], True)
                                Gp, Hp = Gv[p], Hv[p]
                                del Gv, Hv
                            else:
                                Gp, Hp = self.KR.decode_hist(BUF[mode][p], False)
                            sl = fr_slot[cc]
                            isg = (1.0 / sg[sl]).to(torch.float32)
                            ish = (1.0 / sh[sl]).to(torch.float32)
                            bf, bb, bg, gl_, hl_ = self.KR.split_search(Gp, Hp, fr_G[cc], fr_H[cc], isg, ish,
                                                                        self.lam[sl].to(torch.float32), self.mcw[sl].to(torch.float32), fm_level[sl])
                            feat[cc], thr[cc], gain[cc], GL[cc], HL[cc] = bf, bb, bg, gl_, hl_
                            Gp = Hp = None
                do_split = sel & (gain > self.KR.KRTEPS)
                si = torch.nonzero(do_split).squeeze(1)
                nl = sel & ~do_split
                if bool(nl.any()):
                    leaves.append((fr_slot[nl], fr_node[nl], fr_beg[nl], fr_end[nl], fr_G[nl], fr_H[nl]))
                hb = BUF["packed"].numel() * 8 + BUF["wide"].numel() * 8
                peak_hist_bytes = max(peak_hist_bytes, self.ws.bytes())
                if len(si) == 0:
                    continue
                nsi = len(si)
                sw = fr_wide[si]
                kp = keep[si]
                mode_idx = torch.full((nsi,), -1, dtype=torch.int64, device=dev)
                for wflag in (False, True):
                    m = (sw == wflag) & kp
                    k = int(m.sum())
                    if k == 0:
                        continue
                    prev = n_new_w if wflag else n_new_p
                    mode_idx[m] = torch.arange(prev, prev + k, device=dev)
                    src = pos_in_buf[si[m]]
                    if wflag:
                        for pl_ in (0, 1):
                            torch.index_select(BUF["wide"][pl_], 0, src, out=NPH["wide"][pl_, prev:prev + k])
                        n_new_w += k
                    else:
                        torch.index_select(BUF["packed"], 0, src, out=NPH["packed"][prev:prev + k])
                        n_new_p += k
                ph_new_wide.append(sw)
                new_par_mode_idx.append(mode_idx)
                par_index = torch.arange(n_new_par, n_new_par + nsi, device=dev)
                n_new_par += nsi
                lv_slot.append(fr_slot[si]); lv_node.append(fr_node[si]); lv_feat.append(feat[si]); lv_thr.append(thr[si])
                if self.use_worklist:
                    with self.tm("partition"):
                        mid = self.KR.partition(self.X_all, self.xrow, wl, fr_beg[si], fr_end[si], feat[si], thr[si])
                else:
                    mid = self._dense_partition(nid_dense, wl, wl_off, fr_slot[si], fr_node[si], feat[si], thr[si], fr_beg[si], fr_end[si])
                level_rows_part += int(nrows[si].sum())
                GLs, HLs = GL[si], HL[si]
                GRs, HRs = fr_G[si] - GLs, fr_H[si] - HLs
                left_small = HLs <= HRs
                kids.append(dict(slot=fr_slot[si], node=fr_node[si], beg=fr_beg[si], mid=mid, end=fr_end[si],
                                 GL=GLs, HL=HLs, GR=GRs, HR=HRs, left_small=left_small, par=par_index))
            self.counters.append(dict(step=self.step_index, level=l, n_active_slots=len(act_slots), n_frontier=F, n_cand=ncand,
                                      n_split=n_new_par, rows_hist=level_rows_hist, rows_partition=level_rows_part,
                                      cells_buf=level_cells_buf, cells_sub=level_cells_sub, peak_hist_bytes=peak_hist_bytes,
                                      chunk=chunk, n_groups=ngrp, n_keep=n_keep_p + n_keep_w, row_bytes=row_bytes,
                                      split_bytes=split_bytes,
                                      v1_rows=int(self.n_cohort[act_slots].sum()) * (1 if l == 0 else 1),
                                      v1_hist_bytes=int(len(act_slots) * d * B * 8 * (1 if l == 0 else 2 * (1 << l)))))
            self.acc.set_transient(peak_hist_bytes + row_bytes + split_bytes)
            if lv_slot:
                splits.append((torch.cat(lv_slot), torch.cat(lv_node), torch.cat(lv_feat), torch.cat(lv_thr)))
                self.par_hwm = torch.maximum(self.par_hwm, torch.bincount(splits[-1][0], minlength=S))
            if n_new_par == 0:
                break
            cat = lambda key: torch.cat([k[key] for k in kids])
            slot_, node_, beg_, mid_, end_ = cat("slot"), cat("node"), cat("beg"), cat("mid"), cat("end")
            GL_, HL_, GR_, HR_, ls_, par_ = cat("GL"), cat("HL"), cat("GR"), cat("HR"), cat("left_small"), cat("par")
            o = torch.argsort(par_)
            slot_, node_, beg_, mid_, end_, GL_, HL_, GR_, HR_, ls_, par_ = [t[o] for t in (slot_, node_, beg_, mid_, end_, GL_, HL_, GR_, HR_, ls_, par_)]
            ilv = lambda a, b: torch.stack([a, b], 1).reshape(-1)
            fr_slot = ilv(slot_, slot_)
            fr_node = ilv(2 * node_ + 1, 2 * node_ + 2)
            fr_beg = ilv(beg_, mid_)
            fr_end = ilv(mid_, end_)
            fr_G = ilv(GL_, GR_)
            fr_H = ilv(HL_, HR_)
            fr_small = ilv(ls_, ~ls_)
            fr_par = ilv(par_, par_)
            PH = NPH
            ph_bytes = (n_new_p + 2 * n_new_w) * d * B * 8
            self.ws.swap("NPH_p", "PH_p")
            self.ws.swap("NPH_w", "PH_w")
            ph_mode_idx = torch.cat(new_par_mode_idx)
            ph_wide = torch.cat(ph_new_wide)
            if not self.use_worklist:
                self._dense_apply(nid_dense, wl, wl_off, fr_slot, fr_node, fr_beg, fr_end)
        else:
            if len(fr_slot):
                leaves.append((fr_slot, fr_node, fr_beg, fr_end, fr_G, fr_H))
        L_slot = torch.cat([x[0] for x in leaves]); L_node = torch.cat([x[1] for x in leaves])
        L_beg = torch.cat([x[2] for x in leaves]); L_end = torch.cat([x[3] for x in leaves])
        L_G = torch.cat([x[4] for x in leaves]); L_H = torch.cat([x[5] for x in leaves])
        groups, gmax = self._plan_leaf_row_groups(W)
        self.tm_leaf = self.tm("leaf_route"); self.tm_leaf.__enter__()
        lv = self.KR.leaf_values(L_G, L_H, sg[L_slot], sh[L_slot], self.eta[L_slot], self.lam[L_slot])
        self.KR.leaf_update(self.pred, wl, L_beg, L_end, lv)
        lkey = (L_slot << NODE_SHIFT) | L_node
        lo = torch.argsort(lkey)
        levels = []
        for (s_, n_, f_, t_) in splits:
            key = (s_ << NODE_SHIFT) | n_
            o = torch.argsort(key)
            levels.append((key[o], f_[o], t_[o]))
        lkey_s, lv_s = lkey[lo], lv[lo]
        self._route_rows(work, groups, levels, lkey_s, lv_s)
        self.acc.set_transient(self.ws.bytes() + pl.row_phase_bytes(RT, W, gmax))
        self.tm_leaf.__exit__(None, None, None)
        self._round_trees = (splits, lkey_s, lv_s)
        self.h["par_hwm"] = self.par_hwm.cpu().numpy().astype(np.int64)
        self.h["nph_hwm"] = self.nph_hwm.cpu().numpy().astype(np.int64)

    def _dense_init(self, wl, wl_off):
        return torch.zeros(len(wl), dtype=torch.int64, device=self.dev)

    def _dense_build(self, nid, wl, wl_off, gh, slots, nodes, dest, fm_tree, out, wide):
        key_all = (self.slot_of_rt[wl.to(torch.int64)].to(torch.int64) << NODE_SHIFT) | nid
        keys = (slots << NODE_SHIFT) | nodes
        o = torch.argsort(keys)
        ks = keys[o]
        j = torch.searchsorted(ks, key_all).clamp_(max=len(ks) - 1)
        hit = ks[j] == key_all
        ridx = torch.nonzero(hit).squeeze(1)
        dst = dest[o][j[ridx]]
        q = wl[ridx].to(torch.int64)
        xb = self.X_all[self.xrow[q].to(torch.int64)].to(torch.int64)
        d, B = self.d, self.B
        key = (dst[:, None] * d + torch.arange(d, device=self.dev)[None, :]) * B + xb
        m = fm_tree[self.slot_of_rt[q].to(torch.int64)]
        v = gh[q][:, None].expand(-1, d)[m]
        key = key[m]
        if wide:
            G, H = self.KR.split_gh(v)
            plane = out.shape[1] * d * B
            out.view(-1).index_add_(0, key, G)
            out.view(-1).index_add_(0, key + plane, H)
        else:
            out.view(-1).index_add_(0, key, v)
        return len(wl)

    def _dense_partition(self, nid, wl, wl_off, slots, nodes, feat, thr, beg, end):
        mid = self.KR.partition(self.X_all, self.xrow, wl, beg, end, feat, thr)
        return mid

    def _dense_apply(self, nid, wl, wl_off, fr_slot, fr_node, fr_beg, fr_end):
        seg_of, pos, _ = self.KR._expand_segments(fr_beg, fr_end)
        nid[pos] = fr_node[seg_of]

    def _dense_index(self, fr_slot, fr_node, act_slots, l):
        L = 1 << l
        rank = torch.zeros(self.S, dtype=torch.int64, device=self.dev)
        rank[act_slots] = torch.arange(len(act_slots), device=self.dev)
        return rank[fr_slot] * L + (fr_node - (L - 1))

    def _after_round(self, groups=None, gmax=0):
        dev, S, K = self.dev, self.S, self.K
        if getattr(self, "_round_pack", None) is not None:
            s_, n_, f_, t_, lk, lvv, ns, nlf = self._nm_archive_unpack()
        else:
            splits, lkey_s, lv_s = self._round_trees
            i64 = [x[0] for x in splits] + [x[1] for x in splits] + [lkey_s]
            i32 = [x[2] for x in splits] + [x[3] for x in splits]
            assert all(t.dtype == torch.int64 for t in i64) and all(t.dtype == torch.int32 for t in i32)
            parts = [t.contiguous().view(torch.int32) for t in i64] + i32 + [lv_s.contiguous().view(torch.int32)]
            errs = getattr(self, "_route_err", None) or []
            self._route_err = []
            parts += [e.view(torch.int32) for e in errs]
            pack = (parts[0] if len(parts) == 1 else torch.cat(parts)).cpu().numpy()
            ns, nlf = sum(len(x[0]) for x in splits), len(lkey_s)
            if errs:
                n_err = int(pack[len(pack) - len(errs):].astype(np.int64).sum())
                assert n_err == 0, f"route_update: {n_err} routed rows did not land on a leaf"
            s_, n_ = pack[:2 * ns].view(np.int64), pack[2 * ns:4 * ns].view(np.int64)
            lk = pack[4 * ns:4 * ns + 2 * nlf].view(np.int64)
            o = 4 * ns + 2 * nlf
            f_, t_ = pack[o:o + ns], pack[o + ns:o + 2 * ns]
            lvv = pack[o + 2 * ns:o + 2 * ns + nlf].copy().view(np.float32)
        so = np.argsort(s_ * (1 << NODE_SHIFT) + n_, kind="stable")
        s_, n_, f_, t_ = s_[so], n_[so], f_[so], t_[so]
        ls, ln = lk >> NODE_SHIFT, lk & ((1 << NODE_SHIFT) - 1)
        sc = np.bincount(s_, minlength=S); lc = np.bincount(ls, minlength=S)
        so_ = np.concatenate([[0], np.cumsum(sc)]); lo_ = np.concatenate([[0], np.cumsum(lc)])
        n32, f16, t32, ln32 = n_.astype(np.int32), f_.astype(np.int16), t_.astype(np.int32), ln.astype(np.int32)
        num, scale, count = np.zeros(S, dtype=np.int64), np.ones(S), np.zeros(S, dtype=np.int64)
        if self.collect_metrics:
            if groups is None:
                groups, gmax = self._round_groups(0)
            h_act = self.h["active"]
            parts = []
            with self.tm("metric"):
                for s0, s1, a, b in groups:
                    if not h_act[s0:s1].any():
                        continue
                    whole = s0 == 0 and s1 == S
                    sl = self.slot_of_rt if whole else self.slot_of_rt[a:b] - s0
                    rv = (lambda t: t) if whole else (lambda t: t[a:b])
                    has_eval = (~rv(self.is_train)) & (self.active if whole else self.active[s0:s1]).index_select(0, sl)
                    mrow_base, mrow_len = self._mrow(s0, s1, a, b)
                    r = self.KR.metric_fixed_point(self.loss, K, rv(self.pred), rv(self.y_rt), has_eval, sl, s1 - s0,
                                                   mrow_base, mrow_len)
                    parts.append((s0, s1, r))
                    del sl, has_eval, mrow_base, mrow_len
                if len(parts) == 1 and parts[0][0] == 0 and parts[0][1] == S:
                    num, scale, count = (x.cpu().numpy() for x in parts[0][2])
                else:
                    num, scale, count = np.zeros(S, dtype=np.int64), np.ones(S), np.zeros(S, dtype=np.int64)
                    for s0, s1, (n_g, sc_g, c_g) in parts:
                        num[s0:s1], scale[s0:s1], count[s0:s1] = n_g.cpu().numpy(), sc_g.cpu().numpy(), c_g.cpu().numpy()
            self.acc.set_transient(self.ws.bytes() + self.planner.chunk_row_bytes * gmax)
        self.tm.flush()
        finished = []
        arc = RoundArchive(n32, f16, t32, ln32, lvv, so_, lo_)
        for j, mid in enumerate(self.mslots):
            if mid is None:
                continue
            st = self.reg[mid]
            if isinstance(st.trees, LazyTrees):
                st.trees.append_lazy(arc, j * K, K)
            else:
                st.trees.append([arc.tree(j * K + k) for k in range(K)])
            st.local_round += 1
            s0 = j * K
            if count[s0] > 0:
                m = float(num[s0]) / float(scale[s0]) / float(count[s0])
                st.metric_history.append(m)
                st.metric_q_history.append(int(num[s0]))
                if st.best_metric is None or m < st.best_metric:
                    st.best_metric, st.best_round = m, st.local_round - 1
            esr = st.spec.early_stopping_rounds
            if st.local_round >= st.spec.n_rounds:
                finished.append((st, "n_rounds"))
            elif esr > 0 and count[s0] > 0 and st.local_round - 1 - st.best_round >= esr:
                finished.append((st, "early_stop"))
        self._finish_batch(finished)

    def trees(self, mid):
        return self.reg[mid].trees

    def predict(self, mid, Xb, n_rounds=None):
        return self.reg[mid].predict_margin(Xb, n_rounds)

    def margins(self, mid):
        st = self.reg[mid]
        assert st.slot is not None, "model is not resident"
        j, K = st.slot, self.K
        a, b = int(self.h["rt_off"][j * K]), int(self.h["rt_off"][j * K + 1])
        rows = self.crow[a:b].cpu().numpy()
        P = self.pred[a:int(self.h["rt_off"][j * K + K])].cpu().numpy().reshape(K, -1)
        out = [P[k] for k in range(K)]
        return rows, (out[0] if K == 1 else np.stack(out, 1))

    def eval_margin(self, mid):
        st = self.reg[mid]
        if st.eval_margin is not None:
            return st.eval_margin
        return self._ev_gather([(st, st.slot)])[0]

    def timing_report(self):
        return self.tm.report()

    def memory_report(self):
        r = self.acc.snapshot()
        r.update(workspace_bytes=self.ws.bytes(), workspace_grow_events=self.ws.grow_events, workspace_requests=self.ws.requests)
        r["paused_device_bytes"] = self._paused_dev_bytes
        r["accounting_scope"] = "managed histogram capacities plus persistent/shared state and estimated row reserve; not all torch temporaries"
        return r
