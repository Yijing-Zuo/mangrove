import os
import time
import multiprocessing as mp

import numpy as np

from .. import metrics as M
from .driver import Done
from .trainers import _Base, _pool_warmup_mode, _xgb_fastscore_mode

SYNC_ALGOS = ("grid", "sha", "hyperband", "hbpar")


def _env_int(name, default):
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    x = int(v)
    if x < 1:
        raise ValueError(f"{name}={v!r}: expected an integer >= 1")
    return x


def folds_per_worker(P, F, override=None):
    if override is not None:
        return max(1, min(F, int(override)))
    for k in range(1, F + 1):
        if (P * k // F) * F >= P:
            return k
    return F


def assign_parts(P, parts, k):
    parts = sorted(parts)
    F = len(parts)
    return [[parts[(w * k + i) % F] for i in range(k)] for w in range(P)]


def _tuned_worker(wid, spec_dict, data_dir, nthread, device, owned, task_q, res_q, torch_guard, fastscore, warmup,
                  cache):
    try:
        from bench import envinfo
        if device == "cuda" and torch_guard:
            envinfo.gpu_process_guard(required=False)
        from bench.runners.pool import _setup, xgb_params
        import xgboost as xgb
        t0 = time.perf_counter()
        spec, ds, jobs = _setup(spec_dict, data_dir)
        task, K = ds.task, ds.K
        owned = set(owned)
        parts = {}
        for j in jobs.values():
            if j.part in owned:
                parts.setdefault(j.part, j)
        t_data = time.perf_counter()
        dtr, dva, base = {}, {}, {}
        cp = None
        if device == "cuda" and os.environ.get("MANGROVE_BENCH_XGBT_DEVICE_DATA", "1").strip() != "0":
            try:
                import cupy as cp
            except ImportError:
                cp = None
        for part, j in parts.items():
            X = ds.X(j.cohort)
            base[part] = ds.base_margin(j.train_mask)
            if cp is not None:
                Xt = cp.asarray(np.ascontiguousarray(X[j.train_mask], dtype=np.float32))
                yt = cp.asarray(np.ascontiguousarray(ds.y[j.train_mask], dtype=np.float32))
                dtr[part] = xgb.QuantileDMatrix(Xt, label=yt, max_bin=256, nthread=nthread)
                Xv = cp.asarray(np.ascontiguousarray(X[j.val_mask], dtype=np.float32))
                dva[part] = xgb.DMatrix(Xv, nthread=nthread)
                del Xt, yt, Xv
            else:
                dtr[part] = xgb.QuantileDMatrix(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask],
                                                max_bin=256, nthread=nthread)
                dva[part] = xgb.DMatrix(X[j.val_mask].astype(np.float32), nthread=nthread)
        if cp is not None:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
        t1 = time.perf_counter()
        cuts_equal, host_fallback, cut_err = None, [], None
        if cp is not None and os.environ.get("MANGROVE_BENCH_XGBT_CHECK_CUTS", "0").strip() == "1":
            _h = lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
            cuts_equal = True
            for part, j in parts.items():
                X = ds.X(j.cohort)
                ref = xgb.QuantileDMatrix(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask], max_bin=256,
                                          nthread=nthread)
                try:
                    a, b = dtr[part].get_quantile_cut(), ref.get_quantile_cut()
                    same = all(np.array_equal(_h(x), _h(y)) for x, y in zip(a, b))
                except Exception as e:
                    same, cut_err = False, repr(e)[:300]
                if same:
                    del ref
                else:
                    cuts_equal = False
                    host_fallback.append(part)
                    dtr[part] = ref
        t_chk = time.perf_counter()

        def params(j):
            return xgb_params(j, task, K, nthread, device, base[j.part])

        for j0 in (list(parts.values()) if warmup == "all" else list(parts.values())[:1]):
            xgb.train(params(j0), dtr[j0.part], 2)
        scorer = None
        if fastscore is not None:
            from .xgb_fastscore import WorkerScorer
            scorer = WorkerScorer(ds, "cuda" if device == "cuda" else None, fastscore)
            if task == "binary" and scorer.auc_device is None:
                raise RuntimeError("MANGROVE_BENCH_XGB_FASTSCORE: torch sees no CUDA device in this GPU worker; "
                                   "refusing to score on the host")
            scorer.warmup()
        t2 = time.perf_counter()
        res_q.put({"ready": wid, "prep_s": t1 - t0, "data_s": t_data - t0, "build_s": t1 - t_data,
                   "cut_check_s": t_chk - t1, "warmup_s": t2 - t_chk, "parts": sorted(parts),
                   "device_data": cp is not None, "cuts_equal_host": cuts_equal, "host_fallback_parts": host_fallback,
                   "cut_check_error": cut_err, "fastscore": None if scorer is None else scorer.describe()})
        boosters = {}
        live_peak = 0
        while True:
            msg = task_q.get()
            if msg is None:
                break
            if msg[0] == "drop":
                boosters.pop(msg[1], None)
                continue
            _, sid, mid, start, end, t_disp = msg
            j = jobs[mid]
            ts = time.perf_counter()
            bst = boosters.get(mid)
            if bst is None:
                if start != 0:
                    raise RuntimeError(f"model {mid}: no resident booster for continuation from round {start}")
                bst = xgb.Booster(params(j), [dtr[j.part]])
                boosters[mid] = bst
                live_peak = max(live_peak, len(boosters))
            elif bst.num_boosted_rounds() != start:
                raise RuntimeError(f"model {mid}: booster has {bst.num_boosted_rounds()} rounds, segment starts at {start}")
            for i in range(start, end):
                bst.update(dtr[j.part], i)
            t_tr = time.perf_counter()
            mg = bst.predict(dva[j.part], output_margin=True)
            if task == "multiclass":
                mg = np.asarray(mg).reshape(-1, K)
            if scorer is None:
                mt, path = M.metrics(task, ds.y[j.val_mask], mg), None
            else:
                mt, path = scorer.metrics(j, mg)
            t_sc = time.perf_counter()
            if cache == "reset":
                bst.reset()
            t_rs = time.perf_counter()
            r = {"sid": sid, "wid": wid, "loss": M.loss_value(task, mt), "metrics": mt, "rounds": end - start,
                 "train_s": t_tr - ts, "score_s": t_sc - t_tr, "reset_s": t_rs - t_sc, "queue_s": ts - t_disp,
                 "live": len(boosters), "live_peak": live_peak}
            if path is not None:
                r["fs"] = path
            res_q.put(r)
    except Exception:
        import traceback
        res_q.put({"error": traceback.format_exc()[-4000:], "wid": wid})


class TunedXGBTrainer(_Base):
    supports_resume = True
    supports_pause = True

    def __init__(self, spec, ds, W, nthread=1, device="cuda", data_dir=None, torch_guard=True, timeout_s=6 * 3600,
                 cache=None, algo=None):
        super().__init__()
        from ..workload import build_jobs
        self.W, self.nthread, self.device, self.timeout_s = int(W), int(nthread), device, timeout_s
        if self.W < 1:
            raise ValueError("--P must be >= 1")
        self.cache = cache or os.environ.get("MANGROVE_BENCH_XGBT_CACHE", "").strip() or "reset"
        if self.cache not in ("reset", "keep"):
            raise ValueError(f"cache {self.cache!r}: expected reset or keep")
        self.depth = _env_int("MANGROVE_BENCH_XGBT_DEPTH", 2)
        self.qdepth = _env_int("MANGROVE_BENCH_XGBT_QDEPTH", 4)
        self.algo = algo or spec.dynamic.get("algo")
        self.cap = self.W * self.qdepth if self.algo in SYNC_ALGOS else self.W
        self.fastscore = _xgb_fastscore_mode() if device == "cuda" else None
        self.fastscore_counts = {"rank": 0, "host": 0}
        self.warmup_mode = _pool_warmup_mode()
        parts = sorted({j.part for j in build_jobs(spec, ds)})
        self.parts = parts
        fpw = os.environ.get("MANGROVE_BENCH_XGBT_FOLDS_PER_WORKER", "").strip() or None
        self.k = folds_per_worker(self.W, len(parts), fpw)
        self.owned = assign_parts(self.W, parts, self.k)
        self.servers = {p: [w for w in range(self.W) if p in self.owned[w]] for p in parts}
        ctx = mp.get_context("spawn")
        self.res_q = ctx.Queue()
        self.task_qs = [ctx.Queue() for _ in range(self.W)]
        self.procs = [ctx.Process(target=_tuned_worker,
                                  args=(w, spec.to_dict(), None if data_dir is None else str(data_dir), self.nthread,
                                        device, self.owned[w], self.task_qs[w], self.res_q, torch_guard,
                                        self.fastscore, self.warmup_mode, self.cache))
                      for w in range(self.W)]
        for p in self.procs:
            p.start()
        ready = []
        while len(ready) < self.W:
            r = self.res_q.get(timeout=timeout_s)
            if "error" in r:
                self.close()
                raise RuntimeError(f"worker {r['wid']} failed during setup:\n{r['error']}")
            ready.append(r)
        self.ready = sorted(ready, key=lambda r: r["ready"])
        self.prep_s = max(r["prep_s"] for r in ready)
        self.warmup_s = max(r["warmup_s"] for r in ready)
        self.backlog = []
        self.outstanding = [0] * self.W
        self.owner = {}
        self.resident = [0] * self.W
        self.live_peak = [0] * self.W
        self.sum = {"train_s": 0.0, "score_s": 0.0, "reset_s": 0.0, "queue_s": 0.0}
        self.per_w = [{"segments": 0, "rounds": 0, "train_s": 0.0, "score_s": 0.0, "reset_s": 0.0} for _ in range(self.W)]
        self.wait_s = 0.0
        self.n_dropped = 0

    def _send(self, w, seg):
        mid = seg.job.mid
        if mid not in self.owner:
            self.owner[mid] = w
            self.resident[w] += 1
        self.outstanding[w] += 1
        self.task_qs[w].put(("seg", seg.seg_id, mid, seg.start, seg.end, time.perf_counter()))

    def _dispatch(self):
        for limit in (1, self.depth):
            keep = []
            for seg in self.backlog:
                mid = seg.job.mid
                w = self.owner.get(mid)
                if w is not None:
                    if self.outstanding[w] < limit:
                        self._send(w, seg)
                        continue
                else:
                    if seg.start != 0:
                        raise RuntimeError(f"model {mid}: continuation from round {seg.start} without a resident booster")
                    cand = [x for x in self.servers[seg.job.part] if self.outstanding[x] < limit]
                    if cand:
                        self._send(min(cand, key=lambda x: (self.outstanding[x], self.resident[x], x)), seg)
                        continue
                keep.append(seg)
            self.backlog = keep
        self._occ(sum(1 for x in self.outstanding if x > 0))

    def capacity(self):
        return self.cap

    def utilization_now(self):
        return sum(1 for x in self.outstanding if x > 0) / self.W

    def occupancy(self, max_points=20000):
        o = super().occupancy(max_points)
        o["capacity"] = self.W
        o["driver_capacity"] = self.cap
        return o

    def submit(self, seg):
        self.inflight[seg.seg_id] = seg
        self.backlog.append(seg)
        self._dispatch()

    def wait(self):
        if not self.inflight:
            return []
        out = []
        t0 = time.perf_counter()
        r = self.res_q.get(timeout=self.timeout_s)
        self.wait_s += time.perf_counter() - t0
        while True:
            if "error" in r:
                raise RuntimeError(f"worker {r['wid']} failed:\n{r['error']}")
            seg = self.inflight.pop(r["sid"])
            w = r["wid"]
            self.outstanding[w] -= 1
            if "fs" in r:
                self.fastscore_counts[r["fs"]] += 1
            busy = r["train_s"] + r["score_s"] + r["reset_s"]
            self.busy_total += busy
            for k in self.sum:
                self.sum[k] += r[k]
            st = self.per_w[w]
            st["segments"] += 1
            st["rounds"] += r["rounds"]
            for k in ("train_s", "score_s", "reset_s"):
                st[k] += r[k]
            self.live_peak[w] = max(self.live_peak[w], r["live_peak"])
            out.append(Done(seg, r["loss"], r["metrics"], seg.job.mid, seg.end - seg.start, busy))
            try:
                r = self.res_q.get_nowait()
            except Exception:
                break
        self._dispatch()
        return out

    def release(self, token):
        if token is None:
            return
        w = self.owner.pop(token, None)
        if w is not None:
            self.resident[w] -= 1
            self.n_dropped += 1
            self.task_qs[w].put(("drop", token))

    def breakdown(self, wall=None):
        rd = self.ready
        return {"scope": "per-worker setup (untimed, before the driver clock) and segment sums inside the driver run; "
                         "segment sums overlap across workers",
                "workers": self.W, "folds_per_worker": self.k, "parts_of_worker": self.owned,
                "matrix_build_s_max": max(r["build_s"] for r in rd), "matrix_build_s_sum": sum(r["build_s"] for r in rd),
                "data_setup_s_max": max(r["data_s"] for r in rd), "warmup_s_max": max(r["warmup_s"] for r in rd),
                "booster_save_load_s": 0.0,
                "train_s_sum": self.sum["train_s"], "score_s_sum": self.sum["score_s"],
                "cache_reset_s_sum": self.sum["reset_s"], "queue_s_sum": self.sum["queue_s"],
                "driver_wait_s": self.wait_s,
                "driver_bookkeeping_s": None if wall is None else max(0.0, wall - self.wait_s),
                "device_data": [r["device_data"] for r in rd], "cuts_equal_host": [r["cuts_equal_host"] for r in rd],
                "host_fallback_parts": [r["host_fallback_parts"] for r in rd],
                "cut_check_error": [r["cut_check_error"] for r in rd],
                "per_worker_setup": [{k: r[k] for k in ("ready", "data_s", "build_s", "cut_check_s", "warmup_s", "parts")}
                                     for r in rd]}

    def stats(self, wall):
        s = super().stats(wall)
        s["capacity"] = self.W
        s["slot_utilization"] = (self.busy_total / (self.W * wall)) if wall > 0 else None
        s.update({"driver_capacity": self.cap, "device": self.device, "nthread": self.nthread,
                  "continuation": "resident Booster.update in the owner worker (no serialization)", "cache": self.cache,
                  "dispatch_depth": self.depth, "qdepth": self.qdepth, "folds_per_worker": self.k,
                  "parts_of_worker": self.owned, "live_boosters_peak_per_worker": self.live_peak,
                  "boosters_dropped": self.n_dropped, "breakdown": self.breakdown(wall), "per_worker": self.per_w})
        if self.fastscore is not None:
            s["xgb_fastscore"] = {"mode": self.fastscore, "rank_sum_models": self.fastscore_counts["rank"],
                                  "host_models": self.fastscore_counts["host"],
                                  "workers": [r.get("fastscore") for r in self.ready]}
        if self.warmup_mode is not None:
            s["pool_warmup"] = self.warmup_mode
        return s

    def close(self):
        for q in getattr(self, "task_qs", []):
            q.put(None)
        for p in getattr(self, "procs", []):
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
