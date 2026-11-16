import time
import multiprocessing as mp

import numpy as np

from .. import metrics as M
from .driver import Done
from .trainers import PoolTrainer, _Base, _xgb_fastscore_mode

RAM_TOTAL = 0.70
WARMUP_ROUNDS = 2
CONTINUATION = "none_grid_only"
UNSUPPORTED_REASON = (
    "catboost_gpu supports only --algo random (got --algo {algo}): CatBoost's GPU trainer refuses fit(init_model=...) "
    "continuation ('Training continuation for GPU is not yet supported', catboost/cuda/train_lib/train.cpp; init_model "
    "is CPU-only in the docs; GitHub issue #2736); snapshot-file resume was not evaluated (unresolved).  Retraining "
    "every rung from round 0 would charge CatBoost for rounds the other methods do not repeat, so sha / asha / "
    "hyperband-parallel / hyperband-sequential are not run.")


def default_gpu_ram_part(P):
    return round(RAM_TOTAL / max(1, int(P)), 4)


def cb_params(job, task, K, device, T, nthread, gpu_ram_part=None, regression_bfa=False):
    from ..runners.catboost_gpu import params
    p = params(job, task, K, device, T, nthread)
    if device == "cuda" and gpu_ram_part is not None:
        p["gpu_ram_part"] = float(gpu_ram_part)
    if regression_bfa and task == "regression":
        p["boost_from_average"] = True
    return p


def _max_abs_diff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _cb_worker(wid, spec_dict, data_dir, nthread, device, gpu_ram_part, quantize, task_q, res_q, torch_guard,
               fastscore=None, regression_bfa=False):
    try:
        from bench import envinfo
        if device == "cuda" and torch_guard:
            envinfo.gpu_process_guard(required=False)
        from bench.runners.pool import _setup
        import catboost
        from catboost import CatBoost, Pool
        t0 = time.perf_counter()
        spec, ds, jobs = _setup(spec_dict, data_dir)
        task, K = ds.task, ds.K
        parts = {}
        for j in jobs.values():
            parts.setdefault(j.part, j)
        task_type = "GPU" if device == "cuda" else "CPU"
        j_first = next(iter(parts.values()))
        border_count = cb_params(j_first, task, K, device, WARMUP_ROUNDS, nthread)["border_count"]

        def train_pool(j, quant):
            p = Pool(ds.X(j.cohort)[j.train_mask].astype(np.float32), label=ds.y[j.train_mask])
            if quant:
                p.quantize(border_count=border_count, task_type=task_type)
            return p

        def build(quant):
            tr, va = {}, {}
            for part, j in parts.items():
                tr[part] = train_pool(j, quant)
                va[part] = Pool(ds.X(j.cohort)[j.val_mask].astype(np.float32))
            return tr, va

        def fit(j, T, pool):
            m = CatBoost(cb_params(j, task, K, device, T, nthread, gpu_ram_part, regression_bfa))
            m.fit(pool)
            return m

        info = {"version": getattr(catboost, "__version__", None), "quantize": "pre" if quantize else "raw",
                "quantize_fallback": None, "quantize_check": None, "border_count": int(border_count),
                "gpu_ram_part": gpu_ram_part, "thread_count": nthread, "task_type": task_type,
                "regression_boost_from_average": bool(regression_bfa and task == "regression"),
                "warmup": f"{WARMUP_ROUNDS}-round fit + validation predict on every part"}
        prep_s = time.perf_counter() - t0
        warm_s = 0.0
        mode = info["quantize"]
        while True:
            ta, tb = time.perf_counter(), None
            try:
                dtr, dva = build(mode == "pre")
                tb = time.perf_counter()
                prep_s += tb - ta
                for j0 in parts.values():
                    fit(j0, WARMUP_ROUNDS, dtr[j0.part]).predict(dva[j0.part], prediction_type="RawFormulaVal")
                warm_s += time.perf_counter() - tb
                break
            except Exception as e:
                if mode != "pre":
                    raise
                if tb is None:
                    prep_s += time.perf_counter() - ta
                else:
                    warm_s += time.perf_counter() - tb
                info["quantize_fallback"] = f"{type(e).__name__}: {e}"[:800]
                mode = "raw"
        info["quantize"] = mode
        t2 = time.perf_counter()
        if mode == "pre":
            try:
                va0 = dva[j_first.part]
                m_pre = fit(j_first, WARMUP_ROUNDS, dtr[j_first.part]).predict(va0, prediction_type="RawFormulaVal")
                m_pre2 = fit(j_first, WARMUP_ROUNDS, dtr[j_first.part]).predict(va0, prediction_type="RawFormulaVal")
                m_raw = fit(j_first, WARMUP_ROUNDS, train_pool(j_first, False)).predict(va0, prediction_type="RawFormulaVal")
                info["quantize_check"] = {"part": int(j_first.part), "rounds": WARMUP_ROUNDS,
                                          "max_abs_margin_diff_raw_vs_pre": _max_abs_diff(m_raw, m_pre),
                                          "max_abs_margin_diff_pre_repeat": _max_abs_diff(m_pre2, m_pre)}
            except Exception as e:
                info["quantize_check"] = {"error": f"{type(e).__name__}: {e}"[:800]}
        scorer = None
        if fastscore is not None:
            from .xgb_fastscore import WorkerScorer
            scorer = WorkerScorer(ds, "cuda" if device == "cuda" else None, fastscore)
            if task == "binary" and scorer.auc_device is None:
                raise RuntimeError("MANGROVE_BENCH_XGB_FASTSCORE: torch sees no CUDA device in this GPU worker; "
                                   "refusing to score on the host")
            scorer.warmup()
        warm_s += time.perf_counter() - t2
        ready = {"ready": wid, "prep_s": prep_s, "warmup_s": warm_s, "catboost": info}
        if scorer is not None:
            ready["fastscore"] = scorer.describe()
        res_q.put(ready)
        while True:
            msg = task_q.get()
            if msg is None:
                break
            sid, mid, start, end, token = msg
            if start != 0 or token is not None:
                raise RuntimeError(f"catboost_gpu cannot continue a model (no GPU training continuation): segment {sid} "
                                   f"asks for rounds {start}..{end}; only --algo random is supported")
            j = jobs[mid]
            ts = time.perf_counter()
            m = fit(j, end, dtr[j.part])
            mg = np.asarray(m.predict(dva[j.part], prediction_type="RawFormulaVal"), dtype=np.float32)
            if task == "multiclass":
                mg = mg.reshape(-1, K)
            if scorer is None:
                mt = M.metrics(task, ds.y[j.val_mask], mg)
                res_q.put({"sid": sid, "loss": M.loss_value(task, mt), "metrics": mt, "token": None,
                           "busy_s": time.perf_counter() - ts, "wid": wid})
            else:
                mt, path = scorer.metrics(j, mg)
                res_q.put({"sid": sid, "loss": M.loss_value(task, mt), "metrics": mt, "token": None,
                           "busy_s": time.perf_counter() - ts, "wid": wid, "fs": path})
    except Exception:
        import traceback
        res_q.put({"error": traceback.format_exc()[-4000:], "wid": wid})


class CatBoostTrainer(PoolTrainer):
    supports_resume = False
    POLL_S = 5.0

    def __init__(self, spec, W, nthread=1, device="cuda", data_dir=None, torch_guard=True, timeout_s=6 * 3600,
                 gpu_ram_part=None, quantize=True, regression_bfa=False):
        _Base.__init__(self)
        if int(W) < 1:
            raise ValueError("catboost_gpu needs --P >= 1")
        self.W = int(W)
        self.timeout_s = timeout_s
        self.device = device
        self.gpu_ram_part = None
        if device == "cuda":
            self.gpu_ram_part = default_gpu_ram_part(self.W) if gpu_ram_part is None else float(gpu_ram_part)
        self.fastscore = _xgb_fastscore_mode() if device == "cuda" else None
        self.fastscore_workers, self.fastscore_counts = None, {"rank": 0, "host": 0}
        self.warmup_mode = None
        self.quantize = bool(quantize)
        self.regression_bfa = bool(regression_bfa)
        ctx = mp.get_context("spawn")
        self.task_q, self.res_q = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_cb_worker, args=(w, spec.to_dict(), None if data_dir is None else str(data_dir),
                                                            nthread, device, self.gpu_ram_part, self.quantize,
                                                            self.task_q, self.res_q, torch_guard, self.fastscore,
                                                            self.regression_bfa))
                      for w in range(self.W)]
        for p in self.procs:
            p.start()
        ready = []
        try:
            while len(ready) < self.W:
                r = self._get(timeout_s)
                if "error" in r:
                    raise RuntimeError(f"catboost worker {r['wid']} failed during setup:\n{r['error']}")
                ready.append(r)
        except BaseException:
            self.close()
            raise
        ready.sort(key=lambda r: r["ready"])
        self.prep_s = max(r["prep_s"] for r in ready)
        self.warmup_s = max(r["warmup_s"] for r in ready)
        if self.fastscore is not None:
            self.fastscore_workers = [r.get("fastscore") for r in ready]
        self.cb_workers = [r["catboost"] for r in ready]
        self.version = self.cb_workers[0].get("version") if self.cb_workers else None
        modes = sorted({w["quantize"] for w in self.cb_workers})
        self.quantize_effective = modes[0] if len(modes) == 1 else "mixed"
        self.result_notes = [f"catboost worker {i}: pre-quantized pools rejected, fell back to raw pools (quantization "
                             f"inside every timed fit): {w['quantize_fallback']}"
                             for i, w in enumerate(self.cb_workers) if w.get("quantize_fallback")]

    def _get(self, timeout_s):
        import queue
        t_end = time.perf_counter() + timeout_s
        while True:
            try:
                return self.res_q.get(timeout=max(0.05, min(self.POLL_S, t_end - time.perf_counter())))
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs) if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"catboost worker(s) exited without reporting (index, exit code): {dead}; "
                                       "-6 = abort, e.g. a CatBoost GPU-side error (std::terminate)")
                if time.perf_counter() >= t_end:
                    raise TimeoutError(f"no catboost worker result within {timeout_s} s")

    def wait(self):
        if not self.inflight:
            return []
        out = []
        r = self._get(self.timeout_s)
        while True:
            if "error" in r:
                raise RuntimeError(f"catboost worker {r['wid']} failed:\n{r['error']}")
            seg = self.inflight.pop(r["sid"])
            self._occ(min(self.W, len(self.inflight)))
            if "fs" in r:
                self.fastscore_counts[r["fs"]] += 1
            self.busy_total += r["busy_s"]
            out.append(Done(seg, r["loss"], r["metrics"], r["token"], seg.end - seg.start, r["busy_s"]))
            try:
                r = self.res_q.get_nowait()
            except Exception:
                break
        return out

    def stats(self, wall):
        s = super().stats(wall)
        s["catboost"] = {"continuation": CONTINUATION, "device": self.device, "gpu_ram_part": self.gpu_ram_part,
                         "quantize_requested": self.quantize, "quantize_effective": self.quantize_effective,
                         "regression_boost_from_average": self.regression_bfa, "workers": self.cb_workers}
        return s
