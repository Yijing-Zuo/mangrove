import os
import time
import multiprocessing as mp

import numpy as np

from .. import metrics as M
from .driver import Done


class _Base:
    supports_resume = True
    supports_pause = True

    def __init__(self):
        self.inflight = {}
        self.busy_total = 0.0
        self.t_start = time.perf_counter()
        self.clock0 = None
        self.prep_s = 0.0
        self.warmup_s = 0.0
        self.occ = []

    def _occ(self, busy):
        if self.clock0 is not None:
            self.occ.append((round(time.perf_counter() - self.clock0, 4), int(busy)))

    def occupancy(self, max_points=20000):
        o = self.occ
        if len(o) > max_points:
            step = len(o) / max_points
            o = [o[int(i * step)] for i in range(max_points)]
        return {"capacity": self.capacity(), "samples": o}

    def in_flight(self):
        return len(self.inflight)

    def release(self, token):
        pass

    def utilization_now(self):
        return self.in_flight() / max(1, self.capacity())

    def stats(self, wall):
        return {"capacity": self.capacity(), "busy_s": self.busy_total,
                "slot_utilization": (self.busy_total / (self.capacity() * wall)) if wall > 0 else None,
                "prep_s": self.prep_s, "warmup_s": self.warmup_s}

    def close(self):
        pass


class MockTrainer(_Base):
    def __init__(self, W=4, round_s=0.0, resume=True, clock=None):
        super().__init__()
        self.W = W
        self.supports_resume = resume
        self.round_s = round_s
        self.queue = []
        self.sim_t = 0.0

    def capacity(self):
        return self.W

    def submit(self, seg):
        self.inflight[seg.seg_id] = seg

    def loss(self, job, r):
        rng = np.random.default_rng(job.cfg)
        a, b = rng.uniform(0.2, 0.5), rng.uniform(0.1, 1.0)
        return float(a + b / (1.0 + r / 20.0) + 0.001 * job.part)

    def wait(self):
        if not self.inflight:
            return []
        sid = min(self.inflight, key=lambda s: (self.inflight[s].end - self.inflight[s].start, s))
        seg = self.inflight.pop(sid)
        rt = seg.end - seg.start if self.supports_resume else seg.end
        self.busy_total += rt * self.round_s
        self.sim_t += rt * self.round_s / self.W
        L = self.loss(seg.job, seg.end)
        return [Done(seg, L, {"loss": L}, ("tok", seg.job.mid, seg.end), rt, rt * self.round_s)]


def _xgb_fastscore_mode():
    v = os.environ.get("MANGROVE_BENCH_XGB_FASTSCORE", "0").strip()
    if v in ("", "0"):
        return None
    if v in ("1", "auc"):
        return "full" if v == "1" else "auc"
    raise ValueError(f"MANGROVE_BENCH_XGB_FASTSCORE={v!r}: expected 0, 1 or auc")


def _pool_warmup_mode():
    v = os.environ.get("MANGROVE_BENCH_XGB_WARMUP", "").strip()
    if v in ("", "first"):
        return None
    if v == "all":
        return "all"
    raise ValueError(f"MANGROVE_BENCH_XGB_WARMUP={v!r}: expected first or all")


def _pool_worker(wid, lib, spec_dict, data_dir, nthread, device, task_q, res_q, torch_guard, fastscore=None,
                 warmup=None):
    try:
        from bench import envinfo
        if device == "cuda" and torch_guard:
            envinfo.gpu_process_guard(required=False)
        from bench.runners.pool import _setup, xgb_params, lgb_params
        t0 = time.perf_counter()
        spec, ds, jobs = _setup(spec_dict, data_dir)
        task, K = ds.task, ds.K
        parts = {}
        for j in jobs.values():
            parts.setdefault(j.part, j)
        dtr, dva, base = {}, {}, {}
        if lib == "xgb":
            import xgboost as xgb
            for part, j in parts.items():
                X = ds.X(j.cohort)
                base[part] = ds.base_margin(j.train_mask)
                dtr[part] = xgb.QuantileDMatrix(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask], max_bin=256, nthread=nthread)
                dva[part] = xgb.DMatrix(X[j.val_mask].astype(np.float32), nthread=nthread)
        else:
            import lightgbm as lgb
            for part, j in parts.items():
                X = ds.X(j.cohort)
                dtr[part] = lgb.Dataset(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask],
                                        params={"max_bin": 256, "verbose": -1, "num_threads": nthread, "min_data_in_leaf": 1,
                                                "feature_pre_filter": False}, free_raw_data=False)
                dtr[part].construct()
                dva[part] = X[j.val_mask].astype(np.float32)
        t1 = time.perf_counter()
        for j0 in (list(parts.values()) if warmup == "all" else [next(iter(parts.values()))]):
            if lib == "xgb":
                xgb.train(xgb_params(j0, task, K, nthread, device, base[j0.part]), dtr[j0.part], 2)
            else:
                lgb.train(lgb_params(j0, task, K, nthread), dtr[j0.part], 2)
        scorer = None
        if fastscore is not None:
            from .xgb_fastscore import WorkerScorer
            scorer = WorkerScorer(ds, "cuda" if device == "cuda" else None, fastscore)
            if task == "binary" and scorer.auc_device is None:
                raise RuntimeError("MANGROVE_BENCH_XGB_FASTSCORE: torch sees no CUDA device in this GPU worker; "
                                   "refusing to score on the host")
            scorer.warmup()
        t2 = time.perf_counter()
        if scorer is None:
            res_q.put({"ready": wid, "prep_s": t1 - t0, "warmup_s": t2 - t1})
        else:
            res_q.put({"ready": wid, "prep_s": t1 - t0, "warmup_s": t2 - t1, "fastscore": scorer.describe()})
        while True:
            msg = task_q.get()
            if msg is None:
                break
            sid, mid, start, end, token = msg
            j = jobs[mid]
            ts = time.perf_counter()
            n = end - start
            if lib == "xgb":
                prev = None
                if token is not None:
                    prev = xgb.Booster()
                    prev.load_model(bytearray(token))
                bst = xgb.train(xgb_params(j, task, K, nthread, device, base[j.part]), dtr[j.part], n, xgb_model=prev)
                mg = bst.predict(dva[j.part], output_margin=True)
                tok = bytes(bst.save_raw("ubj"))
            else:
                init = lgb.Booster(model_str=token) if token is not None else None
                bst = lgb.train(lgb_params(j, task, K, nthread), dtr[j.part], n, init_model=init, keep_training_booster=False)
                mg = bst.predict(dva[j.part], raw_score=True)
                tok = bst.model_to_string()
            if task == "multiclass":
                mg = np.asarray(mg).reshape(-1, K)
            if scorer is None:
                mt = M.metrics(task, ds.y[j.val_mask], mg)
                res_q.put({"sid": sid, "loss": M.loss_value(task, mt), "metrics": mt, "token": tok, "busy_s": time.perf_counter() - ts,
                           "wid": wid})
            else:
                mt, path = scorer.metrics(j, mg)
                res_q.put({"sid": sid, "loss": M.loss_value(task, mt), "metrics": mt, "token": tok, "busy_s": time.perf_counter() - ts,
                           "wid": wid, "fs": path})
    except Exception:
        import traceback
        res_q.put({"error": traceback.format_exc()[-4000:], "wid": wid})


class PoolTrainer(_Base):
    def __init__(self, lib, spec, W, nthread=1, device="cpu", data_dir=None, torch_guard=True, timeout_s=6 * 3600):
        super().__init__()
        self.W = W
        self.timeout_s = timeout_s
        self.fastscore = _xgb_fastscore_mode() if lib == "xgb" else None
        if device != "cuda":
            self.fastscore = None
        self.fastscore_workers, self.fastscore_counts = None, {"rank": 0, "host": 0}
        self.warmup_mode = _pool_warmup_mode()
        ctx = mp.get_context("spawn")
        self.task_q, self.res_q = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_pool_worker, args=(w, lib, spec.to_dict(), None if data_dir is None else str(data_dir),
                                                              nthread, device, self.task_q, self.res_q, torch_guard,
                                                              self.fastscore, self.warmup_mode))
                      for w in range(W)]
        for p in self.procs:
            p.start()
        ready = []
        while len(ready) < W:
            r = self.res_q.get(timeout=timeout_s)
            if "error" in r:
                self.close()
                raise RuntimeError(f"worker {r['wid']} failed during setup:\n{r['error']}")
            ready.append(r)
        self.prep_s = max(r["prep_s"] for r in ready)
        self.warmup_s = max(r["warmup_s"] for r in ready)
        if self.fastscore is not None:
            self.fastscore_workers = [r.get("fastscore") for r in sorted(ready, key=lambda r: r["ready"])]

    def capacity(self):
        return self.W

    def submit(self, seg):
        self.inflight[seg.seg_id] = seg
        self.task_q.put((seg.seg_id, seg.job.mid, seg.start, seg.end, seg.token))
        self._occ(min(self.W, len(self.inflight)))

    def wait(self):
        if not self.inflight:
            return []
        out = []
        r = self.res_q.get(timeout=self.timeout_s)
        while True:
            if "error" in r:
                raise RuntimeError(f"worker {r['wid']} failed:\n{r['error']}")
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
        if self.fastscore is not None:
            s["xgb_fastscore"] = {"mode": self.fastscore, "rank_sum_models": self.fastscore_counts["rank"],
                                  "host_models": self.fastscore_counts["host"], "workers": self.fastscore_workers}
        if self.warmup_mode is not None:
            s["pool_warmup"] = self.warmup_mode
        return s

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


class MangroveTrainer(_Base):
    supports_resume = True
    supports_pause = True

    def __init__(self, spec, ds, opts, r_max):
        super().__init__()
        import torch
        from ..runners import mangrove
        self.torch = torch
        self.mg = mangrove.import_mangrove(opts)
        self.ds = ds
        self.gpu = (opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"
        backend, why = mangrove.choose_backend(dict(opts, allow_fallback=False), self.gpu)
        o = dict(opts, backend=backend, collect_metrics=False)
        if o.get("memory_budget_gb") is None:
            o["memory_budget_gb"] = 16.0
        if self.gpu:
            from .. import envinfo
            envinfo.gpu_process_guard()
            total = torch.cuda.get_device_properties(0).total_memory
            o = mangrove._gpu_budget_opts(o, total)
        self.alloc_cap_factor = opts.get("alloc_cap_factor")
        self.alloc_fraction = None
        if self.gpu and self.alloc_cap_factor:
            budget = int(o["engine_kw"]["budget_bytes"])
            self.alloc_fraction = min(envinfo.MEM_FRACTION, float(self.alloc_cap_factor) * budget / total)
            torch.cuda.set_per_process_memory_fraction(self.alloc_fraction, 0)
        self.notes = [why] if why else []
        self.fastscore = os.environ.get("MANGROVE_BENCH_MG_FASTSCORE", "0") == "1"
        self.scorer = None
        if self.fastscore:
            from .mg_fastscore import FastScorer
            self.scorer = FastScorer(ds, auc_device="cuda" if self.gpu else None)
            self.notes.append("fastscore: " + self.scorer.describe())
        self.liferec = os.environ.get("MANGROVE_BENCH_MG_LIFEREC", "0") == "1"
        self.steptrace = os.environ.get("MANGROVE_BENCH_MG_STEPTRACE", "0") == "1"
        self.step_trace = []
        self.admit_fold_major = os.environ.get("MANGROVE_BENCH_MG_ADMIT_ORDER", "") == "fold" and spec.dynamic.get("algo") != "asha"
        self._stats_out = None
        kw, missing, self.spec_over = mangrove._engine_kwargs(self.mg, o, self.gpu)
        if missing:
            raise ValueError(f"mangrove.Engine lacks {missing}")
        self.slots = int(opts.get("slots", 1024))
        kw["max_model_slots"] = self.slots
        self.engine_kwargs = dict(kw)
        self.r_max = r_max
        self.opts = o
        self.backend = backend
        t0 = time.perf_counter()
        cohort_ids = sorted({int(c) for c in ([-1] if spec.cuts == "global" else range(ds.n_folds))})
        cohorts = mangrove.make_cohorts(self.mg, ds, cohort_ids)
        self.prep_s = time.perf_counter() - t0
        from ..workload import build_jobs
        wjobs = build_jobs(spec, ds)[:2]
        t0 = time.perf_counter()
        weng = self.mg.Engine(cohorts, mangrove.LOSS[ds.task], n_classes=ds.K, **kw)
        if not all(hasattr(weng, name) for name in ("pause", "resume")):
            raise ValueError("dynamic Mangrove requires Engine.pause/resume to honor rung barriers")
        weng.add_models(mangrove.make_specs(self.mg, wjobs, ds, 2, self.spec_over, opts))
        weng.run()
        if self.gpu:
            torch.cuda.synchronize()
        del weng
        if self.scorer is not None:
            self.scorer.warmup()
        self.warmup_s = time.perf_counter() - t0
        if self.gpu:
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        self.eng = self.mg.Engine(cohorts, mangrove.LOSS[ds.task], n_classes=ds.K, **kw)
        if self.gpu:
            torch.cuda.synchronize()
        self.prep_s += time.perf_counter() - t0
        self.target = {}
        self.mid_of = {}
        self.known_mids = set()
        self.rounds_executed = 0
        self.steps = 0
        self.active_sum = 0

    def capacity(self):
        return self.slots

    def submit(self, seg):
        self.inflight[seg.seg_id] = seg
        if seg.token is None:
            if self.fastscore:
                sp = self._make_spec(seg.job)
            else:
                from ..runners.mangrove import make_specs
                sp = make_specs(self.mg, [seg.job], self.ds, self.r_max, self.spec_over, self.opts)[0]
            mid = self.eng.add_models([sp], admit=False)[0]
        else:
            mid = seg.token
            self.eng.resume([mid], admit=False)
        self.known_mids.add(mid)
        self.target[mid] = seg

    def _make_spec(self, j):
        train_rows, eval_rows, base = self.scorer.spec_rows(j)
        kw = dict(data_key=int(j.cohort), train_rows=train_rows, eval_rows=eval_rows,
                  eta=j.params["eta"], lam=j.params["lam"], mcw=j.params["mcw"], max_depth=j.params["max_depth"],
                  n_rounds=self.r_max, subsample=j.params["subsample"], colsample_bytree=j.params["colsample"],
                  seed=j.seed, base_margin=base, tag=(j.cfg, j.part))
        if self.opts.get("quant_bits"):
            kw["quant_bits"] = int(self.opts["quant_bits"])
        kw.update(self.spec_over)
        return self.mg.ModelSpec(**kw)

    def release(self, token):
        if token is None:
            return
        try:
            st = self.eng.reg[token]
        except (KeyError, IndexError):
            return
        if st.status in ("active", "paused"):
            self.eng.retire([token])
        elif st.status == "pending" and token in getattr(self.eng, "pending", []):
            self.eng.pending.remove(token)
            st.status = "retired"

    def wait(self):
        if not self.inflight:
            return []
        while True:
            out = []
            ready = [(mid, seg) for mid, seg in self.target.items() if self.eng.reg[mid].local_round >= seg.end]
            for mid, seg in ready:
                if self.eng.reg[mid].local_round != seg.end:
                    raise RuntimeError(f"model {mid} overshot rung {seg.end}: reached {self.eng.reg[mid].local_round}")
            self.eng.pause([mid for mid, _ in ready if self.eng.reg[mid].status == "active"])
            if self.fastscore and ready:
                scored = self.scorer.metrics_batch([(seg.job, np.asarray(self.eng.eval_margin(mid))) for mid, seg in ready])
            for i, (mid, seg) in enumerate(ready):
                j = seg.job
                if self.fastscore:
                    mt = scored[i]
                else:
                    mt = M.metrics(self.ds.task, self.ds.y[j.val_mask], np.asarray(self.eng.eval_margin(mid)))
                L = M.loss_value(self.ds.task, mt)
                self.inflight.pop(seg.seg_id)
                del self.target[mid]
                out.append(Done(seg, L, mt, mid, seg.end - seg.start))
            if out:
                return out
            if self.steptrace:
                self._trace_begin()
            ts = time.perf_counter()
            before = {mid: self.eng.reg[mid].local_round for mid in self.known_mids}
            if self.admit_fold_major and self.eng.pending:
                self.eng.pending.sort(key=lambda m: self.eng.reg[m].spec.data_key)
            self.eng.step()
            advanced = sum(self.eng.reg[mid].local_round - r for mid, r in before.items())
            if advanced == 0:
                raise MemoryError("Mangrove made no training progress: the memory/slot budget admits no pending model")
            self.rounds_executed += advanced
            if self.gpu:
                self.torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            n_act = advanced
            if self.clock0 is not None:
                self.occ.append((round(ts - self.clock0, 4), n_act))
                self._occ(0)
            self.active_sum += n_act
            if self.steptrace:
                self._trace_end(n_act)
            self.steps += 1
            self.busy_total += dt * n_act

    def _trace_begin(self):
        self._trace_lv0 = len(getattr(self.eng, "counters", None) or [])
        if self.gpu:
            self.torch.cuda.reset_peak_memory_stats()

    def _trace_end(self, resident):
        allocated = peak = reserved = None
        if self.gpu:
            c = self.torch.cuda
            allocated, peak, reserved = int(c.memory_allocated()), int(c.max_memory_allocated()), int(c.memory_reserved())
        e = self.eng
        rec = dict(step=self.steps, resident=int(resident), active_after=len(e.active_ids()), allocated_bytes=allocated, peak_allocated_bytes=peak,
                   reserved_bytes=reserved, plan_shared_bytes=None, plan_persistent_bytes=None, plan_transient_bytes=None, plan_level_peak_bytes=None,
                   plan_peak_total_bytes=None, plan_budget_bytes=None, plan_reserve_bytes=None)
        acc, pl = getattr(e, "acc", None), getattr(e, "planner", None)
        if acc is not None and pl is not None:
            lv = [r["peak_hist_bytes"] + r["row_bytes"] + r["split_bytes"] for r in (getattr(e, "counters", None) or [])[self._trace_lv0:] if "peak_hist_bytes" in r]
            rec.update(plan_shared_bytes=int(acc.shared), plan_persistent_bytes=int(acc.persistent), plan_transient_bytes=int(acc.transient),
                       plan_level_peak_bytes=int(acc.shared + acc.persistent + max(lv)) if lv else None, plan_peak_total_bytes=int(acc.peak_total),
                       plan_budget_bytes=int(pl.budget), plan_reserve_bytes=int(pl.reserve))
        self.step_trace.append(rec)

    def utilization_now(self):
        try:
            return len(self.eng.active_ids()) / self.slots
        except Exception:
            return None

    def stats(self, wall):
        out = {"capacity": self.slots, "steps": self.steps, "backend": self.backend, "notes": self.notes,
               "engine_kwargs": self.engine_kwargs,
               "actual_model_rounds": self.rounds_executed,
               "mean_active_models": self.active_sum / max(1, self.steps),
               "slot_utilization": self.busy_total / (self.slots * wall) if wall > 0 else None,
               "occupancy_scope": "models advanced during each engine step; other driver work is idle; not GPU utilization",
               "prep_s": self.prep_s,
               "warmup_s": self.warmup_s,
               "alloc_cap_factor": self.alloc_cap_factor, "alloc_fraction_effective": self.alloc_fraction}
        if self.scorer is not None:
            out["fastscore"] = self.scorer.stats()
        if self.admit_fold_major:
            out["admit_order"] = "fold"
        if self.steptrace:
            out["step_trace"] = self.step_trace
            out["step_trace_scope"] = ("one record per engine step of wait(); torch's peak counter is reset before each step, so this run's "
                                       "memory.gpu_peak_* cover only the last step and what followed it, not the whole run")
        if self.liferec:
            out["lifecycle"] = None
            self._stats_out = out
        return out

    def _lifecycle_record(self):
        from collections import Counter
        e = self.eng

        def js(v):
            if isinstance(v, dict):
                return {str(k): js(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [js(x) for x in v]
            if isinstance(v, np.generic):
                return v.item()
            return v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)
        rec = {"flag": "MANGROVE_BENCH_MG_LIFEREC=1"}
        try:
            if self.gpu:
                self.torch.cuda.synchronize()
            e.nm_check()
            rec["nm_check"] = "ok"
        except Exception as x:
            rec["nm_check"] = ("FAILED: " + repr(x))[:1000]
        lv = list(getattr(e, "counters", None) or [])
        rec.update(n_evictions=int(getattr(e, "n_evictions", 0)), nm_stats=js(dict(getattr(e, "nm_stats", {}) or {})),
                   levels=len(lv),
                   chunked_levels=sum(1 for c in lv if c.get("chunk") and c.get("n_frontier")
                                      and c["chunk"] < c["n_frontier"]),
                   max_chunks_per_level=max((-(-int(c["n_frontier"]) // int(c["chunk"])) for c in lv
                                             if c.get("chunk") and c.get("n_frontier")), default=0),
                   max_groups_per_level=max((int(c.get("n_groups") or 0) for c in lv), default=0),
                   events=js(dict(Counter(ev.get("event") for ev in (getattr(e, "events", None) or [])))))
        try:
            rec["memory_report"] = js(e.memory_report())
        except Exception as x:
            rec["memory_report"] = "unavailable: " + repr(x)[:200]
        if hasattr(e, "pin_stats"):
            rec["pin_stats"] = js(dict(e.pin_stats))
        return rec

    def close(self):
        if self.liferec and self._stats_out is not None:
            self._stats_out["lifecycle"] = self._lifecycle_record()
        if self.scorer is not None:
            self.scorer.close()
