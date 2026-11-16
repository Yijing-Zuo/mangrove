import time

import numpy as np

from .. import envinfo
from .base import per_model_entry, unsupported


class _TimeBox(BaseException):
    pass


def run(spec, ds, jobs, opts):
    if ds.task != "regression":
        return unsupported("OpenBoost grouped CUDA path is wired for squared loss only")
    if any(float(j.params.get(k, 1.0)) != 1.0 for j in jobs for k in ("subsample", "colsample")):
        return unsupported("OpenBoost grouped adapter supports subsample=colsample=1 only")
    envinfo.gpu_process_guard(required=False)
    import openboost
    from openboost import NumericData, Problem
    from openboost.binning import Binning
    from openboost.execution import ExecutionContext
    from openboost.device import DeviceOperations
    from openboost import device_inputs, device_recipes
    from openboost.device_runs import RunSpec
    from openboost.device_group_runs import run_many
    import cupy
    group_size = int(opts.get("group_size", 32))
    time_box = float(opts.get("time_box_s", 1200))
    T = spec.rounds
    timing, notes = {}, []
    t0 = time.perf_counter()
    total_bytes = int(cupy.cuda.Device(0).mem_info[1])
    pool_cap = int(total_bytes * envinfo.MEM_FRACTION)
    pool_bytes = int(float(opts.get("memory_budget_gb", min(16.0, pool_cap / 2 ** 30))) * 2 ** 30)
    if not 0 < pool_bytes <= pool_cap:
        raise ValueError("OpenBoost memory budget must be positive and at most device memory * 0.75")
    ctx = ExecutionContext("cuda:0", max_bytes=pool_bytes)
    ctx.__enter__()
    try:
        ops = DeviceOperations(ctx)
        parts = {}
        for j in jobs:
            if j.part in parts:
                continue
            X = ds.X(j.cohort).astype(np.float64)
            names = tuple(f"f{i}" for i in range(ds.d))
            nb = X.max(0).astype(int) + 1
            binning = Binning(names, tuple(np.arange(int(b) - 1, dtype=np.float64) + 0.5 for b in nb))
            tr_idx, va_idx = np.nonzero(j.train_mask)[0], np.nonzero(j.val_mask)[0]
            dtr = NumericData(X[tr_idx], tr_idx.astype(np.int64), names)
            dva = NumericData(X[va_idx], va_idx.astype(np.int64), names)
            ptr = Problem(dtr, ds.y[tr_idx].astype(np.float64)[:, None], dtr.row_ids)
            pva = Problem(dva, ds.y[va_idx].astype(np.float64)[:, None], dva.row_ids)
            prepared = (device_inputs.prepare(ops, binning.transform(dtr)), device_inputs.prepare(ops, binning.transform(dva)))
            parts[j.part] = dict(binning=binning, train=ptr, valid=pva, prepared=prepared, dva=dva)
        timing["prep_s"] = time.perf_counter() - t0 + opts.get("load_s", 0.0)

        def mk(j, rounds):
            p = parts[j.part]
            o = {"rounds": int(rounds), "learning_rate": float(j.params["eta"]), "max_depth": int(j.params["max_depth"]),
                 "reg_lambda": float(j.params["lam"]), "min_child_h": float(j.params["mcw"])}
            return RunSpec(run_id=f"m{j.mid}", seed=int(j.seed), train=p["train"], validation=p["valid"],
                           recipe=device_recipes.squared, options=o, binning=p["binning"], prepared=p["prepared"])
        t0 = time.perf_counter()
        warm = run_many(ops, [mk(jobs[0], 2)], group_size=1)
        if warm[0].error_type is not None:
            raise RuntimeError(f"OpenBoost warmup failed: {warm[0].error_type}: {warm[0].error_message}")
        timing["warmup_s"] = time.perf_counter() - t0
        by_part = {}
        for j in jobs:
            by_part.setdefault(j.part, []).append(j)
        from openboost import device_group_growth as _gg
        _grow, KEY = _gg.depthwise, "grouped_run_round_slots"
        done = {}
        stopped, terminal, aborted, fired = False, None, None, {}
        t0 = time.perf_counter()
        deadline = t0 + time_box

        def _boxed(ops_, jobs_, *a, **kw):
            if time.perf_counter() >= deadline:
                fired["slots"] = len(jobs_)
                raise _TimeBox()
            return _grow(ops_, jobs_, *a, **kw)

        _gg.depthwise = _boxed
        try:
            for part, pj in by_part.items():
                for c0 in range(0, len(pj), group_size):
                    if time.perf_counter() >= deadline:
                        stopped, terminal = True, "time_box"
                        break
                    chunk = pj[c0:c0 + group_size]
                    s0 = int(ctx.metrics.get(KEY, 0))
                    try:
                        outs = run_many(ops, [mk(j, T) for j in chunk], group_size=group_size)
                    except _TimeBox:
                        stopped, terminal = True, "time_box"
                        aborted = {"models": len(chunk), "first_mid": int(chunk[0].mid),
                                   "model_rounds_done": max(0, int(ctx.metrics.get(KEY, 0)) - s0
                                                            - int(fired.get("slots", 0)))}
                        break
                    except Exception as e:
                        if "memory" in (type(e).__name__ + str(e)).lower():
                            stopped, terminal = True, "oom"
                            notes.append(f"OOM in chunk starting at model {chunk[0].mid}: {type(e).__name__}: {str(e)[:200]}")
                            break
                        raise
                    for j, o in zip(chunk, outs):
                        done[j.mid] = o
                        if o.error_type and "memory" in (o.error_type + str(o.error_message)).lower():
                            terminal = terminal or "oom"
                if stopped:
                    break
        finally:
            _gg.depthwise = _grow
        ctx.synchronize()
        timing["train_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        per_model, kept = [], []
        for j in jobs:
            o = done.get(j.mid)
            if o is None:
                continue
            if o.error_type is not None:
                notes.append(f"m{j.mid}: {o.error_type}: {str(o.error_message)[:200]}")
                continue
            if o.result.stop.completed_rounds != T:
                notes.append(f"m{j.mid}: incomplete rounds {o.result.stop.completed_rounds}/{T}")
                continue
            mg = np.asarray(o.result.model.predict(parts[j.part]["dva"]))[:, 0]
            per_model.append(per_model_entry(j, ds.task, ds.y[j.val_mask], mg, o.result.stop.completed_rounds))
            kept.append(j)
        timing["eval_s"] = time.perf_counter() - t0
        mr = len(kept) * T + (aborted or {}).get("model_rounds_done", 0)
        timing["details"] = {"time_box_s": time_box, "aborted_chunk": aborted,
                             "box_overshoot_s": timing["train_s"] - time_box if terminal == "time_box" else None,
                             "model_rounds_completed_models": len(kept) * T, "model_rounds_executed": mr,
                             "model_rounds_per_s": mr / timing["train_s"] if timing["train_s"] else None}
        pool_metrics = dict(ctx.metrics)
    finally:
        ctx.__exit__(None, None, None)
    status = "ok" if len(kept) == len(jobs) else ("partial" if kept else "failed")
    if terminal == "time_box":
        notes.append(f"time box {time_box:.0f}s reached: {len(kept)}/{len(jobs)} models trained")
    if status != "ok" and terminal is None:
        terminal = None
    return {"status": status, "timing": timing, "per_model": per_model, "jobs_kept": kept,
            "memory": {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
                       "host_peak_rss_bytes": envinfo.host_peak_rss(), "openboost_pool_budget_bytes": pool_bytes,
                       "openboost_peak_live_bytes": pool_metrics.get("peak_live_bytes"),
                       "openboost_peak_pool_bytes": pool_metrics.get("peak_pool_bytes")},
            "version": {"openboost": getattr(openboost, "__version__", "?")}, "gpu": True, "notes": notes,
            "variant_extra": {"memory_budget_gb": pool_bytes / 2 ** 30, "group_size": group_size,
                              "time_box_s": time_box},
            "terminal_reason": terminal if status != "ok" else None}
