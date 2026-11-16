import contextlib
import inspect
import math
import sys
import time

import numpy as np

from .. import envinfo, paths
from .base import per_model_entry, unsupported

LOSS = {"binary": "logistic", "regression": "squared", "multiclass": "softmax"}

ABLATIONS = {
    "full": {},
    "no_worklist": {"engine": {"worklist": False}},
    "no_frontier": {"engine": {"frontier": False}},
    "dense_v1like": {"engine": {"worklist": False, "frontier": False}},
    "no_workspace_pool": {"engine": {"workspace_pool": False}},
    "wide_acc": {"spec": {"acc_mode": "wide"}},
    "quant28": {"spec": {"quant_bits": 28}},
    "fixed_chunk": {"engine": {"chunk_parents": None}},
}


def import_mangrove(opts):
    if str(paths.ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(paths.ENGINE_DIR))
    import mangrove
    return mangrove


def source_files():
    return sorted(paths.ENGINE_DIR.glob("mangrove/*.py")) + [p for p in sorted((paths.ENGINE_DIR / "cuda").glob("*"))
            if p.suffix in (".cu", ".h", ".dll", ".so", ".bat", ".sh")]


GPU_CHECK_MARKER = paths.RESULTS_DIR / "gpu_check.json"


def gpu_check_status():
    import json
    try:
        return json.loads(GPU_CHECK_MARKER.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def choose_backend(opts, gpu):
    b = opts.get("backend")
    if not gpu:
        return (b or "ref"), None
    b = b or "gpu"
    if b == "gpu" and opts.get("allow_fallback", True) and gpu_check_status() == "failed":
        return "ref", f"gpu_check marker {GPU_CHECK_MARKER.name} status=failed -> backend='ref' on CUDA"
    return b, None


def _engine_kwargs(mg, opts, gpu):
    kw = {"device": "cuda" if gpu else "cpu"}
    if opts.get("backend"):
        kw["backend"] = opts["backend"]
    if opts.get("memory_budget_gb"):
        kw["budget_bytes"] = int(float(opts["memory_budget_gb"]) * (1 << 30))
    abl = ABLATIONS.get(opts.get("ablation", "full"))
    if abl is None:
        raise ValueError(f"unknown ablation {opts.get('ablation')}")
    kw.update(abl.get("engine", {}))
    if opts.get("ablation") == "fixed_chunk":
        kw["chunk_parents"] = int(opts.get("chunk_parents", 64))
    if opts.get("phases", True):
        kw["phase_timing"] = True
    kw.update(opts.get("engine_kw", {}) or {})
    if "collect_metrics" in opts:
        kw["collect_metrics"] = bool(opts["collect_metrics"])
    sig = inspect.signature(mg.Engine.__init__).parameters
    missing = [k for k in kw if k not in sig and not any(p.kind == p.VAR_KEYWORD for p in sig.values())]
    if "phase_timing" in missing:
        kw.pop("phase_timing")
        missing.remove("phase_timing")
    return kw, missing, abl.get("spec", {})


def _gpu_budget_opts(opts, total_memory):
    limit = int(envinfo.MEM_FRACTION * int(total_memory))
    engine_kw = dict(opts.get("engine_kw") or {})
    if "budget_bytes" in engine_kw:
        requested = engine_kw["budget_bytes"]
    elif opts.get("memory_budget_gb") is not None:
        requested = float(opts["memory_budget_gb"]) * (1 << 30)
    else:
        requested = limit
    if not math.isfinite(float(requested)) or float(requested) <= 0:
        raise ValueError("Mangrove memory budget must be finite and positive")
    budget = int(requested)
    if budget <= 0 or budget > limit:
        raise ValueError(f"Mangrove effective memory budget {budget} B exceeds the process allowance "
                         f"{limit} B ({envinfo.MEM_FRACTION:g} * device total memory), or rounds to zero")
    engine_kw["budget_bytes"] = budget
    return dict(opts, engine_kw=engine_kw)


def make_specs(mg, jobs, ds, T, spec_over, opts):
    out = []
    for j in jobs:
        ev = j.val_mask if j.test_mask is None else (j.val_mask | j.test_mask)
        kw = dict(data_key=int(j.cohort), train_rows=np.nonzero(j.train_mask)[0], eval_rows=np.nonzero(ev)[0],
                  eta=j.params["eta"], lam=j.params["lam"], mcw=j.params["mcw"], max_depth=j.params["max_depth"],
                  n_rounds=T, subsample=j.params["subsample"], colsample_bytree=j.params["colsample"], seed=j.seed,
                  base_margin=ds.base_margin(j.train_mask), tag=(j.cfg, j.part))
        if opts.get("quant_bits"):
            kw["quant_bits"] = int(opts["quant_bits"])
        kw.update(spec_over)
        out.append(mg.ModelSpec(**kw))
    return out


def make_cohorts(mg, ds, cohort_ids):
    y = ds.y.astype(np.int64) if ds.task == "multiclass" else ds.y.astype(np.float32)
    out = []
    for c in cohort_ids:
        nb = ds.nbins(c)
        out.append(mg.Cohort(int(c), np.ascontiguousarray(ds.X(c)), y, nbins=None if nb is None else np.asarray(nb, dtype=np.int64)))
    return out


def phases_of(eng):
    ph = {}

    c = getattr(eng, "counters", None)
    if c:
        ph["levels_recorded"] = len(c)
        keys = sorted({k for r in c[:1000] if isinstance(r, dict) for k in r})
        agg = {}
        for k in keys:
            vals = [r[k] for r in c if isinstance(r, dict) and isinstance(r.get(k), (int, float))]
            if vals:
                agg[k] = {"sum": float(np.sum(vals)), "max": float(np.max(vals)), "mean": float(np.mean(vals))}
        ph["level_counters"] = agg
    try:
        ph["memory_report"] = eng.memory_report()
    except Exception:
        pass
    try:
        tr = eng.timing_report() if hasattr(eng, "timing_report") else None
        if tr:
            ph["phase_times"] = {k: float(v) for k, v in tr.items() if isinstance(v, (int, float))}
    except Exception as e:
        ph["phase_times_error"] = repr(e)
    return ph


def merged_phases(engines):
    ph = phases_of(engines[0])
    if len(engines) > 1:
        tot = {}
        for e in engines:
            for k, v in (phases_of(e).get("phase_times") or {}).items():
                tot[k] = tot.get(k, 0.0) + v
        ph["phase_times"] = tot
        ph["phase_times_note"] = f"summed over {len(engines)} lane engines (overlapping streams: sum > wall time)"
    return ph


def run(spec, ds, jobs, opts):
    import torch
    gpu = (opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"
    if gpu and opts.get("mode", "fused") == "streams":
        return unsupported("Mangrove multi-stream concurrency is not implemented: Engine.step performs blocking "
                           "host archives each round. A sequential stream loop would serialize the lanes; "
                           "use fused or m1 until concurrent dispatch is implemented and GPU-trace validated.")
    cap = None
    if gpu:
        envinfo.gpu_process_guard()
        total = torch.cuda.get_device_properties(0).total_memory
        opts = _gpu_budget_opts(opts, total)
        if opts.get("alloc_cap_factor") is not None:
            cap = envinfo.alloc_cap(opts["alloc_cap_factor"], opts["engine_kw"]["budget_bytes"], total)
    backend, why = choose_backend(opts, gpu)
    notes = [why] if why else []
    o = dict(opts, backend=backend, collect_metrics=False)
    try:
        r = _run_once(spec, ds, jobs, o, gpu)
    except Exception as e:
        if not (gpu and backend == "gpu" and opts.get("allow_fallback", True)):
            raise
        from .base import format_exc
        notes.append(f"backend='gpu' raised {type(e).__name__}: {str(e)[:300]} -> retried with backend='ref' on CUDA")
        notes.append(format_exc()[-1500:])
        if gpu:
            torch.cuda.empty_cache()
        o["backend"] = "ref"
        r = _run_once(spec, ds, jobs, o, gpu)
    r.setdefault("notes", []).extend(notes)
    r["variant_extra"] = {"backend": o["backend"], "fallback_used": o["backend"] != (opts.get("backend") or ("gpu" if gpu else "ref"))}
    if cap is not None:
        r["variant_extra"].update(cap)
    elif opts.get("alloc_cap_factor") is not None:
        r["variant_extra"]["alloc_cap_fraction_effective"] = None
    return r


def _run_once(spec, ds, jobs, opts, gpu):
    import torch
    mg = import_mangrove(opts)
    mode = opts.get("mode", "fused")
    kw, missing, spec_over = _engine_kwargs(mg, opts, gpu)
    if missing:
        return unsupported(f"mangrove.Engine does not accept {missing} (ablation {opts.get('ablation')})")
    md = max(j.params["max_depth"] for j in jobs)
    if md > 12:
        return unsupported("mangrove supports max_depth <= 12")
    task, K, T = ds.task, ds.K, spec.rounds
    loss = LOSS[task]
    sync = torch.cuda.synchronize if gpu else (lambda: None)
    cohort_ids = sorted({int(j.cohort) for j in jobs})
    timing, det = {}, {"engine_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()}}
    t0 = time.perf_counter()
    wco = make_cohorts(mg, ds, cohort_ids[:1])
    weng = mg.Engine(wco, loss, n_classes=K, **kw)
    wj = [j for j in jobs if int(j.cohort) == cohort_ids[0]][:2]
    weng.add_models(make_specs(mg, wj, ds, 2, spec_over, opts))
    weng.run()
    sync()
    del weng, wco
    timing["warmup_s"] = time.perf_counter() - t0
    if gpu:
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    cohorts = make_cohorts(mg, ds, cohort_ids)
    specs = make_specs(mg, jobs, ds, T, spec_over, opts)
    if mode == "fused":
        engines = [mg.Engine(cohorts, loss, n_classes=K, **kw)]
    elif mode == "m1":
        engines = [mg.Engine(cohorts, loss, n_classes=K, max_model_slots=1, **kw)]
    elif mode == "streams":
        Kst = int(opts.get("K", 8))
        engines = [mg.Engine(cohorts, loss, n_classes=K, max_model_slots=1, **kw) for _ in range(Kst)]
        det["streams"] = Kst
        det["note"] = "each lane engine holds its own copy of the cohort matrices"
    else:
        raise ValueError(mode)
    sync()
    timing["prep_s"] = time.perf_counter() - t0 + opts.get("load_s", 0.0)
    envinfo.gpu_reset_peak()
    from .. import gpumon
    sampler = gpumon.from_opts(opts) if gpu else None
    w0 = None
    det["initial_admission_s"] = det["engine_run_s"] = None
    try:
        if sampler is not None:
            sampler.start()
        w0 = time.time()
        t0 = time.perf_counter()
        owner = {}
        if mode in ("fused", "m1"):
            det["segmented_timing_note"] = "initial_admission_s includes add_models and synchronization; engine_run_s includes later admissions, training and final synchronization; train_s also includes owner bookkeeping"
            ta = time.perf_counter()
            ids = engines[0].add_models(specs)
            sync()
            det["initial_admission_s"] = time.perf_counter() - ta
            for i, mid in enumerate(ids):
                owner[i] = (engines[0], mid)
            tr = time.perf_counter()
            engines[0].run()
            sync()
            det["engine_run_s"] = time.perf_counter() - tr
        else:
            det["segmented_timing_note"] = "streams admission and stepping are interleaved; segment durations are unavailable"
            streams = [torch.cuda.Stream() for _ in engines] if gpu else [None] * len(engines)
            ctxf = (lambda s: torch.cuda.stream(s)) if gpu else (lambda s: contextlib.nullcontext())
            queue = list(range(len(specs)))[::-1]
            busy = [None] * len(engines)
            while queue or any(b is not None for b in busy):
                for li, eng in enumerate(engines):
                    with ctxf(streams[li]):
                        if busy[li] is None:
                            if not queue:
                                continue
                            i = queue.pop()
                            mid = eng.add_models([specs[i]])[0]
                            owner[i] = (eng, mid)
                            busy[li] = mid
                        eng.step()
                        if eng.reg[busy[li]].status in ("done", "retired"):
                            busy[li] = None
            sync()
        timing["train_s"] = time.perf_counter() - t0
    finally:
        if sampler is not None:
            try:
                if w0 is not None:
                    sampler.mark_window(w0, time.time())
            finally:
                sampler.stop()
    gpu_util = None
    if sampler is not None:
        gpu_util = sampler.summary(paths.rel_to_bench(opts["gpu_sample_path"]))
    memory = envinfo.gpu_peak() if gpu else {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    memory["host_peak_rss_bytes"] = envinfo.host_peak_rss()
    try:
        memory["engine_accounting"] = engines[0].memory_report()
    except Exception:
        pass
    t0 = time.perf_counter()
    per_model = []
    for i, j in enumerate(jobs):
        eng, mid = owner[i]
        st = eng.reg[mid]
        mg = np.asarray(st.eval_margin if st.eval_margin is not None else eng.eval_margin(mid))
        kw = {}
        if j.test_mask is not None:
            ev = np.nonzero(j.val_mask | j.test_mask)[0]
            vpos = np.searchsorted(ev, np.nonzero(j.val_mask)[0])
            tpos = np.searchsorted(ev, np.nonzero(j.test_mask)[0])
            kw = {"y_test": ds.y[j.test_mask], "test_margin": mg[tpos]}
            mg = mg[vpos]
        per_model.append(per_model_entry(j, task, ds.y[j.val_mask], mg, st.local_round,
                                         {"stop_reason": st.stop_reason}, **kw))
    timing["eval_s"] = time.perf_counter() - t0
    timing["details"] = det
    version = {"engine": "mangrove", "module": getattr(mg, "__file__", "mock"), "ablation": opts.get("ablation", "full"),
               "torch": torch.__version__}
    extra = source_files()
    return {"status": "ok", "timing": timing, "per_model": per_model, "memory": memory, "version": version,
            "gpu": gpu, "extra_sources": extra, "phases": merged_phases(engines) if opts.get("phases", True) else None,
            "gpu_util": gpu_util}
