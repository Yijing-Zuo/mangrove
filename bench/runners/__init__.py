import time

from .. import schema
from .base import assemble, format_exc

METHODS = ("xgb_cpu", "xgb_gpu", "lgb_cpu", "catboost_gpu", "openboost_gpu", "mangrove")
GPU_METHODS = {"xgb_gpu", "catboost_gpu", "openboost_gpu", "mangrove"}


def paths_rel(p):
    from .. import paths
    return paths.rel_to_bench(p) if p else None


def method_id(method, o):
    P, nt = int(o.get("P") or 1), int(o.get("nthread") or 1)
    if method == "xgb_cpu":
        return f"xgb_cpu_{P}x{nt}"
    if method == "xgb_gpu":
        return ("xgb_gpu_seq" if P == 1 else f"xgb_gpu_mp{P}") + ("_devdata" if o.get("xgb_device_data") else "")
    if method == "lgb_cpu":
        return f"lgb_cpu_{P}x{nt}"
    if method == "catboost_gpu":
        return "catboost_gpu" if o.get("device", "cuda") == "cuda" else "catboost_cpu"
    if method == "openboost_gpu":
        return "openboost_gpu"
    if method == "mangrove":
        mode = o.get("mode", "fused")
        s = method + {"fused": "", "m1": "_m1", "streams": f"_m1_streams{o.get('K', 8)}"}[mode]
        if method == "mangrove":
            if o.get("ablation", "full") != "full":
                s += "_" + o["ablation"]
            if o.get("quant_bits") and int(o["quant_bits"]) != 16:
                s += f"_q{int(o['quant_bits'])}"
            if o.get("memory_budget_gb"):
                s += f"_budget{float(o['memory_budget_gb']):g}g"
            if o.get("backend") == "ref":
                s += "_ref"
        return s
    raise ValueError(f"unknown method {method}")


def defaults(method, o):
    o = dict(o)
    if method in ("xgb_cpu", "lgb_cpu"):
        o.setdefault("P", 20)
        o.setdefault("nthread", 1)
    if method == "xgb_gpu":
        o.setdefault("P", 1)
        o.setdefault("nthread", 1)
    return {k: v for k, v in o.items() if v is not None}


def _dispatch(method, spec, ds, jobs, o, data_dir):
    if method in ("xgb_cpu", "xgb_gpu", "lgb_cpu"):
        from .pool import run_pool
        import importlib
        lib = "lgb" if method == "lgb_cpu" else "xgb"
        device = "cuda" if method == "xgb_gpu" else "cpu"
        from .. import gpumon
        sampler = gpumon.from_opts(o)
        status, timing, per_model, memory, extra = run_pool(lib, spec, ds, jobs, int(o["P"]), int(o["nthread"]), device,
                                                            data_dir=data_dir, torch_guard=o.get("torch_guard", True),
                                                            sampler=sampler, xgb_opts={"device_data": bool(o.get("xgb_device_data"))})
        timing["prep_s"] = (timing.get("prep_s") or 0.0) if status == "ok" else None
        mod = importlib.import_module("lightgbm" if lib == "lgb" else "xgboost")
        r = {"status": status, "timing": timing, "per_model": per_model, "memory": memory,
             "version": {lib: mod.__version__}, "gpu": device == "cuda", "error": extra.get("error"),
             "notes": [] if lib == "xgb" else ["LightGBM grows leaf-wise (num_leaves=2^depth, max_depth=depth): reference only"]}
        if sampler is not None:
            r["gpu_util"] = sampler.summary(paths_rel(o.get("gpu_sample_path")))
        return r
    if method == "mangrove":
        from . import mangrove
        return mangrove.run(spec, ds, jobs, o)
    if method == "catboost_gpu":
        from . import catboost_gpu
        return catboost_gpu.run(spec, ds, jobs, o)
    if method == "openboost_gpu":
        from . import openboost_gpu
        return openboost_gpu.run(spec, ds, jobs, o)
    raise ValueError(method)


def run_static(method, opts, spec, ds, jobs, rep=0, data_dir=None):
    started = schema.now()
    o = defaults(method, opts)
    mid = method_id(method, o)
    variant = {k: v for k, v in o.items() if k not in ("load_s", "gpu_sample_path", "load_detail") and isinstance(v, (int, float, str, bool))}
    empty_mem = {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    empty_t = {"prep_s": None, "warmup_s": None, "train_s": None, "eval_s": None}
    try:
        r = _dispatch(method, spec, ds, jobs, o, data_dir)
    except Exception as e:
        tb = format_exc()
        oom = "out of memory" in (str(e) + tb).lower() or type(e).__name__ in ("OutOfMemoryError", "CUDARuntimeError") \
            and "memory" in str(e).lower()
        r = {"status": "failed", "error": tb, "terminal_reason": "oom" if oom else None}
    status = r.get("status", "failed")
    memory = r.get("memory") or dict(empty_mem)
    if "torch_stats" not in memory:
        from .. import envinfo
        memory["torch_stats"] = envinfo.torch_stats()
    if r.get("gpu_util") is not None:
        memory["smi_peak_mib"] = r["gpu_util"].get("mem_used_max_mib")
        memory["smi_peak_all_mib"] = r["gpu_util"].get("mem_used_max_all_mib")
    if "variant_extra" in r:
        variant.update(r["variant_extra"])
    used_jobs = r.get("jobs_kept", jobs)
    from .base import finalize_test
    y_test = ds.y[ds.test_mask] if getattr(ds, "test_mask", None) is not None else None
    per_config_test = finalize_test(r.get("per_model", []), ds.task, y_test, spec.kind) if y_test is not None else None
    for e in r.get("per_model", []):
        e.pop("_test_margin", None)
    extra = {"n_models_planned": len(jobs)}
    if per_config_test is not None:
        extra["per_config_test"] = per_config_test
    if r.get("gpu_util") is not None:
        extra["gpu_util"] = r["gpu_util"]
    extra["terminal_reason"] = r.get("terminal_reason")
    if o.get("load_detail") and isinstance(r.get("timing"), dict) and method not in ("xgb_cpu", "xgb_gpu", "lgb_cpu"):
        r["timing"].setdefault("details", {}).update(o["load_detail"])
    timing = r.get("timing") or dict(empty_t)
    for k in empty_t:
        timing.setdefault(k, None)
    return assemble(spec=spec, ds=ds, jobs=used_jobs, method=method, method_id=mid, variant=variant,
                    version=r.get("version", {}), rep=rep, status=status, timing=timing,
                    per_model=r.get("per_model", []), memory=memory, started_at=started,
                    gpu=r.get("gpu", method in GPU_METHODS), extra_sources=r.get("extra_sources", ()),
                    notes=r.get("notes"), error=r.get("error"), phases=r.get("phases"),
                    extra=extra)
