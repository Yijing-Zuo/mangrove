import traceback

from .. import schema, envinfo, metrics as M
from ..workload import config_list_hash


def per_model_entry(job, task, y_val, margin, rounds, extra=None, y_test=None, test_margin=None):
    e = {"mid": job.mid, "cfg": job.cfg, "part": job.part, "seed": job.seed, "rounds": int(rounds),
         "metrics": M.metrics(task, y_val, margin), "margin_sha256": M.margin_hash(margin)}
    if test_margin is not None:
        e["test_metrics"] = M.metrics(task, y_test, test_margin)
        e["_test_margin"] = test_margin
    if extra:
        e.update(extra)
    return e


def finalize_test(per_model, task, y_test, kind):
    import numpy as np
    by = {}
    for e in per_model:
        tm = e.pop("_test_margin", None)
        if tm is not None:
            by.setdefault(e["cfg"], []).append((tm, e["test_metrics"]))
    if not by or kind != "hpo_cv":
        return None
    out = []
    for c in sorted(by):
        ms = [np.asarray(m, dtype=np.float64) for m, _ in by[c]]
        ens = M.metrics(task, y_test, np.mean(ms, axis=0))
        keys = by[c][0][1].keys()
        mean_fold = {k: float(np.mean([t[k] for _, t in by[c]])) for k in keys}
        out.append({"cfg": int(c), "n_folds": len(ms), "ensemble": ens, "mean_fold": mean_fold})
    return out


def assemble(*, spec, ds, jobs, method, method_id, variant, version, rep, status, timing, per_model, memory,
             started_at, gpu, extra_sources=(), notes=None, error=None, phases=None, trace=None, extra=None):
    T = spec.rounds
    n_models = len(jobs)
    train_s = timing.get("train_s")
    mts = (n_models * T / train_s) if (train_s and status in ("ok", "partial")) else None
    if spec.kind == "dynamic":
        mts = None
    timing = dict(timing)
    timing.setdefault("details", {})
    timing["total_s"] = sum(v for k, v in timing.items() if k.endswith("_s") and k != "total_s" and isinstance(v, (int, float)))
    res = {
        "schema_version": schema.SCHEMA_VERSION,
        "status": status,
        "method": method,
        "method_id": method_id,
        "variant": variant,
        "version": version,
        "dataset": ds.describe() if ds is not None else {"name": spec.dataset, "task": "binary", "n": 0, "d": 0},
        "workload": spec.to_dict(),
        "workload_kind": spec.kind,
        "workload_id": spec.workload_id(),
        "workload_label": spec.label(),
        "config_list_hash": config_list_hash(jobs) if jobs else "",
        "n_models": n_models,
        "rounds": T,
        "trees_per_round": int(ds.K) if ds is not None else 1,
        "timing": timing,
        "throughput": {"model_trees_per_s": mts,
                       "physical_trees_per_s": (mts * (ds.K if ds is not None else 1)) if mts else None,
                       "model_rounds_executed": n_models * T if status == "ok" and spec.kind != "dynamic" else None},
        "per_model": per_model,
        "memory": memory,
        "environment": envinfo.environment(gpu=gpu),
        "exclusivity": envinfo.exclusivity_from_env(),
        "repetition": int(rep),
        "cooling": envinfo.cooling_from_env(),
        "source_hashes": envinfo.source_hashes(extra_sources),
        "started_at": started_at,
        "finished_at": schema.now(),
        "notes": list(notes or []),
        "error": error,
    }
    if phases is not None:
        res["phases"] = phases
    if trace is not None:
        res["trace"] = trace
    if extra:
        res.update(extra)
    return res


def unsupported(reason):
    return {"status": "unsupported", "error": reason}


def format_exc():
    return traceback.format_exc()[-4000:]
