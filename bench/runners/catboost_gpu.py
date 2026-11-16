import time

import numpy as np

from .. import envinfo
from .base import per_model_entry


def params(job, task, K, device, T, nthread):
    cf = job.params
    p = dict(iterations=T, learning_rate=cf["eta"], depth=min(cf["max_depth"], 16), l2_leaf_reg=cf["lam"],
             border_count=254, random_seed=int(job.seed), verbose=False, allow_writing_files=False,
             task_type="GPU" if device == "cuda" else "CPU", thread_count=nthread, boost_from_average=False)
    if cf["subsample"] < 1.0:
        p.update(bootstrap_type="Bernoulli", subsample=cf["subsample"])
    else:
        p.update(bootstrap_type="No")
    if device != "cuda" and cf["colsample"] < 1.0:
        p["rsm"] = cf["colsample"]
    p["loss_function"] = {"binary": "Logloss", "regression": "RMSE", "multiclass": "MultiClass"}[task]
    if task == "multiclass":
        p["classes_count"] = K
    return p


def run(spec, ds, jobs, opts):
    from catboost import CatBoost, Pool
    import catboost
    device = opts.get("device", "cuda")
    nthread = int(opts.get("nthread", 20 if device == "cuda" else 1))
    if device == "cuda":
        envinfo.gpu_process_guard(required=False)
    task, K, T = ds.task, ds.K, spec.rounds
    timing = {}
    t0 = time.perf_counter()
    pools, Xva, Xte = {}, {}, {}
    for j in jobs:
        if j.part in pools:
            continue
        X = ds.X(j.cohort)
        pools[j.part] = Pool(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask])
        Xva[j.part] = Pool(X[j.val_mask].astype(np.float32))
        if j.test_mask is not None:
            Xte[j.part] = Pool(X[j.test_mask].astype(np.float32))
    timing["prep_s"] = time.perf_counter() - t0 + opts.get("load_s", 0.0)
    t0 = time.perf_counter()
    CatBoost(params(jobs[0], task, K, device, 2, nthread)).fit(pools[jobs[0].part])
    timing["warmup_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    models = []
    for j in jobs:
        m = CatBoost(params(j, task, K, device, T, nthread))
        m.fit(pools[j.part])
        models.append(m)
    timing["train_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    per_model = []
    for j, m in zip(jobs, models):
        mg = np.asarray(m.predict(Xva[j.part], prediction_type="RawFormulaVal"))
        kw = {}
        if j.part in Xte:
            kw = {"y_test": ds.y[j.test_mask], "test_margin": np.asarray(m.predict(Xte[j.part], prediction_type="RawFormulaVal"))}
        per_model.append(per_model_entry(j, task, ds.y[j.val_mask], mg, T, **kw))
    timing["eval_s"] = time.perf_counter() - t0
    notes = ["CatBoost grows oblivious (symmetric) trees: reference only, not the same algorithm"]
    if device == "cuda" and any(j.params["colsample"] < 1 for j in jobs):
        notes.append("rsm/colsample ignored on GPU")
    return {"status": "ok", "timing": timing, "per_model": per_model,
            "memory": {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
                       "host_peak_rss_bytes": envinfo.host_peak_rss()},
            "version": {"catboost": catboost.__version__}, "gpu": device == "cuda", "notes": notes}
