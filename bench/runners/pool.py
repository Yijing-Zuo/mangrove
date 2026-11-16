import os
import sys
import time
import multiprocessing as mp

import numpy as np

from .. import envinfo


def xgb_params(job, task, K, nthread, device, base):
    cf = job.params
    p = dict(tree_method="hist", max_bin=256, eta=cf["eta"], reg_lambda=cf["lam"], min_child_weight=cf["mcw"],
             max_depth=cf["max_depth"], reg_alpha=0.0, gamma=0.0, subsample=cf["subsample"],
             colsample_bytree=cf["colsample"], grow_policy="depthwise", nthread=nthread, device=device,
             verbosity=0, seed=int(job.seed))
    if task == "binary":
        p.update(objective="binary:logistic", base_score=0.5)
    elif task == "regression":
        p.update(objective="reg:squarederror", base_score=float(base))
    else:
        p.update(objective="multi:softprob", num_class=K, base_score=0.5)
    return p


def lgb_params(job, task, K, nthread):
    cf = job.params
    obj = {"binary": "binary", "regression": "regression", "multiclass": "multiclass"}[task]
    p = {"objective": obj, "learning_rate": cf["eta"], "lambda_l2": cf["lam"], "min_sum_hessian_in_leaf": cf["mcw"],
         "max_depth": cf["max_depth"], "num_leaves": 2 ** cf["max_depth"], "min_data_in_leaf": 1,
         "num_threads": nthread, "verbose": -1, "max_bin": 256, "force_col_wise": True,
         "bagging_fraction": cf["subsample"], "bagging_freq": 1 if cf["subsample"] < 1 else 0,
         "feature_fraction": cf["colsample"], "seed": int(job.seed), "deterministic": True}
    if task == "multiclass":
        p["num_class"] = K
    return p


def xgb_part_mats(xgb, ds, j, nthread, device, xgb_opts):
    X = ds.X(j.cohort)
    if device == "cuda" and xgb_opts.get("device_data"):
        Xt, yt = _device_arrays(X[j.train_mask], ds.y[j.train_mask])
        dtr = xgb.QuantileDMatrix(Xt, label=yt, max_bin=256)
    else:
        dtr = xgb.QuantileDMatrix(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask],
                                  max_bin=256, nthread=nthread)
    dva = xgb.DMatrix(X[j.val_mask].astype(np.float32), nthread=nthread)
    dte = xgb.DMatrix(X[j.test_mask].astype(np.float32), nthread=nthread) if j.test_mask is not None else None
    return dtr, dva, dte


def eval_entry(lib, ds, j, bst, T, dva, dte, train_job_s):
    from bench import metrics as M
    task, K = ds.task, ds.K
    mg = bst.predict(dva, output_margin=True) if lib == "xgb" else bst.predict(dva, raw_score=True)
    if task == "multiclass":
        mg = np.asarray(mg).reshape(-1, K)
    e = {"mid": j.mid, "cfg": j.cfg, "part": j.part, "seed": j.seed, "rounds": T, "train_job_s": train_job_s,
         "metrics": M.metrics(task, ds.y[j.val_mask], mg), "margin_sha256": M.margin_hash(mg)}
    if dte is not None:
        mt = bst.predict(dte, output_margin=True) if lib == "xgb" else bst.predict(dte, raw_score=True)
        mt = np.asarray(mt, dtype=np.float32)
        if task == "multiclass":
            mt = mt.reshape(-1, K)
        e["test_metrics"] = M.metrics(task, ds.y[j.test_mask], mt)
        e["_test_margin"] = mt
    return e


def _setup(spec_dict, data_dir):
    from bench.workload import WorkloadSpec, build_jobs
    from bench.data import prepare_dataset
    spec = WorkloadSpec.from_dict(spec_dict)
    ds = prepare_dataset(spec, data_dir)
    jobs = {j.mid: j for j in build_jobs(spec, ds)}
    return spec, ds, jobs


def worker_main(wid, lib, spec_dict, data_dir, nthread, device, jobs_q, res_q, b1, b2, torch_guard, xgb_opts=None):
    try:
        _worker(wid, lib, spec_dict, data_dir, nthread, device, jobs_q, res_q, b1, b2, torch_guard, xgb_opts or {})
    except Exception:
        import traceback
        res_q.put({"wid": wid, "error": traceback.format_exc()[-4000:]})
        for b in (b1, b2):
            try:
                b.abort()
            except Exception:
                pass


def _device_arrays(X, y):
    import torch
    return (torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device="cuda"),
            torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32, device="cuda"))


def _worker(wid, lib, spec_dict, data_dir, nthread, device, jobs_q, res_q, b1, b2, torch_guard, xgb_opts):
    from bench import envinfo, metrics as M
    guard = None
    if device == "cuda" and torch_guard:
        guard = envinfo.gpu_process_guard(required=False)
    t0 = time.perf_counter()
    spec, ds, jobs = _setup(spec_dict, data_dir)
    t_data = time.perf_counter()
    T, task, K = spec.rounds, ds.task, ds.K
    parts = {}
    for j in jobs.values():
        parts.setdefault(j.part, j)
    dtr, dva, dte, base = {}, {}, {}, {}
    if lib == "xgb":
        import xgboost as xgb
        for part, j in parts.items():
            base[part] = ds.base_margin(j.train_mask)
            dtr[part], dva[part], te = xgb_part_mats(xgb, ds, j, nthread, device, xgb_opts)
            if te is not None:
                dte[part] = te
    else:
        import lightgbm as lgb
        for part, j in parts.items():
            X = ds.X(j.cohort)
            dtr[part] = lgb.Dataset(X[j.train_mask].astype(np.float32), label=ds.y[j.train_mask],
                                    params={"max_bin": 256, "verbose": -1, "num_threads": nthread,
                                            "min_data_in_leaf": 1, "feature_pre_filter": False}, free_raw_data=False)
            dtr[part].construct()
            dva[part] = X[j.val_mask].astype(np.float32)
            if j.test_mask is not None:
                dte[part] = X[j.test_mask].astype(np.float32)
    t1 = time.perf_counter()
    for part, j in parts.items():
        if lib == "xgb":
            import xgboost as xgb
            xgb.train(xgb_params(j, task, K, nthread, device, base[part]), dtr[part], 2)
        else:
            import lightgbm as lgb
            lgb.train(lgb_params(j, task, K, nthread), dtr[part], 2)
    j0 = next(iter(parts.values()))
    probe = []
    for _ in range(3):
        tp0 = time.perf_counter()
        if lib == "xgb":
            xgb.train(xgb_params(j0, task, K, nthread, device, base[j0.part]), dtr[j0.part], 1)
        else:
            lgb.train(lgb_params(j0, task, K, nthread), dtr[j0.part], 1)
        probe.append(time.perf_counter() - tp0)
    lib_params = (xgb_params(j0, task, K, nthread, device, base[j0.part]) if lib == "xgb" else lgb_params(j0, task, K, nthread))
    t2 = time.perf_counter()
    b1.wait()
    ts = time.perf_counter()
    boosters = []
    while True:
        mid = jobs_q.get()
        if mid is None:
            break
        j = jobs[mid]
        if lib == "xgb":
            bst = xgb.train(xgb_params(j, task, K, nthread, device, base[j.part]), dtr[j.part], T)
        else:
            bst = lgb.train(lgb_params(j, task, K, nthread), dtr[j.part], T)
        boosters.append((mid, bst, time.perf_counter()))
    te = time.perf_counter()
    b2.wait()
    es = time.perf_counter()
    out = []
    prev = ts
    for mid, bst, tend in boosters:
        j = jobs[mid]
        out.append(eval_entry(lib, ds, j, bst, T, dva[j.part], dte.get(j.part), tend - prev))
        prev = tend
    ee = time.perf_counter()
    mem = None
    if guard and guard.get("cuda"):
        mem = envinfo.gpu_peak()
    res_q.put({"wid": wid, "prep_s": t1 - t0, "data_s": t_data - t0, "dmatrix_s": t1 - t_data, "warmup_s": t2 - t1,
               "train_start": ts, "train_end": te,
               "per_call_1round_s": float(np.median(probe)), "lib_params": {k: v for k, v in lib_params.items()},
               "eval_start": es, "eval_end": ee, "n_jobs": len(boosters), "jobs": out,
               "host_peak_rss_bytes": envinfo.host_peak_rss(), "gpu_mem": mem, "guard": guard,
               "torch_stats": envinfo.torch_stats()})


def run_pool(lib, spec, ds, jobs, P, nthread, device, data_dir=None, torch_guard=True, timeout_s=6 * 3600, sampler=None,
             xgb_opts=None):
    ctx = mp.get_context("spawn")
    jobs_q, res_q = ctx.Queue(), ctx.Queue()
    b1 = ctx.Barrier(P + 1, timeout=timeout_s)
    b2 = ctx.Barrier(P + 1, timeout=timeout_s)
    for j in jobs:
        jobs_q.put(j.mid)
    for _ in range(P):
        jobs_q.put(None)
    spec_dict = spec.to_dict()
    t_launch = time.perf_counter()
    procs = [ctx.Process(target=worker_main, args=(w, lib, spec_dict, None if data_dir is None else str(data_dir),
                                                   nthread, device, jobs_q, res_q, b1, b2, torch_guard, xgb_opts), daemon=False)
             for w in range(P)]
    for p in procs:
        p.start()
    errors = []
    if sampler is not None:
        sampler.start()
    try:
        b1.wait()
        w0 = time.time()
        b2.wait()
        w1 = time.time()
        if sampler is not None:
            sampler.mark_window(w0, w1)
    except Exception as e:
        errors.append(f"barrier: {e!r}")
    finally:
        if sampler is not None:
            sampler.stop()
    res = []
    deadline = time.time() + (60 if errors else timeout_s)
    while len(res) < P and time.time() < deadline:
        try:
            res.append(res_q.get(timeout=5))
        except Exception:
            if all(not p.is_alive() for p in procs) and res_q.empty():
                break
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    errors += [r["error"] for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    if errors or len(ok) < P:
        return "failed", {"prep_s": None, "warmup_s": None, "train_s": None, "eval_s": None,
                          "details": {"launch_to_end_s": time.perf_counter() - t_launch}}, [], \
            {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}, {"error": "\n".join(errors)[-6000:] or "worker lost"}
    per_model = sorted([j for r in ok for j in r["jobs"]], key=lambda e: (e["cfg"], e["part"]))
    train_s = max(r["train_end"] for r in ok) - min(r["train_start"] for r in ok)
    eval_s = max(r["eval_end"] for r in ok) - min(r["eval_start"] for r in ok)
    timing = {"prep_s": max(r["prep_s"] for r in ok), "warmup_s": max(r["warmup_s"] for r in ok),
              "train_s": train_s, "eval_s": eval_s,
              "details": {"worker_train_spans_s": [r["train_end"] - r["train_start"] for r in ok],
                          "worker_jobs": [r["n_jobs"] for r in ok],
                          "worker_prep_s": [r["prep_s"] for r in ok],
                          "load_and_binning_s": max(r["data_s"] for r in ok),
                          "dmatrix_build_s": max(r["dmatrix_s"] for r in ok),
                          "mean_job_train_s": float(np.mean([e["train_job_s"] for e in per_model])) if per_model else None,
                          "per_call_1round_s_median": float(np.median([r["per_call_1round_s"] for r in ok])),
                          "library_params": ok[0]["lib_params"],
                          "launch_to_end_s": time.perf_counter() - t_launch}}
    gm = [r["gpu_mem"] for r in ok if r.get("gpu_mem")]
    memory = {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
              "host_peak_rss_bytes": max((r["host_peak_rss_bytes"] or 0) for r in ok),
              "host_peak_rss_sum_bytes": sum((r["host_peak_rss_bytes"] or 0) for r in ok),
              "note": "library GPU memory is not tracked by torch; torch guard only" if device == "cuda" else None}
    if gm:
        memory["torch_guard_peak_allocated_bytes"] = max(g["gpu_peak_allocated_bytes"] or 0 for g in gm)
    memory["torch_stats"] = envinfo.merge_torch_stats([r.get("torch_stats") for r in ok])
    return "ok", timing, per_model, memory, {"guards": [r.get("guard") for r in ok][:1]}
