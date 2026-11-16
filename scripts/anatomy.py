import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MANGROVE_BENCH_DATA", str(HERE.parent / "data")))
sys.path.insert(0, str(HERE.parent))

LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")
COPY = ("cudaMemcpyAsync", "cudaMemcpy", "cuMemcpyAsync", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cudaMemsetAsync", "cudaMemset", "cuMemsetD8Async", "cuMemsetD32Async")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cuStreamSynchronize",
        "cuEventSynchronize", "cudaEventQuery", "cudaStreamWaitEvent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(HERE.parent))
    ap.add_argument("--nodemap", default=None)
    ap.add_argument("--data", default="higgs-1m")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--configs", type=int, default=1, help="configs x folds models (default 1 x fold 0: M=1)")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.engine).resolve()))
    import numpy as np
    import torch
    from torch.profiler import profile, ProfilerActivity
    from mangrove.engine import Engine, Cohort
    from mangrove.registry import ModelSpec
    from bench.workload import WorkloadSpec, sample_configs
    torch.cuda.set_per_process_memory_fraction(0.75)
    z = np.load(DATA / f"{a.data}.npz", allow_pickle=False)
    meta = json.loads((DATA / f"{a.data}.json").read_text())
    task = meta["task"]
    loss = {"binary": "logistic", "regression": "squared"}[task]
    Xf, y, folds, nbf = z["Xb_fold"], z["y"], z["folds"].astype(np.int64), z["nbins_fold"]
    cfgs = sample_configs(WorkloadSpec(dataset=a.data, n_configs=max(a.configs, 1), rounds=100))
    p = cfgs[0]
    fold_ids = [int(x) for x in a.folds.split(",")]
    cohorts = [Cohort(k, np.ascontiguousarray(Xf[k]), y.astype(np.float32), nbins=nbf[k].astype(np.int64))
               for k in fold_ids]
    kw = {} if a.nodemap is None else dict(nodemap=bool(int(a.nodemap)))
    eng = Engine(cohorts, loss, device="cuda", backend="gpu", budget_bytes=10 << 30, phase_timing=False,
                 collect_metrics=False, **kw)
    specs = []
    for c in range(a.configs):
        q = cfgs[c]
        for k in fold_ids:
            tr = folds != k
            specs.append(ModelSpec(data_key=k, train_rows=np.nonzero(tr)[0], eval_rows=np.nonzero(folds == k)[0],
                                   eta=q["eta"], lam=q["lam"], mcw=q["mcw"], max_depth=q["max_depth"], n_rounds=1000,
                                   subsample=q["subsample"], colsample_bytree=q["colsample"], seed=1000 + c * 5 + k,
                                   base_margin=float(np.mean(y[tr])) if task == "regression" else 0.0))
    eng.add_models(specs)
    for _ in range(a.warm):
        eng.step()
    torch.cuda.synchronize()
    R = a.rounds
    t0 = time.perf_counter()
    for _ in range(R):
        eng.step()
    torch.cuda.synchronize()
    wall_np = (time.perf_counter() - t0) * 1e3 / R
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(R):
            eng.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3 / R
    ev = prof.events()
    host = {"launch": 0.0, "copy": 0.0, "sync": 0.0}
    cnt = {"launch": 0, "copy": 0, "sync": 0}
    gk = gc = 0.0
    nk = 0
    for e in ev:
        dur = e.time_range.elapsed_us() / 1e3
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            n = e.name.lower()
            if n.startswith("memcpy") or n.startswith("memset") or "memcpy" in n or "memset" in n:
                gc += dur
            else:
                gk += dur
                nk += 1
            continue
        for kind, names in (("launch", LAUNCH), ("copy", COPY), ("sync", SYNC)):
            if e.name in names:
                host[kind] += dur
                cnt[kind] += 1
    rec = dict(engine=a.engine, nodemap=a.nodemap, data=a.data, rounds=R, max_depth=p["max_depth"],
               wall_ms=wall, wall_noprof_ms=wall_np, gpu_kernel_ms=gk / R, gpu_copy_ms=gc / R, kernels=nk / R,
               host_launch_ms=host["launch"] / R, host_copy_ms=host["copy"] / R, host_sync_ms=host["sync"] / R,
               host_other_ms=wall - sum(host.values()) / R,
               launch_calls=cnt["launch"] / R, copy_calls=cnt["copy"] / R, sync_calls=cnt["sync"] / R,
               nm_stats=getattr(eng, "nm_stats", None))
    if hasattr(eng, "nm_check"):
        eng.nm_check()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
