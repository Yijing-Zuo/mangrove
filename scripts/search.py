import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from bench import schema
from bench.workload import WorkloadSpec, build_jobs

TRAINERS = ("xgb_cpu", "xgb_gpu", "xgb_gpu_tuned", "lgb_cpu", "mangrove", "mock", "catboost_gpu")
ALGOS = {"random": "grid", "sha": "sha", "asha": "asha", "hyperband-parallel": "hbpar", "hyperband-sequential": "hyperband"}


def build_parser():
    from scripts.fixed import parse_range
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", required=True, choices=TRAINERS)
    ap.add_argument("--algo", required=True, choices=list(ALGOS))
    ap.add_argument("--variant", choices=["promote", "stop"], default=None, help="ASHA variant (default promote)")
    ap.add_argument("--eta", type=int, default=3)
    ap.add_argument("--r-min", type=int, default=10)
    ap.add_argument("--r-max", type=int, default=270)
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--n-configs", type=int, default=729)
    ap.add_argument("--config-seed", type=int, default=0)
    ap.add_argument("--sched-seed", type=int)
    ap.add_argument("--gpu-sample-ms", type=int, help="nvidia-smi sampling during the driver run (<out>.gpu.csv)")
    ap.add_argument("--cuts", choices=["fold", "global"], default="fold")
    ap.add_argument("--depth", default="fixed6")
    ap.add_argument("--subsample", type=parse_range, default=[1.0, 1.0], help="row sampling per config, uniform on lo,hi (default 1,1: off)")
    ap.add_argument("--colsample", type=parse_range, default=[1.0, 1.0], help="colsample_bytree per config, uniform on lo,hi (default 1,1: off)")
    ap.add_argument("--folds")
    ap.add_argument("--row-limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--P", type=int, default=20, help="worker processes (default 20; catboost_gpu: 1)")
    ap.add_argument("--nthread", type=int, default=1)
    ap.add_argument("--slots", type=int, default=1024, help="mangrove: max resident models")
    ap.add_argument("--memory-budget-gb", type=float, default=16.0,
                    help="mangrove managed budget in GiB (default 16); must fit the GPU process allowance")
    ap.add_argument("--no-phases", action="store_true", help="mangrove: Engine(phase_timing=False)")
    ap.add_argument("--alloc-cap-factor", type=float,
                    help="mangrove on CUDA: set_per_process_memory_fraction(min(0.75, F*budget/total)) after the 0.75 "
                         "guard; recorded but ignored for other trainers")
    ap.add_argument("--cb-gpu-ram-part", type=float,
                    help="catboost_gpu: CatBoost gpu_ram_part of each worker (default 0.70 / P; at most 0.75)")
    ap.add_argument("--cb-no-quantize", action="store_true",
                    help="catboost_gpu: keep raw training pools (each fit quantizes, as the static runner) instead of "
                         "quantizing every fold once in the untimed preparation")
    ap.add_argument("--cb-regression-boost-from-average", action="store_true",
                    help="catboost_gpu: boost_from_average True for regression (start at the training mean, as XGBoost "
                         "and Mangrove); default False = runners/catboost_gpu.params()")
    ap.add_argument("--backend")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-data", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def trainer_defaults(a, argv):
    if a.trainer == "catboost_gpu" and not any(x == "--P" or x.startswith("--P=") for x in argv):
        a.P = 1
    return a


def spec_of(a):
    from scripts.fixed import parse_depth
    if a.sched_seed is not None:
        a.config_seed = a.sched_seed
    variant = a.variant or ("promote" if a.algo == "asha" else "sync")
    dyn = {"algo": a.algo, "variant": variant, "eta": a.eta, "r_min": a.r_min, "r_max": a.r_max}
    if a.n_trials:
        dyn["n_trials"] = a.n_trials
    return WorkloadSpec(kind="dynamic", dataset=a.dataset, cuts=a.cuts, n_configs=a.n_configs, config_seed=a.config_seed,
                        depth=parse_depth(a.depth), folds=[int(x) for x in a.folds.split(",")] if a.folds else None,
                        rounds=a.r_max, row_limit=a.row_limit, subsample=a.subsample, colsample=a.colsample, dynamic=dyn, tag=a.tag)


def method_id(a, spec):
    d = spec.dynamic
    algo = d["algo"] + (f"-{d['variant']}" if d["algo"] == "asha" else "")
    tr = {"xgb_cpu": f"xgb_cpu_{a.P}x{a.nthread}", "xgb_gpu": "xgb_gpu_seq" if a.P == 1 else f"xgb_gpu_mp{a.P}",
          "lgb_cpu": f"lgb_cpu_{a.P}x{a.nthread}", "mangrove": f"mangrove_s{a.slots}", "mock": "mock",
          "catboost_gpu": ("catboost_cpu" if a.device == "cpu" else "catboost_gpu") + ("_seq" if a.P == 1 else f"_mp{a.P}"),
          "xgb_gpu_tuned": ("xgb_cpu_tuned" if a.device == "cpu" else "xgb_gpu_tuned") + ("_seq" if a.P == 1 else f"_mp{a.P}")
          }[a.trainer]
    return f"{algo}_{tr}"


def validate(argv):
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return False, {"error": "argument parsing failed"}
    a.algo = ALGOS[a.algo]
    trainer_defaults(a, argv)
    spec = spec_of(a)
    errs = spec.validate()
    if a.trainer == "mangrove" and (not math.isfinite(a.memory_budget_gb) or a.memory_budget_gb <= 0):
        errs.append("mangrove memory budget must be finite and positive")
    if a.alloc_cap_factor is not None and (not math.isfinite(a.alloc_cap_factor) or a.alloc_cap_factor <= 0):
        errs.append("--alloc-cap-factor must be finite and positive")
    if a.trainer == "catboost_gpu":
        if a.P < 1:
            errs.append("catboost_gpu needs --P >= 1")
        if a.cb_gpu_ram_part is not None and not (0 < a.cb_gpu_ram_part <= 0.75):
            errs.append("--cb-gpu-ram-part must be in (0, 0.75]")
    from bench.data import find_dataset_file
    p, fmt = find_dataset_file(spec.dataset, a.data_dir)
    info = {"method_id": method_id(a, spec), "dataset_file": None if p is None else str(p), "errors": errs}
    if a.trainer == "catboost_gpu" and spec.dynamic["algo"] != "grid":
        from bench.dynamic.trainers_catboost import UNSUPPORTED_REASON
        info.setdefault("warnings", []).append("status unsupported: " + UNSUPPORTED_REASON.format(algo={v: k for k, v in ALGOS.items()}[spec.dynamic["algo"]]))
    if p is None and a.trainer != "mock":
        (errs if a.require_data else info.setdefault("warnings", [])).append(f"dataset {spec.dataset} not found")
    return not errs, info


def make_trainer(a, spec, ds):
    o = {"device": a.device, "backend": a.backend, "mock": a.mock, "slots": a.slots}
    if a.trainer == "mangrove":
        o["memory_budget_gb"] = a.memory_budget_gb
    o = {k: v for k, v in o.items() if v not in (None, False)}
    if a.trainer == "mangrove":
        o["phases"] = not a.no_phases
        if a.alloc_cap_factor is not None:
            o["alloc_cap_factor"] = a.alloc_cap_factor
    from bench.dynamic import trainers as TR
    if a.trainer == "mock":
        return TR.MockTrainer(W=a.P)
    if a.trainer == "xgb_gpu_tuned":
        from bench.dynamic.trainers_tuned import TunedXGBTrainer
        return TunedXGBTrainer(spec, ds, a.P, a.nthread, "cpu" if a.device == "cpu" else "cuda", data_dir=a.data_dir)
    if a.trainer == "catboost_gpu":
        from bench.dynamic.trainers_catboost import CatBoostTrainer
        return CatBoostTrainer(spec, a.P, a.nthread, "cpu" if a.device == "cpu" else "cuda", data_dir=a.data_dir,
                               gpu_ram_part=a.cb_gpu_ram_part, quantize=not a.cb_no_quantize,
                               regression_bfa=a.cb_regression_boost_from_average)
    if a.trainer in ("xgb_cpu", "xgb_gpu", "lgb_cpu"):
        lib = "lgb" if a.trainer == "lgb_cpu" else "xgb"
        return TR.PoolTrainer(lib, spec, a.P, a.nthread, "cuda" if a.trainer == "xgb_gpu" else "cpu", data_dir=a.data_dir)
    return TR.MangroveTrainer(spec, ds, o, spec.dynamic["r_max"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(argv)
    a.algo = ALGOS[a.algo]
    a = trainer_defaults(a, argv)
    if a.dry_run:
        ok, info = validate(argv)
        print(json.dumps(info, indent=1))
        return 0 if ok else 2
    out = Path(a.out)
    if out.exists() and not a.force and schema.result_is_valid(out)[0]:
        print(f"skip: {out} exists")
        return 0
    ok, info = validate(argv)
    if not ok:
        print(info, file=sys.stderr)
        return 2
    spec = spec_of(a)
    from bench.data import prepare_dataset, make_synthetic
    from bench.dynamic.driver import Driver
    from bench.runners.base import assemble, format_exc
    from bench import envinfo
    started = schema.now()
    t0 = time.perf_counter()
    ds = make_synthetic(n=300, d=4) if (a.trainer == "mock" and a.data_dir is None and a.mock) else prepare_dataset(spec, a.data_dir)
    jobs = build_jobs(spec, ds)
    load_s = time.perf_counter() - t0
    by_cfg = {}
    for j in jobs:
        by_cfg.setdefault(j.cfg, []).append(j)
    status, err, run, tr = "ok", None, None, None
    if a.trainer == "catboost_gpu" and spec.dynamic["algo"] != "grid":
        from bench.dynamic.trainers_catboost import UNSUPPORTED_REASON
        status, err = "unsupported", UNSUPPORTED_REASON.format(algo={v: k for k, v in ALGOS.items()}[spec.dynamic["algo"]])
    gpu = a.trainer in ("xgb_gpu", "xgb_gpu_tuned", "mangrove", "catboost_gpu") and a.device != "cpu"
    from bench import gpumon
    sampler = gpumon.GpuSampler(out.with_suffix(".gpu.csv"), a.gpu_sample_ms) if (a.gpu_sample_ms and status == "ok") else None
    try:
        if status == "ok":
            tr = make_trainer(a, spec, ds)
            if a.trainer == "mangrove":
                gpu = tr.gpu
            envinfo.gpu_reset_peak()
            if sampler is not None:
                sampler.start()
            w0 = time.time()
            run = Driver(tr, by_cfg, ds.task, spec.dynamic).run()
            if sampler is not None:
                sampler.mark_window(w0, time.time())
    except Exception:
        status, err = "failed", format_exc()
    finally:
        if sampler is not None:
            sampler.stop()
        if tr is not None:
            tr.close()
    timing = {"prep_s": load_s + (tr.prep_s if tr else 0.0), "warmup_s": tr.warmup_s if tr else None,
              "train_s": run["wall_s"] if run else None, "eval_s": 0.0,
              "details": {"note": "dynamic workloads: validation scoring is part of the driver loop for every method "
                                  "(decisions need it); train_s = driver wall time"}}
    if a.trainer == "xgb_gpu_tuned" and tr is not None:
        timing["details"]["breakdown"] = tr.breakdown(run["wall_s"] if run else None)
    per_model, used = [], []
    drv_trials = {}
    if run:
        tr_stats = run["trainer"]
        for e in run["trace"]:
            drv_trials[e["trial"]] = max(drv_trials.get(e["trial"], 0), e["rounds"])
        for j in jobs:
            if j.cfg in drv_trials:
                used.append(j)
    memory = envinfo.gpu_peak() if gpu else {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    memory["torch_stats"] = envinfo.torch_stats()
    extra_sources = []
    if a.trainer == "mangrove":
        from bench.runners.mangrove import source_files
        extra_sources = source_files()
    res = assemble(spec=spec, ds=ds, jobs=used, method=f"dynamic_{a.trainer}", method_id=method_id(a, spec),
                   variant={"trainer": a.trainer, "P": a.P, "nthread": a.nthread, "slots": a.slots,
                            "memory_budget_gb": a.memory_budget_gb if a.trainer == "mangrove" else None,
                            "sched_seed": spec.config_seed,
                            "phases": (not a.no_phases) if a.trainer == "mangrove" else None,
                            "alloc_cap_factor": a.alloc_cap_factor,
                            "alloc_fraction_effective": getattr(tr, "alloc_fraction", None),
                            "continuation": ("serialize_reload" if a.trainer in ("xgb_cpu", "xgb_gpu", "lgb_cpu")
                                             else f"resident_booster_{getattr(tr, 'cache', None)}" if a.trainer == "xgb_gpu_tuned"
                                             else "none_grid_only" if a.trainer == "catboost_gpu"
                                             else None),
                            **({"cb_gpu_ram_part": getattr(tr, "gpu_ram_part", None), "cb_quantize": not a.cb_no_quantize,
                                "cb_quantize_effective": getattr(tr, "quantize_effective", None),
                                "cb_regression_boost_from_average": a.cb_regression_boost_from_average}
                               if a.trainer == "catboost_gpu" else {}),
                            **spec.dynamic},
                   version=({"catboost": getattr(tr, "version", None)} if a.trainer == "catboost_gpu" else {}),
                   rep=a.rep, status=status, timing=timing, per_model=[], memory=memory, started_at=started,
                   gpu=gpu, error=err, trace=run["trace"] if run else None, extra_sources=extra_sources,
                   notes=(getattr(tr, "result_notes", None) if a.trainer == "catboost_gpu" else None))
    if run:
        trace = run["trace"]
        final = [e for e in trace if e["rounds"] == run["rungs"][-1]]
        best = min(final, key=lambda e: (e["score"], e["trial"])) if final else None
        res["per_model"] = run["per_model"]
        res["throughput"]["model_rounds_executed"] = int(run["model_rounds_executed"])
        res["throughput"]["model_rounds_per_s"] = run["model_rounds_executed"] / run["wall_s"] if run["wall_s"] else None
        if run.get("occupancy"):
            res["occupancy"] = run["occupancy"]
        if run.get("brackets"):
            res["brackets"] = run["brackets"]
        primary = {"binary": "auc", "regression": "rmse", "multiclass": "mlogloss"}[ds.task]
        fin_m = [e for e in final if (e.get("metrics_mean") or {}).get(primary) is not None]
        sign = 1 if primary == "auc" else -1
        best_p = max(fin_m, key=lambda e: sign * e["metrics_mean"][primary]) if fin_m else None
        res["dynamic_summary"] = {"rungs": run["rungs"], "n_trials": run["n_trials"], "trainer": tr_stats,
                                  "primary_metric": primary,
                                  "final_best_primary": best_p["metrics_mean"][primary] if best_p else None,
                                  "final_primary_of_loss_selected": (best["metrics_mean"] or {}).get(primary) if best else None,
                                  "final_best_score": best["score"] if best else None,
                                  "final_best_trial": best["trial"] if best else None,
                                  "time_to_final_best_s": best["t"] if best else None,
                                  "n_reached_max_rung": len(final)}
    if sampler is not None:
        from bench import paths as _p
        res["gpu_util"] = sampler.summary(_p.rel_to_bench(out.with_suffix(".gpu.csv")))
        res["memory"]["smi_peak_mib"] = res["gpu_util"].get("mem_used_max_mib")
        res["memory"]["smi_peak_all_mib"] = res["gpu_util"].get("mem_used_max_all_mib")
    errs = schema.validate(res)
    if errs:
        res["status"] = "failed"
        res.setdefault("notes", []).append("SCHEMA ERRORS: " + "; ".join(errs[:5]))
    schema.write_json_atomic(out, res, overwrite=out.exists())
    s = res.get("dynamic_summary", {})
    print(f"{res['method_id']} {spec.dataset} status={res['status']} wall={timing['train_s']} "
          f"model-rounds={res['throughput'].get('model_rounds_executed')} best={s.get('final_best_score')} -> {out}")
    if res["status"] == "failed":
        print(err, file=sys.stderr)
        return 1
    if res["status"] == "unsupported":
        print(err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
