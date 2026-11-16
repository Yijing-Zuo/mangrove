import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from bench import schema, paths
from bench.workload import WorkloadSpec, build_jobs
from bench.runners import METHODS, method_id, defaults


def parse_depth(s):
    if s.startswith("fixed"):
        return {"mode": "fixed", "value": int(s[5:] or 6)}
    if s.startswith("mixed"):
        lo, hi = s[5:].split("-")
        return {"mode": "mixed", "lo": int(lo), "hi": int(hi)}
    raise argparse.ArgumentTypeError("depth: fixedN or mixedLO-HI")


def parse_range(s):
    a = [float(x) for x in s.split(",")]
    return [a[0], a[0]] if len(a) == 1 else a[:2]


def _pos_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workload", help="workload JSON (file or inline); flags below override its fields")
    g = ap.add_argument_group("workload")
    g.add_argument("--kind", choices=["hpo_cv", "grouped"])
    g.add_argument("--dataset")
    g.add_argument("--cuts", choices=["fold", "global"])
    g.add_argument("--n-configs", type=int)
    g.add_argument("--config-seed", type=int)
    g.add_argument("--rounds", type=int)
    g.add_argument("--depth", type=parse_depth, help="fixed6 | mixed3-10")
    g.add_argument("--eta-range", type=parse_range)
    g.add_argument("--lam-range", type=parse_range)
    g.add_argument("--mcw-range", type=parse_range)
    g.add_argument("--subsample", type=parse_range, help="lo,hi")
    g.add_argument("--colsample", type=parse_range, help="lo,hi")
    g.add_argument("--folds", help="comma list, default all")
    g.add_argument("--stride", type=int, help="config stride (subset for slow baselines)")
    g.add_argument("--fold-seed", type=int)
    g.add_argument("--row-limit", type=int)
    g.add_argument("--test-frac", type=float, help="outer held-out test fraction, removed before CV")
    g.add_argument("--outer-seed", type=int)
    g.add_argument("--group-val-fold", type=int)
    g.add_argument("--max-groups", type=int)
    g.add_argument("--min-group-rows", type=int)
    g.add_argument("--min-group-size", type=int, help="minimum rows per group over all folds")
    g.add_argument("--tag")
    m = ap.add_argument_group("method options")
    m.add_argument("--P", type=int, help="worker processes (xgb/lgb)")
    m.add_argument("--nthread", type=int)
    m.add_argument("--alloc-cap-factor", type=float,
                   help="mangrove on CUDA: after the 0.75 guard, set_per_process_memory_fraction(min(0.75, F*budget/total)); "
                        "recorded but ignored for other methods")
    m.add_argument("--mode", choices=["fused", "m1", "streams"])
    m.add_argument("--K", type=int, help="CUDA streams for --mode streams")
    m.add_argument("--backend", help="mangrove: gpu|ref")
    m.add_argument("--no-fallback", action="store_true", help="mangrove: do not fall back to backend='ref'")
    m.add_argument("--ablation", help="mangrove: full|no_worklist|no_frontier|no_workspace_pool|wide_acc|quant28|fixed_chunk")
    m.add_argument("--quant-bits", type=int)
    m.add_argument("--memory-budget-gb", type=float)
    m.add_argument("--chunk-bytes", type=float)
    m.add_argument("--chunk-parents", type=int)
    m.add_argument("--no-phases", action="store_true")
    m.add_argument("--group-size", type=int)
    m.add_argument("--time-box-s", type=float)
    m.add_argument("--device", choices=["cuda", "cpu"])
    m.add_argument("--no-torch-guard", action="store_true")
    m.add_argument("--gpu-sample-ms", type=int, help="sample nvidia-smi every N ms during the train scope; raw CSV "
                                                      "is written next to --out as <out>.gpu.csv")
    m.add_argument("--xgb-device-data", action="store_true", help="xgb_gpu: build QuantileDMatrix from CUDA tensors")
    ap.add_argument("--data-dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-data", action="store_true", help="dry-run: missing dataset is an error")
    ap.add_argument("--force", action="store_true")
    return ap


def spec_from_args(a):
    spec = WorkloadSpec.from_json(a.workload) if a.workload else WorkloadSpec()
    d = spec.to_dict()
    mapping = {"kind": "kind", "dataset": "dataset", "cuts": "cuts", "n_configs": "n_configs", "config_seed": "config_seed",
               "rounds": "rounds", "depth": "depth", "subsample": "subsample", "colsample": "colsample",
               "stride": "config_stride", "fold_seed": "fold_seed", "row_limit": "row_limit",
               "test_frac": "test_frac", "outer_seed": "outer_seed",
               "group_val_fold": "group_val_fold", "max_groups": "max_groups", "min_group_rows": "min_group_rows",
               "min_group_size": "min_group_size", "tag": "tag"}
    for k, f in mapping.items():
        v = getattr(a, k)
        if v is not None:
            d[f] = v
    for k in ("eta", "lam", "mcw"):
        v = getattr(a, f"{k}_range")
        if v is not None:
            d["ranges"][k] = v
    if a.folds:
        d["folds"] = [int(x) for x in a.folds.split(",")]
    return WorkloadSpec.from_dict(d)


def opts_from_args(a):
    o = {"P": a.P, "nthread": a.nthread, "mode": a.mode, "K": a.K, "backend": a.backend, "ablation": a.ablation,
         "quant_bits": a.quant_bits, "memory_budget_gb": a.memory_budget_gb, "chunk_bytes": a.chunk_bytes,
         "chunk_parents": a.chunk_parents, "group_size": a.group_size, "time_box_s": a.time_box_s, "device": a.device,
         "alloc_cap_factor": a.alloc_cap_factor}
    o = {k: v for k, v in o.items() if v is not None}
    if a.no_fallback:
        o["allow_fallback"] = False
    if a.no_phases:
        o["phases"] = False
    if a.no_torch_guard:
        o["torch_guard"] = False
    if a.gpu_sample_ms:
        o["gpu_sample_ms"] = a.gpu_sample_ms
        o["gpu_sample_path"] = str(Path(a.out).with_suffix(".gpu.csv"))
    if a.xgb_device_data:
        o["xgb_device_data"] = True
    return o


def validate(argv):
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return False, {"error": "argument parsing failed", "argv": argv}
    info = {"method": a.method, "out": a.out}
    try:
        spec = spec_from_args(a)
    except Exception as e:
        return False, dict(info, error=f"workload: {e}")
    errs = spec.validate()
    if spec.kind == "dynamic":
        errs.append("run_bench.py handles static workloads; use run_dynamic.py")
    o = defaults(a.method, opts_from_args(a))
    try:
        info["method_id"] = method_id(a.method, o)
    except Exception as e:
        errs.append(str(e))
    from bench.data import find_dataset_file
    p, fmt = find_dataset_file(spec.dataset, a.data_dir)
    info["dataset_file"] = None if p is None else str(p)
    info["dataset_format"] = fmt
    if p is None:
        (errs if a.require_data else info.setdefault("warnings", [])).append(f"dataset {spec.dataset} not found")
    info["workload_label"] = spec.label()
    info["errors"] = errs
    return not errs, info


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.dry_run:
        ok, info = validate(argv)
        print(json.dumps(info, indent=1))
        return 0 if ok else 2
    out = Path(a.out)
    if out.exists() and not a.force:
        valid, why = schema.result_is_valid(out)
        if valid:
            print(f"skip: {out} exists ({why})")
            return 0
    spec = spec_from_args(a)
    errs = spec.validate()
    if errs:
        print("invalid workload:", errs, file=sys.stderr)
        return 2
    from bench.data import prepare_dataset, load_dataset
    from bench.runners import run_static
    t0 = time.perf_counter()
    ds0 = load_dataset(spec.dataset, a.data_dir)
    t1 = time.perf_counter()
    ds = prepare_dataset(spec, a.data_dir, ds=ds0)
    jobs = build_jobs(spec, ds)
    t2 = time.perf_counter()
    o = opts_from_args(a)
    o["load_s"] = t2 - t0
    o["load_detail"] = {"load_s": t1 - t0, "binning_s": t2 - t1}
    res = run_static(a.method, o, spec, ds, jobs, rep=a.rep, data_dir=a.data_dir)
    errs = schema.validate(res)
    if errs:
        res.setdefault("notes", []).append("SCHEMA ERRORS: " + "; ".join(errs[:5]))
        res["status"] = "failed"
    schema.write_json_atomic(out, res, overwrite=out.exists())
    tp = res["throughput"]["model_trees_per_s"]
    print(f"{res['method_id']} {res['workload_label']} status={res['status']} models={res['n_models']} "
          f"train_s={res['timing']['train_s']} model-trees/s={tp if tp is None else round(tp, 1)} -> {out}")
    if res["status"] == "failed":
        print(res.get("error") or "", file=sys.stderr)
        return 0 if res.get("terminal_reason") in schema.TERMINAL else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
