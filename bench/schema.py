import json
import math
import os
import time
from pathlib import Path

SCHEMA_VERSION = "mangrove-bench-result/1"
STATUSES = ["ok", "failed", "unsupported", "partial"]
TERMINAL = ("oom", "timeout", "time_box")

_num = {"type": ["number", "null"]}
_int = {"type": ["integer", "null"]}
_obj = {"type": "object"}

RESULT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "bench result",
    "type": "object",
    "required": ["schema_version", "status", "method", "method_id", "variant", "version", "dataset", "workload",
                 "workload_kind", "workload_id", "workload_label", "config_list_hash", "n_models", "rounds",
                 "trees_per_round", "timing", "throughput", "per_model", "memory", "environment", "exclusivity",
                 "repetition", "cooling", "source_hashes", "started_at", "finished_at"],
    "properties": {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "status": {"type": "string", "enum": STATUSES},
        "method": {"type": "string"},
        "method_id": {"type": "string"},
        "variant": _obj,
        "version": _obj,
        "dataset": {"type": "object", "required": ["name", "task", "n", "d"],
                    "properties": {"name": {"type": "string"}, "task": {"type": "string", "enum": ["binary", "regression", "multiclass"]},
                                   "n": {"type": "integer"}, "d": {"type": "integer"}}},
        "workload": _obj,
        "workload_kind": {"type": "string", "enum": ["hpo_cv", "grouped", "dynamic"]},
        "workload_id": {"type": "string"},
        "workload_label": {"type": "string"},
        "config_list_hash": {"type": "string"},
        "n_models": {"type": "integer", "minimum": 0},
        "rounds": {"type": "integer", "minimum": 0},
        "trees_per_round": {"type": "integer", "minimum": 1},
        "timing": {"type": "object", "required": ["prep_s", "warmup_s", "train_s", "eval_s"],
                   "properties": {"prep_s": _num, "warmup_s": _num, "train_s": _num, "eval_s": _num,
                                  "total_s": _num, "details": _obj}},
        "throughput": {"type": "object", "required": ["model_trees_per_s"],
                       "properties": {"model_trees_per_s": _num, "physical_trees_per_s": _num, "model_rounds_executed": _int}},
        "per_model": {"type": "array", "items": {"type": "object", "required": ["cfg", "part", "metrics"],
                                                 "properties": {"cfg": {"type": "integer"}, "part": {"type": "integer"},
                                                                "metrics": _obj, "rounds": _int}}},
        "memory": {"type": "object", "required": ["gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"],
                   "properties": {"gpu_peak_allocated_bytes": _int, "gpu_peak_reserved_bytes": _int,
                                  "host_peak_rss_bytes": _int}},
        "environment": _obj,
        "exclusivity": _obj,
        "repetition": {"type": "integer", "minimum": 0},
        "cooling": _obj,
        "source_hashes": _obj,
        "started_at": {"type": "string"},
        "finished_at": {"type": "string"},
        "trace": {"type": "array"},
        "phases": _obj,
        "error": {"type": ["string", "null"]},
        "notes": {"type": "array", "items": {"type": "string"}},
        "terminal_reason": {"type": ["string", "null"]},
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(obj, schema=RESULT_SCHEMA, path="$"):
    errs = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_is_type(obj, x) for x in ts):
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if "minimum" in schema and isinstance(obj, (int, float)) and obj < schema["minimum"]:
        errs.append(f"{path}: {obj} < {schema['minimum']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing required '{k}'")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs.extend(validate(obj[k], sub, f"{path}.{k}"))
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj[:5000]):
            errs.extend(validate(v, schema["items"], f"{path}[{i}]"))
    return errs


def check_finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: check_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [check_finite(v) for v in obj]
    return obj


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def result_is_valid(path, accept=("ok", "unsupported")):
    p = Path(path)
    if not p.exists():
        return False, "missing"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"unparseable: {e!r}"
    sv = str(obj.get("schema_version", ""))
    if sv.startswith("bench-terminal/"):
        return True, f"terminal:{obj.get('terminal_reason')}"
    if sv.startswith("bench-check/"):
        return (obj.get("status") == "ok"), f"check status={obj.get('status')}"
    if sv != SCHEMA_VERSION:
        return True, "non-result json (exists)"
    errs = validate(obj)
    if errs:
        return False, "schema: " + "; ".join(errs[:3])
    if obj["status"] not in accept:
        if obj["status"] in ("partial", "failed") and obj.get("terminal_reason") in TERMINAL:
            return True, f"terminal:{obj['terminal_reason']}"
        return False, f"status={obj['status']}"
    return True, obj["status"]


def write_json_atomic(path, obj, overwrite=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if not overwrite:
            raise FileExistsError(str(p))
        p.rename(p.with_name(f"{p.stem}.prev-{time.strftime('%Y%m%d-%H%M%S')}{p.suffix}"))
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(check_finite(obj), indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def write_check(path, status, **fields):
    doc = {"schema_version": "mangrove-bench-check/1", "status": status, "time": now(), **fields}
    return write_json_atomic(path, doc, overwrite=True)


def write_terminal(path, reason, **fields):
    doc = {"schema_version": "bench-terminal/1", "status": "failed", "terminal_reason": reason, "time": now(), **fields}
    return write_json_atomic(path, doc, overwrite=True)


def write_schema_file(path):
    Path(path).write_text(json.dumps(RESULT_SCHEMA, indent=1), encoding="utf-8")
