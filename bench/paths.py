import os
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent


def _env_path(var, default):
    v = os.environ.get(var)
    return Path(v).resolve() if v else default


ENGINE_DIR = _env_path("MANGROVE_BENCH_ENGINE", BENCH_DIR)
PREP_DIR = BENCH_DIR / "bench" / "prep"
DATA_DIR = _env_path("MANGROVE_BENCH_DATA", BENCH_DIR / "data")
RAW_DIR = _env_path("MANGROVE_BENCH_RAW", DATA_DIR / "raw")
RESULTS_DIR = _env_path("MANGROVE_BENCH_RESULTS", BENCH_DIR / "results")
LOGS_DIR = _env_path("MANGROVE_BENCH_LOGS", BENCH_DIR / "logs")


def rel_to_bench(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(BENCH_DIR).as_posix()
    except ValueError:
        return p.as_posix()
