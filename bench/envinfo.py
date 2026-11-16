import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import paths

IS_WIN = os.name == "nt"


def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"


def snapshot(exclude_pids=()):
    me = {os.getpid(), os.getppid(), *[int(p) for p in exclude_pids]}
    if shutil.which("nvidia-smi"):
        apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        util = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                     "--format=csv,noheader"])
    else:
        apps, util = "ABSENT nvidia-smi", "ABSENT nvidia-smi"
    procs, load = [], None
    if IS_WIN:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-Process python*,py -ErrorAction SilentlyContinue | ForEach-Object { \"$($_.Id) $($_.ProcessName) "
                    "$([math]::Round($_.CPU,1)) $([math]::Round($_.WorkingSet64/1MB))\" }; "
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                   timeout=60)
        lines = [l for l in out.splitlines() if l.strip()]
        if lines and not lines[0].startswith("ERR"):
            load = lines[-1].strip()
            for l in lines[:-1]:
                try:
                    if int(l.split()[0]) not in me:
                        procs.append(l.strip())
                except ValueError:
                    pass
        else:
            load = out
    else:
        out = _run(["ps", "-eo", "pid,comm,pcpu,rss", "--no-headers"])
        for l in out.splitlines():
            parts = l.split()
            if len(parts) >= 2 and parts[1].startswith("python"):
                try:
                    if int(parts[0]) not in me:
                        procs.append(l.strip())
                except ValueError:
                    pass
        try:
            load = f"{100.0 * os.getloadavg()[0] / (os.cpu_count() or 1):.1f}"
        except Exception:
            load = None
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu_apps": apps, "gpu_util_mem": util,
            "python_procs_other_than_runner": procs, "cpu_load_pct": load}


def ancestor_pids():
    pids, pid = {os.getpid(), os.getppid()}, os.getppid()
    for _ in range(32):
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
        pids.add(pid)
    return pids


def _gpu_util_pct(s):
    try:
        return float(str(s.get("gpu_util_mem", "")).split(",")[0].strip().rstrip("%").strip())
    except (ValueError, IndexError):
        return None


def busy(s, cpu_threshold=30.0, ignore_python=False, platform_name=None, allow=()):
    reasons = []
    apps = s.get("gpu_apps") or ""
    if apps and not apps.startswith(("ERR", "ABSENT")) and apps.strip():
        reasons.append("gpu_compute_app")
    if s.get("python_procs_other_than_runner") and not ignore_python:
        reasons.append("other_python")
    try:
        if float(s.get("cpu_load_pct")) > cpu_threshold:
            reasons.append("cpu_load")
    except (TypeError, ValueError):
        pass
    return bool(reasons), reasons


def exclusivity_from_env():
    return {"pre": snapshot(), "contaminated": None, "source": "self"}


def cooling_from_env():
    return {"protocol": "none", "slept_s": 0}


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def environment(gpu=False):
    env = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
           "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "hostname": socket.gethostname(), "pid": os.getpid(),
           "libs": {m: _ver(m) for m in ("numpy", "xgboost", "lightgbm", "catboost", "torch", "triton", "sklearn")},
           "env_vars": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TRITON_INTERPRET",
                                                       "PYTORCH_CUDA_ALLOC_CONF",
                                                       "MANGROVE_BENCH_MPS")},
           "mangrove_tunables": {k: v for k, v in os.environ.items() if k.startswith("MANGROVE_")}}
    env.update(cpu_info())
    env["power_measurement"] = {
        "gpu": "nvidia-smi power.draw (board power), sampled when --gpu-sample-ms is set (result.gpu_util)",
        "cpu_package": "not measured"}
    if gpu:
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                env["gpu"] = {"name": p.name, "total_memory": p.total_memory, "sm_count": p.multi_processor_count,
                              "capability": f"{p.major}.{p.minor}", "torch_cuda": torch.version.cuda}
        except Exception as e:
            env["gpu"] = {"error": repr(e)}
        if shutil.which("nvidia-smi"):
            env["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                                      "--format=csv,noheader"])
    return env


def cpu_info():
    model = mhz = ram = None
    try:
        if sys.platform.startswith("linux"):
            txt = Path("/proc/cpuinfo").read_text(errors="replace")
            for line in txt.splitlines():
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "model name" and model is None:
                    model = v
                elif k == "cpu MHz" and mhz is None:
                    mhz = float(v)
            if model is None or mhz is None:
                out = _run(["lscpu"]) or ""
                for line in out.splitlines():
                    k, _, v = line.partition(":")
                    if k.strip() == "Model name" and model is None:
                        model = v.strip()
                    elif k.strip() in ("CPU MHz", "CPU max MHz") and mhz is None:
                        mhz = float(v.strip())
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) * 1024 / 2 ** 30
                    break
        elif IS_WIN:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
                mhz = float(winreg.QueryValueEx(key, "~MHz")[0])
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                ram = ms.ullTotalPhys / 2 ** 30
    except Exception:
        pass
    return {"cpu_model": model or platform.processor() or None, "cpu_mhz": mhz,
            "ram_gb": None if ram is None else round(ram, 2)}


TORCH_STAT_KEYS = ("num_alloc_retries", "num_ooms", "allocated_bytes.all.peak", "reserved_bytes.all.peak",
                   "num_device_alloc", "num_device_free")


def torch_stats():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return None
        st = torch.cuda.memory_stats()
    except Exception:
        return None
    return {k: int(st[k]) for k in TORCH_STAT_KEYS if k in st}


def merge_torch_stats(stats):
    stats = [s for s in stats if s]
    if not stats:
        return None
    out = {}
    for s in stats:
        for k, v in s.items():
            out[k] = max(out.get(k, 0), v) if k.endswith(".peak") else out.get(k, 0) + v
    out["n_processes"] = len(stats)
    return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_hashes(extra=()):
    out = {}
    files = sorted(paths.BENCH_DIR.glob("*.py")) + sorted((paths.BENCH_DIR / "bench").rglob("*.py"))
    for p in list(files) + [Path(e) for e in extra]:
        try:
            if p.exists() and p.is_file():
                out[paths.rel_to_bench(p)] = sha256_file(p)[:16]
        except OSError:
            pass
    return out


MEM_FRACTION = float(os.environ.get("MANGROVE_BENCH_MEM_FRACTION", "0.75"))
if not 0.1 <= MEM_FRACTION <= 0.95:
    raise ValueError(f"MANGROVE_BENCH_MEM_FRACTION={MEM_FRACTION} outside [0.1, 0.95]")


def gpu_process_guard(required=True):
    try:
        import torch
    except ImportError:
        if required:
            raise
        return {"torch": None}
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError("CUDA not available in this process")
        return {"cuda": False}
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    return {"cuda": True, "memory_fraction": MEM_FRACTION}


def alloc_cap(factor, budget_bytes, total_bytes):
    F = float(factor)
    if not (F > 0):
        raise ValueError("--alloc-cap-factor must be > 0")
    frac = min(MEM_FRACTION, F * float(budget_bytes) / float(total_bytes))
    import torch
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return {"alloc_cap_factor": F, "alloc_cap_budget_bytes": int(budget_bytes), "alloc_cap_fraction_effective": frac}


def gpu_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def gpu_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def gpu_peak():
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "source": "torch.cuda"}
    except ImportError:
        pass
    return {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None, "source": "n/a"}


def host_peak_rss():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r * 1024) if sys.platform != "darwin" else int(r)
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return int(pmc.PeakWorkingSetSize)
    except Exception:
        return None


class Timer:
    def __init__(self):
        self.t = {}
        self._open = {}

    def start(self, name):
        self._open[name] = time.perf_counter()

    def stop(self, name):
        dt = time.perf_counter() - self._open.pop(name)
        self.t[name] = self.t.get(name, 0.0) + dt
        return dt

    def get(self, name, default=0.0):
        return self.t.get(name, default)
