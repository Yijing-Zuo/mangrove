import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "utilization.gpu", "utilization.memory", "memory.used", "power.draw", "clocks.sm", "temperature.gpu"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_csv(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(FIELDS) or not parts[0][:4].isdigit():
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        rows.append({"t": ts, "util": _num(parts[1]), "util_mem": _num(parts[2]), "mem_mib": _num(parts[3]),
                     "power_w": _num(parts[4]), "clock_mhz": _num(parts[5]), "temp_c": _num(parts[6])})
    return rows


def summarize(rows, t0=None, t1=None):
    import numpy as np
    sel = [r for r in rows if (t0 is None or r["t"] >= t0 - 0.5) and (t1 is None or r["t"] <= t1 + 0.5)]

    def col(k):
        return [r[k] for r in sel if r[k] is not None]
    u, p, m = col("util"), col("power_w"), col("mem_mib")
    out = {"n_samples": len(sel), "window_s": (t1 - t0) if (t0 is not None and t1 is not None) else None}
    if u:
        out.update(util_gpu_mean=float(np.mean(u)), util_gpu_median=float(np.median(u)),
                   util_gpu_p95=float(np.percentile(u, 95)), util_gpu_max=float(np.max(u)))
    if col("util_mem"):
        out["util_mem_mean"] = float(np.mean(col("util_mem")))
    if m:
        out.update(mem_used_max_mib=float(np.max(m)), mem_used_mean_mib=float(np.mean(m)))
    if p:
        out.update(power_mean_w=float(np.mean(p)), power_max_w=float(np.max(p)))
        if out["window_s"]:
            out["energy_j"] = out["power_mean_w"] * out["window_s"]
    if col("clock_mhz"):
        out["clocks_sm_mean_mhz"] = float(np.mean(col("clock_mhz")))
    if col("temp_c"):
        out["temp_max_c"] = float(np.max(col("temp_c")))
    return out


class GpuSampler:
    def __init__(self, path, interval_ms=1000):
        self.path = Path(path)
        self.interval_ms = int(interval_ms)
        self.proc = None
        self.t0 = self.t1 = None
        self.error = None

    def start(self):
        exe = shutil.which("nvidia-smi")
        if not exe:
            self.error = "nvidia-smi not found"
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.proc = subprocess.Popen([exe, f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
                                      f"-lms={self.interval_ms}"], stdout=self._fh, stderr=subprocess.DEVNULL)
        time.sleep(min(1.0, 2 * self.interval_ms / 1000))
        return self

    def mark_window(self, t0, t1):
        self.t0, self.t1 = t0, t1

    def stop(self):
        if self.proc is not None:
            time.sleep(min(1.0, self.interval_ms / 1000))
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            self._fh.close()
            self.proc = None
        return self

    def summary(self, rel=None):
        if self.error:
            return {"error": self.error}
        if not self.path.exists():
            return {"error": "no samples file"}
        rows = parse_csv(self.path)
        s = summarize(rows, self.t0, self.t1)
        allm = [r["mem_mib"] for r in rows if r["mem_mib"] is not None]
        s["mem_used_max_all_mib"] = float(max(allm)) if allm else None
        s["interval_ms"] = self.interval_ms
        s["raw"] = rel or str(self.path)
        return s


def from_opts(opts):
    ms = opts.get("gpu_sample_ms")
    p = opts.get("gpu_sample_path")
    return GpuSampler(p, ms) if (ms and p) else None
