"""Live hardware telemetry from psutil and nvidia-smi."""
import platform
import shutil
import subprocess

import psutil

from .config import DATA_DIR

_cpu_name = None


def cpu_name():
    global _cpu_name
    if _cpu_name is None:
        _cpu_name = platform.processor() or platform.machine()
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        _cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return _cpu_name


def gpus():
    if not shutil.which("nvidia-smi"):
        return []
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,driver_version"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    res = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None
        res.append({"index": int(f[0]), "name": f[1], "util": num(f[2]),
                    "memUsedMB": num(f[3]), "memTotalMB": num(f[4]), "temp": num(f[5]),
                    "power": num(f[6]), "powerLimit": num(f[7]), "driver": f[8]})
    return res


def cuda_version():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for tok in out.split("|"):
        if "CUDA Version" in tok:
            return tok.split("CUDA Version:")[1].strip().split()[0]
    return None


def snapshot():
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(str(DATA_DIR))
    freq = psutil.cpu_freq()
    return {
        "cpu": {"name": cpu_name(), "percent": psutil.cpu_percent(interval=None),
                "perCore": psutil.cpu_percent(interval=None, percpu=True),
                "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
                "freqMHz": freq.current if freq else None},
        "ram": {"used": vm.total - vm.available, "total": vm.total, "percent": vm.percent},
        "disk": {"used": du.used, "total": du.total, "percent": du.percent, "path": str(DATA_DIR)},
        "gpus": gpus(),
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
    }
