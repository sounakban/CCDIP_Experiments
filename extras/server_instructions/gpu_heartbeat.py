import torch
import time
import os
import subprocess
import threading
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
# Calibration-free I-controller.
#
# Tracks a global rolling average of GPU utilization across all phases (rollout
# and update alike) and adjusts the heartbeat sleep via an integral of the error:
#
#   error     = TARGET_UTIL - global_avg      (positive  →  avg below target)
#   sleep_ms -= error * KI                    (reduce sleep when below target)
#
# The controller converges to whatever sleep time produces TARGET_UTIL% on
# average, regardless of GPU model, training config, or phase timing. No
# BASE_UTIL, SLEEP_MAX formula, or busy-threshold calibration required.
#
# Heartbeat kernels run on a dedicated CUDA stream so they interleave with
# training kernels rather than serialising behind them. Competition during the
# gradient-update phase is self-correcting: the higher average pushes the error
# negative, the integral increases sleep, and the firing rate drops until the
# average settles back at TARGET_UTIL.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_UTIL   = float(os.getenv("HB_TARGET",       "55.0"))  # desired avg GPU % (HPC minimum: 50%)
WINDOW        = float(os.getenv("HB_WINDOW",       "60.0"))  # rolling window (s)
KI            = float(os.getenv("HB_KI",            "0.5"))  # integral gain (ms / % / poll)
MIN_SLEEP_MS  = float(os.getenv("HB_MIN_SLEEP_MS",  "0.5"))  # floor
MAX_SLEEP_MS  = float(os.getenv("HB_MAX_SLEEP_MS", "50.0"))  # ceiling
INIT_SLEEP_MS = float(os.getenv("HB_INIT_SLEEP_MS", "5.0"))  # initial guess
POLL          = float(os.getenv("HB_POLL",           "0.2"))  # nvidia-smi poll interval (s)
N             = int(os.getenv("HB_N",              "6144"))   # matrix dimension

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

torch.backends.cuda.matmul.allow_tf32 = True
x      = torch.randn(N, N, device=device, dtype=torch.float16)
y      = torch.randn(N, N, device=device, dtype=torch.float16)
out    = torch.empty(N, N, device=device, dtype=torch.float16)
stream = torch.cuda.Stream()

# Warm up on the dedicated stream
with torch.cuda.stream(stream):
    for _ in range(20):
        torch.mm(x, y, out=out)
stream.synchronize()

def _measure_kernel_ms(n_measure: int = 100) -> float:
    with torch.cuda.stream(stream):
        for _ in range(n_measure):
            torch.mm(x, y, out=out)
    t0 = time.perf_counter()
    stream.synchronize()
    return (time.perf_counter() - t0) / n_measure * 1000

kernel_ms = _measure_kernel_ms()
print(f"[heartbeat] {gpu_name} | kernel={kernel_ms:.2f}ms | "
      f"target={TARGET_UTIL}% window={WINDOW}s N={N}")
print(f"[heartbeat] I-controller: KI={KI} init={INIT_SLEEP_MS}ms "
      f"range=[{MIN_SLEEP_MS}, {MAX_SLEEP_MS}]ms\n")

_sleep_ms: list[float] = [INIT_SLEEP_MS]
_sleep:    list[float] = [INIT_SLEEP_MS / 1000]
_history:  deque       = deque()  # (timestamp, util) — all phases


def monitor() -> None:
    last_log = time.monotonic()
    while True:
        try:
            raw  = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                encoding="utf-8", timeout=1.0,
            )
            util = int(raw.strip())
        except Exception:
            time.sleep(POLL)
            continue

        now = time.monotonic()
        _history.append((now, util))
        while _history and now - _history[0][0] > WINDOW:
            _history.popleft()

        if len(_history) >= 5:
            avg   = sum(u for _, u in _history) / len(_history)
            error = TARGET_UTIL - avg
            _sleep_ms[0] = max(MIN_SLEEP_MS, min(MAX_SLEEP_MS,
                               _sleep_ms[0] - error * KI))
            _sleep[0] = _sleep_ms[0] / 1000

        if now - last_log >= 60.0:
            avg_log = (
                sum(u for _, u in _history) / len(_history)
                if _history else 0.0
            )
            print(
                f"[heartbeat] util={util}% global_avg={avg_log:.1f}% "
                f"sleep={_sleep_ms[0]:.2f}ms",
                flush=True,
            )
            last_log = now

        time.sleep(POLL)


threading.Thread(target=monitor, daemon=True).start()

while True:
    with torch.cuda.stream(stream):
        torch.mm(x, y, out=out)
    stream.synchronize()
    s = _sleep[0]
    if s > 0:
        time.sleep(s)
