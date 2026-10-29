"""Pick where each model runs, based on the hardware available.

* NVIDIA GPU (CUDA): everything on the GPU, Whisper in float16.
* Apple Silicon: pyannote and wav2vec2 on the GPU through Metal (MPS), but Whisper
  on the CPU in int8, because CTranslate2 (faster-whisper's engine) has no Metal
  backend.
* Otherwise: everything on the CPU, Whisper in int8 (about 4x less memory than
  float32, and faster, for a negligible accuracy loss).
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["cuda", "mps", "cpu"]

_MAX_DEFAULT_THREADS = 6


@dataclass(frozen=True)
class DeviceProfile:
    kind: DeviceKind
    description: str  # shown in the UI
    whisper_device: Literal["cuda", "cpu"]
    whisper_compute_type: Literal["float16", "int8"]
    torch_device: DeviceKind  # pyannote + wav2vec2 alignment
    batch_size: int
    cpu_threads: int


def detect_device(
    preference: str = "auto", *, batch_size: int | None = None, cpu_threads: int | None = None
) -> DeviceProfile:
    import torch

    if preference in ("auto", "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        return DeviceProfile(
            kind="cuda",
            description=f"NVIDIA GPU {props.name} ({vram_gb:.0f} GB)",
            whisper_device="cuda",
            whisper_compute_type="float16",
            torch_device="cuda",
            # Batched decoding is what makes WhisperX fast on GPU; 16 fits large-v3
            # in float16 on 10+ GB cards, 8 keeps 6-8 GB cards out of trouble.
            batch_size=batch_size or (16 if vram_gb >= 10 else 8),
            cpu_threads=cpu_threads or 4,
        )

    threads = cpu_threads or _default_cpu_threads()
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return DeviceProfile(
            kind="mps",
            description=f"{_apple_chip_name()} — Whisper on CPU, diarization on GPU (Metal)",
            whisper_device="cpu",
            whisper_compute_type="int8",
            torch_device="mps",
            batch_size=batch_size or 8,
            cpu_threads=threads,
        )

    return DeviceProfile(
        kind="cpu",
        description=f"CPU only ({threads} threads)",
        whisper_device="cpu",
        whisper_compute_type="int8",
        torch_device="cpu",
        batch_size=batch_size or 8,
        cpu_threads=threads,
    )


def _default_cpu_threads() -> int:
    # CTranslate2's CPU decoding stops scaling past a handful of threads, and
    # extra hyper-threads / efficiency cores mostly add contention. Measured on an
    # Apple M4 (4 performance + 6 efficiency cores), large-v3 int8, batch 8:
    # 4 threads ×2.1 real time, 6 → ×2.4, 10 → ×2.2. Tune with SAYWHAT_CPU_THREADS.
    return min(os.cpu_count() or 4, _MAX_DEFAULT_THREADS)


def _apple_chip_name() -> str:
    return _sysctl("machdep.cpu.brand_string") or "Apple Silicon"


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None
