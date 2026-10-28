"""Noise suppression with DeepFilterNet 3.

Why a standalone binary rather than the ``deepfilternet`` Python package: its last
release (0.5.6, 2023) requires numpy<2 while WhisperX requires numpy>=2.1, so
both cannot live in one environment. The DeepFilterNet authors publish a
self-contained Rust build of the same model (DeepFilterNet3 embedded) for every
OS; we download it once, verify its checksum (model_registry.py) and run it as a
subprocess. As a bonus, a crash in the denoiser cannot take the app down.
"""

import subprocess
import tempfile
from pathlib import Path

from app.pipeline.errors import DenoiseError

# DeepFilterNet works on full-band 48 kHz audio; the runner converts to this
# rate before denoising and back to 16 kHz afterwards.
DEEPFILTER_SAMPLE_RATE = 48_000


class DeepFilterNetDenoiser:
    name = "DeepFilterNet3"

    def __init__(self, binary: Path, *, atten_lim_db: float | None = None) -> None:
        self._binary = binary
        self._atten_lim_db = atten_lim_db

    def denoise(self, src: Path, dst: Path) -> None:
        with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".denoise-") as out_dir:
            cmd = [
                str(self._binary),
                # Shift the output back by the STFT + look-ahead delay, so word
                # timestamps computed on denoised audio match the original.
                "--compensate-delay",
                "--output-dir", out_dir,
            ]  # fmt: skip
            if self._atten_lim_db is not None:
                cmd += ["--atten-lim-db", str(self._atten_lim_db)]
            cmd.append(str(src))
            result = subprocess.run(cmd, capture_output=True, text=True)
            produced = Path(out_dir) / src.name
            if result.returncode != 0 or not produced.is_file():
                raise DenoiseError(
                    "Noise reduction failed.",
                    hint=(result.stderr.strip()[-1000:] or None)
                    or "Process the file again without noise reduction.",
                )
            produced.replace(dst)
