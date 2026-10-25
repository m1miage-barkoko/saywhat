"""Audio conversion with ffmpeg: any phone recording → mono PCM WAV.

We call the ffmpeg command-line tool rather than a Python binding: it reads every
format phones produce (AAC/m4a, AMR, 3GP, Opus…), it streams (constant memory on
a 2-hour file) and it behaves identically on Windows, macOS and Linux.
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.pipeline.base import ProgressCallback
from app.pipeline.errors import AudioConversionError, PipelineError

SAMPLE_RATE = 16_000  # what Whisper, wav2vec2 and pyannote all expect

# Accepted by the UI file picker. ffmpeg reads many more; these are the ones
# phone voice recorders produce.
SUPPORTED_EXTENSIONS = (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".opus", ".amr", ".3gp")

# EBU R128 loudness normalisation to -16 LUFS, the usual target for speech.
# Quiet phone recordings come out at a consistent level, which keeps Whisper's
# voice-activity detection and the diarization model in their comfort zone.
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    codec: str
    sample_rate: int
    channels: int


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise PipelineError(
            f"{name} est introuvable.",
            hint="Installez ffmpeg (voir le README) puis relancez Verbatim.",
        )
    return path


def probe(path: Path) -> AudioInfo:
    """Read the duration and format of the first audio track; reject broken files."""
    if not path.is_file():
        raise AudioConversionError(f"Fichier introuvable : {path}")
    cmd = [
        _executable("ffprobe"),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(path),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioConversionError(
            f"Fichier audio illisible ou corrompu : {path.name}",
            hint=result.stderr.strip() or None,
        )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioConversionError(f"Aucune piste audio dans {path.name}.")
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, ValueError):
        raise AudioConversionError(f"Durée illisible pour {path.name}.") from None
    stream = streams[0]
    return AudioInfo(
        duration=duration,
        codec=stream.get("codec_name", "?"),
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
    )


def to_wav(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
    duration: float | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Convert the first audio track of `src` to a mono 16-bit WAV at `dst`.

    `src` is only read, never modified. The output is written to a temporary
    file then renamed, so `dst` never exists half-written (useful if the app is
    stopped mid-conversion). Pass `duration` (from `probe`) to get progress.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    cmd = [
        _executable("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-map", "0:a:0",
        "-ac", "1",
        *(["-af", LOUDNORM_FILTER] if normalize else []),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        # Machine-readable progress on stdout: "out_time_us=…" lines.
        "-progress", "pipe:1", "-nostats",
        str(tmp),
    ]  # fmt: skip
    # stderr goes to a file, not a pipe: a corrupted input can make ffmpeg log
    # thousands of decoding errors, and a full stderr pipe would block ffmpeg
    # while we are blocked reading stdout (deadlock).
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8"
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress and duration and line.startswith("out_time_us="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    progress(min(int(value) / 1e6 / duration, 1.0))
        returncode = proc.wait()
        err.seek(0)
        stderr = err.read()[-2000:]
    if returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AudioConversionError(
            f"La conversion de {src.name} a échoué.", hint=stderr.strip() or None
        )
    tmp.replace(dst)
    if progress:
        progress(1.0)
    return dst


def load_wav(path: Path) -> np.ndarray:
    """Load a WAV produced by `to_wav` as float32 samples in [-1, 1]."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise AudioConversionError(
            f"{path.name} n'est pas un WAV mono {SAMPLE_RATE} Hz "
            f"({sample_rate} Hz, {audio.ndim} dimension(s))."
        )
    return audio
