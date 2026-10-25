import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import get_settings

# Like the app's entry points: models are looked up (and caches written) in the
# project's data folder, never in the home folder.
get_settings().use_local_model_dirs()

DATA_DIR = Path(__file__).parent / "data"
DIALOGUE = DATA_DIR / "dialogue_fr.m4a"  # 2 speakers, ~32 s (scripts/make_test_audio.sh)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg n'est pas installé",
)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def tone(tmp_path: Path):
    """Factory: a stereo 44.1 kHz sine wave encoded with the given ffmpeg arguments."""

    def make(name: str, *codec_args: str, seconds: float = 2.0, volume: float = 0.05) -> Path:
        path = tmp_path / name
        ffmpeg(
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
            "-af", f"volume={volume}", "-ac", "2", *codec_args, str(path),
        )  # fmt: skip
        return path

    return make
