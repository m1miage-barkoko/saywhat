"""Common interface of the swappable pipeline steps.

Each model-backed step is a small Protocol. The runner only knows these
interfaces, so a step can be replaced (by a cloud transcription engine, a test
double, a different diarization model…) without touching the runner, the job
queue or the API. Audio is passed as 16 kHz mono float32 numpy arrays.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from app.pipeline.types import AlignedSegment, RawSegment, SpeakerTurn

# Receives the completed fraction (0.0 → 1.0) of the step being run.
ProgressCallback = Callable[[float], None]


class Denoiser(Protocol):
    name: str

    def denoise(self, src: Path, dst: Path) -> None: ...


class Diarizer(Protocol):
    name: str

    def diarize(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[SpeakerTurn]: ...


class Transcriber(Protocol):
    name: str

    def transcribe(
        self, audio: np.ndarray, progress: ProgressCallback | None = None
    ) -> list[RawSegment]: ...


class Aligner(Protocol):
    name: str

    def align(
        self,
        segments: list[RawSegment],
        audio: np.ndarray,
        progress: ProgressCallback | None = None,
    ) -> list[AlignedSegment]: ...
