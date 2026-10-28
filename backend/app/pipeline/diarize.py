"""Speaker diarization ("who speaks when") with pyannote's community-1 pipeline.

We call pyannote directly rather than through ``whisperx.DiarizationPipeline``:
WhisperX only exposes the regular diarization, in which overlapping speech yields
several simultaneous speakers. community-1 also produces an *exclusive*
diarization (exactly one speaker at any instant, the most likely one), which is
what we need to give every word a single owner.
"""

from pathlib import Path

import numpy as np

from app.pipeline.base import ProgressCallback
from app.pipeline.convert import SAMPLE_RATE
from app.pipeline.errors import ModelNotAvailableError
from app.pipeline.types import SpeakerTurn

# pyannote reports progress for two sub-steps, each counting from 0 to its own
# total; map them onto one monotonic 0 → 1 range.
_SUBSTEP_RANGES = {"segmentation": (0.0, 0.4), "embeddings": (0.4, 1.0)}


class PyannoteDiarizer:
    def __init__(self, model_id: str, model_dir: Path, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.name = model_id
        # Loading from the local folder needs neither a token nor the network.
        pipeline = Pipeline.from_pretrained(model_dir)
        if pipeline is None:
            raise ModelNotAvailableError(f"Could not load the model {model_id}.")
        self._pipeline = pipeline.to(torch.device(device))

    def diarize(
        self,
        audio: np.ndarray,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[SpeakerTurn]:
        import torch

        output = self._pipeline(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            **({"hook": _ProgressHook(progress)} if progress else {}),
        )
        return [
            SpeakerTurn(start=float(segment.start), end=float(segment.end), speaker=str(label))
            for segment, _, label in output.exclusive_speaker_diarization.itertracks(
                yield_label=True
            )
        ]


class _ProgressHook:
    def __init__(self, progress: ProgressCallback) -> None:
        self._progress = progress
        self._last = 0.0

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if not total or completed is None:
            return
        low, high = _SUBSTEP_RANGES.get(step_name, (0.0, 1.0))
        value = low + min(completed / total, 1.0) * (high - low)
        if value > self._last:
            self._last = value
            self._progress(value)
