"""French speech-to-text with faster-whisper, through WhisperX's batched pipeline.

WhisperX first cuts the audio into speech regions with a voice-activity detector,
then decodes ~30 s chunks in batches. Compared to plain sequential Whisper this
is much faster on long recordings and far less prone to Whisper's "hallucination
loops" over silences or background noise.
"""

from pathlib import Path

import numpy as np

from app.pipeline.base import ProgressCallback
from app.pipeline.hardware import DeviceProfile
from app.pipeline.types import RawSegment

LANGUAGE = "fr"  # forced: language auto-detection is slower and can misfire on noise


class WhisperXTranscriber:
    def __init__(self, model_id: str, model_path: Path, device: DeviceProfile) -> None:
        import whisperx

        self.name = model_id
        self._batch_size = device.batch_size
        self._pipeline = whisperx.load_model(
            str(model_path),
            device=device.whisper_device,
            compute_type=device.whisper_compute_type,
            language=LANGUAGE,
            # The pyannote VAD weights ship inside the whisperx package: nothing to
            # download (the "silero" option would fetch code from GitHub at run time).
            vad_method="pyannote",
            threads=device.cpu_threads,
            local_files_only=True,
        )

    def transcribe(
        self, audio: np.ndarray, progress: ProgressCallback | None = None
    ) -> list[RawSegment]:
        result = self._pipeline.transcribe(
            audio,
            batch_size=self._batch_size,
            language=LANGUAGE,
            progress_callback=(lambda pct: progress(pct / 100)) if progress else None,
        )
        return [
            RawSegment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            for seg in result["segments"]
            if seg["text"].strip()
        ]
