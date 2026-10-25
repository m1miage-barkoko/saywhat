"""Word-level timestamps by forced alignment with a French wav2vec2 model.

Whisper's own timestamps are only accurate to about a second. Forced alignment
matches each character of the transcript against a phonetic model of the audio,
giving per-word start/end times: needed both to attribute words to speakers and
to highlight the line being played in the UI. WhisperX also splits the text into
sentences at this stage.
"""

import math

import numpy as np

from app.pipeline.base import ProgressCallback
from app.pipeline.types import AlignedSegment, RawSegment, Word

LANGUAGE = "fr"


class WhisperXAligner:
    def __init__(self, device: str, model_name: str | None = None) -> None:
        import whisperx
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH

        # model_name=None → WhisperX's default for French (torchaudio's VoxPopuli
        # wav2vec2 base, ~360 MB).
        self._model, self._metadata = whisperx.load_align_model(
            language_code=LANGUAGE, device=device, model_name=model_name
        )
        self._device = device
        self.name = model_name or DEFAULT_ALIGN_MODELS_TORCH[LANGUAGE]

    def align(
        self,
        segments: list[RawSegment],
        audio: np.ndarray,
        progress: ProgressCallback | None = None,
    ) -> list[AlignedSegment]:
        import whisperx

        if not segments:
            return []
        result = whisperx.align(
            [seg.model_dump() for seg in segments],
            self._model,
            self._metadata,
            audio,
            self._device,
            return_char_alignments=False,
            progress_callback=(lambda pct: progress(pct / 100)) if progress else None,
        )
        return [
            AlignedSegment(
                start=_seconds(seg.get("start")) or 0.0,
                end=_seconds(seg.get("end")) or 0.0,
                text=seg["text"].strip(),
                words=[
                    Word(
                        text=w["word"],
                        start=_seconds(w.get("start")),
                        end=_seconds(w.get("end")),
                        score=_seconds(w.get("score")),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in result["segments"]
        ]


def _seconds(value: object) -> float | None:
    """WhisperX marks unalignable words with a missing key *or* NaN: normalise to None."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return None if math.isnan(number) else number
