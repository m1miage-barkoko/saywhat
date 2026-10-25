"""Data types shared by the pipeline, the API, the database and the exports.

Pipeline steps exchange these types only, never library objects (pandas frames,
pyannote annotations…), so any step can be replaced by another implementation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A stretch of time during which one speaker talks (diarization output)."""

    start: float
    end: float
    speaker: str


class Word(BaseModel):
    text: str
    # Seconds. None when forced alignment could not place the word (typically
    # digits such as "1 200", absent from the French wav2vec2 alphabet).
    start: float | None = None
    end: float | None = None
    score: float | None = None
    # No per-word speaker: segments are cut wherever the speaker changes, so
    # Segment.speaker is the single source of truth (simpler renames/merges).


class RawSegment(BaseModel):
    """Transcriber output: a chunk of text with coarse timestamps."""

    start: float
    end: float
    text: str


class AlignedSegment(BaseModel):
    """Aligner output: one sentence with word-level timestamps."""

    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Segment(BaseModel):
    """One sentence (or part of a sentence) spoken by a single speaker."""

    id: int
    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = Field(default_factory=list)


class Speaker(BaseModel):
    # Stable identifier referenced by segments ("S1"). Never shown to the user,
    # so renaming a speaker only touches `label` and merging only rewrites ids.
    id: str
    label: str


class ProcessingInfo(BaseModel):
    whisper_model: str
    diarization_model: str | None
    device: str
    denoised: bool
    created_at: datetime
    step_seconds: dict[str, float] = Field(default_factory=dict)


class Transcript(BaseModel):
    schema_version: Literal[1] = 1
    language: str = "fr"
    duration: float
    speakers: list[Speaker]
    segments: list[Segment]
    info: ProcessingInfo

    def speaker_label(self, speaker_id: str) -> str:
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker.label
        return speaker_id


class ProcessingOptions(BaseModel):
    """Per-recording choices made by the user before processing."""

    denoise: bool = True
    diarize: bool = True
    # Giving the number of speakers (exact, or a range) improves diarization a lot.
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    whisper_model: str | None = None  # None = default from settings

    @model_validator(mode="after")
    def _check_speaker_counts(self) -> Self:
        if self.num_speakers is not None and (self.min_speakers or self.max_speakers):
            raise ValueError("Indiquez soit un nombre exact d'intervenants, soit un intervalle.")
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("Le minimum d'intervenants dépasse le maximum.")
        return self
