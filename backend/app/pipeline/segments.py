"""Presentation helpers: turn single-speaker segments into readable paragraphs."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.pipeline.types import Segment, Transcript


@dataclass(frozen=True)
class Paragraph:
    """Consecutive segments of one speaker, displayed as a single block."""

    speaker: str
    segments: tuple[Segment, ...]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def merge_consecutive(segments: Sequence[Segment]) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    for segment in segments:
        if paragraphs and paragraphs[-1].speaker == segment.speaker:
            last = paragraphs[-1]
            paragraphs[-1] = Paragraph(last.speaker, (*last.segments, segment))
        else:
            paragraphs.append(Paragraph(segment.speaker, (segment,)))
    return paragraphs


def format_timestamp(seconds: float) -> str:
    """12.7 → "00:00:12" (truncated, so a line never shows a time after it starts)."""
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(transcript: Transcript) -> str:
    """The conversation as plain text, one paragraph per speaker turn."""
    return "\n".join(
        f"[{format_timestamp(p.start)}] {transcript.speaker_label(p.speaker)}: {p.text}"
        for p in merge_consecutive(transcript.segments)
    )


def format_plain_text(transcript: Transcript) -> str:
    """Only the words: one paragraph per speaker turn, no timestamps or labels."""
    return "\n\n".join(p.text for p in merge_consecutive(transcript.segments))
