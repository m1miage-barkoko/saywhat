from datetime import UTC, datetime

import pytest

from app.pipeline.segments import format_timestamp, format_transcript, merge_consecutive
from app.pipeline.types import ProcessingInfo, Segment, Speaker, Transcript


def seg(i: int, speaker: str, start: float, text: str) -> Segment:
    return Segment(id=i, start=start, end=start + 1, speaker=speaker, text=text)


SEGMENTS = [
    seg(0, "S1", 72.0, "Bonjour, alors pour la livraison de mardi…"),
    seg(1, "S1", 74.0, "On en est où ?"),
    seg(2, "S2", 80.5, "Oui, il faudra prévoir deux palettes."),
    seg(3, "S1", 83.0, "Parfait."),
    seg(4, "S1", 84.0, "Merci."),
]


def test_consecutive_segments_of_one_speaker_form_a_paragraph():
    paragraphs = merge_consecutive(SEGMENTS)

    assert [(p.speaker, [s.id for s in p.segments]) for p in paragraphs] == [
        ("S1", [0, 1]),
        ("S2", [2]),
        ("S1", [3, 4]),
    ]
    assert paragraphs[0].text == "Bonjour, alors pour la livraison de mardi… On en est où ?"
    assert (paragraphs[0].start, paragraphs[0].end) == (72.0, 75.0)


def test_merge_empty():
    assert merge_consecutive([]) == []


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (72.9, "00:01:12"),
        (3599.99, "00:59:59"),
        (7384, "02:03:04"),
        (-1, "00:00:00"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_transcript_uses_speaker_labels():
    transcript = Transcript(
        duration=90,
        speakers=[Speaker(id="S1", label="Speaker 1"), Speaker(id="S2", label="M. Dupont")],
        segments=SEGMENTS,
        info=ProcessingInfo(
            whisper_model="m", diarization_model="d", device="cpu", denoised=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )  # fmt: skip

    assert format_transcript(transcript).splitlines() == [
        "[00:01:12] Speaker 1: Bonjour, alors pour la livraison de mardi… On en est où ?",
        "[00:01:20] M. Dupont: Oui, il faudra prévoir deux palettes.",
        "[00:01:23] Speaker 1: Parfait. Merci.",
    ]
