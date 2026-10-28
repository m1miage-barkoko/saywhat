import pytest

from app.pipeline.assign_speakers import (
    SINGLE_SPEAKER,
    TurnIndex,
    attribute_speakers,
    fill_gaps,
    number_speakers,
    smooth_runs,
)
from app.pipeline.types import AlignedSegment, Segment, SpeakerTurn, Word

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def words(*spec: tuple[str, float | None, float | None]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in spec]


def sentence(*spec: tuple[str, float | None, float | None]) -> AlignedSegment:
    ws = words(*spec)
    timed = [w for w in ws if w.start is not None]
    return AlignedSegment(
        start=timed[0].start if timed else 0.0,
        end=timed[-1].end if timed else 0.0,
        text=" ".join(w.text for w in ws),
        words=ws,
    )


# --- TurnIndex -------------------------------------------------------------


class TestTurnIndex:
    index = TurnIndex([turn(0, 2, A), turn(2, 5, B), turn(7, 9, A)])

    def test_word_inside_a_turn(self):
        assert self.index.speaker_for(0.5, 1.0) == A
        assert self.index.speaker_for(3.0, 3.4) == B

    def test_word_straddling_two_turns_goes_to_the_larger_overlap(self):
        assert self.index.speaker_for(1.8, 2.5) == B  # 0.2 s of A, 0.5 s of B
        assert self.index.speaker_for(1.5, 2.1) == A  # 0.5 s of A, 0.1 s of B

    def test_word_in_a_silence_goes_to_the_nearest_turn(self):
        assert self.index.speaker_for(5.2, 5.5) == B  # 0.2 s after B, 1.5 s before A
        assert self.index.speaker_for(6.6, 6.9) == A  # 1.6 s after B, 0.1 s before A

    def test_word_before_the_first_or_after_the_last_turn(self):
        index = TurnIndex([turn(1, 2, B), turn(3, 4, C)])
        assert index.speaker_for(0.0, 0.5) == B
        assert index.speaker_for(10.0, 10.5) == C

    def test_zero_length_word(self):
        assert self.index.speaker_for(3.0, 3.0) == B

    def test_overlapping_turns_are_supported(self):
        # A long turn that contains a short one (regular, non-exclusive diarization).
        index = TurnIndex([turn(0, 10, A), turn(4, 5, B), turn(11, 12, C)])
        assert index.speaker_for(4.1, 4.9) in {A, B}  # same overlap: either is fine
        assert index.overlaps(4.1, 4.9) == pytest.approx({A: 0.8, B: 0.8})
        assert index.speaker_for(8.0, 9.0) == A  # B ended earlier, A still covers it
        assert index.speaker_for(10.2, 10.3) == A  # silence: A's end is closest

    def test_unsorted_input(self):
        index = TurnIndex([turn(7, 9, A), turn(0, 2, A), turn(2, 5, B)])
        assert index.speaker_for(3.0, 3.5) == B

    def test_empty(self):
        assert not TurnIndex([])
        assert TurnIndex([]).speaker_for(1, 2) is None


# --- helpers ---------------------------------------------------------------


def test_fill_gaps_uses_previous_then_next_neighbour():
    assert fill_gaps([None, A, None, None, B, None]) == [A, A, A, A, B, B]
    assert fill_gaps([None, None]) == [None, None]
    assert fill_gaps([]) == []


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([A, B, A], [A, A, A]),  # lone word in the middle
        ([B, A, A, A], [A, A, A, A]),  # lone word at the start, next to a longer run
        ([A, A, A, B], [A, A, A, A]),  # lone word at the end
        ([A, B, A, B, A], [A] * 5),  # alternating singles collapse
        ([A, A, B, B], [A, A, B, B]),  # a real exchange is kept
        ([A, B], [A, B]),  # two single words: nothing to prefer
        ([A, B, C], [A, B, C]),  # neighbours disagree: keep
        ([A], [A]),
        ([], []),
    ],
)
def test_smooth_runs(labels, expected):
    assert smooth_runs(labels) == expected


# --- attribute_speakers ----------------------------------------------------


def test_sentence_split_where_the_speaker_changes():
    turns = [turn(0.0, 2.0, A), turn(2.2, 4.0, B)]
    s = sentence(
        ("Tu", 0.1, 0.3), ("viens", 0.4, 0.8), ("demain", 0.9, 1.5),
        ("oui", 2.3, 2.6), ("bien", 2.7, 3.0), ("sûr", 3.1, 3.5),
    )  # fmt: skip

    segments = attribute_speakers([s], turns)

    assert [(seg.speaker, seg.text) for seg in segments] == [
        (A, "Tu viens demain"),
        (B, "oui bien sûr"),
    ]
    assert (segments[1].start, segments[1].end) == (2.3, 3.5)
    assert [seg.id for seg in segments] == [0, 1]


def test_untimed_words_follow_their_neighbours():
    # Digits have no letters in the French alignment alphabet, so "1 200" is untimed.
    turns = [turn(0.0, 3.0, A)]
    s = sentence(("Le", 0.1, 0.2), ("montant", 0.3, 0.8), ("est", 0.9, 1.0), ("1 200", None, None))

    [segment] = attribute_speakers([s], turns)

    assert segment.speaker == A
    assert segment.text == "Le montant est 1 200"


def test_boundary_jitter_does_not_split_a_sentence():
    # The first word starts 50 ms before B's turn: it overlaps A slightly more.
    turns = [turn(0.0, 1.03, A), turn(1.03, 5.0, B)]
    s = sentence(("Alors", 0.98, 1.06), ("on", 1.2, 1.3), ("commence", 1.4, 1.9))

    [segment] = attribute_speakers([s], turns)

    assert segment.speaker == B


def test_sentence_without_any_aligned_word():
    turns = [turn(0.0, 5.0, B)]
    s = AlignedSegment(start=1.0, end=2.0, text="...", words=[])

    [segment] = attribute_speakers([s], turns)

    assert (segment.speaker, segment.text, segment.words) == (B, "...", [])


def test_without_diarization_everything_is_one_speaker():
    s1 = sentence(("Bonjour", 0.0, 0.5))
    s2 = sentence(("Au", 1.0, 1.2), ("revoir", 1.3, 1.6))

    segments = attribute_speakers([s1, s2], turns=[])

    assert {seg.speaker for seg in segments} == {SINGLE_SPEAKER}


# --- number_speakers -------------------------------------------------------


def _segment(i: int, speaker: str) -> Segment:
    return Segment(id=i, start=i, end=i + 1, speaker=speaker, text=f"phrase {i}")


def test_speakers_numbered_by_first_appearance():
    raw = [_segment(0, C), _segment(1, A), _segment(2, C), _segment(3, B)]

    segments, speakers = number_speakers(raw)

    assert [s.speaker for s in segments] == ["S1", "S2", "S1", "S3"]
    assert [(s.id, s.label) for s in speakers] == [
        ("S1", "Speaker 1"),
        ("S2", "Speaker 2"),
        ("S3", "Speaker 3"),
    ]
    assert raw[0].speaker == C  # input left untouched
