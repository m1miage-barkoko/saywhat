"""Give every transcribed word to one speaker, then cut sentences where the speaker changes.

Words (from the aligner) and speaker turns (from diarization) come from two
independent models whose boundaries never match exactly. Rules, in order:

1. A timed word goes to the speaker whose turns overlap it the most.
2. A timed word that falls in a silence between turns goes to the nearest turn.
3. A word the aligner could not time (e.g. "1 200") takes its neighbour's speaker.
4. Inside a sentence, a single word attributed to someone else is treated as
   boundary jitter and absorbed by its neighbours (see `smooth_runs`).
5. Each sentence is split wherever the speaker changes.

Speakers are then renumbered by order of first appearance: the first voice heard
becomes "Intervenant 1".
"""

from bisect import bisect_left
from collections.abc import Sequence
from itertools import groupby

from app.pipeline.types import AlignedSegment, Segment, Speaker, SpeakerTurn, Word

SPEAKER_LABEL = "Intervenant {n}"
SINGLE_SPEAKER = "SPEAKER_00"  # when diarization is skipped or finds nobody
_EPSILON = 1e-3  # gives zero-length words a 1 ms extent so overlaps are measurable


class TurnIndex:
    """Speaker turns sorted by start time, for fast "who speaks during [start, end]?".

    Works with overlapping turns (regular diarization) as well as with the
    non-overlapping turns of the exclusive diarization.
    """

    def __init__(self, turns: Sequence[SpeakerTurn]) -> None:
        self._turns = sorted(turns, key=lambda t: (t.start, t.end))
        self._starts = [t.start for t in self._turns]
        # _reach[i] = index of the turn that ends last among turns[0..i]. A
        # backwards scan can stop as soon as no earlier turn reaches the query.
        self._reach: list[int] = []
        for i, turn in enumerate(self._turns):
            best = self._reach[-1] if self._reach else i
            self._reach.append(i if turn.end > self._turns[best].end else best)

    def __bool__(self) -> bool:
        return bool(self._turns)

    def overlaps(self, start: float, end: float) -> dict[str, float]:
        """Seconds of overlap between [start, end] and each speaker's turns."""
        totals: dict[str, float] = {}
        first_after = bisect_left(self._starts, end)  # turns from here start after `end`
        for j in range(first_after - 1, -1, -1):
            if self._turns[self._reach[j]].end <= start:
                break
            turn = self._turns[j]
            overlap = min(turn.end, end) - max(turn.start, start)
            if overlap > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0.0) + overlap
        return totals

    def nearest(self, start: float, end: float) -> str | None:
        """Speaker of the turn closest to [start, end] (used for words in silences)."""
        if not self._turns:
            return None
        first_after = bisect_left(self._starts, end)
        candidates: list[tuple[float, str]] = []
        if first_after > 0:  # listed first so that ties go to the previous speaker
            before = self._turns[self._reach[first_after - 1]]
            candidates.append((start - before.end, before.speaker))
        if first_after < len(self._turns):
            after = self._turns[first_after]
            candidates.append((after.start - end, after.speaker))
        return min(candidates, key=lambda c: c[0])[1]

    def speaker_for(self, start: float, end: float) -> str | None:
        end = max(end, start + _EPSILON)
        totals = self.overlaps(start, end)
        if totals:
            return max(totals, key=totals.__getitem__)
        return self.nearest(start, end)


def assign_words(words: Sequence[Word], index: TurnIndex) -> list[str | None]:
    """Rules 1-2. Untimed words get None."""
    return [
        index.speaker_for(w.start, w.end) if w.start is not None and w.end is not None else None
        for w in words
    ]


def fill_gaps(speakers: Sequence[str | None]) -> list[str | None]:
    """Rule 3: an untimed word takes the previous word's speaker (the next one's at the start)."""
    filled = list(speakers)
    for i in range(1, len(filled)):
        if filled[i] is None:
            filled[i] = filled[i - 1]
    for i in range(len(filled) - 2, -1, -1):
        if filled[i] is None:
            filled[i] = filled[i + 1]
    return filled


def smooth_runs(speakers: Sequence[str], max_len: int = 1) -> list[str]:
    """Rule 4: reassign runs of at most `max_len` words that are most likely jitter.

    A short run is absorbed when it is sandwiched between two runs of the same
    speaker (A B A → A A A), or when it sits at either end of the sentence next to
    a longer run (B A A A → A A A): at a change of speaker, the first word of a
    sentence often starts a few milliseconds before the diarization turn does.
    Runs are recomputed after each change, so A B A B A collapses to A A A A A,
    while a genuine exchange of longer runs (A A B B) is left alone.
    """
    labels = list(speakers)
    while True:
        runs = _runs(labels)
        for k, (_, start, length) in enumerate(runs):
            if length > max_len or len(runs) < 2:
                continue
            prev = runs[k - 1] if k > 0 else None
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            if prev and nxt and prev[0] == nxt[0]:
                new = prev[0]
            elif prev is None and nxt and nxt[2] > max_len:
                new = nxt[0]
            elif nxt is None and prev and prev[2] > max_len:
                new = prev[0]
            else:
                continue
            labels[start : start + length] = [new] * length
            break
        else:  # no run changed: stable
            return labels


def _runs(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """(speaker, first index, length) for each maximal run of equal labels."""
    runs = []
    position = 0
    for speaker, group in groupby(labels):
        length = sum(1 for _ in group)
        runs.append((speaker, position, length))
        position += length
    return runs


def attribute_speakers(
    sentences: Sequence[AlignedSegment], turns: Sequence[SpeakerTurn]
) -> list[Segment]:
    """Rules 1-5: aligned sentences + speaker turns → single-speaker segments."""
    index = TurnIndex(turns)
    segments: list[Segment] = []
    for sentence in sentences:
        sentence_speaker = (
            index.speaker_for(sentence.start, sentence.end) if index else None
        ) or SINGLE_SPEAKER
        if not sentence.words:
            segments.append(
                Segment(
                    id=len(segments),
                    start=sentence.start,
                    end=sentence.end,
                    speaker=sentence_speaker,
                    text=sentence.text,
                )
            )
            continue

        if index:
            raw = fill_gaps(assign_words(sentence.words, index))
            labels = smooth_runs([speaker or sentence_speaker for speaker in raw])
        else:
            labels = [SINGLE_SPEAKER] * len(sentence.words)

        for speaker, group in groupby(zip(sentence.words, labels, strict=True), key=lambda p: p[1]):
            words = [word for word, _ in group]
            timed = [w for w in words if w.start is not None and w.end is not None]
            segments.append(
                Segment(
                    id=len(segments),
                    start=timed[0].start if timed else sentence.start,  # type: ignore[arg-type]
                    end=timed[-1].end if timed else sentence.end,  # type: ignore[arg-type]
                    speaker=speaker,
                    text=" ".join(w.text for w in words),
                    words=words,
                )
            )
    return segments


def number_speakers(segments: Sequence[Segment]) -> tuple[list[Segment], list[Speaker]]:
    """Replace raw labels (SPEAKER_03…) by S1, S2… in order of first appearance."""
    ids: dict[str, str] = {}
    for segment in segments:
        ids.setdefault(segment.speaker, f"S{len(ids) + 1}")
    renamed = [s.model_copy(update={"speaker": ids[s.speaker]}) for s in segments]
    speakers = [
        Speaker(id=new_id, label=SPEAKER_LABEL.format(n=n))
        for n, new_id in enumerate(ids.values(), start=1)
    ]
    return renamed, speakers
