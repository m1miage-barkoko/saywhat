"""End-to-end run of the real models on the bundled 2-speaker dialogue.

Skipped by default (slow, needs ~4 GB of models). To run:

    uv run python -m app.download_models
    uv run pytest -m models
"""

import pytest

from app.pipeline.errors import ModelNotAvailableError
from app.pipeline.segments import merge_consecutive
from app.pipeline.types import ProcessingOptions
from tests.conftest import DIALOGUE

pytestmark = pytest.mark.models

# One distinctive word per line of the dialogue (scripts/make_test_audio.sh);
# speakers alternate, starting with Thomas.
EXPECTED_TURNS = [
    "livraison", "palettes", "entrepôt", "après-midi",
    "facture", "point", "journée", "jeudi",
]  # fmt: skip


@pytest.fixture(scope="module")
def pipeline():
    from app.config import get_settings
    from app.pipeline.hardware import detect_device
    from app.pipeline.runner import LocalModels, TranscriptionPipeline

    settings = get_settings()
    return TranscriptionPipeline(LocalModels(settings, detect_device(settings.device)))


def run(pipeline, tmp_path, **options):
    try:
        return pipeline.run(DIALOGUE, tmp_path, ProcessingOptions(**options))
    except ModelNotAvailableError as exc:
        pytest.skip(str(exc))


def test_two_speakers_alternate(pipeline, tmp_path):
    transcript = run(pipeline, tmp_path).transcript

    assert [s.label for s in transcript.speakers] == ["Intervenant 1", "Intervenant 2"]
    paragraphs = merge_consecutive(transcript.segments)
    assert [p.speaker for p in paragraphs] == ["S1", "S2"] * 4
    for paragraph, keyword in zip(paragraphs, EXPECTED_TURNS, strict=True):
        assert keyword in paragraph.text.lower()


def test_without_denoising_or_diarization(pipeline, tmp_path):
    result = run(pipeline, tmp_path, denoise=False, diarize=False)

    assert result.denoised_audio_path is None
    assert result.audio_path.is_file()
    assert {s.speaker for s in result.transcript.segments} == {"S1"}
    text = " ".join(s.text for s in result.transcript.segments).lower()
    assert all(keyword in text for keyword in EXPECTED_TURNS)
    words = [w for s in result.transcript.segments for w in s.words]
    assert sum(w.start is not None for w in words) / len(words) > 0.9  # word timestamps
