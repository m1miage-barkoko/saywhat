import pytest
import soundfile as sf

from app.config import get_settings
from app.pipeline import convert, model_registry
from app.pipeline.denoise import DEEPFILTER_SAMPLE_RATE, DeepFilterNetDenoiser
from app.pipeline.errors import DenoiseError, ModelNotAvailableError
from tests.conftest import DIALOGUE, requires_ffmpeg


@pytest.fixture(scope="module")
def binary():
    try:
        return model_registry.deepfilter_binary_path(get_settings().bin_dir)
    except ModelNotAvailableError:
        pytest.skip("DeepFilterNet not downloaded (python -m app.download_models)")


@requires_ffmpeg
def test_denoised_audio_keeps_duration_and_timing(binary, tmp_path):
    full_band = convert.to_wav(
        DIALOGUE, tmp_path / "in.wav", sample_rate=DEEPFILTER_SAMPLE_RATE, normalize=False
    )

    DeepFilterNetDenoiser(binary).denoise(full_band, tmp_path / "out.wav")

    before, after = sf.info(full_band), sf.info(tmp_path / "out.wav")
    assert after.samplerate == DEEPFILTER_SAMPLE_RATE
    # Same length: with --compensate-delay, timestamps stay valid after denoising.
    assert after.frames == pytest.approx(before.frames, abs=DEEPFILTER_SAMPLE_RATE * 0.05)
    assert not list(tmp_path.glob(".denoise-*")), "temporary folder left behind"


def test_failure_raises_a_readable_error(binary, tmp_path):
    not_audio = tmp_path / "not_audio.wav"
    not_audio.write_text("this is not an audio file")

    with pytest.raises(DenoiseError, match="Noise reduction failed"):
        DeepFilterNetDenoiser(binary).denoise(not_audio, tmp_path / "out.wav")
