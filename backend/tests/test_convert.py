import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline import convert
from app.pipeline.errors import AudioConversionError
from tests.conftest import DIALOGUE, requires_ffmpeg

pytestmark = requires_ffmpeg

# (file name, ffmpeg encoder arguments) — one per format a phone may produce.
PHONE_FORMATS = [
    ("rec.m4a", ("-c:a", "aac")),
    ("rec.mp3", ("-c:a", "libmp3lame")),
    ("rec.wav", ("-c:a", "pcm_s16le")),
    ("rec.aac", ("-c:a", "aac", "-f", "adts")),
    ("rec.ogg", ("-c:a", "libopus")),
    ("rec.opus", ("-c:a", "libopus")),
    ("rec.3gp", ("-c:a", "aac", "-ar", "16000", "-ac", "1")),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("name", "codec_args"), PHONE_FORMATS)
def test_any_phone_format_becomes_16k_mono_wav(tone, tmp_path, name, codec_args):
    src = tone(name, *codec_args)
    dst = convert.to_wav(src, tmp_path / "out.wav")

    info = sf.info(dst)
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
    assert info.duration == pytest.approx(2.0, abs=0.1)


def test_amr(tmp_path):
    # No AMR encoder ships with common ffmpeg builds, so write a valid AMR-NB file
    # by hand: the "#!AMR\n" magic, then 100 frames of 20 ms. Each frame is a
    # header byte (mode 4.75 kbit/s, quality bit set) and 12 bytes of payload.
    src = tmp_path / "rec.amr"
    src.write_bytes(b"#!AMR\n" + (b"\x04" + bytes(12)) * 100)

    dst = convert.to_wav(src, tmp_path / "out.wav")

    assert sf.info(dst).duration == pytest.approx(2.0, abs=0.05)


def test_original_is_never_modified(tmp_path):
    src = tmp_path / "dialogue.m4a"
    src.write_bytes(DIALOGUE.read_bytes())
    before = _sha256(src)

    convert.to_wav(src, tmp_path / "out.wav")

    assert _sha256(src) == before


def test_loudness_is_normalised(tone, tmp_path):
    quiet = tone("quiet.wav", "-c:a", "pcm_s16le", seconds=5, volume=0.01)

    plain = convert.load_wav(convert.to_wav(quiet, tmp_path / "plain.wav", normalize=False))
    louder = convert.load_wav(convert.to_wav(quiet, tmp_path / "norm.wav", normalize=True))

    rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
    assert rms(louder) > 5 * rms(plain)


def test_probe_reads_duration_and_codec():
    info = convert.probe(DIALOGUE)

    assert info.codec == "aac"
    assert info.duration == pytest.approx(31.9, abs=0.2)


def test_progress_is_reported_up_to_one(tmp_path):
    seen: list[float] = []
    info = convert.probe(DIALOGUE)

    convert.to_wav(DIALOGUE, tmp_path / "out.wav", duration=info.duration, progress=seen.append)

    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_corrupted_file_raises_a_readable_error(tmp_path):
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"garbage" * 100)

    with pytest.raises(AudioConversionError, match="illisible ou corrompu"):
        convert.probe(broken)
    with pytest.raises(AudioConversionError, match="a échoué"):
        convert.to_wav(broken, tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists()
    assert not (tmp_path / "out.wav.part").exists()


def test_file_without_audio_track_is_rejected(tmp_path):
    from tests.conftest import ffmpeg

    video = tmp_path / "video.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=64x64:duration=1", "-c:v", "mpeg4", str(video))

    with pytest.raises(AudioConversionError, match="Aucune piste audio"):
        convert.probe(video)


def test_load_wav_rejects_wrong_sample_rate(tone):
    wav_44k = tone("stereo.wav", "-c:a", "pcm_s16le")

    with pytest.raises(AudioConversionError, match="mono 16000 Hz"):
        convert.load_wav(wav_44k)
