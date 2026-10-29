"""Run the pipeline steps in order, with progress reporting and timing.

The runner depends on the step *interfaces* (base.py) only. `LocalModels` is the
default, fully local provider of those steps; it loads each model on first use
and keeps it in memory, so a long-running process (the server) pays the loading
cost once rather than for every recording.
"""

import gc
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.pipeline import convert, model_registry
from app.pipeline.assign_speakers import attribute_speakers, number_speakers
from app.pipeline.base import Aligner, Denoiser, Diarizer, ProgressCallback, Transcriber
from app.pipeline.denoise import DEEPFILTER_SAMPLE_RATE
from app.pipeline.errors import ModelNotAvailableError
from app.pipeline.hardware import DeviceProfile
from app.pipeline.types import ProcessingInfo, ProcessingOptions, Transcript

STEPS = ("convert", "denoise", "diarize", "transcribe", "align", "assign")
STEP_LABELS = {
    "convert": "Conversion",
    "denoise": "Noise reduction",
    "diarize": "Diarization (who speaks when)",
    "transcribe": "Transcription",
    "align": "Word alignment",
    "assign": "Speaker attribution",
}

# Called as on_progress(step, fraction_of_that_step).
StepProgress = Callable[[str, float], None]


@dataclass
class PipelineResult:
    transcript: Transcript
    audio_path: Path  # 16 kHz, normalised, not denoised
    denoised_audio_path: Path | None  # 16 kHz, normalised, denoised


class LocalModels:
    """Loads the local implementations of the steps lazily and keeps them loaded."""

    def __init__(self, settings: Settings, device: DeviceProfile) -> None:
        self.settings = settings
        self.device = device
        self._transcriber: Transcriber | None = None
        self._aligner: Aligner | None = None
        self._diarizer: Diarizer | None = None
        self._denoiser: Denoiser | None = None

    def transcriber(self, model_id: str) -> Transcriber:
        if self._transcriber is None or self._transcriber.name != model_id:
            from app.pipeline.transcribe import WhisperXTranscriber

            self._transcriber = None  # free the previous model before loading another
            gc.collect()
            path = model_registry.whisper_model_path(model_id, models_dir=self.settings.models_dir)
            self._transcriber = WhisperXTranscriber(model_id, path, self.device)
        return self._transcriber

    def aligner(self) -> Aligner:
        if self._aligner is None:
            from app.pipeline.align import WhisperXAligner

            ready = model_registry.align_model_ready() and model_registry.sentence_splitter_ready()
            if not ready:
                raise ModelNotAvailableError(
                    "The word alignment model has not been downloaded yet.",
                    hint=model_registry.DOWNLOAD_HINT,
                )
            self._aligner = WhisperXAligner(self.device.torch_device)
        return self._aligner

    def diarizer(self) -> Diarizer:
        if self._diarizer is None:
            from app.pipeline.diarize import PyannoteDiarizer

            model_id = self.settings.diarization_model
            path = model_registry.diarization_model_path(model_id, token=None)
            self._diarizer = PyannoteDiarizer(model_id, path, self.device.torch_device)
        return self._diarizer

    def denoiser(self) -> Denoiser:
        if self._denoiser is None:
            from app.pipeline.denoise import DeepFilterNetDenoiser

            binary = model_registry.deepfilter_binary_path(self.settings.bin_dir)
            self._denoiser = DeepFilterNetDenoiser(
                binary, atten_lim_db=self.settings.denoise_atten_lim_db
            )
        return self._denoiser


class TranscriptionPipeline:
    def __init__(self, models: LocalModels) -> None:
        self.models = models

    def run(
        self,
        src: Path,
        work_dir: Path,
        options: ProcessingOptions,
        on_progress: StepProgress | None = None,
    ) -> PipelineResult:
        """Process `src` (left untouched); intermediate audio files go to `work_dir`."""
        settings = self.models.settings
        model_id = options.whisper_model or settings.whisper_model
        model_registry.get_whisper_spec(model_id)  # fail fast on a typo, before any work
        timings: dict[str, float] = {}
        report = on_progress or (lambda step, fraction: None)

        @contextmanager
        def step(name: str) -> Iterator[ProgressCallback]:
            report(name, 0.0)
            started = time.perf_counter()
            yield lambda fraction: report(name, fraction)
            timings[name] = round(time.perf_counter() - started, 2)
            report(name, 1.0)
            _release_accelerator_cache()

        work_dir.mkdir(parents=True, exist_ok=True)
        clean_wav = work_dir / "audio.wav"
        denoised_wav = work_dir / "audio_denoised.wav"

        with step("convert") as progress:
            info = convert.probe(src)
            convert.to_wav(src, clean_wav, duration=info.duration, progress=progress)

        with step("denoise"):
            if options.denoise:
                self._denoise(src, denoised_wav, work_dir)

        audio = convert.load_wav(denoised_wav if options.denoise else clean_wav)

        with step("diarize") as progress:
            turns = (
                self.models.diarizer().diarize(
                    audio,
                    num_speakers=options.num_speakers,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                    progress=progress,
                )
                if options.diarize
                else []
            )

        with step("transcribe") as progress:
            raw_segments = self.models.transcriber(model_id).transcribe(audio, progress)

        with step("align") as progress:
            sentences = self.models.aligner().align(raw_segments, audio, progress)

        with step("assign"):
            segments, speakers = number_speakers(attribute_speakers(sentences, turns))

        transcript = Transcript(
            duration=info.duration,
            speakers=speakers,
            segments=segments,
            info=ProcessingInfo(
                whisper_model=model_id,
                diarization_model=settings.diarization_model if options.diarize else None,
                device=self.models.device.description,
                denoised=options.denoise,
                created_at=datetime.now(UTC),
                step_seconds=timings,
            ),
        )
        return PipelineResult(
            transcript=transcript,
            audio_path=clean_wav,
            denoised_audio_path=denoised_wav if options.denoise else None,
        )

    def _denoise(self, src: Path, dst: Path, work_dir: Path) -> None:
        # DeepFilterNet runs at 48 kHz: original → 48 kHz → denoise → 16 kHz +
        # loudness normalisation (done last, so the noise is not amplified first).
        full_band = work_dir / "tmp_48k.wav"
        full_band_denoised = work_dir / "tmp_48k_denoised.wav"
        try:
            convert.to_wav(src, full_band, sample_rate=DEEPFILTER_SAMPLE_RATE, normalize=False)
            self.models.denoiser().denoise(full_band, full_band_denoised)
            convert.to_wav(full_band_denoised, dst)
        finally:
            full_band.unlink(missing_ok=True)
            full_band_denoised.unlink(missing_ok=True)


def _release_accelerator_cache() -> None:
    # Models stay loaded between steps and jobs, but intermediate tensors cached
    # by PyTorch's allocator are handed back so the next model has room. On a
    # small GPU this is often what prevents an out-of-memory error.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
