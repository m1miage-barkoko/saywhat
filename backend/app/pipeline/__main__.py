"""Command-line pipeline (milestone 1).

    uv run python -m app.pipeline enregistrement.m4a [options]

Prints the speaker-labelled transcript on stdout, progress on stderr, and writes
the full result (words, timestamps, speakers) as JSON, plus the transcript as
text with and without speakers.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from app import privacy
from app.config import get_settings
from app.pipeline.errors import PipelineError
from app.pipeline.model_registry import WHISPER_MODELS
from app.pipeline.runner import STEP_LABELS, STEPS, LocalModels, TranscriptionPipeline
from app.pipeline.segments import format_plain_text, format_transcript
from app.pipeline.types import ProcessingOptions


def main(argv: list[str] | None = None) -> int:
    # First thing, before any model code imports huggingface_hub (both raise otherwise).
    privacy.enforce_offline()
    settings = get_settings()
    settings.use_local_model_dirs()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        options = ProcessingOptions(
            denoise=not args.no_denoise,
            diarize=not args.no_diarize,
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            whisper_model=args.model,
        )
    except ValidationError as exc:
        parser.error("; ".join(err["msg"].removeprefix("Value error, ") for err in exc.errors()))

    if warning := settings.synced_folder_warning():
        print(warning, file=sys.stderr)

    from app.pipeline.hardware import detect_device  # imports torch: a few seconds

    device = detect_device(
        args.device or settings.device,
        batch_size=settings.batch_size,
        cpu_threads=settings.cpu_threads,
    )
    print(f"Hardware: {device.description}", file=sys.stderr)
    print(f"Model: {options.whisper_model or settings.whisper_model}", file=sys.stderr)

    work_dir = args.work_dir or settings.data_dir / "cli" / args.input.stem
    skipped = {"denoise"} if args.no_denoise else set()
    if args.no_diarize:
        skipped.add("diarize")
    progress = _ConsoleProgress(skipped)
    started = time.perf_counter()
    try:
        result = TranscriptionPipeline(LocalModels(settings, device)).run(
            args.input, work_dir, options, progress
        )
    except PipelineError as exc:
        progress.interrupt()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        progress.interrupt()
        print("Interrupted.", file=sys.stderr)
        return 130

    transcript = result.transcript
    output = args.output or work_dir / "transcript.json"
    text_output = output.with_suffix(".txt")
    plain_output = output.with_name(f"{output.stem}_plain.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    text_output.write_text(format_transcript(transcript) + "\n", encoding="utf-8")
    plain_output.write_text(format_plain_text(transcript) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(
        f"\n{len(transcript.speakers)} speaker(s), {len(transcript.segments)} segments. "
        f"{transcript.duration:.0f} s of audio processed in {elapsed:.0f} s "
        f"(×{transcript.duration / elapsed:.1f} real time).",
        file=sys.stderr,
    )
    print(f"JSON: {output}", file=sys.stderr)
    print(f"Text: {text_output}", file=sys.stderr)
    print(f"Plain text: {plain_output}", file=sys.stderr)
    print(f"Original audio: {result.original_audio_path}", file=sys.stderr)
    print(f"Audio: {result.audio_path}", file=sys.stderr)
    if result.denoised_audio_path:
        print(f"Denoised audio: {result.denoised_audio_path}", file=sys.stderr)
    print(file=sys.stderr)
    print(format_transcript(transcript))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline",
        description=(
            "Transcribe a French recording and separate the speakers. "
            "Everything runs on this computer."
        ),
    )
    parser.add_argument("input", type=Path, help="audio file (m4a, mp3, wav, ogg, amr, 3gp…)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON file to write; the .txt files go next to it (default: in --work-dir)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="folder for intermediate audio files (default: data/cli/<file name>)",
    )
    parser.add_argument(
        "--model", choices=list(WHISPER_MODELS), help="transcription model (default: from .env)"
    )
    parser.add_argument("--no-denoise", action="store_true", help="skip noise reduction")
    parser.add_argument(
        "--no-diarize", action="store_true", help="do not separate the speakers (faster)"
    )
    parser.add_argument("--speakers", type=int, help="exact number of speakers, if known")
    parser.add_argument("--min-speakers", type=int, help="minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="maximum number of speakers")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], help="force the hardware to use"
    )
    return parser


class _ConsoleProgress:
    """One line per step; updated in place when stderr is a terminal."""

    def __init__(self, skipped: set[str], stream: TextIO = sys.stderr) -> None:
        self._skipped = skipped
        self._stream = stream
        self._live = stream.isatty()
        self._done: set[str] = set()
        self._started: dict[str, float] = {}
        self._pending_line = False

    def __call__(self, step: str, fraction: float) -> None:
        if step in self._done:
            return
        self._started.setdefault(step, time.perf_counter())
        label = f"[{STEPS.index(step) + 1}/{len(STEPS)}] {STEP_LABELS[step]}"
        if fraction >= 1.0:
            self._done.add(step)
            status = (
                "skipped"
                if step in self._skipped
                else f"done in {time.perf_counter() - self._started[step]:.1f} s"
            )
            line = f"\r{label:<42} {status}\033[K" if self._live else f"{label}: {status}"
            self._write(line + "\n")
            self._pending_line = False
        elif self._live:
            self._write(f"\r{label:<42} {fraction:4.0%}\033[K")
            self._pending_line = True

    def interrupt(self) -> None:
        if self._pending_line:
            self._write("\n")

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()


if __name__ == "__main__":
    sys.exit(main())
