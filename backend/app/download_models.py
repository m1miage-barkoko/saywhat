"""Download the AI models — the only Verbatim command that uses the network.

    uv run python -m app.download_models                         # default model
    uv run python -m app.download_models --model large-v3-turbo  # another one
    uv run python -m app.download_models --all                   # every transcription model

Already-downloaded models are skipped, so it is safe to run again. Afterwards,
Verbatim works entirely offline.
"""

import argparse
import sys
from collections.abc import Callable
from functools import partial

from app.config import get_settings
from app.pipeline import model_registry as registry
from app.pipeline.errors import PipelineError


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    settings.use_local_model_dirs()  # before any download library is imported
    parser = argparse.ArgumentParser(
        prog="python -m app.download_models", description=__doc__.split("\n\n")[0]
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=list(registry.WHISPER_MODELS))
    group.add_argument("--all", action="store_true", help="tous les modèles de transcription")
    args = parser.parse_args(argv)

    default = args.model or settings.whisper_model
    model_ids = list(registry.WHISPER_MODELS) if args.all else [default]
    tasks: list[tuple[str, Callable[[], object]]] = []
    for model_id in model_ids:
        spec = registry.get_whisper_spec(model_id)
        tasks.append(
            (
                f"Transcription : {spec.label} (~{spec.download_gb:g} Go)",
                partial(
                    registry.whisper_model_path,
                    model_id,
                    models_dir=settings.models_dir,
                    download=True,
                ),
            )
        )
    tasks += [
        ("Alignement des mots : wav2vec2 français (360 Mo)", _align_model),
        ("Découpage en phrases : NLTK punkt", registry.download_sentence_splitter),
        (
            f"Diarisation : {settings.diarization_model} (35 Mo)",
            partial(
                registry.diarization_model_path,
                settings.diarization_model,
                token=settings.hf_token,
                download=True,
            ),
        ),
        (
            f"Débruitage : DeepFilterNet {registry.DEEPFILTER_VERSION} (30 Mo)",
            partial(registry.deepfilter_binary_path, settings.bin_dir, download=True),
        ),
    ]

    failures = 0
    for label, task in tasks:
        print(f"• {label}…", flush=True)
        try:
            task()
        except PipelineError as exc:
            failures += 1
            print(f"  ✗ {exc}", flush=True)
        else:
            print("  ✓ prêt", flush=True)

    if failures:
        print(f"\n{failures} élément(s) manquant(s). Corrigez puis relancez cette commande.")
        return 1
    print("\nTout est prêt. Verbatim peut maintenant fonctionner hors ligne.")
    return 0


def _align_model() -> None:
    if not registry.align_model_ready():
        registry.download_align_model()


if __name__ == "__main__":
    sys.exit(main())
