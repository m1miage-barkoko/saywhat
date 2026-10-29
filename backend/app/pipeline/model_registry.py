"""Catalogue of the AI models SayWhat uses, and where to find them on disk.

Every ``*_path(download=False)`` function only looks in local caches; it raises
:class:`ModelNotAvailableError` if something is missing. Passing
``download=True`` is reserved for ``python -m app.download_models``, the only
command allowed to use the network (see app/privacy.py).

Everything lives in the data folder: downloaded models under ``<data_dir>/models``
(the Hugging Face, PyTorch and NLTK caches are redirected there by
``Settings.use_local_model_dirs``), models we convert under
``<data_dir>/models/ctranslate2`` and the DeepFilterNet binary under ``<data_dir>/bin``.
"""

import hashlib
import platform
import shutil
import stat
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.pipeline.errors import MissingTokenError, ModelNotAvailableError

DOWNLOAD_HINT = "Run this first: uv run python -m app.download_models"

# --------------------------------------------------------------------------
# Transcription (Whisper)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WhisperModelSpec:
    id: str
    label: str
    description: str
    repo_id: str
    # "ctranslate2": the repo ships weights faster-whisper can load directly.
    # "transformers": only PyTorch weights; converted once to CTranslate2.
    format: Literal["ctranslate2", "transformers"]
    download_gb: float
    subfolder: str = ""  # where the CTranslate2 files sit inside the repo


WHISPER_MODELS: dict[str, WhisperModelSpec] = {
    spec.id: spec
    for spec in (
        WhisperModelSpec(
            id="bofenghuang/whisper-large-v3-french",
            label="Whisper large-v3 French",
            description="Best accuracy in French (large-v3 fine-tuned on French).",
            repo_id="bofenghuang/whisper-large-v3-french",
            format="ctranslate2",
            subfolder="ctranslate2",
            download_gb=3.1,
        ),
        WhisperModelSpec(
            id="large-v3",
            label="Whisper large-v3",
            description="Official multilingual model from OpenAI.",
            repo_id="Systran/faster-whisper-large-v3",
            format="ctranslate2",
            download_gb=3.1,
        ),
        WhisperModelSpec(
            id="large-v3-turbo",
            label="Whisper large-v3 turbo",
            description="Multilingual, much faster, slightly less accurate.",
            # Formerly mobiuslabsgmbh/faster-whisper-large-v3-turbo (renamed on the Hub).
            repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
            format="ctranslate2",
            download_gb=1.6,
        ),
        WhisperModelSpec(
            id="eustlb/distil-large-v3-fr",
            label="Distil-Whisper large-v3 French",
            description="Much faster, slightly less accurate. Good without a GPU.",
            repo_id="eustlb/distil-large-v3-fr",
            format="transformers",
            download_gb=1.5,
        ),
    )
}
DEFAULT_WHISPER_MODEL = "bofenghuang/whisper-large-v3-french"

# Files faster-whisper needs. preprocessor_config.json matters: large-v3 models
# use 128 mel bins, and without it faster-whisper would assume 80.
_CT2_FILES = (
    "config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json",
)  # fmt: skip
_TRANSFORMERS_FILES = (
    "config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
    "added_tokens.json", "special_tokens_map.json",
)  # fmt: skip


def get_whisper_spec(model_id: str) -> WhisperModelSpec:
    try:
        return WHISPER_MODELS[model_id]
    except KeyError:
        known = ", ".join(WHISPER_MODELS)
        raise ModelNotAvailableError(
            f"Unknown transcription model: {model_id}", hint=f"Available models: {known}"
        ) from None


def whisper_model_path(model_id: str, *, models_dir: Path, download: bool = False) -> Path:
    """Local folder of CTranslate2 weights for `model_id`, ready for faster-whisper."""
    spec = get_whisper_spec(model_id)
    if spec.format == "ctranslate2":
        prefix = f"{spec.subfolder}/" if spec.subfolder else ""
        root = _hf_snapshot(
            spec.repo_id,
            allow_patterns=[prefix + name for name in _CT2_FILES],
            download=download,
            what=spec.label,
        )
        return root / spec.subfolder

    converted = models_dir / "ctranslate2" / model_id.replace("/", "--")
    if (converted / "model.bin").is_file():
        return converted
    if not download:
        raise ModelNotAvailableError(
            f"The model “{spec.label}” has not been downloaded yet.", hint=DOWNLOAD_HINT
        )
    source = _hf_snapshot(
        spec.repo_id, allow_patterns=list(_TRANSFORMERS_FILES), download=True, what=spec.label
    )
    _convert_to_ctranslate2(source, converted)
    return converted


def _convert_to_ctranslate2(source: Path, target: Path) -> None:
    # One-off conversion (a minute or two). float16 weights on disk; CTranslate2
    # still loads them as int8 on CPU or float16 on GPU at run time.
    from ctranslate2.converters import TransformersConverter

    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    TransformersConverter(
        str(source),
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=True,
    ).convert(str(partial), quantization="float16", force=True)
    partial.replace(target)


# --------------------------------------------------------------------------
# Diarization (pyannote)
# --------------------------------------------------------------------------

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def diarization_model_path(model_id: str, *, token: str | None, download: bool = False) -> Path:
    """Local folder of the pyannote pipeline (config.yaml + sub-models).

    The model is gated: downloading needs a Hugging Face token whose account has
    accepted the model's conditions. Once downloaded, it loads from this folder
    without any token or network access.
    """
    if download and not token:
        raise MissingTokenError(
            "A Hugging Face token is needed to download the diarization model.",
            hint=(
                f"Accept the conditions at https://huggingface.co/{model_id}, create a "
                "(read) token at https://huggingface.co/settings/tokens, then add "
                "HF_TOKEN=... to the .env file (see README)."
            ),
        )
    return _hf_snapshot(model_id, download=download, token=token, what="diarization (pyannote)")


# --------------------------------------------------------------------------
# Word alignment (wav2vec2, through WhisperX)
# --------------------------------------------------------------------------


def align_model_ready() -> bool:
    """True if WhisperX's default French alignment model is in the torch cache."""
    import torch
    import torchaudio
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH

    bundle = getattr(torchaudio.pipelines, DEFAULT_ALIGN_MODELS_TORCH["fr"])
    filename = Path(bundle._path).name  # same name torch.hub uses for its cache file
    return (Path(torch.hub.get_dir()) / "checkpoints" / filename).is_file()


def download_align_model() -> None:
    import whisperx

    whisperx.load_align_model(language_code="fr", device="cpu")


def sentence_splitter_ready() -> bool:
    # WhisperX splits aligned text into sentences with NLTK's punkt tokenizer,
    # and would try to download it on first use if it were missing.
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab/french")
        return True
    except LookupError:
        return False


def download_sentence_splitter() -> None:
    import nltk

    if not nltk.download("punkt_tab", quiet=True):
        raise ModelNotAvailableError("Downloading the sentence splitter (NLTK) failed.")


# --------------------------------------------------------------------------
# Noise suppression (DeepFilterNet standalone binary)
# --------------------------------------------------------------------------

DEEPFILTER_VERSION = "0.5.6"
_DEEPFILTER_URL = (
    "https://github.com/Rikorose/DeepFilterNet/releases/download/v{v}/deep-filter-{v}-{target}"
)
# GitHub does not publish checksums for these assets; we computed them when
# pinning the version so a tampered or truncated download is refused.
_DEEPFILTER_BUILDS: dict[tuple[str, str], tuple[str, str]] = {
    ("Darwin", "arm64"): (
        "aarch64-apple-darwin",
        "4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081",
    ),
    ("Darwin", "x86_64"): (
        "x86_64-apple-darwin",
        "d3be84003acb7c23e738ad7f70a158ec779a8d233a82e7fa3e717d112eb5b50f",
    ),
    ("Linux", "x86_64"): (
        "x86_64-unknown-linux-musl",
        "70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da",
    ),
    ("Linux", "aarch64"): (
        "aarch64-unknown-linux-gnu",
        "14e02a1c0028f3ca0bdf83b62b3336e56ba0556894ef295a95e8573f06557166",
    ),
    ("Windows", "AMD64"): (
        "x86_64-pc-windows-msvc.exe",
        "75e11fa16445f560cb6b021521ddb89e89270d13b83089705d98776f58fd7915",
    ),
}


def deepfilter_binary_path(bin_dir: Path, *, download: bool = False) -> Path:
    key = (platform.system(), platform.machine())
    if key not in _DEEPFILTER_BUILDS:
        raise ModelNotAvailableError(
            f"DeepFilterNet is not available for {key[0]} {key[1]}.",
            hint="Turn noise reduction off for this file.",
        )
    target, sha256 = _DEEPFILTER_BUILDS[key]
    path = bin_dir / f"deep-filter-{DEEPFILTER_VERSION}-{target}"
    if path.is_file():
        return path
    if not download:
        raise ModelNotAvailableError(
            "The noise reduction program (DeepFilterNet) has not been downloaded yet.",
            hint=DOWNLOAD_HINT,
        )
    bin_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    url = _DEEPFILTER_URL.format(v=DEEPFILTER_VERSION, target=target)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != sha256:
        partial.unlink()
        raise ModelNotAvailableError(
            "The downloaded DeepFilterNet file is corrupted (checksum mismatch).",
            hint="Run the download again.",
        )
    partial.chmod(partial.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    partial.replace(path)
    return path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _hf_snapshot(
    repo_id: str,
    *,
    download: bool,
    what: str,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    try:
        return Path(
            snapshot_download(
                repo_id,
                allow_patterns=allow_patterns,
                token=token,
                local_files_only=not download,
            )
        )
    except LocalEntryNotFoundError:
        raise ModelNotAvailableError(
            f"The model “{what}” has not been downloaded yet.", hint=DOWNLOAD_HINT
        ) from None
    except GatedRepoError:
        raise MissingTokenError(
            f"Access denied to the model “{what}”.",
            hint=f"Accept its terms of use at https://huggingface.co/{repo_id} "
            "with the account that owns the HF_TOKEN token.",
        ) from None
