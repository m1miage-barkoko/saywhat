"""Application settings, read from environment variables and from ``<repo>/.env``."""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.pipeline.model_registry import DEFAULT_DIARIZATION_MODEL, DEFAULT_WHISPER_MODEL

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]

# Folders that a desktop client mirrors to the cloud. Storing recordings there
# would silently upload them, which defeats the "everything stays local" promise.
_SYNCED_FOLDER_MARKERS = ("Mobile Documents", "CloudStorage", "Dropbox", "OneDrive", "Google Drive")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_DIR / ".env",
        env_prefix="VERBATIM_",
        extra="ignore",
    )

    data_dir: Path = REPO_DIR / "data"
    # Only needed once, to download the gated pyannote model (see README).
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "VERBATIM_HF_TOKEN")
    )
    whisper_model: str = DEFAULT_WHISPER_MODEL
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    device: DevicePreference = "auto"
    # None = pick a sensible value for the detected hardware (see hardware.py).
    batch_size: int | None = None
    cpu_threads: int | None = None
    # DeepFilterNet attenuation limit in dB. None = full noise removal; a lower
    # value (e.g. 20) keeps a bit of background and causes fewer artefacts.
    denoise_atten_lim_db: float | None = None

    @property
    def models_dir(self) -> Path:
        """Every model Verbatim downloads or converts (see use_local_model_dirs)."""
        return self.data_dir / "models"

    @property
    def bin_dir(self) -> Path:
        return self.data_dir / "bin"

    def use_local_model_dirs(self) -> None:
        """Make Hugging Face, PyTorch, NLTK and Matplotlib keep their files in `data/`.

        By default each library writes to its own folder in the user's home
        (~/.cache/huggingface, ~/.cache/torch, ~/nltk_data, ~/.matplotlib). Redirecting
        them here keeps everything Verbatim downloads inside the project, so deleting
        the project removes it all. Most of these libraries read their variable once,
        when imported: entry points call this before importing any model code.
        """
        targets = {  # variable: (folder, module that reads it at import time, if any)
            "HF_HOME": (self.models_dir / "huggingface", "huggingface_hub.constants"),
            "TORCH_HOME": (self.models_dir / "torch", None),  # read on every use
            "NLTK_DATA": (self.models_dir / "nltk", "nltk.data"),
            "MPLCONFIGDIR": (self.data_dir / "cache" / "matplotlib", "matplotlib"),
        }
        for variable, (folder, reader) in targets.items():
            path = str(folder.resolve())
            if os.environ.get(variable) == path:
                continue  # already done (calling twice is harmless)
            if reader in sys.modules:
                # The library already read the old value: setting it now would be
                # silently ignored and files would land in the home folder.
                raise RuntimeError(f"use_local_model_dirs() must run before {reader} is imported")
            folder.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = path

    def synced_folder_warning(self) -> str | None:
        """A French warning if the data folder is inside a cloud-synced folder."""
        resolved = self.data_dir.resolve()
        # iCloud skips any folder whose name ends in ".nosync".
        if any(part.endswith(".nosync") for part in resolved.parts):
            return None
        if any(marker in str(resolved) for marker in _SYNCED_FOLDER_MARKERS):
            return (
                f"Attention : le dossier de données ({resolved}) se trouve dans un dossier "
                "synchronisé avec le cloud. Vos enregistrements y seraient envoyés. "
                "Déplacez le projet ou définissez VERBATIM_DATA_DIR dans .env."
            )
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
