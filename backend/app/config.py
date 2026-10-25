"""Application settings, read from environment variables and from ``<repo>/.env``."""

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
        """Models we convert ourselves (downloaded models stay in the HF cache)."""
        return self.data_dir / "models"

    @property
    def bin_dir(self) -> Path:
        return self.data_dir / "bin"

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
