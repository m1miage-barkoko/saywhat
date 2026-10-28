"""Errors whose message is meant for the end user (printed as is by the CLI)."""


class PipelineError(Exception):
    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n→ {self.hint}" if self.hint else self.message


class AudioConversionError(PipelineError):
    """The input file is unreadable, corrupted or has no audio track."""


class DenoiseError(PipelineError):
    pass


class ModelNotAvailableError(PipelineError):
    """A model is unknown, or has not been downloaded yet."""


class MissingTokenError(PipelineError):
    """The gated pyannote model needs a Hugging Face token to be downloaded."""
