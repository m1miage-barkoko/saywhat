"""Verbatim — local French conversation transcriber (backend)."""

from app import privacy

# Must run before pyannote / huggingface_hub are imported anywhere: both read
# their telemetry switches from the environment at import time.
privacy.disable_telemetry()
