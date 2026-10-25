"""Privacy guards: audio, text *and* usage metadata never leave this machine.

Two of our dependencies contact remote servers by default:

* pyannote.audio 4.x sends usage telemetry (model names, audio duration, number of
  speakers) to otel.pyannote.ai unless ``PYANNOTE_METRICS_ENABLED`` is false.
* huggingface_hub sends anonymous telemetry, and checks the Hub for newer model
  revisions every time a model is loaded.

Both libraries read these switches from the environment when they are *imported*,
so this module only touches environment variables and must run first:
``app/__init__.py`` calls :func:`disable_telemetry`, and the entry points that run
inference (CLI, server) call :func:`enforce_offline` before importing any model code.
Downloading models is the only operation allowed to use the network, and it only
happens through ``python -m app.download_models``.
"""

import os
import sys

_TELEMETRY_OFF = {
    "PYANNOTE_METRICS_ENABLED": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def disable_telemetry() -> None:
    # Forced rather than setdefault: telemetry is never wanted in this app,
    # whatever the user's shell or a library's saved preference says.
    os.environ.update(_TELEMETRY_OFF)


def enforce_offline() -> None:
    """Forbid Hugging Face libraries from using the network for this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and not constants.HF_HUB_OFFLINE:
        # huggingface_hub froze its settings when this module was imported, so
        # setting the variable now would silently do nothing. Fail loudly instead.
        raise RuntimeError("enforce_offline() must be called before huggingface_hub is imported")
