"""The guarantees of app/privacy.py, checked in fresh interpreters (import order matters)."""

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def run_python(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        # Inherit the environment minus anything that would pre-set what we test.
        env={k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"} | env,
    )


def test_importing_the_app_disables_telemetry_even_if_enabled_in_the_shell():
    result = run_python(
        "import os, app; print(os.environ['PYANNOTE_METRICS_ENABLED'], "
        "os.environ['HF_HUB_DISABLE_TELEMETRY'])",
        PYANNOTE_METRICS_ENABLED="true",
    )
    assert result.stdout.split() == ["false", "1"], result.stderr


def test_cli_imports_do_not_load_huggingface_hub_before_going_offline():
    # The CLI calls enforce_offline() inside main(), after its module imports:
    # that only works if none of those imports pulls in huggingface_hub.
    result = run_python(
        "import sys, app.pipeline.__main__; print('huggingface_hub' in sys.modules)"
    )
    assert result.stdout.strip() == "False", result.stderr


def test_enforce_offline_refuses_to_run_too_late():
    result = run_python(
        "from huggingface_hub import snapshot_download\n"
        "from app import privacy\nprivacy.enforce_offline()",
    )
    assert result.returncode != 0
    assert "must be called before huggingface_hub is imported" in result.stderr


def test_enforce_offline_blocks_hub_requests():
    result = run_python(
        "from app import privacy\nprivacy.enforce_offline()\n"
        "from huggingface_hub import constants\nprint(constants.HF_HUB_OFFLINE)"
    )
    assert result.stdout.strip() == "True", result.stderr
