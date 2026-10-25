"""Settings.use_local_model_dirs, checked in fresh interpreters (import order matters)."""

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HF_HOME", "TORCH_HOME", "NLTK_DATA", "MPLCONFIGDIR")
    }
    return subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True, text=True, env=env
    )


def test_libraries_store_their_files_in_the_data_folder(tmp_path):
    result = run_python(
        "from pathlib import Path\n"
        "from app.config import Settings\n"
        f"Settings(data_dir=Path({str(tmp_path)!r})).use_local_model_dirs()\n"
        "import huggingface_hub.constants, nltk.data, torch.hub, matplotlib\n"
        "print(huggingface_hub.constants.HF_HUB_CACHE)\n"
        "print(nltk.data.path[0])\n"
        "print(torch.hub.get_dir())\n"
        "print(matplotlib.get_cachedir())\n"
    )

    data = tmp_path.resolve()
    assert result.stdout.splitlines() == [
        str(data / "models" / "huggingface" / "hub"),
        str(data / "models" / "nltk"),
        str(data / "models" / "torch" / "hub"),
        str(data / "cache" / "matplotlib"),
    ], result.stderr


def test_calling_twice_is_harmless(tmp_path):
    result = run_python(
        "from pathlib import Path\n"
        "from app.config import Settings\n"
        f"s = Settings(data_dir=Path({str(tmp_path)!r}))\n"
        "s.use_local_model_dirs()\n"
        "import huggingface_hub.constants\n"
        "s.use_local_model_dirs()\n"
        "print('ok')"
    )
    assert result.stdout.strip() == "ok", result.stderr


def test_refuses_to_run_after_a_library_read_the_old_value(tmp_path):
    result = run_python(
        "import huggingface_hub.constants\n"
        "from pathlib import Path\n"
        "from app.config import Settings\n"
        f"Settings(data_dir=Path({str(tmp_path)!r})).use_local_model_dirs()"
    )
    assert result.returncode != 0
    assert "must run before huggingface_hub.constants is imported" in result.stderr
