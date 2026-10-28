# What Verbatim installs and generates, and how to remove it all

Verbatim keeps as much as possible **inside the project folder**. A few tools cannot live there: they are listed here with the commands to remove them.

## 1. Inside the project folder (removed with it)

| Location | Contents | Size (measured on macOS) |
|---|---|---|
| `backend/.venv/` | Python environment: PyTorch, WhisperX, pyannote… | about 1.1 GB |
| `data/models/huggingface/` | Whisper and pyannote models | 3 to 4.5 GB depending on the models downloaded |
| `data/models/ctranslate2/` | converted models (`eustlb/distil-large-v3-fr`) | about 1.4 GB per model |
| `data/models/torch/` | word alignment model (wav2vec2) | about 370 MB |
| `data/models/nltk/` | sentence splitting | about 15 MB |
| `data/bin/` | DeepFilterNet noise reduction program | about 30 MB |
| `data/cache/` | Matplotlib font cache | < 1 MB |
| `data/cli/` | command-line results (converted audio, JSON) | depends on use |
| `backend/.pytest_cache/`, `backend/.ruff_cache/`, `__pycache__/` folders | test, lint and Python bytecode caches | a few MB |
| `.env` | your settings and the Hugging Face token | — |

To remove a single model, delete its folder `data/models/huggingface/hub/models--<author>--<name>` (or `data/models/ctranslate2/<author>--<name>`). `uv run python -m app.download_models` downloads it again if it is needed later.

To free space without uninstalling, you can delete `data/cli/` (old command-line results) and the cache folders listed above at any time: they are recreated when needed.

Verbatim redirects the Hugging Face, PyTorch, NLTK and Matplotlib caches to `data/` (`Settings.use_local_model_dirs` in `backend/app/config.py`): it writes **nothing** to `~/.cache/huggingface`, `~/.cache/torch`, `~/nltk_data` or `~/.matplotlib`.

## 2. Outside the project

### Tools installed with Homebrew (macOS)

| Formula | Purpose |
|---|---|
| `ffmpeg` | audio file conversion |
| `uv` | installs Python and the dependencies |

`brew install ffmpeg` also installs the libraries ffmpeg needs, if they are not already present. On the development Mac (25/10/2026) it added 11 formulas: `mpg123`, `lame`, `ca-certificates`, `libvpx`, `openssl@4`, `opus`, `sdl2-compat`, `sdl3`, `svt-av1`, `x264`, `x265` (152 MB in total with ffmpeg and uv).

### Files created by uv

uv can only put these two folders in the project through environment variables that would have to be set before every command, so they stay in their standard location.

| Location | Contents | Size | Needed? |
|---|---|---|---|
| `~/.local/share/uv/python/` | the Python 3.12 interpreter used by `backend/.venv` | about 70 MB | yes, as long as the project is used |
| `~/.cache/uv/` | cache of downloaded packages | up to about 1 GB | no: `uv cache clean` empties it without breaking anything |

### Files created by macOS when the GPU is used (Apple Silicon)

On a Mac, diarization and word alignment run on the GPU through Apple's Metal framework. Each time it compiles a computation, Metal creates an **empty** folder in the system temporary folder and does not remove it:

| Location | Contents |
|---|---|
| `$(getconf DARWIN_USER_TEMP_DIR)com.apple.MetalPerformanceShadersGraph/` | empty `mpsgraph-…` folders (0 bytes of data, several thousand folders after a few runs) |

This folder is not under Verbatim's control (it ignores `TMPDIR`). **Restarting the Mac removes it** (verified): you only need to delete it by hand if you don't want to restart. It can be deleted at any time, and the next transcription creates it again.

### Hugging Face account

The token (`HF_TOKEN`) is only used for the first pyannote download. Once the models are downloaded, you can remove it from `.env` and revoke it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 3. Removing everything (macOS)

In this order (uv must still be installed for step 1):

```bash
# 1. uv's Python interpreter and cache
uv python uninstall 3.12
uv cache clean

# 2. The project itself (environment, models, data, caches, .env)
rm -rf path/to/verbatim      # for example ~/IdeaProjects/verbatim

# 3. Empty folders left by Apple's Metal framework (or simply restart the Mac)
rm -rf "$(getconf DARWIN_USER_TEMP_DIR)com.apple.MetalPerformanceShadersGraph"

# 4. ffmpeg and uv, then the dependencies nothing needs any more
brew uses --installed ffmpeg   # check that no other software uses ffmpeg
brew uninstall ffmpeg uv
brew autoremove
```

`brew autoremove` only removes formulas installed as dependencies that no other software needs any more: the ones other programs use stay in place.

On Linux or Windows, replace step 4 with uninstalling ffmpeg and uv with the package manager you used to install them (`apt`, `winget`…), and skip step 3. uv's locations become `~/.local/share/uv` and `~/.cache/uv` (Linux) or `%APPDATA%\uv` and `%LOCALAPPDATA%\uv\cache` (Windows).
