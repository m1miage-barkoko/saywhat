# Verbatim

**Local** transcription of French conversations, with speaker separation.

Give it a phone recording (m4a, mp3, amr…) and Verbatim returns readable text where each voice is identified (`Speaker 1`, `Speaker 2`…):

```
[00:00:00] Speaker 1: Bonjour Claire, alors pour la livraison de mardi, on en est où ?
[00:00:04] Speaker 2: Bonjour, oui, il faudra prévoir 2 palettes, et le camion arrive vers 10 heures.
```

> **Status: milestone 1 of 6.** The pipeline works from the command line. The API, web interface, editing and exports come in the next milestones.
> This version transcribes French only. Other languages are planned for later versions.

## Privacy

Recordings contain other people's voices and confidential information.

- **Everything runs on your computer.** No audio and no text is sent to an external service.
- Only **one command** uses the network: `python -m app.download_models`, which downloads the models once. After that, Verbatim forces Hugging Face's offline mode (`app/privacy.py`) and works without a connection.
- **Telemetry is forcibly disabled.** By default, pyannote.audio 4 sends usage statistics (file duration, number of speakers) to `otel.pyannote.ai`, and Hugging Face sends its own. Verbatim turns both off before they are loaded (tests check this).
- Data is stored in `data/` (configurable). **Do not put this folder in a synced folder** (iCloud Drive, Dropbox, OneDrive…): your recordings would be uploaded. Verbatim shows a warning if it is.

## Compatibility

| System | Status |
|---|---|
| macOS on Apple Silicon (M1 and later) | **tested** (MacBook Air M4, 16 GB) |
| macOS on Intel | **not supported**: PyTorch 2.8 no longer publishes Intel Mac builds |
| Linux (x86_64, arm64) | configured, not tested yet |
| Windows (x86_64) | configured, not tested yet |

Linux and Windows will be tested in milestone 6. On those systems an NVIDIA GPU makes everything faster but is not required.

- **Memory**: 16 GB recommended. The measured peak is 5 GB, so 8 GB may work, but this has not been tested.
- **Disk space**: about 6 GB free (1 GB of Python dependencies, about 4 GB of models), plus the recordings you transcribe.
- **Internet**: needed only once, for the installation and the model download.

## Installation

> **Not a developer?** Follow the step-by-step guide instead: [how-to-install.md](how-to-install.md).

Check [Compatibility](#compatibility) first.

1. **Prerequisites**: [ffmpeg](https://ffmpeg.org) and [uv](https://docs.astral.sh/uv/) (which installs the right Python version by itself).

   | System | Command |
   |---|---|
   | macOS | `brew install ffmpeg uv` |
   | Linux (Debian/Ubuntu) | `sudo apt install ffmpeg`, then `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
   | Windows | `winget install Gyan.FFmpeg astral-sh.uv` |

2. **Python dependencies** (about 1 GB, including PyTorch):

   ```bash
   cd backend
   uv sync
   ```

3. **Hugging Face token** (see below), then **download the models** (about 4 GB, stored in `data/models/`: nothing is written to your home folder):

   ```bash
   uv run python -m app.download_models
   ```

   Safe to run again: anything already downloaded is skipped.

### Hugging Face token

The diarization model [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) is free but "gated": you must accept its terms of use.

1. Create an account on [huggingface.co](https://huggingface.co/join).
2. Log in, open the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1), fill in the short form at the top of the page and click **Agree and access repository**.
3. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. At the project root: `cp .env.example .env`, then set `HF_TOKEN=hf_...`.

The token is only used for the download. Afterwards the model loads from disk, with no token and no network. You can then remove it from `.env` and revoke it.

## Usage (milestone 1: command line)

```bash
cd backend
uv run python -m app.pipeline ~/Downloads/meeting.m4a
```

The text is printed in the terminal. Everything is saved in `data/cli/<file name>/`:

| File | Contents |
|---|---|
| `transcript.json` | the full result: words, timestamps, speakers |
| `transcript.txt` | the text as printed, one line per speaker turn with timestamp and speaker |
| `transcript_plain.txt` | the words only, one paragraph per speaker turn |
| `original.<ext>` | an unchanged copy of the recording |
| `audio.wav`, `audio_denoised.wav` | the converted and denoised audio, to compare them |

The original file is never modified.

| Option | Effect |
|---|---|
| `--speakers 3` | exact number of speakers, if known (greatly improves separation) |
| `--min-speakers 2 --max-speakers 4` | or a range |
| `--no-denoise` | skip noise reduction (it sometimes makes transcription worse) |
| `--model large-v3-turbo` | another transcription model (see below) |
| `--no-diarize` | do not separate the speakers (faster) |
| `-o result.json` | where to write the JSON (the `.txt` files go next to it) |

## Choosing a model

| Model | Accuracy | Measured speed (Apple M4, CPU) | Download |
|---|---|---|---|
| `bofenghuang/whisper-large-v3-french` (default) | best in French | ×2.5 real time | 3.1 GB |
| `large-v3-turbo` | very good | ×4.5 | 1.6 GB |
| `eustlb/distil-large-v3-fr` | good | ×4.6 | 1.5 GB (converted when downloaded) |
| `large-v3` | very good (multilingual) | ≈ ×2.5 | 3.1 GB |

"×2.5 real time" means one hour of audio is transcribed in about 24 min. Measured on 3 min of dialogue, with the model already loaded, 6 threads and batches of 8. Diarization and alignment come on top.

**Total time for a 1 h 30 recording** (MacBook Air M4, 16 GB):

| Step | `large-v3-turbo` (measured) | default model |
|---|---|---|
| Conversion | 1 min | 1 min |
| Noise reduction | 4 min | 4 min |
| Diarization (GPU) | 7 min | 7 min |
| Transcription | 20 min | ≈ 36 min (estimated at ×2.5) |
| Word alignment | 2 min | 2 min |
| **Total** | **34 min** | **≈ 50 min** (estimate) |

Measured peak memory: 5 GB. The default model stays the most accurate one: accuracy comes before speed. For a faster result on a given file: `--model large-v3-turbo`.

- **NVIDIA GPU**: everything runs on the GPU in float16; keep the default model.
- **Apple Silicon Mac**: diarization and alignment use the GPU (Metal), but Whisper runs on the CPU (int8), because its engine, CTranslate2, has no Metal backend.
- **CPU only**: `large-v3-turbo` is twice as fast for similar accuracy. On a CPU the gain of the "fast" models is capped at about ×2: they make the decoder lighter, but the encoder, the most expensive part, is still large-v3's.

## How it works

```
original.m4a ─ffmpeg─► audio.wav (16 kHz mono, loudness normalised)
     └─ffmpeg 48 kHz─► DeepFilterNet3 ─► audio_denoised.wav
                                              │
          ┌───────────────────────────────────┴───────────────────────┐
  pyannote community-1                            faster-whisper (WhisperX, VAD + batches)
  who speaks when (exclusive diarization)                     │
          │                                       wav2vec2: timestamp for each word
          └──────────────► word → speaker attribution ◄───────┘
                           sentences split where the voice changes
```

Each step is an independent module in `backend/app/pipeline/` behind a common interface (`base.py`): one engine (for example an opt-in cloud service, later) can replace another without touching the rest.

A few non-obvious choices (explained in the code):

- **Exclusive diarization**: pyannote is called directly, because WhisperX only exposes regular diarization, where overlapping speech gives several speakers at once. The exclusive version gives a single speaker at each instant, which is what we need to give each word to one person (`assign_speakers.py`).
- **Untimed words**: the French alignment model's alphabet has no digits, so "1200" gets no timestamp and takes the speaker of the neighbouring word.
- **Small errors at boundaries**: a lone word attributed to another voice in the middle of a sentence is treated as a diarization inaccuracy and given back to its neighbours.
- **DeepFilterNet as a binary**: the `deepfilternet` Python package (last released in 2023) requires numpy < 2, which is incompatible with WhisperX (numpy ≥ 2.1). Verbatim uses the standalone program published by its authors (same DeepFilterNet3 model), checked against a pinned checksum.

## Pinned versions

| Package | Version | Why |
|---|---|---|
| whisperx | 3.8.6 | latest stable release (3.8.7 is a release candidate); uses pyannote 4 |
| torch / torchaudio | 2.8.0 | required by whisperx 3.8.6 (`~=2.8.0`). The `weights_only` pitfall of torch ≥ 2.6 does not affect pyannote 4: its models load correctly with torch 2.8 (verified). |
| pyannote.audio | 4.0.7 | needed for community-1 and its exclusive diarization |
| faster-whisper | 1.2.1 | latest release; whisperx requires ≥ 1.2 |
| ctranslate2 | 4.8.2 | recent but proven (4.8.4 was released the day the versions were pinned) |
| DeepFilterNet | 0.5.6 | latest published release (standalone binary) |

All transitive dependencies are locked in `backend/uv.lock`.

FFmpeg: any recent version. pyannote warns that `torchcodec` cannot find FFmpeg 4 to 7. This does not matter: Verbatim always hands it audio already decoded in memory (the warning is hidden).

## Uninstalling

Everything Verbatim installs or generates, inside the project and outside it, and the commands to remove it: [UNINSTALL.md](UNINSTALL.md).

## Development

```bash
cd backend
uv run pytest                 # fast tests (about 5 s)
uv run pytest -m models       # end to end with the real models (about 1 min)
uv run ruff check . && uv run ruff format .
```

`tests/data/dialogue_fr.m4a` is a 32 s French dialogue between two synthetic voices with background noise, generated by `backend/scripts/make_test_audio.sh` (macOS). To test with a real recording, just use the usage command above.

## Troubleshooting

| Problem | Solution |
|---|---|
| `ffmpeg not found` | install ffmpeg (see Installation), then reopen the terminal |
| `… has not been downloaded yet` | `uv run python -m app.download_models` |
| `A Hugging Face token is needed` / `Access denied` | see "Hugging Face token"; check that the terms were accepted with the same account as the token |
| Poor transcription of a noisy file | run it again with `--no-denoise` and compare |
| "cloud-synced folder" warning | move the project out of iCloud Drive / Dropbox / OneDrive, or set `VERBATIM_DATA_DIR` in `.env` |
