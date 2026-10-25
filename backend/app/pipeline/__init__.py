"""Audio → speaker-labelled French transcript.

Steps (each in its own module, each independently testable):

    convert → denoise → diarize → transcribe → align → assign_speakers

Importing this package is cheap: modules that need torch / WhisperX / pyannote
import them lazily, so the CLI and the API can configure privacy settings and
start quickly.
"""

import logging
import warnings

# pyannote (used by our diarizer and by WhisperX's voice-activity detector) warns
# at import when torchcodec cannot find a supported FFmpeg (4 to 7). Irrelevant
# here: we always hand pyannote in-memory waveforms, never files, so Verbatim
# works with whatever FFmpeg version is installed.
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly")

# WhisperX installs its own INFO-level handler on *stdout* unless its logger
# already has one, which would mix progress chatter into the transcript the CLI
# prints there. A NullHandler prevents that; warnings still reach the root logger.
_whisperx_log = logging.getLogger("whisperx")
_whisperx_log.addHandler(logging.NullHandler())
_whisperx_log.setLevel(logging.WARNING)
# Lightning notes a harmless checkpoint-format upgrade every time the VAD loads.
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.WARNING)
