"""
Configuration constants for audio and video processing.

This module loads environment variables and defines paths and parameters
used throughout the audio diarization and video processing pipeline.
"""

from pathlib import Path
from typing import Optional

HUGGINGFACE_TOKEN: Optional[str]
"""
Hugging Face authentication token for accessing gated models.

Loaded from the HUGGINGFACE_TOKEN environment variable. Required for
downloading and using pyannote models (segmentation, diarization, embeddings).
"""

AUDIO_DB_PATH: Path
"""
Path to the directory containing known speaker voice embeddings.

Each subdirectory represents a known speaker and contains their voice
embedding stored as 'embedding.npy'. Used by AudioEmbedding for speaker
identification and comparison.
"""

AUDIO_THRESHOLD: float
"""
Cosine distance threshold for speaker identification matching.

Embeddings with cosine distance below this threshold are considered matches
to known speakers. Lower values (e.g., 0.3-0.4) are more strict, higher values
(e.g., 0.5-0.6) are more lenient. Default: 0.45
"""

SAMPLE_RATE: int
"""
Audio sample rate in Hz for processing.

Standard sample rate used for audio processing throughout the pipeline.
Default: 16000 Hz (16 kHz)
"""

VIDEO_DB_PATH: Path
"""
Path to the directory containing known face embeddings or images.

Used for face recognition and identification in video processing.
Each subdirectory may represent a known person with their face data.
"""

VIDEO_THRESHOLD: float
"""
Cosine distance threshold for speaker identification matching.

Embeddings with cosine distance below this threshold are considered matches
to known speakers. Lower values (e.g., 0.3-0.4) are more strict, higher values
(e.g., 0.5-0.6) are more lenient. Default: 0.45
"""
