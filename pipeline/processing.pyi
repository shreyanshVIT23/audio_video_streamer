"""
Audio and video processing pipeline for real-time media streams.

This module provides processors for handling audio transcription with speaker
diarization and video face recognition from WebRTC media streams. Processes
are designed to run asynchronously for real-time performance.
"""

from pathlib import Path
from typing import List
from aiortc import MediaStreamTrack
from numpy import ndarray
from pandas import DataFrame
from pipeline.audio_detect import AudioRecognizer
from pipeline.audio_diarization import AudioEmbedding, AudioSegmentationFile
from pipeline.streaming import VideoStreamRecorder

class AudioProcessor:
    """
    Real-time audio processor for speech recognition and speaker identification.

    Processes audio streams to transcribe speech and identify speakers using
    voice activity detection, speaker embeddings, and Whisper transcription.
    Results are written to a conversation transcript file.
    """

    ar: AudioRecognizer
    aseg: AudioSegmentationFile
    aem: AudioEmbedding
    track: MediaStreamTrack
    conversation: Path

    def __init__(self, track: MediaStreamTrack) -> None:
        """
        Initialize the AudioProcessor with a media stream track.

        :param track: WebRTC audio media stream track to process

        Note:
            Creates or appends to 'conversation.txt' for storing transcriptions.
        """
        ...

    async def process_audio(self) -> None:
        """
        Process audio stream for transcription and speaker identification.

        Continuously processes incoming audio chunks:
        1. Detects voice activity segments
        2. Extracts speaker embeddings for each segment
        3. Identifies or registers new speakers
        4. Transcribes speech using Whisper
        5. Writes results to conversation file with speaker labels

        Note:
            This is a long-running coroutine that processes audio until
            the stream ends. Uses asyncio.to_thread for CPU-intensive operations.
        """
        ...

class VideoProcessor:
    """
    Real-time video processor for face detection and recognition.

    Processes video streams to detect and identify faces, tracking when
    people enter or leave the video frame. Maintains a database of known
    faces and registers new faces automatically.
    """

    recorder: VideoStreamRecorder

    def __init__(self, track: MediaStreamTrack) -> None:
        """
        Initialize the VideoProcessor with a media stream track.

        :param track: WebRTC video media stream track to process
        """
        ...

    async def start(self) -> None:
        """
        Start the video processing pipeline.

        Launches two concurrent tasks:
        1. Video frame recording/buffering
        2. Face detection and recognition processing

        Note:
            This coroutine runs until the video stream ends. Both tasks
            run concurrently using asyncio.gather.
        """
        ...

    def _show_frame(self, frame: ndarray, text: str) -> None:
        """
        Display a video frame in an OpenCV window.

        :param frame: Video frame as numpy array (BGR format)
        :param text: Window title/name for the display

        Note:
            Helper method for debugging/visualization. Uses cv2.imshow
            with a 1ms wait to allow window updates.
        """
        ...

    async def process_video(self) -> None:
        """
        Process video frames for face detection and person tracking.

        Continuously processes video frames:
        1. Extracts faces from each frame
        2. Identifies faces against known database
        3. Registers new faces when encountered
        4. Tracks people entering/leaving the frame
        5. Writes person detection events to 'all_people.txt'

        The processor maintains state of currently visible people and
        logs when new people appear or when people leave the frame.

        Note:
            This is a long-running coroutine that processes frames until
            the stream ends. Uses asyncio.to_thread for CPU-intensive
            face recognition operations. Displays live video feed in
            an OpenCV window for monitoring.
        """
        ...
