"""
Audio and video streaming utilities for WebRTC media processing.

This module provides buffering, recording, and transformation utilities for
real-time audio and video streams from WebRTC. Handles frame buffering,
resampling, and format conversion for downstream processing pipelines.
"""

from typing import AsyncGenerator, Tuple, Optional
from aiortc import MediaStreamTrack
import asyncio
import numpy as np
from numpy.typing import NDArray
from av import AudioFrame, VideoFrame
import torch

class AudioFrameBuffer:
    """
    Asynchronous buffer for accumulating audio frames into processable chunks.

    Collects incoming audio frames and provides them in fixed-duration chunks
    suitable for speech recognition models like Whisper. Automatically handles
    resampling to 16kHz and conversion to mono float32 format.
    """

    sample_rate: int
    chunk_seconds: int
    buffer: asyncio.Queue[Tuple[NDArray[np.float32], NDArray[np.float32]]]
    total_samples: int
    timeout: int
    lock: asyncio.Lock
    ready_event: asyncio.Event
    sample_rate_in: int

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_seconds: int = 30,
        timeout: int = 50,
    ) -> None:
        """
        Initialize the AudioFrameBuffer.

        :param sample_rate: Target sample rate for output audio (default: 16000 Hz)
        :param chunk_seconds: Duration of each audio chunk in seconds (default: 30).
                              30 seconds matches Whisper's training data length
        :param timeout: Maximum seconds to wait for enough audio before flushing
                       incomplete chunks (default: 50)
        """
        ...

    def _preprocess(
        self, frame: AudioFrame
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Convert PyAV AudioFrame to Whisper-compatible format.

        Performs the following transformations:
        1. Converts multi-channel audio to mono by averaging
        2. Converts to float32 format
        3. Normalizes amplitude to [-1.0, 1.0] range
        4. Resamples to 16kHz if necessary

        :param frame: Input AudioFrame from PyAV
        :return: Tuple of (resampled_audio_16kHz, original_sample_rate_audio)
                 Both as 1D float32 numpy arrays
        """
        ...

    async def add_frame(self, frame: AudioFrame) -> None:
        """
        Add an audio frame to the buffer.

        Preprocesses the frame and adds it to the internal queue. Updates
        the total sample count and sets the ready event when enough samples
        are accumulated for a complete chunk.

        :param frame: AudioFrame to add to the buffer
        """
        ...

    def ready(self) -> bool:
        """
        Check if enough audio has been buffered for a complete chunk.

        :return: True if total_samples >= sample_rate * chunk_seconds, False otherwise
        """
        ...

    async def _flush(
        self,
    ) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        """
        Flush all frames from the buffer.

        Internal method that extracts all buffered frames without waiting.
        Used by get_chunk() to retrieve accumulated audio.

        :return: Tuple of (list_of_16kHz_frames, list_of_original_rate_frames)
        """
        ...

    async def get_chunk(
        self,
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]]:
        """
        Get the next chunk of audio, waiting if necessary.

        Waits for either:
        - Enough samples to be buffered (chunk_seconds worth), or
        - Timeout to expire, then returns whatever is available

        After retrieving audio, resets the buffer and sample count.

        :return: Tuple of (concatenated_16kHz_audio, concatenated_original_rate_audio)
                 as 1D numpy arrays, or (None, None) if no audio available
        """
        ...

class AudioFrameRecorder(MediaStreamTrack):
    """
    MediaStreamTrack wrapper that records audio frames to a buffer.

    Passes through audio frames while simultaneously storing them in an
    AudioFrameBuffer for processing. Useful for tee-ing audio streams.
    """

    kind: str
    track: MediaStreamTrack
    buffer: AudioFrameBuffer

    def __init__(self, track: MediaStreamTrack, buffer: AudioFrameBuffer) -> None:
        """
        Initialize the AudioFrameRecorder.

        :param track: Source audio MediaStreamTrack to record from
        :param buffer: AudioFrameBuffer to store frames in
        """
        ...

    async def recv(self) -> AudioFrame:
        """
        Receive and record the next audio frame.

        Retrieves a frame from the source track, adds it to the buffer,
        and returns it for further processing.

        :return: The received AudioFrame
        """
        ...

async def audio_consumer(track: MediaStreamTrack, buffer: AudioFrameBuffer) -> None:
    """
    Consume audio frames from a track and store them in a buffer.

    Creates an AudioFrameRecorder and continuously receives frames until
    the track ends. This is a background task for buffering audio.

    :param track: Source audio MediaStreamTrack to consume from
    :param buffer: AudioFrameBuffer to store frames in

    Note:
        This coroutine runs until the track's readyState becomes "ended".
        Typically launched as a background task with asyncio.create_task().
    """
    ...

def prepare_audio_for_whisper(
    frame: AudioFrame,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert PyAV AudioFrame to Whisper-compatible format.

    Standalone function that performs the same preprocessing as
    AudioFrameBuffer._preprocess(). Converts to mono, float32,
    normalizes, and resamples to 16kHz.

    :param frame: Input AudioFrame from PyAV
    :return: Tuple of (resampled_audio_16kHz, original_sample_rate_audio)
             Both as 1D float32 numpy arrays
    """
    ...

async def audio_transformer(
    track: MediaStreamTrack,
) -> AsyncGenerator[Tuple[torch.Tensor, NDArray[np.float32]], None]:
    """
    Transform audio stream into chunks suitable for processing.

    Creates a buffering pipeline that yields audio chunks as they become
    available. Each chunk is provided as both a PyTorch tensor (for model input)
    and a numpy array (for other processing).

    :param track: Source audio MediaStreamTrack to transform
    :return: AsyncGenerator yielding (tensor, numpy_array) tuples where:
             - tensor: PyTorch tensor of shape (1, samples) at 16kHz, suitable
                      for speech recognition models
             - numpy_array: Numpy array of audio at 16kHz for other processing

    Note:
        This generator runs until the track ends. The background consumer
        task is automatically created. Chunks are yielded when either:
        - chunk_seconds worth of audio is buffered, or
        - The timeout expires with incomplete audio
    """
    ...

class VideoStreamRecorder(MediaStreamTrack):
    """
    MediaStreamTrack that records and buffers video frames with frame skipping.

    Buffers video frames in a queue with configurable frame skipping to reduce
    processing load. Automatically manages queue size and provides frames for
    downstream processing.
    """

    kind: str
    track: MediaStreamTrack
    queue: asyncio.Queue[VideoFrame]
    _skip_ratio: int
    _running: bool

    def __init__(self, track: MediaStreamTrack, maxsize: int = 10) -> None:
        """
        Initialize the VideoStreamRecorder.

        :param track: Source video MediaStreamTrack to record from
        :param maxsize: Maximum number of frames to buffer in the queue (default: 10)

        Note:
            By default, processes every 5th frame (skip_ratio=5) to reduce
            computational load for face recognition tasks.
        """
        ...

    async def recv(self) -> VideoFrame:
        """
        Receive the next video frame from the buffer.

        :return: VideoFrame from the queue

        Note:
            This method blocks until a frame is available in the queue.
        """
        ...

    async def recv_loop(self) -> None:
        """
        Continuously receive frames from source track and add to buffer.

        Background loop that:
        1. Receives frames from the source track
        2. Skips frames according to _skip_ratio (keeps every 5th frame)
        3. Adds frames to the queue, dropping oldest if full
        4. Stops when track ends or stop() is called

        Note:
            This coroutine should be launched as a background task with
            asyncio.create_task(). Runs until the track ends or is stopped.
        """
        ...

    def stop(self) -> None:
        """
        Stop the recorder and clear the frame buffer.

        Sets the running flag to False and empties the frame queue.
        Call this to cleanly shut down the recorder.
        """
        ...
