from typing import AsyncGenerator, Tuple
from aiortc import MediaStreamTrack
import asyncio
import librosa
import numpy as np
from av import AudioFrame, VideoFrame
import torch
import torchaudio


class AudioFrameBuffer:
    def __init__(
        self,
        sample_rate=16000,
        chunk_seconds=30,
        timeout=50,
    ) -> None:  # The chunk seconds can be reduuced or increased based on the need
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds  # Here we use 30s as it is the length of whisper's audio transcription it was trained on
        self.buffer = asyncio.Queue()
        self.total_samples = 0
        self.timeout = timeout
        self.lock = asyncio.Lock()
        self.ready_event = asyncio.Event()
        self.sample_rate_in: int

    def _preprocess(self, frame: AudioFrame):
        """
        Convert PyAV AudioFrame to Whisper-compatible mono float32 waveform at 16 kHz.
        Input shape: (channels, samples)
        Output shape: (n_samples,) 1D NumPy array
        """
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)

        arr = arr.astype(np.float32)

        if np.abs(arr).max() > 1.0:
            arr = arr / np.abs(arr).max()

        arr2 = arr.copy()
        if frame.sample_rate != 16000:
            arr = librosa.resample(arr, orig_sr=frame.sample_rate, target_sr=16000)

        return arr, arr2

    async def add_frame(
        self,
        frame: AudioFrame,
        # preprocess: Optional[Callable[[AudioFrame], np.ndarray]] = None,
    ):
        # if preprocess:
        frame_np, frame_tensor = self._preprocess(frame)
        self.sample_rate_in = frame.sample_rate
        await self.buffer.put((frame_np, frame_tensor))
        async with self.lock:
            self.total_samples += frame_np.shape[0]
        # else:
        #     await self.buffer.put(frame)
        #     async with self.lock:
        #         self.total_samples += frame.shape[0]
        if self.ready():
            self.ready_event.set()

    def ready(self):
        return self.total_samples >= self.sample_rate * self.chunk_seconds

    async def _flush(self):
        frames_np = []
        frames_tensor = []
        while not self.buffer.empty():
            try:
                frame_np, frame_tensor = self.buffer.get_nowait()
                frames_np.append(frame_np)
                frames_tensor.append(frame_tensor)
            except asyncio.QueueEmpty:
                break
        return frames_np, frames_tensor

    async def get_chunk(self):
        frames_np = []
        frames_tensor = []
        try:  # To prevent indefinite hanging as it can wait indefinitly for audio but it never gets the audio
            await asyncio.wait_for(
                self.ready_event.wait(), timeout=self.timeout
            )  # Waits for 40seconds if no output throws and flushes the Queue
        except asyncio.TimeoutError:
            pass
        self.ready_event.clear()
        frames_np, frames_tensor = await self._flush()
        async with self.lock:
            self.total_samples = 0
        if not frames_np or not frames_tensor:
            return None, None
        return np.concatenate(frames_np), np.concatenate(
            frames_tensor
        )  # if frames else np.zeros((0,), dtype=np.float32)


class AudioFrameRecorder(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack, buffer: AudioFrameBuffer) -> None:
        super().__init__()
        self.track = track
        self.buffer = buffer

    async def recv(self):
        frame = await self.track.recv()
        await self.buffer.add_frame(frame)
        return frame  # To prevent ruff lint error


async def audio_consumer(track, buffer):
    afr = AudioFrameRecorder(track=track, buffer=buffer)
    while afr.track.readyState != "ended":
        await afr.recv()


def prepare_audio_for_whisper(frame: AudioFrame):
    """
    Convert PyAV AudioFrame to Whisper-compatible mono float32 waveform at 16 kHz.
    Input shape: (channels, samples)
    Output shape: (n_samples,) 1D NumPy array
    """
    arr = frame.to_ndarray()
    if arr.ndim == 2:
        arr = arr.mean(axis=0)

    arr = arr.astype(np.float32)

    if np.abs(arr).max() > 1.0:
        arr = arr / np.abs(arr).max()

    arr2 = arr.copy()
    if frame.sample_rate != 16000:
        arr = librosa.resample(arr, orig_sr=frame.sample_rate, target_sr=16000)

    return arr, arr2


async def audio_transformer(
    track: MediaStreamTrack,
) -> AsyncGenerator[Tuple[torch.Tensor, np.ndarray], None]:
    afb = AudioFrameBuffer()
    asyncio.create_task(audio_consumer(track=track, buffer=afb))
    while track.readyState != "ended":
        frames_np, frames_tensor = await afb.get_chunk()
        if frames_np is None or frames_tensor is None:
            continue
        # tensor = torch.from_numpy(chunk).unsqueeze(0)
        tensor = torch.from_numpy(frames_tensor).unsqueeze(0)
        tensor = torchaudio.functional.resample(
            tensor, afb.sample_rate_in, afb.sample_rate
        )
        print("sending chunk")
        print(f"received numpy shape: {frames_np.shape}")
        yield tensor, frames_np


class VideoStreamRecorder(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, maxsize=10) -> None:
        super().__init__()
        self.track = track
        self.queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=maxsize)
        self._skip_ratio = 5
        self._running = True

    async def recv(self):
        return await self.queue.get()

    async def recv_loop(self):
        frame_count = 0
        count = 0
        while self._running:
            try:
                frame = await self.track.recv()
                count += 1
            except Exception as e:
                print(f"[Video Transformer error]: {e}")
                break
            # print(f"frame received {count}")
            if self.track.readyState == "ended":
                self.stop()
                break

            frame_count = (frame_count + 1) % self._skip_ratio
            if frame_count != 0:
                continue

            try:
                self.queue.put_nowait(frame)  # type: ignore
            except asyncio.QueueFull:
                _ = self.queue.get_nowait()
                self.queue.put_nowait(frame)  # type: ignore

            # print(f"put frame {count} in queue")

    def stop(self) -> None:
        self._running = False
        while not self.queue.empty():
            self.queue.get_nowait()


# class VideoFrameBuffer:
#     def __init__(self) -> None:
#         self.queue = asyncio.Queue(maxsize=10)
#
#
# async def video_transformer(track: MediaStreamTrack, buffer: VideoFrameBuffer):
#     vsr = VideoStreamRecorder(track)
#     frame_count = 0
#     while vsr.track.readyState != "ended":
#         frame: VideoFrame = await vsr.recv()
#         frame_count = (frame_count + 1) % 3
#         if frame_count == 0:
#             try:
#                 buffer.queue.put_nowait(frame.to_ndarray(format="bgr24"))
#             except asyncio.QueueFull as e:
#                 buffer.queue.get_nowait()
#         else:
#             continue
#         # yield frame.to_ndarray(format="bgr24")
