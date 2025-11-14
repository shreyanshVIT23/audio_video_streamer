import asyncio
from pathlib import Path
from typing import List
from aiortc import MediaStreamTrack
import aiofiles
from av import VideoFrame
import cv2
from numpy import ndarray
from pandas import DataFrame
from pipeline.audio_detect import AudioRecognizer
from pipeline.audio_diarization import AudioEmbedding, AudioSegmentationFile
from pipeline.face_detect import extract_faces, find_person, save_new_person_pyav
from pipeline.streaming import VideoStreamRecorder, audio_transformer


class AudioProcessor:
    def __init__(self, track: MediaStreamTrack) -> None:
        self.ar = AudioRecognizer()
        self.aseg = AudioSegmentationFile()
        self.aem = AudioEmbedding()
        self.track = track
        self.conversation = Path(r"./conversation.txt")

    async def process_audio(self):
        async with aiofiles.open(self.conversation, "a") as f:
            async for tensors, chunk in audio_transformer(self.track):
                print(f"{tensors.layout}: layout, {tensors.shape}: shape")
                audioFile = {"waveform": tensors, "sample_rate": 16000}
                segments = []
                for segment, _ in self.aseg.voice_detection(audioFile):
                    emb = await asyncio.to_thread(
                        self.aem.get_embeddings, audioFile, segment
                    )
                    print("did embedding")
                    name = await asyncio.to_thread(self.aem.compare_embeddings, emb)
                    print("compared embedding")
                    if name is None:
                        name = self.aem.save_embeddings(emb)
                    segments.append(segment.start)
                    segments.append(segment.end)
                result = await asyncio.to_thread(
                    self.ar.whisper_model, path_to_audio=chunk
                )
                print("whisper transcribed")
                print(result)
                # await f.write(f"[{name}]: {result}\n")


class VideoProcessor:
    def __init__(self, track: MediaStreamTrack) -> None:
        self.recorder = VideoStreamRecorder(track)

    # async def start(self):
    #     asyncio.create_task(self.recorder.recv_loop())
    #     asyncio.create_task(self.process_video())

    async def start(self):
        await asyncio.gather(self.recorder.recv_loop(), self.process_video())

    def _show_frame(self, frame, text):
        cv2.imshow(text, frame)
        cv2.waitKey(1)

    async def process_video(self):
        async with aiofiles.open("./all_people.txt", "a") as f:
            prev_names = set()
            # async for frame in video_transformer(self.track):
            while self.recorder._running or self.recorder.track.readyState != "ended":
                try:
                    frame = self.recorder.queue.get_nowait()
                    # print("I got frame")
                    frame = frame.to_ndarray(format="bgr24")
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)
                    continue
                # cv2.imshow("face see", frame)
                # print("Hey0")
                self._show_frame(frame=frame, text="face see")
                names = set()
                # print("Hey1")
                faces: List[ndarray] = await asyncio.to_thread(extract_faces, frame)  # type: ignore

                # print("Hey2")
                if not faces:
                    continue
                for cropped_face in faces:
                    persons: List[DataFrame] = await asyncio.to_thread(
                        # find_person, cropped_face
                        find_person,
                        frame,
                    )  # type: ignore
                    for person in persons:
                        if person.empty:
                            if cropped_face.dtype != "uint8":
                                cropped_face = (cropped_face * 255).astype("uint8")
                            name = save_new_person_pyav(
                                VideoFrame.from_ndarray(cropped_face)
                            )
                            # print("Hey3")
                        else:
                            name = Path(person["identity"].iloc[0]).parent.name
                        names.add(name)
                if not prev_names:
                    prev_names = names.copy()
                    for name in prev_names:
                        await f.write(f"New person detected: {name}\n")
                    await f.flush()
                    continue

                new_names = names - prev_names
                removed_names = prev_names - names

                if new_names:
                    for new_name in new_names:
                        await f.write(f"New person detected: {new_name}\n")
                    await f.flush()

                if removed_names:
                    for removed_name in removed_names:
                        await f.write(f"Removed person detected: {removed_name}\n")
                    await f.flush()

                prev_names = names.copy()
        await asyncio.to_thread(cv2.destroyAllWindows)
