from pathlib import Path
from pipeline.audio_diarization import (
    simulated_diarization,
)
from pipeline.audio_detect import AudioRecognizer


def transcription(audio_path):
    ar = AudioRecognizer()
    for segment, person in simulated_diarization(audio_path):
        result = ar.fast_whisper_model(path_to_audio=audio_path, segment=segment)
        yield f"[{person}]: {result}"


conversation = []
audio_path = Path("./test_data/2 people conversation.opus")
for speaking in transcription(str(audio_path)):
    conversation.append(speaking)

for speaker in conversation:
    print(speaker)
