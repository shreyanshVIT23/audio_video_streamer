from pathlib import Path
from typing import List
from faster_whisper import WhisperModel
from pyannote.core import Segment
import whisper
import time
import torch
import torchaudio

# from espnet_model_zoo.downloader import ModelDownloader
#
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import string
import soundfile as sf
import librosa
# from espnet2.bin.asr_inference import Speech2Text

"""
It is best to use faster whisper or openai whisper as they are far superior compared to others.
Technically faster whisper should as its name suggests be faster than openai whisper but according to the test in this page
the output is somewhat different in my laptop so others can check and see if its better in theirs.
I am only using cpu but it should run far faster in a gpu i believe
"""


class AudioRecognizer:
    def __init__(self) -> None:
        self.whisper = None
        self.faster_whisper = None
        self.wav2vec2 = None
        self.wav2vec2_claude = None
        self.esp2net = None

    def faster_whisper_load(self, model="small", device="cpu", compute_type="int8"):
        return WhisperModel(model, device, compute_type=compute_type)

    def fast_whisper_model(
        self, path_to_audio, segment: Segment | None | List[float]
    ) -> str:
        if self.faster_whisper is None:
            self.faster_whisper = self.faster_whisper_load()
        model = self.faster_whisper
        if isinstance(segment, Segment):
            segments, _ = model.transcribe(
                path_to_audio, beam_size=5, clip_timestamps=[segment.start, segment.end]
            )
        elif isinstance(segment, list):
            segments, _ = model.transcribe(audio=path_to_audio, clip_timestamps=segment)
        else:
            segments, _ = model.transcribe(path_to_audio, beam_size=5)
        return " ".join(segment.text for segment in segments)

    def whisper_load(self, model="small"):
        return whisper.load_model(model)

    def whisper_model(self, path_to_audio, segment=None) -> str:
        if self.whisper is None:
            self.whisper = self.whisper_load()
        print("load model")
        model = self.whisper
        result = ""
        if isinstance(segment, Segment):
            result = model.transcribe(
                audio=path_to_audio,
                temperature=0,
                clip_timestamps=[segment.start, segment.end],
                language="en",
            )
        elif isinstance(segment, list):
            result = model.transcribe(
                audio=path_to_audio,
                temperature=0,
                clip_timestamps=segment,
                language="en",
            )
        else:
            result = model.transcribe(audio=path_to_audio, temperature=0, language="en")
        print("got the result")
        print(result["text"])
        return "".join(result["text"])

    def wav2_vec2_load(self):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        return bundle.get_model(), bundle

    def wav2vec2_model(self, path_to_audio) -> str:
        if self.wav2vec2 is None:
            self.wav2vec2 = self.wav2_vec2_load()
        model, bundle = self.wav2vec2
        waveform, sr = torchaudio.load(path_to_audio)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        with torch.inference_mode():
            emissions, _ = self.wav2vec2(waveform)
        tokens = torch.argmax(emissions, dim=-1)

        vocab = bundle.get_labels()
        transcription = "".join([vocab[i] for i in tokens[0]])
        print("Transcription:", transcription.replace("|", " ").strip().lower())
        return transcription

    def wav2_vec2_model_with_greedy(self, path_to_audio) -> str:
        def greedy_ctc_decode(emissions, labels):
            tokens = torch.argmax(emissions, dim=-1)
            prev_token = None
            transcript = []

            for token in tokens:
                token = token.item()
                if token != prev_token and labels[token] != "|":
                    if labels[token] != "<blk>":
                        transcript.append(labels[token])
                prev_token = token

            return "".join(transcript).replace("|", " ").strip().lower()

        if self.wav2vec2 is None:
            self.wav2vec2 = self.wav2_vec2_load()
        model, bundle = self.wav2vec2
        waveform, sr = torchaudio.load(path_to_audio)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        with torch.inference_mode():
            emissions, _ = model(waveform)

        labels = bundle.get_labels()
        transcription = greedy_ctc_decode(emissions[0], labels)
        print("Transcription:", transcription)
        return transcription

    def wav2vec2_claude_load(self):
        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h", use_safetensors=True
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h", use_safetensors=True
        )
        return processor, model

    def wav2vec2_claude_model(self, path_to_audio, segment: Segment | None) -> str:
        """
        Simple Wav2Vec2 ASR Script - CPU Only
        Quick and easy speech-to-text transcription
        """
        if self.wav2vec2_claude is None:
            self.wav2vec2_claude = self.wav2vec2_claude_load()
        processor, model = self.wav2vec2_claude

        # Load model and processor
        model.eval()

        # Load audio file
        audio_file = path_to_audio

        info = torchaudio.info(audio_file)
        sample_rate = info.sample_rate

        if segment is None:
            # Entire audio
            waveform, sample_rate = torchaudio.load(audio_file)
        else:
            # Specific segment
            frame_offset = int(segment.start * sample_rate)
            num_frames = int(segment.duration * sample_rate)
            waveform, sample_rate = torchaudio.load(
                audio_file, frame_offset=frame_offset, num_frames=num_frames
            )

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Prepare input
        audio = waveform.squeeze().numpy()
        inputs = processor(
            audio,
            sampling_rate=16000,  # type:ignore
            return_tensors="pt",  # type:ignore
            padding=True,  # type:ignore
        )

        # Transcribe
        print("Transcribing...")
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Print result
        print(f"\nTranscription: {transcription}")
        return transcription

    def esp2net_load(
        self,
        tag="Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave",
    ):
        d = ModelDownloader()
        speech2text = Speech2Text(
            **d.download_and_unpack(tag),  # type: ignore
            device="cuda" if torch.cuda.is_available() else "cpu",
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.3,
            beam_size=10,
            batch_size=0,
            nbest=1,
        )

        return speech2text

    def esp2net_model(
        self,
        path_to_audio,
        segment: None | Segment,
        lang="en",
        fs=16000,
    ) -> str | None:
        # -------------------------
        # 1. Model setup
        # -------------------------
        if self.esp2net is None:
            self.esp2net = self.esp2net_load()
        speech2text = self.esp2net

        def text_normalizer(text):
            text = text.upper()
            return text.translate(str.maketrans("", "", string.punctuation))

        # -------------------------
        # 2. Recognize a pre-recorded audio file
        # -------------------------
        audio_file = path_to_audio  # replace with your file
        info = sf.info(audio_file)
        sample_rate = info.samplerate
        if segment is None:
            speech, rate = sf.read(audio_file)
        else:
            start_sample = int(segment.start * sample_rate)
            stop_sample = int(segment.end * sample_rate)
            speech, rate = sf.read(audio_file, start=start_sample, stop=stop_sample)
        if rate != fs:
            speech = librosa.resample(speech, rate, fs)
            rate = fs

        nbests = speech2text(speech)
        text, *_ = nbests[0]

        print(f"Input file: {audio_file}")
        print(f"ASR hypothesis: {text_normalizer(text)}, {text}")
        return text


if __name__ == "__main__":
    ar = AudioRecognizer()
    path_to_audio = str(Path("./test_data/audio.wav"))
    models = {
        # "ESP2NET": ar.esp2net_model,
        "FAST WHISPER": ar.fast_whisper_model,
        # "WAV2VEC2 Claude": ar.wav2vec2_claude_model,
        "OPENAI WHISPER": ar.whisper_model,
    }

    results = {}
    for _ in range(10):
        for name, model_func in models.items():
            start = time.time()
            output = model_func(path_to_audio, None)
            elapsed = time.time() - start
            results[name] = results.get(name, 0) + elapsed
            print(f"{name} ({elapsed:.2f}s): {output}\n")

    print("=== Summary ===")
    for name, t in results.items():
        print(f"{name}: {t:.2f} sec")
