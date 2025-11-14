from typing import Optional, Tuple, List
from faster_whisper import WhisperModel
from pyannote.core import Segment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
import torch

class AudioRecognizer:
    """
    Audio recognition class supporting multiple ASR (Automatic Speech Recognition) models.

    Provides interfaces for Faster-Whisper, OpenAI Whisper, Wav2Vec2, and ESPnet2 models.
    """

    whisper: Optional[object]
    faster_whisper: Optional[WhisperModel]
    wav2vec2: Optional[Tuple[torch.nn.Module, object]]
    wav2vec2_claude: Optional[Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]]
    esp2net: Optional[Speech2Text]

    def __init__(self) -> None:
        """Initialize the AudioRecognizer with all model attributes set to None."""
        ...

    def faster_whisper_load(
        self, model: str = "small", device: str = "cpu", compute_type: str = "int8"
    ) -> WhisperModel:
        """
        Load a Faster-Whisper model.

        :param model: Model size to load (e.g., "tiny", "base", "small", "medium", "large")
        :param device: Device to run the model on ("cpu" or "cuda")
        :param compute_type: Computation type for quantization (e.g., "int8", "float16", "float32")
        :return: Loaded WhisperModel instance
        """
        ...

    def fast_whisper_model(self, path_to_audio: str, segment: Optional[Segment]) -> str:
        """
        Run transcription using Faster-Whisper.

        :param path_to_audio: Path to the audio file to transcribe
        :param segment: Optional audio segment to transcribe (from pyannote)
        :return: Transcribed text
        """
        ...

    def whisper_load(self, model: str = "small") -> object:
        """
        Load OpenAI Whisper model.

        :param model: Model size to load (e.g., "tiny", "base", "small", "medium", "large")
        :return: Loaded Whisper model instance
        """
        ...

    def whisper_model(
        self, path_to_audio: str, segment: Optional[Segment | list] = None
    ) -> str:
        """
        Transcribe using OpenAI Whisper model.

        :param path_to_audio: Path to the audio file to transcribe
        :param segment: Optional audio segment or list to transcribe
        :return: Transcribed text
        """
        ...

    def wav2_vec2_load(self) -> Tuple[torch.nn.Module, object]:
        """
        Load Wav2Vec2 ASR model from torchaudio pipelines.

        :return: Tuple of (model, decoder) for Wav2Vec2
        """
        ...

    def wav2vec2_model(self, path_to_audio: str) -> str:
        """
        Run Wav2Vec2 transcription.

        :param path_to_audio: Path to the audio file to transcribe
        :return: Transcribed text
        """
        ...

    def wav2_vec2_model_with_greedy(self, path_to_audio: str) -> str:
        """
        Run Wav2Vec2 with greedy CTC decoding.

        :param path_to_audio: Path to the audio file to transcribe
        :return: Transcribed text using greedy decoding
        """
        ...

    def wav2vec2_claude_load(self) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
        """
        Load Wav2Vec2 (Claude version) model and processor.

        :return: Tuple of (processor, model) for Wav2Vec2
        """
        ...

    def wav2vec2_claude_model(
        self, path_to_audio: str, segment: Optional[Segment]
    ) -> str:
        """
        Transcribe with pretrained Wav2Vec2 model using Claude's setup.

        :param path_to_audio: Path to the audio file to transcribe
        :param segment: Optional audio segment to transcribe (from pyannote)
        :return: Transcribed text
        """
        ...

    def esp2net_load(
        self,
        tag: str = (
            "Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_"
            "hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave"
        ),
    ) -> Speech2Text:
        """
        Load ESPnet2 ASR model.

        :param tag: Model tag identifier from ESPnet model zoo
        :return: Loaded Speech2Text model instance
        """
        ...

    def esp2net_model(
        self,
        path_to_audio: str,
        segment: Optional[Segment],
        lang: str = "en",
        fs: int = 16000,
    ) -> Optional[str]:
        """
        Run speech recognition using ESPnet2 model.

        :param path_to_audio: Path to the audio file to transcribe
        :param segment: Optional audio segment to transcribe (from pyannote)
        :param lang: Language code for recognition (default: "en")
        :param fs: Sample rate in Hz (default: 16000)
        :return: Transcribed text or None if recognition fails
        """
        ...
