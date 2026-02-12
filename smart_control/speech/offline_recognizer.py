"""Offline speech recognition templates (Vosk stream + optional Whisper file transcription)."""

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from smart_control.schemas import VoiceEvent

LOGGER = logging.getLogger(__name__)


class MicrophoneError(RuntimeError):
    """Raised when no usable input microphone is available."""


@dataclass
class VoskSpeechConfig:
    """Runtime settings for Vosk live transcription."""

    model_path: str
    sample_rate: int = 16000
    device: Optional[int] = None


class VoskSpeechRecognizer:
    """Streaming offline speech-to-text with Vosk and sounddevice."""

    def __init__(self, config: VoskSpeechConfig):
        if not Path(config.model_path).exists():
            raise FileNotFoundError(f"Vosk model path not found: {config.model_path}")

        self.config = config
        self.model = Model(config.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
        self.recognizer.SetWords(True)
        self.audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _resolve_input_device(self) -> int:
        if self.config.device is not None:
            info = sd.query_devices(self.config.device)
            if info.get("max_input_channels", 0) <= 0:
                raise MicrophoneError("Configured audio device has no input channels.")
            return self.config.device

        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                return idx
        raise MicrophoneError("No microphone found. Connect a mic and try again.")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            LOGGER.warning("Audio callback status: %s", status)
        self.audio_queue.put(indata.tobytes())

    def start(self, on_event: Callable[[VoiceEvent], None], on_error: Callable[[Exception], None]) -> None:
        """Start asynchronous recognition loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(on_event, on_error), daemon=True)
        self._thread.start()

    def _run_loop(self, on_event: Callable[[VoiceEvent], None], on_error: Callable[[Exception], None]) -> None:
        try:
            device = self._resolve_input_device()
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                blocksize=4000,
                device=device,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            ):
                LOGGER.info("Microphone stream started on device %s", device)
                while self._running:
                    try:
                        data = self.audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            on_event(VoiceEvent(text=text, confidence=0.70))
        except Exception as exc:
            on_error(exc)
        finally:
            self._running = False
            LOGGER.info("Microphone stream stopped")

    def stop(self) -> None:
        """Stop background recognition thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None


class WhisperFileRecognizer:
    """Optional offline file transcription template (for recorded audio clips)."""

    def __init__(self, model_size: str = "base"):
        try:
            import whisper
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("Install openai-whisper to use WhisperFileRecognizer") from exc
        self._whisper = whisper.load_model(model_size)

    def transcribe_file(self, audio_path: str) -> VoiceEvent:
        """Transcribe a WAV/MP3 file to text using Whisper."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        result = self._whisper.transcribe(audio_path)
        text = result.get("text", "").strip()
        confidence = float(result.get("avg_logprob", -1.0))
        return VoiceEvent(text=text, confidence=confidence if confidence > 0 else 0.60)
