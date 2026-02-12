import json
import queue
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import sounddevice as sd
from vosk import Model, KaldiRecognizer


@dataclass
class VoskConfig:
    model_path: str
    sample_rate: int = 16000
    device: Optional[Union[int, str]] = None   # allow index or name
    grammar_phrases: Optional[List[str]] = None


class VoskStreamRecognizer:
    """
    Stable streaming recognizer using Vosk + sounddevice (Windows-safe).
    """

    def __init__(self, config: VoskConfig):
        self.config = config
        self._model = Model(self.config.model_path)
        if self.config.grammar_phrases:
            grammar = json.dumps(self.config.grammar_phrases)
            self._recognizer = KaldiRecognizer(
                self._model, self.config.sample_rate, grammar
            )
        else:
            self._recognizer = KaldiRecognizer(self._model, self.config.sample_rate)
        self._recognizer.SetWords(True)

        self._queue: queue.Queue[bytes] = queue.Queue()
        self._run_flag = False
        self._thread: Optional[threading.Thread] = None
        self._device_index: Optional[int] = None

    def _resolve_device(self) -> Optional[int]:
        if self.config.device is None:
            default_in = sd.default.device[0] if sd.default.device else None
            if isinstance(default_in, int) and default_in >= 0:
                info = sd.query_devices(default_in)
                if info and info.get("max_input_channels", 0) > 0:
                    return default_in
            for idx, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) > 0:
                    return idx
            raise RuntimeError("No input audio devices found")

        if isinstance(self.config.device, int):
            info = sd.query_devices(self.config.device)
            if info.get("max_input_channels", 0) < 1:
                raise RuntimeError("Selected device has no input channels")
            return self.config.device

        target = str(self.config.device).strip().lower()
        if not target:
            return None
        for idx, info in enumerate(sd.query_devices()):
            name = str(info.get("name", "")).lower()
            if target in name and info.get("max_input_channels", 0) > 0:
                return idx
        raise RuntimeError(f"Input device not found: {self.config.device}")


    # ─────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        self._queue.put(indata.tobytes())

    # ─────────────────────────────────────
    def start(
        self,
        on_text: Callable[[str], None],
        on_error: Optional[Callable[[str], None]] = None,
    ):
        if self._run_flag:
            return

        self._run_flag = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(on_text, on_error),
            daemon=True
        )
        self._thread.start()

    # ─────────────────────────────────────
    def _run_loop(self, on_text, on_error):
        try:
            device = self._resolve_device()
            self._device_index = device
            sd.check_input_settings(
                device=device,
                samplerate=self.config.sample_rate,
                channels=1,
            )

            samplerate = self.config.sample_rate

            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=samplerate,
                dtype="int16",
                blocksize=4000,      # ✅ SAFE
                callback=self._audio_callback,
            ):
                while self._run_flag:
                    try:
                        data = self._queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    if self._recognizer.AcceptWaveform(data):
                        result = json.loads(self._recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            on_text(text)

        except Exception as exc:
            if on_error:
                on_error(f"Mic error: {exc}")

        finally:
            self._run_flag = False

    # ─────────────────────────────────────
    def stop(self):
        self._run_flag = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
