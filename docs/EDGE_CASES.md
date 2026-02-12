# Edge Cases and Handling

## 1) Missing webcam or blocked camera access
- Symptom: `CvzoneGestureDetector.open()` fails.
- Handling: catch `WebcamError`, show non-blocking UI warning, disable gesture pipeline, keep voice-only mode active.

## 2) Missing microphone or wrong audio device
- Symptom: `VoskSpeechRecognizer` raises `MicrophoneError`.
- Handling: show available devices, auto-fallback to first valid input device, allow text-input fallback for testing.

## 3) Garbled or non-JSON LLM output
- Symptom: model returns prose or fenced text instead of JSON.
- Handling: `Phi3OllamaIntentClassifier` extracts first JSON-like block; on failure returns `IntentResult(error=...)` and no action is executed.

## 4) Ambiguous command intent
- Example: voice says "open it" without target.
- Handling: if `target` missing or confidence < threshold, ask follow-up prompt instead of acting.

## 5) Contradictory multimodal input
- Example: gesture=FIST (stop) while voice="shutdown".
- Handling: define precedence policy (gesture veto for high-risk commands) and require explicit confirmation.

## 6) Repeated commands due to STT duplicates
- Symptom: same phrase emitted repeatedly.
- Handling: debounce by command hash + timestamp window (e.g., ignore duplicates within 2 seconds).

## 7) Unsafe high-risk execution
- Example: accidental "shutdown".
- Handling: require two-factor confirmation (`voice=confirm` and/or allowed gesture like `OPEN_PALM`) before execution.

## 8) Ollama not running or model missing
- Symptom: connection refused from `http://127.0.0.1:11434`.
- Handling: detect startup failure, switch system into offline fallback mapper or no-op mode, and report clear recovery steps.
