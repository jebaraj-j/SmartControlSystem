# Smart Control Redesign

## Recommended Project Structure

```text
smart_control/
  __init__.py
  schemas.py
  logging_config.py
  gesture/
    __init__.py
    cvzone_detector.py
  speech/
    __init__.py
    offline_recognizer.py
  llm/
    __init__.py
    phi3_ollama.py
  actions/
    __init__.py
    system_actions.py
  pipeline/
    __init__.py
    controller.py
examples/
  gesture_cvzone_template.py
  speech_offline_template.py
  phi3_ollama_template.py
docs/
  SMART_CONTROL_REDESIGN.md
  EDGE_CASES.md
```

## Data Flow

1. `gesture/cvzone_detector.py` captures webcam frames and emits labels like `FIST`, `OPEN_PALM`, `TWO_FINGERS`.
2. `speech/offline_recognizer.py` captures mic audio and emits offline transcripts (`VoskSpeechRecognizer`) or transcribes saved files (`WhisperFileRecognizer`).
3. `pipeline/controller.py` fuses latest gesture + voice into `MultimodalInput`.
4. `llm/phi3_ollama.py` sends that payload to local Ollama (`phi3:mini`) and expects JSON intent output.
5. `actions/system_actions.py` maps the intent JSON into safe control actions with confirmation requirements.

## Migration Guidance

- Keep your current `ui/main_window.py` running while you integrate the new package incrementally.
- Start by replacing gesture capture in the camera thread with `CvzoneGestureDetector`.
- Route voice transcript text into `SmartControlPipeline.update_voice(...)`.
- Display `ActionResult.message` in your existing voice log area.
- Keep `SystemActionMapper(dry_run=True)` until end-to-end intent quality is stable.
