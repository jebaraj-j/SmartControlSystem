from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    vosk_model_path: Path
    hand_landmarker_path: Path
    camera_index: int
    gesture_print_interval_sec: float


def _resolve_vosk_model_path(root_dir: Path) -> Path:
    env_model = os.environ.get("VOSK_MODEL_PATH")
    if env_model:
        return Path(env_model)

    candidates = [
        root_dir / "vosk-model-small-en-us-0.15",
        root_dir / "voice" / "vosk-model-small-en-us-0.15",
        root_dir / "voice" / "vosk-model-en-us-0.22",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_config() -> AppConfig:
    root_dir = Path(__file__).resolve().parent
    return AppConfig(
        root_dir=root_dir,
        vosk_model_path=_resolve_vosk_model_path(root_dir),
        hand_landmarker_path=Path(os.environ.get("HAND_LANDMARKER_PATH", root_dir / "hand_landmarker.task")),
        camera_index=int(os.environ.get("CAMERA_INDEX", "0")),
        gesture_print_interval_sec=float(os.environ.get("GESTURE_PRINT_INTERVAL_SEC", "0.5")),
    )
