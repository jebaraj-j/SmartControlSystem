"""Template script: cvzone gesture detection loop."""

import cv2

from smart_control.gesture import CvzoneGestureDetector, WebcamError
from smart_control.logging_config import configure_logging


def main() -> None:
    configure_logging()
    detector = CvzoneGestureDetector(camera_index=0)

    try:
        detector.open()
        while True:
            frame, event = detector.read()
            if frame is None:
                continue

            if event:
                print(f"Gesture: {event.label} (conf={event.confidence:.2f})")

            cv2.imshow("Gesture Template", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except WebcamError as exc:
        print(f"Webcam error: {exc}")
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
