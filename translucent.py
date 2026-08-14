"""Translucent: turn 'invisible' by throwing a peace sign at your webcam.

Controls (while the preview window is focused):
  b       - capture the current camera view as the background plate
            (step out of frame first)
  d       - toggle debug overlay (recognized gesture + status text)
  q / Esc - quit

Usage:
  python translucent.py [--camera 0] [--background background.jpg]
  python translucent.py --virtual-cam    # also expose the effect as a webcam
                                          # device for Zoom/Discord/OBS/etc.
                                          # Requires pyvirtualcam plus a system
                                          # virtual camera backend: either OBS
                                          # Studio (provides "OBS Virtual
                                          # Camera") or the standalone driver
                                          # from
                                          # https://github.com/letmaik/pyvirtualcam/releases
"""

import argparse
import contextlib
import os
import time

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    VIRTUALCAM_AVAILABLE = True
except ImportError:
    VIRTUALCAM_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BACKGROUND_PATH = os.path.join(SCRIPT_DIR, "background.jpg")
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_recognizer.task")

PEACE_HOLD_FRAMES = 6       # consecutive frames the gesture must be seen before it counts
TOGGLE_COOLDOWN = 1.5       # seconds to ignore new toggles after one fires
BLEND_SPEED = 0.15          # how fast the crossfade catches up to its target each frame
GESTURE_CONFIDENCE = 0.6


def load_background(path, frame_shape):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = frame_shape[:2]
    return cv2.resize(img, (w, h))


def save_background_from_frame(frame, path):
    cv2.imwrite(path, frame)
    return frame.copy()


def run_loop(cap, recognizer, background, background_path, vcam, window, show_preview):
    peace_streak = 0
    target_alpha = 0.0   # 0 = fully visible (camera), 1 = fully translucent (background)
    current_alpha = 0.0
    last_toggle_time = 0.0
    debug = False
    start_time = time.time()

    print("Translucent running. Press 'b' to set your background, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        top_gesture = None
        if result.gestures:
            candidate = result.gestures[0][0]
            if candidate.score >= GESTURE_CONFIDENCE:
                top_gesture = candidate.category_name

        peace_now = top_gesture == "Victory"

        if peace_now and background is not None:
            peace_streak += 1
        else:
            peace_streak = 0

        now = time.time()
        if (
            peace_streak >= PEACE_HOLD_FRAMES
            and background is not None
            and now - last_toggle_time > TOGGLE_COOLDOWN
        ):
            target_alpha = 1.0 - target_alpha
            last_toggle_time = now
            peace_streak = 0

        current_alpha += (target_alpha - current_alpha) * BLEND_SPEED

        if background is not None and current_alpha > 0.003:
            output = cv2.addWeighted(background, current_alpha, frame, 1 - current_alpha, 0)
        else:
            output = frame

        # vcam/preview get the clean feed; overlay text is drawn on a separate
        # copy so debug/status text never leaks into the outgoing video call.
        if vcam is not None:
            vcam.send(output)
            vcam.sleep_until_next_frame()

        if show_preview:
            display = output
            if debug:
                display = output.copy()
                status = "TRANSLUCENT" if target_alpha > 0.5 else "VISIBLE"
                bg_status = "background: set" if background is not None else "background: NOT SET (press b)"
                gesture_status = f"gesture: {top_gesture or '-'}"
                cv2.putText(display, f"state: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, bg_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, gesture_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif background is None:
                display = output.copy()
                cv2.putText(display, "Step out of frame and press 'b' to set background",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            background = save_background_from_frame(frame, background_path)
            print(f"Background captured and saved to {background_path}")
        elif key == ord("d"):
            debug = not debug


def main():
    parser = argparse.ArgumentParser(description="Turn translucent with a peace sign.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND_PATH, help="path to background image")
    parser.add_argument(
        "--virtual-cam",
        action="store_true",
        help="also send the output to a virtual camera device (needs pyvirtualcam + a backend driver)",
    )
    parser.add_argument("--no-preview", action="store_true", help="skip the local preview window")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Missing gesture model at {MODEL_PATH}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
            "gesture_recognizer/float16/latest/gesture_recognizer.task"
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    ok, probe = cap.read()
    if not ok:
        raise SystemExit("Could not read an initial frame from the camera")
    frame_h, frame_w = probe.shape[:2]

    recognizer = GestureRecognizer.create_from_options(
        GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )

    background = None
    if os.path.exists(args.background):
        background = load_background(args.background, probe.shape)

    vcam_ctx = contextlib.nullcontext()
    if args.virtual_cam:
        if not VIRTUALCAM_AVAILABLE:
            raise SystemExit("pyvirtualcam is not installed. Run: pip install pyvirtualcam")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        try:
            vcam_ctx = pyvirtualcam.Camera(width=frame_w, height=frame_h, fps=fps, fmt=PixelFormat.BGR)
        except RuntimeError as e:
            raise SystemExit(
                f"Could not start virtual camera: {e}\n"
                "On Windows you need a virtual camera backend installed: either OBS Studio "
                "(provides 'OBS Virtual Camera') or the standalone driver from "
                "https://github.com/letmaik/pyvirtualcam/releases"
            )

    window = "Translucent"
    show_preview = not args.no_preview
    if show_preview:
        cv2.namedWindow(window)

    try:
        with vcam_ctx as vcam:
            if vcam is not None:
                print(f"Virtual camera active: {vcam.device}")
            run_loop(cap, recognizer, background, args.background, vcam, window, show_preview)
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
