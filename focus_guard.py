"""Focus Guard: nags you with a mock course-withdrawal form when you stop
looking at the screen (doomscrolling on your phone, staring off, etc).

Detection is a head-pose proxy, not true gaze/pupil tracking: it estimates
which way your face is oriented (yaw/pitch) using MediaPipe FaceLandmarker +
a classic 6-point solvePnP fit, and treats "face turned/tilted away beyond a
threshold" or "no face visible" as not-looking-at-the-screen.

The popup is purely cosmetic - no data is entered or sent anywhere, no real
withdrawal happens. It auto-closes once your attention returns.

Controls (while the preview window is focused):
  d       - toggle debug overlay (yaw/pitch/state)
  q / Esc - quit

Usage:
  python focus_guard.py [--camera 0] [--lost-seconds 4] [--recovery-seconds 2]
"""

import argparse
import os
import threading
import time
import tkinter as tk

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PIL import Image as PILImage, ImageTk

try:
    import pymupdf as fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
DEFAULT_FORM_PDF_PATH = os.path.join(SCRIPT_DIR, "withdrawal_form.pdf")

# Standard 6-point landmark indices used for solvePnP head pose (MediaPipe
# FaceMesh canonical topology).
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Generic anthropometric 3D model points (millimeters, arbitrary origin at
# the nose tip) - a coarse approximation, sufficient for threshold-level
# yaw/pitch, not precision tracking.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # nose tip
    (0.0, -63.6, -12.5),     # chin
    (-43.3, 32.7, -26.0),    # left eye, left corner
    (43.3, 32.7, -26.0),     # right eye, right corner
    (-28.9, -28.9, -24.1),   # left mouth corner
    (28.9, -28.9, -24.1),    # right mouth corner
], dtype="double")

LANDMARK_INDICES = [NOSE_TIP, CHIN, LEFT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER,
                     LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.should_show_popup = False
        self.debug_text = ""


def estimate_head_pose(landmarks, frame_w, frame_h):
    image_points = np.array(
        [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in LANDMARK_INDICES],
        dtype="double",
    )

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    pitch, yaw, _roll = euler_angles.flatten()

    # decomposeProjectionMatrix can report pitch as a value near +/-180
    # instead of near 0 depending on the solved rotation branch; fold it
    # back into a signed -90..90 range around "facing the camera".
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return pitch, yaw


def camera_worker(state, args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}")
        with state.lock:
            state.running = False
        return

    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    window = "Focus Guard"
    if not args.no_preview:
        cv2.namedWindow(window)

    debug = False
    start_time = time.time()
    attentive_state = True
    state_start_time = time.time()

    print("Focus Guard running. Look away or leave frame to test the trigger. 'q' to quit.")

    while True:
        with state.lock:
            if not state.running:
                break

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        attentive_now = False
        pose_text = "no face"
        if result.face_landmarks:
            pose = estimate_head_pose(result.face_landmarks[0], frame_w, frame_h)
            if pose is not None:
                pitch, yaw = pose
                attentive_now = abs(yaw) < args.yaw_threshold and abs(pitch) < args.pitch_threshold
                pose_text = f"yaw={yaw:.0f} pitch={pitch:.0f}"

        if attentive_now != attentive_state:
            attentive_state = attentive_now
            state_start_time = time.time()

        held_for = time.time() - state_start_time
        with state.lock:
            if not attentive_state and held_for >= args.lost_seconds:
                state.should_show_popup = True
            elif attentive_state and held_for >= args.recovery_seconds:
                state.should_show_popup = False
            state.debug_text = f"{'attentive' if attentive_state else 'distracted'} {held_for:.1f}s | {pose_text}"

        if not args.no_preview:
            display = frame
            if debug:
                display = frame.copy()
                with state.lock:
                    text = state.debug_text
                    popup_text = f"popup: {'ON' if state.should_show_popup else 'off'}"
                cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, popup_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                with state.lock:
                    state.running = False
                break
            elif key == ord("d"):
                debug = not debug

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    with state.lock:
        state.running = False


def build_mock_popup(root):
    popup = tk.Toplevel(root)
    popup.title("Course Withdrawal Form")
    popup.attributes("-fullscreen", True)
    popup.attributes("-topmost", True)
    popup.configure(bg="#f4f1e8")

    frame = tk.Frame(popup, bg="#f4f1e8")
    frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(frame, text="COURSE WITHDRAWAL FORM", font=("Georgia", 32, "bold"),
              bg="#f4f1e8", fg="#7a1f1f").pack(pady=(0, 10))
    tk.Label(frame, text="Your attention appears to have left the course material.",
              font=("Georgia", 16), bg="#f4f1e8", fg="#333").pack(pady=(0, 20))

    fields = [
        ("Reason for withdrawal:", "Doomscrolling"),
        ("Last observed activity:", "Not looking at screen"),
        ("Status:", "Pending your return"),
    ]
    for label, value in fields:
        row = tk.Frame(frame, bg="#f4f1e8")
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, font=("Georgia", 14, "bold"), width=22, anchor="e",
                  bg="#f4f1e8", fg="#333").pack(side="left", padx=6)
        tk.Label(row, text=value, font=("Georgia", 14), anchor="w",
                  bg="white", fg="#333", width=30, relief="sunken", padx=6).pack(side="left")

    tk.Label(frame, text="Look back at the screen to automatically cancel this withdrawal.",
              font=("Georgia", 13, "italic"), bg="#f4f1e8", fg="#666").pack(pady=(24, 0))

    return popup


def render_pdf_pages(path, max_w, max_h):
    """Render each page of a PDF to a PhotoImage sized to fit (max_w, max_h)."""
    doc = fitz.open(path)
    images = []
    for page in doc:
        zoom = min(max_w / page.rect.width, max_h / page.rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        mode = "RGBA" if pix.alpha else "RGB"
        pil_img = PILImage.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(ImageTk.PhotoImage(pil_img))
    doc.close()
    return images


def build_pdf_popup(root, page_images):
    popup = tk.Toplevel(root)
    popup.title("Course Withdrawal Form")
    popup.attributes("-fullscreen", True)
    popup.attributes("-topmost", True)
    popup.configure(bg="#1a1a1a")

    page_idx = {"i": 0}

    image_label = tk.Label(popup, bg="#1a1a1a")
    image_label.place(relx=0.5, rely=0.45, anchor="center")

    caption_text = "Look back at the screen to automatically cancel this withdrawal."
    if len(page_images) > 1:
        caption_text += "  (← / → to view other pages)"
    tk.Label(popup, text=caption_text, font=("Georgia", 13, "italic"),
              bg="#1a1a1a", fg="#ccc").place(relx=0.5, rely=0.94, anchor="center")

    page_indicator = tk.Label(popup, font=("Georgia", 11), bg="#1a1a1a", fg="#888")
    page_indicator.place(relx=0.99, rely=0.01, anchor="ne")

    def show_page(i):
        page_idx["i"] = i % len(page_images)
        image_label.configure(image=page_images[page_idx["i"]])
        if len(page_images) > 1:
            page_indicator.configure(text=f"Page {page_idx['i'] + 1}/{len(page_images)}")

    if len(page_images) > 1:
        popup.bind("<Left>", lambda e: show_page(page_idx["i"] - 1))
        popup.bind("<Right>", lambda e: show_page(page_idx["i"] + 1))

    show_page(0)
    return popup


def main():
    parser = argparse.ArgumentParser(description="Nag you back to focus with a mock withdrawal form.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--lost-seconds", type=float, default=0.2,
                         help="seconds of sustained inattention before the popup appears")
    parser.add_argument("--recovery-seconds", type=float, default=2.0,
                         help="seconds of sustained attention before the popup auto-closes")
    parser.add_argument("--yaw-threshold", type=float, default=25.0, help="max degrees left/right turn")
    parser.add_argument("--pitch-threshold", type=float, default=20.0, help="max degrees up/down tilt")
    parser.add_argument("--no-preview", action="store_true", help="skip the local preview window")
    parser.add_argument("--form-pdf", default=DEFAULT_FORM_PDF_PATH,
                         help="PDF to display in the popup (falls back to a mock form if missing)")
    parser.add_argument("--mock-form", action="store_true", help="always use the plain mock form, ignore any PDF")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Missing face landmark model at {MODEL_PATH}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )

    state = SharedState()
    worker = threading.Thread(target=camera_worker, args=(state, args), daemon=True)
    worker.start()

    root = tk.Tk()
    root.withdraw()
    popup_holder = {"window": None}

    page_images = None
    if not args.mock_form and args.form_pdf and os.path.exists(args.form_pdf):
        if PDF_AVAILABLE:
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            try:
                page_images = render_pdf_pages(args.form_pdf, screen_w * 0.8, screen_h * 0.75)
            except Exception as e:
                print(f"Could not render {args.form_pdf}: {e}. Falling back to mock form.")
        else:
            print("pymupdf is not installed (pip install pymupdf). Falling back to mock form.")

    def make_popup():
        if page_images:
            return build_pdf_popup(root, page_images)
        return build_mock_popup(root)

    def poll():
        with state.lock:
            running = state.running
            should_show = state.should_show_popup
        if not running:
            if popup_holder["window"] is not None:
                popup_holder["window"].destroy()
            root.quit()
            return
        if should_show and popup_holder["window"] is None:
            popup_holder["window"] = make_popup()
        elif not should_show and popup_holder["window"] is not None:
            popup_holder["window"].destroy()
            popup_holder["window"] = None
        root.after(200, poll)

    root.after(200, poll)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        with state.lock:
            state.running = False

    with state.lock:
        state.running = False
    worker.join(timeout=2)


if __name__ == "__main__":
    main()
