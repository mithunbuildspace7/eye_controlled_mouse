import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
import os
import math
import threading
import urllib.request
import csv as _csv
from collections import deque

import gaze_lib as gl

# ============================================
# CONFIG (camera / calibration-flow specific)
# ============================================

CAMERA_RESOLUTIONS = [(1920, 1080), (1280, 720), (960, 540), (640, 480)]
DETECTION_SCALE = 1.0  # item 7: test 0.85 vs 1.0 — full-res kept here for max eye pixels

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

CALIB_GRID_COLS = 7
CALIB_GRID_ROWS = 7
CALIB_MARGIN = 0.04
VALIDATION_ROWCOLS = {1, 3, 5}
CORNER_EDGE_MARGIN = 0.015   # item 11: dedicated near-boundary points, tighter than the main grid

CALIB_SETTLE_TIME = 0.4
CALIB_SAMPLE_TIME = 0.9
MIN_VALID_SAMPLES = 5
CALIB_MIN_CONFIDENCE = 0.5

BLINK_EAR_THRESHOLD = gl.BLINK_EAR_THRESHOLD
BLINK_HOLD_SECONDS = 0.2
CLICK_COOLDOWN = 0.8

MOUTH_OPEN_THRESHOLD = 0.55
MODE_TOGGLE_HOLD_SECONDS = 0.2
MODE_TOGGLE_COOLDOWN = 1.0

SCROLL_DEAD_ZONE = 0.02
SCROLL_SENSITIVITY = 45
SCROLL_MAX_OFFSET = 0.15

MAX_CURSOR_SPEED_PX_PER_SEC = 2200

DIAGNOSTIC_POINTS = [
    (0.50, 0.50, "center"),
    (0.50, 0.05, "top"), (0.50, 0.95, "bottom"),
    (0.05, 0.50, "left"), (0.95, 0.50, "right"),
    (0.05, 0.05, "top-left"), (0.95, 0.05, "top-right"),
    (0.05, 0.95, "bottom-left"), (0.95, 0.95, "bottom-right"),
    (0.25, 0.30, "mid"), (0.72, 0.62, "mid"),
]
DIAGNOSTIC_SETTLE = 0.5
DIAGNOSTIC_DWELL = 1.2

DEBUG_WINDOW_W, DEBUG_WINDOW_H = 480, 360
COUNTDOWN_SECONDS = 3

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ============================================
# THREADED CAMERA
# ============================================

class ThreadedCamera:
    def __init__(self, resolutions):
        self.cap = None
        self._open_camera(resolutions)
        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.3)

    def _open_camera(self, resolutions):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0). Check camera permissions/connection.")
        for w, h in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera opened at {aw}x{ah}")
                self.cap = cap
                return
        raise RuntimeError("Could not open webcam at any supported resolution.")

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.ok = frame, True
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face landmarker model (one-time, ~few MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    except Exception as e:
        print("Could not auto-download model:", e)
        print(f"Please manually download it from:\n{MODEL_URL}\nand place it as '{MODEL_PATH}' in this folder.")
        raise SystemExit(1)


ensure_model()

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = vision.FaceLandmarker.create_from_options(options)
camera = ThreadedCamera(CAMERA_RESOLUTIONS)
start_time = time.time()


def now_ts():
    return time.time() - start_time


def get_timestamp_ms():
    return int(now_ts() * 1000)


def get_landmarks(frame):
    small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE) if DETECTION_SCALE != 1.0 else frame
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)
    small = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, get_timestamp_ms())
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def _mouth_ratio(landmarks, w, h):
    top, bottom = landmarks[gl.MOUTH_VERT[0]], landmarks[gl.MOUTH_VERT[1]]
    left, right = landmarks[gl.MOUTH_CORNERS[0]], landmarks[gl.MOUTH_CORNERS[1]]
    vert = math.hypot((top.x - bottom.x) * w, (top.y - bottom.y) * h)
    horiz = math.hypot((left.x - right.x) * w, (left.y - right.y) * h)
    return vert / horiz if horiz > 1e-6 else 0.0


def draw_iris_markers(frame, landmarks, w, h):
    """Draw visual-only eye reference points and yellow iris-center markers."""
    for idx in (*gl.LEFT_EYE_CORNERS, *gl.RIGHT_EYE_CORNERS,
                *gl.LEFT_UPPER_LID, *gl.LEFT_LOWER_LID,
                *gl.RIGHT_UPPER_LID, *gl.RIGHT_LOWER_LID):
        pt = landmarks[idx]
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (100, 100, 100), -1)

    # Yellow = BGR (0, 255, 255). These are display markers only; they do
    # not alter the gaze features, model prediction, or cursor movement.
    for idx in (gl.LEFT_IRIS_CENTER, gl.RIGHT_IRIS_CENTER):
        pt = landmarks[idx]
        center = (int(pt.x * w), int(pt.y * h))
        cv2.circle(frame, center, 3, (0, 255, 255), -1)
        cv2.circle(frame, center, 7, (0, 255, 255), 1)
    return frame


# ============================================
# CALIBRATION GRID
# ============================================

def _generate_grid_points():
    xs = np.linspace(CALIB_MARGIN, 1 - CALIB_MARGIN, CALIB_GRID_COLS)
    ys = np.linspace(CALIB_MARGIN, 1 - CALIB_MARGIN, CALIB_GRID_ROWS)

    points = []
    for row_i, gy in enumerate(ys):
        col_order = range(CALIB_GRID_COLS) if row_i % 2 == 0 else reversed(range(CALIB_GRID_COLS))
        for col_i in col_order:
            is_val = (row_i in VALIDATION_ROWCOLS) and (col_i in VALIDATION_ROWCOLS)
            points.append((row_i, col_i, float(xs[col_i]), float(gy), is_val))

    # Item 11: dedicated near-boundary anchors, closer to the true edge than
    # the main grid's margin. Train-only (not validation).
    m = CORNER_EDGE_MARGIN
    extra = [
        (m, m), (1 - m, m), (m, 1 - m), (1 - m, 1 - m),   # corners
        (0.5, m), (0.5, 1 - m), (m, 0.5), (1 - m, 0.5),   # edge midpoints
    ]
    for gx, gy in extra:
        points.append((-1, -1, gx, gy, False))

    return points


def _sample_point(target_x, target_y, progress_text, extra_prompt="", csv_writer=None, is_val=False):
    vt = gl.VelocityTracker()
    settle_start = time.time()
    samples = []

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        elapsed = time.time() - settle_start

        cv2.circle(canvas, (target_x, target_y), 20, (0, 0, 255), -1)
        cv2.circle(canvas, (target_x, target_y), 20, (255, 255, 255), 2)

        if elapsed > CALIB_SETTLE_TIME:
            landmarks = get_landmarks(frame)
            if landmarks is not None:
                feat = gl.extract_features(landmarks, frame, fw, fh)
                v = vt.update(feat.fused_gx, feat.fused_gy, now_ts())
                if feat.confidence >= CALIB_MIN_CONFIDENCE and v < gl.SACCADE_VELOCITY_THRESHOLD:
                    samples.append(feat.vector)
                    if csv_writer is not None:
                        gl.log_calibration_sample(csv_writer, feat.vector, target_x, target_y,
                                                   is_val, feat.confidence, now_ts())
            cv2.circle(canvas, (target_x, target_y), 9, (0, 255, 0), -1)
            cv2.putText(canvas, f"samples: {len(samples)}", (target_x - 45, target_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{progress_text} - look at the dot, keep head still - ESC to cancel",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if extra_prompt:
            cv2.putText(canvas, extra_prompt, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

        cv2.imshow("Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if elapsed > CALIB_SETTLE_TIME + CALIB_SAMPLE_TIME:
            break

    return samples


def _collect_calibration_data():
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    grid = _generate_grid_points()
    total_events = len(grid) + sum(1 for p in grid if p[4])
    event_i = 0

    csv_file, csv_writer, csv_name = gl.open_calibration_log()
    print(f"Logging raw calibration samples to {csv_name}")

    train_X, train_y, val_X, val_y = [], [], [], []

    for row_i, col_i, gx_norm, gy_norm, is_val in grid:
        target_x, target_y = int(gx_norm * SCREEN_W), int(gy_norm * SCREEN_H)
        event_i += 1
        samples = _sample_point(target_x, target_y, f"Point {event_i}/{total_events}",
                                 csv_writer=csv_writer, is_val=is_val)
        if samples is None:
            cv2.destroyWindow("Calibration")
            csv_file.close()
            return None

        dest_X, dest_y = (val_X, val_y) if is_val else (train_X, train_y)
        if len(samples) < MIN_VALID_SAMPLES:
            print(f"Warning: point ({row_i},{col_i}) only got {len(samples)} valid samples — skipping pass.")
        else:
            for s in samples:
                dest_X.append(s)
                dest_y.append((target_x, target_y))

        if is_val:
            event_i += 1
            samples2 = _sample_point(
                target_x, target_y, f"Point {event_i}/{total_events}",
                extra_prompt="Shift your head slightly (lean/tilt) - keep looking at the SAME dot",
                csv_writer=csv_writer, is_val=True,
            )
            if samples2 is None:
                cv2.destroyWindow("Calibration")
                csv_file.close()
                return None
            if len(samples2) < MIN_VALID_SAMPLES:
                print(f"Warning: augmented pass at ({row_i},{col_i}) only got {len(samples2)} samples.")
            else:
                for s in samples2:
                    val_X.append(s)
                    val_y.append((target_x, target_y))

    cv2.destroyWindow("Calibration")
    csv_file.close()

    if len(train_X) < 100 or len(val_X) < 20:
        print(f"Calibration failed: too little data (train={len(train_X)}, val={len(val_X)}).")
        return None

    return (np.array(train_X), np.array(train_y, dtype=np.float64),
            np.array(val_X), np.array(val_y, dtype=np.float64))


def verify_calibration(gm):
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    verify_kf = gl.GazeKalman()

    while True:
        success, frame = camera.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        for gx_norm in np.linspace(0.1, 0.9, 5):
            x = int(gx_norm * SCREEN_W)
            cv2.line(canvas, (x, 0), (x, SCREEN_H), (40, 40, 40), 1)
        for gy_norm in np.linspace(0.1, 0.9, 5):
            y = int(gy_norm * SCREEN_H)
            cv2.line(canvas, (0, y), (SCREEN_W, y), (40, 40, 40), 1)

        landmarks = get_landmarks(frame)
        if landmarks is not None:
            feat = gl.extract_features(landmarks, frame, fw, fh)
            nx, ny = gm.predict_norm(feat.vector)
            sx, sy = verify_kf.step(nx * SCREEN_W, ny * SCREEN_H, feat.confidence, now_ts())
            sx = int(np.clip(sx, 0, SCREEN_W - 1))
            sy = int(np.clip(sy, 0, SCREEN_H - 1))
            cv2.drawMarker(canvas, (sx, sy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(canvas, (sx, sy), 14, (0, 255, 0), 2)
            cv2.putText(canvas, f"confidence: {feat.confidence:.2f}", (40, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Model: {gm.model_name} - look around to check accuracy",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(canvas, "SPACE = accept   R = redo calibration   ESC = cancel",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow("Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.destroyWindow("Calibration")
            return True
        elif key == ord("r"):
            cv2.destroyWindow("Calibration")
            return False
        elif key == 27:
            cv2.destroyWindow("Calibration")
            return None


def run_calibration():
    print("Starting calibration: 49 grid points + 9 head-shifted validation reps + 8 corner/edge anchors.")
    while True:
        data = _collect_calibration_data()
        if data is None:
            return None
        X_train, y_train, X_val, y_val = data
        gm = gl.train_and_select_model(X_train, y_train, X_val, y_val, SCREEN_W, SCREEN_H)
        verdict = verify_calibration(gm)
        if verdict is True:
            print("Calibration accepted.")
            return gm
        elif verdict is None:
            return None
        else:
            print("Redoing calibration...")


# ============================================
# DIAGNOSTIC TEST SCREEN (items 9, 10, 15)
# ============================================

def run_diagnostic_test(gm):
    """Untouched test points — never seen during calibration — with a live
    target-vs-raw-vs-filtered overlay and a final region-broken-down error
    report. This is the honest, independent 'final test set' (item 15)."""
    cv2.namedWindow("Diagnostic", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Diagnostic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    test_kf = gl.GazeKalman()
    csv_file = open(f"diagnostic_log_{int(time.time())}.csv", "w", newline="")
    writer = _csv.writer(csv_file)
    writer.writerow(["target_x", "target_y", "raw_x", "raw_y", "filtered_x", "filtered_y",
                      "error_px", "error_x", "error_y", "confidence", "yaw", "pitch", "roll", "region"])

    per_point_errors = []

    for gx_norm, gy_norm, label in DIAGNOSTIC_POINTS:
        target_x, target_y = int(gx_norm * SCREEN_W), int(gy_norm * SCREEN_H)
        region = gl.region_label(gx_norm, gy_norm)
        start = time.time()
        point_errors = []

        while time.time() - start < DIAGNOSTIC_SETTLE + DIAGNOSTIC_DWELL:
            success, frame = camera.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (target_x, target_y), 18, (0, 0, 255), -1)

            landmarks = get_landmarks(frame)
            settled = (time.time() - start) > DIAGNOSTIC_SETTLE

            if landmarks is not None:
                feat = gl.extract_features(landmarks, frame, fw, fh)
                nx, ny = gm.predict_norm(feat.vector)
                raw_x, raw_y = nx * SCREEN_W, ny * SCREEN_H
                fx, fy = test_kf.step(raw_x, raw_y, feat.confidence, now_ts())

                cv2.drawMarker(canvas, (int(np.clip(raw_x, 0, SCREEN_W - 1)), int(np.clip(raw_y, 0, SCREEN_H - 1))),
                                (0, 165, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                cv2.circle(canvas, (int(np.clip(fx, 0, SCREEN_W - 1)), int(np.clip(fy, 0, SCREEN_H - 1))),
                           14, (0, 255, 0), 2)

                err_px = math.hypot(fx - target_x, fy - target_y)
                err_x, err_y = fx - target_x, fy - target_y

                info = [
                    f"target ({target_x},{target_y})  region: {region}",
                    f"raw ({raw_x:.0f},{raw_y:.0f})  filtered ({fx:.0f},{fy:.0f})",
                    f"error: {err_px:.0f}px  (x:{err_x:+.0f} y:{err_y:+.0f})",
                    f"confidence: {feat.confidence:.2f}",
                    f"yaw:{feat.yaw:.1f} pitch:{feat.pitch:.1f} roll:{feat.roll:.1f}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(canvas, line, (40, 50 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)

                if settled:
                    point_errors.append(err_px)
                    writer.writerow([target_x, target_y, raw_x, raw_y, fx, fy, err_px, err_x, err_y,
                                      feat.confidence, feat.yaw, feat.pitch, feat.roll, region])

            cv2.imshow("Diagnostic", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                csv_file.close()
                cv2.destroyWindow("Diagnostic")
                print("Diagnostic test cancelled.")
                return

        if point_errors:
            per_point_errors.append((region, float(np.mean(point_errors))))

    csv_file.close()
    cv2.destroyWindow("Diagnostic")

    print("\n--- Diagnostic test results (untouched points) ---")
    by_region = {}
    for r, e in per_point_errors:
        by_region.setdefault(r, []).append(e)
    all_errs = [e for _, e in per_point_errors]
    print(f"  Overall MAE: {np.mean(all_errs):.1f}px  (~{gl.px_to_degrees(np.mean(all_errs), SCREEN_W):.2f} deg)")
    for r, vals in sorted(by_region.items()):
        print(f"  {r:12s}: {np.mean(vals):6.1f}px")
    print("Saved raw results to diagnostic_log_*.csv\n")


# ============================================
# HUD
# ============================================

def draw_hud(frame, mode, fps, status_text, status_color, click_hold_ratio, model_name, confidence):
    h, w = frame.shape[:2]
    mode_color = (0, 200, 255) if mode == "SCROLL" else (0, 255, 120)
    cv2.putText(frame, f"MODE: {mode}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA)

    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    cv2.putText(frame, status_text, (w - status_size[0] - 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, status_color, 2, cv2.LINE_AA)

    bar_w, bar_h = 180, 10
    bar_x, bar_y = (w - bar_w) // 2, 42
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    fill = int(bar_w * np.clip(click_hold_ratio, 0, 1))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 255, 0), -1)

    conf_color = (0, 255, 0) if confidence > 0.6 else ((0, 255, 255) if confidence > 0.3 else (0, 0, 255))
    cv2.putText(frame, f"{model_name} | conf {confidence:.2f} | {fps:4.1f}fps", (12, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "ESC quit | R recalib | P pause | C correct | D diagnostic", (12, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "mouth-open: toggle scroll", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_countdown(seconds):
    for i in range(seconds, 0, -1):
        for _ in range(10):
            success, frame = camera.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (DEBUG_WINDOW_W, DEBUG_WINDOW_H))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (DEBUG_WINDOW_W, DEBUG_WINDOW_H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, str(i), (DEBUG_WINDOW_W // 2 - 15, DEBUG_WINDOW_H // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "Get ready - eye-gaze tracking starts soon", (40, DEBUG_WINDOW_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Eye Cursor - Project", frame)
            cv2.waitKey(1)
            time.sleep(0.03)


# ============================================
# MAIN
# ============================================

gaze_model = gl.load_calibration()
if gaze_model is None:
    gaze_model = run_calibration()
    if gaze_model is None:
        print("No calibration available. Exiting.")
        camera.stop()
        raise SystemExit(0)
    gl.save_calibration(gaze_model)

kalman = gl.GazeKalman()

blink_start_time = None
last_click_time = 0
click_message_until = 0
eyes_closed = False

mouth_open_start = None
last_mode_toggle = 0
mode = "MOVE"
scroll_baseline_y = None

prev_cursor_pos = None
prev_cursor_time = None
raw_prev_pos = None
raw_prev_time = None

correction_flash_until = 0
paused = False
frame_times = deque(maxlen=30)

cv2.namedWindow("Eye Cursor - Project", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Cursor - Project", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
cv2.moveWindow("Eye Cursor - Project", max(SCREEN_W - DEBUG_WINDOW_W - 30, 0), 30)

print("Eye-gaze cursor running. ESC quit | R recalibrate | P pause | C correct | D diagnostic test.")
run_countdown(COUNTDOWN_SECONDS)

while True:
    loop_start = time.time()
    success, frame = camera.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    t = now_ts()

    status_text = "No face detected"
    status_color = (0, 0, 255)
    click_hold_ratio = 0.0
    live_confidence = 0.0

    if not paused:
        landmarks = get_landmarks(frame)

        if landmarks is not None:
            feat = gl.extract_features(landmarks, frame, w, h)
            live_confidence = feat.confidence
            mouth_ratio = _mouth_ratio(landmarks, w, h)
            frame = draw_iris_markers(frame, landmarks, w, h)

            is_closed = feat.avg_ear < BLINK_EAR_THRESHOLD
            is_mouth_open = mouth_ratio > MOUTH_OPEN_THRESHOLD

            if is_mouth_open:
                if mouth_open_start is None:
                    mouth_open_start = t
                held_mouth = t - mouth_open_start
                if held_mouth >= MODE_TOGGLE_HOLD_SECONDS and t - last_mode_toggle > MODE_TOGGLE_COOLDOWN:
                    mode = "SCROLL" if mode == "MOVE" else "MOVE"
                    last_mode_toggle = t
                    mouth_open_start = t
                    if mode == "SCROLL":
                        scroll_baseline_y = feat.fused_gy
                    else:
                        kalman = gl.GazeKalman()
                        prev_cursor_pos = None
                        raw_prev_pos = None
            else:
                mouth_open_start = None

            if mode == "SCROLL":
                offset = feat.fused_gy - scroll_baseline_y
                if abs(offset) > SCROLL_DEAD_ZONE:
                    clamped = float(np.clip(offset, -SCROLL_MAX_OFFSET, SCROLL_MAX_OFFSET))
                    scroll_amount = -int(clamped * SCROLL_SENSITIVITY / SCROLL_MAX_OFFSET * 10)
                    if scroll_amount != 0:
                        pyautogui.scroll(scroll_amount)
                        status_text = "Scrolling up" if scroll_amount > 0 else "Scrolling down"
                        status_color = (0, 200, 255)
                    else:
                        status_text, status_color = "Scroll ready", (0, 200, 255)
                else:
                    status_text, status_color = "Scroll ready", (0, 200, 255)
                blink_start_time = None
                eyes_closed = False

            else:
                if is_closed:
                    if blink_start_time is None:
                        blink_start_time = t
                    held = t - blink_start_time
                    click_hold_ratio = held / BLINK_HOLD_SECONDS
                    if held >= BLINK_HOLD_SECONDS and not eyes_closed:
                        if t - last_click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = t
                            click_message_until = t + 0.50
                            eyes_closed = True
                else:
                    blink_start_time = None
                    eyes_closed = False

                if not is_closed:
                    nx, ny = gaze_model.predict_norm(feat.vector)
                    mapped_x, mapped_y = nx * SCREEN_W, ny * SCREEN_H

                    # --- runtime outlier rejection (items 20, 25) ---
                    effective_conf = feat.confidence
                    if raw_prev_pos is not None:
                        dt_raw = max(t - raw_prev_time, 1e-3)
                        v_raw = math.hypot(mapped_x - raw_prev_pos[0], mapped_y - raw_prev_pos[1]) / dt_raw
                        if v_raw > gl.RUNTIME_MAX_PLAUSIBLE_SPEED_PX and feat.confidence < 0.85:
                            effective_conf = 0.0  # implausible jump -> Kalman predicts-only this frame
                    if effective_conf > 0.0:
                        raw_prev_pos, raw_prev_time = (mapped_x, mapped_y), t

                    fx, fy = kalman.step(mapped_x, mapped_y, effective_conf, t)
                    fx = float(np.clip(fx, 0, SCREEN_W - 1))
                    fy = float(np.clip(fy, 0, SCREEN_H - 1))

                    if prev_cursor_pos is not None:
                        dt = max(t - prev_cursor_time, 1e-3)
                        max_step = MAX_CURSOR_SPEED_PX_PER_SEC * dt
                        dx, dy = fx - prev_cursor_pos[0], fy - prev_cursor_pos[1]
                        dist = math.hypot(dx, dy)
                        if dist > max_step:
                            scale = max_step / dist
                            fx = prev_cursor_pos[0] + dx * scale
                            fy = prev_cursor_pos[1] + dy * scale

                    prev_cursor_pos, prev_cursor_time = (fx, fy), t
                    pyautogui.moveTo(int(fx), int(fy))

                    if status_text != "CLICK!":
                        status_text = "Tracking" if feat.confidence >= gl.CONFIDENCE_FREEZE_THRESHOLD else "Low confidence"
                        status_color = (0, 255, 255) if feat.confidence >= gl.CONFIDENCE_FREEZE_THRESHOLD else (0, 140, 255)
                        if t < click_message_until:
                            status_text, status_color = "CLICK!", (0, 255, 0)
                        if t < correction_flash_until:
                            status_text, status_color = "Corrected!", (255, 255, 0)
    else:
        status_text, status_color = "Paused", (0, 0, 255)

    frame_times.append(time.time() - loop_start)
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 1
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    display_frame = cv2.resize(frame, (DEBUG_WINDOW_W, DEBUG_WINDOW_H))
    display_frame = draw_hud(display_frame, mode, fps, status_text, status_color,
                              click_hold_ratio, gaze_model.model_name, live_confidence)
    cv2.imshow("Eye Cursor - Project", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    elif key == ord("r"):
        new_gm = run_calibration()
        if new_gm is not None:
            gaze_model = new_gm
            gl.save_calibration(gaze_model)
            kalman = gl.GazeKalman()
            prev_cursor_pos = None
            raw_prev_pos = None

    elif key == ord("p"):
        paused = not paused
        if paused:
            blink_start_time = None
            eyes_closed = False
        else:
            kalman = gl.GazeKalman()
            prev_cursor_pos = None
            raw_prev_pos = None

    elif key == ord("d"):
        run_diagnostic_test(gaze_model)
        kalman = gl.GazeKalman()
        prev_cursor_pos = None
        raw_prev_pos = None

    elif key == ord("c"):
        if not paused and mode == "MOVE" and prev_cursor_pos is not None:
            success2, frame2 = camera.read()
            if success2:
                frame2 = cv2.flip(frame2, 1)
                fh2, fw2 = frame2.shape[:2]
                landmarks2 = get_landmarks(frame2)
                if landmarks2 is not None:
                    feat2 = gl.extract_features(landmarks2, frame2, fw2, fh2)
                    if feat2.confidence >= CALIB_MIN_CONFIDENCE:
                        gaze_model.add_sample_and_refit(
                            feat2.vector, prev_cursor_pos[0] / SCREEN_W, prev_cursor_pos[1] / SCREEN_H)
                        gl.save_calibration(gaze_model)
                        correction_flash_until = now_ts() + 0.6
                        print("Added correction sample and refit model.")

camera.stop()
cv2.destroyAllWindows()