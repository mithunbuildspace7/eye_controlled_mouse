import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import Delaunay
import pyautogui
import time
import os
import math
import threading
import urllib.request
from collections import deque

# ============================================
# CONFIG
# ============================================

CAMERA_RESOLUTIONS = [(1280, 720), (960, 540), (640, 480)]

# Iris landmarks are tiny, so we don't downscale as aggressively as a
# head-pose tracker would. Higher = more precise gaze, lower = more FPS.
DETECTION_SCALE = 0.85

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Stores the raw calibration point cloud (gaze ratios + matching screen
# coords), NOT a single homography matrix. Renamed so an old
# calibration_head.npy / calibration_gaze.npy is never accidentally loaded.
CALIBRATION_FILE = "calibration_points.npz"

# --- Calibration grid ---
# A dense grid trains local accuracy across the WHOLE screen instead of
# fitting one global transform. More points = more accurate but longer to
# calibrate. 5x5 = 25 points, roughly 45-60 seconds one-time cost.
CALIB_GRID_COLS = 5
CALIB_GRID_ROWS = 5
CALIB_MARGIN = 0.04          # how close to the true screen edge points go
CALIB_SETTLE_TIME = 0.5      # time to let your eye actually settle on a new dot
CALIB_SAMPLE_TIME = 1.2      # time spent collecting samples once settled
MIN_VALID_SAMPLES = 5        # below this, warn + skip the point

# --- One Euro Filter tuning ---
ONEEURO_MIN_CUTOFF = 0.03
ONEEURO_BETA = 0.25
ONEEURO_DCUTOFF = 1.0

# Hard speed cap applied after the filter, in screen px/sec. Whatever a
# single frame's gaze estimate does, the cursor can't jump farther than
# this per second — the actual fix for "flying" cursor jumps.
MAX_CURSOR_SPEED_PX_PER_SEC = 2200

# --- Blink click ---
BLINK_EAR_THRESHOLD = 0.19
BLINK_HOLD_SECONDS = 0.2
CLICK_COOLDOWN = 0.8

# --- Mouth-open mode toggle (MOVE <-> SCROLL) ---
MOUTH_OPEN_THRESHOLD = 0.55
MODE_TOGGLE_HOLD_SECONDS = 0.2
MODE_TOGGLE_COOLDOWN = 1.0

# --- Scroll (active only in SCROLL mode, driven by vertical gaze offset) ---
SCROLL_DEAD_ZONE = 0.02
SCROLL_SENSITIVITY = 45
SCROLL_MAX_OFFSET = 0.15

# --- Stillness Lock ---
STILLNESS_ENTER_RADIUS = 10
STILLNESS_EXIT_RADIUS = 18
STILLNESS_HOLD_TIME = 0.3

DEBUG_WINDOW_W, DEBUG_WINDOW_H = 480, 360
COUNTDOWN_SECONDS = 3

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ============================================
# LANDMARK INDICES
# ============================================
# MediaPipe's FaceLandmarker task outputs 478 landmarks by default (468 face
# + 10 iris), no extra config needed. Iris centers: 468 pairs with the
# (33,133) eye, 473 pairs with the (362,263) eye.

LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_VERT = (159, 145)
LEFT_IRIS_CENTER = 468

RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERT = (386, 374)
RIGHT_IRIS_CENTER = 473

MOUTH_CORNERS = (61, 291)
MOUTH_VERT = (13, 14)

# ============================================
# ONE EURO FILTER
# ============================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, te):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        te = max(t - self.t_prev, 1e-3)
        freq = 1.0 / te

        dx = (x - self.x_prev) * freq
        a_d = self._alpha(self.d_cutoff, te)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, te)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


filter_x = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_DCUTOFF)
filter_y = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_DCUTOFF)

# ============================================
# THREADED CAMERA (fixes lag: capture never blocks the main loop)
# ============================================

class ThreadedCamera:
    def __init__(self, resolutions):
        self.cap = None
        self.width = None
        self.height = None
        self._open_camera(resolutions)

        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.3)  # warm up

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
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera opened at {actual_w}x{actual_h}")
                self.cap = cap
                self.width, self.height = actual_w, actual_h
                return

        raise RuntimeError("Could not open webcam at any supported resolution.")

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.ok = True
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


# ============================================
# MODEL DOWNLOAD
# ============================================

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

# ============================================
# FACE LANDMARKER SETUP
# ============================================

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

# ============================================
# CAMERA
# ============================================

camera = ThreadedCamera(CAMERA_RESOLUTIONS)

start_time = time.time()


def now_ts():
    return time.time() - start_time


def get_timestamp_ms():
    return int(now_ts() * 1000)


# ============================================
# HELPERS
# ============================================

def aspect_ratio(landmarks, vert_pair, corner_pair, w, h):
    top = landmarks[vert_pair[0]]
    bottom = landmarks[vert_pair[1]]
    left = landmarks[corner_pair[0]]
    right = landmarks[corner_pair[1]]

    vert_dist = np.hypot((top.x - bottom.x) * w, (top.y - bottom.y) * h)
    horiz_dist = np.hypot((left.x - right.x) * w, (left.y - right.y) * h)

    if horiz_dist == 0:
        return 0.0

    return vert_dist / horiz_dist


def _eye_gaze_ratio(landmarks, corners, vert, iris_idx):
    """Where the iris sits inside ONE eye socket, as a 0-1 ratio on each axis.
    This is head-position-invariant (it's a ratio of the iris offset to the
    eye's own width/height), so small head drift doesn't blow up the mapping
    the way raw iris coordinates would."""
    corner_a = landmarks[corners[0]]
    corner_b = landmarks[corners[1]]
    top = landmarks[vert[0]]
    bottom = landmarks[vert[1]]
    iris = landmarks[iris_idx]

    h_span = corner_b.x - corner_a.x
    v_span = bottom.y - top.y

    gx = (iris.x - corner_a.x) / h_span if abs(h_span) > 1e-6 else 0.5
    gy = (iris.y - top.y) / v_span if abs(v_span) > 1e-6 else 0.5

    return gx, gy


def get_gaze_point(landmarks):
    """Average both eyes' gaze ratio into a single (gx, gy) point — driven
    purely by where the iris sits inside each eye, not by head position."""
    l_gx, l_gy = _eye_gaze_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_VERT, LEFT_IRIS_CENTER)
    r_gx, r_gy = _eye_gaze_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_VERT, RIGHT_IRIS_CENTER)

    gaze_x = (l_gx + r_gx) / 2
    gaze_y = (l_gy + r_gy) / 2
    return gaze_x, gaze_y


def get_landmarks(frame):
    small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, get_timestamp_ms())

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


def eyes_open(landmarks, w, h):
    l_ear = aspect_ratio(landmarks, LEFT_EYE_VERT, LEFT_EYE_CORNERS, w, h)
    r_ear = aspect_ratio(landmarks, RIGHT_EYE_VERT, RIGHT_EYE_CORNERS, w, h)
    return (l_ear + r_ear) / 2 >= BLINK_EAR_THRESHOLD


# ============================================
# GAZE MAPPER — dense-grid piecewise-linear interpolation
# ============================================
# A single global homography (the old approach) forces one rigid transform
# onto the WHOLE screen, but eyeball rotation isn't actually planar/
# projective — so a global fit is systematically worse near the edges and
# corners than near the calibration centroid. Instead, this trains many
# points across the screen and, at runtime, interpolates only from the
# 2-3 calibration points immediately surrounding your current gaze
# (a Delaunay triangulation + barycentric interpolation). Each point you
# calibrate only has to be locally accurate to its own neighborhood — which
# is exactly what "train each point" means in practice.

class GazeMapper:
    def __init__(self, gaze_pts, screen_pts):
        self.gaze_pts = np.asarray(gaze_pts, dtype=np.float64)
        self.screen_pts = np.asarray(screen_pts, dtype=np.float64)
        self.tri = Delaunay(self.gaze_pts)

    def map(self, gx, gy):
        p = np.array([gx, gy])
        simplex = self.tri.find_simplex(p)

        if simplex >= 0:
            # Inside the calibrated area: exact barycentric interpolation
            # between the 3 surrounding calibration points.
            b = self.tri.transform[simplex, :2].dot(p - self.tri.transform[simplex, 2])
            bary = np.array([b[0], b[1], 1 - b[0] - b[1]])
            verts = self.tri.simplices[simplex]
            screen_pt = bary @ self.screen_pts[verts]
            return float(screen_pt[0]), float(screen_pt[1])

        # Outside the calibrated area (e.g. looking past the outermost
        # trained points) — fall back to an inverse-distance-weighted blend
        # of the nearest few calibration points instead of wildly
        # extrapolating a triangle's plane.
        k = min(4, len(self.gaze_pts))
        d = np.linalg.norm(self.gaze_pts - p, axis=1)
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        w /= w.sum()
        screen_pt = (w[:, None] * self.screen_pts[idx]).sum(axis=0)
        return float(screen_pt[0]), float(screen_pt[1])


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    try:
        data = np.load(CALIBRATION_FILE)
        return GazeMapper(data["gaze"], data["screen"])
    except Exception as e:
        print(f"Could not load existing calibration ({e}); will recalibrate.")
        return None


def save_calibration(mapper):
    np.savez(CALIBRATION_FILE, gaze=mapper.gaze_pts, screen=mapper.screen_pts)


# ============================================
# CALIBRATION
# ============================================

def _generate_grid_points():
    """Boustrophedon (snake) order across the grid so consecutive targets
    are always adjacent — no big eye jumps between calibration points."""
    xs = np.linspace(CALIB_MARGIN, 1 - CALIB_MARGIN, CALIB_GRID_COLS)
    ys = np.linspace(CALIB_MARGIN, 1 - CALIB_MARGIN, CALIB_GRID_ROWS)

    points = []
    for row_i, gy in enumerate(ys):
        row_xs = xs if row_i % 2 == 0 else xs[::-1]
        for gx in row_xs:
            points.append((float(gx), float(gy)))
    return points


def _collect_calibration_points():
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    grid = _generate_grid_points()
    total = len(grid)

    screen_points = []
    gaze_points = []

    for i, (gx_norm, gy_norm) in enumerate(grid):
        target_x = int(gx_norm * SCREEN_W)
        target_y = int(gy_norm * SCREEN_H)

        settle_start = time.time()
        samples = []

        while True:
            success, frame = camera.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            elapsed = time.time() - settle_start

            cv2.circle(canvas, (target_x, target_y), 20, (0, 0, 255), -1)
            cv2.circle(canvas, (target_x, target_y), 20, (255, 255, 255), 2)

            if elapsed > CALIB_SETTLE_TIME:
                landmarks = get_landmarks(frame)
                if landmarks is not None and eyes_open(landmarks, frame_w, frame_h):
                    samples.append(get_gaze_point(landmarks))
                cv2.circle(canvas, (target_x, target_y), 9, (0, 255, 0), -1)
                cv2.putText(canvas, f"samples: {len(samples)}", (target_x - 45, target_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.putText(canvas, f"Point {i + 1}/{total} - look at the dot, keep head still - ESC to cancel",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Calibration", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cv2.destroyWindow("Calibration")
                return None

            if elapsed > CALIB_SETTLE_TIME + CALIB_SAMPLE_TIME:
                break

        if len(samples) < MIN_VALID_SAMPLES:
            print(f"Warning: point {i + 1}/{total} only got {len(samples)} valid samples — skipping it.")
            continue

        # Median, not mean: robust to any stray outlier frame that slipped
        # past the open-eyes check (motion blur, momentary detection noise).
        gaze_points.append(np.median(samples, axis=0))
        screen_points.append((target_x, target_y))

    cv2.destroyWindow("Calibration")

    if len(gaze_points) < 6:
        print("Calibration failed: too few valid points collected.")
        return None

    return GazeMapper(gaze_points, screen_points)


def verify_calibration(mapper):
    """Show your live tracked point on screen so you can SEE the accuracy
    before committing to it, instead of trusting a single summary number."""
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    verify_filter_x = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_DCUTOFF)
    verify_filter_y = OneEuroFilter(ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_DCUTOFF)

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        # light reference grid so you can judge position against something
        for gx_norm in np.linspace(0.1, 0.9, 5):
            x = int(gx_norm * SCREEN_W)
            cv2.line(canvas, (x, 0), (x, SCREEN_H), (40, 40, 40), 1)
        for gy_norm in np.linspace(0.1, 0.9, 5):
            y = int(gy_norm * SCREEN_H)
            cv2.line(canvas, (0, y), (SCREEN_W, y), (40, 40, 40), 1)

        landmarks = get_landmarks(frame)
        if landmarks is not None:
            gx, gy = get_gaze_point(landmarks)
            sx, sy = mapper.map(gx, gy)
            t = now_ts()
            sx = verify_filter_x(sx, t)
            sy = verify_filter_y(sy, t)
            sx = int(np.clip(sx, 0, SCREEN_W - 1))
            sy = int(np.clip(sy, 0, SCREEN_H - 1))
            cv2.drawMarker(canvas, (sx, sy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(canvas, (sx, sy), 14, (0, 255, 0), 2)

        cv2.putText(canvas, "Look around the screen to check accuracy",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "SPACE = accept   R = redo calibration   ESC = cancel",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            cv2.destroyWindow("Calibration")
            return True
        elif key == ord("r"):
            cv2.destroyWindow("Calibration")
            return False
        elif key == 27:
            cv2.destroyWindow("Calibration")
            return None


def run_calibration():
    print("Starting calibration. LOOK at each red dot with your EYES only — keep your head still.")
    while True:
        mapper = _collect_calibration_points()
        if mapper is None:
            return None

        verdict = verify_calibration(mapper)
        if verdict is True:
            print("Calibration accepted.")
            return mapper
        elif verdict is None:
            return None
        else:
            print("Redoing calibration...")
            continue


# ============================================
# HUD / VISUAL POLISH
# ============================================

def draw_hud(frame, mode, fps, status_text, status_color, click_hold_ratio):
    h, w = frame.shape[:2]

    mode_color = (0, 200, 255) if mode == "SCROLL" else (0, 255, 120)

    cv2.putText(frame, f"MODE: {mode}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA)

    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    status_x = w - status_size[0] - 12
    cv2.putText(frame, status_text, (status_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)

    bar_w, bar_h = 180, 10
    bar_x = (w - bar_w) // 2
    bar_y = 42
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    fill = int(bar_w * np.clip(click_hold_ratio, 0, 1))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 255, 0), -1)

    cv2.putText(frame, "ESC quit | R recalib | P pause | mouth-open: toggle scroll | keep head still",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

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

mapper = load_calibration()

if mapper is None:
    mapper = run_calibration()
    if mapper is None:
        print("No calibration available. Exiting.")
        camera.stop()
        raise SystemExit(0)
    save_calibration(mapper)

blink_start_time = None
last_click_time = 0
click_message_until = 0
eyes_closed = False

mouth_open_start = None
last_mode_toggle = 0
mode = "MOVE"
scroll_baseline_y = None

# stillness lock state
stillness_locked = False
lock_anchor = None
still_ref_point = None
still_ref_time = None


def reset_stillness():
    global stillness_locked, lock_anchor, still_ref_point, still_ref_time
    stillness_locked = False
    lock_anchor = None
    still_ref_point = None
    still_ref_time = None

# Tracks the cursor's last actual on-screen position/time so movement can be
# speed-capped, independent of how noisy a single frame's gaze read is.
prev_cursor_pos = None
prev_cursor_time = None

paused = False
frame_times = deque(maxlen=30)

cv2.namedWindow("Eye Cursor - Project", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Cursor - Project", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
win_x = max(SCREEN_W - DEBUG_WINDOW_W - 30, 0)
win_y = 30
cv2.moveWindow("Eye Cursor - Project", win_x, win_y)

print("Eye-gaze cursor running. Press ESC to quit, R to recalibrate, P to pause.")
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

    if not paused:
        landmarks = get_landmarks(frame)

        if landmarks is not None:
            gaze_point = get_gaze_point(landmarks)

            left_ear = aspect_ratio(landmarks, LEFT_EYE_VERT, LEFT_EYE_CORNERS, w, h)
            right_ear = aspect_ratio(landmarks, RIGHT_EYE_VERT, RIGHT_EYE_CORNERS, w, h)
            avg_ear = (left_ear + right_ear) / 2
            mar = aspect_ratio(landmarks, MOUTH_VERT, MOUTH_CORNERS, w, h)

            is_closed = avg_ear < BLINK_EAR_THRESHOLD
            is_mouth_open = mar > MOUTH_OPEN_THRESHOLD

            # ---- Mode toggle (mouth open, held briefly) ----
            if is_mouth_open:
                if mouth_open_start is None:
                    mouth_open_start = t
                held_mouth = t - mouth_open_start
                if held_mouth >= MODE_TOGGLE_HOLD_SECONDS and t - last_mode_toggle > MODE_TOGGLE_COOLDOWN:
                    mode = "SCROLL" if mode == "MOVE" else "MOVE"
                    last_mode_toggle = t
                    mouth_open_start = t
                    if mode == "SCROLL":
                        scroll_baseline_y = gaze_point[1]
                    else:
                        filter_x.reset()
                        filter_y.reset()
                        reset_stillness()
                        prev_cursor_pos = None
                        prev_cursor_time = None
            else:
                mouth_open_start = None

            if mode == "SCROLL":
                # ---- Scroll mode: look up/down with your EYES, clicking disabled ----
                offset = gaze_point[1] - scroll_baseline_y
                if abs(offset) > SCROLL_DEAD_ZONE:
                    clamped = float(np.clip(offset, -SCROLL_MAX_OFFSET, SCROLL_MAX_OFFSET))
                    scroll_amount = -int(clamped * SCROLL_SENSITIVITY / SCROLL_MAX_OFFSET * 10)
                    if scroll_amount != 0:
                        pyautogui.scroll(scroll_amount)
                        status_text = "Scrolling up" if scroll_amount > 0 else "Scrolling down"
                        status_color = (0, 200, 255)
                    else:
                        status_text = "Scroll ready"
                        status_color = (0, 200, 255)
                else:
                    status_text = "Scroll ready"
                    status_color = (0, 200, 255)
                blink_start_time = None
                eyes_closed = False

            else:
                # ---- MOVE mode: blink click + filtered eye-gaze cursor movement ----
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
                    mapped_x, mapped_y = mapper.map(gaze_point[0], gaze_point[1])

                    raw_x = float(np.clip(mapped_x, 0, SCREEN_W - 1))
                    raw_y = float(np.clip(mapped_y, 0, SCREEN_H - 1))

                    # ---- Stillness Lock: freeze cursor completely when gaze is steady ----
                    newly_locked = False

                    if not stillness_locked:
                        if still_ref_point is None:
                            still_ref_point = (raw_x, raw_y)
                            still_ref_time = t
                        else:
                            d_ref = math.hypot(raw_x - still_ref_point[0], raw_y - still_ref_point[1])
                            if d_ref > STILLNESS_ENTER_RADIUS:
                                still_ref_point = (raw_x, raw_y)
                                still_ref_time = t
                            elif t - still_ref_time >= STILLNESS_HOLD_TIME:
                                stillness_locked = True
                                lock_anchor = still_ref_point
                                newly_locked = True

                    if stillness_locked:
                        d_lock = math.hypot(raw_x - lock_anchor[0], raw_y - lock_anchor[1])
                        if d_lock > STILLNESS_EXIT_RADIUS:
                            stillness_locked = False
                            still_ref_point = (raw_x, raw_y)
                            still_ref_time = t
                            filter_x.reset()
                            filter_y.reset()
                            prev_cursor_pos = lock_anchor
                            prev_cursor_time = t
                        else:
                            if newly_locked:
                                pyautogui.moveTo(int(lock_anchor[0]), int(lock_anchor[1]))
                            if status_text != "CLICK!":
                                status_text = "Locked (still)"
                                status_color = (150, 255, 150)

                    if not stillness_locked:
                        fx = filter_x(raw_x, t)
                        fy = filter_y(raw_y, t)

                        if prev_cursor_pos is not None:
                            dt = max(t - prev_cursor_time, 1e-3)
                            max_step = MAX_CURSOR_SPEED_PX_PER_SEC * dt
                            dx = fx - prev_cursor_pos[0]
                            dy = fy - prev_cursor_pos[1]
                            dist = math.hypot(dx, dy)
                            if dist > max_step:
                                scale = max_step / dist
                                fx = prev_cursor_pos[0] + dx * scale
                                fy = prev_cursor_pos[1] + dy * scale

                        prev_cursor_pos = (fx, fy)
                        prev_cursor_time = t

                        pyautogui.moveTo(int(fx), int(fy))

                        if status_text != "CLICK!":
                            status_text = "Tracking"
                            status_color = (0, 255, 255)

                            if t < click_message_until:
                                status_text = "CLICK!"
                                status_color = (0, 255, 0)
    else:
        status_text = "Paused"
        status_color = (0, 0, 255)

    frame_times.append(time.time() - loop_start)
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 1
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    display_frame = cv2.resize(frame, (DEBUG_WINDOW_W, DEBUG_WINDOW_H))
    display_frame = draw_hud(display_frame, mode, fps, status_text, status_color, click_hold_ratio)

    cv2.imshow("Eye Cursor - Project", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord("r"):
        new_mapper = run_calibration()
        if new_mapper is not None:
            mapper = new_mapper
            save_calibration(mapper)
            filter_x.reset()
            filter_y.reset()
            reset_stillness()
            prev_cursor_pos = None
            prev_cursor_time = None
    elif key == ord("p"):
        paused = not paused
        if paused:
            blink_start_time = None
            eyes_closed = False
        else:
            filter_x.reset()
            filter_y.reset()
            reset_stillness()
            prev_cursor_pos = None
            prev_cursor_time = None

camera.stop()
cv2.destroyAllWindows()