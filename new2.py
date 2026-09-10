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
import joblib
from collections import deque, namedtuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================================
# CONFIG
# ============================================

CAMERA_RESOLUTIONS = [(1280, 720), (960, 540), (640, 480)]
DETECTION_SCALE = 0.85

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Now stores a fitted regression model + scaler + raw training data (for
# online correction), not a geometric point cloud. Renamed again on purpose.
CALIBRATION_FILE = "calibration_model.pkl"

# --- Calibration grid ---
CALIB_GRID_COLS = 7
CALIB_GRID_ROWS = 7
CALIB_MARGIN = 0.04
CALIB_SETTLE_TIME = 0.4
CALIB_SAMPLE_TIME = 0.9
MIN_VALID_SAMPLES = 5          # per pass, below this the pass is dropped
CALIB_MIN_CONFIDENCE = 0.5     # frames below this confidence aren't sampled

# The interior (row,col) indices (of the 7x7 grid) that get a SECOND
# collection pass with a slightly different head position, and whose
# samples (both passes) are held out entirely as validation data — never
# trained on. This is what lets us report genuine, not self-reported,
# accuracy, and what teaches the model some head-position tolerance.
VALIDATION_ROWCOLS = {1, 3, 5}

# --- Saccade gating ---
# The eye is genuinely "between fixations" during a saccade — that data
# is motion blur, not signal. Velocity is in gaze-ratio-units/sec.
SACCADE_VELOCITY_THRESHOLD = 1.2
SACCADE_VELOCITY_MAX = 3.0

# --- Eye-openness confidence ---
EAR_FULLY_OPEN = 0.30   # ~fully open reference EAR, confidence ramps to 1 here

# --- Blink click ---
BLINK_EAR_THRESHOLD = 0.19
BLINK_HOLD_SECONDS = 0.2
CLICK_COOLDOWN = 0.8

# --- Mouth-open mode toggle (MOVE <-> SCROLL) ---
MOUTH_OPEN_THRESHOLD = 0.55
MODE_TOGGLE_HOLD_SECONDS = 0.2
MODE_TOGGLE_COOLDOWN = 1.0

# --- Scroll (driven by vertical gaze offset) ---
SCROLL_DEAD_ZONE = 0.02
SCROLL_SENSITIVITY = 45
SCROLL_MAX_OFFSET = 0.15

# --- Kalman filter (screen-pixel space) ---
KALMAN_PROCESS_NOISE = 0.6
KALMAN_MEASUREMENT_NOISE = 150.0
# Below this confidence, the Kalman filter predicts only (no correction) —
# i.e. it coasts on its own velocity estimate instead of trusting a noisy
# or blink-corrupted measurement. This is the real fix for shaking: don't
# feed bad frames into the estimator at all, rather than smoothing after.
CONFIDENCE_FREEZE_THRESHOLD = 0.15

# Hard safety net regardless of the above.
MAX_CURSOR_SPEED_PX_PER_SEC = 2200

# --- For converting pixel error to a visual-angle estimate (reporting only,
# doesn't affect tracking). Set these to your actual setup for a meaningful
# number. ---
APPROX_SCREEN_WIDTH_CM = 34.0
APPROX_VIEWING_DISTANCE_CM = 50.0

DEBUG_WINDOW_W, DEBUG_WINDOW_H = 480, 360
COUNTDOWN_SECONDS = 3

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ============================================
# LANDMARK INDICES
# ============================================

LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_VERT = (159, 145)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = (469, 470, 471, 472)

RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERT = (386, 374)
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = (474, 475, 476, 477)

MOUTH_CORNERS = (61, 291)
MOUTH_VERT = (13, 14)

NOSE_TIP = 1
CHIN = 152

# Generic 6-point 3D face model (mm) used for solvePnP head-pose estimation.
# It's not metrically exact for any specific face, but it doesn't need to
# be — it just needs to be a CONSISTENT feature frame to frame, since the
# personalized regression model learns to correct for whatever systematic
# offset it has.
HEAD_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),         # nose tip
    (0.0, -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0), # landmark 33
    (225.0, 170.0, -135.0),  # landmark 263
    (-150.0, -150.0, -125.0),# landmark 61
    (150.0, -150.0, -125.0), # landmark 291
], dtype=np.float64)

# ============================================
# THREADED CAMERA
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
# MODEL DOWNLOAD / FACE LANDMARKER SETUP
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
    small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    # CLAHE on the luminance channel: cheap robustness against backlight /
    # uneven lighting, which was visibly hurting landmark stability before.
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


# ============================================
# FEATURE EXTRACTION
# ============================================

def _dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def _aspect_ratio(landmarks, vert_pair, corner_pair, w, h):
    vert_dist = _dist(landmarks[vert_pair[0]], landmarks[vert_pair[1]], w, h)
    horiz_dist = _dist(landmarks[corner_pair[0]], landmarks[corner_pair[1]], w, h)
    return vert_dist / horiz_dist if horiz_dist > 1e-6 else 0.0


def _eye_gaze_ratio(landmarks, corners, vert, iris_idx):
    corner_a, corner_b = landmarks[corners[0]], landmarks[corners[1]]
    top, bottom = landmarks[vert[0]], landmarks[vert[1]]
    iris = landmarks[iris_idx]
    h_span = corner_b.x - corner_a.x
    v_span = bottom.y - top.y
    gx = (iris.x - corner_a.x) / h_span if abs(h_span) > 1e-6 else 0.5
    gy = (iris.y - top.y) / v_span if abs(v_span) > 1e-6 else 0.5
    return gx, gy


def _iris_radius_ratio(landmarks, center_idx, ring, corners, w, h):
    center = landmarks[center_idx]
    cx, cy = center.x * w, center.y * h
    radii = [math.hypot(landmarks[i].x * w - cx, landmarks[i].y * h - cy) for i in ring]
    radius = sum(radii) / len(radii)
    eye_w = _dist(landmarks[corners[0]], landmarks[corners[1]], w, h)
    return radius / eye_w if eye_w > 1e-6 else 0.0


def _ear_confidence(ear):
    span = max(EAR_FULLY_OPEN - BLINK_EAR_THRESHOLD, 1e-6)
    return float(np.clip((ear - BLINK_EAR_THRESHOLD) / span, 0.0, 1.0))


def _head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),
        (landmarks[CHIN].x * w, landmarks[CHIN].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(HEAD_MODEL_3D, image_points, camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    euler = cv2.decomposeProjectionMatrix(proj)[6].flatten()  # pitch, yaw, roll
    pitch, yaw, roll = euler[0], euler[1], euler[2]
    fx, fy, fz = tvec.flatten()
    return yaw, pitch, roll, fx, fy, fz


FeatureResult = namedtuple("FeatureResult", [
    "vector", "fused_gx", "fused_gy", "confidence", "left_conf", "right_conf",
    "avg_ear",
])

FEATURE_NAMES = [
    "left_gx", "left_gy", "right_gx", "right_gy",
    "left_ear", "right_ear",
    "left_eye_w", "left_eye_h", "right_eye_w", "right_eye_h",
    "left_iris_r", "right_iris_r",
    "yaw", "pitch", "roll", "face_x", "face_y", "face_z",
    "left_conf", "right_conf",
]


def extract_features(landmarks, w, h):
    l_gx, l_gy = _eye_gaze_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_VERT, LEFT_IRIS_CENTER)
    r_gx, r_gy = _eye_gaze_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_VERT, RIGHT_IRIS_CENTER)

    l_ear = _aspect_ratio(landmarks, LEFT_EYE_VERT, LEFT_EYE_CORNERS, w, h)
    r_ear = _aspect_ratio(landmarks, RIGHT_EYE_VERT, RIGHT_EYE_CORNERS, w, h)

    interocular = _dist(landmarks[33], landmarks[263], w, h)
    interocular = interocular if interocular > 1e-6 else 1.0

    l_eye_w = _dist(landmarks[LEFT_EYE_CORNERS[0]], landmarks[LEFT_EYE_CORNERS[1]], w, h) / interocular
    l_eye_h = _dist(landmarks[LEFT_EYE_VERT[0]], landmarks[LEFT_EYE_VERT[1]], w, h) / interocular
    r_eye_w = _dist(landmarks[RIGHT_EYE_CORNERS[0]], landmarks[RIGHT_EYE_CORNERS[1]], w, h) / interocular
    r_eye_h = _dist(landmarks[RIGHT_EYE_VERT[0]], landmarks[RIGHT_EYE_VERT[1]], w, h) / interocular

    l_iris_r = _iris_radius_ratio(landmarks, LEFT_IRIS_CENTER, LEFT_IRIS_RING, LEFT_EYE_CORNERS, w, h)
    r_iris_r = _iris_radius_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_IRIS_RING, RIGHT_EYE_CORNERS, w, h)

    yaw, pitch, roll, fx, fy, fz = _head_pose(landmarks, w, h)

    l_conf = _ear_confidence(l_ear)
    r_conf = _ear_confidence(r_ear)
    avg_ear = (l_ear + r_ear) / 2
    confidence = min(l_conf, r_conf)

    # Confidence-weighted fusion instead of a flat 50/50 average — an eye
    # that's more closed / less reliable this frame gets less say.
    total_w = l_conf + r_conf
    if total_w > 1e-6:
        fused_gx = (l_gx * l_conf + r_gx * r_conf) / total_w
        fused_gy = (l_gy * l_conf + r_gy * r_conf) / total_w
    else:
        fused_gx, fused_gy = (l_gx + r_gx) / 2, (l_gy + r_gy) / 2

    vector = np.array([
        l_gx, l_gy, r_gx, r_gy,
        l_ear, r_ear,
        l_eye_w, l_eye_h, r_eye_w, r_eye_h,
        l_iris_r, r_iris_r,
        yaw, pitch, roll, fx, fy, fz,
        l_conf, r_conf,
    ], dtype=np.float64)

    return FeatureResult(vector, fused_gx, fused_gy, confidence, l_conf, r_conf, avg_ear)


class VelocityTracker:
    """Tracks how fast the (fused) gaze point is moving, in ratio-units/sec —
    used to detect an in-progress saccade so that frame doesn't get sampled
    into calibration or trusted at full weight during tracking."""
    def __init__(self):
        self.prev = None
        self.prev_t = None

    def update(self, x, y, t):
        if self.prev is None:
            self.prev, self.prev_t = (x, y), t
            return 0.0
        dt = max(t - self.prev_t, 1e-3)
        v = math.hypot(x - self.prev[0], y - self.prev[1]) / dt
        self.prev, self.prev_t = (x, y), t
        return v


def saccade_factor(velocity):
    if velocity <= SACCADE_VELOCITY_THRESHOLD:
        return 1.0
    if velocity >= SACCADE_VELOCITY_MAX:
        return 0.0
    span = SACCADE_VELOCITY_MAX - SACCADE_VELOCITY_THRESHOLD
    return 1.0 - (velocity - SACCADE_VELOCITY_THRESHOLD) / span


# ============================================
# KALMAN FILTER (screen-pixel space, confidence-gated)
# ============================================

class GazeKalman:
    """Constant-velocity Kalman filter. Below CONFIDENCE_FREEZE_THRESHOLD it
    predicts only (coasts on its velocity estimate) instead of correcting
    with a measurement it doesn't trust — this is what actually replaces the
    old stillness-lock/speed-cap stack with something principled."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self._base_meas_noise = KALMAN_MEASUREMENT_NOISE
        self.initialized = False

    def reset(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = True

    def step(self, x, y, confidence):
        if not self.initialized:
            self.reset(x, y)
            return x, y

        noise_scale = 1.0 + (1.0 - confidence) * 20.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._base_meas_noise * noise_scale

        pred = self.kf.predict()

        if confidence < CONFIDENCE_FREEZE_THRESHOLD:
            return float(pred[0, 0]), float(pred[1, 0])

        meas = np.array([[np.float32(x)], [np.float32(y)]])
        est = self.kf.correct(meas)
        return float(est[0, 0]), float(est[1, 0])


# ============================================
# GAZE MODEL — feature vector -> screen (x, y), via personalized regression
# ============================================

def px_to_degrees(px_error):
    px_per_cm = SCREEN_W / APPROX_SCREEN_WIDTH_CM
    cm_error = px_error / px_per_cm
    return math.degrees(math.atan2(cm_error, APPROX_VIEWING_DISTANCE_CM))


class GazeModel:
    def __init__(self, scaler, model_name, model, X_train, y_train, X_val, y_val):
        self.scaler = scaler
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def predict(self, feature_vector):
        Xs = self.scaler.transform(feature_vector.reshape(1, -1))
        pred = self.model.predict(Xs)[0]
        return float(pred[0]), float(pred[1])

    def add_sample_and_refit(self, feature_vector, screen_x, screen_y):
        """Online drift correction: append a new (features -> known screen
        point) example and refit. Only meaningful when the label is a real,
        independent confirmation (e.g. the user explicitly confirming 'I am
        looking exactly at the cursor right now') — NOT just wherever the
        cursor already is, since that would be circular."""
        self.X_train = np.vstack([self.X_train, feature_vector.reshape(1, -1)])
        self.y_train = np.vstack([self.y_train, np.array([[screen_x, screen_y]])])
        Xs = self.scaler.transform(self.X_train)
        self.model.fit(Xs, self.y_train)


def train_and_select_model(X_train, y_train, X_val, y_val):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    candidates = {
        "Ridge": Ridge(alpha=5.0),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=0),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=0),
    }

    print("\n--- Model comparison (held-out validation points, never trained on) ---")
    results = {}
    for name, model in candidates.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xv)
        err = np.linalg.norm(pred - y_val, axis=1)
        mae_px = float(np.mean(err))
        rmse_px = float(np.sqrt(np.mean(err ** 2)))
        p95_px = float(np.percentile(err, 95))
        deg = px_to_degrees(mae_px)
        results[name] = (model, mae_px)
        print(f"  {name:12s} MAE: {mae_px:6.1f}px   RMSE: {rmse_px:6.1f}px   "
              f"95th pct: {p95_px:6.1f}px   (~{deg:.2f} deg visual angle)")

    best_name = min(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"Selected model: {best_name}\n")

    return GazeModel(scaler, best_name, best_model, X_train, y_train, X_val, y_val)


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    try:
        data = joblib.load(CALIBRATION_FILE)
        return GazeModel(data["scaler"], data["model_name"], data["model"],
                          data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    except Exception as e:
        print(f"Could not load existing calibration ({e}); will recalibrate.")
        return None


def save_calibration(gm):
    joblib.dump({
        "scaler": gm.scaler, "model_name": gm.model_name, "model": gm.model,
        "X_train": gm.X_train, "y_train": gm.y_train, "X_val": gm.X_val, "y_val": gm.y_val,
    }, CALIBRATION_FILE)


# ============================================
# CALIBRATION
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
    return points


def _sample_point(target_x, target_y, progress_text, extra_prompt=""):
    """Runs one settle+collect window at (target_x, target_y). Returns a
    list of feature vectors, or None if the user cancelled (ESC)."""
    vt = VelocityTracker()
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
                feat = extract_features(landmarks, fw, fh)
                v = vt.update(feat.fused_gx, feat.fused_gy, now_ts())
                if feat.confidence >= CALIB_MIN_CONFIDENCE and v < SACCADE_VELOCITY_THRESHOLD:
                    samples.append(feat.vector)
            cv2.circle(canvas, (target_x, target_y), 9, (0, 255, 0), -1)
            cv2.putText(canvas, f"samples: {len(samples)}", (target_x - 45, target_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{progress_text} - look at the dot, keep head still - ESC to cancel",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if extra_prompt:
            cv2.putText(canvas, extra_prompt, (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

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

    train_X, train_y, val_X, val_y = [], [], [], []

    for row_i, col_i, gx_norm, gy_norm, is_val in grid:
        target_x = int(gx_norm * SCREEN_W)
        target_y = int(gy_norm * SCREEN_H)

        event_i += 1
        samples = _sample_point(target_x, target_y, f"Point {event_i}/{total_events}")
        if samples is None:
            cv2.destroyWindow("Calibration")
            return None

        dest_X, dest_y = (val_X, val_y) if is_val else (train_X, train_y)
        if len(samples) < MIN_VALID_SAMPLES:
            print(f"Warning: point at ({row_i},{col_i}) only got {len(samples)} valid samples — skipping pass.")
        else:
            for s in samples:
                dest_X.append(s)
                dest_y.append((target_x, target_y))

        if is_val:
            event_i += 1
            samples2 = _sample_point(
                target_x, target_y, f"Point {event_i}/{total_events}",
                extra_prompt="Shift your head slightly (lean/tilt) - keep looking at the SAME dot",
            )
            if samples2 is None:
                cv2.destroyWindow("Calibration")
                return None
            if len(samples2) < MIN_VALID_SAMPLES:
                print(f"Warning: augmented pass at ({row_i},{col_i}) only got {len(samples2)} samples — skipping.")
            else:
                for s in samples2:
                    val_X.append(s)
                    val_y.append((target_x, target_y))

    cv2.destroyWindow("Calibration")

    if len(train_X) < 100 or len(val_X) < 20:
        print(f"Calibration failed: too little data collected (train={len(train_X)}, val={len(val_X)}).")
        return None

    return (np.array(train_X), np.array(train_y, dtype=np.float64),
            np.array(val_X), np.array(val_y, dtype=np.float64))


def verify_calibration(gm):
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    verify_kf = GazeKalman()

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
            feat = extract_features(landmarks, fw, fh)
            sx, sy = gm.predict(feat.vector)
            sx, sy = verify_kf.step(sx, sy, feat.confidence)
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
    print("Starting calibration: 49 points + 9 head-shifted validation reps (~58 total).")
    print("LOOK at each red dot with your EYES only.")
    while True:
        data = _collect_calibration_data()
        if data is None:
            return None
        X_train, y_train, X_val, y_val = data

        gm = train_and_select_model(X_train, y_train, X_val, y_val)

        verdict = verify_calibration(gm)
        if verdict is True:
            print("Calibration accepted.")
            return gm
        elif verdict is None:
            return None
        else:
            print("Redoing calibration...")
            continue


# ============================================
# HUD
# ============================================

def draw_hud(frame, mode, fps, status_text, status_color, click_hold_ratio, model_name, confidence):
    h, w = frame.shape[:2]
    mode_color = (0, 200, 255) if mode == "SCROLL" else (0, 255, 120)

    cv2.putText(frame, f"MODE: {mode}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA)

    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    status_x = w - status_size[0] - 12
    cv2.putText(frame, status_text, (status_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)

    bar_w, bar_h = 180, 10
    bar_x, bar_y = (w - bar_w) // 2, 42
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    fill = int(bar_w * np.clip(click_hold_ratio, 0, 1))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 255, 0), -1)

    conf_color = (0, 255, 0) if confidence > 0.6 else ((0, 255, 255) if confidence > 0.3 else (0, 0, 255))
    cv2.putText(frame, f"{model_name} | conf {confidence:.2f} | {fps:4.1f}fps", (12, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "ESC quit | R recalib | P pause | C correct-point | mouth: toggle scroll",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

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

gaze_model = load_calibration()

if gaze_model is None:
    gaze_model = run_calibration()
    if gaze_model is None:
        print("No calibration available. Exiting.")
        camera.stop()
        raise SystemExit(0)
    save_calibration(gaze_model)

kalman = GazeKalman()

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

correction_flash_until = 0
paused = False
frame_times = deque(maxlen=30)

cv2.namedWindow("Eye Cursor - Project", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye Cursor - Project", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
win_x = max(SCREEN_W - DEBUG_WINDOW_W - 30, 0)
win_y = 30
cv2.moveWindow("Eye Cursor - Project", win_x, win_y)

print("Eye-gaze cursor running. ESC quit | R recalibrate | P pause | C correct current point.")
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
            feat = extract_features(landmarks, w, h)
            live_confidence = feat.confidence
            mar = _aspect_ratio(landmarks, MOUTH_VERT, MOUTH_CORNERS, w, h)

            is_closed = feat.avg_ear < BLINK_EAR_THRESHOLD
            is_mouth_open = mar > MOUTH_OPEN_THRESHOLD

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
                        kalman = GazeKalman()
                        prev_cursor_pos = None
                        prev_cursor_time = None
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
                        status_text = "Scroll ready"
                        status_color = (0, 200, 255)
                else:
                    status_text = "Scroll ready"
                    status_color = (0, 200, 255)
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
                    mapped_x, mapped_y = gaze_model.predict(feat.vector)
                    fx, fy = kalman.step(mapped_x, mapped_y, feat.confidence)

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

                    prev_cursor_pos = (fx, fy)
                    prev_cursor_time = t

                    pyautogui.moveTo(int(fx), int(fy))

                    if status_text != "CLICK!":
                        status_text = "Tracking" if feat.confidence >= CONFIDENCE_FREEZE_THRESHOLD else "Low confidence"
                        status_color = (0, 255, 255) if feat.confidence >= CONFIDENCE_FREEZE_THRESHOLD else (0, 140, 255)
                        if t < click_message_until:
                            status_text = "CLICK!"
                            status_color = (0, 255, 0)
                        if t < correction_flash_until:
                            status_text = "Corrected!"
                            status_color = (255, 255, 0)
    else:
        status_text = "Paused"
        status_color = (0, 0, 255)

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
            save_calibration(gaze_model)
            kalman = GazeKalman()
            prev_cursor_pos = None
            prev_cursor_time = None

    elif key == ord("p"):
        paused = not paused
        if paused:
            blink_start_time = None
            eyes_closed = False
        else:
            kalman = GazeKalman()
            prev_cursor_pos = None
            prev_cursor_time = None

    elif key == ord("c"):
        # Online drift correction: only meaningful if you are ACTUALLY
        # looking exactly at the current cursor position right now.
        if not paused and mode == "MOVE" and prev_cursor_pos is not None:
            success2, frame2 = camera.read()
            if success2:
                frame2 = cv2.flip(frame2, 1)
                fh2, fw2 = frame2.shape[:2]
                landmarks2 = get_landmarks(frame2)
                if landmarks2 is not None:
                    feat2 = extract_features(landmarks2, fw2, fh2)
                    if feat2.confidence >= CALIB_MIN_CONFIDENCE:
                        gaze_model.add_sample_and_refit(feat2.vector, prev_cursor_pos[0], prev_cursor_pos[1])
                        save_calibration(gaze_model)
                        correction_flash_until = now_ts() + 0.6
                        print("Added correction sample and refit model.")

camera.stop()
cv2.destroyAllWindows()