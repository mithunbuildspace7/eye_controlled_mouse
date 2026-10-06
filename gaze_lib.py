"""
gaze_lib.py — shared gaze-estimation logic, no camera/display side effects.
Imported by eye_gaze_cursor.py (live) and offline_eval.py (offline).
"""
import math
import time
import csv
import os

import cv2
import numpy as np
import joblib
from collections import namedtuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================================
# LANDMARK INDICES (verified against MediaPipe's published 478-point map)
# ============================================

LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_VERT = (159, 145)          # kept for backward reference
LEFT_UPPER_LID = (160, 159, 158)    # multi-point upper lid -> stabilized EAR
LEFT_LOWER_LID = (144, 145, 153)    # multi-point lower lid
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = (469, 470, 471, 472)

RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERT = (386, 374)
RIGHT_UPPER_LID = (387, 386, 385)
RIGHT_LOWER_LID = (373, 374, 380)
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = (474, 475, 476, 477)

MOUTH_CORNERS = (61, 291)
MOUTH_VERT = (13, 14)

NOSE_TIP = 1
CHIN = 152

HEAD_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# ============================================
# CONFIG (feature/model/filter side — no hardware here)
# ============================================

BLINK_EAR_THRESHOLD = 0.19
EAR_FULLY_OPEN = 0.30

SACCADE_VELOCITY_THRESHOLD = 1.2
SACCADE_VELOCITY_MAX = 3.0

KALMAN_PROCESS_NOISE = 3.0       # was 0.6 — higher = filter adapts to velocity changes faster (less lag)
KALMAN_MEASUREMENT_NOISE = 70.0  # was 150.0 — lower = trusts incoming measurements more readily
CONFIDENCE_FREEZE_THRESHOLD = 0.10  # was 0.15 — corner gaze reads as lower confidence; don't stall short of it

# Runtime plausibility guard (item 20/25): a raw prediction implying a
# faster-than-this jump gets treated as low-confidence rather than trusted.
RUNTIME_MAX_PLAUSIBLE_SPEED_PX = 4000.0

APPROX_SCREEN_WIDTH_CM = 34.0
APPROX_VIEWING_DISTANCE_CM = 50.0

# Was 4 — the eye crop is upscaled this much before thresholding. 4x meant
# thresholding/contour-finding on 4x the pixels for a precision gain that
# wasn't worth the latency; 2x is still enough detail for a good pupil blob.
PUPIL_ROI_UPSCALE = 2

_PUPIL_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

FEATURE_NAMES = (
    ["l_" + n for n in ["gx", "gy", "ear", "eye_w", "eye_h", "iris_r", "ring_disagree",
                         "px_gx", "px_gy", "px_quality", "conf"]] +
    ["r_" + n for n in ["gx", "gy", "ear", "eye_w", "eye_h", "iris_r", "ring_disagree",
                         "px_gx", "px_gy", "px_quality", "conf"]] +
    ["yaw", "pitch", "roll", "face_x", "face_y", "face_z", "interocular_norm", "eye_disagree"]
)
N_FEATURES = len(FEATURE_NAMES)  # 30


# ============================================
# GEOMETRY HELPERS
# ============================================

def _dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def _pt(lm, w, h):
    return lm.x * w, lm.y * h


def _local_eye_axes(landmarks, corners, w, h):
    """Eye-local coordinate frame: X = corner_a->corner_b (rotates with head
    roll), Y = perpendicular. Returns (center, ux, uy_perp, eye_w)."""
    ax, ay = _pt(landmarks[corners[0]], w, h)
    bx, by = _pt(landmarks[corners[1]], w, h)
    ex, ey = bx - ax, by - ay
    eye_w = math.hypot(ex, ey)
    if eye_w < 1e-6:
        return (ax, ay), (1.0, 0.0), (0.0, 1.0), 1.0
    ux, uy = ex / eye_w, ey / eye_w
    perp = (-uy, ux)
    center = ((ax + bx) / 2, (ay + by) / 2)
    return center, (ux, uy), perp, eye_w


def _project_local(point_px, py, center, axis_x, axis_y_perp, eye_w, eye_h):
    dx, dy = point_px - center[0], py - center[1]
    lx = (dx * axis_x[0] + dy * axis_x[1]) / eye_w
    ly = (dx * axis_y_perp[0] + dy * axis_y_perp[1]) / (eye_h if eye_h > 1e-6 else 1.0)
    return lx + 0.5, ly + 0.5


def _lid_center(landmarks, indices, w, h):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def detect_pupil_pixel(gray_full, landmarks, corners, w, h, upscale=PUPIL_ROI_UPSCALE):
    """Crop the eye region from the ORIGINAL-resolution grayscale frame,
    upscale it, and find the pupil as the largest sufficiently-circular dark
    blob. Returns (px, py, quality) in original-frame pixel coords, or None
    if nothing plausible was found."""
    xs = [landmarks[i].x * w for i in (corners[0], corners[1])]
    ys = [landmarks[i].y * h for i in (corners[0], corners[1])]
    eye_w = max(xs) - min(xs)
    if eye_w < 4:
        return None
    mx, my = eye_w * 0.4, eye_w * 0.7
    x0, x1 = max(int(min(xs) - mx), 0), min(int(max(xs) + mx), w)
    y0, y1 = max(int(min(ys) - my), 0), min(int(max(ys) + my), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    roi = gray_full[y0:y1, x0:x1]
    roi = _PUPIL_CLAHE.apply(roi)  # CLAHE only on this tiny crop now, not the whole frame — much cheaper
    roi_up = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    roi_up = cv2.GaussianBlur(roi_up, (5, 5), 0)

    try:
        _, thresh = cv2.threshold(roi_up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error:
        return None

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    rh, rw = roi_up.shape[:2]
    ccx, ccy = rw / 2, rh / 2
    best, best_score = None, -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > rw * rh * 0.6:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 1:
            continue
        circularity = float(np.clip(area / (math.pi * radius * radius + 1e-6), 0, 1))
        center_penalty = math.hypot(cx - ccx, cy - ccy) / (rw + 1e-6)
        score = circularity - center_penalty
        if score > best_score:
            best_score = score
            best = (cx, cy, circularity)

    if best is None:
        return None

    cx, cy, circularity = best
    px = x0 + cx / upscale
    py = y0 + cy / upscale
    return px, py, circularity


def _head_pose(landmarks, w, h):
    image_points = np.array([
        _pt(landmarks[NOSE_TIP], w, h),
        _pt(landmarks[CHIN], w, h),
        _pt(landmarks[33], w, h),
        _pt(landmarks[263], w, h),
        _pt(landmarks[61], w, h),
        _pt(landmarks[291], w, h),
    ], dtype=np.float64)

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(HEAD_MODEL_3D, image_points, camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    euler = cv2.decomposeProjectionMatrix(proj)[6].flatten()
    pitch, yaw, roll = euler[0], euler[1], euler[2]
    fx, fy, fz = tvec.flatten()
    return yaw, pitch, roll, fx, fy, fz


def _ear_confidence(ear):
    span = max(EAR_FULLY_OPEN - BLINK_EAR_THRESHOLD, 1e-6)
    return float(np.clip((ear - BLINK_EAR_THRESHOLD) / span, 0.0, 1.0))


def _eye_block(landmarks, gray_full, corners, upper_lid, lower_lid, iris_center_idx, ring, w, h):
    center, ax, ay_perp, eye_w = _local_eye_axes(landmarks, corners, w, h)

    top_cx, top_cy = _lid_center(landmarks, upper_lid, w, h)
    bot_cx, bot_cy = _lid_center(landmarks, lower_lid, w, h)
    eye_h = math.hypot(top_cx - bot_cx, top_cy - bot_cy)
    ear = eye_h / eye_w if eye_w > 1e-6 else 0.0

    ring_pts = [_pt(landmarks[i], w, h) for i in ring]
    ring_cx = sum(p[0] for p in ring_pts) / len(ring_pts)
    ring_cy = sum(p[1] for p in ring_pts) / len(ring_pts)

    raw_center = _pt(landmarks[iris_center_idx], w, h)
    ring_disagree = math.hypot(ring_cx - raw_center[0], ring_cy - raw_center[1]) / eye_w if eye_w > 1e-6 else 0.0

    iris_radii = [math.hypot(p[0] - ring_cx, p[1] - ring_cy) for p in ring_pts]
    iris_r = (sum(iris_radii) / len(iris_radii)) / eye_w if eye_w > 1e-6 else 0.0

    gx, gy = _project_local(ring_cx, ring_cy, center, ax, ay_perp, eye_w, eye_h)

    pupil = detect_pupil_pixel(gray_full, landmarks, corners, w, h)
    if pupil is not None:
        px_gx, px_gy = _project_local(pupil[0], pupil[1], center, ax, ay_perp, eye_w, eye_h)
        px_quality = pupil[2]
    else:
        px_gx, px_gy, px_quality = gx, gy, 0.0

    conf = _ear_confidence(ear)

    return dict(gx=gx, gy=gy, ear=ear, eye_w=eye_w, eye_h=eye_h, iris_r=iris_r,
                ring_disagree=ring_disagree, px_gx=px_gx, px_gy=px_gy,
                px_quality=px_quality, conf=conf, interocular_denom=eye_w)


FeatureResult = namedtuple("FeatureResult", [
    "vector", "fused_gx", "fused_gy", "confidence", "left_conf", "right_conf",
    "avg_ear", "eye_disagree", "yaw", "pitch", "roll",
])


def extract_features(landmarks, frame_bgr, w, h):
    # Plain grayscale here — CLAHE now happens inside detect_pupil_pixel on
    # just the small eye crop, not on the whole frame (much cheaper, same
    # contrast-normalization benefit where it actually matters).
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    L = _eye_block(landmarks, gray_full, LEFT_EYE_CORNERS, LEFT_UPPER_LID, LEFT_LOWER_LID,
                    LEFT_IRIS_CENTER, LEFT_IRIS_RING, w, h)
    R = _eye_block(landmarks, gray_full, RIGHT_EYE_CORNERS, RIGHT_UPPER_LID, RIGHT_LOWER_LID,
                    RIGHT_IRIS_CENTER, RIGHT_IRIS_RING, w, h)

    interocular = _dist(landmarks[33], landmarks[263], w, h)
    interocular = interocular if interocular > 1e-6 else 1.0
    interocular_norm = interocular / math.hypot(w, h)

    eye_disagree = math.hypot(L["gx"] - R["gx"], L["gy"] - R["gy"])

    yaw, pitch, roll, fx, fy, fz = _head_pose(landmarks, w, h)

    avg_ear = (L["ear"] + R["ear"]) / 2
    total_w = L["conf"] + R["conf"]
    if total_w > 1e-6:
        fused_gx = (L["gx"] * L["conf"] + R["gx"] * R["conf"]) / total_w
        fused_gy = (L["gy"] * L["conf"] + R["gy"] * R["conf"]) / total_w
    else:
        fused_gx, fused_gy = (L["gx"] + R["gx"]) / 2, (L["gy"] + R["gy"]) / 2

    disagree_penalty = float(np.clip(eye_disagree * 0.9, 0.0, 0.35))
    confidence = min(L["conf"], R["conf"]) * (1.0 - disagree_penalty)

    vector = np.array([
        L["gx"], L["gy"], L["ear"], L["eye_w"] / interocular, L["eye_h"] / interocular,
        L["iris_r"], L["ring_disagree"], L["px_gx"], L["px_gy"], L["px_quality"], L["conf"],
        R["gx"], R["gy"], R["ear"], R["eye_w"] / interocular, R["eye_h"] / interocular,
        R["iris_r"], R["ring_disagree"], R["px_gx"], R["px_gy"], R["px_quality"], R["conf"],
        yaw, pitch, roll, fx, fy, fz, interocular_norm, eye_disagree,
    ], dtype=np.float64)

    return FeatureResult(vector, fused_gx, fused_gy, confidence, L["conf"], R["conf"],
                          avg_ear, eye_disagree, yaw, pitch, roll)


class VelocityTracker:
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
# KALMAN FILTER — dt-aware, confidence-gated
# ============================================

class GazeKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.initialized = False
        self.last_t = None

    def reset(self, x, y, t):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = True
        self.last_t = t

    def step(self, x, y, confidence, t):
        if not self.initialized:
            self.reset(x, y, t)
            return x, y

        dt = max(t - self.last_t, 1e-3)
        dt = min(dt, 0.2)  # guard against huge dt after a pause/stall
        self.last_t = t

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE * dt

        noise_scale = 1.0 + (1.0 - confidence) * 8.0  # was *20.0 — less penalty at moderate confidence
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE * noise_scale

        pred = self.kf.predict()

        if confidence < CONFIDENCE_FREEZE_THRESHOLD:
            return float(pred[0, 0]), float(pred[1, 0])

        meas = np.array([[np.float32(x)], [np.float32(y)]])
        est = self.kf.correct(meas)
        return float(est[0, 0]), float(est[1, 0])


# ============================================
# REGION LABELING (for error breakdown)
# ============================================

def region_label(x_norm, y_norm):
    left, right = x_norm < 0.15, x_norm > 0.85
    top, bottom = y_norm < 0.15, y_norm > 0.85
    if top and left:
        return "top-left"
    if top and right:
        return "top-right"
    if bottom and left:
        return "bottom-left"
    if bottom and right:
        return "bottom-right"
    if left:
        return "left"
    if right:
        return "right"
    if top:
        return "top"
    if bottom:
        return "bottom"
    if 0.35 <= x_norm <= 0.65 and 0.35 <= y_norm <= 0.65:
        return "center"
    return "mid"


# ============================================
# MODEL
# ============================================

def px_to_degrees(px_error, screen_w):
    px_per_cm = screen_w / APPROX_SCREEN_WIDTH_CM
    cm_error = px_error / px_per_cm
    return math.degrees(math.atan2(cm_error, APPROX_VIEWING_DISTANCE_CM))


class GazeModel:
    """Predicts NORMALIZED [0,1] screen coordinates (item 33) — resolution
    independent. Caller multiplies by their actual SCREEN_W/H."""
    def __init__(self, scaler, model_name, model, X_train, y_train_norm, X_val, y_val_norm):
        self.scaler = scaler
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.y_train = y_train_norm
        self.X_val = X_val
        self.y_val = y_val_norm

    def predict_norm(self, feature_vector):
        Xs = self.scaler.transform(feature_vector.reshape(1, -1))
        pred = self.model.predict(Xs)[0]
        return float(pred[0]), float(pred[1])

    def add_sample_and_refit(self, feature_vector, x_norm, y_norm):
        self.X_train = np.vstack([self.X_train, feature_vector.reshape(1, -1)])
        self.y_train = np.vstack([self.y_train, np.array([[x_norm, y_norm]])])
        Xs = self.scaler.transform(self.X_train)
        self.model.fit(Xs, self.y_train)


def _eval_region_breakdown(pred_norm, y_norm, screen_w, screen_h):
    err_px = np.linalg.norm((pred_norm - y_norm) * [screen_w, screen_h], axis=1)
    regions = [region_label(x, y) for x, y in y_norm]
    by_region = {}
    for r, e in zip(regions, err_px):
        by_region.setdefault(r, []).append(e)
    return err_px, by_region


def train_and_select_model(X_train, y_train_px, X_val, y_val_px, screen_w, screen_h, seeds=(0, 1)):
    y_train = y_train_px / [screen_w, screen_h]
    y_val = y_val_px / [screen_w, screen_h]

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    def build(name, seed):
        if name == "Ridge":
            return Ridge(alpha=5.0)
        if name == "RandomForest":
            return RandomForestRegressor(n_estimators=150, max_depth=12, random_state=seed)
        if name == "MLP":
            return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=seed)
        if name == "HistGB":
            # HistGradientBoostingRegressor doesn't support multi-output natively
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=6, random_state=seed))
        raise ValueError(name)

    print("\n--- Model comparison (held-out validation points, never trained on) ---")
    results = {}
    for name in ["Ridge", "RandomForest", "MLP", "HistGB"]:
        use_seeds = seeds if name in ("RandomForest", "MLP", "HistGB") else (0,)
        maes = []
        fitted = None
        for s in use_seeds:
            model = build(name, s)
            model.fit(Xtr, y_train)
            pred = model.predict(Xv)
            err_px, by_region = _eval_region_breakdown(pred, y_val, screen_w, screen_h)
            maes.append(float(np.mean(err_px)))
            if fitted is None:
                fitted = model
                fitted_err_px, fitted_regions = err_px, by_region

        mean_mae = float(np.mean(maes))
        rmse_px = float(np.sqrt(np.mean(fitted_err_px ** 2)))
        p95_px = float(np.percentile(fitted_err_px, 95))
        deg = px_to_degrees(mean_mae, screen_w)
        results[name] = (fitted, mean_mae)

        seed_note = f" (avg over {len(use_seeds)} seeds)" if len(use_seeds) > 1 else ""
        print(f"  {name:12s} MAE: {mean_mae:6.1f}px{seed_note}   RMSE: {rmse_px:6.1f}px   "
              f"95th pct: {p95_px:6.1f}px   (~{deg:.2f} deg)")
        region_str = "  ".join(f"{r}:{np.mean(v):.0f}px" for r, v in sorted(fitted_regions.items()))
        print(f"    by region -> {region_str}")

    best_name = min(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"Selected model: {best_name}\n")

    return GazeModel(scaler, best_name, best_model, X_train, y_train, X_val, y_val)


# ============================================
# PERSISTENCE
# ============================================

CALIBRATION_FILE = "calibration_model.pkl"


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    try:
        data = joblib.load(CALIBRATION_FILE)
        n_saved = data["X_train"].shape[1]
        if n_saved != N_FEATURES:
            print(f"Saved calibration was trained with {n_saved} features but the current "
                  f"code extracts {N_FEATURES} (feature set changed since last calibration) — "
                  f"ignoring old file, will recalibrate.")
            return None
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
# CSV LOGGING (item 30) — consumed offline by offline_eval.py
# ============================================

CSV_HEADER = list(FEATURE_NAMES) + ["target_x_px", "target_y_px", "is_val", "confidence", "timestamp"]


def open_calibration_log():
    fname = f"calibration_log_{int(time.time())}.csv"
    f = open(fname, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    return f, writer, fname


def log_calibration_sample(writer, feature_vector, target_x_px, target_y_px, is_val, confidence, t):
    writer.writerow(list(feature_vector) + [target_x_px, target_y_px, int(is_val), confidence, t])
