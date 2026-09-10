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
from collections import deque

# ============================================
# CONFIG
# ============================================

CAMERA_RESOLUTIONS = [(1280, 720), (960, 540), (640, 480)]
DETECTION_SCALE = 0.6          # downscale frame before landmark detection (speed boost)

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

CALIBRATION_FILE = "calibration_head.npy"

# --- One Euro Filter tuning ---
ONEEURO_MIN_CUTOFF = 0.1
ONEEURO_BETA = 0.5
ONEEURO_DCUTOFF = 1.0

# --- Blink click ---
BLINK_EAR_THRESHOLD = 0.19
BLINK_HOLD_SECONDS = 0.2 # It can 0.4
CLICK_COOLDOWN = 0.8

# --- Mouth-open mode toggle (MOVE <-> SCROLL) ---
MOUTH_OPEN_THRESHOLD = 0.55
MODE_TOGGLE_HOLD_SECONDS = 0.2 # You can change it 0.5
MODE_TOGGLE_COOLDOWN = 1.0

# --- Scroll (active only in SCROLL mode) ---
SCROLL_DEAD_ZONE = 0.015        # normalized head-y dead zone around baseline
SCROLL_SENSITIVITY = 45         # scroll units per normalized offset
SCROLL_MAX_OFFSET = 0.15

CALIB_SETTLE_TIME = 0.6
CALIB_SAMPLE_TIME = 1.0

# --- Stillness Lock (alternative to filtering: freezes cursor completely when head is steady) ---
STILLNESS_ENTER_RADIUS = 6       # px of raw movement allowed to still count as "not moving"
STILLNESS_EXIT_RADIUS = 14       # px of raw movement needed to break the lock (hysteresis)
STILLNESS_HOLD_TIME = 0.25       # seconds of steadiness required before locking

DEBUG_WINDOW_W, DEBUG_WINDOW_H = 480, 360
COUNTDOWN_SECONDS = 3

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ============================================
# LANDMARK INDICES
# ============================================

NOSE_TIP = 1
BETWEEN_EYEBROWS = 168

LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_VERT = (159, 145)

RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERT = (386, 374)

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


def get_head_point(landmarks):
    nose = landmarks[NOSE_TIP]
    brow = landmarks[BETWEEN_EYEBROWS]
    return (nose.x + brow.x) / 2, (nose.y + brow.y) / 2


def get_landmarks(frame):
    small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, get_timestamp_ms())

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================
# CALIBRATION
# ============================================

def run_calibration():
    print("Starting calibration. Turn/move your HEAD to point at each red dot.")

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    margin = 0.08
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]

    screen_points = []
    head_points = []

    for gy in ys:
        for gx in xs:
            target_x = int(gx * SCREEN_W)
            target_y = int(gy * SCREEN_H)

            settle_start = time.time()
            samples = []

            while True:
                success, frame = camera.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                elapsed = time.time() - settle_start

                cv2.circle(canvas, (target_x, target_y), 22, (0, 0, 255), -1)
                cv2.circle(canvas, (target_x, target_y), 22, (255, 255, 255), 2)

                if elapsed > CALIB_SETTLE_TIME:
                    landmarks = get_landmarks(frame)
                    if landmarks is not None:
                        samples.append(get_head_point(landmarks))
                    cv2.circle(canvas, (target_x, target_y), 10, (0, 255, 0), -1)

                cv2.putText(canvas, "Point your HEAD at the dot - ESC to cancel", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Calibration", canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    cv2.destroyWindow("Calibration")
                    return None, None

                if elapsed > CALIB_SETTLE_TIME + CALIB_SAMPLE_TIME:
                    break

            if len(samples) == 0:
                print("Warning: no face detected for a calibration point, skipping.")
                continue

            head_points.append(np.mean(samples, axis=0))
            screen_points.append((target_x, target_y))

    cv2.destroyWindow("Calibration")

    if len(head_points) < 4:
        print("Calibration failed: not enough valid points.")
        return None, None

    src = np.array(head_points, dtype=np.float32)
    dst = np.array(screen_points, dtype=np.float32)

    homography, _ = cv2.findHomography(src, dst, method=0)

    # --- calibration quality check (reprojection error) ---
    reproj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(reproj - dst, axis=1)
    mean_err = float(np.mean(errors))
    screen_diag = math.hypot(SCREEN_W, SCREEN_H)
    err_ratio = mean_err / screen_diag

    if err_ratio < 0.01:
        quality = "Good"
    elif err_ratio < 0.025:
        quality = "Fair"
    else:
        quality = "Poor - consider recalibrating (R) with steadier head positioning"

    print(f"Calibration complete. Mean reprojection error: {mean_err:.1f}px -> Quality: {quality}")
    return homography, quality


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        return np.load(CALIBRATION_FILE)
    return None


def save_calibration(h):
    np.save(CALIBRATION_FILE, h)


# ============================================
# HUD / VISUAL POLISH
# ============================================

def draw_hud(frame, mode, fps, status_text, status_color, click_hold_ratio, calib_quality, paused):
    h, w = frame.shape[:2]

    # =========================
    # TOP LEFT : MODE
    # =========================
    mode_color = (0, 200, 255) if mode == "SCROLL" else (0, 255, 120)

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    # =========================
    # TOP RIGHT : STATUS
    # =========================
    status_size = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        2
    )[0]

    status_x = w - status_size[0] - 12

    cv2.putText(
        frame,
        status_text,
        (status_x, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # =========================
    # CLICK HOLD BAR
    # =========================
    bar_w = 180
    bar_h = 10

    bar_x = (w - bar_w) // 2
    bar_y = 42

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (180, 180, 180),
        1,
    )

    fill = int(bar_w * np.clip(click_hold_ratio, 0, 1))

    if fill > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill, bar_y + bar_h),
            (0, 255, 0),
            -1,
        )

    # =========================
    # BOTTOM
    # =========================
    cv2.putText(
        frame,
        "ESC quit | mouth-open : toggle scroll",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

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
            cv2.putText(frame, "Get ready - tracking starts soon", (40, DEBUG_WINDOW_H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Head Cursor - Project", frame)
            cv2.waitKey(1)
            time.sleep(0.03)


# ============================================
# MAIN
# ============================================

homography = load_calibration()
calib_quality = None

if homography is None:
    homography, calib_quality = run_calibration()
    if homography is None:
        print("No calibration available. Exiting.")
        camera.stop()
        raise SystemExit(0)
    save_calibration(homography)

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

paused = False
frame_times = deque(maxlen=30)

cv2.namedWindow("Head Cursor - Project", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Head Cursor - Project", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
win_x = max(SCREEN_W - DEBUG_WINDOW_W - 30, 0)
win_y = 30
cv2.moveWindow("Head Cursor - Project", win_x, win_y)

print("Head cursor running. Press ESC to quit, R to recalibrate, P to pause.")
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
            head_point = get_head_point(landmarks)

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
                        scroll_baseline_y = head_point[1]
                    else:
                        filter_x.reset()
                        filter_y.reset()
                        reset_stillness()
            else:
                mouth_open_start = None

            if mode == "SCROLL":
                # ---- Scroll mode: tilt head up/down, clicking disabled ----
                offset = head_point[1] - scroll_baseline_y
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
                # ---- MOVE mode: blink click + filtered cursor movement ----
                if is_closed:
                    if blink_start_time is None:
                        blink_start_time = t
                    held = t - blink_start_time
                    click_hold_ratio = held / BLINK_HOLD_SECONDS

                    if held >= BLINK_HOLD_SECONDS and not eyes_closed:
                        if t - last_click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = t
                            click_message_until = t + 0.50      # Show CLICK! for 300 ms
                            eyes_closed = True
                else:
                    blink_start_time = None
                    eyes_closed = False

                if not is_closed:
                    point = np.array([[head_point]], dtype=np.float32)
                    mapped = cv2.perspectiveTransform(point, homography)[0][0]

                    raw_x = float(np.clip(mapped[0], 0, SCREEN_W - 1))
                    raw_y = float(np.clip(mapped[1], 0, SCREEN_H - 1))

                    # ---- Stillness Lock: freeze cursor completely when head is steady ----
                    newly_locked = False

                    if not stillness_locked:
                        if still_ref_point is None:
                            still_ref_point = (raw_x, raw_y)
                            still_ref_time = t
                        else:
                            d_ref = math.hypot(raw_x - still_ref_point[0], raw_y - still_ref_point[1])
                            if d_ref > STILLNESS_ENTER_RADIUS:
                                # still moving meaningfully -> restart the steadiness timer here
                                still_ref_point = (raw_x, raw_y)
                                still_ref_time = t
                            elif t - still_ref_time >= STILLNESS_HOLD_TIME:
                                stillness_locked = True
                                lock_anchor = still_ref_point
                                newly_locked = True

                    if stillness_locked:
                        d_lock = math.hypot(raw_x - lock_anchor[0], raw_y - lock_anchor[1])
                        if d_lock > STILLNESS_EXIT_RADIUS:
                            # deliberate movement detected -> break the lock
                            stillness_locked = False
                            still_ref_point = (raw_x, raw_y)
                            still_ref_time = t
                            filter_x.reset()
                            filter_y.reset()
                        else:
                            if newly_locked:
                                pyautogui.moveTo(int(lock_anchor[0]), int(lock_anchor[1]))
                            if status_text != "CLICK!":
                                status_text = "Locked (still)"
                                status_color = (150, 255, 150)

                    if not stillness_locked:
                        fx = filter_x(raw_x, t)
                        fy = filter_y(raw_y, t)
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

    # ---- FPS ----
    frame_times.append(time.time() - loop_start)
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 1
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    display_frame = cv2.resize(frame, (DEBUG_WINDOW_W, DEBUG_WINDOW_H))
    display_frame = draw_hud(display_frame, mode, fps, status_text, status_color,
                              click_hold_ratio, calib_quality, paused)

    cv2.imshow("Head Cursor - Project", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord("r"):
        new_h, new_quality = run_calibration()
        if new_h is not None:
            homography = new_h
            calib_quality = new_quality
            save_calibration(homography)
            filter_x.reset()
            filter_y.reset()
            reset_stillness()
    elif key == ord("p"):
        paused = not paused
        if paused:
            blink_start_time = None
            eyes_closed = False
        else:
            filter_x.reset()
            filter_y.reset()
            reset_stillness()

camera.stop()
cv2.destroyAllWindows()
