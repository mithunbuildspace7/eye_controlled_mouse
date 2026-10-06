# Eye-Gaze Cursor

Control your mouse cursor using only your eyes, via a regular webcam — no head movement required, no special hardware. Blink to click, open your mouth to toggle scroll mode, and a built-in diagnostic screen tells you exactly how accurate your setup is, region by region.

This isn't a simple "iris-ratio-to-screen" hack — it's a personalized machine-learning pipeline: it extracts ~30 features per frame from both eyes and your head pose, trains and automatically compares four different regression models on your own calibration data, and filters the result through a confidence-gated Kalman filter.

---

## Features

- **Eyes-only cursor control** — no head tracking, no extra hardware beyond a webcam
- **Blink-to-click**, with a cooldown to avoid accidental double clicks
- **Mouth-open gesture** to toggle between cursor-move mode and scroll mode
- **Personalized calibration** — 66 calibration points (7×7 grid + validation reps + dedicated corner/edge anchors), trained specifically to your face, camera, and sitting position
- **Automatic model selection** — Ridge, Random Forest, MLP, and HistGradientBoosting are all trained and compared; whichever generalizes best to data it's never seen wins
- **Dual pupil detection** — MediaPipe's iris landmarks *and* an independent classical computer-vision pupil detector (thresholding + contours), fed to the model together
- **Head-pose compensation** via `solvePnP` (yaw/pitch/roll, approximate head position)
- **Confidence-gated Kalman filtering** — smooths jitter without introducing noticeable lag, and freezes rather than guesses when confidence is low (blinking, poor detection)
- **Built-in diagnostic test** (`D` key) — live raw-vs-filtered-vs-target overlay on points never used in calibration, with a center/edge/corner accuracy breakdown
- **Offline re-evaluation tool** — re-run model comparisons on previously logged calibration data without touching the camera again
- **Online drift correction** (`C` key) — add a correction sample mid-session if you notice the cursor drifting

---

## How it works (short version)

```
Webcam
  → MediaPipe Face Landmarker (478 landmarks incl. iris)
  → Feature extraction (both eyes' local-axis gaze ratio, eye shape,
     pixel-based pupil position, head yaw/pitch/roll, face position,
     confidence — 30 features total)
  → Regression model (Ridge / Random Forest / MLP / HistGradientBoosting
     — best one auto-selected via held-out validation)
  → Kalman filter (confidence-gated, time-aware)
  → Runtime outlier rejection + speed cap
  → Cursor movement
```

Calibration isn't a single geometric formula — it's real supervised learning. You look at a grid of dots, the system logs (your eye/head features) → (the exact screen point you were looking at) for each one, trains several models on that data, and keeps whichever one is most accurate on dots it was *not* trained on.

For the full technical explanation (every algorithm, why each one was chosen, version history), see the project write-up — ask for it if it's not included in this repo.

---

## Requirements

- **Python 3.9 – 3.12** (MediaPipe does not currently support newer versions)
- A webcam
- Packages listed in `requirements.txt`:

```
opencv-python>=4.8
mediapipe>=0.10
numpy>=1.24
pyautogui>=0.9.54
scikit-learn>=1.3
joblib>=1.3
```

---

## Installation

```bash
git clone <this-repo-url>
cd eye-gaze-cursor
pip install -r requirements.txt
```

Verify everything installed correctly:

```bash
python -c "
import importlib
for pkg in ['cv2','mediapipe','numpy','pyautogui','sklearn','joblib']:
    try:
        importlib.import_module(pkg)
        print(pkg, 'OK')
    except ImportError:
        print(pkg, 'MISSING')
"
```

---

## Usage

```bash
python eye_gaze_cursor.py
```

On first run:
1. The MediaPipe face-landmark model (`face_landmarker.task`, a few MB) downloads automatically — needs internet the first time only.
2. A 3-second countdown gives you time to get in position.
3. **Calibration starts automatically** — look at each red dot as it appears (66 total: a 7×7 grid, 9 of those points get a repeat pass with your head shifted slightly, plus 8 dedicated anchors right at the screen edges/corners). Try not to blink while a dot is active. Takes roughly 2 minutes.
4. After calibration, a **verification screen** shows your live tracked point so you can visually confirm accuracy before accepting — `SPACE` to accept, `R` to redo, `ESC` to cancel.
5. Once accepted, the cursor goes live.

Calibration is saved (`calibration_model.pkl`) and reused automatically on future runs — you won't need to recalibrate every time unless you press `R`, your setup changes significantly, or the file is deleted.

### Controls (while running)

| Key / Action | Effect |
|---|---|
| Look around | Moves the cursor |
| Blink and hold briefly | Click |
| Open your mouth briefly | Toggle MOVE ↔ SCROLL mode (in scroll mode, look up/down to scroll) |
| `R` | Recalibrate |
| `P` | Pause / resume tracking |
| `C` | Add an online correction sample (only use this while looking exactly at the current cursor position) |
| `D` | Run the diagnostic accuracy test |
| `ESC` | Quit |

### Diagnostic test (`D`)

Shows 11 points never used in calibration, overlays your raw prediction, filtered prediction, and the actual target in real time, then prints a full accuracy report broken down by screen region (center, edges, corners) — both to the console and to a `diagnostic_log_*.csv` file. This is the honest way to check how accurate your setup really is.

### Offline re-evaluation

Every calibration run also saves a `calibration_log_*.csv` of every raw sample collected. You can re-run model comparison on that data later, without a camera:

```bash
python offline_eval.py calibration_log_<timestamp>.csv
```

---

## Project structure

| File | Role |
|---|---|
| `eye_gaze_cursor.py` | The app you run — camera, calibration flow, diagnostic screen, live cursor loop |
| `gaze_lib.py` | The engine — feature extraction, Kalman filter, model training/comparison, calibration save/load. No camera code; pure logic, imported by the other two files |
| `offline_eval.py` | Standalone tool to re-run model comparison on a saved calibration CSV, no camera needed |
| `requirements.txt` | Python dependencies |
| `face_landmarker.task` | MediaPipe's face-landmark model — auto-downloaded on first run, not tracked in git |
| `calibration_model.pkl` | Your personal trained calibration (generated locally, not tracked in git) |

---

## Known limitations

- No true webcam intrinsic calibration — head pose uses an approximate generic face/camera model, not this exact camera's measured optical properties.
- Corner/edge accuracy is inherently harder than center accuracy (partial eyelid occlusion at extreme gaze angles, natural perspective differences between the two eyes).
- Calibration is personal and setup-specific — a meaningfully different sitting distance, camera position, or lighting setup will need recalibration.
- The pixel-based pupil detector uses classical thresholding, not a trained detector, so it can occasionally be fooled by eyelashes, shadows, or glare — it's used as a secondary signal alongside MediaPipe's landmarks, not a replacement.
- The `C` online-correction key only helps if you are genuinely looking exactly at the cursor when you press it; otherwise it can reinforce existing drift.

## Possible future improvements

- Real webcam intrinsic calibration (checkerboard capture) for more accurate head pose
- A trained pupil/iris detector instead of threshold-based detection
- A CNN trained on raw eye-image crops, fine-tuned on personal calibration data, if classical features prove insufficient

---

## Acknowledgments

Built on [MediaPipe](https://github.com/google-ai-edge/mediapipe) (face/iris landmark detection), [OpenCV](https://opencv.org/) (image processing, head pose, Kalman filtering), [scikit-learn](https://scikit-learn.org/) (regression models), and [PyAutoGUI](https://pyautogui.readthedocs.io/) (cursor control).
