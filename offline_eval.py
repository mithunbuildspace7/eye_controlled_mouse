"""
offline_eval.py — reload a calibration_log_*.csv (written by eye_gaze_cursor.py
during calibration) and re-run model training/comparison without a webcam.

Usage:
    python offline_eval.py calibration_log_1234567890.csv
    python offline_eval.py calibration_log_1234567890.csv 1920 1080   (override screen size)

Limitation: this can only re-test what was already computed into the logged
FEATURE_NAMES columns (different models, regularization, confidence/velocity
thresholds applied post-hoc, different train/val splits). It can't test a
totally different feature-extraction scheme without re-collecting data —
that needs a live camera.
"""
import sys
import csv
import numpy as np
import pyautogui

import gaze_lib as gl


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_feat = gl.N_FEATURES
    X, y, is_val, conf = [], [], [], []
    for row in rows:
        X.append([float(row[name]) for name in gl.FEATURE_NAMES])
        y.append([float(row["target_x_px"]), float(row["target_y_px"])])
        is_val.append(int(row["is_val"]))
        conf.append(float(row["confidence"]))

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    is_val = np.array(is_val, dtype=bool)
    assert X.shape[1] == n_feat, f"CSV has {X.shape[1]} feature columns, expected {n_feat} — stale log format?"
    return X, y, is_val


def main():
    if len(sys.argv) < 2:
        print("Usage: python offline_eval.py <calibration_log.csv> [screen_w screen_h]")
        sys.exit(1)

    path = sys.argv[1]
    if len(sys.argv) >= 4:
        screen_w, screen_h = int(sys.argv[2]), int(sys.argv[3])
    else:
        screen_w, screen_h = pyautogui.size()
        print(f"No screen size given, using this machine's: {screen_w}x{screen_h} "
              f"(pass explicit values if this CSV was recorded on a different monitor)")

    X, y, is_val = load_csv(path)
    print(f"Loaded {len(X)} samples ({is_val.sum()} validation, {(~is_val).sum()} training) from {path}")

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    gl.train_and_select_model(X_train, y_train, X_val, y_val, screen_w, screen_h)


if __name__ == "__main__":
    main()
