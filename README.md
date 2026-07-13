# 👁️ Eye Controlled Mouse

> Control your computer cursor using only your eyes and head movements with the power of AI and Computer Vision.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Tracking-orange?style=for-the-badge)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

## 📌 Overview

**Eye Controlled Mouse** is a real-time computer vision project that allows users to control the mouse cursor without touching a physical mouse.

Using **MediaPipe Face Landmarker**, **OpenCV**, and **PyAutoGUI**, the system tracks facial landmarks and maps eye/head movements to cursor movement.

The project is designed to provide a simple hands-free interaction experience while demonstrating the power of AI-based human-computer interaction.

---

## ✨ Features

- 👁️ Real-time eye/head tracking
- 🖱️ Hands-free mouse control
- 🎯 9-point calibration system
- ⚡ Smooth cursor movement using One Euro Filter
- 👀 Blink to Left Click
- 👄 Mouth Open to Toggle Scroll Mode
- 📜 Head-controlled scrolling
- 📷 Automatic MediaPipe model download

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- PyAutoGUI
- NumPy

---

# 📂 Project Structure

```
eye-controlled-mouse/
│
├── eye_cursor.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── face_landmarker.task
```

---

# 💻 Requirements

- Python 3.10+
- Webcam

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Tahsan-Habib/eye-controlled-mouse.git
```

Move into the project folder.

```bash
cd eye-controlled-mouse
```

---

## 2. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## 3. Run the project

```bash
python eye_cursor.py
```

The MediaPipe model will automatically download the first time you run the project (internet required only once).

---

# 🎮 Controls

| Key | Function |
|------|----------|
| **R** | Recalibrate |
| **P** | Pause / Resume Tracking |
| **ESC** | Exit |

---

# 🖱️ Mouse Controls

| Gesture | Action |
|----------|--------|
| Move Eyes / Head | Move Cursor |
| Blink (Hold) | Left Click |
| Open Mouth | Toggle Scroll Mode |
| Tilt Head Up/Down (Scroll Mode) | Scroll |

---

# 🎯 Calibration Guide

Calibration is the most important step.

When you launch the project for the first time, a calibration screen will appear.

## Follow these steps carefully:

1. Sit comfortably in front of your webcam.
2. Make sure your face is fully visible.
3. Look directly at each red calibration dot.
4. Move your eyes naturally while keeping your head as steady as possible.
5. Wait until each point is recorded.
6. Repeat for all calibration points.

After calibration, your cursor will move according to your recorded eye/head positions.

---

## ❗ If Cursor Tracking Isn't Accurate

No problem!

Simply press:

```
R
```

to recalibrate.

During recalibration:

- Look carefully at each red dot.
- Avoid sudden movements.
- Stay in a well-lit environment.
- Keep your webcam stable.
- Repeat the calibration if necessary.

Better calibration leads to significantly better cursor accuracy.

---

# 💡 Tips for Best Performance

✅ Use the project in a well-lit room.

✅ Sit around **40–70 cm** from your webcam.

✅ Avoid covering your face.

✅ Keep your webcam stable.

✅ Recalibrate whenever tracking becomes inaccurate.

---

# ⚙️ How It Works

1. Webcam captures your face.
2. MediaPipe detects facial landmarks.
3. Head position is calculated.
4. Homography maps head movement to screen coordinates.
5. One Euro Filter smooths the cursor.
6. Blink detection performs clicks.
7. Mouth opening switches between Move Mode and Scroll Mode.

---

# 🚀 Future Improvements

- Right Click
- Drag & Drop
- Double Click
- Custom Gestures
- Adjustable Sensitivity
- Multi-monitor Support
- Eye Typing
- Voice Commands
- Cross-platform Support

---

# 🤝 Contributing

Contributions are always welcome.

If you have ideas for improving the project, feel free to:

- Fork the repository
- Create a new branch
- Commit your changes
- Submit a Pull Request

---

# 🐛 Known Limitations

- Tested on Windows.
- Performance depends on webcam quality.
- Poor lighting may reduce tracking accuracy.

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Tahsan Habib**

If you enjoyed this project, consider giving it a ⭐ on GitHub. It helps others discover the project and motivates future improvements.

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork it

💬 Share your feedback

🚀 Happy Coding!