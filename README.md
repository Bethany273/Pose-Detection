# Pose Detection (MediaPipe + OpenCV)

This small project demonstrates using the MediaPipe Pose Landmarker (Tasks API) with OpenCV.

Quick steps to get running:

1. Use Python 3.11 (recommended). If you have Python 3.13 currently, install Python 3.11 and create a virtual environment.

2. Install dependencies:

   python -m pip install -r requirements.txt

3. Place your MediaPipe Pose Landmarker model file named `pose_landmarker.task` next to `pose detection.py`, or update the `model_path` variable in the script.

4. Place a test image named `pose.jpg` next to the script (or update the script).

5. Run:

   python "pose detection.py"

Notes:
- If you see "No matching distribution found for mediapipe" when installing, your Python version is likely unsupported by prebuilt MediaPipe wheels. Use Python 3.11 or 3.10.
- On Windows, prefer the official CPython installer and create a venv for reproducibility.
