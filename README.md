# 🟦 APRILTAGMVP — Calibration Branch  
### AprilTag-Based Camera ↔ Robot Calibration for Kinova Gen3 / Gen3-N

This branch is **exclusively dedicated to calibration** of the **Kinova K3 camera mounted on the end-effector**, using AprilTags to estimate accurate spatial transforms between:

- **Robot Base → End-Effector (live from Kortex)**
- **End-Effector → Camera (fixed extrinsics)**
- **Camera → AprilTag (vision-based pose)**
- **Robot Base → AprilTag (composed result)**

It is **not** intended for task execution, motion planning, or runtime perception — only **data collection, validation, and calibration**.

---

## ✅ What This Branch Provides

✔ Offline AprilTag viewer (no robot required)  
✔ Robot-integrated calibration capture tool  
✔ Live Base → Tag pose visualization  
✔ Clean modular structure for reuse in downstream projects  
✔ JSON-based calibration dataset output  

---

## 📦 1. Requirements

### Python
- **Python 3.11**
- ❗ **Python 3.12 is NOT supported** (Kortex protobuf incompatibility)

Verify:
```bash
py -3.11 --version
```

---

### Libraries

Install via `pip` (Python 3.11):

- `opencv-python`
- `numpy`
- `pupil-apriltags`

### Kinova Kortex SDK
Download:
```
kortex_api-2.2.0.post31-py3-none-any.whl
```

Install:
```bash
py -3.11 -m pip install kortex_api-2.2.0.post31-py3-none-any.whl
```

---

## 📥 2. Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/AprilTagMVP.git
cd AprilTagMVP
```

Ensure you are on the **calibration branch**:
```bash
git checkout calibration
```

---

## 🧰 3. Install Remaining Dependencies

```bash
py -3.11 -m pip install opencv-python numpy pupil-apriltags
```

---

## 🎥 4. Offline AprilTag Viewer (NO ROBOT REQUIRED)

This is a **standalone sanity check** for:
- AprilTag detection
- Camera intrinsics
- Pose stability

### Run (webcam):
```bash
py -3.11 -m robot.offline_apriltag_viewer --webcam 0
```

### Run (video file):
```bash
py -3.11 -m robot.offline_apriltag_viewer --video path/to/test.mp4
```

If this works, your **vision stack is correctly configured**.

---

## 🤖 5. Robot Calibration Capture Tool (LAB USE)

This is the **core purpose of this branch**.

The script:
- Connects to the Kinova robot using Kortex API
- Streams live **Base → Tool** pose
- Detects AprilTags from the **K3 camera**
- Computes **Base → Tag** pose
- Saves synchronized samples on command

### Run:
```bash
py -3.11 -m robot.apriltag_calibration --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

---

## 🎮 Controls

| Key | Action |
|----|-------|
| SPACE | Save calibration sample |
| Q | Quit program |

---

## 📂 Output

Calibration samples are written to:

```
calibration_samples.json
```

---

## 📁 Repository Structure

```
AprilTagMVP/
├── robot/
│   ├── apriltag_calibration.py
│   ├── apriltag_viewer.py
│   ├── offline_apriltag_viewer.py
│   └── device_connection.py
│
├── common/
│   ├── utils.py
│   ├── webcam_config.py
│   └── tool_cam_config.py
│
├── calibration_samples.json
└── README.md
```

---

## 🚧 Scope Disclaimer

This branch is intentionally **narrow in scope**.

Its sole purpose is:
> **Reliable geometric calibration between the Kinova robot base and the camera-mounted AprilTag frame.**