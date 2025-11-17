# 🟦 APRILTAGMVP – AprilTag Calibration for Kinova Gen3 / Gen3-N

A modular toolkit for detecting AprilTags, streaming camera pose, and collecting calibration samples between the **camera mounted on the end-effector** and the **Kinova Kortex robot base**.

This repo provides:
- ✔ A standalone **AprilTag viewer** (webcam only)  
- ✔ A robot-integrated **calibration capture tool**  
- ✔ Clean modular folder structure  
- ✔ Utilities for pose, detection, and Kortex API connections  

---

## 📦 1. Requirements

### **Python**
- Python **3.11**  
❗ *Python 3.12 will NOT work with Kortex API (protobuf incompatibility).*

### **Libraries**
Install via `pip`:
- `opencv-python`
- `numpy`
- `pupil-apriltags`

### **Kinova Kortex SDK**
Download the Python Kortex SDK: [kortex_api-2.2.0.post31-py3-none-any.whl](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public%2Fkortex%2FAPI%2F2.2.0%2Fkortex_api-2.2.0.post31-py3-none-any.whl)

Place it in the project root.

---

## 📥 2. Clone the Repository

```sh
git clone https://github.com/<YOUR_USERNAME>/AprilTagMVP.git
cd AprilTagMVP
```

---

## 🧰 3. Install Dependencies

### Option A — Install using your system Python 3.11

```sh
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pip install kortex_api-2.2.0.post31-py3-none-any.whl
```

### Required packages:
- `opencv-python`
- `numpy`
- `pupil-apriltags`


## 🎥 4. Test AprilTag Detection (NO ROBOT REQUIRED)

Run the AprilTag webcam viewer:

```sh
py -3.11 -m local.apriltag_viewer
```

This will:
- Open your default webcam
- Detect AprilTags
- Display pose (x, y, z, yaw, pitch, roll)

If this works, your AprilTag + OpenCV setup is correct.

---


## 📌 5. Run the Full Calibration Tool (Robot + Camera)

This script:
- Connects to the robot using Kortex API
- Streams AprilTag pose
- Streams robot end-effector pose
- Saves samples when you press **SPACE**
- Writes results to `calibration_samples.json`

Run it:

```sh
py -3.11 -m robot/apriltag_calibration.py --ip <ROBOT_IP> -u <ROBOT_USERNAME> -p <ROBOT_PASSWORD>
```

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Save a calibration sample |
| **Q** | Quit program |

Calibration samples are saved to:
```
calibration_samples.json
```

---
