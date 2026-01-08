# AprilTagMVP – AprilTag Detection for Kinova Gen3

A toolkit for detecting AprilTags and computing pose coordinates for the Kinova Gen3 robotic arm.

**Two Viewers:**

- `offline_apriltag_viewer.py` – Works with any webcam (no robot required)
- `apriltag_viewer.py` – Full integration with Kinova Gen3 arm

---

## 📦 Requirements

### Python

- **Python 3.11** (required)
- ❗ Python 3.12 will NOT work with Kortex API

### Libraries

```
opencv-python
numpy
pupil-apriltags
```

### Kinova Kortex SDK (for robot viewer only)

Download: `kortex_api-2.2.0.post31-py3-none-any.whl`

---

## 🧰 Installation

```bash
# Install dependencies
py -3.11 -m pip install opencv-python numpy pupil-apriltags

# Install Kortex API (for robot viewer only)
py -3.11 -m pip install kortex_api-2.2.0.post31-py3-none-any.whl
```

---

## 🎥 Offline AprilTag Viewer (No Robot)

Use this to test AprilTag detection with any webcam outside the lab.

### Run

```bash
py -3.11 -m robot.offline_apriltag_viewer
```

### Options

| Argument       | Description           | Default        |
| -------------- | --------------------- | -------------- |
| `--camera`     | Camera index          | `0`            |
| `--width`      | Capture width         | `1280`         |
| `--height`     | Capture height        | `720`          |
| `--tag-size`   | Tag size in meters    | `0.05` (5 cm)  |
| `--tag-family` | AprilTag family       | `tag36h11`     |
| `--fx`, `--fy` | Focal length (pixels) | Auto-estimated |
| `--cx`, `--cy` | Principal point       | Image center   |

### Example

```bash
# Use second camera with 10cm tag
py -3.11 -m robot.offline_apriltag_viewer --camera 1 --tag-size 0.10
```

### Controls

| Key | Action |
| --- | ------ |
| `Q` | Quit   |

---

## 🤖 Kinova AprilTag Viewer (Robot Required)

Full integration with Kinova Gen3 arm. Computes tag position in both camera frame and robot base frame.

### Run

```bash
py -3.11 -m robot.apriltag_viewer --ip <ROBOT_IP> -u <USERNAME> -p <PASSWORD>
```

**All connection arguments are required** (no defaults for security).

### Options

| Argument                     | Description                             | Required |
| ---------------------------- | --------------------------------------- | -------- |
| `--ip`                       | Robot IP address                        | ✅ Yes   |
| `-u`, `--username`           | Robot username                          | ✅ Yes   |
| `-p`, `--password`           | Robot password                          | ✅ Yes   |
| `--log`                      | Print coordinates to console each frame | No       |
| `--log-file`                 | Save coordinates to CSV file            | No       |
| `--gt-x`, `--gt-y`, `--gt-z` | Ground truth position for validation    | No       |

### Example

```bash
# Basic usage
py -3.11 -m robot.apriltag_viewer --ip 192.168.1.10 -u MyUser -p MyPass

# With CSV logging
py -3.11 -m robot.apriltag_viewer --ip 192.168.1.10 -u MyUser -p MyPass --log-file session.csv
```

### Controls

| Key | Action                                   |
| --- | ---------------------------------------- |
| `Q` | Quit                                     |
| `S` | Print current tag coordinates to console |

### Output

The viewer displays:

- **CAM→TAG**: Position relative to camera (meters)
- **BASE→TAG**: Position relative to robot base (meters) ← _Use for arm motion planning_
- **Orientation**: Roll, pitch, yaw (degrees)

### CSV Log Format

```
timestamp,tag_id,cam_x,cam_y,cam_z,cam_dist,base_x,base_y,base_z,base_dist,roll,pitch,yaw
```

---

## 📁 Project Structure

```
AprilTagMVP/
├── robot/
│   ├── apriltag_viewer.py        # Kinova robot viewer
│   ├── offline_apriltag_viewer.py # Standalone webcam viewer
│   └── device_connection.py       # Kortex API connection
├── common/
│   ├── utils.py                   # Detection & transformation utilities
│   ├── webcam_config.py           # Camera intrinsics & RTSP config
│   └── tool_cam_config.py         # Tool→Camera extrinsics
└── README.md
```
