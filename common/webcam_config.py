# common/webcam_config.py

import numpy as np
import cv2
import json
import os

# ================================================================
#  Camera Configuration for Kinova Gen3 / RTSP Camera
#  Intrinsics from checkerboard calibration (9x7 squares, 2.2 cm)
# ================================================================

# ---- Intrinsic Parameters (from calibration script) ----
fx = 572.39097753
fy = 556.52636418
cx = 722.73029237
cy = 205.79086075

camera_params = (fx, fy, cx, cy)

# ---- Distortion Coefficients (k1, k2, p1, p2, k3) ----
dist_coeffs = np.array([
    0.00390371,
   -0.00449676,
   -0.00132384,
   -0.00363592,
   -0.00253139
])

# ---- AprilTag parameters ----
tag_size = 0.132         # meters (132 mm)
tag_family = "tag36h11"  # AprilTag family


# ---- Camera Matrix K ----
K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])


# ---- RTSP Camera Stream ----
DEFAULT_ROBOT_IP = "192.168.1.10"

def rtsp_url(ip: str = DEFAULT_ROBOT_IP) -> str:
    """
    Build the RTSP URL for the Kinova Gen3 color stream.
    """
    return f"rtsp://{ip}/color"


def undistort_frame(frame):
    """
    Optionally undistort a frame using the calibrated intrinsics.
    Right now the AprilTag pipeline is using the raw image with
    these intrinsics; so this is provided for other tools.

    If you start using this in the AprilTag pipeline, you *must*
    also use the corresponding newK intrinsics for pose.
    """
    h, w = frame.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)
    undistorted = cv2.undistort(frame, K, dist_coeffs, None, newK)
    return undistorted


CONFIG_FILE = "camera_intrinsics.json"

def save_json(path: str = CONFIG_FILE):
    """
    Save current intrinsics & distortion to JSON so other scripts
    can load them without editing code.
    """
    data = {
        "camera_matrix": K.tolist(),
        "distortion_coeffs": dist_coeffs.tolist(),
        "camera_params": camera_params,
        "tag_size": tag_size,
        "tag_family": tag_family,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[✓] Saved camera settings to {path}")


def load_json(path: str = CONFIG_FILE):
    """
    Load intrinsics & distortion from JSON. Only needed if you
    recalibrate later and want all tools to pick up the new values.
    """
    global K, dist_coeffs, camera_params, tag_size, tag_family

    if not os.path.exists(path):
        print(f"[!] JSON file '{path}' not found. Using built-in calibration.")
        return False

    with open(path, "r") as f:
        data = json.load(f)

    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"], dtype=float)
        fx_, fy_, cx_, cy_ = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        camera_params = (fx_, fy_, cx_, cy_)

    if "distortion_coeffs" in data:
        dist = data["distortion_coeffs"]
        dist_coeffs[:] = np.array(dist[0], dtype=float)

    if "tag_size" in data:
        tag_size = float(data["tag_size"])

    if "tag_family" in data:
        tag_family = data["tag_family"]

    print("[✓] Loaded calibration from JSON")
    return True


# =================================================================
#  Self-Test
# =================================================================

if __name__ == "__main__":
    print("Camera Matrix K:")
    print(K)

    print("\nDistortion Coefficients:")
    print(dist_coeffs)

    print("\nRTSP URL Example:")
    print(rtsp_url())
