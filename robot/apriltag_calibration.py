# robot/apriltag_calibration.py

import cv2
import numpy as np
from collections import deque
import json

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from robot.device_connection import DeviceConnection, parse_connection_arguments
from common.apriltag_utils import create_detector, rotation_to_euler_xyz
from common.webcam_config import (
    rtsp_url,
    tag_size,
    tag_family,
    camera_params,  
)


# ================================================================
# GLOBAL BASE CLIENT (Kinova)
# ================================================================
base = None

POSE_SCALE = 1.6730539   

# ================================================================
# APRILTAG CAMERA PARAMS WITH EMPIRICAL Z-SCALE FIX
# ================================================================

fx_calib, fy_calib, cx, cy = camera_params

SCALE_Z = 0.30 / 0.225  

fx_tag = fx_calib * SCALE_Z
fy_tag = fy_calib * SCALE_Z

camera_params_tag = (fx_tag, fy_tag, cx, cy)

print("[INFO] Calibrated intrinsics (checkerboard):")
print(f"  fx = {fx_calib:.3f}, fy = {fy_calib:.3f}, cx = {cx:.3f}, cy = {cy:.3f}")
print("[INFO] Empirical Z scale factor:", SCALE_Z)
print("[INFO] AprilTag intrinsics used for pose:")
print(f"  fx_tag = {fx_tag:.3f}, fy_tag = {fy_tag:.3f}, cx = {cx:.3f}, cy = {cy:.3f}")


# ================================================================
# EULER → ROTATION MATRIX
# ================================================================

def euler_xyz_to_R(rx, ry, rz):
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,  0, 1]])

    return Rz @ Ry @ Rx


# ================================================================
# GET CURRENT END-EFFECTOR POSE
# ================================================================

def get_robot_pose():
    global base
    if base is None:
        raise RuntimeError("BaseClient (base) is not initialized")

    pose = base.GetMeasuredCartesianPose()

    x = float(pose.x)
    y = float(pose.y)
    z = float(pose.z)

    rx = np.deg2rad(pose.theta_x)
    ry = np.deg2rad(pose.theta_y)
    rz = np.deg2rad(pose.theta_z)

    R = euler_xyz_to_R(rx, ry, rz)
    t = np.array([x, y, z], dtype=float)

    return R, t


def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x


# ================================================================
# MAIN CALIBRATION SCRIPT
# ================================================================

def main():
    global base

    # -----------------------------------------
    # CONNECT TO ROBOT
    # -----------------------------------------

    args = parse_connection_arguments()
    with DeviceConnection.create_tcp_connection(args) as router:
        base = BaseClient(router)
        print("\n[✓] Connected to robot. Starting AprilTag calibration...\n")

        calibration_samples = []

        # -----------------------------------------
        # OPEN ROBOT CAMERA STREAM (RTSP)
        # -----------------------------------------

        ip = args.ip if hasattr(args, "ip") else "192.168.1.10"
        url = rtsp_url(ip)

        print(f"Opening camera: {url}")
        cap = cv2.VideoCapture(url)

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[INFO] RTSP stream resolution: {width} × {height}")

        if not cap.isOpened():
            print(f"[!] ERROR: Could not open RTSP stream: {url}")
            return

        detector = create_detector()

        yaw_hist   = deque(maxlen=10)
        pitch_hist = deque(maxlen=10)
        roll_hist  = deque(maxlen=10)

        latest_R_cam_tag = None
        latest_t_cam_tag = None

        # ============================================================
        # MAIN LOOP
        # ============================================================

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---------------------------
            # APRILTAG DETECTION
            # ---------------------------
            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params_tag,  
                tag_size=tag_size,
            )

            latest_R_cam_tag = None
            latest_t_cam_tag = None

            for r in results:
                pts = np.array([r.corners], dtype=np.int32)
                cv2.polylines(frame, pts, True, (0, 255, 0), 2)

                ptA = r.corners[0]
                cv2.putText(frame, f"ID: {r.tag_id}",
                            (int(ptA[0]), int(ptA[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                t_raw = r.pose_t     
                R = r.pose_R

                t = t_raw * POSE_SCALE

                latest_R_cam_tag = R
                latest_t_cam_tag = t[:, 0]

                yaw, pitch, roll = rotation_to_euler_xyz(R)

                yaw_hist.append(np.degrees(yaw))
                pitch_hist.append(np.degrees(pitch))
                roll_hist.append(np.degrees(roll))

                avg_yaw = np.mean(yaw_hist)
                avg_pitch = np.mean(pitch_hist)
                avg_roll = np.mean(roll_hist)

                cv2.rectangle(frame, (5, 5), (530, 110), (0, 0, 0), -1)
                cv2.putText(frame,
                            f"Tag Pos: x={t[0][0]:.3f} y={t[1][0]:.3f} z={t[2][0]:.3f} m",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Tag Rot: yaw={avg_pitch:.0f} pitch={avg_yaw:.0f} roll={avg_roll:.0f}",
                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)


            # Display
            cv2.imshow("AprilTag Calibration (Robot Camera)", frame)

            key = cv2.waitKey(1) & 0xFF

            # Save sample on SPACE
            if key == ord(" "):
                if latest_R_cam_tag is None:
                    print("[!] No tag detected — cannot save sample.")
                else:
                    R_base_tool, t_base_tool = get_robot_pose()
                    sample = {
                        "R_base_tool": to_list(R_base_tool),
                        "t_base_tool": to_list(t_base_tool),
                        "R_cam_tag":   to_list(latest_R_cam_tag),
                        "t_cam_tag":   to_list(latest_t_cam_tag),
                    }
                    calibration_samples.append(sample)
                    print(f"[✓] Saved calibration sample #{len(calibration_samples)}")

            # Quit on 'q'
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save collected samples
        if calibration_samples:
            with open("calibration_samples.json", "w") as f:
                json.dump(calibration_samples, f, indent=4)
            print(
                f"\n[✓] Saved {len(calibration_samples)} calibration "
                "samples to calibration_samples.json\n"
            )
        else:
            print("\n[!] No calibration samples collected.\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
