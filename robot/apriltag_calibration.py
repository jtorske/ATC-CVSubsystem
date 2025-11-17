# robot/apriltag_calibration.py

import cv2
import numpy as np
from collections import deque
import json

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from robot.device_connection import DeviceConnection, parse_connection_arguments
from common.apriltag_utils import create_detector, detect_tags, rotation_to_euler_xyz

# Global BaseClient
base = None

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

def main():
    global base

    args = parse_connection_arguments()
    with DeviceConnection.create_tcp_connection(args) as router:
        base = BaseClient(router)
        print("BaseClient initialized, starting AprilTag calibration logger...")

        calibration_samples = []

        # Use RTSP camera on the arm if desired, or just 0 if webcam:
        cap = cv2.VideoCapture(0)

        detector = create_detector()

        yaw_hist, pitch_hist, roll_hist = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)

        latest_R_cam_tag = None
        latest_t_cam_tag = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = detect_tags(detector, gray)

            latest_R_cam_tag = None
            latest_t_cam_tag = None

            for r in results:
                (ptA, ptB, ptC, ptD) = r.corners
                pts = np.array([ptA, ptB, ptC, ptD], dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                cv2.putText(frame, f"ID: {r.tag_id}",
                            (int(ptA[0]), int(ptA[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                t = r.pose_t
                R = r.pose_R

                latest_R_cam_tag = R
                latest_t_cam_tag = t[:, 0]

                yaw, pitch, roll = rotation_to_euler_xyz(R)
                yaw_hist.append(np.degrees(yaw))
                pitch_hist.append(np.degrees(pitch))
                roll_hist.append(np.degrees(roll))

                avg_yaw = np.mean(yaw_hist)
                avg_pitch = np.mean(pitch_hist)
                avg_roll = np.mean(roll_hist)

                cv2.rectangle(frame, (5, 5), (500, 90), (0, 0, 0), -1)

                def draw_text(img, text, pos, color):
                    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 3)
                    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

                draw_text(frame,
                          f"Pos: x={t[0][0]:.2f} y={t[1][0]:.2f} z={t[2][0]:.2f} m",
                          (10, 35), (0, 255, 0))
                draw_text(frame,
                          f"Rot: yaw={avg_yaw:.0f} pitch={avg_pitch:.0f} roll={avg_roll:.0f}",
                          (10, 70), (0, 200, 255))

            cv2.imshow("AprilTag Calibration (Robot)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if latest_R_cam_tag is None:
                    print("[!] No tag detected this frame, cannot save sample.")
                else:
                    try:
                        R_base_tool, t_base_tool = get_robot_pose()
                        sample = {
                            "R_base_tool": to_list(R_base_tool),
                            "t_base_tool": to_list(t_base_tool),
                            "R_cam_tag":   to_list(latest_R_cam_tag),
                            "t_cam_tag":   to_list(latest_t_cam_tag),
                        }
                        calibration_samples.append(sample)
                        print(f"[✓] Saved calibration sample #{len(calibration_samples)}")
                    except Exception as e:
                        print(f"[!] Error getting robot pose: {e}")

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if calibration_samples:
            with open("calibration_samples.json", "w") as f:
                json.dump(calibration_samples, f, indent=4)
            print(f"\nSaved {len(calibration_samples)} calibration samples to calibration_samples.json")
        else:
            print("\nNo calibration samples collected.")

if __name__ == "__main__":
    main()
