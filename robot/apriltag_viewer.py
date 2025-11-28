import cv2
import numpy as np
import argparse
from collections import deque

from common.apriltag_utils import create_detector, rotation_to_euler_xyz
from common.webcam_config import (
    rtsp_url,
    tag_size,
    camera_params,   
)
from common.base_tag_config import R_base_cam, t_base_cam  
from common.tool_cam_config import R_tool_cam, t_tool_cam


# ================================================================
# DEPTH / SCALE CONSTANTS – MATCH apriltag_calibration.py
# ================================================================

POSE_SCALE = 1.6730539

fx_calib, fy_calib, cx_calib, cy_calib = camera_params
SCALE_Z = 0.30 / 0.225

fx_tag = fx_calib * SCALE_Z
fy_tag = fy_calib * SCALE_Z
camera_params_tag = (fx_tag, fy_tag, cx_calib, cy_calib)

print("[INFO] Calibrated intrinsics (checkerboard):")
print(f"  fx = {fx_calib:.3f}, fy = {fy_calib:.3f}, cx = {cx_calib:.3f}, cy = {cy_calib:.3f}")
print("[INFO] Empirical Z scale factor (intrinsics):", SCALE_Z)
print("[INFO] Global pose scale factor (POSE_SCALE):", POSE_SCALE)
print("[INFO] AprilTag intrinsics used for pose:")
print(f"  fx_tag = {fx_tag:.3f}, fy_tag = {fy_tag:.3f}, cx = {cx_calib:.3f}, cy = {cy_calib:.3f}")
print("[INFO] Base→Cam from common.base_tag_config:")
print("R_base_cam =\n", R_base_cam)
print("t_base_cam =", t_base_cam)


# ================================================================
# COMPOSE BASE→TAG FROM BASE→CAM AND CAM→TAG
# ================================================================

def compose_base_tag(R_base_tool, t_base_tool, R_cam_tag, t_cam_tag):
    R_base_cam = R_base_tool @ R_tool_cam
    t_base_cam = R_base_tool @ t_tool_cam + t_base_tool

    R_base_tag = R_base_cam @ R_cam_tag
    t_base_tag = R_base_cam @ t_cam_tag + t_base_cam

    return R_base_tag, t_base_tag

# ================================================================
# ROTATION ERROR (OPTIONAL)
# ================================================================

def rotation_error_deg(R_ref, R_live):
    R_err = R_ref.T @ R_live
    tr = np.trace(R_err)
    cos_theta = max(min((tr - 1.0) / 2.0, 1.0), -1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# ================================================================
# ARG PARSER
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="AprilTag base→tag viewer using Kinova K3 camera"
    )
    p.add_argument(
        "--ip",
        type=str,
        default="192.168.1.10",
        help="Robot IP (RTSP camera). Default: 192.168.1.10",
    )
    p.add_argument("--gt-x", type=float, default=None, help="Ground-truth Base→Tag X [m]")
    p.add_argument("--gt-y", type=float, default=None, help="Ground-truth Base→Tag Y [m]")
    p.add_argument("--gt-z", type=float, default=None, help="Ground-truth Base→Tag Z [m]")
    return p.parse_args()


# ================================================================
# MAIN VIEWER
# ================================================================

def main():
    args = parse_args()

    gt_vec = None
    if args.gt_x is not None and args.gt_y is not None and args.gt_z is not None:
        gt_vec = np.array([args.gt_x, args.gt_y, args.gt_z], dtype=float)
        print("[INFO] Using ground-truth Base→Tag for comparison:")
        print(f"       x={gt_vec[0]:.3f}, y={gt_vec[1]:.3f}, z={gt_vec[2]:.3f} m")

    url = rtsp_url(args.ip)
    print(f"[INFO] Opening camera: {url}")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"[!] ERROR: Could not open RTSP stream: {url}")
        return

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] RTSP stream resolution: {w} × {h}")

    detector = create_detector()

    yaw_hist   = deque(maxlen=10)
    pitch_hist = deque(maxlen=10)
    roll_hist  = deque(maxlen=10)

    print("\n[*] AprilTag Base→Tag viewer")
    print("    Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params_tag,
            tag_size=tag_size,
        )

        frame_has_tag = False
        cam_pos = None
        base_pos = None
        base_rot_euler = None
        pos_err = None
        rot_err_deg = None

        for r in results:
            frame_has_tag = True

            pts = np.array([r.corners], dtype=np.int32)
            cv2.polylines(frame, pts, True, (0, 255, 0), 2)

            ptA = r.corners[0]
            cv2.putText(
                frame,
                f"ID: {r.tag_id}",
                (int(ptA[0]), int(ptA[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            t_raw = r.pose_t        
            R_cam_tag = r.pose_R     

            t_scaled = t_raw * POSE_SCALE
            t_cam_tag = t_scaled[:, 0]  

            cam_pos = t_cam_tag.copy()

            yaw, pitch, roll = rotation_to_euler_xyz(R_cam_tag)
            yaw_hist.append(np.degrees(yaw))
            pitch_hist.append(np.degrees(pitch))
            roll_hist.append(np.degrees(roll))

            avg_yaw = np.mean(yaw_hist)
            avg_pitch = np.mean(pitch_hist)
            avg_roll = np.mean(roll_hist)

            base_rot_euler = (avg_pitch, avg_yaw, avg_roll)

            R_base_tag, t_base_tag = compose_base_tag(
                R_base_cam, t_base_cam, R_cam_tag, t_cam_tag
            )
            base_pos = t_base_tag.copy()

            if gt_vec is not None:
                pos_err = base_pos - gt_vec
                R_ref = np.eye(3)
                rot_err_deg = rotation_error_deg(R_ref, R_base_tag)

            break

        # ---------------- HUD ----------------
        cv2.rectangle(frame, (5, 5), (780, 190), (0, 0, 0), -1)
        y0 = 30
        dy = 25

        if frame_has_tag and cam_pos is not None and base_pos is not None:
            cv2.putText(
                frame,
                f"Cam→Tag:  x={cam_pos[0]:.3f}  y={cam_pos[1]:.3f}  z={cam_pos[2]:.3f} m",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y0 += dy

            cv2.putText(
                frame,
                f"Base→Tag: x={base_pos[0]:.3f}  y={base_pos[1]:.3f}  z={base_pos[2]:.3f} m",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            y0 += dy

            if base_rot_euler is not None:
                yaw_disp, pitch_disp, roll_disp = base_rot_euler
                cv2.putText(
                    frame,
                    f"Tag Rot (cam): yaw={yaw_disp:.0f}  pitch={pitch_disp:.0f}  roll={roll_disp:.0f}",
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 255),
                    2,
                )
                y0 += dy

            if gt_vec is not None and pos_err is not None:
                dx_cm = pos_err[0] * 100.0
                dy_cm = pos_err[1] * 100.0
                dz_cm = pos_err[2] * 100.0
                norm_cm = float(np.linalg.norm(pos_err) * 100.0)

                cv2.putText(
                    frame,
                    f"GT Base→Tag: x={gt_vec[0]:.3f} y={gt_vec[1]:.3f} z={gt_vec[2]:.3f} m",
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                y0 += dy

                cv2.putText(
                    frame,
                    f"Δ vs GT: dx={dx_cm:+.1f} dy={dy_cm:+.1f} dz={dz_cm:+.1f} cm  |Δ|={norm_cm:.1f} cm",
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 0),
                    2,
                )
                y0 += dy

                if rot_err_deg is not None:
                    cv2.putText(
                        frame,
                        f"Approx rot error vs flat: {rot_err_deg:.1f}°",
                        (15, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 180, 180),
                        2,
                    )
        else:
            cv2.putText(
                frame,
                "No AprilTag detected",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("AprilTag Base→Tag Viewer (Robot Camera)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Viewer closed.")


if __name__ == "__main__":
    main()
