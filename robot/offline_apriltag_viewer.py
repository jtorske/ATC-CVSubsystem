# robot/offline_apriltag_viewer.py
#
# Offline AprilTag viewer for use outside the lab (no robotic arm needed).
#   - Works with any standard webcam
#   - Detects AprilTags and displays pose in camera frame
#   - No robot connection or Base→Tag computation
#   - Useful for testing, development, and demonstrations

import cv2
import numpy as np
import argparse
from collections import deque

from common.utils import create_detector, rotation_to_euler_xyz


# ================================================================
# DEFAULT CAMERA INTRINSICS (generalized for typical webcams)
# ================================================================

# These are reasonable defaults for a typical 720p/1080p webcam.
# Most consumer webcams have a horizontal FOV of ~60-70 degrees.
# For a 1280x720 image with ~65 deg HFOV: fx ≈ 1280 / (2 * tan(32.5°)) ≈ 1000
# Principal point is assumed at image center (will be auto-adjusted based on resolution)

DEFAULT_FOCAL_LENGTH = 800.0   # Approximate focal length (pixels) - works for most webcams
DEFAULT_TAG_SIZE = 0.05        # 5 cm tag (common for testing)
DEFAULT_TAG_FAMILY = "tag36h11"


# ================================================================
# ARG PARSER
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Offline AprilTag viewer using a standard webcam (no robot required)"
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0 = default webcam). Default: 0",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width. Default: 1280",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height. Default: 720",
    )
    p.add_argument(
        "--tag-size",
        type=float,
        default=DEFAULT_TAG_SIZE,
        help=f"AprilTag size in meters. Default: {DEFAULT_TAG_SIZE} (5 cm)",
    )
    p.add_argument(
        "--tag-family",
        type=str,
        default=DEFAULT_TAG_FAMILY,
        help=f"AprilTag family. Default: {DEFAULT_TAG_FAMILY}",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=None,
        help=f"Focal length X (pixels). Default: {DEFAULT_FOCAL_LENGTH} or auto-estimated from resolution",
    )
    p.add_argument(
        "--fy",
        type=float,
        default=None,
        help=f"Focal length Y (pixels). Default: {DEFAULT_FOCAL_LENGTH} or auto-estimated from resolution",
    )
    p.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Principal point X (pixels). Default: image center",
    )
    p.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Principal point Y (pixels). Default: image center",
    )
    return p.parse_args()


# ================================================================
# MAIN VIEWER
# ================================================================
def main():
    args = parse_args()

    used_tag_size = args.tag_size
    used_tag_family = args.tag_family

    print("[INFO] Offline AprilTag Viewer (No Robot)")
    print(f"[INFO] Camera index: {args.camera}")
    print(f"[INFO] Tag family: {used_tag_family}")
    print(f"[INFO] Tag size: {used_tag_size:.3f} m ({used_tag_size * 100:.1f} cm)")

    # ---------- Open webcam ----------
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera index {args.camera}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam resolution: {w} × {h}")

    # Build camera intrinsics - use provided values or estimate from resolution
    # Principal point defaults to image center
    cx = args.cx if args.cx is not None else w / 2.0
    cy = args.cy if args.cy is not None else h / 2.0

    # Focal length: if not provided, estimate based on resolution
    # Assuming ~65 degree horizontal FOV (typical for webcams)
    # fx = width / (2 * tan(HFOV/2)) ≈ width * 0.78 for 65 deg FOV
    estimated_focal = max(w, h) * 0.78
    fx = args.fx if args.fx is not None else estimated_focal
    fy = args.fy if args.fy is not None else estimated_focal

    cam_params = (fx, fy, cx, cy)
    print(f"[INFO] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    if args.fx is None or args.fy is None:
        print("[INFO] (Focal length auto-estimated assuming ~65° HFOV)")

    # Create detector with specified tag family
    from pupil_apriltags import Detector
    detector = Detector(families=used_tag_family)

    # Smoothing history for orientation
    yaw_hist = deque(maxlen=10)
    pitch_hist = deque(maxlen=10)
    roll_hist = deque(maxlen=10)

    print("\n[*] Offline AprilTag Viewer")
    print("    Point your webcam at an AprilTag to detect it.")
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
            camera_params=cam_params,
            tag_size=used_tag_size,
        )

        frame_has_tag = False
        cam_pos = None
        cam_rot_euler = None

        for r in results:
            frame_has_tag = True

            # Draw tag outline
            pts = np.array([r.corners], dtype=np.int32)
            cv2.polylines(frame, pts, True, (0, 255, 0), 2)

            # Draw tag center
            center = r.center.astype(int)
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

            # Label with tag ID
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

            # Pose in camera frame
            t_cam_tag = r.pose_t[:, 0]  # (3,)
            R_cam_tag = r.pose_R        # 3x3

            cam_pos = t_cam_tag.copy()

            yaw, pitch, roll = rotation_to_euler_xyz(R_cam_tag)
            yaw_hist.append(np.degrees(yaw))
            pitch_hist.append(np.degrees(pitch))
            roll_hist.append(np.degrees(roll))

            avg_yaw = np.mean(yaw_hist)
            avg_pitch = np.mean(pitch_hist)
            avg_roll = np.mean(roll_hist)
            cam_rot_euler = (avg_yaw, avg_pitch, avg_roll)

            # Draw coordinate axes on the tag
            draw_axes(frame, R_cam_tag, t_cam_tag, cam_params, used_tag_size)

            # Only process first detected tag for HUD
            break

        # ---------------- HUD ----------------
        hud_height = 140
        cv2.rectangle(frame, (5, 5), (500, hud_height), (0, 0, 0), -1)
        y0 = 30
        dy = 28

        if frame_has_tag and cam_pos is not None:
            cv2.putText(
                frame,
                f"Cam->Tag: x={cam_pos[0]:.3f}  y={cam_pos[1]:.3f}  z={cam_pos[2]:.3f} m",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y0 += dy

            distance = np.linalg.norm(cam_pos)
            cv2.putText(
                frame,
                f"Distance: {distance:.3f} m ({distance * 100:.1f} cm)",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            y0 += dy

            if cam_rot_euler is not None:
                yaw_disp, pitch_disp, roll_disp = cam_rot_euler
                cv2.putText(
                    frame,
                    f"Rotation: yaw={yaw_disp:.1f}  pitch={pitch_disp:.1f}  roll={roll_disp:.1f} deg",
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 255),
                    2,
                )
                y0 += dy

            cv2.putText(
                frame,
                f"Tags detected: {len(results)}",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
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
            y0 += dy
            cv2.putText(
                frame,
                f"Looking for {used_tag_family} tags...",
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )

        cv2.imshow("Offline AprilTag Viewer (Webcam)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Viewer closed.")


def draw_axes(frame, R, t, camera_params, axis_length):
    """
    Draw 3D coordinate axes on the frame at the tag location.
    Red = X, Green = Y, Blue = Z
    """
    fx, fy, cx, cy = camera_params
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Axis endpoints in tag frame
    axis_len = axis_length * 0.5
    axes_3d = np.array([
        [0, 0, 0],              # origin
        [axis_len, 0, 0],       # X
        [0, axis_len, 0],       # Y
        [0, 0, -axis_len],      # Z (towards camera)
    ], dtype=np.float32).T  # 3x4

    # Transform to camera frame
    axes_cam = R @ axes_3d + t.reshape(3, 1)

    # Project to image plane
    axes_2d = K @ axes_cam
    axes_2d = axes_2d[:2, :] / axes_2d[2, :]
    axes_2d = axes_2d.T.astype(int)

    origin = tuple(axes_2d[0])
    x_end = tuple(axes_2d[1])
    y_end = tuple(axes_2d[2])
    z_end = tuple(axes_2d[3])

    # Draw axes
    cv2.line(frame, origin, x_end, (0, 0, 255), 2)   # X = Red
    cv2.line(frame, origin, y_end, (0, 255, 0), 2)   # Y = Green
    cv2.line(frame, origin, z_end, (255, 0, 0), 2)   # Z = Blue


if __name__ == "__main__":
    main()
