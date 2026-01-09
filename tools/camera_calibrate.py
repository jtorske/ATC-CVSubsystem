"""
Complete Camera Intrinsic Calibration for Kinova Gen3

This script:
1. Captures images from the Kinova RTSP stream with a checkerboard
2. Computes camera intrinsics (fx, fy, cx, cy) and distortion coefficients
3. Validates the calibration with reprojection error
4. Saves results to webcam_config.py format

Requirements:
- Print a checkerboard pattern (default: 9x6 inner corners)
- Measure your square size accurately in meters

Usage:
    py -3.11 -m robot.test --ip 192.168.1.10

Instructions:
1. Hold the checkerboard at various angles and distances
2. Press 'c' to capture when corners are detected (green overlay)
3. Capture 15-25 images with good variety
4. Press 'q' when done to compute calibration
"""

import cv2
import numpy as np
import json
import argparse
from datetime import datetime
from threading import Thread


class RTSPStream:
    """Threaded RTSP stream reader - always has the latest frame."""
    
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
    
    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def get(self, prop):
        return self.cap.get(prop)
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def calibrate_camera(ip: str, checkerboard: tuple, square_size: float, num_samples: int):
    """
    Run interactive camera calibration.
    
    Args:
        ip: Robot IP address
        checkerboard: (columns, rows) of inner corners
        square_size: Size of one square in meters
        num_samples: Target number of calibration images
    """
    url = f"rtsp://{ip}/color"
    print(f"[INFO] Opening RTSP stream: {url}")
    
    # Use threaded reader for real-time performance
    cap = RTSPStream(url)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open stream: {url}")
        return None
    
    # Get actual stream resolution - THIS IS CRITICAL
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream resolution: {w} x {h}")
    print(f"[INFO] Checkerboard: {checkerboard[0]}x{checkerboard[1]} inner corners")
    print(f"[INFO] Square size: {square_size*100:.2f} cm ({square_size} m)")
    
    # Prepare object points (3D coordinates of checkerboard corners)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size
    
    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image
    
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print("\n" + "=" * 60)
    print("CAMERA CALIBRATION")
    print("=" * 60)
    print("Instructions:")
    print("  - Hold checkerboard at various angles and distances")
    print("  - Cover different parts of the image (corners, center)")
    print("  - Vary the distance (close, medium, far)")
    print("  - Tilt the board in different directions")
    print("")
    print("Controls:")
    print("  'c' - Capture current frame (when corners detected)")
    print("  'q' - Finish and compute calibration")
    print("  'r' - Reset (discard all captures)")
    print("=" * 60 + "\n")
    
    while len(obj_points) < num_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray, checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        
        display = frame.copy()
        
        if found:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(display, checkerboard, corners_refined, found)
            
            # Status text
            cv2.putText(display, "CORNERS FOUND - Press 'c' to capture", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No checkerboard detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show capture count
        cv2.putText(display, f"Captures: {len(obj_points)}/{num_samples}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Show resolution
        cv2.putText(display, f"Resolution: {w}x{h}", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Camera Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and found:
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"[✓] Captured sample {len(obj_points)}/{num_samples}")
            
            # Brief flash to confirm capture
            cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)
            cv2.imshow("Camera Calibration", display)
            cv2.waitKey(200)
            
        elif key == ord('q'):
            print("\n[INFO] Finishing capture...")
            break
            
        elif key == ord('r'):
            obj_points.clear()
            img_points.clear()
            print("[!] Reset - all captures discarded")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(obj_points) < 5:
        print(f"[ERROR] Not enough samples ({len(obj_points)}). Need at least 5.")
        return None
    
    print(f"\n[INFO] Computing calibration from {len(obj_points)} samples...")
    
    # Run calibration
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error
    mean_error = total_error / len(obj_points)
    
    # Extract parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Resolution: {w} x {h}")
    print(f"RMS Reprojection Error: {ret:.4f} pixels")
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")
    print("")
    print("Camera Intrinsics:")
    print(f"  fx = {fx:.8f}")
    print(f"  fy = {fy:.8f}")
    print(f"  cx = {cx:.8f}")
    print(f"  cy = {cy:.8f}")
    print("")
    print("Distortion Coefficients (k1, k2, p1, p2, k3):")
    print(f"  {dist_coeffs.ravel()}")
    print("=" * 60)
    
    if ret > 1.0:
        print("\n[WARNING] Reprojection error is high (>1 pixel).")
        print("Consider recalibrating with better samples.")
    
    # Save results
    result = {
        "resolution": {"width": w, "height": h},
        "camera_matrix": K.tolist(),
        "distortion_coeffs": dist_coeffs.tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "rms_error": ret,
        "mean_error": mean_error,
        "num_samples": len(obj_points),
        "checkerboard": checkerboard,
        "square_size_m": square_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save to JSON
    json_path = "camera_calibration.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] Saved calibration to {json_path}")
    
    # Generate webcam_config.py content
    config_content = f'''# common/webcam_config.py
#
# AUTO-GENERATED by robot/test.py (camera calibration)
# Timestamp: {datetime.now().isoformat()}
# Resolution: {w} x {h}
# RMS Error: {ret:.4f} pixels
#

import numpy as np

# ================================================================
#  Camera Configuration for Kinova Gen3 / RTSP Camera
#  Calibrated at resolution: {w} x {h}
# ================================================================

# ---- Intrinsic Parameters ----
fx = {fx:.8f}
fy = {fy:.8f}
cx = {cx:.8f}
cy = {cy:.8f}

camera_params = (fx, fy, cx, cy)

# ---- Distortion Coefficients (k1, k2, p1, p2, k3) ----
dist_coeffs = np.array([
    {dist_coeffs[0, 0]:.8f},
    {dist_coeffs[0, 1]:.8f},
    {dist_coeffs[0, 2]:.8f},
    {dist_coeffs[0, 3]:.8f},
    {dist_coeffs[0, 4]:.8f}
])

# ---- AprilTag parameters ----
# IMPORTANT: Measure your tag accurately! Edge to edge of outer black border.
tag_size = 0.132         # meters - UPDATE THIS TO YOUR ACTUAL TAG SIZE
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
    """Build the RTSP URL for the Kinova Gen3 color stream."""
    return f"rtsp://{{ip}}/color"
'''
    
    # Save the config file
    config_path = "common/webcam_config_NEW.py"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"[✓] Generated {config_path}")
    print(f"\n[ACTION REQUIRED] Review and rename to webcam_config.py:")
    print(f"    1. Verify the tag_size value (currently 0.132 m = 13.2 cm)")
    print(f"    2. Run: copy common\\webcam_config_NEW.py common\\webcam_config.py")
    
    return result


def validate_calibration(ip: str):
    """
    Validate existing calibration by measuring a known distance.
    """
    from common.webcam_config import camera_params, tag_size
    from common.utils import create_detector
    
    fx, fy, cx, cy = camera_params
    
    url = f"rtsp://{ip}/color"
    cap = RTSPStream(url)
    detector = create_detector()
    
    print("\n" + "=" * 60)
    print("CALIBRATION VALIDATION")
    print("=" * 60)
    print(f"Current camera_params: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Current tag_size: {tag_size} m ({tag_size*100} cm)")
    print("")
    print("Instructions:")
    print("  1. Place the AprilTag at a KNOWN distance from camera")
    print("  2. Compare reported distance vs actual measured distance")
    print("  3. Press 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size
        )
        
        display = frame.copy()
        
        if results:
            r = results[0]
            t = r.pose_t[:, 0]
            distance = np.linalg.norm(t)
            
            # Draw tag
            pts = np.array([r.corners], dtype=np.int32)
            cv2.polylines(display, pts, True, (0, 255, 0), 3)
            
            # Show distance
            cv2.putText(display, f"Reported distance: {distance:.3f} m ({distance*100:.1f} cm)", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Z (depth): {t[2]:.3f} m", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display, "Compare with your measured distance!", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display, "No AprilTag detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Calibration Validation", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration for Kinova Gen3")
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Robot IP address")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners (columns)")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners (rows)")
    parser.add_argument("--square-size", type=float, default=0.022, 
                       help="Checkerboard square size in meters (default: 0.022 = 2.2cm)")
    parser.add_argument("--samples", type=int, default=20, help="Number of calibration samples")
    parser.add_argument("--validate", action="store_true", help="Validate existing calibration")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_calibration(args.ip)
    else:
        calibrate_camera(
            ip=args.ip,
            checkerboard=(args.cols, args.rows),
            square_size=args.square_size,
            num_samples=args.samples
        )
