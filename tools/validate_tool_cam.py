"""
Validate Tool→Camera Calibration

This script helps validate the hand-eye calibration by:
1. Showing the current t_tool_cam values
2. Letting you physically measure the camera offset
3. Computing Base→Tag and comparing with physical measurements

Usage:
    py -3.11 -m robot.validate_tool_cam --ip 192.168.1.10 -u SSE_Student -p KinovaG3
"""

import cv2
import numpy as np
import argparse
from threading import Thread

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from robot.device_connection import DeviceConnection
from common.utils import create_detector, euler_xyz_to_R
from common.webcam_config import rtsp_url, tag_size, camera_params
from common.tool_cam_config import R_tool_cam, t_tool_cam


class RTSPStream:
    """Threaded RTSP stream reader."""
    
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
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.10")
    parser.add_argument("-u", "--username", default="SSE_Student")
    parser.add_argument("-p", "--password", default="KinovaG3")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TOOL→CAMERA CALIBRATION VALIDATION")
    print("=" * 70)
    
    print("\n[CURRENT t_tool_cam VALUES]")
    print(f"  X = {t_tool_cam[0]:.4f} m ({t_tool_cam[0]*100:.1f} cm)")
    print(f"  Y = {t_tool_cam[1]:.4f} m ({t_tool_cam[1]*100:.1f} cm)")
    print(f"  Z = {t_tool_cam[2]:.4f} m ({t_tool_cam[2]*100:.1f} cm)")
    print(f"  |t| = {np.linalg.norm(t_tool_cam):.4f} m ({np.linalg.norm(t_tool_cam)*100:.1f} cm)")
    
    print("\n[SANITY CHECK]")
    if np.linalg.norm(t_tool_cam) > 0.3:
        print("  ⚠️  WARNING: t_tool_cam magnitude > 30cm seems too large!")
        print("  ⚠️  The camera is typically only 5-15cm from the tool flange.")
        print("  ⚠️  This suggests calibration may be incorrect.")
    else:
        print("  ✓ t_tool_cam magnitude looks reasonable")
    
    print("\n[ROTATION MATRIX CHECK]")
    det = np.linalg.det(R_tool_cam)
    print(f"  det(R_tool_cam) = {det:.6f} (should be ~1.0)")
    if abs(det - 1.0) > 0.01:
        print("  ⚠️  WARNING: Rotation matrix determinant is not 1.0!")
    else:
        print("  ✓ Rotation matrix is valid")
    
    print("\n" + "-" * 70)
    print("PHYSICAL MEASUREMENT INSTRUCTIONS")
    print("-" * 70)
    print("""
The Tool→Camera transform describes where the camera is relative to the
robot's tool flange (the mounting plate at the end of the arm).

To validate, you need to PHYSICALLY MEASURE:

1. t_tool_cam[0] (X): Left/right offset from tool center to camera lens
2. t_tool_cam[1] (Y): Forward/back offset  
3. t_tool_cam[2] (Z): Up/down offset (usually small)

For Kinova Gen3 with wrist camera, typical values are ~5-10cm total offset.
    """)
    
    input("Press Enter to start live validation...")
    
    # Connect to robot
    class ConnArgs:
        def __init__(self, ip, u, p):
            self.ip = ip
            self.username = u
            self.password = p
    
    conn_args = ConnArgs(args.ip, args.username, args.password)
    
    with DeviceConnection.create_tcp_connection(conn_args) as router:
        base_client = BaseClient(router)
        cap = RTSPStream(rtsp_url(args.ip))
        detector = create_detector()
        
        print("\n" + "=" * 70)
        print("LIVE VALIDATION")
        print("=" * 70)
        print("""
Instructions:
1. Place AprilTag at a KNOWN position relative to robot base
2. Measure the tag position physically (X, Y, Z from base)
3. Compare with the computed Base→Tag shown on screen

Press 'q' to quit
Press 's' to save current readings to compare
        """)
        
        saved_readings = []
        
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
            
            # Get robot pose
            pose = base_client.GetMeasuredCartesianPose()
            t_base_tool = np.array([pose.x, pose.y, pose.z])
            R_base_tool = euler_xyz_to_R(
                np.deg2rad(pose.theta_x),
                np.deg2rad(pose.theta_y),
                np.deg2rad(pose.theta_z)
            )
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Show robot pose
            cv2.putText(display, f"Tool Position (m): X={pose.x:.3f} Y={pose.y:.3f} Z={pose.z:.3f}",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if results:
                r = results[0]
                t_cam_tag = r.pose_t[:, 0]
                R_cam_tag = r.pose_R
                
                # Compute Base→Tag
                # T_base_tag = T_base_tool @ T_tool_cam @ T_cam_tag
                R_base_cam = R_base_tool @ R_tool_cam
                t_base_cam = R_base_tool @ t_tool_cam + t_base_tool
                
                R_base_tag = R_base_cam @ R_cam_tag
                t_base_tag = R_base_cam @ t_cam_tag + t_base_cam
                
                # Draw tag
                pts = np.array([r.corners], dtype=np.int32)
                cv2.polylines(display, pts, True, (0, 255, 0), 3)
                
                # Show Cam→Tag
                cv2.putText(display, f"Cam->Tag (m): X={t_cam_tag[0]:.3f} Y={t_cam_tag[1]:.3f} Z={t_cam_tag[2]:.3f}",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show Base→Tag (THE KEY VALUE)
                cv2.putText(display, f"Base->Tag (m): X={t_base_tag[0]:.3f} Y={t_base_tag[1]:.3f} Z={t_base_tag[2]:.3f}",
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show t_tool_cam being used
                cv2.putText(display, f"t_tool_cam (m): X={t_tool_cam[0]:.3f} Y={t_tool_cam[1]:.3f} Z={t_tool_cam[2]:.3f}",
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Instructions
                cv2.putText(display, "Compare Base->Tag with your physical measurement!",
                           (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, "Press 's' to save reading, 'q' to quit",
                           (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            else:
                cv2.putText(display, "No AprilTag detected", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Tool-Cam Validation", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and results:
                saved_readings.append({
                    't_base_tool': t_base_tool.copy(),
                    't_cam_tag': t_cam_tag.copy(),
                    't_base_tag': t_base_tag.copy(),
                })
                print(f"\n[SAVED READING #{len(saved_readings)}]")
                print(f"  Tool:     X={t_base_tool[0]:.4f} Y={t_base_tool[1]:.4f} Z={t_base_tool[2]:.4f}")
                print(f"  Cam→Tag:  X={t_cam_tag[0]:.4f} Y={t_cam_tag[1]:.4f} Z={t_cam_tag[2]:.4f}")
                print(f"  Base→Tag: X={t_base_tag[0]:.4f} Y={t_base_tag[1]:.4f} Z={t_base_tag[2]:.4f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        if len(saved_readings) >= 2:
            print("\n" + "=" * 70)
            print("CONSISTENCY CHECK")
            print("=" * 70)
            
            base_tags = np.array([r['t_base_tag'] for r in saved_readings])
            mean = base_tags.mean(axis=0)
            std = base_tags.std(axis=0)
            
            print(f"\nBase→Tag across {len(saved_readings)} readings:")
            print(f"  Mean: X={mean[0]:.4f} Y={mean[1]:.4f} Z={mean[2]:.4f}")
            print(f"  Std:  X={std[0]:.4f} Y={std[1]:.4f} Z={std[2]:.4f}")
            
            if np.all(std < 0.02):
                print("\n  ✓ Good! Standard deviation < 2cm on all axes")
            else:
                print("\n  ⚠️  High variance! Base→Tag should be constant if tag is fixed.")
                print("      This indicates the Tool→Cam calibration is incorrect.")


if __name__ == "__main__":
    main()
