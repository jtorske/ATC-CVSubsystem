"""
Hand-Eye Calibration: Find the Tool→Camera transform.

Procedure:
1. Place an AprilTag in a FIXED location (don't move it!)
2. Move the robot arm to multiple poses while keeping the tag visible
3. Record (Base→Tool, Cam→Tag) pairs at each pose
4. Solve AX=XB problem
"""
import cv2
import numpy as np
import json

def collect_calibration_samples(ip, username, password, num_samples=10):
    """Collect pose pairs for hand-eye calibration."""
    from robot.device_connection import DeviceConnection
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from common.utils import create_detector, euler_xyz_to_R
    from common.webcam_config import rtsp_url, tag_size, camera_params
    
    samples = []
    
    class ConnArgs:
        def __init__(self, ip, u, p):
            self.ip = ip
            self.username = u
            self.password = p
    
    conn_args = ConnArgs(ip, username, password)
    
    with DeviceConnection.create_tcp_connection(conn_args) as router:
        base_client = BaseClient(router)
        cap = cv2.VideoCapture(rtsp_url(ip))
        detector = create_detector()
        
        print(f"Collect {num_samples} samples.")
        print("Move arm to different poses, press 'c' to capture, 'q' to finish.\n")
        
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=True,
                                      camera_params=camera_params, tag_size=tag_size)
            
            display = frame.copy()
            tag_found = len(results) > 0
            
            if tag_found:
                r = results[0]
                pts = np.array([r.corners], dtype=np.int32)
                cv2.polylines(display, pts, True, (0, 255, 0), 3)
                cv2.putText(display, "Tag found - 'c' to capture", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No tag visible", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(display, f"Samples: {len(samples)}/{num_samples}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Hand-Eye Calibration", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and tag_found:
                # Get robot pose
                pose = base_client.GetMeasuredCartesianPose()
                t_base_tool = np.array([pose.x, pose.y, pose.z])
                R_base_tool = euler_xyz_to_R(
                    np.deg2rad(pose.theta_x),
                    np.deg2rad(pose.theta_y),
                    np.deg2rad(pose.theta_z)
                )
                
                # Get tag pose
                r = results[0]
                t_cam_tag = r.pose_t[:, 0]
                R_cam_tag = r.pose_R
                
                samples.append({
                    "R_base_tool": R_base_tool.tolist(),
                    "t_base_tool": t_base_tool.tolist(),
                    "R_cam_tag": R_cam_tag.tolist(),
                    "t_cam_tag": t_cam_tag.tolist(),
                })
                print(f"Captured sample {len(samples)}")
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Save samples
    with open("calibration_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved {len(samples)} samples to calibration_samples.json")
    return samples


def solve_hand_eye(samples):
    """Solve for Tool→Camera transform using OpenCV."""
    R_gripper2base = []  # R_base_tool.T (inverted)
    t_gripper2base = []
    R_target2cam = []    # R_cam_tag
    t_target2cam = []
    
    for s in samples:
        R_bt = np.array(s["R_base_tool"])
        t_bt = np.array(s["t_base_tool"])
        R_ct = np.array(s["R_cam_tag"])
        t_ct = np.array(s["t_cam_tag"])
        
        # OpenCV wants gripper2base (inverse of base2tool)
        R_gripper2base.append(R_bt.T)
        t_gripper2base.append(-R_bt.T @ t_bt)
        R_target2cam.append(R_ct)
        t_target2cam.append(t_ct)
    
    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # We want Tool→Cam, so invert
    R_tool_cam = R_cam2tool.T
    t_tool_cam = -R_cam2tool.T @ t_cam2tool.flatten()
    
    print("\n=== HAND-EYE CALIBRATION RESULTS ===")
    print("R_tool_cam =")
    print(repr(R_tool_cam))
    print(f"\nt_tool_cam (meters) =")
    print(repr(t_tool_cam))
    
    # Save
    result = {
        "R_tool_cam": R_tool_cam.tolist(),
        "t_tool_cam": t_tool_cam.tolist(),
    }
    with open("tool_cam_calibration.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to tool_cam_calibration.json")
    
    return R_tool_cam, t_tool_cam


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="192.168.1.10")
    p.add_argument("-u", "--username", default="SSE_Student")
    p.add_argument("-p", "--password", default="Kinova G3")
    p.add_argument("--solve-only", action="store_true", help="Just solve from existing samples")
    args = p.parse_args()
    
    if args.solve_only:
        with open("calibration_samples.json") as f:
            samples = json.load(f)
    else:
        samples = collect_calibration_samples(args.ip, args.username, args.password)
    
    if len(samples) >= 3:
        solve_hand_eye(samples)