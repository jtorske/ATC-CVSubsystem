import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque
from K3N_Connection import *

# Imported Kinova API
import sys
import os

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
    

def main():
    # Camera intrinsic parameters
    # Will need to replace this with some callibrated values (requires seperate script)
    fx = 600 
    fy = 600 
    cx = 320
    cy = 240 

    # Tag size in meters
    tag_size = 0.05

    import utilities
    
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)

        # example core
        vision_device_id = example_vision_get_device_id(device_manager)

    # Changed the video capture to the ip of the K3N arm
    cap = cv2.VideoCapture("rtsp://192.168.1.10/color")

    detector = Detector(families="tag36h11")

    # Buffers for smoothing
    yaw_hist, pitch_hist, roll_hist = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags with pose estimation
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size
        )

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            pts = np.array([ptA, ptB, ptC, ptD], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Draw ID
            cv2.putText(frame, f"ID: {r.tag_id}",
                        (int(ptA[0]), int(ptA[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Extract pose
            t = r.pose_t  # translation vector (x, y, z) in meters
            R = r.pose_R  # 3x3 rotation matrix

            # convert rotation matrix to Euler angles
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            if sy > 1e-6:
                yaw = np.arctan2(R[2, 1], R[2, 2])
                pitch = np.arctan2(-R[2, 0], sy)
                roll = np.arctan2(R[1, 0], R[0, 0])
            else:
                yaw = np.arctan2(-R[1, 2], R[1, 1])
                pitch = np.arctan2(-R[2, 0], sy)
                roll = 0

            # Add values to smoothing buffers
            yaw_hist.append(np.degrees(yaw))
            pitch_hist.append(np.degrees(pitch))
            roll_hist.append(np.degrees(roll))

            avg_yaw = np.mean(yaw_hist)
            avg_pitch = np.mean(pitch_hist)
            avg_roll = np.mean(roll_hist)

            # Black rectangle background
            cv2.rectangle(frame, (5, 5), (470, 90), (0, 0, 0), -1)

            # New function to make text drawing more visible
            def draw_text(img, text, pos, color):
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3) 
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)    

            # Translation values
            draw_text(frame,
                    f"Pos: x={t[0][0]:.2f} y={t[1][0]:.2f} z={t[2][0]:.2f} m",
                    (10, 35), (0, 255, 0)) 

            # Rotation values (smoothed)
            draw_text(frame,
                    f"Rot: yaw={avg_yaw:.0f} pitch={avg_pitch:.0f} roll={avg_roll:.0f}",
                    (10, 70), (0, 200, 255)) 


        cv2.imshow("AprilTag Detection with Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()