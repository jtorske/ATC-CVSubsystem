import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque
from K3N_Connection import *

# Imported Kinova API
import sys
import os

#from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
#from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

#from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
    
# Changed the video capture to the ip of the K3N arm
cap = cv2.VideoCapture("rtsp://192.168.1.10/color")

def main():
    # Camera intrinsic parameters
    # Will need to replace this with some callibrated values (requires seperate script)
    fx = 600 
    fy = 600 
    cx = 320
    cy = 240 

while True:
    ret, frame = cap.read()
    if not ret:
        break

        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)

        # example core
        vision_device_id = example_vision_get_device_id(device_manager)

    

    detector = Detector(families="tag36h11")

    # Buffers for smoothing
    yaw_hist, pitch_hist, roll_hist = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Rotation values
        draw_text(frame,
                f"Rot: yaw={np.degrees(yaw):.1f} pitch={np.degrees(pitch):.1f} roll={np.degrees(roll):.1f}",
                (10, 70), (0, 200, 255)) 


        cv2.imshow("AprilTag Detection with Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()