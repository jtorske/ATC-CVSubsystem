import cv2
import numpy as np
from pupil_apriltags import Detector

# Camera intrinsic parameters
# Will need to replace this with some callibrated values (requires seperate script)
fx = 600 
fy = 600 
cx = 320
cy = 240 

# Tag size in meters
tag_size = 0.05

cap = cv2.VideoCapture(0)

detector = Detector(families="tag36h11")

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

        # Show translation
        cv2.putText(frame,
                    f"Pos: x={t[0][0]:.2f} y={t[1][0]:.2f} z={t[2][0]:.2f} m",
                    (int(ptA[0]), int(ptA[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # convert rotation matrix to Euler angles
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            yaw = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            roll = np.arctan2(R[1,0], R[0,0])
        else:
            yaw = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            roll = 0

        cv2.putText(frame,
                    f"Rot: yaw={np.degrees(yaw):.1f} pitch={np.degrees(pitch):.1f} roll={np.degrees(roll):.1f}",
                    (int(ptA[0]), int(ptA[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    cv2.imshow("AprilTag Detection with Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
