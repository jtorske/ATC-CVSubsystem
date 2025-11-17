# local/apriltag_viewer.py

import cv2
import numpy as np
from collections import deque

from common.apriltag_utils import create_detector, detect_tags, rotation_to_euler_xyz

def main():
    # Use your webcam at home
    cap = cv2.VideoCapture(0)

    detector = create_detector()

    yaw_hist = deque(maxlen=10)
    pitch_hist = deque(maxlen=10)
    roll_hist = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detect_tags(detector, gray)

        latest_t = None
        latest_R = None

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            pts = np.array([ptA, ptB, ptC, ptD], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            cv2.putText(frame, f"ID: {r.tag_id}",
                        (int(ptA[0]), int(ptA[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            t = r.pose_t
            R = r.pose_R

            latest_t = t
            latest_R = R

        if latest_R is not None:
            yaw, pitch, roll = rotation_to_euler_xyz(latest_R)

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

            draw_text(
                frame,
                f"Pos: x={latest_t[0][0]:.2f} y={latest_t[1][0]:.2f} z={latest_t[2][0]:.2f} m",
                (10, 35),
                (0, 255, 0),
            )
            draw_text(
                frame,
                f"Rot: yaw={avg_yaw:.0f} pitch={avg_pitch:.0f} roll={avg_roll:.0f}",
                (10, 70),
                (0, 200, 255),
            )

        cv2.imshow("AprilTag Viewer (Local – No Robot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
