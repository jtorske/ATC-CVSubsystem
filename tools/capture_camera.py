import cv2
import os

# Robot camera RTSP URL
ROBOT_IP = "192.168.1.10"
RTSP_URL = f"rtsp://{ROBOT_IP}/color"

SAVE_DIR = "calib_images"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"[!] Could not open RTSP stream: {RTSP_URL}")
        return

    print("[*] Calibration capture running")
    print("    - Show the checkerboard on your phone screen")
    print("    - Move it around the FOV (near/far, left/right, tilt)")
    print("    - Press 'c' to capture a frame")
    print("    - Press 'q' to quit")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame from camera")
            break

        cv2.imshow("Calibration Capture (Robot Camera)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            filename = os.path.join(SAVE_DIR, f"calib_{i:03d}.png")
            cv2.imwrite(filename, frame)
            print(f"[✓] Saved {filename}")
            i += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[*] Done. Saved {i} images in '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
