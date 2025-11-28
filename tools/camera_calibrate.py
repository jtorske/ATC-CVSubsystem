import cv2
import numpy as np
import glob
import json
import os

# ==========================================================
# Checkerboard parameters
# ==========================================================
# Board: 9 squares wide x 7 squares tall → 8 x 6 INNER corners

CHECKERBOARD = (10, 7)
SQUARE_SIZE = 0.022    # 2.2 cm → meters

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==========================================================
# Prepare 3D object points for checkerboard corners
# ==========================================================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] 
imgpoints = []  

# ==========================================================
# Load calibration images
# ==========================================================

IMAGE_DIR = "calib_images"
images = glob.glob(os.path.join(IMAGE_DIR, "*.png"))

if len(images) == 0:
    print("[!] No images found in calib_images/.")
    exit()

print(f"[INFO] Found {len(images)} calibration images.")

gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray_shape is None:
        gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)
        cv2.imshow("Corners", vis)
        cv2.waitKey(100)
    else:
        print(f"[WARN] Corners not found in {fname}")

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print(f"[!] Only {len(objpoints)} valid images – try to capture more with varied poses.")
    exit()

# ==========================================================
# Run calibration
# ==========================================================

print("\n[INFO] Running calibration...")

ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

print("\n========== CALIBRATION RESULTS ==========")
print("Camera Matrix (K):\n", K)
print("\nDistortion Coeffs:\n", dist_coeffs)
print("Reprojection Error:", ret)

# ==========================================================
# Save calibration to JSON
# ==========================================================

data = {
    "camera_matrix": K.tolist(),
    "distortion_coeffs": dist_coeffs.tolist(),
    "reprojection_error": float(ret)
}

with open("camera_intrinsics.json", "w") as f:
    json.dump(data, f, indent=4)

print("\n[✓] Saved calibration to camera_intrinsics.json")
