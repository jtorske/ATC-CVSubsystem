import cv2
import numpy as np
from pupil_apriltags import Detector

from .webcam_config import camera_params, tag_size, tag_family

def create_detector():
    return Detector(families=tag_family)

def detect_tags(detector, gray_image):
    """
    Runs AprilTag detection with pose estimation on a grayscale image.
    Returns the raw 'results' from pupil_apriltags.
    """
    return detector.detect(
        gray_image,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size,
    )

def rotation_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to XYZ Euler angles (radians).
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        yaw = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return yaw, pitch, roll
