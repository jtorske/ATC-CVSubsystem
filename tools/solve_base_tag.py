import json
import numpy as np
import cv2
import os

SAMPLES_FILE = os.path.join(os.path.dirname(__file__), "..", "calibration_samples.json")


def load_samples(path):
    with open(path, "r") as f:
        samples = json.load(f)
    print(f"[INFO] Loaded {len(samples)} samples from {path}")
    return samples


def to_np_Rt(sample):
    R_base_tool = np.array(sample["R_base_tool"], dtype=float) 
    t_base_tool = np.array(sample["t_base_tool"], dtype=float)  

    R_cam_tag = np.array(sample["R_cam_tag"], dtype=float)     
    t_cam_tag = np.array(sample["t_cam_tag"], dtype=float)    

    return R_base_tool, t_base_tool, R_cam_tag, t_cam_tag


def invert_transform(R, t):
    """Invert a rigid transform x' = R x + t"""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def average_rotation(Rs):
    """
    Average a list of rotation matrices using SVD on the sum.
    """
    M = np.zeros((3, 3))
    for R in Rs:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def main():
    if not os.path.exists(SAMPLES_FILE):
        print(f"[!] Cannot find {SAMPLES_FILE}")
        return

    samples = load_samples(SAMPLES_FILE)

    R_gripper2base = []   
    t_gripper2base = []
    R_target2cam  = []   
    t_target2cam  = []

    for s in samples:
        R_b_g, t_b_g, R_c_t, t_c_t = to_np_Rt(s)

        R_g_b, t_g_b = invert_transform(R_b_g, t_b_g)

        R_t_c = R_c_t
        t_t_c = t_c_t

        R_gripper2base.append(R_g_b)
        t_gripper2base.append(t_g_b.reshape(3, 1))
        R_target2cam.append(R_t_c)
        t_target2cam.append(t_t_c.reshape(3, 1))

    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam   = np.array(R_target2cam)
    t_target2cam   = np.array(t_target2cam)

    print("[INFO] Solving hand-eye (camera ↔ tool)...")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,  t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    t_cam2gripper = t_cam2gripper.reshape(3)

    print("\n========== HAND–EYE RESULT ==========")
    print("R_cam2tool (camera in tool frame):\n", R_cam2gripper)
    print("t_cam2tool (m):\n", t_cam2gripper)

    # ---------------------------------------------------------
    # Compute T_base_cam for each sample: T_b_c = T_b_g * T_g_c
    # ---------------------------------------------------------

    R_base_cam_list = []
    t_base_cam_list = []
    R_base_tag_list = []
    t_base_tag_list = []

    R_tool_cam, t_tool_cam = invert_transform(R_cam2gripper, t_cam2gripper)

    for s in samples:
        R_b_g, t_b_g, R_c_t, t_c_t = to_np_Rt(s)

        R_b_c = R_b_g @ R_tool_cam
        t_b_c = R_b_g @ t_tool_cam + t_b_g

        R_base_cam_list.append(R_b_c)
        t_base_cam_list.append(t_b_c)

        R_t_c = R_c_t
        t_t_c = t_c_t

        R_c_t_inv, t_c_t_inv = invert_transform(R_t_c, t_t_c)

        R_b_t = R_b_c @ R_c_t_inv
        t_b_t = R_b_c @ t_c_t_inv + t_b_c

        R_base_tag_list.append(R_b_t)
        t_base_tag_list.append(t_b_t)

    R_base_cam = average_rotation(R_base_cam_list)
    t_base_cam = np.mean(np.vstack(t_base_cam_list), axis=0)

    R_base_tag = average_rotation(R_base_tag_list)
    t_base_tag = np.mean(np.vstack(t_base_tag_list), axis=0)

    print("\n========== BASE → CAMERA ==========")
    print("R_base_cam:\n", R_base_cam)
    print("t_base_cam (m):\n", t_base_cam)

    print("\n========== BASE → TAG ==========")
    print("R_base_tag:\n", R_base_tag)
    print("t_base_tag (m):\n", t_base_tag)

    out = {
        "R_base_cam": R_base_cam.tolist(),
        "t_base_cam": t_base_cam.tolist(),
        "R_base_tag": R_base_tag.tolist(),
        "t_base_tag": t_base_tag.tolist(),
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "base_tag_solution.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"\n[✓] Saved solution to {out_path}")


if __name__ == "__main__":
    main()
