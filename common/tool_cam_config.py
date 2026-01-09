import numpy as np

# Rotation from tool frame to camera frame
# Best so far: flip X gave X=0.009, Y=0.003, Z=0.097
# Try: flip X and flip Y (180 rotation around Z)
R_tool_cam = np.array([
    [-1.0,  0.0, 0.0],
    [0.0,  -1.0, 0.0],
    [0.0,   0.0, 1.0],
])

# Translation from tool flange to camera lens (in meters)
# MEASURE THESE ON YOUR ROBOT:
#   X: left(-) / right(+) offset
#   Y: backward(-) / forward(+) offset  
#   Z: down(-) / up(+) offset
t_tool_cam = np.array([
    0.0,    # X: camera centered
    0.006,  # Y: camera 0.6cm forward of flange
    0.0,    # Z: no vertical offset
])