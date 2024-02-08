import sys
import cv2
import numpy as np
import time

def find_depth(circles_right_x, circles_right_y, circles_left_x, circles_left_y, frame_right, frame_left, baseline, f, alpha):
    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print('Left and right camera frames do not have the same pixel width')
        return None

    x_right, y_right = circles_right_x, circles_right_y
    x_left, y_left = circles_left_x, circles_left_y

    # CALCULATE THE DISPARITY:
    disparity = abs(x_left - x_right)  # Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    if disparity != 0:
        zDepth = (baseline * f_pixel) / disparity  # Depth in [cm]
        return abs(zDepth)
    else:
        print('Disparity is zero, cannot calculate depth.')
        return None
