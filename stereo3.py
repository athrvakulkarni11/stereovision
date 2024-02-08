import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

def detect_red_and_blue_balls_from_webcam(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    # lower_red = np.array([0, 144, 133])
    # upper_red = np.array([10, 255, 255])

    # Define the lower and upper bounds for the blue color in HSV
    lower_blue = np.array([90, 120, 100])
    upper_blue = np.array([120, 255, 255])

    # Create masks using the inRange function
    # red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the masks
    # red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours, also Used for detecting balls at a far distance 
    min_contour_area = 10

    # Detect red balls
    # red_centroids = detect_balls(frame,  red_contours, min_contour_area, color=(0, 0, 255))

    # Detect blue balls
    x,y = detect_balls(frame, blue_contours, min_contour_area, color=(255, 0, 0))

    return x,y
def detect_balls(frame, contours, min_contour_area, color):
    max_contour_area = 0
    max_contour_centroid = (0, 0)

    for cnt in contours:
        epsilon = 0.08 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                if cv2.contourArea(cnt) > min_contour_area:
                    if cv2.contourArea(cnt) > max_contour_area:
                        max_contour_area = cv2.contourArea(cnt)
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            max_contour_centroid = (cX, cY)
                            # Draw the outer circle
                            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                            cv2.circle(frame, (int(x), int(y)), int(radius), color, 2)
                            # Draw the center of the circle
                            cv2.circle(frame, max_contour_centroid, 5, color, -1)

    return max_contour_centroid



# Open both cameras     
cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap_left =  cv2.VideoCapture(1, cv2.CAP_DSHOW)


frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 10              # Distance between the cameras [cm]
f = 6               # Camera lense's focal length [mm]
alpha = 35 # Camera field of view in the horizontal plane [degrees]

# Initial values
count = -1

while True:
    count += 1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    # If cannot catch any frame, break
    if ret_right == False or ret_left == False:
        break
    else:
        # APPLYING HSV-FILTER:
        circles_right_x,circles_right_y = detect_red_and_blue_balls_from_webcam(frame_right)
        circles_left_x ,circles_left_y= detect_red_and_blue_balls_from_webcam(frame_left)

        ################## CALIBRATION #########################################################
        # frame_right, frame_left = calib.undistorted(frame_right, frame_left)
        ########################################################################################

        # If no ball can be caught in one camera, show text "TRACKING LOST"
        if not (circles_left_x or circles_left_y) or not(circles_right_x,circles_right_y):
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth are in video presentation
            depth = tri.find_depth(circles_right_x,circles_right_y, circles_left_x ,circles_left_y, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("Depth: ", depth)

        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
