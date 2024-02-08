import cv2
import numpy as np
import cvzone

from cvzone.FaceMeshModule import FaceMeshDetector

# Function to detect the blue ball
def detect_blue_ball(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Find the contour with the maximum area (assumed to be the blue ball)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center and radius of the circle
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the circle on the frame
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Display the center coordinates
        cv2.putText(frame, f'Ball Center: {center}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Open camera sources
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Check if the cameras opened successfully
if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Couldn't open cameras.")
    exit(-1)

# Initialize SIFT
sift = cv2.SIFT_create()

# Initialize FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

while True:
    # Read frames from both cameras
    ret0, fr0 = cap0.read()
    ret1, fr1 = cap1.read()

    # Break the loop if either of the cameras has an issue
    if not ret0 or not ret1:
        print("Error: Couldn't read frames from cameras.")
        break

    # Detect keypoints and compute descriptors
    kp0, des0 = sift.detectAndCompute(fr0, None)
    kp1, des1 = sift.detectAndCompute(fr1, None)

    # Match descriptors using BFMatcher
    matches = bf.knnMatch(des0, des1, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # Adjust the ratio threshold as needed
            good_matches.append(m)

    # Check if there are enough good matches
    if len(good_matches) >= 4:
        src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Check if a valid homography matrix is found
        if H is not None:
            # Ensure the homography matrix is of the correct type
            H = H.astype(np.float32)

            # Warp the first image to align with the second
            stitched_img = cv2.warpPerspective(fr0, H, (fr1.shape[1] + fr0.shape[1], fr1.shape[0]))

            # Combine the two images
            stitched_img[:, :fr1.shape[1]] = fr1

            # Detect the blue ball in the stitched image
            detect_blue_ball(stitched_img)

            # Display frames and stitched image
            cv2.imshow("Camera 0", fr0)
            cv2.imshow("Camera 1", fr1)
            cv2.imshow("Stitched Image", stitched_img)

            # Visualize correspondences (debugging)
            img_matches = cv2.drawMatches(fr0, kp0, fr1, kp1, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches", img_matches)

    # Detect the blue ball in each individual frame
    detect_blue_ball(fr0)
    detect_blue_ball(fr1)

    # Display individual frames
    cv2.imshow("Camera 0", fr0)
    cv2.imshow("Camera 1", fr1)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap0.release()
cap1.release()
cv2.destroyAllWindows()
