import cv2

# Create a stitcher object
stitcher = cv2.Stitcher_create()

# Create video capture objects for camera source 0 and 1
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Check if camera sources are opened successfully
if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Unable to open one or more camera sources.")
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    exit()

while True:
    # Capture frames from camera source 0
    ret0, frame0 = cap0.read()

    # Capture frames from camera source 1
    ret1, frame1 = cap1.read()

    # Check if frames are captured successfully
    if not ret0 or not ret1:
        print("Error: Unable to capture frames from one or more sources.")
        break

    # Stitch the frames together
    status, result = stitcher.stitch((frame0, frame1))

    # Check if stitching was successful
    if status == cv2.Stitcher_OK:
        cv2.imshow('Stitched Image', result)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects
cap0.release()
cap1.release()
cv2.destroyAllWindows()
