import cv2
import os

def convert_video_to_images(video_path, output_folder, frame_interval=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through frames and save images
    for frame_count in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image
            image_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(image_filename, frame)
            print(f"Saved: {image_filename}")
        else:
            print(f"Error reading frame {frame_count}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = "silo - Made with Clipchamp.mp4"
    output_folder = "silo"
    frame_interval = 20  # Number of frames to skip between saved frames

    convert_video_to_images(video_path, output_folder, frame_interval)
