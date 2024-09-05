import cv2
import os

def video_to_screenshots(video_path, output_folder, frame_interval=1):
    print(f"===== ENTERING video_to_screenshots FUNCTION =====")
    screenshots_folder = os.path.join(output_folder, "originals")
    os.makedirs(screenshots_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    
    frame_count = 0
    screenshot_count = 0

    print(f"Converting video: {video_path}")
    print(f"Output folder: {screenshots_folder}")

    while True:
        success, frame = video.read()
        
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            screenshot_path = os.path.join(screenshots_folder, f"screenshot_{screenshot_count:04d}.png")
            cv2.imwrite(screenshot_path, frame)
            screenshot_count += 1
        
        frame_count += 1

    video.release()

    print(f"Converted {screenshot_count} frames to screenshots.")
    print(f"===== EXITING video_to_screenshots FUNCTION =====")

def main():
    print("===== ENTERING main FUNCTION =====")
    try:
        video_to_screenshots("crafter_video.mp4", "screenshots")
    except Exception as e:
        print(f"===== ERROR IN main: {str(e)} =====")
    print("===== EXITING main FUNCTION =====")

if __name__ == "__main__":
    print("===== SCRIPT STARTED =====")
    main()
    print("===== SCRIPT ENDED =====")
