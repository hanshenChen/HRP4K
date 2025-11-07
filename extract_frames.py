import cv2
import os

def extract_frames_from_video(video_path, output_dir, fps=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video Information:")
    print(f"  - Original FPS: {video_fps}")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Extraction Rate: {fps} frames per second")
    
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            output_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_filename, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} images...")
        
        frame_count += 1
    
    video.release()
    
    print(f"\nCompleted!")
    print(f"Total extracted: {saved_count} images")
    print(f"Images saved in: {output_dir}")


if __name__ == "__main__":
    video_path = r"E:\city_result2222.mp4"    # Video file path
    output_dir = "output"     # Output directory
    frames_per_second = 3     # Extract 3 frames per second
    
    extract_frames_from_video(video_path, output_dir, frames_per_second)