import os
import cv2
import json
from tqdm import tqdm

# Paths
DATASET_PATH = r"C:\Users\nandh\deepfake_project"
METADATA_FILE = os.path.join(DATASET_PATH, 'metadata.json')
VIDEO_PATH = os.path.join(DATASET_PATH, 'videos')
FRAME_PATH = os.path.join(DATASET_PATH, 'frames')
os.makedirs(FRAME_PATH, exist_ok=True)

# Step 3: Extract Frames from Videos
def extract_frames(video_dir, frame_dir):
    metadata = json.load(open(METADATA_FILE))
    for video_name, info in metadata.items():
        label = info['label']
        video_file = os.path.join(video_dir, video_name)
        label_dir = os.path.join(frame_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_file)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_id = 0
        saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_rate == 0:
                resized = cv2.resize(frame, (640, 360))
                filename = f"{video_name}_{frame_id}.jpg"
                cv2.imwrite(os.path.join(label_dir, filename), resized)
                saved += 1
            frame_id += 1
        cap.release()
        print(f"Extracted {saved} frames from {video_name}")


if __name__ == '__main__':
    extract_frames(VIDEO_PATH, FRAME_PATH)
