import os, json, shutil, random
from tqdm import tqdm

# Paths
DATASET_PATH = r"C:\Users\nandh\deepfake_project"
CROPPED_PATH = os.path.join(DATASET_PATH, 'faces')
METADATA_FILE = os.path.join(DATASET_PATH, 'metadata.json')
SPLIT_PATH = os.path.join(DATASET_PATH, 'dataset_split')

TRAIN_PATH = os.path.join(SPLIT_PATH, 'train')
VAL_PATH   = os.path.join(SPLIT_PATH, 'val')
TEST_PATH  = os.path.join(SPLIT_PATH, 'test')

for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    for label in ['REAL', 'FAKE']:
        os.makedirs(os.path.join(path, label), exist_ok=True)

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

# Organize by label
real_videos = []
fake_videos = []

for video, info in metadata.items():
    if info['label'] == 'REAL':
        real_videos.append(video.split('.')[0])
    elif info['label'] == 'FAKE':
        fake_videos.append(video.split('.')[0])

# Balance classes
min_len = min(len(real_videos), len(fake_videos))
real_videos = real_videos[:min_len]
fake_videos = fake_videos[:min_len]

# Shuffle and split
random.seed(42)
all_videos = real_videos + fake_videos
random.shuffle(all_videos)

train_split = int(0.7 * len(all_videos))
val_split   = int(0.9 * len(all_videos))

train_videos = all_videos[:train_split]
val_videos   = all_videos[train_split:val_split]
test_videos  = all_videos[val_split:]

# Copy cropped faces to respective folders
def copy_data(video_list, target_path):
    for video in tqdm(video_list):
        label = 'REAL' if video in real_videos else 'FAKE'
        src_folder = os.path.join(CROPPED_PATH, video)
        dst_folder = os.path.join(target_path, label)
        if os.path.exists(src_folder):
            for file in os.listdir(src_folder):
                src = os.path.join(src_folder, file)
                dst = os.path.join(dst_folder, f"{video}_{file}")
                shutil.copy(src, dst)

copy_data(train_videos, TRAIN_PATH)
copy_data(val_videos, VAL_PATH)
copy_data(test_videos, TEST_PATH)
