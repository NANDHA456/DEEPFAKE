import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

# Paths
DATASET_PATH = '/content/drive/MyDrive/deepfake_dataset'
FRAME_PATH = os.path.join(DATASET_PATH, 'frames')
CROPPED_PATH = os.path.join(DATASET_PATH, 'faces')
os.makedirs(CROPPED_PATH, exist_ok=True)

detector = MTCNN()

# Crop faces
def crop_faces_from_frames():
    for video_folder in tqdm(os.listdir(FRAME_PATH)):
        input_folder = os.path.join(FRAME_PATH, video_folder)
        output_folder = os.path.join(CROPPED_PATH, video_folder)
        os.makedirs(output_folder, exist_ok=True)

        for frame_file in os.listdir(input_folder):
            frame_path = os.path.join(input_folder, frame_file)
            image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(image)
            if results:
                x, y, w, h = results[0]['box']
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face_path = os.path.join(output_folder, frame_file)
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    crop_faces_from_frames()
