# 🎭 Deepfake Detection System

A deep learning-based deepfake detection system that processes video datasets to classify faces as REAL or FAKE. Built with face detection and CNN classification for robust performance.

## 🔍 Overview

This project implements an end-to-end pipeline for deepfake detection:
- Extract frames from video datasets  
- Detect and crop faces for consistent inputs
- Train a CNN-based binary classifier
- Provide real-time inference on images

**Dataset Used:** Celebrity Dataset containing **1000+ videos** with balanced REAL/FAKE samples.

## ✨ Features

- **Automated Video Processing**: Extracts frames from videos with metadata-driven labeling
- **Face Detection**: Precise face cropping for consistent model inputs  
- **Transfer Learning**: CNN pre-trained on ImageNet for superior feature extraction
- **Robust Training**: Early stopping, dropout regularization, and train/test splitting
- **Real-time Inference**: Single-image prediction with confidence scoring

## 🛠️ Technologies

- **Python 3.8+**
- **TensorFlow 2.x** 
- **MTCNN** (face detection)
- **EfficientNetB0** (classification backbone)
- **OpenCV** (video/image processing)
- **scikit-learn** (data utilities)

## 📁 Project Structure
```
deepfake-detection/
│
├── data/
│ ├── videos/ # Input video files
│ ├── frames/ # Extracted frames
│ ├── faces/ # Cropped faces
│ ├── split/ # Train/test datasets
│ └── metadata.json # Video labels
│
├── notebooks/
│ └── deepfake_detection.ipynb
│---deepfake_detector.py (integration of all files)
├── requirements.txt
└── README.md
```

## ⚙️ Installation

pip install tensorflow opencv-python mtcnn efficientnet scikit-learn numpy pandas

text

## 🚀 Usage

1. **Prepare Dataset**
```
metadata.json format
{
"video1.mp4": {"label": "REAL"},
"video2.mp4": {"label": "FAKE"}
}

```

2. **Run Pipeline**
 ```
Extract frames from videos
extract_frames(VIDEO_PATH, FRAME_PATH)

Crop faces
crop_faces(FRAME_PATH, CROPPED_PATH)

Create train/test split
split_path = prepare_split(CROPPED_PATH)

Train model
model = train_model(split_path)

Inference
predict_image("path/to/image.jpg", model)
```


## 🔧 Technical Details

### Pipeline
1. **Frame Extraction**: 1fps sampling from videos
2. **Face Detection**: Detect faces with bounding boxes
3. **Preprocessing**: Resize to 128×128, normalize to [0,1]
4. **Classification**: Binary REAL/FAKE prediction

### Model Architecture
- **Backbone**: EfficientNetB0 (ImageNet pretrained)
- **Head**: Global Average Pooling + Dense (Sigmoid)
- **Input**: 128×128×3 face crops
- **Output**: Binary probability (REAL/FAKE)

## 📈 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~85% |
| Model Size | 5.3M parameters |
| Inference Time | <100ms per image |
| Dataset | 1000+ videos |

## 💻 Example Usage
```
def predict_deepfake(image_path, model):
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)
```
```text
prediction = model.predict(img)
label = "FAKE" if prediction > 0.6 else "REAL"
confidence = max(prediction, 1-prediction)

return label, confidence
```

## 🎯 Applications

- Social media content verification
- News authenticity checking
- Security and anti-spoofing systems
- Research benchmarking

## 📄 License

MIT License

## 👤 Author

**Nandha Kumar V**  
Chennai Institute of Technology

---

*Efficient deepfake detection using deep learning and computer vision*
