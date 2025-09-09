import torch
import cv2
import numpy as np
import librosa
import mediapipe as mp
from model import MultimodalModel

MODEL_PATH = 'saved_models/best_model.pth'
NUM_CLASSES = 7
FACE_SIZE = (224, 224)
VIDEO_FRAMES = 20
SAMPLE_RATE = 16000
N_MELS = 128

idx_to_emotion = {
    0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
    4: 'fearful', 5: 'disgust', 6: 'surprised'
}


def preprocess_single_video(video_path, device):
    """Preprocesses a single video file for inference using MediaPipe."""
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, VIDEO_FRAMES, dtype=int)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                if x < 0: x = 0
                if y < 0: y = 0
                face = frame[y:y+h, x:x+w]
                
                if face.size > 0:
                    face_resized = cv2.resize(face, FACE_SIZE)
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    frames.append(face_gray)
    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not extract any frames with faces.")
    
    while len(frames) < VIDEO_FRAMES:
        frames.append(frames[-1])
    
    video_data = np.array(frames)
    video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0) / 255.0

    y, sr = librosa.load(video_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel_spec.shape[1] < 200:
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 200 - log_mel_spec.shape[1])), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :200]
    audio_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)

    return video_tensor, audio_tensor


def predict(video_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = MultimodalModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    try:
        video_tensor, audio_tensor = preprocess_single_video(video_path, device)
        video_tensor, audio_tensor = video_tensor.to(device), audio_tensor.to(device)
        
        with torch.no_grad():
            output = model(video_tensor, audio_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_emotion = idx_to_emotion[predicted_idx.item()]
            
            print(f"Predicted Emotion: {predicted_emotion}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == '__main__':
    test_video_path = 'data/RAVDESS/Actor_01/01-01-03-02-02-02-01.mp4' 
    predict(test_video_path)