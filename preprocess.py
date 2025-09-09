import os
import cv2
import librosa
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import mediapipe as mp

DATA_DIR = 'data/RAVDESS'
PROCESSED_VIDEO_DIR = 'processed_data/video'
PROCESSED_AUDIO_DIR = 'processed_data/audio'
FACE_SIZE = (224, 224)
VIDEO_FRAMES = 20 
SAMPLE_RATE = 16000
N_MELS = 128

emotions_to_use = {
    '01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry',
    '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

emotion_labels_map = {label: i for i, label in enumerate(emotions_to_use.values())}

def preprocess_data():
    """
    Main function to preprocess video and audio data.
    It iterates through the RAVDESS dataset, extracts faces and spectrograms,
    and saves them as NumPy arrays along with their labels.
    """
    os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

    metadata = []

    actor_folders = [d for d in os.listdir(DATA_DIR) if d.startswith('Actor_')]
    for actor_folder in tqdm(actor_folders, desc="Processing Actors"):
        actor_path = os.path.join(DATA_DIR, actor_folder)
        for filename in os.listdir(actor_path):
            if filename.endswith('.mp4'):
                file_path = os.path.join(actor_path, filename)
                parts = filename.split('-')
                emotion_code = parts[2]
                
                if emotion_code not in emotions_to_use:
                    continue
                emotion_label = emotions_to_use[emotion_code]
                label_idx = emotion_labels_map[emotion_label]


                cap = cv2.VideoCapture(file_path)
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
                if len(frames) > 0:
                    print(f"SUCCESS: Video processed for {filename}") 
                    while len(frames) < VIDEO_FRAMES:
                        frames.append(frames[-1])

                    frames = frames[:VIDEO_FRAMES]

                    video_data = np.array(frames)
                    video_filename = os.path.join(PROCESSED_VIDEO_DIR, f"{os.path.splitext(filename)[0]}.npy")
                    np.save(video_filename, video_data)
                else:
                    print(f"Warning: Skipped video {filename}, no faces detected at all.")
                    continue

                try:
                    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    
                    if log_mel_spectrogram.shape[1] < 200:
                        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, 200 - log_mel_spectrogram.shape[1])), mode='constant')
                    else:
                        log_mel_spectrogram = log_mel_spectrogram[:, :200]
                    
                    audio_filename = os.path.join(PROCESSED_AUDIO_DIR, f"{os.path.splitext(filename)[0]}.npy")
                    np.save(audio_filename, log_mel_spectrogram)
                    print(f"SUCCESS: Audio processed for {filename}")

                except Exception as e:
                    print(f"FAILURE: Could not process AUDIO for {filename}. Error: {e}")
                    continue

                metadata.append({
                    'video_file': video_filename,
                    'audio_file': audio_filename,
                    'label': label_idx
                })

    df = pd.DataFrame(metadata)
    df.to_csv('processed_data/metadata.csv', index=False)
    print("Preprocessing complete!")

if __name__ == '__main__':
    preprocess_data()