import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, metadata_file):
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_path = self.metadata.iloc[idx]['video_file']
        audio_path = self.metadata.iloc[idx]['audio_file']
        label = self.metadata.iloc[idx]['label']

        video_frames = np.load(video_path)
        video_tensor = torch.tensor(video_frames, dtype=torch.float32) / 255.0

        spectrogram = np.load(audio_path)
        audio_tensor = torch.tensor(spectrogram, dtype=torch.float32)
        
        return video_tensor, audio_tensor, torch.tensor(label, dtype=torch.long)