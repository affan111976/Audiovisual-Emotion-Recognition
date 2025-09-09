import torch
import torch.nn as nn
import torchvision.models as models

class VideoModel(nn.Module):
    def __init__(self, num_features=128):
        super(VideoModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)
        
        self.lstm = nn.LSTM(num_features, num_features, batch_first=True)

    def forward(self, x):
        batch_size, num_frames, h, w = x.shape
        x = x.unsqueeze(2) 
        
        features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :] 
            feature = self.resnet(frame)
            features.append(feature)
            
        features = torch.stack(features, 1) 
        lstm_out, _ = self.lstm(features)
        
        return lstm_out[:, -1, :]

class AudioModel(nn.Module):
    def __init__(self, num_features=128):
        super(AudioModel, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * 16 * 25, num_features)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv_stack(x)
        x = self.fc(x)
        return x

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=7, fusion_features=256):
        super(MultimodalModel, self).__init__()
        self.video_model = VideoModel()
        self.audio_model = AudioModel()
        
        # Fusion and classification layers
        self.fusion = nn.Linear(128 + 128, fusion_features) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(fusion_features, num_classes)

    def forward(self, video_x, audio_x):
        video_features = self.video_model(video_x)
        audio_features = self.audio_model(audio_x)
        
        combined_features = torch.cat((video_features, audio_features), dim=1)
        
        fused = self.fusion(combined_features)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        
        output = self.classifier(fused)
        return output