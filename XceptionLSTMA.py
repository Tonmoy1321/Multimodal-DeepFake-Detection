import torch.nn as nn
import torch.nn.functional as F
from .Xception import xception  # Ensure this import is correct

class XceptionLSTMA(nn.Module):
    def __init__(self, hidden_dim):
        super(XceptionLSTMA, self).__init__()
        self.feature_extractor = xception(pretrained=True)
        self.feature_extractor.fc = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze Xception parameters

        self.lstm = nn.LSTM(
            input_size=2048,  # Xception output size
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_out = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def extract_features(self, audio_batch, device):
        self.feature_extractor.to(device)
        
        batch_size, time_steps, c, n_mfcc = audio_batch.shape  

        # Reshape MFCC features to match Xception input requirements
        frames = audio_batch.view(batch_size * time_steps, c, n_mfcc, 1)  # Shape: (batch_size * time_steps, 3, 13, 1)
        frames = F.interpolate(frames, size=(64, 64), mode="bilinear", align_corners=False)  # Resize to 64x64

        frame_features = self.feature_extractor(frames)  # Extract features using Xception
        feature_dim = frame_features.shape[-1]

        features = frame_features.view(batch_size, time_steps, feature_dim)  # Reshape to (batch_size, time_steps, feature_dim)

        return features

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Use the final hidden state
        dense_out = self.fc_layers(lstm_out)
        return self.sigmoid(self.fc_out(dense_out))