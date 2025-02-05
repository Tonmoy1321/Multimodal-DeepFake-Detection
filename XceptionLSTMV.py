import os
import numpy as np
import torch
import torch.nn as nn
from .Xception import xception



class XceptionLSTMV(nn.Module):
    def __init__(self, hidden_dim):
        super(XceptionLSTMV, self).__init__()
        self.feature_extractor = xception(pretrained=True)
        self.feature_extractor.fc = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  

        self.lstm = nn.LSTM(
            input_size=2048,  
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

    def extract_features(self, video_batch, device):
        """
        Extracts frame-wise features from a batch of videos.
        """
        self.feature_extractor.to(device)
        
        batch_size, seq_len, c, h, w = video_batch.shape 

        # Reshape to process frames individually
        frames = video_batch.view(batch_size * seq_len, c, h, w)  # Shape: (batch_size * seq_len, 3, 256, 256)

        # Pass frames through feature extractor
        frame_features = self.feature_extractor(frames)  # Shape: (batch_size * seq_len, 2048)

        # Reshape back to (batch_size, seq_len, feature_dim)
        features = frame_features.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, 2048)
        
        return features


    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Take only last time step (video-level prediction)
        dense_out = self.fc_layers(lstm_out)
        return self.sigmoid(self.fc_out(dense_out))  # Shape: (batch_size, 1)

