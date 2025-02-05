import torch
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from Models.XceptionLSTMV import XceptionLSTMV
from Dataset.video_dataloader import get_face_dataloader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_folder = "Dataset/processed_FAVC/test"
test_dataloader = get_face_dataloader(test_folder, batch_size=8, shuffle=False)

# Load the best saved model
model_path = "Checkpoints/best_model.pth"  
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

# Initialize model
model = XceptionLSTMV(hidden_dim=512).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Loss function
criterion = nn.BCELoss()

# Evaluation
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for video_batch, labels in tqdm(test_dataloader, desc="Evaluating", leave=True):
        video_batch, labels = video_batch.to(device), labels.to(device)

        # Extract Features
        features = model.module.extract_features(video_batch, device) if isinstance(model, nn.DataParallel) else model.extract_features(video_batch, device)

        # Forward Pass
        outputs = model(features)

        # Compute Loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Compute Accuracy
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Compute final accuracy and loss
test_loss /= len(test_dataloader)
test_accuracy = correct / total

# Print results
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{total} correct)")