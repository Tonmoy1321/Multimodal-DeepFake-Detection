import torch
import torch.optim as optim
import torch.nn as nn
from Models.XceptionLSTMV import XceptionLSTMV
from Dataset.video_dataloader import get_face_dataloader


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_folder = "Dataset/processed_FAVC/train"
eval_folder = "Dataset/processed_FAVC/eval"

train_dataloader = get_face_dataloader(train_folder, batch_size=2, shuffle=False)
eval_dataloader = get_face_dataloader(eval_folder, batch_size=2, shuffle=False)

# Initialize model
model = XceptionLSTMV(hidden_dim=512).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early stopping parameters
best_eval_loss = float("inf")
early_stop_count = 0
patience = 10

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for video_batch, labels in train_dataloader:
        video_batch, labels = video_batch.to(device), labels.to(device)

        # Extract Features
        features = model.module.extract_features(video_batch, device) if isinstance(model, nn.DataParallel) else model.extract_features(video_batch, device)

        # Forward Pass
        outputs = model(features)

        # Compute Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_dataloader):.4f}")

    # Evaluation Every 10 Epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for video_batch, labels in eval_dataloader:
                video_batch, labels = video_batch.to(device), labels.to(device)

                # Extract Features
                features = model.module.extract_features(video_batch, device) if isinstance(model, nn.DataParallel) else model.extract_features(video_batch, device)

                # Forward Pass
                outputs = model(features)

                # Compute Loss
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

                # Compute Accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        eval_loss /= len(eval_dataloader)
        eval_accuracy = correct / total
        print(f"Evaluation Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

        # Update learning rate based on evaluation loss
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(eval_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Print learning rate update manually
        if new_lr < prev_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        # Save best model based on evaluation loss
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            early_stop_count = 0
            print("New best model found. Saving...")
            torch.save(model.state_dict(), "Checkpoints/best_model_FAVC.pth")
        else:
            early_stop_count += 1

        # Early stopping condition
        if early_stop_count >= patience:
            print("Early stopping triggered. Training stopped.")
            break

print("Training Finished!")