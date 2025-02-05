import torch 
import torch.nn as nn
import torch.optim as optim
from Models.XceptionLSTMA import XceptionLSTMA
from Dataset.audio_dataloader import get_audio_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_folder = "Dataset/processed_audio/train"
eval_folder = "Dataset/processed_audio/eval"

train_dataloader = get_audio_dataloader(train_folder, batch_size=8, shuffle=False)
eval_dataloader = get_audio_dataloader(eval_folder, batch_size=8, shuffle=False)

model = XceptionLSTMA(hidden_dim=512).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_eval_loss = float("inf")
early_stop_count = 0
patience = 10

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for audio_batch, labels in train_dataloader:
        audio_batch, labels = audio_batch.to(device), labels.to(device)

        # Extract features and forward pass
        features = model.module.extract_features(audio_batch, device) if isinstance(model, nn.DataParallel) else model.extract_features(audio_batch, device)
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_dataloader):.4f}")

    # Evaluation
    if (epoch + 1) % 10 == 0:
        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for audio_batch, labels in eval_dataloader:
                audio_batch, labels = audio_batch.to(device), labels.to(device)

                features = model.module.extract_features(audio_batch, device) if isinstance(model, nn.DataParallel) else model.extract_features(audio_batch, device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        eval_loss /= len(eval_dataloader)
        eval_accuracy = correct / total
        print(f"Evaluation Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

        # Learning rate scheduling
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(eval_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr < prev_lr:
            print(f"Learning rate reduced to {new_lr:.6f}")

        # Early stopping
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            early_stop_count = 0
            print("New best model found. Saving...")
            torch.save(model.state_dict(), "Checkpoints/best_model_audio.pth")
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("Early stopping triggered. Training stopped.")
            break

print("Training Finished!")