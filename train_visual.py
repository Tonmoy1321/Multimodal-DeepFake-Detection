# import os
# import time
# import torch
# import wandb
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
# from collections import Counter
# from torch.amp import autocast, GradScaler
# from Models.XceptionLSTMV import XceptionLSTMV
# from Dataset.video_dataloader import get_face_dataloader

# # Initialize WandB
# wandb.init(project="FAVC_Video_Balanced_Training_evm", resume=True)

# # Device setup
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Dataset paths
# train_folder = "Dataset/FAVC_Full_evm/frames/train"
# eval_folder = "Dataset/FAVC_Full_evm/frames/eval"

# train_dataloader = get_face_dataloader(train_folder, batch_size=8, augment_minority=True, shuffle=True)
# eval_dataloader = get_face_dataloader(eval_folder, batch_size=8, augment_minority=True, shuffle=False)

# # Compute class distribution dynamically
# class_counts = Counter()
# for _, labels, _ in train_dataloader:
#     class_counts.update(labels.cpu().numpy().astype(int))

# num_real = class_counts.get(0, 1)
# num_fake = class_counts.get(1, 1)
# total_samples = num_real + num_fake

# # Compute dynamic class weights
# real_weight = total_samples / num_real
# fake_weight = total_samples / num_fake

# class_weight = torch.tensor([real_weight, fake_weight], dtype=torch.float32).to(device)

# # Model
# model = XceptionLSTMV(hidden_dim=128).to(device)

# # Custom Loss Functions
# class LabelSmoothingBCEWithLogitsLoss(nn.Module):
#     def __init__(self, smoothing=0.5):   # was 0.1
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, logits, targets):
#         targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
#         return self.bce(logits, targets)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=0.5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, targets):
#         bce_loss = self.bce(logits, targets)
#         probas = torch.sigmoid(logits)
#         focal_weight = (1 - probas) ** self.gamma
#         return (self.alpha * focal_weight * bce_loss).mean()

# # Select the loss function
# criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.5)

# # Optimizer, Scheduler, and Gradient Scaler
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
# scaler = GradScaler()

# # Early stopping parameters
# best_eval_loss = float("inf")
# early_stop_count = 0
# patience = 6

# # Log hyperparameters
# wandb.config.update({
#     "learning_rate": 0.0001,
#     "batch_size": 8,
#     "optimizer": "Adam",
#     "loss_function": "Label Smoothing BCE",
#     "scheduler": "ReduceLROnPlateau",
#     "hidden_dim": 128,
#     "patience": patience,
#     "class_weight_real": real_weight,
#     "class_weight_fake": fake_weight
# })

# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     start_time = time.time()

#     correct_train = 0
#     total_train = 0

#     print(f"\nEpoch {epoch + 1}/{num_epochs}")
#     for video_batch, labels, seq_lengths in tqdm(train_dataloader, desc="Training Batches", leave=True):
#         video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

#         optimizer.zero_grad()
#         with autocast("cuda"):
#             features = model.extract_features(video_batch, seq_lengths)
#             outputs = model(features, seq_lengths).squeeze(-1)

#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()

#         probabilities = torch.sigmoid(outputs / 7.0)  # Temperature Scaling was 2.0
#         predicted = (probabilities > 0.5).float()
#         correct_train += (predicted == labels).sum().item()
#         total_train += labels.size(0)

#     train_loss = running_loss / len(train_dataloader)
#     train_accuracy = correct_train / total_train if total_train > 0 else 0
#     epoch_time = time.time() - start_time

#     print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Time: {epoch_time:.2f}s")
    
#     wandb.log({
#         "Loss/Train": train_loss,
#         "Accuracy/Train": train_accuracy,
#         "Epoch Time": epoch_time
#     })

#     # Evaluation loop
#     model.eval()
#     eval_loss = 0.0
#     correct_real, correct_fake, total_real, total_fake = 0, 0, 0, 0

#     with torch.no_grad():
#         for video_batch, labels, seq_lengths in tqdm(eval_dataloader, desc="Evaluation Batches", leave=True):
#             video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

#             with autocast("cuda"):
#                 features = model.extract_features(video_batch, seq_lengths)
#                 outputs = model(features, seq_lengths).squeeze(-1)

#                 loss = criterion(outputs, labels)
#                 eval_loss += loss.item()

#                 probabilities = torch.sigmoid(outputs / 2.0)
#                 predicted = (probabilities > 0.5).float()

#                 correct_real += ((predicted == 0) & (labels == 0)).sum().item()
#                 correct_fake += ((predicted == 1) & (labels == 1)).sum().item()
#                 total_real += (labels == 0).sum().item()
#                 total_fake += (labels == 1).sum().item()

#     eval_loss /= len(eval_dataloader)
#     eval_accuracy = (correct_real + correct_fake) / (total_real + total_fake)

#     print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
    
#     wandb.log({
#         "Loss/Eval": eval_loss,
#         "Accuracy/Eval": eval_accuracy,
#         "Correct Real": correct_real,
#         "Correct Fake": correct_fake
#     })

#     scheduler.step(eval_loss)

#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         early_stop_count = 0
#         print("New best model found. Saving...")
#         torch.save(model.state_dict(), os.path.join("Checkpoints", "FAVC_full_evm.pth"))
#     else:
#         early_stop_count += 1

#     if early_stop_count >= patience:
#         print("Early stopping triggered. Training stopped.")
#         break

# wandb.finish()
# print("Training Finished!")



# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from collections import Counter
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader
# from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, auc
# import torch.multiprocessing as mp

# from Models.XceptionLSTMV import XceptionLSTMV
# from Dataset.video_dataloader_enhanced import get_face_dataloader, collate_fn


# # ---------------- Device ----------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # ---------------- Metrics ----------------
# def compute_metrics(labels, probs):
#     if len(np.unique(labels)) <= 1:
#         return 0.0, 0.0, 0.0, 1.0, 0.5, 0.5
#     auc_score = roc_auc_score(labels, probs)
#     ap_score = average_precision_score(labels, probs)
#     fpr, tpr, thresholds = roc_curve(labels, probs)
#     pauc_score = (
#         auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1
#         if np.sum(fpr <= 0.1) >= 2
#         else 0.0
#     )
#     fnr = 1 - tpr
#     abs_diffs = np.abs(fpr - fnr)
#     eer_idx = np.nanargmin(abs_diffs)
#     eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
#     eer_thresh = thresholds[eer_idx]
#     youden_idx = np.argmax(tpr - fpr)
#     youden_thresh = thresholds[youden_idx]
#     return auc_score, pauc_score, ap_score, eer, eer_thresh, youden_thresh


# def main():
#     # ---------------- Load Data ----------------
#     train_folder = "/media/rt0706/Lab/LAV-DF"
#     eval_folder = "/media/rt0706/Lab/LAV-DF"
#     lavdf_json = "Dataset/LAV-DF/metadata.json"

#     print("Loading training data...")
#     train_dataset = get_face_dataloader(
#         folder_path=train_folder,
#         mode="lavdf_raw",
#         subset="train",
#         lavdf_json=lavdf_json,
#         batch_size=1,
#         augment_minority=False,
#         shuffle=False,
#         raw_video=True,
#         use_face_detection=True,
#         frame_size=(224, 224),
#         max_frames=50,
#     ).dataset

#     print("Loading eval data...")
#     eval_dataset = get_face_dataloader(
#         folder_path=eval_folder,
#         mode="lavdf_raw",
#         subset="dev",
#         lavdf_json=lavdf_json,
#         batch_size=1,
#         augment_minority=False,
#         shuffle=False,
#         raw_video=True,
#         use_face_detection=True,
#         frame_size=(224, 224),
#         max_frames=50,
#     ).dataset

#     # ---------------- Fast Class Counting ----------------
#     print("Counting training class distribution (fast)...")
#     labels = [lbl for _, lbl, _ in train_dataset.samples]
#     class_counts = Counter(labels)
#     # print("Class counts:", class_counts)

#     # ---------------- Model ----------------
#     model = XceptionLSTMV(hidden_dim=128).to(device)

#     # ---------------- Loss, Optimizer, Scheduler ----------------
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=3
#     )
#     scaler = GradScaler()

#     # ---------------- Early Stopping ----------------
#     best_eval_loss = float("inf")
#     best_eer = float("inf")
#     early_stop_count = 0
#     patience = 6

#     # ---------------- Training ----------------
#     num_epochs = 50
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=2,
#         collate_fn=collate_fn,
#     )
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=2,
#         collate_fn=collate_fn,
#     )

#     epoch_bar = tqdm(range(num_epochs), desc="Epochs")
#     for epoch in epoch_bar:
#         model.train()
#         running_loss = 0.0
#         all_probs, all_labels = [], []
#         correct_real, correct_fake, total_real, total_fake = 0, 0, 0, 0

#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         for video_batch, labels, seq_lengths in tqdm(
#             train_dataloader, desc=f"Training Batches (Epoch {epoch+1})"
#         ):
#             video_batch, labels, seq_lengths = (
#                 video_batch.to(device),
#                 labels.to(device),
#                 seq_lengths.to(device),
#             )

#             optimizer.zero_grad()
#             with autocast("cuda"):
#                 features = model.extract_features(video_batch, seq_lengths)
#                 outputs = model(features, seq_lengths).squeeze(-1)
#                 outputs = torch.clamp(outputs, -10, 10)
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item()
#             probs = torch.sigmoid(outputs).detach().cpu().numpy()
#             preds = (probs > 0.5).astype(int)

#             all_probs.extend(probs)
#             all_labels.extend(labels.cpu().numpy())
#             correct_real += ((preds == 0) & (labels.cpu().numpy() == 0)).sum()
#             total_real += (labels.cpu().numpy() == 0).sum()
#             correct_fake += ((preds == 1) & (labels.cpu().numpy() == 1)).sum()
#             total_fake += (labels.cpu().numpy() == 1).sum()

#         train_loss = running_loss / max(1, len(train_dataloader))
#         train_auc, train_pauc, train_ap, train_eer, _, _ = compute_metrics(
#             np.array(all_labels), np.array(all_probs)
#         )
#         train_acc = (correct_real + correct_fake) / (total_real + total_fake + 1e-6)
#         print(
#             f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
#             f"AUC={train_auc:.4f}, pAUC={train_pauc:.4f}, AP={train_ap:.4f}, EER={train_eer:.4f}"
#         )
#         print(
#             f"Train Correct Real: {correct_real}/{total_real} | Correct Fake: {correct_fake}/{total_fake}"
#         )

#         # ---------------- Evaluation ----------------
#         model.eval()
#         eval_loss = 0.0
#         all_probs, all_labels = [], []
#         correct_real, correct_fake, total_real, total_fake = 0, 0, 0, 0

#         with torch.no_grad():
#             for video_batch, labels, seq_lengths in tqdm(
#                 eval_dataloader, desc=f"Evaluation Batches (Epoch {epoch+1})"
#             ):
#                 video_batch, labels, seq_lengths = (
#                     video_batch.to(device),
#                     labels.to(device),
#                     seq_lengths.to(device),
#                 )
#                 with autocast("cuda"):
#                     features = model.extract_features(video_batch, seq_lengths)
#                     outputs = model(features, seq_lengths).squeeze(-1)
#                     outputs = torch.clamp(outputs, -10, 10)
#                     loss = criterion(outputs, labels)
#                 eval_loss += loss.item()
#                 probs = torch.sigmoid(outputs).detach().cpu().numpy()
#                 preds = (probs > 0.5).astype(int)
#                 all_probs.extend(probs)
#                 all_labels.extend(labels.cpu().numpy())
#                 correct_real += ((preds == 0) & (labels.cpu().numpy() == 0)).sum()
#                 total_real += (labels.cpu().numpy() == 0).sum()
#                 correct_fake += ((preds == 1) & (labels.cpu().numpy() == 1)).sum()
#                 total_fake += (labels.cpu().numpy() == 1).sum()

#         eval_loss /= max(1, len(eval_dataloader))
#         eval_auc, eval_pauc, eval_ap, eval_eer, eer_thresh, youden_thresh = compute_metrics(
#             np.array(all_labels), np.array(all_probs)
#         )
#         eval_acc = (correct_real + correct_fake) / (total_real + total_fake + 1e-6)
#         print(
#             f"Eval: Loss={eval_loss:.4f}, Acc={eval_acc:.4f}, AUC={eval_auc:.4f}, "
#             f"pAUC={eval_pauc:.4f}, AP={eval_ap:.4f}, EER={eval_eer:.4f}, "
#             f"EER_thresh={eer_thresh:.4f}, Youden_thresh={youden_thresh:.4f}"
#         )
#         print(
#             f"Eval Correct Real: {correct_real}/{total_real} | Correct Fake: {correct_fake}/{total_fake}"
#         )

#         scheduler.step(eval_loss)

#         if eval_loss < best_eval_loss and eval_eer < best_eer:
#             best_eval_loss = eval_loss
#             best_eer = eval_eer
#             early_stop_count = 0
#             print("New best model (loss + eer). Saving...")
#             os.makedirs("Checkpoints", exist_ok=True)
#             torch.save(
#                 model.state_dict(),
#                 os.path.join("Checkpoints", f"NormalTraining_LAVDFRAW_Best.pth"),
#             )
#         else:
#             early_stop_count += 1
#             print(f"Current Early Stopping Count:{early_stop_count}")
#         if early_stop_count >= patience:
#             print("Early stopping triggered. Training stopped.")
#             break

#     print("Normal Training Finished!")


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     main()



import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, auc
import torch.multiprocessing as mp

from Models.XceptionLSTMV import XceptionLSTMV
from Dataset.video_dataloader_enhanced import get_face_dataloader, collate_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        x = nn.functional.normalize(features)
        W = nn.functional.normalize(self.weight)
        cos_theta = torch.matmul(x, W.t())
        if labels is None:
            return self.s * cos_theta
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        return self.s * output

def compute_metrics(labels, probs):
    if len(np.unique(labels)) <= 1:
        return 0.0, 0.0, 0.0, 1.0, 0.5
    auc_score = roc_auc_score(labels, probs)
    ap_score = average_precision_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    pauc_score = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_idx = np.nanargmin(abs_diffs)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return auc_score, pauc_score, ap_score, eer, thresholds[eer_idx]

def main():
    train_folder = "/media/rt0706/Lab/LAV-DF"
    eval_folder = "/media/rt0706/Lab/LAV-DF"
    lavdf_json = "Dataset/LAV-DF/metadata.json"

    print("Loading training data...")
    train_dataset = get_face_dataloader(
        folder_path=train_folder,
        mode="lavdf_raw",
        subset="train",
        lavdf_json=lavdf_json,
        batch_size=1,
        augment_minority=False,
        shuffle=False,
        raw_video=True,
        use_face_detection=True,
        frame_size=(224, 224),
        max_frames=50,
    ).dataset

    print("Loading eval data...")
    eval_dataset = get_face_dataloader(
        folder_path=eval_folder,
        mode="lavdf_raw",
        subset="dev",
        lavdf_json=lavdf_json,
        batch_size=1,
        augment_minority=False,
        shuffle=False,
        raw_video=True,
        use_face_detection=True,
        frame_size=(224, 224),
        max_frames=50,
    ).dataset

    print("Counting training class distribution...")
    labels = [lbl for _, lbl, _ in train_dataset.samples]
    class_counts = Counter(labels)
    print("Class counts:", class_counts)

    model = XceptionLSTMV(hidden_dim=128).to(device)
    arcface_head = ArcFaceHead(128, 2, s=30.0, m=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(arcface_head.parameters()), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler()

    best_eval_loss = float("inf")
    best_eer = float("inf")
    patience = 6
    early_stop_count = 0

    num_epochs = 50
    freeze_epochs = 3

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        if epoch < freeze_epochs:
            for p in model.feature_extractor.parameters():
                p.requires_grad = False
        else:
            for p in model.feature_extractor.parameters():
                p.requires_grad = True

        model.train()
        arcface_head.train()
        running_loss, all_probs, all_labels = 0.0, [], []
        correct_real = correct_fake = total_real = total_fake = 0

        for video_batch, labels, seq_lengths in tqdm(train_loader, desc="Training"):
            video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                features = model.extract_features(video_batch, seq_lengths)
                embeddings = model.lstm(features)[0][:, -1, :]
                labels_long = labels.long()
                logits = arcface_head(embeddings, labels_long)
                loss = criterion(logits, labels_long)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(arcface_head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            correct_real += ((preds == 0) & (labels.cpu().numpy() == 0)).sum()
            total_real += (labels.cpu().numpy() == 0).sum()
            correct_fake += ((preds == 1) & (labels.cpu().numpy() == 1)).sum()
            total_fake += (labels.cpu().numpy() == 1).sum()

        train_loss = running_loss / len(train_loader)
        train_auc, train_pauc, train_ap, train_eer, _ = compute_metrics(np.array(all_labels), np.array(all_probs))
        train_acc = (correct_real + correct_fake) / (total_real + total_fake + 1e-6)
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}, pAUC={train_pauc:.4f}, AP={train_ap:.4f}, EER={train_eer:.4f}")
        print(f"Train Correct Real: {correct_real}/{total_real} | Correct Fake: {correct_fake}/{total_fake}")

        model.eval()
        arcface_head.eval()
        eval_loss, all_probs, all_labels = 0.0, [], []
        correct_real = correct_fake = total_real = total_fake = 0

        with torch.no_grad():
            for video_batch, labels, seq_lengths in tqdm(eval_loader, desc="Evaluation"):
                video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)
                with autocast("cuda"):
                    features = model.extract_features(video_batch, seq_lengths)
                    embeddings = model.lstm(features)[0][:, -1, :]
                    labels_long = labels.long()
                    logits = arcface_head(embeddings, labels_long)
                    loss = criterion(logits, labels_long)
                eval_loss += loss.item()
                probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                correct_real += ((preds == 0) & (labels.cpu().numpy() == 0)).sum()
                total_real += (labels.cpu().numpy() == 0).sum()
                correct_fake += ((preds == 1) & (labels.cpu().numpy() == 1)).sum()
                total_fake += (labels.cpu().numpy() == 1).sum()

        eval_loss /= len(eval_loader)
        eval_auc, eval_pauc, eval_ap, eval_eer, _ = compute_metrics(np.array(all_labels), np.array(all_probs))
        eval_acc = (correct_real + correct_fake) / (total_real + total_fake + 1e-6)
        print(f"Eval: Loss={eval_loss:.4f}, Acc={eval_acc:.4f}, AUC={eval_auc:.4f}, pAUC={eval_pauc:.4f}, AP={eval_ap:.4f}, EER={eval_eer:.4f}")
        print(f"Eval Correct Real: {correct_real}/{total_real} | Correct Fake: {correct_fake}/{total_fake}")

        scheduler.step(eval_loss)

        if eval_loss < best_eval_loss and eval_eer < best_eer:
            best_eval_loss, best_eer = eval_loss, eval_eer
            early_stop_count = 0
            os.makedirs("Checkpoints", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "arcface": arcface_head.state_dict()},
                os.path.join("Checkpoints", "XceptionLSTMV_ArcFace_Best.pth"),
            )
            print("New best model saved.")
        else:
            early_stop_count += 1
            print(f"Early stopping counter: {early_stop_count}/{patience}")
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()