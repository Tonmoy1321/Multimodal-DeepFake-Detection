# import os
# import time
# import torch
# import wandb
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from collections import Counter
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from torch.amp import autocast, GradScaler

# from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
# from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention

# # --- Metrics ---
# def compute_eer_auc(labels, scores):
#     fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2
#     return auc_score, pauc, eer

# # --- Setup ---
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# train_loader, test_loader, eval_loader = get_patch_image_loaders(
#     "Dataset/AU_Files/fakeavceleb_whole_image_patches", batch_size=2, image_size=128, max_frames=50
# )

# # --- Class Weights ---
# class_counts = Counter()
# for _, _, label in train_loader.dataset:
#     class_counts.update([label.item()])
# total = sum(class_counts.values())
# weights = torch.tensor([total / class_counts.get(0, 1), total / class_counts.get(1, 1)], dtype=torch.float32).to(device)

# # --- Model ---
# model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)

# # --- Loss & Optimizer ---
# class LabelSmoothingBCEWithLogitsLoss(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss()
#     def forward(self, logits, targets):
#         targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
#         return self.bce(logits, targets)

# criterion = LabelSmoothingBCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
# scaler = GradScaler()

# # --- WandB Config ---
# wandb.config.update({
#     "lr": 1e-4, "batch_size": 2, "loss": "Label Smoothing BCE",
#     "class_weight_real": weights[0].item(), "class_weight_fake": weights[1].item()
# })

# best_eval_loss = float("inf")
# early_stop_count = 0
# patience = 5

# # --- Training Loop ---
# for epoch in range(100):
#     model.train()
#     start_time = time.time()
#     train_loss, correct, total = 0.0, 0, 0
#     all_probs, all_labels = [], []

#     print(f"\nEpoch {epoch + 1}")
#     for patches, weights_batch, labels in tqdm(train_loader, desc="Training"):
#         patches, weights_batch, labels = patches.to(device), weights_batch.to(device), labels.to(device)

#         optimizer.zero_grad()
#         with autocast("cuda"):
#             outputs = model(patches, au_patch_weights=weights_batch).squeeze(-1)
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         probs = torch.sigmoid(outputs / 7.0).detach().cpu().numpy()
#         preds = (probs > 0.5).astype(float)
#         all_probs.extend(probs)
#         all_labels.extend(labels.cpu().numpy())
#         correct += (preds == labels.cpu().numpy()).sum()
#         total += labels.size(0)
#         train_loss += loss.item()

#     acc = correct / total
#     auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)
#     train_loss /= len(train_loader)

#     print(f"Train: Loss={train_loss:.4f}, Acc={acc:.4f}, AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}")
#     wandb.log({
#         "Train/Loss": train_loss, "Train/Accuracy": acc,
#         "Train/AUC": auc_score, "Train/pAUC": pauc_score, "Train/EER": eer,
#         "Train/Time": time.time() - start_time
#     })

#     # --- Evaluation ---
#     model.eval()
#     eval_loss, correct, total = 0.0, 0, 0
#     all_probs, all_labels = [], []

#     with torch.no_grad():
#         for patches, weights_batch, labels in tqdm(eval_loader, desc="Evaluation"):
#             patches, weights_batch, labels = patches.to(device), weights_batch.to(device), labels.to(device)
#             with autocast("cuda"):
#                 outputs = model(patches, au_patch_weights=weights_batch).squeeze(-1)
#                 loss = criterion(outputs, labels)

#             probs = torch.sigmoid(outputs / 2.0).detach().cpu().numpy()
#             preds = (probs > 0.5).astype(float)
#             all_probs.extend(probs)
#             all_labels.extend(labels.cpu().numpy())
#             correct += (preds == labels.cpu().numpy()).sum()
#             total += labels.size(0)
#             eval_loss += loss.item()

#     acc = correct / total
#     auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)
#     eval_loss /= len(eval_loader)

#     print(f"Eval: Loss={eval_loss:.4f}, Acc={acc:.4f}, AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}")
#     wandb.log({
#         "Eval/Loss": eval_loss, "Eval/Accuracy": acc,
#         "Eval/AUC": auc_score, "Eval/pAUC": pauc_score, "Eval/EER": eer
#     })

#     scheduler.step(eval_loss)
#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         early_stop_count = 0
#         torch.save(model.state_dict(), os.path.join("Checkpoints", "ff_au_patch_60fps_lstm_best_model_with_attn_5.pth"))
#         print("Model saved.")
#     else:
#         early_stop_count += 1
#         # if early_stop_count >= patience:
#         #     print("Early stopping triggered.")
#         #     break

# wandb.finish()
# print("Training Complete.")


import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.amp import autocast, GradScaler

from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention


# ---------------- Metrics ----------------
def compute_eer_auc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)
    pauc = (
        auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1
        if np.sum(fpr <= 0.1) >= 2 else 0.0
    )
    eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2
    return auc_score, pauc, eer


# ---------------- Device ----------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ---------------- Data ----------------
print("Loading data...")
train_loader, test_loader, eval_loader = get_patch_image_loaders(
    data_root="Dataset/AU_Files/fakeavceleb_whole_image_patches",
    mode="fakeavceleb",
    csv_path="Dataset/meta_data.csv",
    batch_size=2,
    image_size=128,
    max_frames=60,
    augment_train=True,   # balance + augment training
    augment_eval=True,   # keep eval clean
    augment_test=False    # keep test clean
)

# ---------------- Model ----------------
model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)


# ---------------- Loss ----------------
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


criterion = LabelSmoothingBCEWithLogitsLoss()

# ---------------- Optimizer / Scheduler ----------------
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
scaler = GradScaler()

# ---------------- Early stopping ----------------
best_eval_loss = float("inf")
early_stop_count = 0
patience = 5

# ---------------- Training loop ----------------
for epoch in range(100):
    model.train()
    start_time = time.time()
    train_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    print(f"\nEpoch {epoch + 1}")
    for patches, weights_batch, labels in tqdm(train_loader, desc="Training"):
        patches, weights_batch, labels = (
            patches.to(device),
            weights_batch.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(patches, au_patch_weights=weights_batch)
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.view(-1)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(outputs / 7.0).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels.cpu().numpy()).sum()
        total += labels.size(0)
        train_loss += loss.item()

    acc = correct / total
    auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)
    train_loss /= len(train_loader)
    print(
        f"Train: Loss={train_loss:.4f}, Acc={acc:.4f}, "
        f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}, "
        f"Time={time.time()-start_time:.2f}s"
    )

    # ---------------- Evaluation ----------------
    model.eval()
    eval_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for patches, weights_batch, labels in tqdm(eval_loader, desc="Evaluation"):
            patches, weights_batch, labels = (
                patches.to(device),
                weights_batch.to(device),
                labels.to(device),
            )

            with autocast("cuda"):
                outputs = model(patches, au_patch_weights=weights_batch)
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.view(-1)
                loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs / 2.0).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels.cpu().numpy()).sum()
            total += labels.size(0)
            eval_loss += loss.item()

    acc = correct / total
    auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)
    eval_loss /= len(eval_loader)
    print(
        f"Eval: Loss={eval_loss:.4f}, Acc={acc:.4f}, "
        f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}"
    )

    # ---------------- Scheduler / Early stopping ----------------
    scheduler.step(eval_loss)
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        early_stop_count = 0
        os.makedirs("Checkpoints", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("Checkpoints", "best_au_patch_model_last2.pth"))
        print("Model saved.")
    else:
        early_stop_count += 1
        print(f"Early stopping patience: {early_stop_count}/{patience}")
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

print("Training Complete.")



# import os
# import time
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from collections import Counter
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from torch.amp import autocast, GradScaler

# from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
# from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention


# # --- Metrics ---
# def compute_eer_auc(labels, scores):
#     labels = np.asarray(labels).astype(int).ravel()
#     scores = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2
#     return auc_score, pauc, eer


# def youden_threshold(labels, scores):
#     """Return threshold that maximizes TPR - FPR (Youden's J)."""
#     labels = np.asarray(labels).astype(int).ravel()
#     scores = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
#     j = tpr - fpr
#     j_idx = int(np.argmax(j))
#     return float(thr[j_idx])


# # --- Setup ---
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# # NEW: pass CSV path + toggle unmatched real handling to the loader
# train_loader, test_loader, eval_loader = get_patch_image_loaders(
#     data_root="Dataset/AU_Files/fakeavceleb_whole_image_patches",
#     csv_path="Dataset/meta_data.csv",
#     batch_size=2,
#     num_workers=2,
#     image_size=128,
#     max_frames=50,
#     max_aus=17,
#     include_unmatched_real=True,
#     unmatched_split_seed=42
# )

# # --- Class Weights (fast scan over labels with tqdm) ---
# # Use dataset.samples to avoid loading images here
# class_counts = Counter()
# for _, lbl in tqdm(getattr(train_loader.dataset, "samples", []), desc="Scanning train set for class counts"):
#     class_counts.update([int(lbl)])
# total = sum(class_counts.values())
# weights = torch.tensor(
#     [total / class_counts.get(0, 1), total / class_counts.get(1, 1)],
#     dtype=torch.float32
# ).to(device)

# # --- Model ---
# model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)

# # --- Loss & Optimizer ---
# class LabelSmoothingBCEWithLogitsLoss(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, logits, targets):
#         targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
#         return self.bce(logits, targets)

# criterion = LabelSmoothingBCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
# scaler = GradScaler()

# best_eval_loss = float("inf")
# early_stop_count = 0
# patience = 5

# os.makedirs("Checkpoints", exist_ok=True)


# def _normalize_logits_targets(raw_logits: torch.Tensor, raw_labels: torch.Tensor):
#     """
#     Make shapes safe for BCEWithLogitsLoss:
#       - logits → (-1,)
#       - labels → (-1,)
#     Works for logits shaped [], [1], [B], [B,1].
#     """
#     logits = raw_logits
#     labels = raw_labels

#     # Float labels
#     if labels.dtype not in (torch.float32, torch.float64):
#         labels = labels.float()

#     # Flatten labels to (B,)
#     labels = labels.view(-1)

#     # Fix logits shape robustly
#     if logits.dim() == 2 and logits.size(-1) == 1:
#         logits = logits.squeeze(-1)
#     elif logits.dim() == 0:  # scalar (e.g., batch size = 1 and over-squeezed)
#         logits = logits.unsqueeze(0)
#     logits = logits.view(-1)

#     if logits.shape != labels.shape:
#         raise RuntimeError(f"Shape mismatch after normalization: logits{tuple(logits.shape)} vs labels{tuple(labels.shape)}")

#     return logits, labels


# # --- Training Loop ---
# for epoch in range(100):
#     model.train()
#     start_time = time.time()
#     train_loss = 0.0
#     all_probs, all_labels = [], []

#     print(f"\nEpoch {epoch + 1}")
#     for patches, weights_batch, labels in tqdm(train_loader, desc="Training"):
#         patches, weights_batch, labels = patches.to(device), weights_batch.to(device), labels.to(device).float()

#         optimizer.zero_grad(set_to_none=True)
#         with autocast(device_type="cuda", enabled=(device.type == "cuda")):
#             outputs = model(patches, au_patch_weights=weights_batch)
#             logits, targets = _normalize_logits_targets(outputs, labels)
#             loss = criterion(logits, targets)

#         scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         # collect probs/labels for Youden-based accuracy (keep your train-time scaling)
#         probs = torch.sigmoid((logits / 7.0).float()).detach().cpu().numpy()
#         all_probs.extend(probs.tolist())
#         all_labels.extend(targets.detach().cpu().numpy().tolist())
#         train_loss += float(loss.item())

#     # --- Train metrics (using Youden's threshold) ---
#     train_loss /= max(1, len(train_loader))
#     thr_train = youden_threshold(all_labels, all_probs)
#     preds_train = (np.asarray(all_probs) >= thr_train).astype(int)
#     acc_train = float((preds_train == np.asarray(all_labels).astype(int)).mean())
#     auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)

#     print(f"Train: Loss={train_loss:.4f}, Acc@Youden={acc_train:.4f}, "
#           f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}, thr*={thr_train:.3f}")

#     # --- Evaluation ---
#     model.eval()
#     eval_loss = 0.0
#     all_probs, all_labels = [], []

#     with torch.no_grad():
#         for patches, weights_batch, labels in tqdm(eval_loader, desc="Evaluation"):
#             patches, weights_batch, labels = patches.to(device), weights_batch.to(device), labels.to(device).float()
#             with autocast(device_type="cuda", enabled=(device.type == "cuda")):
#                 outputs = model(patches, au_patch_weights=weights_batch)
#                 logits, targets = _normalize_logits_targets(outputs, labels)
#                 loss = criterion(logits, targets)

#             # collect probs/labels for Youden (keep your eval-time scaling)
#             probs = torch.sigmoid((logits / 2.0).float()).detach().cpu().numpy()
#             all_probs.extend(probs.tolist())
#             all_labels.extend(targets.detach().cpu().numpy().tolist())
#             eval_loss += float(loss.item())

#     # --- Eval metrics (Youden) ---
#     eval_loss /= max(1, len(eval_loader))
#     thr_eval = youden_threshold(all_labels, all_probs)
#     preds_eval = (np.asarray(all_probs) >= thr_eval).astype(int)
#     acc_eval = float((preds_eval == np.asarray(all_labels).astype(int)).mean())
#     auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)

#     print(f"Eval:  Loss={eval_loss:.4f}, Acc@Youden={acc_eval:.4f}, "
#           f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}, thr*={thr_eval:.3f}")

#     # --- LR schedule & checkpointing (unchanged logic) ---
#     scheduler.step(eval_loss)
#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         early_stop_count = 0
#         torch.save(model.state_dict(), os.path.join("Checkpoints", "ff_au_patch_60fps_lstm_best_model_with_attn_5.pth"))
#         print("Model saved.")
#     else:
#         early_stop_count += 1
#         # if early_stop_count >= patience:
#         #     print("Early stopping triggered.")
#         #     break

# print("Training Complete.")