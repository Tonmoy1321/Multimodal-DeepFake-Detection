# import os
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# import numpy as np
# from Models.XceptionLSTMV import XceptionLSTMV
# from Dataset.video_dataloader import get_face_dataloader

# # Device setup
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Load test dataset
# test_folder = "Dataset/FAVC_Full_evm/frames/test"
# test_dataloader = get_face_dataloader(test_folder, batch_size=1, augment_minority=True, shuffle=False, sample_percentage=1.0)

# # Load the best saved model
# model_path = "Checkpoints/best_model_FAVC_Whole_8th.pth"
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

# # Initialize model
# model = XceptionLSTMV(hidden_dim=128).to(device)

# # Load model weights
# model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
# model.eval()

# # Loss function
# criterion = nn.BCEWithLogitsLoss()

# # Evaluation metrics
# correct = 0
# total = 0
# test_loss = 0.0
# all_labels = []
# all_probabilities = []

# # New Class Counters
# label_0_count = 0  # Real-Real pairs (label = 0)
# label_1_count = 0  # Fake-involved pairs (label = 1)
# correct_label_0 = 0  # Correctly classified real-real
# correct_label_1 = 0  # Correctly classified fake-involved

# def compute_eer_auc(labels, scores):
#     """Computes EER, thresholds, AUC, and pAUC (for FPR ≤ 0.1)."""

#     # Compute ROC curve
#     fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr  # False Negative Rate

#     # Compute AUC
#     auc_score = auc(fpr, tpr)

#     # Compute partial AUC (pAUC) for FPR ≤ 0.1
#     max_fpr = 0.1
#     valid_mask = fpr <= max_fpr
#     fpr_filtered, tpr_filtered = fpr[valid_mask], tpr[valid_mask]

#     if len(fpr_filtered) >= 2:
#         pauc_score = auc(fpr_filtered, tpr_filtered) / max_fpr  # Normalize by max FPR
#     else:
#         pauc_score = 0.0  # Handle cases with insufficient points

#     # Compute absolute differences between FPR and FNR for EER
#     abs_diff = np.abs(fpr - fnr)

#     # Handle NaN cases
#     if np.isnan(abs_diff).any():
#         return None, None, None, None, None

#     # Find the index where the difference is minimized (EER point)
#     min_index = np.nanargmin(abs_diff)
#     eer = (fpr[min_index] + fnr[min_index]) / 2
#     eer_threshold = thresholds[min_index]

#     # Compute optimal threshold using Youden's J statistic (maximizing TPR - FPR)
#     youden_idx = np.argmax(tpr - fpr)
#     youden_threshold = thresholds[youden_idx]

#     return eer, eer_threshold, auc_score, pauc_score, youden_threshold


# # Evaluation loop
# with torch.no_grad():
#     for video_batch, labels, seq_lengths in tqdm(test_dataloader, desc="Evaluating", leave=True):
#         video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

#         # Count occurrences of labels 0 and 1
#         label_0_count += (labels == 0).sum().item()
#         label_1_count += (labels == 1).sum().item()

#         # Extract Features
#         features = model.extract_features(video_batch, seq_lengths)

#         # Forward Pass
#         outputs = model(features, seq_lengths).squeeze(-1)
#         probabilities = torch.sigmoid(outputs)

#         # Store outputs and labels for metric computation
#         all_probabilities.extend(probabilities.cpu().numpy())
#         # print(all_probabilities)
#         all_labels.extend(labels.cpu().numpy())

#         # Compute Loss
#         loss = criterion(outputs, labels.float())
#         test_loss += loss.item()

#         # Compute Accuracy
#         predicted = (probabilities > 0.5).float()
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

#         # Count correctly predicted labels
#         correct_label_0 += ((predicted == 0) & (labels == 0)).sum().item()
#         correct_label_1 += ((predicted == 1) & (labels == 1)).sum().item()

# # Compute final accuracy and loss
# test_loss /= len(test_dataloader)
# test_accuracy = correct / total

# # Convert lists to numpy arrays
# all_probabilities = np.array(all_probabilities)
# all_labels = np.array(all_labels)

# # # Compute AUC
# # auc_score = roc_auc_score(all_labels, all_probabilities)

# # Compute pAUC (partial AUC) for FPR ≤ 0.1
# # fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
# # max_fpr = 0.1
# # fpr_threshold_idx = np.searchsorted(fpr, max_fpr, side="right")

# # if fpr_threshold_idx > 1:
# #     partial_auc = auc(fpr[:fpr_threshold_idx], tpr[:fpr_threshold_idx]) / max_fpr
# # else:
# #     partial_auc = 0.0

# # Compute EER
# eer, eer_threshold, auc_score, partial_auc, y_thr = compute_eer_auc(all_labels, all_probabilities)
# # print(f"Threshold:{y_thr}")

# # Print final results
# print(f"\n===== Evaluation Results =====")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{total} correct)")
# print(f"AUC: {auc_score:.4f}")
# print(f"pAUC (FPR <= 0.1): {partial_auc:.4f}")
# if eer is not None:
#     print(f"EER: {eer:.4f} (Threshold: {eer_threshold:.4f})")
# else:
#     print("EER could not be computed due to numerical issues.")

# # Print class-wise statistics
# print("\n===== Class-wise Accuracy =====")
# print(f"Total label 0 (Real-Real): {label_0_count}")
# print(f"Total label 1 (Fake-involved): {label_1_count}")
# print(f"Correctly classified label 0: {correct_label_0} / {label_0_count} ({(correct_label_0 / label_0_count) * 100:.2f}%)")
# print(f"Correctly classified label 1: {correct_label_1} / {label_1_count} ({(correct_label_1 / label_1_count) * 100:.2f}%)")



# import os
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc
# import numpy as np

# from Models.XceptionLSTMV import XceptionLSTMV
# from Dataset.video_dataloader import get_face_dataloader

# # Device setup
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # ---------------- CONFIG ----------------
# config = {
#     "dataset_root": "/media/rt0706/Media/VCBSL-Dataset/FAVC_Whole/frames",
#     "mode": "fakeavceleb",   # or "lavdf"
#     "subset": "eval",
#     "csv_path": "Dataset/meta_data.csv",
#     "lavdf_json": "Dataset/LAV-DF/metadata.min.json",
#     "batch_size": 1,
#     "augment_minority": False,
#     "sample_percentage": 1.0,
#     "checkpoint": "Checkpoints/FakeAVCeleb_Correct_Labels.pth",
# }
# # ----------------------------------------

# # Load test dataset
# test_loader = get_face_dataloader(
#     config["dataset_root"],
#     mode=config["mode"],
#     subset=config["subset"],
#     csv_path=config["csv_path"],
#     lavdf_json=config["lavdf_json"],
#     batch_size=config["batch_size"],
#     shuffle=False,
#     augment_minority=config["augment_minority"],
#     sample_percentage=config["sample_percentage"]
# )

# # Load the best saved model
# if not os.path.exists(config["checkpoint"]):
#     raise FileNotFoundError(f"Model checkpoint not found at {config['checkpoint']}")

# # Initialize model
# model = XceptionLSTMV(hidden_dim=128).to(device)

# # Load model weights
# ckpt = torch.load(config["checkpoint"], map_location=device, weights_only=False)
# model.load_state_dict(ckpt, strict=False)
# model.eval()

# # Loss function
# criterion = nn.BCEWithLogitsLoss()

# # Evaluation metrics
# correct = 0
# total = 0
# test_loss = 0.0
# all_labels = []
# all_probabilities = []

# # Class counters
# label_0_count = 0
# label_1_count = 0
# correct_label_0 = 0
# correct_label_1 = 0


# def compute_eer_auc(labels, scores):
#     """Computes EER, thresholds, AUC, pAUC (FPR ≤ 0.1), and Youden’s J threshold."""
#     fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr

#     auc_score = auc(fpr, tpr)

#     # pAUC
#     max_fpr = 0.1
#     valid_mask = fpr <= max_fpr
#     pauc_score = auc(fpr[valid_mask], tpr[valid_mask]) / max_fpr if valid_mask.sum() > 1 else 0.0

#     # EER
#     abs_diff = np.abs(fpr - fnr)
#     min_index = np.nanargmin(abs_diff)
#     eer = (fpr[min_index] + fnr[min_index]) / 2
#     eer_threshold = thresholds[min_index]

#     # Youden’s J
#     youden_idx = np.argmax(tpr - fpr)
#     youden_threshold = thresholds[youden_idx]

#     return eer, eer_threshold, auc_score, pauc_score, youden_threshold


# # Evaluation loop
# with torch.no_grad():
#     for video_batch, labels, seq_lengths in tqdm(test_loader, desc="Evaluating", leave=True):
#         video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

#         label_0_count += (labels == 0).sum().item()
#         label_1_count += (labels == 1).sum().item()

#         features = model.extract_features(video_batch, seq_lengths)
#         outputs = model(features, seq_lengths).squeeze(-1)
#         probabilities = torch.sigmoid(outputs)

#         all_probabilities.extend(probabilities.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

#         loss = criterion(outputs, labels.float())
#         test_loss += loss.item()

#         total += labels.size(0)

# # Compute metrics
# test_loss /= len(test_loader)
# all_probabilities = np.array(all_probabilities)
# all_labels = np.array(all_labels)

# eer, eer_thr, auc_score, pauc_score, youden_thr = compute_eer_auc(all_labels, all_probabilities)

# # Final predictions using Youden threshold
# predicted = (all_probabilities > youden_thr).astype(int)
# correct = (predicted == all_labels).sum()
# test_accuracy = correct / len(all_labels)

# correct_label_0 = ((predicted == 0) & (all_labels == 0)).sum()
# correct_label_1 = ((predicted == 1) & (all_labels == 1)).sum()

# # Print results
# print(f"\n===== Evaluation Results =====")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{len(all_labels)} correct)")
# print(f"AUC: {auc_score:.4f}")
# print(f"pAUC (FPR <= 0.1): {pauc_score:.4f}")
# print(f"EER: {eer:.4f} (Threshold: {eer_thr:.4f})")
# print(f"Youden’s J Threshold: {youden_thr:.4f}")

# print("\n===== Class-wise Accuracy =====")
# print(f"Total label 0 (Real): {label_0_count}")
# print(f"Total label 1 (Fake): {label_1_count}")
# print(f"Correctly classified label 0: {correct_label_0} / {label_0_count} "
#       f"({(correct_label_0 / label_0_count * 100) if label_0_count else 0:.2f}%)")
# print(f"Correctly classified label 1: {correct_label_1} / {label_1_count} "
#       f"({(correct_label_1 / label_1_count * 100) if label_1_count else 0:.2f}%)")



# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch.amp import autocast
# from torch.utils.data import DataLoader
# from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, auc
# import torch.multiprocessing as mp

# from Models.XceptionLSTMV import XceptionLSTMV
# from Dataset.video_dataloader_enhanced import get_face_dataloader, collate_fn


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# # -------------------------
# # ArcFace Head (same as train)
# # -------------------------
# class ArcFaceHead(nn.Module):
#     def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.5):
#         super().__init__()
#         self.num_classes = num_classes
#         self.s = s
#         self.m = m
#         self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, features, labels=None):
#         x = F.normalize(features)
#         W = F.normalize(self.weight)
#         cos_theta = torch.matmul(x, W.t())
#         if labels is None:
#             return self.s * cos_theta
#         theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
#         target_logits = torch.cos(theta + self.m)
#         one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
#         output = cos_theta * (1 - one_hot) + target_logits * one_hot
#         return self.s * output


# # -------------------------
# # Metrics
# # -------------------------
# def compute_metrics(labels, probs):
#     if len(np.unique(labels)) <= 1:
#         return {"AUC": 0.0, "pAUC": 0.0, "AP": 0.0, "EER": 1.0}

#     auc_score = roc_auc_score(labels, probs)
#     ap_score = average_precision_score(labels, probs)
#     fpr, tpr, _ = roc_curve(labels, probs)
#     pauc_score = (
#         auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     )
#     fnr = 1 - tpr
#     abs_diffs = np.abs(fpr - fnr)
#     eer_idx = np.nanargmin(abs_diffs)
#     eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
#     return {"AUC": auc_score, "pAUC": pauc_score, "AP": ap_score, "EER": eer}


# # -------------------------
# # Test Loop
# # -------------------------
# def test():
#     test_folder = "/media/rt0706/Lab/LAV-DF"
#     lavdf_json = "Dataset/LAV-DF/metadata.json"
#     ckpt_path = "Checkpoints/XceptionLSTMV_ArcFace_Best.pth"

#     print("Loading test data...")
#     test_dataset = get_face_dataloader(
#         folder_path=test_folder,
#         mode="lavdf_raw",
#         subset="test",
#         lavdf_json=lavdf_json,
#         batch_size=1,
#         augment_minority=False,
#         shuffle=False,
#         raw_video=True,
#         use_face_detection=True,
#         frame_size=(224, 224),
#         max_frames=50,
#     ).dataset

#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

#     # Load model + ArcFace head
#     model = XceptionLSTMV(hidden_dim=128).to(device)
#     arcface_head = ArcFaceHead(128, 2, s=30.0, m=0.5).to(device)

#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     arcface_head.load_state_dict(ckpt["arcface"])
#     print(f"Loaded checkpoint from {ckpt_path}")

#     model.eval()
#     arcface_head.eval()

#     all_labels, all_probs = [], []
#     correct_real = correct_fake = total_real = total_fake = 0
#     correct = total = 0

#     with torch.no_grad():
#         for video_batch, labels, seq_lengths in tqdm(test_loader, desc="Testing"):
#             video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

#             with autocast("cuda", enabled=(device.type == "cuda")):
#                 features = model.extract_features(video_batch, seq_lengths)
#                 embeddings = model.lstm(features)[0][:, -1, :]
#                 logits = arcface_head(embeddings)
#                 probs = torch.softmax(logits, dim=1)[:, 1]

#             preds = (probs > 0.5).long()
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())

#             lbls, preds_np = labels.cpu().numpy(), preds.cpu().numpy()
#             correct += (preds_np == lbls).sum()
#             total += len(lbls)

#             correct_real += ((preds_np == 0) & (lbls == 0)).sum()
#             total_real += (lbls == 0).sum()
#             correct_fake += ((preds_np == 1) & (lbls == 1)).sum()
#             total_fake += (lbls == 1).sum()

#     acc = correct / max(total, 1)
#     metrics = compute_metrics(np.array(all_labels), np.array(all_probs))

#     print("\n=== Test Results ===")
#     print(f"Accuracy: {acc:.4f}")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#     print(f"Classwise: Real {correct_real}/{total_real}, Fake {correct_fake}/{total_fake}")


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     test()



import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, auc
import torch.multiprocessing as mp

from Models.XceptionLSTMV import XceptionLSTMV
from Dataset.video_dataloader_enhanced import get_face_dataloader, collate_fn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# -------------------------
# ArcFace Head (same as train)
# -------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        x = F.normalize(features)
        W = F.normalize(self.weight)
        cos_theta = torch.matmul(x, W.t())
        if labels is None:
            return self.s * cos_theta
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        return self.s * output


# -------------------------
# Metrics
# -------------------------
# def compute_metrics(labels, probs):
#     if len(np.unique(labels)) <= 1:
#         return {"AUC": 0.0, "pAUC": 0.0, "AP": 0.0, "EER": 1.0}

#     auc_score = roc_auc_score(labels, probs)
#     ap_score = average_precision_score(labels, probs)
#     fpr, tpr, _ = roc_curve(labels, probs)
#     pauc_score = (
#         auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     )
#     fnr = 1 - tpr
#     abs_diffs = np.abs(fpr - fnr)
#     eer_idx = np.nanargmin(abs_diffs)
#     eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
#     return {"AUC": auc_score, "pAUC": pauc_score, "AP": ap_score, "EER": eer}

def compute_metrics(labels, probs, alpha=0.1):
    labels = np.asarray(labels).astype(int)
    probs  = np.asarray(probs, dtype=float)

    if len(np.unique(labels)) < 2:
        return {"AUC": 0.0, "pAUC": 0.0, "AP": 0.0, "EER": 1.0}

    auc_score = roc_auc_score(labels, probs)
    ap_score  = average_precision_score(labels, probs)

    fpr, tpr, thresholds = roc_curve(labels, probs)

    # ---- pAUC on [0, alpha] with interpolation + normalized to [0,1] (0 = random) ----
    grid = np.linspace(0.0, alpha, 2001)
    tpr_i = np.interp(grid, fpr, tpr)           # interp TPR at desired FPR grid
    pauc_raw = auc(grid, tpr_i)
    pauc_norm = (pauc_raw - (alpha**2)/2) / (alpha - (alpha**2)/2)

    # ---- EER via linear interpolation at the crossing ----
    fnr = 1 - tpr
    diff = fpr - fnr
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]  # sign change
    if len(idx) == 0:
        # fallback to closest point if no exact crossing
        j = np.argmin(np.abs(diff))
        eer = (fpr[j] + fnr[j]) / 2.0
    else:
        j = idx[0]
        # linearly interpolate between (fpr[j], fnr[j]) and (fpr[j+1], fnr[j+1])
        x1, y1 = fpr[j], fnr[j]
        x2, y2 = fpr[j+1], fnr[j+1]
        # solve for x where x = y on the segment
        # diff(x) = (x - y) = 0 -> linear interpolation weight
        w = (y1 - x1) / ((x2 - x1) - (y2 - y1) + 1e-12)
        w = np.clip(w, 0.0, 1.0)
        eer = x1 + w*(x2 - x1)  # equals y1 + w*(y2 - y1)

    # (optional) Accuracy at Youden's J
    j_scores = tpr - fpr
    j_ix = np.argmax(j_scores)
    thr_j = thresholds[j_ix]
    acc_j = ((probs >= thr_j).astype(int) == labels).mean()

    return {
        "AUC": float(auc_score),
        "AP": float(ap_score),
        "pAUC": float(pauc_norm),
        "EER": float(eer),
        "ACC@J": float(acc_j),     # optional, remove if not needed
        "THR@J": float(thr_j)      # optional, remove if not needed
    }



# -------------------------
# Test Loop (FakeAVCeleb)
# -------------------------
def test():
    test_folder = "/media/rt0706/Media/VCBSL-Dataset/FAVC_Whole/frames"
    csv_path = "Dataset/meta_data.csv" 
    ckpt_path = "Checkpoints/XceptionLSTMV_ArcFace_Best.pth"

    print("Loading FakeAVCeleb test data...")
    test_dataset = get_face_dataloader(
        folder_path=test_folder,
        mode="fakeavceleb",
        subset="test",
        csv_path=csv_path,
        batch_size=1,
        augment_minority=False,
        shuffle=False,
        raw_video=False,
        use_face_detection=False,
        frame_size=(224, 224),
        max_frames=75,
    ).dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Load model + ArcFace head
    model = XceptionLSTMV(hidden_dim=128).to(device)
    arcface_head = ArcFaceHead(128, 2, s=30.0, m=0.5).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    arcface_head.load_state_dict(ckpt["arcface"])
    print(f"Loaded checkpoint from {ckpt_path}")

    model.eval()
    arcface_head.eval()

    all_labels, all_probs = [], []
    correct_real = correct_fake = total_real = total_fake = 0
    correct = total = 0

    with torch.no_grad():
        for video_batch, labels, seq_lengths in tqdm(test_loader, desc="Testing"):
            video_batch, labels, seq_lengths = video_batch.to(device), labels.to(device), seq_lengths.to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                features = model.extract_features(video_batch, seq_lengths)
                embeddings = model.lstm(features)[0][:, -1, :]
                logits = arcface_head(embeddings)
                probs = torch.softmax(logits, dim=1)[:, 1]

            preds = (probs > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            lbls, preds_np = labels.cpu().numpy(), preds.cpu().numpy()
            correct += (preds_np == lbls).sum()
            total += len(lbls)

            correct_real += ((preds_np == 0) & (lbls == 0)).sum()
            total_real += (lbls == 0).sum()
            correct_fake += ((preds_np == 1) & (lbls == 1)).sum()
            total_fake += (lbls == 1).sum()

    acc = correct / max(total, 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))

    print("\n=== FakeAVCeleb Test Results ===")
    print(f"Accuracy: {acc:.4f}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Classwise: Real {correct_real}/{total_real}, Fake {correct_fake}/{total_fake}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test()