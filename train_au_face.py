# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc
# from torch.amp import autocast, GradScaler
# from torch.optim.swa_utils import AveragedModel

# from Models.AUFaceModel import EVM_AU_GradCAM_Model
# from Dataset.AuVidDataset import get_joint_dataloader

# def compute_eer_auc(labels, scores):
#     fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2
#     return auc_score, pauc, eer

# class AdaptiveDeepfakeLoss(nn.Module):
#     def __init__(self, pos_weight):
#         super().__init__()
#         self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.beta = nn.Parameter(torch.tensor(0.3))

#     def forward(self, logits, labels, v_tokens, au_tokens):
#         loss_cls = self.cls_loss(logits, labels)
#         loss_align = torch.mean((v_tokens - au_tokens)**2)
#         delta = v_tokens[:,1:] - v_tokens[:,:-1]
#         loss_temp = torch.mean(delta**2)
#         total = loss_cls + torch.sigmoid(self.alpha)*loss_align + torch.sigmoid(self.beta)*loss_temp
#         return total, loss_cls.item(), loss_align.item(), loss_temp.item()

# # --- Setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_amp = True
# accum_steps = 4
# patience = 8

# train_loader, test_loader, eval_loader = get_joint_dataloader(
#     video_root="Dataset/FAVC_Whole/frames",
#     au_root="Dataset/AU_Files/AU_Image_Patches",
#     batch_size=2,
#     shuffle=True,
#     max_frames=30,
#     max_aus=17,
#     image_size=128,
#     num_workers=8
# )

# model = EVM_AU_GradCAM_Model(embed_dim=512).to(device)
# ema_model = AveragedModel(model)

# pos_weight = torch.tensor([5.0], device=device)
# criterion = AdaptiveDeepfakeLoss(pos_weight=pos_weight)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=1e-3,
#     total_steps=100 * len(train_loader),
#     pct_start=0.3
# )
# scaler = GradScaler(enabled=use_amp)

# best_auc = 0.0
# early_stop_count = 0

# for epoch in range(100):
#     model.train()
#     train_loss = 0.0
#     all_probs, all_labels = [], []

#     print(f"\nEpoch {epoch + 1}")
#     optimizer.zero_grad()

#     for i, (videos, au_patches, labels) in enumerate(tqdm(train_loader, desc="Training")):
#         videos, au_patches = videos.to(device), au_patches.to(device)
#         labels = labels.float().to(device)

#         with autocast(device_type="cuda", enabled=use_amp):
#             logits, v_tokens, au_tokens, _ = model(videos, au_patches)
#             loss, loss_cls, loss_align, loss_temp = criterion(logits.squeeze(-1), labels, v_tokens, au_tokens)

#         scaler.scale(loss).backward()

#         if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             scheduler.step()
#             ema_model.update_parameters(model)


#         probs = torch.sigmoid(logits.detach()).cpu().numpy()
#         all_probs.extend(probs)
#         all_labels.extend(labels.cpu().numpy())
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_auc, train_pauc, train_eer = compute_eer_auc(all_labels, all_probs)
#     print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, pAUC={train_pauc:.4f}, EER={train_eer:.4f}")

#     # --- Evaluation ---
#     ema_model.eval()
#     eval_probs, eval_labels = [], []

#     with torch.no_grad():
#         for videos, au_patches, labels in tqdm(eval_loader, desc="Evaluation"):
#             videos, au_patches = videos.to(device), au_patches.to(device)
#             labels = labels.float().cpu().numpy()

#             logits, _, _, _ = ema_model(videos, au_patches)
#             probs = torch.sigmoid(logits).cpu().numpy()

#             eval_probs.extend(probs)
#             eval_labels.extend(labels)

#     eval_auc, eval_pauc, eval_eer = compute_eer_auc(eval_labels, eval_probs)
#     print(f"Eval: AUC={eval_auc:.4f}, pAUC={eval_pauc:.4f}, EER={eval_eer:.4f}")

#     if eval_auc > best_auc:
#         best_auc = eval_auc
#         early_stop_count = 0
#         torch.save(ema_model.module.state_dict(), "Checkpoints/evm_au_gradcam_best_auc_2.pth")
#         print(f"New best AUC: {eval_auc:.4f} - Model saved.")
#     else:
#         early_stop_count += 1
#         if early_stop_count >= patience:
#             print(f"Early stopping at AUC {best_auc:.4f}")
#             break

# print("Training Complete.")



# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc, average_precision_score
# from torch.amp import autocast, GradScaler
# from torch.optim.swa_utils import AveragedModel

# from Models.AUFaceModel import AUFaceCrossDetector
# from Dataset.AuVidDataset import get_joint_dataloader  # must support return_weights=True


# # ---------------- Metrics & thresholding ----------------
# def compute_eer_auc(labels, scores):
#     y = np.asarray(labels).astype(int).ravel()
#     s = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     idx = int(np.nanargmin(np.abs(fpr - fnr)))
#     eer = float((fpr[idx] + fnr[idx]) / 2.0)
#     return auc_score, pauc, eer, (fpr, tpr)


# def pick_threshold(labels, scores, mode="youden", fpr_target=0.01):
#     y = np.asarray(labels).astype(int).ravel()
#     s = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)

#     if mode == "youden":
#         j = tpr - fpr
#         j_idx = int(np.argmax(j))
#         return float(thr[j_idx]), float(fpr[j_idx]), float(tpr[j_idx])

#     ok = np.where(fpr <= float(fpr_target))[0]
#     if len(ok) == 0:
#         return float(thr[0]), float(fpr[0]), float(tpr[0])
#     idx = int(ok[-1])
#     return float(thr[idx]), float(fpr[idx]), float(tpr[idx])


# def compute_acc_ap_and_counts(labels, scores, thr):
#     y = np.asarray(labels).astype(int).ravel()
#     s = np.asarray(scores).astype(float).ravel()
#     preds = (s >= float(thr)).astype(int)
#     acc = float((preds == y).mean())
#     total_real = int((y == 0).sum())
#     total_fake = int((y == 1).sum())
#     correct_real = int(((preds == 0) & (y == 0)).sum())
#     correct_fake = int(((preds == 1) & (y == 1)).sum())
#     try:
#         ap = float(average_precision_score(y, s)) if (y.min() != y.max()) else float("nan")
#     except Exception:
#         ap = float("nan")
#     return acc, ap, correct_real, total_real, correct_fake, total_fake


# # ---------------- Loss (Option B: uniform) ----------------
# class AdaptiveDeepfakeLoss(nn.Module):
#     """Unweighted BCE-with-logits + alignment + temporal smoothness."""
#     def __init__(self):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.beta  = nn.Parameter(torch.tensor(0.3))

#     def forward(self, logits, labels, v_tokens, au_tokens):
#         if logits.dim() == 2 and logits.size(-1) == 1:
#             logits = logits.squeeze(-1)

#         loss_cls = F.binary_cross_entropy_with_logits(logits, labels)

#         loss_align = torch.mean((v_tokens - au_tokens) ** 2)

#         if v_tokens.size(1) > 1:
#             delta = v_tokens[:, 1:] - v_tokens[:, :-1]
#             loss_temp = torch.mean(delta ** 2)
#         else:
#             loss_temp = logits.new_tensor(0.0)

#         total = loss_cls + torch.sigmoid(self.alpha) * loss_align + torch.sigmoid(self.beta) * loss_temp
#         return total, float(loss_cls.detach().cpu()), float(loss_align.detach().cpu()), float(loss_temp.detach().cpu())


# # ---------------- Helpers ----------------
# def unpack_batch(batch):
#     """
#     Supports both (videos, aus, labels) and (videos, aus, labels, au_mask, au_weight).
#     Returns tensors with au_mask/au_weight possibly None.
#     """
#     if len(batch) == 5:
#         videos, au_patches, labels, au_mask, au_weight = batch
#     elif len(batch) == 3:
#         videos, au_patches, labels = batch
#         au_mask, au_weight = None, None
#     else:
#         raise RuntimeError(f"Unexpected batch structure of length {len(batch)}")
#     return videos, au_patches, labels, au_mask, au_weight


# # ---------------- Main (single GPU) ----------------
# primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_amp = True
# accum_steps = 4
# patience = 8
# epochs = 100

# train_loader, test_loader, eval_loader = get_joint_dataloader(
#     video_root="/media/rt0706/Media/VCBSL-Dataset/FAVC_Whole/frames",
#     au_root="Dataset/AU_Files/fakeavceleb_whole_image_patches",
#     batch_size=2,
#     shuffle=True,
#     max_frames=75,
#     max_aus=17,
#     image_size=128,
#     num_workers=4,
#     csv_path="Dataset/meta_data.csv",
#     # make the dataloader return (au_mask, au_weight)
#     return_weights=True,
# )

# model = AUFaceCrossDetector(num_aus=17, face_dim=512, au_dim=512, lstm_hidden=256).to(primary_device)
# ema_model = AveragedModel(model).to(primary_device)

# criterion = AdaptiveDeepfakeLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# # OneCycleLR with grad accumulation
# steps_per_epoch = math.ceil(len(train_loader) / max(1, accum_steps))
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3
# )

# scaler = GradScaler(enabled=use_amp)
# os.makedirs("Checkpoints", exist_ok=True)
# best_auc = 0.0
# early_stop_count = 0

# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     all_probs, all_labels = [], []

#     print(f"\nEpoch {epoch + 1}")
#     optimizer.zero_grad(set_to_none=True)
#     last_step_count = int(getattr(optimizer, "_step_count", 0))

#     for i, batch in enumerate(tqdm(train_loader, desc="Training")):
#         videos, au_patches, labels, au_mask, au_weight = unpack_batch(batch)

#         # Normalize to (B, C, T, H, W) if needed
#         if videos.dim() == 5 and videos.size(1) != 3 and videos.size(2) == 3:
#             videos = videos.permute(0, 2, 1, 3, 4).contiguous()

#         videos     = videos.to(primary_device, non_blocking=True)
#         au_patches = au_patches.to(primary_device, non_blocking=True)
#         labels     = labels.float().to(primary_device, non_blocking=True)
#         if au_mask is not None:
#             au_mask = au_mask.to(primary_device, non_blocking=True).float()
#         if au_weight is not None:
#             au_weight = au_weight.to(primary_device, non_blocking=True).float()

#         with autocast(device_type="cuda", enabled=(primary_device.type == "cuda" and use_amp)):
#             logits, v_tokens, au_tokens = model(videos, au_patches, au_mask=au_mask, au_weight=au_weight)
#             loss, _, _, _ = criterion(logits, labels, v_tokens, au_tokens)

#         scaler.scale(loss).backward()

#         if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)

#             step_count_now = int(getattr(optimizer, "_step_count", 0))
#             if step_count_now > last_step_count:
#                 scheduler.step()
#                 ema_model.update_parameters(model)
#             last_step_count = step_count_now

#         probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
#         all_probs.extend(probs)
#         all_labels.extend(labels.detach().cpu().numpy())
#         train_loss += float(loss.item())

#     train_loss /= max(1, len(train_loader))
#     train_auc, train_pauc, train_eer, _ = compute_eer_auc(all_labels, all_probs)
#     thr_youden, _, _ = pick_threshold(all_labels, all_probs, mode="youden")
#     tr_acc, tr_ap, tr_cr, tr_tr, tr_cf, tr_tf = compute_acc_ap_and_counts(all_labels, all_probs, thr=thr_youden)
#     print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, pAUC={train_pauc:.4f}, EER={train_eer:.4f}")
#     print(f"Train: Acc@Youden={tr_acc:.4f}, AP={tr_ap:.4f}, Correct[real]={tr_cr}/{tr_tr}, "
#           f"Correct[fake]={tr_cf}/{tr_tf}, thr={thr_youden:.3f}")

#     # -------- Evaluation (EMA) --------
#     ema_model.eval()
#     eval_probs, eval_labels = [], []
#     with torch.no_grad():
#         for batch in tqdm(eval_loader, desc="Evaluation"):
#             videos, au_patches, labels, au_mask, au_weight = unpack_batch(batch)
#             if videos.dim() == 5 and videos.size(1) != 3 and videos.size(2) == 3:
#                 videos = videos.permute(0, 2, 1, 3, 4).contiguous()

#             videos     = videos.to(primary_device, non_blocking=True)
#             au_patches = au_patches.to(primary_device, non_blocking=True)
#             if au_mask is not None:
#                 au_mask = au_mask.to(primary_device, non_blocking=True).float()
#             if au_weight is not None:
#                 au_weight = au_weight.to(primary_device, non_blocking=True).float()

#             logits, _, _ = ema_model(videos, au_patches, au_mask=au_mask, au_weight=au_weight)
#             probs = torch.sigmoid(logits).float().cpu().numpy()
#             eval_probs.extend(probs)
#             eval_labels.extend(labels.float().cpu().numpy())

#     eval_auc, eval_pauc, eval_eer, _ = compute_eer_auc(eval_labels, eval_probs)
#     thr_eval, fpr_eval, tpr_eval = pick_threshold(eval_labels, eval_probs, mode="youden")
#     ev_acc, ev_ap, ev_cr, ev_tr, ev_cf, ev_tf = compute_acc_ap_and_counts(eval_labels, eval_probs, thr=thr_eval)
#     print(f"Eval: AUC={eval_auc:.4f}, pAUC={eval_pauc:.4f}, EER={eval_eer:.4f}")
#     print(f"Eval: Acc@thr={ev_acc:.4f}, AP={ev_ap:.4f}, Correct[real]={ev_cr}/{ev_tr}, "
#           f"Correct[fake]={ev_cf}/{ev_tf}, thr={thr_eval:.3f}, FPR={fpr_eval:.3f}, TPR={tpr_eval:.3f}")

#     # Low-FPR operating point (optional)
#     thr_eval_f, fpr_f, tpr_f = pick_threshold(eval_labels, eval_probs, mode="fpr", fpr_target=0.05)
#     ev_acc_f, ev_ap_f, ev_cr_f, ev_tr_f, ev_cf_f, ev_tf_f = compute_acc_ap_and_counts(eval_labels, eval_probs, thr=thr_eval_f)
#     print(f"Eval@FPR≤5%: Acc={ev_acc_f:.4f}, AP={ev_ap_f:.4f}, thr={thr_eval_f:.3f}, "
#           f"FPR={fpr_f:.3f}, TPR={tpr_f:.3f}, Correct[real]={ev_cr_f}/{ev_tr_f}, Correct[fake]={ev_cf_f}/{ev_tf_f}")

#     if eval_auc > best_auc:
#         best_auc = eval_auc
#         early_stop_count = 0
#         torch.save(ema_model.state_dict(), "Checkpoints/auface_cross_best_auc_optionB.pth")
#         print(f"New best AUC: {eval_auc:.4f} - Model saved.")
#     else:
#         early_stop_count += 1
#         if early_stop_count >= patience:
#             print(f"Early stopping at AUC {best_auc:.4f}")
#             break

# print("Training Complete.")


# train_auface_arcface_cbfocal.py

import os
import math
import numpy as np
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc as sk_auc, average_precision_score

from Models.AUFaceModel import AUFaceCrossDetector
from Dataset.AuVidDataset import get_joint_dataloader  # returns (train, test, eval); supports return_weights=True


# =================== Reproducibility ===================
def set_seed(seed: Optional[int] = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =================== ArcFace + CB-Focal ===================
class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2, s=30.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        x = F.normalize(features)
        W = F.normalize(self.weight)
        cos = x @ W.t()
        if labels is None:
            return self.s * cos
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = cos * (1 - one_hot) + target * one_hot
        return self.s * logits


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss on logits (after ArcFace)."""
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(samples_per_cls)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# =================== Metrics ===================
def compute_eer_auc(labels, scores):
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    fnr = 1 - tpr
    auc_score = sk_auc(fpr, tpr) if len(fpr) else float("nan")
    mask = fpr <= 0.1
    pauc = sk_auc(fpr[mask], tpr[mask]) / 0.1 if np.sum(mask) >= 2 else float("nan")
    idx = int(np.nanargmin(np.abs(fpr - fnr))) if len(fpr) else 0
    eer = float((fpr[idx] + fnr[idx]) / 2.0) if len(fpr) else float("nan")
    return auc_score, pauc, eer, (fpr, tpr)


def pick_threshold(labels, scores, mode="youden", fpr_target=0.01):
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    if len(fpr) == 0:
        return 0.5, 0.0, 0.0
    if mode == "youden":
        j = tpr - fpr
        j_idx = int(np.argmax(j))
        return float(thr[j_idx]), float(fpr[j_idx]), float(tpr[j_idx])
    ok = np.where(fpr <= float(fpr_target))[0]
    if len(ok) == 0:
        return float(thr[0]), float(fpr[0]), float(tpr[0])
    idx = int(ok[-1])
    return float(thr[idx]), float(fpr[idx]), float(tpr[idx])


def compute_acc_ap_and_counts(labels, scores, thr):
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()
    preds = (s >= float(thr)).astype(int)
    acc = float((preds == y).mean())
    total_real = int((y == 0).sum())
    total_fake = int((y == 1).sum())
    correct_real = int(((preds == 0) & (y == 0)).sum())
    correct_fake = int(((preds == 1) & (y == 1)).sum())
    try:
        ap = float(average_precision_score(y, s)) if (y.min() != y.max()) else float("nan")
    except Exception:
        ap = float("nan")
    return acc, ap, correct_real, total_real, correct_fake, total_fake


# =================== Helpers ===================
def unpack_batch(batch):
    """Supports both (videos, aus, labels) and (videos, aus, labels, au_mask, au_weight)."""
    if len(batch) == 5:
        videos, au_patches, labels, au_mask, au_weight = batch
    elif len(batch) == 3:
        videos, au_patches, labels = batch
        au_mask, au_weight = None, None
    else:
        raise RuntimeError(f"Unexpected batch of length {len(batch)}")
    return videos, au_patches, labels, au_mask, au_weight


def make_weighted_sampler_from_dataset(train_loader: DataLoader):
    """
    Build a WeightedRandomSampler using dataset-provided labels (preferred)
    or fall back to uniform if unavailable.
    """
    ds = train_loader.dataset
    labels = None

    # Preferred: your dataset exposes all labels here
    if hasattr(ds, "all_labels"):
        arr = getattr(ds, "all_labels")
        labels = list(arr) if not torch.is_tensor(arr) else arr.cpu().tolist()

    if labels is not None and len(labels) == len(ds):
        counts = np.bincount(np.array(labels), minlength=2)
        w_real = 0.5 / max(int(counts[0]), 1)
        w_fake = 0.5 / max(int(counts[1]), 1)
        sample_weights = [w_fake if y == 1 else w_real for y in labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return None


# =================== Config ===================
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = True
accum_steps = 4
patience = 8
epochs = 100
grad_clip = 1.0
seed = 42

# regularizer weights
lambda_align = 0.2
lambda_temp = 0.1


def main():
    set_seed(seed)
    os.makedirs("Checkpoints", exist_ok=True)

    # --------------- Data ---------------
    train_loader, test_loader, eval_loader = get_joint_dataloader(
        video_root="/media/rt0706/Media/VCBSL-Dataset/FAVC_Whole/frames",
        au_root="Dataset/AU_Files/fakeavceleb_whole_image_patches",
        batch_size=2,
        shuffle=True,
        max_frames=75,
        max_aus=17,
        image_size=128,
        num_workers=4,
        csv_path="Dataset/meta_data.csv",
        return_weights=True,   # loader returns (videos, aus, labels, mask, weights)
    )

    # Optional: wrap with balanced sampler (dataset is already oversampled on train,
    # but this enforces class-balanced minibatches even when shuffling)
    sampler = make_weighted_sampler_from_dataset(train_loader)
    if sampler is not None:
        print("[Info] Using WeightedRandomSampler for class-balanced batches.")
        train_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            collate_fn=getattr(train_loader, "collate_fn", None),
            drop_last=False,
        )
    else:
        print("[Info] Proceeding without a custom sampler (dataset oversampling + CB-Focal still help).")

    # --------------- Model ---------------
    model = AUFaceCrossDetector(num_aus=17, face_dim=512, au_dim=512, lstm_hidden=256).to(primary_device)
    ema_model = AveragedModel(model).to(primary_device)

    # Embed head → 128-D features for ArcFace
    embed_head = nn.Sequential(
        nn.LazyLinear(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
    ).to(primary_device)
    ema_embed = AveragedModel(embed_head).to(primary_device)

    # ArcFace classifier
    arcface = ArcFaceHead(feat_dim=128, num_classes=2, s=30.0, m=0.30).to(primary_device)

    # CB-Focal init from dataset labels (fast, no warm-scan)
    counts = Counter(getattr(train_loader.dataset, "all_labels", []))
    samples_per_cls = [max(counts.get(0, 1), 1), max(counts.get(1, 1), 1)]
    print(f"[Info] Class counts (for CB-Focal): real={samples_per_cls[0]}, fake={samples_per_cls[1]}")
    cbfocal = CBFocalLoss(samples_per_cls=samples_per_cls, beta=0.9999, gamma=2.0).to(primary_device)

    # Optimizer / Scheduler / AMP
    optimizer = optim.AdamW(
        list(model.parameters()) + list(embed_head.parameters()) + list(arcface.parameters()),
        lr=1e-4, weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, accum_steps))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3
    )
    scaler = GradScaler(enabled=use_amp)

    best_auc = 0.0
    early_stop_count = 0

    # --------------- Train Loop ---------------
    for epoch in range(epochs):
        model.train(); embed_head.train(); arcface.train()
        train_loss = 0.0
        all_probs, all_labels = [], []

        print(f"\nEpoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        last_step_count = int(getattr(optimizer, "_step_count", 0))

        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            videos, au_patches, labels, au_mask, au_weight = unpack_batch(batch)

            # Normalize to (B, C, T, H, W) if needed
            if videos.dim() == 5 and videos.size(1) != 3 and videos.size(2) == 3:
                videos = videos.permute(0, 2, 1, 3, 4).contiguous()

            videos     = videos.to(primary_device, non_blocking=True)
            au_patches = au_patches.to(primary_device, non_blocking=True)
            labels     = labels.long().to(primary_device, non_blocking=True)
            if au_mask is not None:
                au_mask = au_mask.to(primary_device, non_blocking=True).float()
            if au_weight is not None:
                au_weight = au_weight.to(primary_device, non_blocking=True).float()

            with autocast(device_type="cuda", enabled=(primary_device.type == "cuda" and use_amp)):
                # Model returns: logits_raw (ignored), v_tokens [B,Tv,Dv], au_tokens [B,Ta,Da]
                _, v_tokens, au_tokens = model(videos, au_patches, au_mask=au_mask, au_weight=au_weight)

                # Build 128-D embedding for ArcFace
                v_pool = v_tokens.mean(1)        # [B, Dv]
                au_pool = au_tokens.mean(1)      # [B, Da]
                pooled = torch.cat([v_pool, au_pool], dim=1)
                embed = embed_head(pooled)       # [B, 128]

                # ArcFace + CB-Focal classification
                logits_arc = arcface(embed, labels)  # [B,2]
                loss_cls = cbfocal(logits_arc, labels)

                # Regularizers
                loss_align = F.mse_loss(v_pool, au_pool)
                loss_temp_v = (v_tokens[:, 1:] - v_tokens[:, :-1]).pow(2).mean() if v_tokens.size(1) > 1 else v_tokens.new_tensor(0.0)
                loss_temp_au = (au_tokens[:, 1:] - au_tokens[:, :-1]).pow(2).mean() if au_tokens.size(1) > 1 else au_tokens.new_tensor(0.0)
                loss_temp = 0.5 * (loss_temp_v + loss_temp_au)

                loss = loss_cls + lambda_align * loss_align + lambda_temp * loss_temp

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(embed_head.parameters()) + list(arcface.parameters()),
                    grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                step_count_now = int(getattr(optimizer, "_step_count", 0))
                if step_count_now > last_step_count:
                    scheduler.step()
                    ema_model.update_parameters(model)
                    ema_embed.update_parameters(embed_head)
                last_step_count = step_count_now

            probs = torch.softmax(logits_arc.detach(), dim=1)[:, 1].float().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))
        train_auc, train_pauc, train_eer, _ = compute_eer_auc(all_labels, all_probs)
        thr_youden, _, _ = pick_threshold(all_labels, all_probs, mode="youden")
        tr_acc, tr_ap, tr_cr, tr_tr, tr_cf, tr_tf = compute_acc_ap_and_counts(all_labels, all_probs, thr=thr_youden)
        print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, pAUC={train_pauc:.4f}, EER={train_eer:.4f}")
        print(f"Train: Acc@Youden={tr_acc:.4f}, AP={tr_ap:.4f}, Correct[real]={tr_cr}/{tr_tr}, "
              f"Correct[fake]={tr_cf}/{tr_tf}, thr={thr_youden:.3f}")

        # -------- Evaluation (EMA backbone + EMA embed; ArcFace current head) --------
        ema_model.eval(); ema_embed.eval(); arcface.eval()
        eval_probs, eval_labels = [], []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation"):
                videos, au_patches, labels, au_mask, au_weight = unpack_batch(batch)
                if videos.dim() == 5 and videos.size(1) != 3 and videos.size(2) == 3:
                    videos = videos.permute(0, 2, 1, 3, 4).contiguous()

                videos     = videos.to(primary_device, non_blocking=True)
                au_patches = au_patches.to(primary_device, non_blocking=True)
                labels     = labels.long().to(primary_device, non_blocking=True)
                if au_mask is not None:
                    au_mask = au_mask.to(primary_device, non_blocking=True).float()
                if au_weight is not None:
                    au_weight = au_weight.to(primary_device, non_blocking=True).float()

                _, v_tokens, au_tokens = ema_model(videos, au_patches, au_mask=au_mask, au_weight=au_weight)
                v_pool = v_tokens.mean(1)
                au_pool = au_tokens.mean(1)
                embed = ema_embed(torch.cat([v_pool, au_pool], dim=1))
                logits_arc = arcface(embed)  # labels=None -> plain logits
                probs = torch.softmax(logits_arc, dim=1)[:, 1].float().cpu().numpy()
                eval_probs.extend(probs)
                eval_labels.extend(labels.cpu().numpy())

        eval_auc, eval_pauc, eval_eer, _ = compute_eer_auc(eval_labels, eval_probs)
        thr_eval, fpr_eval, tpr_eval = pick_threshold(eval_labels, eval_probs, mode="youden")
        ev_acc, ev_ap, ev_cr, ev_tr, ev_cf, ev_tf = compute_acc_ap_and_counts(eval_labels, eval_probs, thr=thr_eval)
        print(f"Eval: AUC={eval_auc:.4f}, pAUC={eval_pauc:.4f}, EER={eval_eer:.4f}")
        print(f"Eval: Acc@thr={ev_acc:.4f}, AP={ev_ap:.4f}, Correct[real]={ev_cr}/{ev_tr}, "
              f"Correct[fake]={ev_cf}/{ev_tf}, thr={thr_eval:.3f}, FPR={fpr_eval:.3f}, TPR={tpr_eval:.3f}")

        # Optional: low-FPR operating point (5%)
        thr_eval_f, fpr_f, tpr_f = pick_threshold(eval_labels, eval_probs, mode="fpr", fpr_target=0.05)
        ev_acc_f, ev_ap_f, ev_cr_f, ev_tr_f, ev_cf_f, ev_tf_f = compute_acc_ap_and_counts(eval_labels, eval_probs, thr=thr_eval_f)
        print(f"Eval@FPR≤5%: Acc={ev_acc_f:.4f}, AP={ev_ap_f:.4f}, thr={thr_eval_f:.3f}, "
              f"FPR={fpr_f:.3f}, TPR={tpr_f:.3f}, Correct[real]={ev_cr_f}/{ev_tr_f}, Correct[fake]={ev_cf_f}/{ev_tf_f}")

        # Save best by AUC
        if eval_auc > best_auc:
            best_auc = eval_auc
            early_stop_count = 0
            torch.save({
                "model": ema_model.state_dict(),
                "embed": ema_embed.state_dict(),
                "arcface": arcface.state_dict(),
                "best_auc": best_auc
            }, "Checkpoints/auface_cross_best_auc_arcface_cb.pth")
            print(f"New best AUC: {eval_auc:.4f} - Model saved.")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"Early stopping at AUC {best_auc:.4f}")
                break

    print("Training Complete.")


if __name__ == "__main__":
    main()