# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc
# from torch.amp import autocast

# from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
# from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention

# # Metric Calculation
# def compute_eer_auc(labels, scores):
#     fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2
#     return auc_score, pauc, eer

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load test data
# _, test_loader, _ = get_patch_image_loaders(
#     "Dataset/LAVDF_Patches", batch_size=2, image_size=128, max_frames=90
# )

# # Load model
# model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)
# model.load_state_dict(torch.load("Checkpoints/au_patch_lstm_best_model_with_attn_2.pth", map_location=device), strict=False)
# model.eval()

# # Testing
# test_loss, correct, total = 0.0, 0, 0
# all_probs, all_labels = [], []
# true_positives, true_negatives = 0, 0

# criterion = torch.nn.BCEWithLogitsLoss()

# with torch.no_grad():
#     for patches, weights_batch, labels in tqdm(test_loader, desc="Testing"):
#         patches = patches.to(device)
#         weights_batch = weights_batch.to(device)
#         labels = labels.to(device)

#         with autocast("cuda"):
#             outputs = model(patches, au_patch_weights=weights_batch).squeeze(-1)
#             loss = criterion(outputs, labels)

#         probs = torch.sigmoid(outputs / 2.0).detach().cpu().numpy()
#         print(probs)
#         preds = (probs > 0.5).astype(float)
#         print(preds)

#         all_probs.extend(probs)
#         all_labels.extend(labels.cpu().numpy())

#         correct += (preds == labels.cpu().numpy()).sum()
#         total += labels.size(0)
#         test_loss += loss.item()

#         true_positives += np.sum((preds == 1) & (labels.cpu().numpy() == 1))
#         true_negatives += np.sum((preds == 0) & (labels.cpu().numpy() == 0))

# # --- Metrics ---
# acc = correct / total
# auc_score, pauc_score, eer = compute_eer_auc(all_labels, all_probs)
# test_loss /= len(test_loader)

# real_total = np.sum(np.array(all_labels) == 0)
# fake_total = np.sum(np.array(all_labels) == 1)
# real_correct = true_negatives
# fake_correct = true_positives
# real_acc = real_correct / real_total * 100 if real_total > 0 else 0
# fake_acc = fake_correct / fake_total * 100 if fake_total > 0 else 0

# # --- Print Summary ---
# print(f"\nTest Results:")
# print(f"Loss={test_loss:.4f}, Accuracy={acc:.4f}")
# print(f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}")
# print(f"Real samples (class: 0) --> correct ({real_correct}/{real_total}) ({real_acc:.1f}%)")
# print(f"Fake samples (class: 1) --> correct ({fake_correct}/{fake_total}) ({fake_acc:.1f}%)")

# print("Testing complete.")



# import os
# import torch
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc
# from torch.amp import autocast

# from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
# from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention

# # --- Metrics ---
# def compute_eer_auc(labels, scores):
#     fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     eer_idx = np.nanargmin(np.abs(fpr - fnr))
#     eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
#     thr_eer = thr[eer_idx]
#     # Youden's J (optional)
#     j = tpr - fpr
#     j_idx = np.nanargmax(j)
#     thr_j = thr[j_idx]
#     return auc_score, pauc, eer, float(thr_eer), float(thr_j)

# def safe_pick_threshold(labels, scores, mode="eer"):
#     y = np.asarray(labels).astype(int)
#     s = np.asarray(scores).astype(float)
#     if y.size == 0:
#         return float(np.median(s) if s.size else 0.0)
#     # If only one class present, ROC is undefined → fall back to median score
#     if np.min(y) == np.max(y):
#         return float(np.median(s))
#     _, _, _, thr_eer, thr_j = compute_eer_auc(y, s)
#     return float(thr_eer if mode == "eer" else thr_j)

# # --- Setup ---
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Load test data
# _, test_loader, _ = get_patch_image_loaders(
#     "Dataset/AU_Files/AU_Image_Patches", batch_size=2, image_size=128, max_frames=90
# )

# # Load model
# model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)
# state = torch.load("Checkpoints/ff_au_patch_60fps_lstm_best_model_with_attn_5.pth", map_location=device)
# model.load_state_dict(state, strict=False)
# model.eval()

# criterion = torch.nn.BCEWithLogitsLoss()

# # --- Testing loop (no thresholding inside) ---
# test_loss = 0.0
# all_probs, all_labels = [], []

# with torch.no_grad():
#     for patches, weights_batch, labels in tqdm(test_loader, desc="Testing"):
#         patches = patches.to(device)
#         weights_batch = weights_batch.to(device)
#         labels = labels.to(device).float()

#         with autocast(device_type="cuda", enabled=(device.type == "cuda")):
#             logits = model(patches, au_patch_weights=weights_batch)

#         # Normalize shapes → logits: [B], labels: [B]
#         if logits.dim() == 2 and logits.size(-1) == 1:
#             logits = logits.squeeze(-1)
#         elif logits.dim() == 0:
#             logits = logits.unsqueeze(0)
#         labels = labels.view_as(logits)

#         # Loss in fp32
#         loss = criterion(logits.float(), labels.float())
#         test_loss += float(loss.item())

#         # Calibrated probs (keep your /2.0 choice)
#         probs = torch.sigmoid((logits / 2.0).float()).cpu().numpy()

#         all_probs.extend(probs.tolist())
#         all_labels.extend(labels.cpu().numpy().tolist())

# # --- Post-hoc thresholding (EER) + metrics ---
# all_probs_np = np.asarray(all_probs, dtype=float)
# all_labels_np = np.asarray(all_labels, dtype=int)
# test_loss /= max(1, len(test_loader))

# # Pick thresholds (EER primary, Youden optional)
# thr_eer = safe_pick_threshold(all_labels_np, all_probs_np, mode="eer")
# thr_j   = safe_pick_threshold(all_labels_np, all_probs_np, mode="youden")

# # Predictions @ EER threshold
# preds = (all_probs_np >= 0.5).astype(int)

# # Aggregate metrics
# acc = float((preds == all_labels_np).mean()) if all_labels_np.size else 0.0
# auc_score, pauc_score, eer, _, _ = compute_eer_auc(all_labels_np, all_probs_np)

# # Per-class breakdown
# real_total = int(np.sum(all_labels_np == 0))
# fake_total = int(np.sum(all_labels_np == 1))
# real_correct = int(np.sum((preds == 0) & (all_labels_np == 0)))
# fake_correct = int(np.sum((preds == 1) & (all_labels_np == 1)))
# real_acc = (real_correct / real_total * 100) if real_total > 0 else 0.0
# fake_acc = (fake_correct / fake_total * 100) if fake_total > 0 else 0.0

# # --- Print Summary ---
# print(f"\nOperating threshold (EER): {thr_eer:.4f}  |  Youden-J: {thr_j:.4f}")
# print(f"Test Results:")
# print(f"Loss={test_loss:.4f}, Accuracy={acc:.4f}")
# print(f"AUC={auc_score:.4f}, pAUC={pauc_score:.4f}, EER={eer:.4f}")
# print(f"Real samples (class: 0) --> correct ({real_correct}/{real_total}) ({real_acc:.1f}%)")
# print(f"Fake samples (class: 1) --> correct ({fake_correct}/{fake_total}) ({fake_acc:.1f}%)")
# print("Testing complete.")


# test_au_patch.py
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use("Agg")  # headless save
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc, average_precision_score
# from sklearn.manifold import TSNE

# from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
# from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention


# # ================== Config (edit here, no CLI) ==================
# DATA_ROOT = "Dataset/AU_Files/fakeavceleb_whole_image_patches"
# CSV_PATH  = "Dataset/meta_data.csv"
# CKPT_PATH = "Checkpoints/ff_au_patch_60fps_lstm_best_model_with_attn_5.pth"

# DEVICE = "cuda:1"  # falls back to CPU if not available
# SPLIT  = "eval"    # "eval" or "test"

# BATCH_SIZE = 2
# NUM_WORKERS = 2
# IMAGE_SIZE = 128
# MAX_FRAMES = 50
# MAX_AUS    = 17

# INCLUDE_UNMATCHED_REAL = True
# UNMATCHED_SPLIT_SEED   = 42

# TSNE_MAX_SAMPLES = 2000
# TSNE_OUT = "tsne_plot.png"
# # ================================================================


# # ---------------- Metrics ----------------
# def compute_eer_auc(labels, scores):
#     labels = np.asarray(labels).astype(int).ravel()
#     scores = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
#     fnr = 1 - tpr
#     auc_score = auc(fpr, tpr)
#     pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
#     # EER by closest point to FPR=FNR
#     idx = int(np.nanargmin(np.abs(fpr - fnr)))
#     eer = float((fpr[idx] + fnr[idx]) / 2.0)
#     return auc_score, pauc, eer, (fpr, tpr)


# def youden_threshold(labels, scores):
#     labels = np.asarray(labels).astype(int).ravel()
#     scores = np.asarray(scores).astype(float).ravel()
#     fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
#     j = tpr - fpr
#     j_idx = int(np.argmax(j))
#     return float(thr[j_idx])


# @torch.no_grad()
# def extract_pooled_embeddings(model, patches, au_patch_weights=None):
#     """
#     Forward up to the pooled feature (before classifier).
#     Returns (B, 2*lstm_hidden) on CPU.
#     """
#     B, T, A, C, H, W = patches.size()
#     x = patches.view(B * T * A, C, H, W)
#     features = model.feature_extractor(x).view(B * T * A, -1)
#     features = model.au_fc(features)  # (B*T*A, hidden_dim)
#     features = features.view(B, T, A, model.hidden_dim)

#     attn_scores = model.attn(features)               # (B, T, A, 1)
#     attn_weights = torch.softmax(attn_scores, dim=2) # (B, T, A, 1)

#     if au_patch_weights is not None:
#         lw = au_patch_weights.unsqueeze(-1)          # (B, T, A, 1)
#         combined = attn_weights * lw
#         attn_weights = combined / (combined.sum(dim=2, keepdim=True) + 1e-6)

#     attended = (attn_weights * features).sum(dim=2)  # (B, T, hidden_dim)
#     lstm_out, _ = model.lstm(attended)               # (B, T, 2*lstm_hidden)
#     pooled = lstm_out.mean(dim=1)                    # (B, 2*lstm_hidden)
#     return pooled.detach().cpu()


# def main():
#     device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

#     # Load data (mirror training config)
#     train_loader, test_loader, eval_loader = get_patch_image_loaders(
#         data_root=DATA_ROOT,
#         csv_path=CSV_PATH,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         image_size=IMAGE_SIZE,
#         max_frames=MAX_FRAMES,
#         max_aus=MAX_AUS,
#         include_unmatched_real=INCLUDE_UNMATCHED_REAL,
#         unmatched_split_seed=UNMATCHED_SPLIT_SEED,
#     )
#     loader = eval_loader if SPLIT == "eval" else test_loader
#     print(f"Evaluating on split: {SPLIT} with {len(loader.dataset)} samples")

#     # Model
#     model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)
#     sd = torch.load(CKPT_PATH, map_location=device)
#     model.load_state_dict(sd, strict=True)
#     model.eval()

#     # Gather scores/labels/embeddings
#     all_scores, all_labels = [], []
#     all_embeds = []

#     with torch.no_grad():
#         for patches, weights_batch, labels in tqdm(loader, desc=f"Running {SPLIT}"):
#             patches = patches.to(device)
#             weights_batch = weights_batch.to(device)
#             labels = labels.to(device).float().view(-1)

#             logits = model(patches, au_patch_weights=weights_batch).view(-1)
#             # match training eval scaling (/2.0) then sigmoid
#             probs = torch.sigmoid((logits / 2.0).float())

#             all_scores.extend(probs.detach().cpu().numpy().tolist())
#             all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

#             pooled = extract_pooled_embeddings(model, patches, au_patch_weights=weights_batch)
#             all_embeds.append(pooled)

#     # Stack
#     all_scores = np.asarray(all_scores, dtype=np.float64)
#     all_labels = np.asarray(all_labels, dtype=int)
#     all_embeds = torch.cat(all_embeds, dim=0).numpy() if len(all_embeds) else np.zeros((0, 256), dtype=np.float32)

#     # Metrics
#     auc_score, pauc_score, eer, _ = compute_eer_auc(all_labels, all_scores)
#     thr = youden_threshold(all_labels, all_scores)
#     preds = (all_scores >= thr).astype(int)
#     acc = float((preds == all_labels).mean())
#     try:
#         ap = float(average_precision_score(all_labels, all_scores)) if (all_labels.min() != all_labels.max()) else float("nan")
#     except Exception:
#         ap = float("nan")

#     n_real = int((all_labels == 0).sum())
#     n_fake = int((all_labels == 1).sum())
#     cr = int(((preds == 0) & (all_labels == 0)).sum())
#     cf = int(((preds == 1) & (all_labels == 1)).sum())

#     print(f"\n[{SPLIT.upper()}] Results")
#     print(f"AUC={auc_score:.4f} | pAUC@0.1={pauc_score:.4f} | EER={eer:.4f}")
#     print(f"AP={ap:.4f} | Acc@Youden={acc:.4f} | thr*={thr:.3f}")
#     print(f"Counts: real={n_real} (correct={cr}), fake={n_fake} (correct={cf})")

#     # t-SNE
#     if all_embeds.shape[0] > 2:
#         N = all_embeds.shape[0]
#         if N > TSNE_MAX_SAMPLES:
#             idx = np.random.RandomState(0).choice(N, size=TSNE_MAX_SAMPLES, replace=False)
#             X = all_embeds[idx]
#             y = all_labels[idx]
#         else:
#             X = all_embeds
#             y = all_labels

#         perplexity = max(5, min(30, (X.shape[0] // 3) - 1)) if X.shape[0] > 10 else 5
#         print(f"Running t-SNE on {X.shape[0]} embeddings (perplexity={perplexity})...")
#         Z = TSNE(n_components=2, init="random", learning_rate="auto",
#                  perplexity=perplexity, n_iter=1000).fit_transform(X)

#         plt.figure(figsize=(8, 6))
#         mask_real = (y == 0)
#         mask_fake = (y == 1)
#         plt.scatter(Z[mask_real, 0], Z[mask_real, 1], s=8, alpha=0.7, label="Real (0)")
#         plt.scatter(Z[mask_fake, 0], Z[mask_fake, 1], s=8, alpha=0.7, label="Fake (1)")
#         plt.title(f"t-SNE of pooled features ({SPLIT})")
#         plt.legend(loc="best")
#         plt.tight_layout()
#         plt.savefig(TSNE_OUT, dpi=200)
#         print(f"t-SNE saved to: {TSNE_OUT}")
#     else:
#         print("Not enough embeddings for t-SNE; skipping.")


# if __name__ == "__main__":
#     main()



import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.amp import autocast

from Dataset.AUPatchFeatureLoader import get_patch_image_loaders
from Models.ResNetLSTM import AUPatchResNetClassifierWithAUAttention


# ---------------- Metrics ----------------
def compute_metrics(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)
    pauc = (
        auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1
        if np.sum(fpr <= 0.1) >= 2 else 0.0
    )
    ap = average_precision_score(labels, scores)

    # Equal Error Rate (EER)
    abs_diffs = np.abs(fpr - fnr)
    eer_idx = np.nanargmin(abs_diffs)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_thresh = thresholds[eer_idx]

    # Youden's J (maximizing TPR - FPR)
    j_scores = tpr - fpr
    youden_idx = np.argmax(j_scores)
    youden_thresh = thresholds[youden_idx]

    return auc_score, pauc, ap, eer, eer_thresh, youden_thresh


# ---------------- Device ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- Data ----------------
print("Loading data...")
# Now dataloader itself balances/augments if augment_* is set True
_, test_loader, _ = get_patch_image_loaders(
    data_root="Dataset/AU_Files/fakeavceleb_whole_image_patches",
    mode="fakeavceleb",
    csv_path="Dataset/meta_data.csv",
    lavdf_json=None,
    batch_size=2,
    image_size=128,
    max_frames=60,
    augment_train=True,   # training loader balanced+augmented
    augment_eval=False,   # keep eval clean (or True if you want balancing)
    augment_test=True     # duplicate+augment test for fairness
)

# ---------------- Model ----------------
model = AUPatchResNetClassifierWithAUAttention(hidden_dim=128, lstm_hidden=128).to(device)
ckpt_path = "Checkpoints/best_au_patch_model_last.pth"
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded checkpoint from {ckpt_path}")

# ---------------- Collect all scores ----------------
all_probs, all_labels = [], []

with torch.no_grad():
    for patches, weights_batch, labels in tqdm(test_loader, desc="Collecting Scores"):
        patches, weights_batch, labels = (
            patches.to(device),
            weights_batch.to(device),
            labels.to(device),
        )
        with autocast("cuda"):
            outputs = model(patches, au_patch_weights=weights_batch)
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.view(-1)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ---------------- Metrics + optimal thresholds ----------------
auc_score, pauc_score, ap_score, eer, eer_thresh, youden_thresh = compute_metrics(all_labels, all_probs)

print("\n=== Global Metrics (threshold-independent) ===")
print(f"AUC:   {auc_score:.4f}")
print(f"pAUC:  {pauc_score:.4f}")
print(f"AP:    {ap_score:.4f}")
print(f"EER:   {eer:.4f}")
print(f"EER-threshold:    {eer_thresh:.4f}")
print(f"Youden-threshold: {youden_thresh:.4f}")

# ---------------- Evaluate at optimal thresholds ----------------
for name, thresh in [("0.5 (default)", 0.5),
                     ("EER-optimal", eer_thresh),
                     ("Youden-optimal", youden_thresh)]:
    preds = (all_probs > thresh).astype(int)

    acc = (preds == all_labels).mean()
    correct_real = ((preds == 0) & (all_labels == 0)).sum()
    total_real = (all_labels == 0).sum()
    correct_fake = ((preds == 1) & (all_labels == 1)).sum()
    total_fake = (all_labels == 1).sum()

    print(f"\n=== Evaluation at {name} threshold ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Correct Real: {correct_real}/{total_real} | Correct Fake: {correct_fake}/{total_fake}")