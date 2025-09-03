# test_joint_au_face.py
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
from sklearn.manifold import TSNE
from torch.amp import autocast

from Models.AUFaceModel import AUFaceCrossDetector
from Dataset.AuVidDataset import get_joint_dataloader


# =========================
# Config (edit here)
# =========================
DEVICE             = "cuda:0" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH       = "Checkpoints/auface_cross_best_auc_optionB.pth"
OUTPUT_DIR         = "Checkpoints"

# Data roots (LAVDF structure)
VIDEO_ROOT         = "/media/rt0706/Media/VCBSL-Dataset/FAVC_Whole/frames"  # npy face-crop arrays
AU_ROOT            = "Dataset/AU_Files/fakeavceleb_whole_image_patches"               # AU patch folders

# LAVDF label source
LAVDF_MODE         = False
LAVDF_JSON_PATH    = "Dataset/LAV-DF/metadata.min.json"

# If you ever want CSV mode, set LAVDF_MODE=False and pass csv_path to get_joint_dataloader from here.
CSV_PATH         = "Dataset/meta_data.csv"

BATCH_SIZE         = 2
MAX_FRAMES         = 75
MAX_AUS            = 17
IMAGE_SIZE         = 128
NUM_WORKERS        = 2

# Which split to evaluate first ("eval" or "test"); will auto-fallback if empty
PRIMARY_SPLIT      = "test"

# Optional sign auto-fix (if logits are inverted relative to labels)
SIGN_POLICY        = "auto"   # "none" | "auto" | "flip"
SIGN_MARGIN        = 1e-6

TSNE_N_COMPONENTS  = 2
TSNE_PERPLEXITY    = 30
TSNE_N_ITER        = 1500
TSNE_MAX_SAMPLES   = 2000

SEED               = 42


# =========================
# Metrics (mirrors train code)
# =========================
def compute_eer_auc(labels, scores):
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    fnr = 1 - tpr
    auc_score = auc(fpr, tpr)
    pauc = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1]) / 0.1 if np.sum(fpr <= 0.1) >= 2 else 0.0
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return auc_score, pauc, eer


def pick_threshold(labels, scores, mode="youden", fpr_target=0.01):
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)

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


# =========================
# Safe checkpoint loading
# =========================
def _unwrap_state_dict(raw):
    """Handle common checkpoint containers: EMA, Lightning, DP, etc."""
    state = raw
    # common top-level keys
    for k in ["state_dict", "model", "ema_state_dict", "model_ema", "ema", "net", "module"]:
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]

    # drop EMA bookkeeping if present
    if isinstance(state, dict):
        state.pop("n_averaged", None)

    # strip leading "module." if saved under DataParallel
    if isinstance(state, dict):
        new_state = {}
        for kk, vv in state.items():
            new_state[kk[7:]] = vv if kk.startswith("module.") else vv
        state = new_state
    return state


def load_state_dict_flexible(model, path):
    raw = torch.load(path, map_location="cpu")
    state = _unwrap_state_dict(raw)
    try:
        model.load_state_dict(state, strict=True)
        print(f"[Load] {path} ✓ (strict=True)")
    except Exception as e:
        print(f"[Load] strict=True failed -> {e.__class__.__name__}: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Load] strict=False | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("        First few missing:", missing[:8])
        if unexpected:
            print("        First few unexpected:", unexpected[:8])


# =========================
# Feature collection
# =========================
@torch.no_grad()
def collect_features(loader, model, device, use_amp=True):
    """
    Returns:
      feats_face : (N, D_face) mean over time from v_tokens
      feats_au   : (N, D_au)   mean over time from au_tokens
      labels     : (N,)
      scores     : (N,) sigmoid(logits)
    """
    model.eval()
    all_face, all_au, all_lab, all_score = [], [], [], []

    for videos, au_patches, labels in tqdm(loader, desc="Collecting features"):
        # videos may be (B, T, C, H, W) or (B, C, T, H, W). Model expects (B, C, T, H, W).
        if videos.dim() == 5 and videos.shape[1] != 3 and videos.shape[2] == 3:
            videos = videos.permute(0, 2, 1, 3, 4).contiguous()

        videos     = videos.to(device, non_blocking=True)
        au_patches = au_patches.to(device, non_blocking=True)
        labels_np  = labels.cpu().numpy().astype(int)

        with autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp)):
            logits, v_tokens, au_tokens = model(videos, au_patches)

        probs    = torch.sigmoid(logits).detach().flatten().cpu().numpy()
        face_mu  = v_tokens.mean(dim=1).detach().cpu().numpy()  # (B, D_face)
        au_mu    = au_tokens.mean(dim=1).detach().cpu().numpy() # (B, D_au)

        all_face.append(face_mu)
        all_au.append(au_mu)
        all_lab.append(labels_np)
        all_score.append(probs)

    feats_face = np.concatenate(all_face, axis=0) if all_face else np.zeros((0, 1))
    feats_au   = np.concatenate(all_au, axis=0)   if all_au   else np.zeros((0, 1))
    labels_out = np.concatenate(all_lab, axis=0)  if all_lab  else np.zeros((0,), dtype=int)
    scores_out = np.concatenate(all_score, axis=0) if all_score else np.zeros((0,), dtype=float)
    return feats_face, feats_au, labels_out, scores_out


# =========================
# t-SNE plotting
# =========================
def run_tsne_and_plot(X, y, title, save_path, seed=SEED):
    if X.shape[0] == 0:
        print(f"[t-SNE] No data for {title}; skipped.")
        return

    if TSNE_MAX_SAMPLES is not None and X.shape[0] > TSNE_MAX_SAMPLES:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=TSNE_MAX_SAMPLES, replace=False)
        X = X[idx]
        y = y[idx]

    tsne = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=min(TSNE_PERPLEXITY, max(5, (X.shape[0] - 1) // 3)),
        n_iter=TSNE_N_ITER,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(7, 6))
    mask_real = (y == 0)
    mask_fake = (y == 1)
    plt.scatter(Z[mask_real, 0], Z[mask_real, 1], s=12, alpha=0.6, label="real")
    plt.scatter(Z[mask_fake, 0], Z[mask_fake, 1], s=12, alpha=0.6, label="fake")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"[t-SNE] Saved -> {save_path}")


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = torch.device(DEVICE)

    # --- Data ---
    if LAVDF_MODE:
        train_loader, test_loader, eval_loader = get_joint_dataloader(
            video_root=VIDEO_ROOT,
            au_root=AU_ROOT,
            batch_size=BATCH_SIZE,
            shuffle=False,
            max_frames=MAX_FRAMES,
            max_aus=MAX_AUS,
            image_size=IMAGE_SIZE,
            num_workers=NUM_WORKERS,
            lavdf_mode=True,
            lavdf_json_path=LAVDF_JSON_PATH,
        )
        print("[Data] LAVDF mode ON (JSON labels).")
    else:
        train_loader, test_loader, eval_loader = get_joint_dataloader(
            video_root=VIDEO_ROOT,
            au_root=AU_ROOT,
            batch_size=BATCH_SIZE,
            shuffle=False,
            max_frames=MAX_FRAMES,
            max_aus=MAX_AUS,
            image_size=IMAGE_SIZE,
            num_workers=NUM_WORKERS,
            csv_path=CSV_PATH,
        )
        print("[Data] CSV mode ON.")

    # --- Choose split with fallback ---
    split_request = (PRIMARY_SPLIT or "eval").lower().strip()
    loader = eval_loader if split_request == "eval" else test_loader
    split_used = "eval" if split_request == "eval" else "test"

    def _len_safe(dl):
        try:
            return len(dl.dataset)
        except Exception:
            return 0

    if (loader is None) or (_len_safe(loader) == 0):
        other = test_loader if split_used == "eval" else eval_loader
        if (other is None) or (_len_safe(other) == 0):
            raise RuntimeError("No valid data in either eval or test splits.")
        loader = other
        split_used = "test" if split_used == "eval" else "eval"
        print(f"[Data] Requested '{split_request}' is empty. Falling back to '{split_used}'.")

    print(f"[Data] Evaluating split: {split_used}  |  N={len(loader.dataset)}")

    # --- Model ---
    model = AUFaceCrossDetector(num_aus=MAX_AUS, face_dim=512, au_dim=512, lstm_hidden=256).to(device)
    assert os.path.isfile(WEIGHTS_PATH), f"Missing weights: {WEIGHTS_PATH}"
    load_state_dict_flexible(model, WEIGHTS_PATH)
    model.eval()

    # --- Collect features & scores (aligned with train forward signature) ---
    feats_face, feats_au, labels, scores = collect_features(loader, model, device, use_amp=True)

    # --- Optional sign auto-fix (if training used opposite logit convention) ---
    if SIGN_POLICY != "none" and labels.size > 0 and np.unique(labels).size > 1:
        auc_p  = roc_auc_score(labels, scores)
        auc_np = roc_auc_score(labels, 1.0 - scores)
        if SIGN_POLICY == "flip" or (SIGN_POLICY == "auto" and (auc_np > auc_p + SIGN_MARGIN)):
            scores = 1.0 - scores
            print(f"[Sign] Applied sign flip -> using 1 - p (AUC={max(auc_p, auc_np):.4f})")
        else:
            print(f"[Sign] Kept original sign (AUC={auc_p:.4f})")

    # --- Metrics & operating points (mirrors train printouts) ---
    auc_score, pauc, eer = compute_eer_auc(labels, scores)
    thr_youden, fpr_y, tpr_y = pick_threshold(labels, scores, mode="youden")
    acc_y, ap_y, cr_y, tr_y, cf_y, tf_y = compute_acc_ap_and_counts(labels, scores, thr=thr_youden)
    print(f"[{split_used.upper()}] AUC={auc_score:.4f} | pAUC@0.1={pauc:.4f} | EER={eer:.4f}")
    print(f"[{split_used.upper()}] Acc@thr={acc_y:.4f}, AP={ap_y:.4f}, "
          f"Correct[real]={cr_y}/{tr_y}, Correct[fake]={cf_y}/{tf_y}, "
          f"thr={thr_youden:.3f}, FPR={fpr_y:.3f}, TPR={tpr_y:.3f}")

    # Low-FPR operating point (5%) for reference
    thr_f, fpr_f, tpr_f = pick_threshold(labels, scores, mode="fpr", fpr_target=0.05)
    acc_f, ap_f, cr_f, tr_f, cf_f, tf_f = compute_acc_ap_and_counts(labels, scores, thr=thr_f)
    print(f"[{split_used.upper()}@FPR≤5%] Acc={acc_f:.4f}, AP={ap_f:.4f}, thr={thr_f:.3f}, "
          f"FPR={fpr_f:.3f}, TPR={tpr_f:.3f}, Correct[real]={cr_f}/{tr_f}, Correct[fake]={cf_f}/{tf_f}")

    # Save scores/labels
    np.savez(
        os.path.join(OUTPUT_DIR, f"{split_used}_scores_and_labels.npz"),
        scores=scores, labels=labels
    )

    # --- t-SNE plots ---
    run_tsne_and_plot(
        feats_face, labels,
        title=f"t-SNE (Face stream)  N={feats_face.shape[0]}  D={feats_face.shape[1]}",
        save_path=os.path.join(OUTPUT_DIR, f"tsne_face_{split_used}.png")
    )
    run_tsne_and_plot(
        feats_au, labels,
        title=f"t-SNE (AU stream)  N={feats_au.shape[0]}  D={feats_au.shape[1]}",
        save_path=os.path.join(OUTPUT_DIR, f"tsne_au_{split_used}.png")
    )
    run_tsne_and_plot(
        np.concatenate([feats_face, feats_au], axis=1),
        labels,
        title=f"t-SNE (Concat face+AU)  N={feats_face.shape[0]}  D={feats_face.shape[1]+feats_au.shape[1]}",
        save_path=os.path.join(OUTPUT_DIR, f"tsne_concat_{split_used}.png")
    )

    print("Done.")


if __name__ == "__main__":
    main()