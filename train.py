# """
# =============================================================================
#   Duality AI — Offroad Semantic Segmentation
#   DeepLabV3+ Full Training Script

#   Expected file structure:
#       train/
#           Color_Images/     <- RGB .png/.jpg images
#           Segmentation/     <- Label images (pixel values = raw class IDs)
#       val/
#           Color_Images/
#           Segmentation/
#       test/
#           Color_Images/
#           Segmentation/

#   Run:
#       conda activate EDU
#       python train.py

#   Outputs (saved to runs/):
#       best_model.pth         <- Best checkpoint by val IoU
#       training_curves.png    <- Loss + IoU over epochs
#       per_class_iou.png      <- Bar chart of per-class IoU
#       confusion_matrix.png   <- Normalized confusion matrix
#       sample_predictions.png <- Input | GT | Prediction side-by-side
# =============================================================================
# """

# import os
# import cv2
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import segmentation_models_pytorch as smp
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")


# # =============================================================================
# #  CONFIGURATION  — Edit these values before running
# # =============================================================================

# CONFIG = {
#     # ── Paths ──────────────────────────────────────────────────────────────
#     "train_rgb":  "train/Color_Images",
#     "train_seg":  "train/Segmentation",
#     "val_rgb":    "val/Color_Images",
#     "val_seg":    "val/Segmentation",
#     "test_rgb":   "test/Color_Images",
#     "test_seg":   "test/Segmentation",
#     "output_dir": "runs",

#     # ── Model ──────────────────────────────────────────────────────────────
#     "encoder":         "resnet101",   # or: resnet50, efficientnet-b4
#     "encoder_weights": "imagenet",
#     "num_classes":     10,

#     # ── Training ───────────────────────────────────────────────────────────
#     "image_size":    512,
#     "batch_size":    8,               # reduce to 4 if GPU runs out of memory
#     "epochs":        60,
#     "backbone_lr":   1e-4,            # lower LR for pretrained encoder
#     "head_lr":       5e-4,            # higher LR for decoder + head
#     "weight_decay":  1e-4,
#     "warmup_epochs": 5,

#     # ── Loss weights ───────────────────────────────────────────────────────
#     "ce_weight":   0.6,
#     "dice_weight": 0.4,

#     # ── Misc ───────────────────────────────────────────────────────────────
#     "num_workers": 4,                 # set 0 on Windows if errors occur
#     "seed":        42,
#     "save_every":  5,
# }


# # =============================================================================
# #  CLASS DEFINITIONS
# # =============================================================================

# # Raw pixel value in segmentation image → compact class index 0-9
# CLASS_MAP = {
#     100:   0,   # Trees
#     200:   1,   # Lush Bushes
#     300:   2,   # Dry Grass
#     500:   3,   # Dry Bushes
#     550:   4,   # Ground Clutter
#     600:   5,   # Flowers
#     700:   6,   # Logs
#     800:   7,   # Rocks
#     7100:  8,   # Landscape
#     10000: 9,   # Sky
# }

# CLASS_NAMES = [
#     "Trees",          # 0
#     "Lush Bushes",    # 1
#     "Dry Grass",      # 2
#     "Dry Bushes",     # 3
#     "Ground Clutter", # 4
#     "Flowers",        # 5
#     "Logs",           # 6
#     "Rocks",          # 7
#     "Landscape",      # 8
#     "Sky",            # 9
# ]

# CLASS_COLORS = [
#     (34,  139, 34),   # Trees          Forest Green
#     (0,   200, 100),  # Lush Bushes    Bright Green
#     (210, 180, 140),  # Dry Grass      Tan
#     (139, 90,  43),   # Dry Bushes     Brown
#     (128, 128, 128),  # Ground Clutter Gray
#     (255, 20,  147),  # Flowers        Deep Pink
#     (101, 67,  33),   # Logs           Dark Brown
#     (200, 200, 200),  # Rocks          Light Gray
#     (245, 222, 179),  # Landscape      Sandy Wheat
#     (135, 206, 250),  # Sky            Sky Blue
# ]


# # =============================================================================
# #  UTILITY FUNCTIONS
# # =============================================================================

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def remap_label(seg_image):
#     """
#     Convert raw pixel values (100, 200 ... 10000) → class indices (0–9).
#     Unmapped pixels are set to 255 (ignored in loss computation).
#     """
#     out = np.full(seg_image.shape, 255, dtype=np.int64)
#     for raw_val, class_idx in CLASS_MAP.items():
#         out[seg_image == raw_val] = class_idx
#     return out


# def colorize_prediction(pred_map):
#     """Convert 2D class index map → RGB image for visualization."""
#     h, w = pred_map.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     for cls_idx, color in enumerate(CLASS_COLORS):
#         rgb[pred_map == cls_idx] = color
#     return rgb


# def find_image_pairs(rgb_dir, seg_dir):
#     """
#     Match each RGB image with its segmentation file.
#     Tries multiple common naming conventions automatically.
#     """
#     if not os.path.exists(rgb_dir):
#         print(f"  [WARNING] Directory not found: {rgb_dir}")
#         return []
#     if not os.path.exists(seg_dir):
#         print(f"  [WARNING] Directory not found: {seg_dir}")
#         return []

#     rgb_files = sorted([
#         f for f in os.listdir(rgb_dir)
#         if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#     ])

#     pairs, missing = [], []

#     for rgb_name in rgb_files:
#         stem = os.path.splitext(rgb_name)[0]
#         candidates = [
#             rgb_name,
#             stem + ".png",
#             stem + "_seg.png",
#             stem + "_segmentation.png",
#             stem + "_label.png",
#             stem.replace("_rgb", "") + ".png",
#             stem.replace("_color", "") + ".png",
#             stem.replace("_Color", "") + ".png",
#         ]
#         found = None
#         for c in candidates:
#             if os.path.exists(os.path.join(seg_dir, c)):
#                 found = c
#                 break
#         if found:
#             pairs.append((
#                 os.path.join(rgb_dir, rgb_name),
#                 os.path.join(seg_dir, found)
#             ))
#         else:
#             missing.append(rgb_name)

#     if missing:
#         print(f"  [WARNING] No segmentation match for {len(missing)} images "
#               f"(first: {missing[0]})")

#     return pairs


# def compute_class_weights(pairs, num_classes=10):
#     """
#     Inverse-frequency class weights so rare classes (Flowers, Logs)
#     get higher loss weight during training.
#     """
#     print("\n  Computing class frequency (this may take a minute) ...")
#     freq = np.zeros(num_classes, dtype=np.float64)

#     for _, seg_path in tqdm(pairs, desc="  Scanning seg files"):
#         seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
#         if seg is None:
#             continue
#         if seg.ndim == 3:
#             seg = seg[:, :, 0]
#         seg_mapped = remap_label(seg)
#         for c in range(num_classes):
#             freq[c] += (seg_mapped == c).sum()

#     weights = 1.0 / (freq + 1.0)
#     weights = weights / weights.mean()

#     print("\n  Class Weights:")
#     for i, (name, w, f) in enumerate(zip(CLASS_NAMES, weights, freq)):
#         bar = "█" * min(int(w * 4), 35)
#         print(f"    [{i}] {name:<16} pixels={f:>12.0f}  w={w:6.3f}  {bar}")

#     return torch.tensor(weights, dtype=torch.float32)


# def compute_sample_weights(pairs):
#     """
#     Oversample images containing rare classes (Flowers=5, Logs=6).
#     """
#     print("\n  Computing sample weights for oversampling ...")
#     RARE = {5, 6}
#     weights = []
#     for _, seg_path in tqdm(pairs, desc="  Scanning"):
#         seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
#         if seg is None:
#             weights.append(1.0)
#             continue
#         if seg.ndim == 3:
#             seg = seg[:, :, 0]
#         seg_mapped = remap_label(seg)
#         rare_px = sum((seg_mapped == r).sum() for r in RARE)
#         weights.append(1.0 + rare_px * 0.005)
#     return weights


# # =============================================================================
# #  DATASET
# # =============================================================================

# class DesertSegDataset(Dataset):

#     def __init__(self, pairs, transform=None):
#         self.pairs     = pairs
#         self.transform = transform

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         rgb_path, seg_path = self.pairs[idx]

#         # Load RGB
#         img = cv2.imread(rgb_path)
#         if img is None:
#             raise FileNotFoundError(f"Cannot read: {rgb_path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Load segmentation
#         seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
#         if seg is None:
#             raise FileNotFoundError(f"Cannot read: {seg_path}")
#         if seg.ndim == 3:
#             seg = seg[:, :, 0]          # use first channel only

#         seg = remap_label(seg).astype(np.int64)

#         if self.transform:
#             aug = self.transform(image=img, mask=seg.astype(np.int32))
#             img = aug["image"]
#             seg = aug["mask"].long()
#         else:
#             img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
#             seg = torch.from_numpy(seg).long()

#         return img, seg


# # =============================================================================
# #  AUGMENTATION PIPELINES
# # =============================================================================

# def get_train_transform(image_size):
#     """
#     Training augmentation pipeline - compatible with all Albumentations versions.
#     Achieves multi-scale training through combination of Resize + RandomScale.
#     """
#     return A.Compose([
#         # Resize to base size, then apply random scaling
#         A.LongestMaxSize(max_size=int(image_size * 1.5), p=1.0),
#         A.RandomScale(scale_limit=0.5, p=0.8),  # 0.5x to 1.5x scale variation
#         A.PadIfNeeded(
#             min_height=image_size, 
#             min_width=image_size,
#             border_mode=cv2.BORDER_REFLECT_101,
#             p=1.0
#         ),
#         A.RandomCrop(height=image_size, width=image_size, p=1.0),
        
#         # Geometric transforms
#         A.HorizontalFlip(p=0.5),
#         A.ShiftScaleRotate(
#             shift_limit=0.1, 
#             scale_limit=0.1, 
#             rotate_limit=10,
#             border_mode=cv2.BORDER_REFLECT_101, 
#             p=0.5
#         ),
        
#         # Color transforms
#         A.ColorJitter(
#             brightness=0.3, 
#             contrast=0.3, 
#             saturation=0.2, 
#             hue=0.05, 
#             p=0.8
#         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2, 
#             contrast_limit=0.2, 
#             p=0.4
#         ),
        
#         # Blur & noise
#         A.OneOf([
#             A.GaussianBlur(blur_limit=(3, 7), p=1.0),
#             A.MedianBlur(blur_limit=5, p=1.0),
#         ], p=0.3),
        
#         A.ToGray(p=0.05),
        
#         # Normalization (ImageNet stats)
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])


# def get_val_transform(image_size):
#     return A.Compose([
#         A.Resize(image_size, image_size),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])


# # =============================================================================
# #  MODEL
# # =============================================================================

# def build_model(cfg):
#     model = smp.DeepLabV3Plus(
#         encoder_name    = cfg["encoder"],
#         encoder_weights = cfg["encoder_weights"],
#         in_channels     = 3,
#         classes         = cfg["num_classes"],
#         activation      = None,     # raw logits
#     )
#     return model


# # =============================================================================
# #  LOSS
# # =============================================================================

# class CombinedLoss(nn.Module):
#     """
#     Weighted CrossEntropy + Dice.
#     CE handles class imbalance via per-class weights.
#     Dice improves boundary sharpness and rare object recall.
#     """
#     def __init__(self, class_weights, ce_w=0.6, dice_w=0.4, ignore_index=255):
#         super().__init__()
#         self.ce_w   = ce_w
#         self.dice_w = dice_w
#         self.ce = nn.CrossEntropyLoss(
#             weight=class_weights, ignore_index=ignore_index
#         )
#         self.dice = smp.losses.DiceLoss(
#             mode="multiclass", ignore_index=ignore_index
#         )

#     def forward(self, logits, targets):
#         return self.ce_w * self.ce(logits, targets) + \
#                self.dice_w * self.dice(logits, targets)


# # =============================================================================
# #  EVALUATION
# # =============================================================================

# @torch.no_grad()
# def evaluate(model, loader, device, num_classes=10):
#     model.eval()
#     intersection = np.zeros(num_classes, dtype=np.float64)
#     union_arr    = np.zeros(num_classes, dtype=np.float64)

#     for imgs, masks in tqdm(loader, desc="  Evaluating", leave=False):
#         imgs  = imgs.to(device)
#         preds = model(imgs).argmax(dim=1).cpu().numpy().flatten()
#         m     = masks.numpy().flatten()

#         valid = m != 255
#         preds = preds[valid]
#         m     = m[valid]

#         for cls in range(num_classes):
#             pc = (preds == cls)
#             mc = (m == cls)
#             intersection[cls] += (pc & mc).sum()
#             union_arr[cls]    += (pc | mc).sum()

#     per_cls = np.where(
#         union_arr > 0,
#         intersection / (union_arr + 1e-8),
#         np.nan
#     )
#     return float(np.nanmean(per_cls)), per_cls


# def print_metrics_table(per_cls, mean_iou):
#     print("\n" + "=" * 55)
#     print(f"  {'CLASS':<18} {'IoU':>8}  STATUS")
#     print("=" * 55)
#     for name, iou in zip(CLASS_NAMES, per_cls):
#         if np.isnan(iou):
#             status = "(not in val set)"
#             print(f"  {name:<18} {'N/A':>8}  {status}")
#         else:
#             status = "✓ Good" if iou >= 0.5 else ("△ Ok" if iou >= 0.3 else "✗ Low")
#             print(f"  {name:<18} {iou:>8.4f}  {status}")
#     print("=" * 55)
#     print(f"  {'MEAN IoU':<18} {mean_iou:>8.4f}")
#     print("=" * 55 + "\n")


# # =============================================================================
# #  VISUALIZATION
# # =============================================================================

# def save_loss_curves(train_losses, val_ious, out_dir):
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     ep = range(1, len(train_losses) + 1)

#     axes[0].plot(ep, train_losses, "b-o", markersize=3, label="Train Loss")
#     axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].grid(True, alpha=0.4)

#     axes[1].plot(ep, val_ious, "g-o", markersize=3, label="Val Mean IoU")
#     axes[1].axhline(y=0.70, color="orange", linestyle="--", label="Target 0.70")
#     axes[1].set_title("Validation Mean IoU", fontsize=13, fontweight="bold")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Mean IoU")
#     axes[1].set_ylim(0, 1)
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.4)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "training_curves.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# def save_per_class_iou(per_cls, mean_iou, out_dir):
#     vals   = [float(v) if not np.isnan(v) else 0.0 for v in per_cls]
#     colors = ["green" if v >= 0.5 else "orange" if v >= 0.3 else "red" for v in vals]

#     fig, ax = plt.subplots(figsize=(12, 5))
#     bars = ax.barh(CLASS_NAMES, vals, color=colors, edgecolor="white", height=0.6)
#     ax.axvline(x=mean_iou, color="navy", linestyle="--", linewidth=2,
#                label=f"Mean IoU = {mean_iou:.4f}")

#     for bar, val in zip(bars, vals):
#         ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
#                 f"{val:.3f}", va="center", fontsize=9)

#     ax.set_xlim(0, 1.08)
#     ax.set_title(f"Per-Class IoU  |  Mean = {mean_iou:.4f}",
#                  fontsize=13, fontweight="bold")
#     ax.set_xlabel("IoU Score")
#     ax.legend(loc="lower right")
#     ax.grid(True, axis="x", alpha=0.3)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "per_class_iou.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# def save_confusion_matrix(model, loader, device, out_dir, num_classes=10):
#     print("\n  Building confusion matrix ...")
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for imgs, masks in tqdm(loader, desc="  CM pass", leave=False):
#             preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy().flatten()
#             m     = masks.numpy().flatten()
#             valid = m != 255
#             all_preds.extend(preds[valid])
#             all_labels.extend(m[valid])

#     cm      = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
#     cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

#     fig, ax = plt.subplots(figsize=(13, 11))
#     sns.heatmap(cm_norm, annot=True, fmt=".2f",
#                 xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
#                 cmap="Blues", ax=ax, linewidths=0.3)
#     ax.set_title("Normalized Confusion Matrix — Val Set",
#                  fontsize=13, fontweight="bold")
#     ax.set_ylabel("True Class")
#     ax.set_xlabel("Predicted Class")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()

#     path = os.path.join(out_dir, "confusion_matrix.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# def save_sample_predictions(model, dataset, device, out_dir, n=6):
#     print(f"\n  Saving {n} sample predictions ...")
#     model.eval()
#     MEAN = np.array([0.485, 0.456, 0.406])
#     STD  = np.array([0.229, 0.224, 0.225])
#     idxs = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

#     patches = [
#         mpatches.Patch(color=[c / 255 for c in CLASS_COLORS[i]], label=CLASS_NAMES[i])
#         for i in range(len(CLASS_NAMES))
#     ]

#     fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))
#     if n == 1:
#         axes = axes[np.newaxis, :]

#     for row, idx in enumerate(idxs):
#         img_t, mask_t = dataset[int(idx)]
#         img_np = (img_t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)

#         with torch.no_grad():
#             pred = model(img_t.unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()

#         axes[row, 0].imshow(img_np)
#         axes[row, 0].set_title("Input", fontsize=11)
#         axes[row, 0].axis("off")

#         axes[row, 1].imshow(colorize_prediction(mask_t.numpy()))
#         axes[row, 1].set_title("Ground Truth", fontsize=11)
#         axes[row, 1].axis("off")

#         axes[row, 2].imshow(colorize_prediction(pred))
#         axes[row, 2].set_title("Prediction", fontsize=11)
#         axes[row, 2].axis("off")

#     fig.legend(handles=patches, loc="lower center", ncol=5,
#                fontsize=9, bbox_to_anchor=(0.5, -0.01))
#     plt.suptitle("DeepLabV3+ Sample Predictions", fontsize=14,
#                  fontweight="bold", y=1.01)
#     plt.tight_layout()

#     path = os.path.join(out_dir, "sample_predictions.png")
#     plt.savefig(path, dpi=120, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# # =============================================================================
# #  TRAINING LOOP
# # =============================================================================

# def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total):
#     model.train()
#     total_loss = 0.0
#     loop = tqdm(loader, desc=f"  Epoch {epoch:03d}/{total:03d} [Train]", leave=False)

#     for imgs, masks in loop:
#         imgs  = imgs.to(device)
#         masks = masks.to(device)

#         optimizer.zero_grad(set_to_none=True)

#         with autocast():
#             logits = model(imgs)
#             loss   = criterion(logits, masks)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()
#         loop.set_postfix(loss=f"{loss.item():.4f}")

#     return total_loss / len(loader)


# # =============================================================================
# #  MAIN
# # =============================================================================

# def main():
#     set_seed(CONFIG["seed"])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"\n{'='*60}")
#     print("  Duality AI — DeepLabV3+ Segmentation Training")
#     print(f"{'='*60}")
#     print(f"  Device  : {device}")
#     if device.type == "cuda":
#         print(f"  GPU     : {torch.cuda.get_device_name(0)}")
#         gb = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"  VRAM    : {gb:.1f} GB")
#     print(f"  Encoder : {CONFIG['encoder']}")
#     print(f"  Epochs  : {CONFIG['epochs']}")
#     print(f"  Batch   : {CONFIG['batch_size']}")
#     print(f"  ImgSize : {CONFIG['image_size']}x{CONFIG['image_size']}")
#     print(f"{'='*60}\n")

#     os.makedirs(CONFIG["output_dir"], exist_ok=True)

#     # ── File pairs ──────────────────────────────────────────────────────────
#     print("  Resolving dataset file pairs ...")
#     train_pairs = find_image_pairs(CONFIG["train_rgb"], CONFIG["train_seg"])
#     val_pairs   = find_image_pairs(CONFIG["val_rgb"],   CONFIG["val_seg"])
#     test_pairs  = find_image_pairs(CONFIG["test_rgb"],  CONFIG["test_seg"])

#     print(f"\n  Train: {len(train_pairs)} pairs")
#     print(f"  Val  : {len(val_pairs)} pairs")
#     print(f"  Test : {len(test_pairs)} pairs")

#     if not train_pairs:
#         raise RuntimeError(
#             "No training image pairs found!\n"
#             f"Expected RGB images in  : {CONFIG['train_rgb']}\n"
#             f"Expected seg images in  : {CONFIG['train_seg']}\n"
#             "Check that filenames match between the two folders."
#         )

#     # ── Weights ─────────────────────────────────────────────────────────────
#     class_weights  = compute_class_weights(train_pairs, CONFIG["num_classes"])
#     sample_weights = compute_sample_weights(train_pairs)

#     # ── Datasets & Loaders ──────────────────────────────────────────────────
#     train_ds = DesertSegDataset(train_pairs, transform=get_train_transform(CONFIG["image_size"]))
#     val_ds   = DesertSegDataset(val_pairs,   transform=get_val_transform(CONFIG["image_size"]))
#     test_ds  = DesertSegDataset(test_pairs,  transform=get_val_transform(CONFIG["image_size"]))

#     sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

#     train_loader = DataLoader(
#         train_ds, batch_size=CONFIG["batch_size"], sampler=sampler,
#         num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
#         num_workers=CONFIG["num_workers"], pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
#         num_workers=CONFIG["num_workers"]
#     )

#     # ── Model ───────────────────────────────────────────────────────────────
#     print("\n  Building DeepLabV3+ ...")
#     model = build_model(CONFIG).to(device)
#     n_params = sum(p.numel() for p in model.parameters()) / 1e6
#     print(f"  Parameters: {n_params:.1f}M")

#     # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
#     criterion = CombinedLoss(
#         class_weights.to(device), CONFIG["ce_weight"], CONFIG["dice_weight"]
#     )

#     optimizer = AdamW([
#         {"params": model.encoder.parameters(),          "lr": CONFIG["backbone_lr"]},
#         {"params": model.decoder.parameters(),          "lr": CONFIG["head_lr"]},
#         {"params": model.segmentation_head.parameters(),"lr": CONFIG["head_lr"]},
#     ], weight_decay=CONFIG["weight_decay"])

#     warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
#                       total_iters=CONFIG["warmup_epochs"])
#     cosine = CosineAnnealingWarmRestarts(
#         optimizer, T_0=max(1, CONFIG["epochs"] - CONFIG["warmup_epochs"]), T_mult=1
#     )
#     scheduler = SequentialLR(optimizer, [warmup, cosine],
#                               milestones=[CONFIG["warmup_epochs"]])

#     scaler = GradScaler(enabled=(device.type == "cuda"))

#     # ── Training ────────────────────────────────────────────────────────────
#     train_losses, val_ious = [], []
#     best_iou, best_per_cls = 0.0, None

#     print(f"\n{'='*60}")
#     print("  Training ...")
#     print(f"{'='*60}\n")

#     for epoch in range(1, CONFIG["epochs"] + 1):
#         t0 = time.time()

#         loss = train_one_epoch(
#             model, train_loader, optimizer, criterion,
#             scaler, device, epoch, CONFIG["epochs"]
#         )
#         train_losses.append(loss)

#         mean_iou, per_cls = evaluate(model, val_loader, device, CONFIG["num_classes"])
#         val_ious.append(mean_iou)
#         scheduler.step()
#         lr = optimizer.param_groups[0]["lr"]

#         elapsed = time.time() - t0
#         print(
#             f"  Epoch {epoch:03d}/{CONFIG['epochs']:03d}  "
#             f"Loss={loss:.4f}  "
#             f"Val IoU={mean_iou:.4f}  "
#             f"LR={lr:.2e}  "
#             f"({elapsed:.0f}s)"
#         )

#         # Save best model
#         if mean_iou > best_iou:
#             best_iou     = mean_iou
#             best_per_cls = per_cls.copy()
#             ckpt_path    = os.path.join(CONFIG["output_dir"], "best_model.pth")
#             torch.save({
#                 "epoch":         epoch,
#                 "model_state":   model.state_dict(),
#                 "optimizer":     optimizer.state_dict(),
#                 "mean_iou":      mean_iou,
#                 "per_class_iou": per_cls.tolist(),
#                 "config":        CONFIG,
#             }, ckpt_path)
#             print(f"  *** New best: {best_iou:.4f}  → saved {ckpt_path}")

#         # Periodic checkpoint
#         if epoch % CONFIG["save_every"] == 0:
#             ep_ckpt = os.path.join(CONFIG["output_dir"], f"ckpt_ep{epoch:03d}.pth")
#             torch.save(model.state_dict(), ep_ckpt)

#     # ── Post-training reporting ─────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print("  Training complete — Generating reports")
#     print(f"{'='*60}")

#     # Load best weights
#     ckpt = torch.load(os.path.join(CONFIG["output_dir"], "best_model.pth"),
#                       map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     print(f"\n  Best model: Epoch {ckpt['epoch']}  IoU = {best_iou:.4f}")

#     print_metrics_table(best_per_cls, best_iou)
#     save_loss_curves(train_losses, val_ious, CONFIG["output_dir"])
#     save_per_class_iou(best_per_cls, best_iou, CONFIG["output_dir"])
#     save_confusion_matrix(model, val_loader, device, CONFIG["output_dir"])
#     save_sample_predictions(model, val_ds, device, CONFIG["output_dir"], n=6)

#     # Test set evaluation
#     if test_pairs:
#         print("\n  Evaluating on TEST set ...")
#         test_iou, test_per_cls = evaluate(model, test_loader, device, CONFIG["num_classes"])
#         print("\n  TEST SET:")
#         print_metrics_table(test_per_cls, test_iou)

#     print(f"\n  All outputs: ./{CONFIG['output_dir']}/")
#     print(f"  Best Val IoU: {best_iou:.4f}\n")
#     print("  Done!")


# if __name__ == "__main__":
#     main()


"""
=============================================================================
  Duality AI — DeepLabV3+ Inference / Test Script
=============================================================================
"""

import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Class definitions ───────────────────────────────────────────────────────

CLASS_MAP = {
    100:   0, 200:   1, 300:   2, 500:   3, 550:   4,
    600:   5, 700:   6, 800:   7, 7100:  8, 10000: 9,
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = [
    (34,139,34),(0,200,100),(210,180,140),(139,90,43),
    (128,128,128),(255,20,147),(101,67,33),(200,200,200),
    (245,222,179),(135,206,250),
]

NUM_CLASSES = 10
IMAGE_SIZE  = 512
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# =============================================================================
# HELPERS
# =============================================================================

def remap_label(seg):
    out = np.full(seg.shape, 255, dtype=np.int64)
    for raw_val, idx in CLASS_MAP.items():
        out[seg == raw_val] = idx
    return out

def colorize_prediction(pred_map):
    rgb = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        rgb[pred_map == cls_idx] = color
    return rgb

def overlay_prediction(img_rgb, pred_map, alpha=0.55):
    colored = colorize_prediction(pred_map).astype(np.float32)
    blended = (1 - alpha) * img_rgb.astype(np.float32) + alpha * colored
    return blended.clip(0, 255).astype(np.uint8)

def get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    encoder = cfg.get("encoder", "resnet101")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    print(f"  Checkpoint loaded")
    return model

# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict_tta(model, img_tensor, device):
    x = img_tensor.unsqueeze(0).to(device)
    h, w = x.shape[2:]
    probs_sum = torch.zeros(1, NUM_CLASSES, h, w, device=device)
    count = 0

    for flip in [False, True]:
        xi = x.flip(-1) if flip else x
        for scale in [0.75, 1.0, 1.25]:
            xs = F.interpolate(xi, size=(int(h*scale), int(w*scale)),
                               mode="bilinear", align_corners=False)
            logits = model(xs)
            logits = F.interpolate(logits, size=(h, w),
                                   mode="bilinear", align_corners=False)
            if flip:
                logits = logits.flip(-1)
            probs_sum += logits.softmax(dim=1)
            count += 1

    return (probs_sum / count).argmax(1).squeeze(0).cpu().numpy()

@torch.no_grad()
def predict_single(model, img_tensor, device):
    return model(img_tensor.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()

# =============================================================================
# IoU
# =============================================================================

def compute_full_iou(all_preds, all_labels):
    inter = np.zeros(NUM_CLASSES)
    uni   = np.zeros(NUM_CLASSES)

    for pred, label in zip(all_preds, all_labels):
        p = pred.flatten()
        l = label.flatten()
        valid = l != 255
        p, l = p[valid], l[valid]

        for c in range(NUM_CLASSES):
            pc = (p == c); lc = (l == c)
            inter[c] += (pc & lc).sum()
            uni[c]   += (pc | lc).sum()

    per_cls = np.where(uni > 0, inter / (uni + 1e-8), np.nan)
    return float(np.nanmean(per_cls)), per_cls

def save_iou_report(mean_iou, per_cls, out_dir):
    lines = ["="*50, "DeepLabV3+ IoU Report", "="*50, ""]
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_cls)):
        if np.isnan(iou):
            lines.append(f"[{i}] {name:<18}  N/A")
        else:
            lines.append(f"[{i}] {name:<18}  {iou:.4f}")
    lines += ["", "="*50, f"MEAN IoU : {mean_iou:.4f}", "="*50]

    path = os.path.join(out_dir, "test_iou_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nReport saved: {path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="runs/best_model.pth")
    parser.add_argument("--input",  default="test/Color_Images")
    parser.add_argument("--labels", default="test/Segmentation")
    parser.add_argument("--output", default="runs/test_output")
    parser.add_argument("--tta",    action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")

    # NEW
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of test images")
    
    args = parser.parse_args()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.model, device)
    transform = get_transform()

    img_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])
    # LIMIT NUMBER OF IMAGES
    max_images = 310   # <-- change this number anytime you want

    if len(img_files) > max_images:
        img_files = img_files[:max_images]
        print(f"Using only first {len(img_files)} images")

    print(f"\nFound {len(img_files)} images")

    if args.max_images is not None:
        img_files = img_files[:args.max_images]
        print(f"Using only first {len(img_files)} images")

    has_labels = os.path.isdir(args.labels) and bool(os.listdir(args.labels))
    all_preds, all_labels = [], []
    total_time = 0.0

    for fname in tqdm(img_files, desc="Inferencing"):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.input, fname)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        img_t = transform(image=img_rgb)["image"]

        t0 = time.time()
        pred_sm = predict_tta(model, img_t, device) if args.tta else \
                  predict_single(model, img_t, device)
        total_time += time.time() - t0

        pred_full = cv2.resize(pred_sm.astype(np.uint8),
                               (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)

        all_preds.append(pred_full)

        if has_labels:
            label_path = os.path.join(args.labels, fname)
            if os.path.exists(label_path):
                seg = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                if seg is not None:
                    if seg.ndim == 3:
                        seg = seg[:,:,0]
                    all_labels.append(remap_label(seg))

    avg_ms = total_time / max(len(img_files),1) * 1000
    print(f"\nAvg time: {avg_ms:.1f} ms/image")

    if all_labels and len(all_labels) == len(all_preds):
        mean_iou, per_cls = compute_full_iou(all_preds, all_labels)
        save_iou_report(mean_iou, per_cls, args.output)

    print(f"\nDone. Outputs saved to: {args.output}")

if __name__ == "__main__":
    main()
