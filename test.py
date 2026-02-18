"""
=============================================================================
  Duality AI — DeepLabV3+ Inference / Test Script

  Loads best_model.pth and runs inference on:
      test/Color_Images/          <- unlabeled test images

  Outputs saved to runs/test_output/:
      <image_name>_pred.png       <- colorized segmentation overlay
      <image_name>_raw.png        <- raw class-index map
      test_iou_report.txt         <- IoU report (if test/Segmentation/ exists)

  Usage:
      conda activate EDU
      python test.py
      python test.py --model runs/best_model.pth --input test/Color_Images
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

# ── Class definitions (must match train.py) ──────────────────────────────────

CLASS_MAP = {
    100:   0,
    200:   1,
    300:   2,
    500:   3,
    550:   4,
    600:   5,
    700:   6,
    800:   7,
    7100:  8,
    10000: 9,
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = [
    (34,  139, 34),
    (0,   200, 100),
    (210, 180, 140),
    (139, 90,  43),
    (128, 128, 128),
    (255, 20,  147),
    (101, 67,  33),
    (200, 200, 200),
    (245, 222, 179),
    (135, 206, 250),
]

NUM_CLASSES = 10
IMAGE_SIZE  = 512
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# =============================================================================
#  HELPERS
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
    ckpt    = torch.load(ckpt_path, map_location=device)
    cfg     = ckpt.get("config", {})
    encoder = cfg.get("encoder", "resnet101")

    model = smp.DeepLabV3Plus(
        encoder_name    = encoder,
        encoder_weights = None,
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    mean_iou = ckpt.get("mean_iou", "?")
    print(f"  Checkpoint: epoch={epoch}  val_iou={mean_iou}")
    return model


# =============================================================================
#  INFERENCE (with optional TTA)
# =============================================================================

@torch.no_grad()
def predict_tta(model, img_tensor, device):
    """Horizontal flip + multi-scale TTA. Best accuracy."""
    x = img_tensor.unsqueeze(0).to(device)
    h, w = x.shape[2], x.shape[3]
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
    return (probs_sum / count).argmax(dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_single(model, img_tensor, device):
    """Single-pass inference. Faster."""
    return model(img_tensor.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()


# =============================================================================
#  IoU (when ground truth labels are available)
# =============================================================================

def compute_full_iou(all_preds, all_labels):
    inter = np.zeros(NUM_CLASSES)
    uni   = np.zeros(NUM_CLASSES)
    for pred, label in zip(all_preds, all_labels):
        p = pred.flatten()
        l = label.flatten()
        valid = l != 255
        p, l  = p[valid], l[valid]
        for c in range(NUM_CLASSES):
            pc = (p == c); lc = (l == c)
            inter[c] += (pc & lc).sum()
            uni[c]   += (pc | lc).sum()
    per_cls  = np.where(uni > 0, inter / (uni + 1e-8), np.nan)
    return float(np.nanmean(per_cls)), per_cls


def save_iou_report(mean_iou, per_cls, out_dir):
    lines = ["=" * 50, "  DeepLabV3+ Test Set IoU Report", "=" * 50, ""]
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_cls)):
        if np.isnan(iou):
            lines.append(f"  [{i}] {name:<18}  N/A  (not present)")
        else:
            status = "GOOD" if iou >= 0.5 else ("OK" if iou >= 0.3 else "LOW")
            lines.append(f"  [{i}] {name:<18}  {iou:.4f}  [{status}]")
    lines += ["", "=" * 50, f"  MEAN IoU : {mean_iou:.4f}", "=" * 50]

    path = os.path.join(out_dir, "test_iou_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\n  Report saved: {path}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="runs/best_model.pth")
    parser.add_argument("--input",  default="test/Color_Images")
    parser.add_argument("--labels", default="test/Segmentation")
    parser.add_argument("--output", default="runs/test_output")
    parser.add_argument("--tta",    action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print("  DeepLabV3+ Inference")
    print(f"{'='*50}")
    print(f"  Device : {device}")
    print(f"  Model  : {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  TTA    : {args.tta}")
    print(f"{'='*50}\n")

    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model}\n"
            "Run train.py first to generate it."
        )

    model     = load_model(args.model, device)
    transform = get_transform()

    img_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not img_files:
        raise RuntimeError(f"No images in {args.input}")
    print(f"\n  Found {len(img_files)} test images\n")

    has_labels = os.path.isdir(args.labels) and bool(os.listdir(args.labels))
    all_preds, all_labels = [], []
    total_time = 0.0

    # for fname in tqdm(img_files, desc="  Inferencing"):
    #     stem    = os.path.splitext(fname)[0]
    #     img_bgr = cv2.imread(os.path.join(args.input, fname))
    #     if img_bgr is None:
    #         continue
    #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #     orig_h, orig_w = img_rgb.shape[:2]

    #     img_t = transform(image=img_rgb)["image"]

    #     t0 = time.time()
    #     pred_sm = predict_tta(model, img_t, device) if args.tta else \
    #               predict_single(model, img_t, device)
    #     total_time += time.time() - t0

    #     # Resize back to original resolution
    #     pred_full = cv2.resize(
    #         pred_sm.astype(np.uint8), (orig_w, orig_h),
    #         interpolation=cv2.INTER_NEAREST
    #     )
    #     all_preds.append(pred_full)

    #     # Save overlay
    #     overlay = overlay_prediction(img_rgb, pred_full, alpha=0.55)
    #     cv2.imwrite(
    #         os.path.join(args.output, f"{stem}_pred.png"),
    #         cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    #     )
    #     # Save raw class map
    #     cv2.imwrite(
    #         os.path.join(args.output, f"{stem}_raw.png"),
    #         pred_full.astype(np.uint8)
    #     )

    #     # Load label if present
    #     if has_labels:
    #         for cand in [fname, stem + ".png", stem + "_seg.png"]:
    #             sp = os.path.join(args.labels, cand)
    #             if os.path.exists(sp):
    #                 seg = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
    #                 if seg is not None:
    #                     if seg.ndim == 3:
    #                         seg = seg[:, :, 0]
    #                     all_labels.append(remap_label(seg))
    #                 break

    # # Summary
    # avg_ms = total_time / max(len(img_files), 1) * 1000
    # print(f"\n  Avg time : {avg_ms:.1f} ms/image  "
    #       f"({'✓ <50ms' if avg_ms < 50 else '△ >50ms — try --no-tta'})")

    # IoU report
    if all_labels and len(all_labels) == len(all_preds):
        mean_iou, per_cls = compute_full_iou(all_preds, all_labels)
        save_iou_report(mean_iou, per_cls, args.output)

        vals   = [float(v) if not np.isnan(v) else 0.0 for v in per_cls]
        colors = ["green" if v >= 0.5 else "orange" if v >= 0.3 else "red" for v in vals]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(CLASS_NAMES, vals, color=colors, edgecolor="white", height=0.6)
        ax.axvline(x=mean_iou, color="navy", linestyle="--",
                   label=f"Mean IoU = {mean_iou:.4f}")
        ax.set_title("Test Set Per-Class IoU", fontsize=13, fontweight="bold")
        ax.set_xlabel("IoU Score")
        ax.set_xlim(0, 1.05)
        ax.legend(); ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "test_iou_chart.png"), dpi=150)
        plt.close()

    print(f"\n  Outputs saved to: {args.output}/")
    print("  Done!\n")


if __name__ == "__main__":
    main()
