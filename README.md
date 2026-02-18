# Duality AI — Offroad Semantic Segmentation
## DeepLabV3+ Training & Inference

**Project Report:** [View Full Report](https://drive.google.com/drive/folders/1iEL4NvOoinpvk_k5R0GjRhBF_ejBTquN?usp=sharing)
---

## File Structure

```
project/
  train/
    Color_Images/       <- RGB training images
    Segmentation/       <- Label images (raw class pixel values)
  val/
    Color_Images/
    Segmentation/
  test/
    Color_Images/       <- Unlabeled test images
    Segmentation/       <- (optional, for evaluation)
  train.py              <- Training script
  test.py               <- Inference script
  requirements.txt      <- Python dependencies
  runs/                 <- Created automatically
    best_model.pth
    training_curves.png
    per_class_iou.png
    confusion_matrix.png
    sample_predictions.png
    test_output/
```

---

## Setup

```bash
# Windows
cd ENV_SETUP && setup_env.bat

# Mac / Linux
conda create -n EDU python=3.9 -y
conda activate EDU
pip install -r requirements.txt
```

---

## Training

```bash
conda activate EDU
python train.py
```

Key settings in `CONFIG` at the top of `train.py`:
- `encoder`: `"resnet101"` (or `"resnet50"` for faster training)
- `batch_size`: `8` (reduce to `4` if GPU OOM)
- `epochs`: `60`
- `image_size`: `512`

---

## Inference / Test

```bash
# With TTA (best accuracy, slightly slower)
python test.py

# Without TTA (faster, < 50ms per image)
python test.py --no-tta

# Custom paths
python test.py --model runs/best_model.pth --input test/Color_Images
```

Outputs saved to `runs/test_output/`:
- `*_pred.png` — colorized segmentation overlay on original image
- `*_raw.png`  — raw class index map (integer values 0-9)
- `test_iou_report.txt` — per-class IoU if labels available

---

## Class Mapping

| Raw Pixel Value | Class Index | Class Name     |
|----------------|-------------|----------------|
| 100            | 0           | Trees          |
| 200            | 1           | Lush Bushes    |
| 300            | 2           | Dry Grass      |
| 500            | 3           | Dry Bushes     |
| 550            | 4           | Ground Clutter |
| 600            | 5           | Flowers        |
| 700            | 6           | Logs           |
| 800            | 7           | Rocks          |
| 7100           | 8           | Landscape      |
| 10000          | 9           | Sky            |

---

## Expected Outputs

- Training loss should **steadily decrease** over epochs
- Val IoU should climb and plateau — target **≥ 0.60** mean IoU
- Rare classes (Flowers, Logs) start low — oversampling + class weights help
- Benchmark inference speed: **< 50ms/image** (single pass, no TTA)

---

## Troubleshooting

**GPU out of memory:** reduce `batch_size` to 4 or `image_size` to 384

**No seg file found for image:** check filenames match between `Color_Images/` and `Segmentation/` — the script tries common patterns automatically

**Windows `num_workers` error:** set `num_workers: 0` in `CONFIG`

**Slow training:** switch encoder to `"resnet50"` for faster iteration
