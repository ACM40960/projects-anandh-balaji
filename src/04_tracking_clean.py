# =============================================================================
# 04_tracking.py — BóSight Week 4: Multi-Object Tracking
# =============================================================================
# Combines the YOLOv8s detector (week 2) and ResNet-50 re-ID model (week 3)
# into an end-to-end tracking pipeline. For every frame: detect cows →
# crop each detection → identify which cow via re-ID → output structured table.
#
# Traditional trackers (ByteTrack, BoT-SORT) were not used because MmCows
# cam_1 frames are ~17 seconds apart — too sparse for motion prediction.
# Instead, appearance-based re-ID handles identity assignment per frame.
#
# Pipeline:
#   1. Load both trained models (detector + re-ID)
#   2. For each of 5,040 frames: detect → crop → classify identity
#   3. Evaluate against ground truth labels (IoU matching)
#   4. Save results as tracked_cows.parquet
#
# Results:
#   Detection rate:    99.29% (53,129 / 53,508 ground truth boxes matched)
#   ID accuracy:       99.07% (of matched detections, correctly identified)
#   Overall accuracy:  98.37% (correct identity / total ground truth)
#
# Platform: Kaggle T4 GPU (~19 minutes for full inference)
# =============================================================================


# --- Cell 1: Install dependencies ---
# !pip install ultralytics timm -q


# --- Cell 2: Setup paths ---
# Three Kaggle datasets attached:
#   - model-weights: best.pt (detector) + reid_best.pt (re-ID)
#   - bosigh-dataset: images in train/val/test split
#   - original-labels: labels with cow_id 1-16 for evaluation
from pathlib import Path

WEIGHTS_DIR = Path("/kaggle/input/datasets/anandhvenkataraman/model-weights")
IMAGES_DIR = Path("/kaggle/input/datasets/anandhvenkataraman/bosigh-dataset/content/bosight_dataset/images")
ORIG_LABELS = Path("/kaggle/input/datasets/anandhvenkataraman/original-labels/content/drive/MyDrive/MmCows/extracted/visual_data/labels/combined/0725/cam_1")

DET_WEIGHTS = WEIGHTS_DIR / "best.pt"
REID_WEIGHTS = WEIGHTS_DIR / "reid_best.pt"

# Verify all inputs exist
print(f"Detector: {DET_WEIGHTS.exists()}")
print(f"Re-ID: {REID_WEIGHTS.exists()}")
print(f"Original labels: {len(list(ORIG_LABELS.glob('*.txt')))} files")


# --- Cell 3: Load both models ---
import torch
import timm
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8s detector
detector = YOLO(str(DET_WEIGHTS))

# Load ResNet-50 re-ID classifier (16 classes: C01-C16)
reid_model = timm.create_model("resnet50", pretrained=False, num_classes=16)
reid_model.load_state_dict(torch.load(str(REID_WEIGHTS), map_location=DEVICE))
reid_model = reid_model.to(DEVICE)
reid_model.eval()

# Same preprocessing as re-ID training (no augmentation)
reid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Cow name lookup: index 0 = C01, index 15 = C16
COW_NAMES = [f"C{i:02d}" for i in range(1, 17)]

print(f"Detector loaded: {DET_WEIGHTS.name}")
print(f"Re-ID loaded: {REID_WEIGHTS.name}")
print(f"Device: {DEVICE}")
print(f"Classes: {COW_NAMES}")


# --- Cell 4: Run detection + re-ID on all frames ---
# For every frame across all splits:
#   1. Run detector to get bounding boxes
#   2. Crop each detection from the image
#   3. Feed crop through re-ID model to get cow identity
#   4. Store result with frame stem, cow ID, confidences, and bbox coords
import time
from collections import defaultdict

all_splits = ["train", "val", "test"]
results_list = []
total_frames = 0
total_dets = 0
start = time.time()

for split in all_splits:
    img_dir = IMAGES_DIR / split
    image_paths = sorted(img_dir.glob("*.jpg"))

    for img_path in image_paths:
        stem = img_path.stem
        total_frames += 1

        # Run YOLOv8 detector (conf=0.25 threshold)
        det_results = detector(str(img_path), imgsz=640, conf=0.25, device=0, verbose=False)
        boxes = det_results[0].boxes

        if len(boxes) == 0:
            continue

        # Open full image for cropping
        img_pil = Image.open(img_path)
        w, h = img_pil.size

        # Prepare crops and metadata for batch re-ID
        crops = []
        box_coords = []
        confs = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            # Skip tiny detections
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img_pil.crop((x1, y1, x2, y2))
            crop_tensor = reid_transform(crop).unsqueeze(0).to(DEVICE)
            crops.append(crop_tensor)
            box_coords.append((x1, y1, x2, y2))
            confs.append(float(box.conf[0]))

        if not crops:
            continue

        # Batch re-ID inference: classify all crops at once
        batch = torch.cat(crops, dim=0)
        with torch.no_grad():
            logits = reid_model(batch)
            preds = logits.argmax(1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).max(1).values.cpu().numpy()

        # Store results
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            cow_name = COW_NAMES[pred]
            results_list.append({
                "stem": stem,
                "split": split,
                "cow_id": cow_name,
                "reid_conf": float(prob),
                "det_conf": confs[i],
                "x1": box_coords[i][0],
                "y1": box_coords[i][1],
                "x2": box_coords[i][2],
                "y2": box_coords[i][3],
            })
            total_dets += 1

        # Progress update every 500 frames
        if total_frames % 500 == 0:
            elapsed = time.time() - start
            print(f"Processed {total_frames} frames, {total_dets} detections, {elapsed:.0f}s")

elapsed = time.time() - start
print(f"\nDone: {total_frames} frames, {total_dets} detections, {elapsed:.0f}s")


# --- Cell 5: Evaluate against ground truth ---
# For each ground truth box, find the prediction with highest IoU overlap.
# If IoU >= 0.5, check whether the predicted cow ID matches the true ID.
# Reports detection rate, ID accuracy, and overall accuracy.
import pandas as pd

df = pd.DataFrame(results_list)
print(f"Total predictions: {len(df)}")

correct = 0
matched = 0
total_gt = 0

for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    img_dir = IMAGES_DIR / split

    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        lbl_path = ORIG_LABELS / f"{stem}.txt"

        if not lbl_path.exists():
            continue

        lines = [l.strip() for l in lbl_path.read_text().strip().split("\n") if l.strip()]
        if not lines:
            continue

        # Parse ground truth bounding boxes with cow identity
        img_pil = Image.open(img_path)
        w, h = img_pil.size

        gt_boxes = []
        for line in lines:
            parts = line.split()
            cow_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            gx1 = int((xc - bw / 2) * w)
            gy1 = int((yc - bh / 2) * h)
            gx2 = int((xc + bw / 2) * w)
            gy2 = int((yc + bh / 2) * h)
            gt_boxes.append({"cow_id": f"C{cow_id:02d}", "x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2})

        total_gt += len(gt_boxes)

        # Get predictions for this frame
        frame_preds = split_df[split_df["stem"] == stem]

        # Match each ground truth box to the best overlapping prediction
        for gt in gt_boxes:
            best_iou = 0
            best_pred_cow = None

            for _, pred in frame_preds.iterrows():
                # Compute IoU (Intersection over Union)
                ix1 = max(gt["x1"], pred["x1"])
                iy1 = max(gt["y1"], pred["y1"])
                ix2 = min(gt["x2"], pred["x2"])
                iy2 = min(gt["y2"], pred["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                area_gt = (gt["x2"] - gt["x1"]) * (gt["y2"] - gt["y1"])
                area_pred = (pred["x2"] - pred["x1"]) * (pred["y2"] - pred["y1"])
                union = area_gt + area_pred - inter

                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_pred_cow = pred["cow_id"]

            # Count as matched if IoU >= 0.5, correct if cow ID also matches
            if best_iou >= 0.5:
                matched += 1
                if best_pred_cow == gt["cow_id"]:
                    correct += 1

print(f"\nGround truth boxes: {total_gt}")
print(f"Matched (IoU >= 0.5): {matched}")
print(f"Correctly identified: {correct}")
print(f"Detection rate: {matched/total_gt:.4f}")
print(f"ID accuracy (of matched): {correct/matched:.4f}")
print(f"Overall accuracy (correct/total_gt): {correct/total_gt:.4f}")


# --- Cell 6: Save tracking output ---
# tracked_cows.parquet is the primary input for downstream stages:
# behaviour classification (week 5), sensor fusion (week 7), dashboard (week 9)
df.to_parquet("/kaggle/working/tracked_cows.parquet", index=False)
print(f"Saved {len(df)} tracking records")

from IPython.display import FileLink
display(FileLink("/kaggle/working/tracked_cows.parquet"))
