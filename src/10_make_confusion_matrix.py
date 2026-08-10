"""
BóSight Week 10: Behaviour Confusion Matrix (evaluation figure)

Re-runs the trained behaviour classifier over the held-out test crops and writes the
confusion matrix (raw counts + row-normalised) and per-class report to evaluation/.
The metrics themselves are recorded during training; this script exists to regenerate
the figures for the report.

Pipeline:
  1. Build the test index by joining each crop to its behaviour label (by timestamp)
  2. Run inference with behaviour_best.pt over the test crops
  3. Save confusion_matrix.png, confusion_matrix_normalised.png, confusion_matrix.csv

Notes:
  - CPU-safe: OpenMP guards are set BEFORE importing numpy/torch (full training crashed
    this machine's CPU with a duplicate-OpenMP access violation; inference is lighter and
    these guards keep it stable).
  - Model path is repo-relative (models/); output goes to evaluation/. Crop and label
    inputs are external (not in the repo) — update CROPS and BEH to your local copies.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # tolerate the duplicate OpenMP runtime
os.environ['OMP_NUM_THREADS'] = '2'           # keep CPU thread count modest
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
torch.set_num_threads(2)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True         # some crops are slightly truncated
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')                          # non-interactive backend (saves files, no display)
import matplotlib.pyplot as plt
import seaborn as sns

REPO  = Path(__file__).resolve().parent.parent      # repo root (this file is in src/)
CKPT  = str(REPO / 'models' / 'behaviour_best.pt')  # trained model, in the repo
OUT   = REPO / 'evaluation'                          # where the figures are written
CROPS = Path('D:/D2/bosight/data/content/bosight_crops/test')          # external; update to your path
BEH   = Path('D:/D2/sensor_data/sensor_data/behavior_labels/individual')  # external; update to your path
CLASSES = ['lying', 'standing', 'feeding', 'moving']
C2I = {c: i for i, c in enumerate(CLASSES)}
CODE2CLASS = {1: 'moving', 2: 'standing', 3: 'feeding', 4: 'feeding', 7: 'lying'}  # 0/5/6 excluded

# Build a {timestamp: behaviour_code} lookup per cow so each crop can be labelled cheaply.
beh = {}
for c in range(1, 17):
    cow = f'C{c:02d}'
    d = pd.read_csv(BEH / f'{cow}_0725.csv')
    beh[cow] = dict(zip(d['timestamp'].astype(int), d['behavior'].astype(int)))

# Index the test crops, attaching each one's 4-class label via its filename timestamp.
rows = []
for cow in sorted(os.listdir(CROPS)):
    cd = CROPS / cow
    if not cd.is_dir(): continue
    for fn in os.listdir(cd):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        ts = int(fn.split('_')[0]); code = beh[cow].get(ts)
        if code is None: continue          # no label for this second
        cls = CODE2CLASS.get(code)
        if cls is None: continue           # excluded behaviour (unknown/licking/drinking)
        rows.append({'path': str(cd / fn), 'label': C2I[cls]})
test = pd.DataFrame(rows)
print('test crops:', len(test), flush=True)

# Evaluation transform: resize + ImageNet normalisation (no augmentation).
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DS(Dataset):
    def __init__(s, df): s.df = df.reset_index(drop=True)
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]; return tf(Image.open(r['path']).convert('RGB')), r['label']

loader = DataLoader(DS(test), batch_size=64, shuffle=False, num_workers=0)

# Rebuild the ResNet-50 head to 4 classes and load the trained weights.
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(CKPT, map_location='cpu')); model.eval()
print('model loaded', flush=True)

# Inference over the test set.
preds, gts, done = [], [], 0
with torch.no_grad():
    for x, y in loader:
        preds += model(x).argmax(1).tolist(); gts += y.tolist()
        done += len(y)
        if done % 1280 == 0: print(f'  {done}/{len(test)}', flush=True)

# Confusion matrix + per-class report.
cm = confusion_matrix(gts, preds)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT / 'confusion_matrix.csv')
print(classification_report(gts, preds, target_names=CLASSES, digits=3, zero_division=0), flush=True)

# Raw-count heatmap.
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Behaviour — Test Confusion Matrix')
plt.tight_layout(); plt.savefig(OUT / 'confusion_matrix.png', dpi=150); plt.close()

# Row-normalised heatmap (recall per true class).
cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
plt.figure(figsize=(6, 5))
sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Behaviour — Normalised (recall per row)')
plt.tight_layout(); plt.savefig(OUT / 'confusion_matrix_normalised.png', dpi=150); plt.close()
print('SAVED confusion_matrix.png + _normalised.png + .csv to', OUT, flush=True)
