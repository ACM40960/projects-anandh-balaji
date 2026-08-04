# Reproducing the GPU-trained stages on Kaggle

The behaviour classifier (`src/05_behaviour_classifier.ipynb`) and the BCS notebook
(`src/06_bcs.ipynb`) are designed to run on a free **Kaggle T4 GPU**. This guide reproduces
the behaviour classifier end-to-end; the BCS notebook follows the same attach-and-run pattern.

## Prerequisites

- A free [Kaggle](https://www.kaggle.com) account.
- The two inputs the notebook consumes, uploaded as Kaggle Datasets:
  1. **Cow crops** — per-cow image crops, organised as `.../bosight_crops/{train,val,test}/<cow>/*.jpg`
     (produced by the detection + tracking stages, `src/02`–`src/04`).
  2. **Behaviour labels** — the 16 per-cow CSVs `C01_0725.csv` … `C16_0725.csv` from MmCows.

## Steps

1. **Upload the datasets** — Kaggle → *Datasets → New Dataset*. Drag in the crops (a zip is
   auto-extracted) and, as a second dataset, the behaviour-label CSVs. Titles don't matter;
   the notebook auto-detects the folders under `/kaggle/input`.
2. **Import the notebook** — Kaggle → *Code → New Notebook → File → Import Notebook*, and
   upload `src/05_behaviour_classifier.ipynb`.
3. **Attach inputs + GPU** — in the right sidebar, *Add Input* for both datasets, then
   *Settings → Accelerator → GPU T4 x1*.
4. **Enable internet** — *Settings → Internet → On* (required so torchvision can download the
   ImageNet-pretrained ResNet-50 weights).
5. **Run All.** Cell 1 should print `Device: cuda`. A full 15-epoch run takes ~15–25 minutes.

## Outputs

Artefacts are written to `/kaggle/working/`:

- `behaviour_best.pt` — the trained model (also stored in the repo under `models/`).
- `behaviour_meta.json` — per-class metrics and the test classification report.

Use **Save Version** (or download from the *Output* tab) to persist them — Kaggle wipes the
working directory when a session ends.

## Notes

- Keep `NUM_WORKERS = 4`; `0` or `2` can stall the dataloader on Kaggle.
- The `moving` class is the hardest (rare + temporal); a low F1 there is expected and is
  discussed in `evaluation/consolidated_metrics.md`.
- The notebook resolves all paths automatically, so no `/kaggle/input/...` hardcoding is needed.
