# Training

## Setup

```bash
# 1. Clone repo
git clone <repo-url>
cd MultiModal

# 2. Install Python deps
pip install -r requirements.txt

# 3. Install CUDA PyTorch (see pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Requires Python >= 3.10 and a CUDA-capable GPU with >= 6GB VRAM (tested on RTX 4050 6GB).

## Data

The preprocessed KG data (`simulation/data/kg/clinical_kg_efficient/`) is included in the repo (~100 MB). No raw data downloads needed for training.

For the full multimodal features (CXR, ECG, text, structured — ~5.4 GB), download from [link TBD]. These are used for cross-modal attention when `--feature-dir` is provided to training.

## Quick Start

Train the full multimodal cascade model:

```bash
python -u simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --embed-dim 256 \
    --batch-size 3000 \
    --grad-accum-steps 1 \
    --epochs 1000 \
    --lr 3e-4 \
    --temperature 0.2 \
    --label-smoothing 0.05 \
    --n3-weight 0.01 \
    --checkpoint-dir ckpts/cascade_v3 \
    --keep-last-n 0 \
    --eval \
    --eval-every-epochs 1 \
    --patience 10 \
    --eval-batch-size 1024 > log.txt 2>&1
```

The `--feature-dir` flag is optional. When omitted, cross-modal context is disabled (the model gracefully degrades with `precompute_features = False`, zeroing out cross-modal attention context). When provided (path to a directory of precomputed entity features), it loads the features and enables cross-modal attention.

Multimodal run with precomputed features:

```bash
python -u simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text,image,ecg \
    --feature-dir data/features \
    --embed-dim 256 \
    --batch-size 3000 \
    --grad-accum-steps 1 \
    --epochs 1000 \
    --lr 3e-4 \
    --temperature 0.2 \
    --label-smoothing 0.05 \
    --n3-weight 0.01 \
    --checkpoint-dir ckpts/cascade_v3 \
    --keep-last-n 0 \
    --eval \
    --eval-every-epochs 1 \
    --patience 10 \
    --eval-batch-size 1024 > log.txt 2>&1
```

## Ablations

Replace `--ablation` flag with one of:
- `no_pid` — disables PID (cross-modal interaction)
- `no_type` — disables entity type embeddings
- `no_modality` — disables modality embeddings
- `all` — disables all auxiliary components

## Resume from Crash

On Windows WDDM, CUDA kernel timeouts (>2s) may crash. Add `--resume` to continue:

```bash
python -u simulation/kg/train_manual.py \
    ...same args... \
    --checkpoint-dir ckpts/cascade_v3 \
    --resume > log.txt 2>&1
```

## Results

Test metrics, best model weights, and training logs are saved under `ckpts/ablation/{name}/`.

## Known Issues

- **Windows WDDM**: Training may crash every 1-20 epochs due to GPU driver timeout. Kill the hung process and resume with `--resume`.
- **Expandable Segments**: `expandable_segments:True` not supported on Windows. Default allocator may cause OOM on fragmentation.
- **First Epoch Slow**: CUDA kernel compilation for the 216M-param model can take several minutes on the first epoch.
