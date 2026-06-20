# Training

## Setup

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10 and a CUDA-capable GPU (tested on RTX 4050 6GB).

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

## Ablations

Replace `--ablation` flag with one of:
- `no_pid` — disables PID (cross-modal interaction)
- `no_type` — disables entity type embeddings
- `no_modality` — disables modality embeddings
- `all` — disables all auxiliary components

## Resume from Crash

Add `--resume` to resume from the latest checkpoint:

```bash
python -u simulation/kg/train_manual.py \
    ...same args... \
    --checkpoint-dir ckpts/cascade_v3 \
    --resume > log.txt 2>&1
```

Results (test metrics, best model weights, training log) are saved under `results/` with seed info.
