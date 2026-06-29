# Instructions

## Setup

```bash
git clone <repo-url>
cd MultiModal
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Requires Python >= 3.10, CUDA GPU with >= 6 GB VRAM.

## Data

All preprocessed data is in the repo:
- KG triples: `simulation/data/kg/clinical_kg_efficient/`
- Multimodal features: `simulation/data/kg/multimodal/features/` (CXR, ECG, text, structured)

No external downloads needed.

## Training

Three models available: `transE`, `complex`, `multimodal_cascade`.

### TransE (107M params, fastest)

```bash
python simulation/kg/train_manual.py --model transE --batch-size 38000 --num-negatives 4 --grad-accum-steps 2 --epochs 50     --eval --patience 0 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000 --max-test-triples 5000 --eval-chunk-size 512 --keep-last-n -1 --checkpoint-dir ckpts/transe --seed 42
```

### ComplEx (216M params)

```bash
python simulation/kg/train_manual.py --model complex --batch-size 1024 --num-negatives 4 --grad-accum-steps 2 --epochs 50     --eval --patience 0 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000 --max-test-triples 5000 --eval-chunk-size 512 --keep-last-n -1 --checkpoint-dir ckpts/complex --seed 42
```

### Multimodal Cascade (216M params)

```bash
python simulation/kg/train_manual.py --model multimodal_cascade --modalities text --batch-size 512 --num-negatives 4 --grad-accum-steps 2 --epochs 50     --eval --patience 0 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000 --max-test-triples 5000 --eval-chunk-size 512 --keep-last-n -1 --checkpoint-dir ckpts/cascade --seed 42
```

Cascade also supports `--ablation` (no_pid, no_type, no_modality) for ablation studies.

### Key Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--batch-size` | 256 | Per-model: transE=38000, complex=1024, cascade=512 |
| `--grad-accum-steps` | 2 | Effective batch = batch-size × grad-accum-steps |
| `--eval-chunk-size` | 512 | Lower if OOM during eval (uses less GPU memory) |
| `--max-eval-triples` | None | Cap val triples for faster eval (recommend 5000) |
| `--max-test-triples` | 5000 | Cap test triples for faster final eval |
| `--keep-last-n` | 3 | Set to -1 to skip saving epoch checkpoints (saves disk) |
| `--resume` | off | Resume from latest checkpoint in --checkpoint-dir |

## Ablation Runner

Runs all 18 combos (3 models × 3 seeds, cascade also × 4 ablations):

```bash
python run_ablation.py
```

Skips completed runs (checks for `meta.json`). Resumes partial runs automatically.

## Resume

If training crashes (OOM, power loss), re-run with `--resume`:

```bash
python simulation/kg/train_manual.py ...same args... --resume
```

## Output

- `ckpts/ablation/{name}/meta.json` — training metadata (epoch, MRR)
- `ckpts/ablation/{name}/test_results.json` — final test metrics
- `ckpts/ablation/{name}.log` — per-run stdout
- `ckpts/ablation/ablation_master.log` — overall progress

## Notes

- 6 GB VRAM limit: batch sizes above will OOM on backward pass
- `--eval-chunk-size 512` is safe for all models; higher (1024) may OOM on complex/cascade
- After training completes, delete `*.pt` checkpoints to save space (only `meta.json` + `test_results.json` needed for records)
