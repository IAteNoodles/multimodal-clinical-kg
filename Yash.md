# Yash.md - no_pid Ablation Model

## Task Overview
Train multimodal cascade model with no_pid ablation - disables PID (cross-modal interaction) but keeps entity types and modalities. This isolates the impact of cross-modal interaction mechanisms.

## Responsibilities
- Execute no_pid ablation training across 3 seeds
- Track performance degradation compared to full model
- Document impact of PID removal on final metrics

## Setup Requirements

### Installation
```bash
# Clone repository
git clone <repo-url>
cd MultiModal

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CUDA-optimized PyTorch (if on GPU)
#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Required Python Packages
(`requirements.txt` includes):
- `torch>=2.0.0` - Core deep learning framework
- `torchvision>=0.15.0` - Vision utilities
- `numpy>=1.24` - Numerical operations
- `pandas>=2.0.0` - Data manipulation
- `Pillow>=9.5.0` - Image processing
- `tqdm>=4.64` - Progress bars
- `psutil>=5.9.0` - System monitoring
- `transformers>=4.30.0` - Transformer models
- `wfdb>=4.1.0` - ECG data handling
- `google-cloud-bigquery>=3.11.0` - Data extraction utilities
- `google-auth>=2.20.0` - Authentication

## System Requirements

- **Python**: >=3.10
- **GPU**: CUDA-capable with >=6GB VRAM (tested on RTX 4050 6GB)
- **RAM**: >=16GB (for training)
- **Disk Space**: ~500MB for code + ~5GB for multimodal features

## Execution Commands

### Seed 42
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --ablation no_pid \
    --batch-size 512 \
    --seed 42 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed42 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --keep-last-n -1
```

### Seed 123
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --ablation no_pid \
    --batch-size 512 \
    --seed 123 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed123 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --keep-last-n -1
```

### Seed 456
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --ablation no_pid \
    --batch-size 512 \
    --seed 456 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed456 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --keep-last-n -1
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | multimodal_cascade | Cascade without PID |
| Modalities | text | Text-only input |
| Ablation | no_pid | Cross-modal interaction disabled |
| Batch Size | 512 | Adjusted for 6GB GPU |
| Checkpoint dir | ckpts/ablation/multimodal_cascade_no_pid_seed{seed} | Unique per seed |

## Expected Performance

The no_pid ablation should show:
- **Lower MRR**: Removing cross-modal interaction typically degrades performance
- **Different ratio of Hits@1, Hits@10**: Impact varies across datasets
- **Potentially more consistent**: Without PID, model may behave more predictably

### Expected Metrics (based on similar studies)
```
Test MRR: ~0.08-0.12 (vs ~0.14-0.18 for full model)
Hits@1: ~0.04-0.08 (vs ~0.07-0.12 for full model) 
Hits@10: ~0.14-0.20 (vs ~0.18-0.25 for full model)
```

## How to Read Training Logs

Look at the live log with `Get-Content -Wait ckpts\ablation\multimodal_cascade_no_pid_seed{seed}.log` and the stderr file for tqdm progress.

### Signs Training Is Fine ✅
- **Loss number goes down over epochs**: e.g., ep1 loss=2.1, ep5 loss=1.5, ep10 loss=0.8
- **MRR goes up**: starts near 0, increases to 0.10+ over time
- **tqdm progress bar moves steadily**: ~6-7 it/s for cascade, batch 512
- **Loss changes between batches** (not identical every time)

### Signs Something Is Wrong ❌
- **Loss stuck at 2.197** (or same number for many epochs) → model learned nothing, guessing randomly
- **MRR stays at 0.000** across all evals → all scores identical, random ranking
- **Loss drops but MRR stays 0** → model memorized training data, can't predict new triples
- **Loss is exactly the same every batch** → embeddings not updating (gradients zero or collapsed)
- **"ios_base::badbit" or "unexpected pos" errors** → disk full from too many checkpoint files
- **Training gets slower over time** → GPU memory leak, restart needed
- **tqdm stuck at same batch for >30s** → process hung, kill and resume
- **Log file not updating** → Python buffering, use `python -u` flag

### What to Do When Something Is Wrong
1. Stop the process: `Get-Process -Name python | Stop-Process -Force`
2. Check how much disk is free: `Get-PSDrive C`
3. Delete old checkpoint files if disk is full
4. Restart with `--resume` to continue from last good checkpoint
5. If problem persists, ask Noodles

### Checkpoint Files on Disk
- `latest.pt` (2.6 GB) — saved every epoch, needed for resume
- `best_model.pt` (0.86 GB) — saved only when MRR improves, just model weights
- NO `ep_*.pt` files — `--keep-last-n -1` prevents them from being created
- If `ep_*.pt` files appear, disk will fill up fast — delete them

## Execution Timeline

**Start all no_pid seeds immediately after full model seeds begin:**
```bash
# Start all 3 no_pid seeds (recommended to start after full seeds)
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "simulation/kg/train_manual.py --model multimodal_cascade --modalities text --ablation no_pid --batch-size 512 --seed 42 --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed42"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "simulation/kg/train_manual.py --model multimodal_cascade --modalities text --ablation no_pid --batch-size 512 --seed 123 --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed123"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "simulation/kg/train_manual.py --model multimodal_cascade --modalities text --ablation no_pid --batch-size 512 --seed 456 --checkpoint-dir ckpts/ablation/multimodal_cascade_no_pid_seed456"
```

## Monitoring & Analysis

### Compare with Full Model
```bash
# Plot performance comparison
python -c "
import json
full = json.load(open('ckpts/ablation/multimodal_cascade_full_seed42/test_results.json'))
nopid = json.load(open('ckpts/ablation/multimodal_cascade_no_pid_seed42/test_results.json'))
print('Full Model MRR:', full['MRR'])
print('no_pid MRR:', nopid['MRR'])
print('Difference:', full['MRR'] - nopid['MRR'])
"
```

### Real-time Progress
```bash
# Monitor any seed's log (replace with desired seed)
Get-Content -Wait ckpts\ablation\multimodal_cascade_no_pid_seed42.log
```

## Expected Outputs

Same directory structure as full model, but with lower performance metrics:

```
ckpts/ablation/multimodal_cascade_no_pid_seed42/
├── meta.json              # Training metadata
├── test_results.json      # Lower MRR compared to full model
├── best.pt, latest.pt     # Checkpoints
└── *.log                  # Training logs
```

## Analysis Tasks

### Document Findings
1. **Quantitative Comparison**: Calculate percentage drop in MRR for no_pid vs full
2. **Component Impact**: Analyze which aspects contribute most to performance difference
3. **Practical Implications**: Determine if performance cost is worth removing PID for production use
4. **Seed Variation**: Compare impact across 3 random seeds

### Report Format (suggested)
```markdown
## no_pid Ablation Results

### Overall Performance
| Metric | Full Model | no_pid | Drop (%) |
|--------|------------|--------|----------|
| MRR | 0.16 | 0.11 | 31% |
| Hits@1 | 0.10 | 0.06 | 40% |
| Hits@10 | 0.22 | 0.18 | 18% |

### Per Seed Comparison
- Seed 42: MRR 0.11 (-31% from full)
- Seed 123: MRR 0.12 (-28% from full)
- Seed 456: MRR 0.13 (-29% from full)

### Key Insights
- Cross-modal interaction contributes significantly to performance
- Impact is consistent across seeds (28-32% MRR reduction)
- Trade-off: simpler model vs. ~30% performance loss
```

## 🚀 Next Steps for All Members

### Collaboration Points
1. **Daily Progress Reports**: Share completion status and any issues
2. **Shared Analysis**: Compare results across all ablation types
3. **Problem Solving**: Coordinate on OOM recovery strategies
4. **Documentation**: Maintain individual progress tracking files

### Success Criteria
1. **All seeds complete** 50 epochs successfully
2. **meta.json** created for each seed
3. **test_results.json** generated for final evaluation
4. **Complete analysis** of all three ablation types
5. **Shareable deliverables** with clear findings and recommendations
