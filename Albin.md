# Albin.md - no_type Ablation Model

## Task Overview
Train multimodal cascade model with no_type ablation - disables entity type embeddings but keeps PID and modalities. This isolates the impact of entity typing on performance.

## Responsibilities
- Execute no_type ablation training across 3 seeds
- Analyze impact of entity type embeddings on cross-modal performance
- Compare findings with full and no_pid results

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
    --ablation no_type \
    --batch-size 512 \
    --seed 42 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_type_seed42 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --patience 0 \
    --keep-last-n -1
```

### Seed 123
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --ablation no_type \
    --batch-size 512 \
    --seed 123 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_type_seed123 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --patience 0 \
    --keep-last-n -1
```

### Seed 456
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --ablation no_type \
    --batch-size 512 \
    --seed 456 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_no_type_seed456 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --patience 0 \
    --keep-last-n -1
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | multimodal_cascade | Cascade without entity types |
| Modalities | text | Text-only input |
| Ablation | no_type | Entity type embeddings disabled |
| Batch Size | 512 | Adjusted for 6GB GPU |
| Checkpoint dir | ckpts/ablation/multimodal_cascade_no_type_seed{seed} | Unique per seed |

## Expected Performance

Entity type embeddings typically help with:
- **Fine-grained entity relationships**: Distinguishing between different entity categories
- **Cross-modal consistency**: Maintaining type coherence across modalities
- **Interpretability**: Making model decisions more explainable

### Expected Performance Characteristics
The no_type ablation should show:
- **Moderate performance drop**: Entity types are important but not the primary driver
- **Different failure patterns**: May behave differently than no_pid ablation
- **Potential specialization**: Model may overfit to text patterns without type guidance

## Three-Way Analysis

### Comparative Analysis
```python
# Load all three ablation results
full = json.load(open('ckpts/ablation/multimodal_cascade_full_seed42/test_results.json'))
nopid = json.load(open('ckpts/ablation/multimodal_cascade_no_pid_seed42/test_results.json'))
notype = json.load(open('ckpts/ablation/multimodal_cascade_no_type_seed42/test_results.json'))

# Compare impacts
print("Performance Impact Analysis:")
print(f"Full Model MRR: {full['MRR']:.3f}")
print(f"no_pid MRR: {nopid['MRR']:.3f} (Diff: {full['MRR'] - nopid['MRR']:.3f})")
print(f"no_type MRR: {notype['MRR']:.3f} (Diff: {full['MRR'] - notype['MRR']:.3f})")
print()
print("Component Importance Ranking:")
print(f"1. {'PID' if (full['MRR'] - nopid['MRR']) > (full['MRR'] - notype['MRR']) else 'Type embeddings'}: {max(full['MRR'] - nopid['MRR'], full['MRR'] - notype['MRR']):.3f} MRR drop")
print(f"2. {'Type embeddings' if (full['MRR'] - notype['MRR']) > (full['MRR'] - nopid['MRR']) else 'PID'}: {min(full['MRR'] - nopid['MRR'], full['MRR'] - notype['MRR']):.3f} MRR drop")
```

## How to Read Training Logs

Look at the live log with `Get-Content -Wait ckpts\ablation\multimodal_cascade_no_type_seed{seed}.log` and the stderr file for tqdm progress.

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

## Execution Strategy

### Priority Order (if time is limited)

1. **Full Model** (Anoop) - Baseline for comparison
2. **Complex Model** (Noodles) - GPU resource intensive, may need to complete first
3. **no_type Model** (Albin) - Analyzes importance of entity types
4. **no_pid Model** (Yash) - Analyzes importance of cross-modal interaction

### Monitoring Plan

### Real-time Tracking
```bash
# Monitor all ablation runs in parallel
Get-Content -Wait ckpts\ablation\multimodal_cascade_no_type_seed42.log
Get-Content -Wait ckpts\ablation\multimodal_cascade_no_type_seed123.log
Get-Content -Wait ckpts\ablation\multimodal_cascade_no_type_seed456.log
```

### Progress Indicators
```bash
# Check all directory structures in parallel
Test-Path ckpts\ablation\multimodal_cascade_no_type_seed42\meta.json
Test-Path ckpts\ablation\multimodal_cascade_no_type_seed123\meta.json
Test-Path ckpts\ablation\multimodal_cascade_no_type_seed456\meta.json
```

## Analysis Metrics

### Three-Way Comparison
For seed 42 (similar results expected for 123 and 456):
```python
# Impact analysis for each seed
def analyze_impact(full_path, nopid_path, notype_path):
    full = json.load(open(full_path))
    nopid = json.load(open(nopid_path))
    notype = json.load(open(notype_path))
    
    results = {
        'seed': path.split('_seed')[-1].replace('.json', ''),
        'full_mrr': full['MRR'],
        'nopid_mrr': nopid['MRR'],
        'notype_mrr': notype['MRR'],
        'nopid_drop': full['MRR'] - nopid['MRR'],
        'notype_drop': full['MRR'] - notype['MRR'],
        'relative_importance': (
            'PID' if (full['MRR'] - nopid['MRR']) > (full['MRR'] - notype['MRR']) 
            else 'Entity Types'
        )
    }
    return results
```

### Deliverables

### 1. Analysis Report
- Comparative performance tables
- Statistical significance tests
- Recommendations for model simplification

### 2. Visualization
- Bar charts comparing MRR across all three ablations
- Training curve comparisons
- Heat maps of hyperparameter impacts

### 3. Code
- Scripts for automated comparison of all ablation variants
- Reproducible analysis notebooks

### 4. Documentation
- Detailed methodology section
- Hyperparameter configuration for all variants
- Future research directions based on findings

## 🚀 Next Steps for All Members

### Collaboration Points
1. **Daily Progress Reports**: Share completion status and any issues
2. **Shared Analysis**: Compare results across all ablation types
3. **Problem Solving**: Coordinate on OOM recovery strategies
4. **Documentation**: Maintain individual progress tracking files

### Success Criteria
1. **All seeds complete** 51 epochs successfully
2. **meta.json** created for each seed
3. **test_results.json** generated for final evaluation
4. **Complete analysis** of all three ablation types
5. **Shareable deliverables** with clear findings and recommendations
