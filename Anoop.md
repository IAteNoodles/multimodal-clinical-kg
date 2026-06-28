# Anoop.md - Full Multimodal Cascade Model

## Task Overview
Train the complete multimodal cascade model with text modality only. This is the baseline model that includes all functionalities: PID (cross-modal interaction), entity types, and modalities.

## Responsibilities
- Execute full cascade model training across 3 different random seeds
- Monitor training progress and ensure completion of all 50 epochs for each seed
- Track and report final test metrics for each seed

## Execution Commands

### Seed 42
```bash
python simulation/kg/train_manual.py \
    --model multimodal_cascade \
    --modalities text \
    --batch-size 512 \
    --seed 42 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_full_seed42 \
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
    --batch-size 512 \
    --seed 123 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_full_seed123 \
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
    --batch-size 512 \
    --seed 456 \
    --checkpoint-dir ckpts/ablation/multimodal_cascade_full_seed456 \
    --eval-every-epochs 3 \
    --max-eval-triples 5000 \
    --eval-chunk-size 512 \
    --keep-last-n -1
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | multimodal_cascade | Full cascade with all components |
| Modalities | text | Text-only input modality |
| Batch Size | 512 | Adjusted for 6GB GPU memory |
| Epochs | 50 | Full training duration |
| Eval every epochs | 3 | Validation every 3 epochs |
| Checkpoint dir | ckpts/ablation/multimodal_cascade_full_seed{seed} | Unique per seed |
| Keep last n | -1 | Save all epoch checkpoints |

## Expected Duration

**Per seed**: ~40 minutes per epoch × 50 epochs = ~35 hours  
**Total (3 seeds)**: ~105 hours (parallel execution recommended)

## Monitoring & Tracking

### Check Training Progress
```bash
# Watch live log output (replace SEED with 42, 123, or 456)
Get-Content -Wait ckpts\ablation\multimodal_cascade_full_seed{seed}.log

# Check master progress log
Get-Content ckpts\ablation\ablation_master.log -Tail 10
```

### Verify Completion
```bash
# Check if meta.json exists (indicates completed training)
Test-Path ckpts\ablation\multimodal_cascade_full_seed{seed}\meta.json

# View final metrics
if (Test-Path ckpts\ablation\multimodal_cascade_full_seed{seed}\test_results.json) {
    Get-Content ckpts\ablation\multimodal_cascade_full_seed{seed}\test_results.json | ConvertFrom-Json
}
```

### System Status
```bash
# See all active training processes
Get-Process -Name python | Select-Object Id, @{N='Cmd';E={(Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine}} | Format-Table -AutoSize -Wrap
```

## Expected Outputs

After each seed completes, you should have:

```
ckpts/ablation/multimodal_cascade_full_seed{seed}/
├── meta.json              # Training metadata (epoch, MRR, best_ep)
├── test_results.json      # Final test evaluation metrics
├── best.pt                # Best model checkpoint
├── best_model.pt          # Model weights only
├── latest.pt              # Latest checkpoint
└── *.log                  # Training logs
```

## Error Handling

### Out of Memory (OOM)
1. Kill the process:
   ```powershell
   Get-Process -Name python | Stop-Process -Force
   ```
2. Remove incomplete checkpoint directory:
   ```powershell
   Remove-Item -Recurse -Force ckpts\ablation\multimodal_cascade_full_seed{seed}
   ```
3. Resume training:
   ```bash
   python simulation/kg/train_manual.py ... --checkpoint-dir ckpts/ablation/multimodal_cascade_full_seed{seed} --resume
   ```

### GPU Timeout
1. If training hangs for >2 seconds, kill and restart with `--resume`
2. Check GPU memory before starting:
   ```bash
   python -c "import torch; print(f'alloc={torch.cuda.memory_allocated()/1e9:.2f}GB')"
   ```

## Timeline (Parallel Execution)

| Time | Action |
|------|--------|
| 00:00 | Start all 3 cascade full seeds in parallel |
| 00:00-35h | Monitor progress and check logs |
| 35h+ | All seeds should be completed |

---

# Yash.md - no_pid Ablation Model

## Task Overview
Train multimodal cascade model with no_pid ablation - disables PID (cross-modal interaction) but keeps entity types and modalities. This isolates the impact of cross-modal interaction mechanisms.

## Responsibilities
- Execute no_pid ablation training across 3 seeds
- Track performance degradation compared to full model
- Document impact of PID removal on final metrics

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

---

# Albin.md - no_type Ablation Model

## Task Overview
Train multimodal cascade model with no_type ablation - disables entity type embeddings but keeps PID and modalities. This isolates the impact of entity typing on performance.

## Responsibilities
- Execute no_type ablation training across 3 seeds
- Analyze impact of entity type embeddings on cross-modal performance
- Compare findings with full and no_pid results

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

## Expected Outputs

All no_type seeds should produce similar directory structure:

```
ckpts/ablation/multimodal_cascade_no_type_seed42/
├── meta.json              # Training metadata
├── test_results.json      # Metrics showing impact of removing entity types
├── best.pt, latest.pt     # Checkpoints
└── *.log                  # Training logs
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

### Immediate Actions
1. **Set up dedicated monitoring** for your assigned ablation
2. **Configure parallel execution** if running multiple seeds simultaneously
3. **Create log monitoring scripts** for real-time progress tracking
4. **Establish backup procedures** for checkpoint management

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
