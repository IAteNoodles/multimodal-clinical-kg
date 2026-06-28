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
