# Cascade Ablation Results

## Experiment Setup

- **Base model**: ComplEx scoring function
- **Embedding dim**: 256
- **Batch size**: 3000
- **Learning rate**: 3e-4
- **Epochs**: 50
- **grad_accum_steps**: 1
- **Dataset**: 3.77M triples, multimodal KG
- **GPU**: RTX 4050 6GB VRAM
- **Text-only mode** (use_modality_encoders=False)

The full "cascade" model adds three auxiliary embedding components on top of bare ComplEx:

1. **PID synergy**: Hypernetwork + modulation MLP that generates relation-specific adjustments to entity embeddings (20 relations)
2. **Entity type embeddings**: Added to entity embeddings based on entity type (12 types, but 78% of entities are "Finding" type)
3. **Modality embeddings**: Added to entity embeddings based on data modality (5 modalities: CXR, ECG, text, structured, lab)

Each ablation removes one component while keeping the others.

## Results

| Config | Description | Test MRR | Best Val MRR | Best Val Epoch | Δ vs Full Cascade |
|---|---|---|---|---|---|
| Full Cascade | All components (PID + type + modality) | 0.8209 | — | — | baseline |
| no_pid | Strips PID synergy | 0.8251 | — | — | +0.0042 |
| no_type | Strips entity type embeddings | 0.8289 | 0.8284 | 48 | +0.0080 |
| no_modality | Strips modality embeddings | 0.8308 | 0.8305 | 49 | +0.0099 |
| all (bare ComplEx) | Strips all three | — | 0.1588 | — | −0.6621 (val) |

Earlier baseline results for context:

- TransE (50ep): test MRR = 0.5752
- ComplEx (50ep, standalone): test MRR = 0.7690

## Key Observations

1. **Removing ANY single auxiliary component improves performance.** This is counterintuitive — each component was designed to help, yet removing any one of them yields better test MRR than the full cascade.

2. **Ranking**: no_modality (0.8308) > no_type (0.8289) > no_pid (0.8251) > full cascade (0.8209). The best config strips modality embeddings.

3. **Bare ComplEx catastrophically fails** (val MRR 0.1588 when all three are removed), yet the full cascade with all three underperforms any single-ablation. This means the auxiliary components are necessary in aggregate but harmful individually when combined.

4. **The magnitude of improvement is small**: best vs worst (excluding bare) is +0.0099 MRR. Without multi-seed runs, it's unclear if this is within noise.

5. **Overparameterization pattern**: All three auxiliary components have very few unique values (5 modalities, 12 types, 20 relations for PID) relative to the entity vocabulary. The embeddings for these may add noise without sufficient signal.

## Open Questions (Under Investigation)

- Is the +0.0099 difference within noise? (Requires multi-seed runs to answer)
- Is there a bug in how auxiliary embeddings are combined with entity embeddings?
- Is there a scaling/initialization issue that causes auxiliary embeddings to dominate?
- Is this a known phenomenon in KGE literature (auxiliary features causing negative transfer)?

## File Paths (Checkpoint Directories)

- `ckpts/cascade_v2/` — Full cascade (test MRR=0.8209)
- `ckpts/cascade_ablation_no_pid/` — no_pid ablation (test MRR=0.8251)
- `ckpts/cascade_ablation_no_type/` — no_type ablation (test MRR=0.8289)
- `ckpts/cascade_ablation_no_modality/` — no_modality ablation (test MRR=0.8308)