# Cascade Ablation Results

## Experiment Setup

- **Base model**: ComplEx scoring function
- **Embedding dim**: 256
- **Batch size**: 3000
- **Learning rate**: 3e-4
- **Epochs**: 50
- **grad_accum_steps**: 1
- **Dataset**: 3.77M triples, multimodal KG (421,216 entities, 20 relations)
- **GPU**: RTX 4050 6GB VRAM
- **Text-only mode** (use_modality_encoders=False)
- **Eval protocol**: 50-negative sampled ranking (NOT full filtered ranking)
- **Loss**: InfoNCE (temperature=0.2, label_smoothing=0.05, 4 training negatives)
- **Optimizer**: AdamW (weight_decay=1e-3), CosineAnnealingLR, 1000-step warmup
- **clamp_norm**: 0.0 (DISABLED — no embedding norm constraint)

The full "cascade" model adds three auxiliary embedding components on top of bare ComplEx:

1. **PID synergy**: Hypernetwork + modulation MLP that generates relation-specific adjustments to entity embeddings (20 relations). Only applied to cross-modal triples.
2. **Entity type embeddings**: Added to entity embeddings based on entity type (12 types, but 78% of entities are "Finding" type)
3. **Modality embeddings**: Gated addition to entity embeddings based on data modality (5 modalities: CXR, ECG, RAD, STR, None)

Each ablation removes one component while keeping the others.

## Results

| Config | Description | Test MRR | Best Val MRR | Best Val Epoch | Δ vs Full Cascade |
|---|---|---|---|---|---|
| Full Cascade | All components (PID + type + modality) | 0.8209 | — | — | baseline |
| no_pid | Strips PID synergy | 0.8251 | — | — | +0.0042 |
| no_type | Strips entity type embeddings | 0.8289 | 0.8284 | 48 | +0.0080 |
| no_modality | Strips modality embeddings | 0.8308 | 0.8305 | 49 | +0.0099 |
| all (bare ComplEx) | Strips all three | **UNVERIFIED** | 0.1588 (?) | — | — |

Earlier baseline results for context:

- TransE (50ep): test MRR = 0.5752
- ComplEx (50ep, standalone `ComplExModel`): test MRR = 0.7690

> **NOTE on "all" ablation**: The 0.1588 value is **unverified**. No checkpoint directory (`ckpts/cascade_ablation_all/`), no log file, and no `meta.json` exist for this run. The `run_ablation.bat` script only handles `no_pid`, `no_type`, `no_modality` — it does NOT include `all`. The value likely came from an ad-hoc run whose artifacts were lost. This value should not be trusted until re-run with proper logging.

## Key Observations

1. **Removing ANY single auxiliary component improves performance.** This is counterintuitive — each component was designed to help, yet removing any one of them yields better test MRR than the full cascade.

2. **Ranking**: no_modality (0.8308) > no_type (0.8289) > no_pid (0.8251) > full cascade (0.8209). The best config strips modality embeddings.

3. **The magnitude of improvement is small**: best vs worst is +0.0099 MRR. Without multi-seed runs, it's unclear if this is within noise. However, the DIRECTION is consistent across all three ablations (all remove → all improve), which is unlikely to be pure noise.

4. **Overparameterization pattern**: All three auxiliary components have very few unique values (5 modalities, 12 types, 20 relations for PID) relative to the entity vocabulary (421K). The embeddings for these may add noise without sufficient signal.

## Root Cause: 40x Initialization Scaling Mismatch

### The Bug

All embedding layers use `nn.init.xavier_uniform_`, which computes its bound as:

```
bound = sqrt(6 / (fan_in + fan_out))
```

where `fan_in = embedding_dim` and `fan_out = num_embeddings` (the number of rows).

| Layer | num_embeddings | embed_dim | xavier bound | row L2 norm (mean) |
|---|---|---|---|---|
| `entity_embeddings` | 421,216 | 512 | 0.00377 | 0.0493 (full), **0.0348** (re/im half) |
| `modality_embeddings` | 5 | 256 | 0.15162 | **1.3931** |
| `entity_type_embeddings` | 12 | 256 | 0.14963 | **1.3911** |

**Scaling ratio**: auxiliary / base = 1.39 / 0.035 = **~40x**

### Why This Breaks the Model

In `_get_entity_emb()` (models.py:804-824), the combination is additive:

```python
re = base_re                          # norm ~0.035
re = re + mod_emb * has_mod           # mod_emb norm ~1.39, has_mod=0.5 at init
re = re + type_emb                    # type_emb norm ~1.39
```

After combination, `re` is dominated by the auxiliary terms (~1.39) with the base entity embedding (~0.035) contributing <3% of the signal magnitude. The model cannot distinguish between entities that share the same type and modality — the entity-specific signal is drowned out.

Additionally:
- `clamp_embed_norm` (which would normalize base to 1.0) is **disabled** (`--clamp-norm` default=0.0, not set in run scripts)
- `has_modality_logit` starts at 0 → sigmoid(0) = 0.5, so modality is added at 50% strength from the start
- Type embedding is added **ungated** (no learnable gate, always full strength)
- Only `re` is modified — `im` is never augmented (asymmetric, breaks ComplEx symmetry)
- `n3_penalty` only regularizes base embeddings, not the combined embedding

### Why Removing Auxiliary Components Helps

With the 40x mismatch, each auxiliary component adds a large-magnitude, low-information signal:
- **Modality**: only 5 unique values → near-constant for most entities
- **Type**: only 12 unique values, 78% are "Finding" → near-constant for most entities
- **PID**: only affects cross-modal triples, adds same modulation to re and im

Removing any one component:
1. Reduces the noise injected into the embedding
2. Slightly increases the relative contribution of the base entity embedding
3. → Small but consistent improvement

This is a **signal-to-noise ratio** problem: the auxiliary features have high magnitude but low information content (few unique values), while the base embedding has low magnitude but high information content (entity-specific).

### Additional Code Issues Found

1. **`PIDSynergy.pair_encoder`**: Uses PyTorch default `N(0,1)` init instead of xavier_uniform_ — inconsistent with `relation_encoder` which IS xavier-init'd. Output norm ~11.3 dominates combiner input.

2. **PID modulation asymmetry**: Same modulation vector added to both `re` and `im` (`augmented_re = h_re + mod; augmented_im = h_im + mod`). This is not complex-aware — ComplEx scoring treats re and im differently, so identical modulation breaks the complex bilinear structure.

3. **PID head-only augmentation**: Only head entities are augmented, tails are not. Asymmetric.

4. **Dead parameters in ablation modes**: All auxiliary components (PID synergy, cross-modal attention, modulation MLP, etc.) are always constructed in `__init__` regardless of ablation flag. Ablated components receive no gradients but are still affected by weight_decay and waste optimizer memory.

5. **`_feature_tensor`/`_feature_mask` not registered buffers**: If model is moved to GPU after `set_precomputed_features()`, these tensors stay on CPU → device mismatch.

6. **Resume doesn't restore optimizer/scheduler state**: Fresh optimizer and scheduler are constructed on resume, losing momentum and LR schedule position.

## Verification

The scaling mismatch was confirmed numerically by creating matching `nn.Embedding` layers with `xavier_uniform_` init and measuring row norms:

- `entity_embeddings` re/im half norm: 0.0348
- `modality_embeddings` row norm: 1.3931 (40.0x ratio)
- `entity_type_embeddings` row norm: 1.3911 (39.9x ratio)

Script: `verify_scaling.py`

## Proposed Fix

Scale auxiliary embeddings to match base embedding magnitude at initialization:

```python
# Option A: Scale down auxiliary init to match base
scale = entity_embedding_norm / auxiliary_embedding_norm  # ~0.025
nn.init.xavier_uniform_(self.modality_embeddings.weight)
self.modality_embeddings.weight.data.mul_(scale)
nn.init.xavier_uniform_(self.entity_type_embeddings.weight)
self.entity_type_embeddings.weight.data.mul_(scale)

# Option B: Use uniform init with explicit small bound for all embeddings
# Option C: Normalize combined embedding after addition (F.normalize)
# Option D: Enable --clamp-norm 1.0 (clamps base to 1.0, but auxiliary still at 1.39)
```

After fixing, re-run ablations. If the pattern reverses (auxiliary components help), the scaling mismatch was the root cause.

## Open Questions

- [ ] Is the +0.0099 difference within noise? (Requires multi-seed runs)
- [ ] Does fixing the scaling mismatch reverse the ablation pattern?
- [ ] Was the "all" ablation (0.1588) ever actually run? (No artifacts found)
- [ ] Does the PID modulation asymmetry (same vector to re/im, head-only) hurt performance?

## File Paths

- `ckpts/cascade_v2/` — Full cascade (test MRR=0.8209)
- `ckpts/cascade_ablation_no_pid/` — no_pid ablation (test MRR=0.8251)
- `ckpts/cascade_ablation_no_type/` — no_type ablation (test MRR=0.8289)
- `ckpts/cascade_ablation_no_modality/` — no_modality ablation (test MRR=0.8308)
- `simulation/kg/models.py` — Model code (`MultimodalCASCADEModel` at line 688)
- `simulation/kg/train_manual.py` — Training script
- `simulation/kg/run_ablation.bat` — Ablation run script (no_pid, no_type, no_modality only)
- `verify_scaling.py` — Scaling mismatch verification script
- `.opencode/summaries/models_deep_analysis.md` — Full model code analysis
- `.opencode/summaries/train_manual_deep_analysis.md` — Full training code analysis