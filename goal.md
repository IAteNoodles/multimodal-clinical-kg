# CASCADE-KG: PID-Aware Multimodal Clinical Knowledge Graph

## Problem
Current "multimodal" KG in `simulation/kg/` is **not truly multimodal** — it uses modality labels (FND_CXR_*, FND_ECG_*, FND_RAD_*) as entity prefixes but never loads or processes actual images, waveforms, or text. It's a standard KG embedding system with modality tags.

## Goal
Build a **truly multimodal clinical KG** that:
1. Encodes **real data** from each modality (CXR images, ECG waveforms, radiology text, structured clinical data)
2. Uses **cross-modal fusion** (not just label prefixes) to learn joint representations
3. Evaluates across **4 modality settings** to measure each modality's contribution

## Modality Settings (Ablation Study)
| Setting | Modalities | Purpose |
|---------|-----------|---------|
| **T** | Text only (RadGraph entities + clinical notes) | Baseline — symbolic KG |
| **T+I** | Text + CXR images | Add imaging |
| **T+I+E** | Text + CXR + ECG waveforms | Add time-series |
| **T+I+E+S** | Text + CXR + ECG + Structured (labs, vitals, demographics) | Full multimodal |

## Data Sources
| Modality | Source | Location | Status |
|----------|--------|----------|--------|
| CXR Images | MIMIC-CXR-JPG | `simulation/data/mimic_cxr_jpg/images/` (566 JPGs, p10 prefix) | On disk, tiny sample |
| CXR Labels | CheXpert (14 pathologies) | `simulation/data/mimic_cxr_jpg/mimic-cxr-2.0.0-chexpert.csv` | On disk, 227K rows |
| CXR Metadata | DICOM headers | `simulation/data/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv` | On disk |
| ECG Waveforms | MIMIC-IV-ECG demo | `simulation/data/mimic_iv_ecg_demo/waveforms/` (659 records) | On disk |
| ECG Reference | PTB-XL (21,837 records) | `simulation/data/ptb_xl/` | On disk |
| Radiology Text | RadGraph (NLP extractions) | `simulation/data/radgraph/` | On disk |
| Structured Clinical | MIMIC-IV demo (22 tables) | `simulation/data/mimic_iv_demo/` | On disk |
| Structured (5K) | MIMIC-IV v3.1 via BigQuery | `Exploration-MJ/data/mimic_5k/` | On disk |
| UMLS Ontology | 2026AA release | `Exploration-MJ/data/umls-2026AA-mrconso/` | On disk |
| Clinical Notes | MIMIC-IV-Note (discharge summaries) | **NOT DOWNLOADED** | Need via BigQuery |

## Key Gaps to Address
1. **No actual image data** — need CXR encoder (DenseNet-121/ViT) producing visual embeddings
2. **No actual waveform data** — need ECG encoder (1D-CNN) producing signal embeddings
3. **No actual text data** — need ClinicalBERT encoder for radiology reports/notes
4. **No structured data integration** — labs, vitals, demographics not in KG
5. **Cross-modal edges are hand-crafted** — need data-driven cross-modal alignment
6. **Only 566 CXR images** — need more (full MIMIC-CXR = ~377K images, ~250GB)
7. **Only 659 ECG waveforms** — PTB-XL has 21,837, MIMIC-IV-ECG has ~1.6M

## Architecture (Redesign from Scratch)
```
Modality Encoders:
  CXR Image → DenseNet-121 → 512-d visual embedding
  ECG Signal → 1D-CNN → 256-d temporal embedding
  Clinical Text → ClinicalBERT → 768-d text embedding
  Structured → MLP → 128-d clinical embedding

Fusion:
  Cross-attention between modality embeddings
  PID-weighted synergy (core innovation)
  Unified entity representation

KG Embedding:
  ComplEx base (best performing in v1)
  PID-modulated cross-modal scoring
```

## Evaluation
- **Task**: Link prediction (MRR, Hits@1/3/10)
- **Primary metric**: Cross-modal MRR (predicting across modality boundaries)
- **Stratified evaluation**: By relation type and modality pair
- **Bootstrap CIs**: 1000 samples, 95% CI

## Baselines
1. TransE (unimodal KG embedding)
2. ComplEx (best v1 baseline)
3. MKGFormer-style (image + text fusion, if data permits)
4. MEDMKG-style (UMLS + imaging, if data permits)

## Success Criteria
- CASCADE-KG (T+I+E+S) > ComplEx on cross-modal MRR by ≥10%
- Each modality addition shows measurable improvement: T < T+I < T+I+E < T+I+E+S
- Learned PID synergy matrix shows clinically interpretable patterns

## Hardware Constraints
- RTX 4050 6GB VRAM
- ~2GB free RAM
- ~24GB free disk
- Gradient accumulation required
- Batch size auto-tuned per modality setting

## Timeline
1. **Phase 1**: Data audit + download missing data (CXR images, clinical notes)
2. **Phase 2**: Build modality encoders (CXR, ECG, text, structured)
3. **Phase 3**: Rebuild KG with real multimodal entities
4. **Phase 4**: Implement multimodal fusion model
5. **Phase 5**: Train all 4 settings + baselines
6. **Phase 6**: Analyze PID synergy + write results
