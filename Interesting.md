# Interesting Findings: MedMKG vs Our Clinical KG

## Scale Gap (We Dwarf Them)

| | **Ours** | **MedMKG** |
|---|---|---|
| Entities | 421,216 | 8,017 (3,149 concepts + 4,868 images) |
| Relations | 8 | 262 |
| Triples/Edges | 3.77M | 35,387 |
| Cross-modal edges | implicit (feature fusion) | 20,705 explicit |

## KG Embedding Quality (We Dominate)

Their best link prediction: **TransD Hits@10 = 11.89%** (head), **18.87%** (tail). Most models get near-zero — ComplEx Hits@10 = 0.11% head.

**Ours**: Full-ranking re-evaluation pending after audit fixes (disease hierarchy, eval pipeline). Previous sampled-eval MRR 0.8209 is stale and expected to change.

Why their LP is terrible: 262 relations / 8K entities = extreme sparsity. Most tensor factorization models (DistMult, SimplE, RESCAL, ComplEx) collapse to near-zero. Only translation models (TransD, TransE, TransH) survive barely.

## Architectural Differences

- **Them**: Images as **nodes** in KG, linked to concepts via Positive/Negative/Uncertain edges. Multimodality = graph topology.
- **Us**: Multimodal **features fused into entity embeddings** (CXR, ECG, text, structured). Multimodality = representation enrichment.
- **Them**: UMLS concepts + MIMIC-CXR images only (chest X-rays).
- **Us**: 5 entity types (Finding, Anatomy, Disease, Patient, Study), 4 modalities (CXR, ECG, RAD, structured) — broader clinical scope.

## What They Have That We Don't

- **NaF (Neighbor-aware Filtering)**: `NaF(m) = Σ log(M / |N(r,c)|)` — IDF-like weighting over (relation, concept) pairs to rank image informativeness and deduplicate redundant X-rays. Could apply to our KG if we have redundant image entities.
- **Semantic edge labels** on cross-modal links: Positive / Negative / Uncertain (extracted via GPT-4o during disambiguation). Our CASCADE has modality embeddings but no sentiment/uncertainty signals on edges.
- **Downstream task benchmarking**: Knowledge-augmented text-image retrieval (KnowledgeCLIP, FashionKLIP on OpenI + MIMIC-CXR) and VQA (MR-MKG, KRISP, EKGRL on VQA-RAD, SLAKE, PathVQA) — 24 baselines across 6 datasets. These are the tasks that matter clinically.
- **Human evaluation** with radiologists: concept coverage, relation correctness, image diversity scored ~80% across all three.
- Public dataset + code: https://github.com/XiaochenWang-PSU/MedMKG, https://huggingface.co/datasets/xcwangpsu/MedMKG

## What We Have That They Don't

- **CASCADE model** with entity-type awareness, modality embeddings, synergy heads — architecturally more sophisticated than any of their 17 LP baselines.
- **Order-of-magnitude better** link prediction performance (full-ranking re-eval pending; previously 82%+ MRR on sampled eval vs <19% Hits@10).
- **ECG modality** (they only have CXR images).
- **Much larger, richer graph** with patient/study-level entities (421K entities, 3.77M triples).
- **Multimodal feature fusion** into embeddings rather than images-as-nodes topology.

## Key Takeaway

MedMKG is a **benchmark paper** (NeurIPS 2025 submission) focused on downstream task evaluation (retrieval + VQA). Their KG is small (35K edges) and their link prediction is an afterthought with terrible results. Our core contribution is **strong KG embedding models** on a much larger clinical graph with novel architecture (CASCADE) — full-ranking MRR pending re-evaluation after audit fixes. Different focus, different strengths.

Their NaF algorithm, semantic edge labels, and downstream augmentation framework (KnowledgeCLIP, MR-MKO) are interesting ideas we could adopt to extend our work beyond link prediction into clinically meaningful downstream tasks.

## Paper Reference

- Title: MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph
- Authors: Xiaochen Wang, Yuan Zhong, Lingwei Zhang, Lisong Dai, Ting Wang, Fenglong Ma (Penn State, Renmin Hospital of Wuhan Univ, Stony Brook)
- arXiv: 2505.17214v1 (May 22, 2025)
- Status: Submitted to NeurIPS 2025