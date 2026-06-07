# train.py summary

## lines 1-55
- imports: __future__, sys, pathlib, argparse, json, pickle, gc, random, datetime, typing (Dict, List, Optional, Set, Tuple), numpy, torch, torch.nn, FloatTensor, LongTensor, torch.nn.functional, AdamW, CosineAnnealingLR, LambdaLR
- local imports: KGTriplesDataset, MultimodalKGTriplesDataset, LinkPredictionEvaluator, NegativeSampler, MODALITY_SET_MAP, ClinicalKG, Entity, Relation, CASCADEKGModel, ComplExModel, MultimodalComplExModel, MultimodalCASCADEModel, InfoNCELoss, TransEModel
- tqdm fallback: defines minimal tqdm replacement if tqdm not installed (prints every 100 items)

## lines 57-63
- set_seed(seed) → sets random, numpy, torch, cuda seeds for reproducibility

## lines 65-81
- CASCADEWrapper(nn.Module) → wraps CASCADEKGModel to adapt score() from 5-arg to 3-arg (h, r, t) by storing entity_type_ids and entity_modality_ids
- __getattr__ delegates to inner model for non-wrapper attrs

## lines 83-98
- MultimodalWrapper(nn.Module) → wraps MultimodalComplExModel to adapt score() from 4-arg to 3-arg (h, r, t) by storing entity_modality_ids
- __getattr__ delegates to inner model

## lines 100-106
- _is_multimodal(model) → checks if model is MultimodalComplExModel or MultimodalCASCADEModel
- _is_multimodal_cascade(model) → checks if model is MultimodalCASCADEModel

## lines 108-367
- train_model(model, dataset, evaluator, args, active_modalities) → main training loop
  - Moves model to device, enables cuDNN benchmark on CUDA
  - Two param groups: high_lr (modulation, cross_modal, pid_synergy, modality_embed, type_embed, has_modality) and base, plus loss_fn params
  - Uses InfoNCELoss with temperature and label_smoothing
  - AdamW optimizer with separate LR groups; NegativeSampler for negatives
  - AMP GradScaler on CUDA
  - Cosine LR schedule with warmup via LambdaLR
  - Gradient accumulation, gradient clipping (max_norm=1.0), embedding clamping
  - SWA (Stochastic Weight Averaging) from configurable epoch fraction
  - Early stopping on val MRR patience
  - Checkpoint save/resume support
  - Final evaluation on test split after loading best checkpoint
  - Returns dict with final_train_loss, val_mrr, test metrics

## lines 370-430
- evaluate_model(model, dataset, evaluator, split, device, batch_size, max_triples, num_eval_negatives, return_details) → evaluates model on a split
  - Wraps model via CASCADEWrapper/MultimodalWrapper as needed
  - Falls back to CPU on CUDA OOM, then restores model to original device
  - Delegates to LinkPredictionEvaluator.evaluate()

## lines 433-467
- compute_model_params(model_name, num_entities, num_relations, embed_dim, num_entity_types, num_modalities, ablation) → computes total parameter count for a model config
  - Handles transe, complex, cascade (with ablation variants), multimodal_complex, multimodal_cascade

## lines 470-491
- _find_matching_embed_dim(target_params, model_name, ...) → searches embed_dim in [16..2048) to match target param count for fair comparison

## lines 494-551
- build_model(args, dataset) → constructs model by name with config from args
  - transe: optional param-matching via _find_matching_embed_dim
  - complex: same param-matching logic
  - cascade: with ablation support
  - multimodal_complex: with modality set, synergy_dim=64, num_heads=4
  - multimodal_cascade: full config with ablation + modalities

## lines 554-708
- auto_tune_config(model, dataset, args) → auto-tunes batch_size, num_negatives, grad_accum_steps, eval_batch_size based on VRAM
  - Runs trial forward+backward pass to measure per-sample memory
  - Computes optimal batch sizes from available VRAM (95% safety factor)
  - Falls back to minimum settings on OOM
  - Adjusts eval_batch_size with higher multiplier for cascade models

## lines 711-762
- bootstrap_ci(per_triple_data, n_bootstrap, ci) → bootstrap confidence intervals for MRR, Hits@1/3/10 from per-triple rank data
- bootstrap_ci_cross_modal(per_triple_data, ...) → same but filtered to cross-modal triples only (head_type != tail_type)

## lines 765-820
- tune_hparams(dataset, evaluator, args) → random search over lr, dropout, weight_decay, temperature, label_smoothing, n3_weight
  - Runs args.tune_trials trials, tracks best val_mrr, saves results to tune_results.json

## lines 823-901
- main() → argparse setup with extensive CLI options
  - Model selection, training hyperparams, evaluation params, SWA, ablation, modality, benchmark, tuning, feature precomputation, test KG flag

## lines 902-1000
- main() continued: dataset loading logic
  - Routes to MultimodalKGTriplesDataset or KGTriplesDataset based on model type
  - Tries efficient format first, falls back to pickle
  - Constructs run_list: benchmark mode runs 11 configs (transe, complex, cascade variants, multimodal variants with different modality sets)

## lines 1001-1106
- main() continued: training loop over run_list
  - Resets args per run, builds model, auto-tunes config, trains, evaluates
  - Computes bootstrap CIs if n_bootstrap > 0
  - Prints PID synergy matrix for cascade models
  - Saves per-model JSON results; benchmark mode saves aggregated benchmark JSON
  - Entry point: if __name__ == "__main__": main()