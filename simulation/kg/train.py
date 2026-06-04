from __future__ import annotations

import argparse
import json
import pickle
import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from simulation.kg.dataset import (
    KGTriplesDataset,
    MultimodalKGTriplesDataset,
    LinkPredictionEvaluator,
    NegativeSampler,
    MODALITY_SET_MAP,
)
from simulation.kg.extract_entities import ClinicalKG, Entity, Relation
from simulation.kg.models import (
    CASCADEKGModel,
    ComplExModel,
    MultimodalComplExModel,
    MultimodalCASCADEModel,
    InfoNCELoss,
    TransEModel,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        for i, item in enumerate(iterable):
            if i % 100 == 0:
                if total:
                    print(f"\r{desc}: {i}/{total}", end="", flush=True)
                else:
                    print(f"\r{desc}: {i}", end="", flush=True)
            yield item
        print()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CASCADEWrapper(nn.Module):
    """Wraps CASCADEKGModel so its score() matches the 3-arg signature (h, r, t)."""

    def __init__(self, model: nn.Module, entity_type_ids: LongTensor, entity_modality_ids: LongTensor):
        super().__init__()
        self.model = model
        self._entity_type_ids = entity_type_ids
        self._entity_modality_ids = entity_modality_ids

    def score(self, heads: LongTensor, relations: LongTensor, tails: LongTensor) -> FloatTensor:
        return self.model.score(heads, relations, tails, self._entity_type_ids, self._entity_modality_ids)

    def __getattr__(self, name: str):
        if name in ("model", "_entity_type_ids", "_entity_modality_ids", "score"):
            return super().__getattr__(name)
        return getattr(self.model, name)


class MultimodalWrapper(nn.Module):
    """Wraps MultimodalComplExModel so its score() matches the 3-arg signature (h, r, t)."""

    def __init__(self, model: nn.Module, entity_modality_ids: LongTensor):
        super().__init__()
        self.model = model
        self._entity_modality_ids = entity_modality_ids

    def score(self, heads: LongTensor, relations: LongTensor, tails: LongTensor) -> FloatTensor:
        return self.model.score(heads, relations, tails, self._entity_modality_ids)

    def __getattr__(self, name: str):
        if name in ("model", "_entity_modality_ids", "score"):
            return super().__getattr__(name)
        return getattr(self.model, name)


def _is_multimodal(model: nn.Module) -> bool:
    return isinstance(model, (MultimodalComplExModel, MultimodalCASCADEModel))


def _is_multimodal_cascade(model: nn.Module) -> bool:
    return isinstance(model, MultimodalCASCADEModel)


def train_model(
    model: nn.Module,
    dataset: KGTriplesDataset,
    evaluator: LinkPredictionEvaluator,
    args: argparse.Namespace,
    active_modalities: Optional[Set[str]] = None,
) -> Dict[str, float]:
    device = torch.device(args.device)
    model = model.to(device)

    high_lr_params = []
    base_params = []
    high_lr_names = ('modulation', 'cross_modal', 'pid_synergy', 'modality_embed', 'type_embed', 'has_modality')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in high_lr_names):
            high_lr_params.append(param)
        else:
            base_params.append(param)
    loss_fn = InfoNCELoss(temperature=args.temperature, label_smoothing=args.label_smoothing)
    loss_fn = loss_fn.to(device)

    optimizer = AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': high_lr_params, 'lr': args.lr * args.lr_multiplier, 'weight_decay': args.weight_decay},
        {'params': loss_fn.parameters(), 'lr': args.lr, 'weight_decay': 0.0},
    ], lr=args.lr)
    neg_sampler = NegativeSampler(dataset, num_negatives=args.num_negatives)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type) if device_type == 'cuda' else None

    is_cascade = isinstance(model, CASCADEKGModel)
    is_mm = _is_multimodal(model)
    is_mm_cascade = _is_multimodal_cascade(model)

    swa_avg = None
    swa_count = 0
    swa_start_epoch = 0
    if args.swa:
        swa_start_epoch = max(1, int(args.epochs * args.swa_start))

    train_triples, train_weights = dataset.get_train_triples()
    val_triples, val_weights = dataset.get_val_triples()

    warmup_steps = getattr(args, 'warmup_steps', 0)
    total_steps = ((train_triples.size(0) + args.batch_size - 1) // args.batch_size // args.grad_accum_steps) * args.epochs
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * max(0.0, min(1.0, progress))))
    scheduler = LambdaLR(optimizer, lr_lambda)

    if is_cascade or is_mm:
        entity_type_ids = dataset.get_entity_type_ids().to(device)
        entity_modality_ids = dataset.get_entity_modality_ids().to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = []
    if args.ablation:
        suffix_parts.append(args.ablation)
    if active_modalities is not None:
        mod_str = "+".join(sorted(active_modalities))
        suffix_parts.append(mod_str)
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    best_ckpt_path = output_dir / f"{args.model}{suffix}_best.pt"

    best_val_mrr = 0.0
    patience_counter = 0
    start_epoch = 1
    results: Dict[str, float] = {}

    if args.resume and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if args.resume_optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_mrr = ckpt.get('metrics', {}).get('MRR', 0.0)
        print(f"Resumed from epoch {start_epoch - 1} (val MRR={best_val_mrr:.4f})")

    if active_modalities is not None:
        print(f"Active modalities: {', '.join(sorted(active_modalities))}")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_triples.size(0))
        shuffled = train_triples[perm]
        epoch_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        pbar = tqdm(
            range(0, shuffled.size(0), args.batch_size),
            desc=f"Epoch {epoch}/{args.epochs}",
            total=(shuffled.size(0) + args.batch_size - 1) // args.batch_size,
        )

        for accum_step, start in enumerate(pbar):
            end = min(start + args.batch_size, shuffled.size(0))
            batch_triples = shuffled[start:end].to(device)

            neg_triples, _ = neg_sampler.sample(batch_triples.cpu())
            neg_triples = neg_triples.to(device)

            num_neg_total = neg_triples.size(0) // batch_triples.size(0)
            if num_neg_total == 0:
                continue

            pos_heads = batch_triples[:, 0]
            pos_rels = batch_triples[:, 1]
            pos_tails = batch_triples[:, 2]

            neg_heads = neg_triples[:, 0]
            neg_rels = neg_triples[:, 1]
            neg_tails = neg_triples[:, 2]

            if accum_step % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                if is_mm_cascade:
                    pos_scores = model.score(pos_heads, pos_rels, pos_tails, entity_type_ids, entity_modality_ids)
                    neg_scores = model.score(neg_heads, neg_rels, neg_tails, entity_type_ids, entity_modality_ids)
                elif is_mm:
                    pos_scores = model.score(pos_heads, pos_rels, pos_tails, entity_modality_ids)
                    neg_scores = model.score(neg_heads, neg_rels, neg_tails, entity_modality_ids)
                elif is_cascade:
                    pos_scores = model.score(pos_heads, pos_rels, pos_tails, entity_type_ids, entity_modality_ids)
                    neg_scores = model.score(neg_heads, neg_rels, neg_tails, entity_type_ids, entity_modality_ids)
                else:
                    pos_scores = model.score(pos_heads, pos_rels, pos_tails)
                    neg_scores = model.score(neg_heads, neg_rels, neg_tails)

                neg_scores = neg_scores.view(batch_triples.size(0), num_neg_total)
                pos_scores = pos_scores.unsqueeze(1)

                loss = loss_fn(pos_scores, neg_scores)
                if args.n3_weight > 0 and hasattr(model, 'n3_penalty'):
                    loss = loss + args.n3_weight * model.n3_penalty(pos_heads, pos_rels, pos_tails)

            loss_val = loss.detach()
            scaled_loss = loss / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
                if scaler.get_scale() == 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                scaled_loss.backward()

            del pos_scores, neg_scores, scaled_loss, loss, batch_triples, neg_triples
            del pos_heads, pos_rels, pos_tails, neg_heads, neg_rels, neg_tails

            if (accum_step + 1) % args.grad_accum_steps == 0 or (start + args.batch_size) >= shuffled.size(0):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if args.embed_norm > 0:
                        with torch.no_grad():
                            model.entity_embeddings.weight.clamp_(-args.embed_norm, args.embed_norm)
                    if args.embed_max_norm > 0 and hasattr(model, 'clamp_embed_norm'):
                        model.clamp_embed_norm(max_norm=args.embed_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if args.embed_norm > 0:
                        with torch.no_grad():
                            model.entity_embeddings.weight.clamp_(-args.embed_norm, args.embed_norm)
                    if args.embed_max_norm > 0 and hasattr(model, 'clamp_embed_norm'):
                        model.clamp_embed_norm(max_norm=args.embed_max_norm)
                    optimizer.step()
                scheduler.step()
                global_step += 1

            epoch_loss = epoch_loss + loss_val
            num_batches += 1

            if num_batches % 10 == 0:
                pbar.set_postfix(loss=f"{epoch_loss.item() / max(num_batches, 1):.4f}")

        avg_loss = epoch_loss.item() / max(num_batches, 1)

        val_metrics: Dict[str, float] = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            val_metrics = evaluate_model(model, dataset, evaluator, split='val', device=args.device, batch_size=args.eval_batch_size, max_triples=args.max_eval_triples, num_eval_negatives=args.eval_negatives)
            val_mrr = val_metrics.get("MRR", 0.0)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                patience_counter = 0
                if args.save_checkpoints:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'metrics': val_metrics,
                    }, best_ckpt_path)
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"Val MRR: {val_metrics.get('MRR', 0.0):.4f} | "
                f"H@1: {val_metrics.get('Hits@1', 0.0):.4f} | "
                f"H@3: {val_metrics.get('Hits@3', 0.0):.4f} | "
                f"H@10: {val_metrics.get('Hits@10', 0.0):.4f} | "
                f"Patience: {patience_counter}/{args.patience}"
            )

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break
        else:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if swa_avg is not None and epoch >= swa_start_epoch:
            for name, param in model.named_parameters():
                if name not in swa_avg:
                    swa_avg[name] = param.data.clone()
                else:
                    swa_avg[name].add_(param.data)
            swa_count += 1
        elif args.swa and epoch >= swa_start_epoch and swa_avg is None:
            swa_avg = {}
            for name, param in model.named_parameters():
                swa_avg[name] = param.data.clone()
            swa_count = 1

    if swa_avg is not None and swa_count > 0:
        for name, param in model.named_parameters():
            if name in swa_avg:
                param.data.copy_(swa_avg[name] / swa_count)
        print(f"SWA: averaged weights over {swa_count} epochs")
        del swa_avg

    if args.save_checkpoints and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print("No best checkpoint saved, using current model state")

    results["final_train_loss"] = avg_loss
    results["val_mrr"] = best_val_mrr

    if args.device == "cuda":
        torch.cuda.empty_cache()
    test_metrics = evaluate_model(model, dataset, evaluator, split='test', device=args.device, batch_size=args.eval_batch_size, max_triples=args.max_eval_triples, num_eval_negatives=args.eval_negatives)
    for k, v in test_metrics.items():
        results[f"test_{k.lower().replace('@', '_')}"] = v

    return results


def evaluate_model(
    model: nn.Module,
    dataset: KGTriplesDataset,
    evaluator: LinkPredictionEvaluator,
    split: str = 'test',
    device: Optional[str] = None,
    batch_size: int = 256,
    max_triples: Optional[int] = None,
    num_eval_negatives: int = 50,
    return_details: bool = False,
):
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    device_str = str(dev)

    if split == 'val':
        triples, weights = dataset.get_val_triples()
    elif split == 'test':
        triples, weights = dataset.get_test_triples()
    else:
        triples, weights = dataset.get_train_triples()

    is_cascade = isinstance(model, CASCADEKGModel)
    is_mm = _is_multimodal(model)
    is_mm_cascade = _is_multimodal_cascade(model)

    if is_cascade or is_mm:
        entity_type_ids = dataset.get_entity_type_ids().to(dev)
        entity_modality_ids = dataset.get_entity_modality_ids().to(dev)

    if is_mm_cascade:
        wrapped = CASCADEWrapper(model, entity_type_ids, entity_modality_ids)
    elif is_mm:
        wrapped = MultimodalWrapper(model, entity_modality_ids)
    elif is_cascade:
        wrapped = CASCADEWrapper(model, entity_type_ids, entity_modality_ids)
    else:
        wrapped = model

    try:
        result = evaluator.evaluate(wrapped, triples, weights, batch_size=batch_size, device=device_str, max_triples=max_triples, num_eval_negatives=num_eval_negatives, return_details=return_details)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[WARN] CUDA OOM during evaluation, falling back to CPU")
            torch.cuda.empty_cache()
            model_cpu = model.cpu()
            if is_mm_cascade or is_cascade:
                wrapped = CASCADEWrapper(model_cpu, entity_type_ids.cpu(), entity_modality_ids.cpu())
            elif is_mm:
                wrapped = MultimodalWrapper(model_cpu, entity_modality_ids.cpu())
            else:
                wrapped = model_cpu
            result = evaluator.evaluate(wrapped, triples, weights, batch_size=batch_size, device='cpu', max_triples=max_triples, num_eval_negatives=num_eval_negatives, return_details=return_details)
            del model_cpu
            gc.collect()
            model.to(dev)
        else:
            raise

    return result


def compute_model_params(
    model_name: str,
    num_entities: int,
    num_relations: int,
    embed_dim: int,
    num_entity_types: int = 5,
    num_modalities: int = 4,
    ablation: Optional[str] = None,
) -> int:
    """Compute total parameter count for a given model configuration."""
    if model_name == 'transe':
        return num_entities * embed_dim + num_relations * embed_dim
    elif model_name == 'complex':
        return num_entities * embed_dim * 2 + num_relations * embed_dim * 2
    elif model_name == 'cascade':
        params = num_entities * embed_dim * 2 + num_relations * embed_dim * 2
        if ablation != 'no_modality':
            params += num_modalities * embed_dim
        if ablation != 'no_type':
            params += num_entity_types * embed_dim
        if ablation != 'no_pid':
            params += num_modalities * num_modalities
        return params
    elif model_name in ('multimodal_complex', 'multimodal_cascade'):
        params = num_entities * embed_dim * 2 + num_relations * embed_dim * 2
        params += num_modalities * embed_dim
        params += num_entities * 1
        params += 64 * 64 + 64
        params += 256 * 64 + 64
        params += num_relations * 128 + (128 * 2) * 128 + 128
        if model_name == 'multimodal_cascade':
            params += num_entity_types * embed_dim
        return params
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _find_matching_embed_dim(
    target_params: int,
    model_name: str,
    num_entities: int,
    num_relations: int,
    num_entity_types: int = 5,
    num_modalities: int = 4,
) -> int:
    """Find embed_dim for baseline model that gives closest param count to target."""
    best_dim = 16
    best_diff = abs(compute_model_params(model_name, num_entities, num_relations, 16, num_entity_types, num_modalities) - target_params)

    for dim in range(24, 2048, 8):
        params = compute_model_params(model_name, num_entities, num_relations, dim, num_entity_types, num_modalities)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim
        if params > target_params * 1.5:
            break

    return best_dim


def build_model(args: argparse.Namespace, dataset: KGTriplesDataset) -> nn.Module:
    num_ents = dataset.num_entities
    num_rels = dataset.num_relations

    if args.model == 'transe':
        embed_dim = args.embed_dim
        if args.match_params:
            cascade_params = compute_model_params('cascade', num_ents, num_rels, args.embed_dim)
            embed_dim = _find_matching_embed_dim(cascade_params, 'transe', num_ents, num_rels)
            matched_params = compute_model_params('transe', num_ents, num_rels, embed_dim)
            print(f"  Param-matched TransE: embed_dim={embed_dim}, params={matched_params:,} (CASCADE ref={cascade_params:,})")
        return TransEModel(num_ents, num_rels, embed_dim, margin=args.margin, dropout=args.dropout)
    elif args.model == 'complex':
        embed_dim = args.embed_dim
        if args.match_params:
            cascade_params = compute_model_params('cascade', num_ents, num_rels, args.embed_dim)
            embed_dim = _find_matching_embed_dim(cascade_params, 'complex', num_ents, num_rels)
            matched_params = compute_model_params('complex', num_ents, num_rels, embed_dim)
            print(f"  Param-matched ComplEx: embed_dim={embed_dim}, params={matched_params:,} (CASCADE ref={cascade_params:,})")
        return ComplExModel(num_ents, num_rels, embed_dim, dropout=args.dropout)
    elif args.model == 'cascade':
        ablation = getattr(args, 'ablation', None)
        return CASCADEKGModel(
            num_ents, num_rels, args.embed_dim,
            num_entity_types=5,
            num_modalities=4,
            ablation=ablation,
            dropout=args.dropout,
        )
    elif args.model == 'multimodal_complex':
        modalities = MODALITY_SET_MAP.get(args.modalities, {"text"})
        return MultimodalComplExModel(
            num_ents, num_rels, args.embed_dim,
            num_modalities=4,
            synergy_dim=64,
            num_heads=4,
            dropout=args.dropout,
            use_modality_encoders=args.use_modality_encoders,
            use_pretrained_encoders=False,
            modalities=modalities,
        )
    elif args.model == 'multimodal_cascade':
        modalities = MODALITY_SET_MAP.get(args.modalities, {"text"})
        ablation = getattr(args, 'ablation', None)
        return MultimodalCASCADEModel(
            num_ents, num_rels, args.embed_dim,
            num_entity_types=5,
            num_modalities=4,
            synergy_dim=64,
            num_heads=4,
            dropout=args.dropout,
            use_modality_encoders=args.use_modality_encoders,
            use_pretrained_encoders=False,
            modalities=modalities,
            ablation=ablation,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def auto_tune_config(
    model: nn.Module,
    dataset: KGTriplesDataset,
    args: argparse.Namespace,
) -> None:
    """Auto-tune batch_size, num_negatives, grad_accum_steps, eval_batch_size
    based on available VRAM. Does a trial forward+backward to measure per-sample
    memory, then computes optimal settings. Modifies args in place."""

    if args.device == 'cpu':
        return

    print("  [AUTO-TUNE] Probing VRAM...", flush=True)
    device = torch.device(args.device)
    is_cascade = isinstance(model, CASCADEKGModel)
    is_mm = _is_multimodal(model)
    is_mm_cascade = _is_multimodal_cascade(model)

    model = model.to(device)
    torch.cuda.empty_cache()

    entity_type_ids = dataset.get_entity_type_ids().to(device) if (is_cascade or is_mm) else None
    entity_modality_ids = dataset.get_entity_modality_ids().to(device) if (is_cascade or is_mm) else None
    train_triples, _ = dataset.get_train_triples()

    print("  [AUTO-TUNE] Warming up forward pass...", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        sample = train_triples[:1].to(device)
        if is_mm_cascade:
            model.score(sample[:, 0], sample[:, 1], sample[:, 2], entity_type_ids, entity_modality_ids)
        elif is_mm:
            model.score(sample[:, 0], sample[:, 1], sample[:, 2], entity_modality_ids)
        elif is_cascade:
            model.score(sample[:, 0], sample[:, 1], sample[:, 2], entity_type_ids, entity_modality_ids)
        else:
            model.score(sample[:, 0], sample[:, 1], sample[:, 2])
    model_mem = torch.cuda.max_memory_allocated(device)
    print(f"  [AUTO-TUNE] Model memory: {model_mem/1e9:.2f} GB", flush=True)

    print("  [AUTO-TUNE] Running trial backward pass...", flush=True)
    trial_bs = min(1, train_triples.size(0))
    neg_sampler = NegativeSampler(dataset, num_negatives=1)
    batch = train_triples[:trial_bs].to(device)
    neg_triples, _ = neg_sampler.sample(batch.cpu())
    neg_triples = neg_triples.to(device)
    num_neg_total = max(neg_triples.size(0) // trial_bs, 1)

    if is_mm or is_mm_cascade:
        if entity_modality_ids is not None and entity_modality_ids.shape[0] > 1:
            h_mod = entity_modality_ids[batch[0, 0].item()].item()
            t_mod = entity_modality_ids[batch[0, 2].item()].item()
            if h_mod == t_mod:
                alt_idx = (entity_modality_ids != h_mod).nonzero(as_tuple=True)[0]
                if alt_idx.numel() > 0:
                    batch[0, 2] = alt_idx[0]

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)

        with torch.amp.autocast(device_type='cuda', enabled=True):
            if is_mm_cascade:
                pos_s = model.score(batch[:, 0], batch[:, 1], batch[:, 2], entity_type_ids, entity_modality_ids)
                neg_s = model.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2], entity_type_ids, entity_modality_ids)
            elif is_mm:
                pos_s = model.score(batch[:, 0], batch[:, 1], batch[:, 2], entity_modality_ids)
                neg_s = model.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2], entity_modality_ids)
            elif is_cascade:
                pos_s = model.score(batch[:, 0], batch[:, 1], batch[:, 2], entity_type_ids, entity_modality_ids)
                neg_s = model.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2], entity_type_ids, entity_modality_ids)
            else:
                pos_s = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
                neg_s = model.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
            neg_s = neg_s.view(trial_bs, num_neg_total)
            pos_s = pos_s.unsqueeze(1)
            logits = torch.cat([pos_s, neg_s], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits / 0.1, labels)

        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated(device)
        batch_overhead = peak_mem - model_mem
        print(f"  [AUTO-TUNE] Trial peak: {peak_mem/1e9:.2f} GB, overhead: {batch_overhead/1e9:.2f} GB", flush=True)

        optimizer.zero_grad(set_to_none=True)
        del optimizer
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            model.zero_grad()
            args.batch_size = 2
            args.num_negatives = 1
            target_eff = args.batch_size * args.grad_accum_steps
            args.grad_accum_steps = max(1, target_eff // args.batch_size)
            args.eval_batch_size = 8
            print(f"  [AUTO-TUNE] Trial OOM — falling back to minimum: bs=2, neg=1, ga={args.grad_accum_steps}")
            return
        raise

    model.zero_grad()
    del batch, neg_triples, pos_s, neg_s, loss, sample, neg_sampler
    if is_cascade or is_mm:
        del entity_type_ids, entity_modality_ids
    torch.cuda.empty_cache()

    total_samples = trial_bs * (1 + num_neg_total)
    mem_per_sample = (batch_overhead / total_samples) * 1.2
    if is_mm or is_mm_cascade:
        mem_per_sample *= 1.3

    total_vram = torch.cuda.get_device_properties(device).total_memory
    safety_factor = 0.85
    usable_vram = total_vram * safety_factor
    available = usable_vram - model_mem

    orig_bs = args.batch_size
    orig_neg = args.num_negatives
    orig_ga = args.grad_accum_steps
    orig_eval_bs = args.eval_batch_size
    target_effective = args.batch_size * args.grad_accum_steps

    if available <= 0:
        print(f"  [AUTO-TUNE] Model ({model_mem/1e9:.2f} GB) exceeds {safety_factor*100:.0f}% of {total_vram/1e9:.1f} GB VRAM")
        args.batch_size = 2
        args.num_negatives = 1
        args.grad_accum_steps = max(1, target_effective // args.batch_size)
        args.eval_batch_size = 8
        return

    max_bs = int(available / (mem_per_sample * (1 + args.num_negatives)))
    max_bs = max(4, min(max_bs, 256))

    if max_bs >= args.batch_size:
        args.batch_size = max_bs
    else:
        args.batch_size = max(4, max_bs)
        while args.num_negatives > 2 and args.batch_size < 16:
            args.num_negatives -= 1
            max_bs = int(available / (mem_per_sample * (1 + args.num_negatives)))
            args.batch_size = max(4, min(max_bs, 256))

    args.grad_accum_steps = max(1, (target_effective + args.batch_size - 1) // args.batch_size)

    eval_multiplier = 3.5 if is_cascade else 2.5
    eval_mem_per_triple = mem_per_sample * (args.eval_negatives + 1) * eval_multiplier
    max_eval_bs = int(available / max(eval_mem_per_triple, 1))
    args.eval_batch_size = max(4, min(max_eval_bs, 256))

    print(f"  [AUTO-TUNE] VRAM {total_vram/1e9:.1f}GB | Model {model_mem/1e9:.2f}GB | Available {available/1e9:.2f}GB | Per-sample {mem_per_sample/1e6:.1f}MB")
    print(f"  [AUTO-TUNE] bs {orig_bs}->{args.batch_size} | neg {orig_neg}->{args.num_negatives} | ga {orig_ga}->{args.grad_accum_steps} | eff {args.batch_size*args.grad_accum_steps} | eval_bs {orig_eval_bs}->{args.eval_batch_size}")


def bootstrap_ci(
    per_triple_data: List[Dict],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals from per-triple rank data.

    Each entry in per_triple_data must have a 'rank' key.
    Returns dict with mean and CI bounds for MRR, Hits@1, Hits@3, Hits@10.
    """
    if not per_triple_data:
        return {}

    ranks = np.array([d['rank'] for d in per_triple_data])
    n = len(ranks)

    rng = np.random.RandomState(42)
    boot_metrics = {'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': []}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = ranks[idx]
        boot_metrics['MRR'].append(np.mean(1.0 / sample))
        boot_metrics['Hits@1'].append(np.mean(sample <= 1))
        boot_metrics['Hits@3'].append(np.mean(sample <= 3))
        boot_metrics['Hits@10'].append(np.mean(sample <= 10))

    alpha = (1.0 - ci) / 2.0
    result = {}
    for metric_name, values in boot_metrics.items():
        arr = np.array(values)
        result[f'{metric_name}_mean'] = float(np.mean(arr))
        result[f'{metric_name}_ci_low'] = float(np.percentile(arr, alpha * 100))
        result[f'{metric_name}_ci_high'] = float(np.percentile(arr, (1.0 - alpha) * 100))

    return result


def bootstrap_ci_cross_modal(
    per_triple_data: List[Dict],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Compute bootstrap CIs for cross-modal triples only.

    Each entry must have 'rank', 'head_type', 'tail_type' keys.
    A triple is cross-modal if head_type != tail_type.
    """
    cross_modal = [d for d in per_triple_data if d.get('head_type') != d.get('tail_type')]
    if not cross_modal:
        return {}
    return bootstrap_ci(cross_modal, n_bootstrap=n_bootstrap, ci=ci)


def tune_hparams(
    dataset: KGTriplesDataset,
    evaluator: LinkPredictionEvaluator,
    args: argparse.Namespace,
) -> None:
    """Random search over hyperparameters."""
    import math

    search_space = {
        'lr': [1e-4, 3e-4, 5e-4, 1e-3],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'weight_decay': [1e-4, 1e-3, 1e-2],
        'temperature': [0.05, 0.1, 0.2, 0.5],
        'label_smoothing': [0.0, 0.05, 0.1, 0.15],
        'n3_weight': [0.0, 1e-4, 1e-3, 1e-2],
    }

    best_val_mrr = -1.0
    best_config = None
    results_log = []

    for trial in range(args.tune_trials):
        config = {k: random.choice(v) for k, v in search_space.items()}
        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{args.tune_trials}: {config}")

        for k, v in config.items():
            setattr(args, k, v)

        set_seed(args.seed)
        model = build_model(args, dataset)
        auto_tune_config(model, dataset, args)

        display_name = f"tune_{args.model}"
        print(f"Training {display_name} (trial {trial+1})...")
        trial_results = train_model(model, dataset, evaluator, args)

        val_mrr = trial_results.get('val_mrr', 0.0)
        config['val_mrr'] = val_mrr
        results_log.append(config)
        print(f"Trial {trial+1} val MRR: {val_mrr:.4f}")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_config = config.copy()
            print(f"  New best! MRR={best_val_mrr:.4f}")

    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tune_results.json", "w") as f:
        json.dump({"best": best_config, "all_trials": results_log}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="KG Link Prediction Training")
    parser.add_argument('--model', type=str, default='cascade',
                        choices=['transe', 'complex', 'cascade', 'multimodal_complex', 'multimodal_cascade', 'all'])
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability for embeddings (0=disabled)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for ranking loss (0=disabled)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for InfoNCE loss (lower=sharper)')
    parser.add_argument('--n3-weight', type=float, default=1e-3,
                        help='Weight for N3 regularization (L3 norm on embeddings, 0=disabled)')
    parser.add_argument('--embed-norm', type=float, default=1.0,
                        help='Clamp embedding values to [-N, N] (0=disabled)')
    parser.add_argument('--embed-max-norm', type=float, default=0.0,
                        help='Max L2 norm for embeddings via clamp_embed_norm (0=disabled, try 1.0)')
    parser.add_argument('--lr-multiplier', type=float, default=3.0,
                        help='LR multiplier for modulation/cross-attention params')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps for LR scheduler (0=no warmup)')
    parser.add_argument('--num-negatives', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--max-eval-triples', type=int, default=10000,
                        help='Max triples to evaluate (subsampled)')
    parser.add_argument('--eval-negatives', type=int, default=50,
                        help='Number of type-constrained negative samples for evaluation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='simulation/data/kg/results')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True,
                        help='Save best checkpoint when validation MRR improves')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from best checkpoint if available')
    parser.add_argument('--resume-optimizer', action='store_true',
                        help='Load optimizer state when resuming (uses more RAM)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run all models sequentially and produce a comparison JSON')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience on val MRR')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=[None, 'no_pid', 'no_modality', 'no_type'],
                        help='CASCADE ablation variant')
    parser.add_argument('--match-params', action='store_true',
                        help='Match parameter budgets across models')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples for CIs')
    parser.add_argument('--ci', type=float, default=0.95,
                        help='Confidence interval level')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning with random search')
    parser.add_argument('--tune-trials', type=int, default=20,
                        help='Number of hyperparameter trials')
    parser.add_argument('--modalities', type=str, default='text',
                        choices=['text', 'text+image', 'text+image+ecg', 'text+image+ecg+structured'],
                        help='Modality setting for multimodal models')
    parser.add_argument('--use-modality-encoders', action='store_true', default=False,
                        help='Enable running encoders during training (default: precompute features)')
    parser.add_argument('--precompute-features', action=argparse.BooleanOptionalAction, default=True,
                        help='Run encoders once and save features (default True)')
    parser.add_argument('--feature-dim', type=int, default=256,
                        help='Dimension for unified feature space')
    parser.add_argument('--test-kg', action='store_true',
                        help='Use the small 500-entity test KG instead of the full clinical KG')
    parser.add_argument('--swa', action='store_true',
                        help='Enable Stochastic Weight Averaging for last N epochs')
    parser.add_argument('--swa-start', type=float, default=0.75,
                        help='Fraction of training after which SWA starts (default 0.75)')
    parser.add_argument('--swa-lr', type=float, default=0.05,
                        help='Learning rate for SWA (default 0.05 * base lr)')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        args.device = 'cpu'

    set_seed(args.seed)

    if args.test_kg:
        efficient_dir = Path("simulation/data/kg/test_kg")
        if not efficient_dir.exists():
            efficient_dir = Path(__file__).parent.parent / "data" / "kg" / "test_kg"
    else:
        efficient_dir = Path("simulation/data/kg/clinical_kg_efficient")
        if not efficient_dir.exists():
            efficient_dir = Path(__file__).parent.parent / "data" / "kg" / "clinical_kg_efficient"

    is_multimodal_model = args.model in ('multimodal_complex', 'multimodal_cascade')
    if args.test_kg:
        feature_dir = Path("simulation/data/kg/test_kg/features")
        if not feature_dir.exists():
            feature_dir = Path(__file__).parent.parent / "data" / "kg" / "test_kg" / "features"
    else:
        feature_dir = Path("simulation/data/kg/multimodal/features")
        if not feature_dir.exists():
            feature_dir = Path(__file__).parent.parent / "data" / "kg" / "multimodal" / "features"

    if is_multimodal_model and args.precompute_features and feature_dir.exists():
        if efficient_dir.exists() and (efficient_dir / "metadata.json").exists():
            print(f"Loading multimodal dataset from: {efficient_dir}")
            dataset = MultimodalKGTriplesDataset.from_efficient(
                efficient_dir, seed=args.seed,
                feature_dir=feature_dir, feature_dim=args.feature_dim,
                modalities=args.modalities,
            )
        else:
            kg_path = Path("simulation/data/kg/clinical_kg.pkl")
            if not kg_path.exists():
                kg_path = Path(__file__).parent.parent / "data" / "kg" / "clinical_kg.pkl"
            if not kg_path.exists():
                print(f"KG not found. Searched: {efficient_dir} and {kg_path}")
                raise SystemExit(1)
            print(f"Loading from pickle: {kg_path}")
            with open(kg_path, "rb") as f:
                kg: ClinicalKG = pickle.load(f)
            dataset = MultimodalKGTriplesDataset(
                kg, seed=args.seed,
                feature_dir=feature_dir, feature_dim=args.feature_dim,
                modalities=args.modalities,
            )
    else:
        if efficient_dir.exists() and (efficient_dir / "metadata.json").exists():
            print(f"Loading from efficient format: {efficient_dir}")
            dataset = KGTriplesDataset.from_efficient(efficient_dir, seed=args.seed)
        else:
            kg_path = Path("simulation/data/kg/clinical_kg.pkl")
            if not kg_path.exists():
                kg_path = Path(__file__).parent.parent / "data" / "kg" / "clinical_kg.pkl"
            if not kg_path.exists():
                print(f"KG not found. Searched: {efficient_dir} and {kg_path}")
                raise SystemExit(1)
            print(f"Loading from pickle: {kg_path}")
            with open(kg_path, "rb") as f:
                kg: ClinicalKG = pickle.load(f)
            dataset = KGTriplesDataset(kg, seed=args.seed)

    print(f"Entities: {dataset.num_entities}, Relations: {dataset.num_relations}")

    train_t, _ = dataset.get_train_triples()
    val_t, _ = dataset.get_val_triples()
    test_t, _ = dataset.get_test_triples()
    print(f"Train: {train_t.size(0)}, Val: {val_t.size(0)}, Test: {test_t.size(0)}")

    evaluator = LinkPredictionEvaluator(dataset)

    if args.tune:
        tune_hparams(dataset, evaluator, args)
        return

    orig_batch_size = args.batch_size
    orig_num_negatives = args.num_negatives
    orig_grad_accum_steps = args.grad_accum_steps
    orig_eval_batch_size = args.eval_batch_size

    if args.benchmark or args.model == "all":
        run_list = [
            ('transe', None, 'text'),
            ('complex', None, 'text'),
            ('cascade', None, 'text'),
            ('cascade', 'no_pid', 'text'),
            ('cascade', 'no_modality', 'text'),
            ('cascade', 'no_type', 'text'),
            ('multimodal_complex', None, 'text'),
            ('multimodal_cascade', None, 'text'),
            ('multimodal_cascade', None, 'text+image'),
            ('multimodal_cascade', None, 'text+image+ecg'),
            ('multimodal_cascade', None, 'text+image+ecg+structured'),
        ]
    else:
        run_list = [(args.model, args.ablation, args.modalities)]

    all_results = {}

    for model_name, ablation, modalities in run_list:
        args.batch_size = orig_batch_size
        args.num_negatives = orig_num_negatives
        args.grad_accum_steps = orig_grad_accum_steps
        args.eval_batch_size = orig_eval_batch_size

        args.model = model_name
        args.ablation = ablation
        args.modalities = modalities
        set_seed(args.seed)

        model = build_model(args, dataset)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        mod_display = f"_{modalities}" if model_name in ('multimodal_complex', 'multimodal_cascade') else ""
        display_name = model_name if ablation is None else f"{model_name}_{ablation}"
        display_name = f"{display_name}{mod_display}"
        print(f"\nModel: {display_name} | Total params: {total_params:,} | Trainable: {trainable_params,}")

        auto_tune_config(model, dataset, args)

        active_mods = MODALITY_SET_MAP.get(modalities, {"text"})
        print(f"Training {display_name} on {args.device}...")
        results = train_model(model, dataset, evaluator, args, active_modalities=active_mods)

        if args.n_bootstrap > 0:
            print(f"Computing bootstrap CIs (n={args.n_bootstrap}, ci={args.ci})...")
            eval_result = evaluate_model(
                model, dataset, evaluator, split='test',
                device=args.device, batch_size=args.eval_batch_size,
                max_triples=args.max_eval_triples,
                num_eval_negatives=args.eval_negatives,
                return_details=True,
            )
            if isinstance(eval_result, tuple):
                test_metrics, per_triple_data = eval_result
                overall_ci = bootstrap_ci(per_triple_data, n_bootstrap=args.n_bootstrap, ci=args.ci)
                cross_modal_ci = bootstrap_ci_cross_modal(per_triple_data, n_bootstrap=args.n_bootstrap, ci=args.ci)

                for k, v in overall_ci.items():
                    results[f"test_bootstrap_{k}"] = v
                for k, v in cross_modal_ci.items():
                    results[f"test_cross_modal_bootstrap_{k}"] = v

        if model_name in ('cascade', 'multimodal_cascade'):
            synergy = model.get_pid_synergy_matrix().detach().cpu().numpy()
            print("\nLearned PID Synergy Weights (modality pairs):")
            mod_names = ["CXR", "ECG", "RAD", "None"]
            header = "        " + "  ".join(f"{n:>8s}" for n in mod_names[:synergy.shape[1]])
            print(header)
            for i, row in enumerate(synergy):
                name = mod_names[i] if i < len(mod_names) else f"M{i}"
                vals = "  ".join(f"{v:8.4f}" for v in row)
                print(f"  {name:<5s}  {vals}")
            print()

        output = {
            "model": display_name,
            "embed_dim": args.embed_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "ablation": ablation,
            "modalities": modalities,
            "epochs": args.epochs,
        }
        output.update(results)

        if not args.benchmark:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = output_dir / f"{display_name}_{timestamp}.json"
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {out_path}")

        all_results[display_name] = output

        print("\n" + "=" * 60)
        print(f"RESULTS: {display_name}")
        print("=" * 60)
        print(f"{'Metric':<35} {'Value':>10}")
        print("-" * 45)
        for k, v in sorted(output.items()):
            if isinstance(v, float):
                print(f"{k:<35} {v:>10.4f}")
            elif v is None:
                print(f"{k:<35} {'N/A':>10}")
            else:
                print(f"{k:<35} {v:>10}")

    if args.benchmark:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_path = output_dir / f"benchmark_{timestamp}.json"
        with open(bench_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBenchmark results saved to {bench_path}")


if __name__ == "__main__":
    main()
