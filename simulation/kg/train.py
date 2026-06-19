from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.8,max_split_size_mb:256",
)

import argparse
import json
import pickle
import gc
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW


class VRAMExceeded(Exception):
    def __init__(self, vram_gb: float, shared_gb: float, batch_size: int, num_negatives: int):
        self.vram_gb = vram_gb
        self.shared_gb = shared_gb
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        super().__init__(f"Shared GPU memory {shared_gb:.2f}GB detected (total used {vram_gb:.2f}GB) with bs={batch_size} neg={num_negatives}")
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from simulation.kg.dataset import (
    ENTITY_TYPE_TO_ID,
    MODALITY_TO_ID,
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


def mem_trace(label, device, verbose=True):
    """Print VRAM and RAM usage at a given point."""
    vram_alloc = torch.cuda.memory_allocated(device) / 1e9
    vram_reserved = torch.cuda.memory_reserved(device) / 1e9
    vram_peak = torch.cuda.max_memory_allocated(device) / 1e9
    ram_used = psutil.Process().memory_info().rss / 1e9
    ram_avail = psutil.virtual_memory().available / 1e9
    if verbose:
        print(f"  [MEM] {label}: VRAM alloc={vram_alloc:.2f}GB reserved={vram_reserved:.2f}GB peak={vram_peak:.2f}GB | RAM used={ram_used:.1f}GB avail={ram_avail:.1f}GB")
    return vram_alloc, vram_reserved, vram_peak, ram_used


def _cuda_device_index(device) -> int:
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str) and device.startswith('cuda'):
        parts = device.split(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return torch.cuda.current_device()
    return torch.cuda.current_device()


def _nvidia_smi_memory_gb(device) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        gpu_idx = _cuda_device_index(device)
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={gpu_idx}',
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(',')]
        total_gb = float(parts[0]) * 1024 * 1024 / 1e9
        used_gb = float(parts[1]) * 1024 * 1024 / 1e9
        free_gb = float(parts[2]) * 1024 * 1024 / 1e9
        shared_gb = max(0.0, used_gb - (total_gb - free_gb))
        return total_gb, used_gb, shared_gb
    except Exception:
        return None, None, None


def _enforce_vram_ceiling(args: argparse.Namespace, device) -> float:
    """Hard-cap PyTorch's CUDA allocator so an over-budget allocation raises a
    clean OOM (which the existing retry loop handles) instead of letting the
    Windows WDDM driver silently spill into *shared* GPU memory.

    Why this is needed: torch.cuda.memory_allocated() does NOT include the CUDA
    context, cuDNN/cuBLAS workspaces, or allocator fragmentation -- but Task
    Manager and the driver count all of it. On a 6 GB card those extras are
    ~0.7-1.0 GB, so a budget that targets ~5.3 GB of *allocated* memory
    overshoots the physical 6 GB and pages the overflow (~0.4 GB) into shared
    memory. set_per_process_memory_fraction caps the allocator's *reserved*
    bytes below that spill line, keeping every commitment inside dedicated VRAM.

    Returns the cap in GB (0.0 if not applicable / failed).
    """
    if str(getattr(device, 'type', device)) == 'cpu':
        return 0.0
    try:
        idx = _cuda_device_index(device)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        # Hold back room for the CUDA context + workspaces + fragmentation that
        # the driver counts on top of torch's allocated bytes.
        context_reserve_gb = getattr(args, 'cuda_context_reserve_gb', 1.0)
        cap_gb = total_gb - context_reserve_gb
        configured = getattr(args, 'max_vram_gb', 0.0)
        if configured and configured > 0:
            cap_gb = min(cap_gb, configured)
        cap_gb = max(0.5, cap_gb)
        frac = max(0.05, min(0.95, cap_gb / total_gb))
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        torch.cuda.empty_cache()
        print(
            f"  [VRAM-CEILING] device total={total_gb:.2f}GB, context reserve="
            f"{context_reserve_gb:.2f}GB -> torch allocator capped at "
            f"{cap_gb:.2f}GB ({frac*100:.0f}%). Over-budget allocations now OOM "
            f"(and auto-retry smaller) instead of spilling to shared memory."
        )
        return cap_gb
    except Exception as exc:
        print(f"  [VRAM-CEILING] could not set per-process memory fraction: {exc}")
        return 0.0


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad and n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile wrapper if present."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def _strip_state_dict_prefix(state_dict: dict, prefix: str = '_orig_mod.') -> dict:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _is_incompatible_state_dict_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return (
        'Missing key(s) in state_dict' in msg
        or 'Unexpected key(s) in state_dict' in msg
        or 'size mismatch for' in msg
    )


def _load_model_state_with_prefix_fallback(
    model: nn.Module,
    state_dict: dict,
    checkpoint_path: Path,
) -> bool:
    original_state = copy.deepcopy(model.state_dict())
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as exc:
        if not _is_incompatible_state_dict_error(exc):
            model.load_state_dict(original_state)
            raise

    model.load_state_dict(original_state)
    try:
        model.load_state_dict(_strip_state_dict_prefix(state_dict))
        return True
    except RuntimeError as exc:
        model.load_state_dict(original_state)
        if not _is_incompatible_state_dict_error(exc):
            raise
        print(
            f"Skipping checkpoint {checkpoint_path}: incompatible with the current model/dataset ({exc}). Starting fresh."
        )
        return False


def _safe_empty_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


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
    is_cascade = isinstance(model, CASCADEKGModel)
    is_mm = _is_multimodal(model)
    is_mm_cascade = _is_multimodal_cascade(model)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Cap the allocator BEFORE calibration so the probe steps OOM-and-shrink
        # instead of spilling into shared GPU memory.
        _enforce_vram_ceiling(args, device)
        torch.backends.cudnn.benchmark = True

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
    neg_sampler = NegativeSampler(dataset, num_negatives=args.num_negatives, device=device)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(device_type) if device_type == 'cuda' else None

    swa_avg = None
    swa_count = 0
    swa_start_epoch = 0
    if args.swa:
        swa_start_epoch = max(1, int(args.epochs * args.swa_start))

    train_triples, train_weights = dataset.get_train_triples()
    train_dataset = TensorDataset(train_triples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )
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
    if args.test_kg:
        suffix_parts.append('testkg')
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
    global_step = 0
    smooth_loss = None
    results: Dict[str, float] = {}

    last_ckpt_path = output_dir / f"{args.model}{suffix}_last.pt"
    resume_path = last_ckpt_path if args.resume and last_ckpt_path.exists() else (best_ckpt_path if args.resume and best_ckpt_path.exists() else None)
    print(f"  [RESUME] resume={args.resume} last={last_ckpt_path} exists={last_ckpt_path.exists()} best={best_ckpt_path} exists={best_ckpt_path.exists()} chosen={resume_path}")
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if _load_model_state_with_prefix_fallback(model, ckpt['model_state_dict'], resume_path):
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt and 'scheduler' in dir():
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if scaler is not None and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_mrr = ckpt.get('best_val_mrr', ckpt.get('metrics', {}).get('MRR', 0.0))
            patience_counter = ckpt.get('patience_counter', 0)
            global_step = ckpt.get('global_step', 0)
            smooth_loss = ckpt.get('smooth_loss', None)
            print(f"Resumed from epoch {start_epoch - 1} (val MRR={best_val_mrr:.4f}, patience={patience_counter})")

    avg_loss = float(smooth_loss or 0.0)
    if start_epoch > args.epochs:
        print(f"Checkpoint already finished epoch {args.epochs}, skipping training loop")

    if active_modalities is not None:
        print(f"Active modalities: {', '.join(sorted(active_modalities))}")

    ema = EMA(model, decay=0.999) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if getattr(args, 'skip_calibrate', False):
            print("  [CALIBRATE] Skipped by --skip-calibrate; using explicit bs, neg, ga")
            cal = {'batch_size': args.batch_size, 'num_negatives': args.num_negatives,
                   'grad_accum_steps': args.grad_accum_steps, 'eval_batch_size': args.eval_batch_size}
        else:
            cal = _calibrate_vram(model, dataset, neg_sampler, loss_fn, optimizer, scaler, args, device)
        if cal['batch_size'] != args.batch_size or cal['num_negatives'] != args.num_negatives:
            print(f"  [CALIBRATE] Applying: bs {args.batch_size}->{cal['batch_size']} neg {args.num_negatives}->{cal['num_negatives']} ga {args.grad_accum_steps}->{cal['grad_accum_steps']}")
            args.batch_size = cal['batch_size']
            args.num_negatives = cal['num_negatives']
            args.grad_accum_steps = cal['grad_accum_steps']
            args.eval_batch_size = cal['eval_batch_size']
            neg_sampler = NegativeSampler(dataset, num_negatives=args.num_negatives, device=device)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2 if args.num_workers > 0 else None,
                drop_last=True,
            )
            warmup_steps = getattr(args, 'warmup_steps', 0)
            total_steps = ((train_triples.size(0) + args.batch_size - 1) // args.batch_size // args.grad_accum_steps) * args.epochs
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / max(1, warmup_steps)
                decay_steps = max(1, total_steps - warmup_steps)
                progress = (step - warmup_steps) / decay_steps
                return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * max(0.0, min(1.0, progress))))
            scheduler = LambdaLR(optimizer, lr_lambda)
            print(f"  [CALIBRATE] DataLoader rebuilt: bs={args.batch_size} neg={args.num_negatives} ga={args.grad_accum_steps}")
        else:
            print(f"  [CALIBRATE] Current config optimal: bs={args.batch_size} neg={args.num_negatives} ga={args.grad_accum_steps}")
        # Fully release the large transient segments the calibration probes
        # created, so they do not linger as reserved/oversize blocks and inflate
        # driver-visible VRAM during the real training loop.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            total=len(train_loader),
        )

        for batch_idx, (batch_triples,) in pbar:
            batch_triples = batch_triples.to(device, non_blocking=True)
            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after batch to device", device)

            neg_triples, _ = neg_sampler.sample(batch_triples)
            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after neg sample", device)

            num_neg_total = neg_triples.size(0) // batch_triples.size(0)
            if num_neg_total == 0:
                continue

            pos_heads = batch_triples[:, 0]
            pos_rels = batch_triples[:, 1]
            pos_tails = batch_triples[:, 2]

            neg_heads = neg_triples[:, 0]
            neg_rels = neg_triples[:, 1]
            neg_tails = neg_triples[:, 2]
            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after all tensors on device", device)

            if batch_idx % args.grad_accum_steps == 0:
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

            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after fwd+loss", device)

            loss_val = loss.detach()
            scaled_loss = loss / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
                if scaler.get_scale() == 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
                scaled_loss.backward()

            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after bwd", device)

            del pos_scores, neg_scores, scaled_loss, loss, batch_triples, neg_triples
            del pos_heads, pos_rels, pos_tails, neg_heads, neg_rels, neg_tails

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
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
                if global_step % 50 == 0:
                    _safe_empty_cache()
                if global_step % 20 == 0 and args.device == 'cuda':
                    _, _used, _shared = _nvidia_smi_memory_gb(args.device)
                    _loss_str = f"{smooth_loss:.4f}" if smooth_loss is not None else "n/a"
                    if _used is not None:
                        pbar.set_postfix(sm=_loss_str, vram=f"{_used:.1f}G", sh=f"{_shared:.2f}G")
                    if _shared is not None and _shared > 0.5:
                        print(f"\n  [VRAM] shared={_shared:.2f}GB > 0.5GB threshold — saving checkpoint and aborting")
                        ckpt_path = output_dir / f"{args.model}{suffix}_last.pt"
                        torch.save({
                            'model_state_dict': _unwrap_model(model).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'global_step': global_step,
                            'best_val_mrr': best_val_mrr,
                            'patience_counter': patience_counter,
                            'smooth_loss': smooth_loss,
                            'args': {k: v for k, v in vars(args).items() if k != 'device'},
                        }, ckpt_path)
                        raise VRAMExceeded(vram_gb=_used, shared_gb=_shared, batch_size=args.batch_size, num_negatives=args.num_negatives)

            if batch_idx == 0 and epoch == start_epoch:
                mem_trace("train: after step", device)

            epoch_loss = epoch_loss + loss_val
            num_batches += 1

            if ema is not None:
                ema.update(model)

            raw_loss = epoch_loss.item() / max(num_batches, 1)
            if smooth_loss is None:
                smooth_loss = raw_loss
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * raw_loss
            if num_batches % 10 == 0:
                pbar.set_postfix(loss=f"{raw_loss:.4f}", sm=f"{smooth_loss:.4f}")

        avg_loss = epoch_loss.item() / max(num_batches, 1)

        val_metrics: Dict[str, float] = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            if ema is not None:
                ema.apply(model)
            val_metrics = evaluate_model(model, dataset, evaluator, split='val', device=args.device, batch_size=args.eval_batch_size, max_triples=args.max_eval_triples, num_eval_negatives=args.eval_negatives)
            if ema is not None:
                ema.restore(model)
            val_mrr = val_metrics.get("MRR", 0.0)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                patience_counter = 0
                if args.save_checkpoints:
                    torch.save({
                        'model_state_dict': _unwrap_model(model).state_dict(),
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

        if args.device == 'cuda':
            _, used_vram, shared_vram = _nvidia_smi_memory_gb(args.device)
            if used_vram is not None and shared_vram is not None:
                print(f"  [VRAM] end-epoch: used={used_vram:.2f}GB shared={shared_vram:.2f}GB")

        if args.save_checkpoints:
            last_ckpt_path = output_dir / f"{args.model}{suffix}_last.pt"
            torch.save({
                'model_state_dict': _unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'epoch': epoch,
                'best_val_mrr': best_val_mrr,
                'patience_counter': patience_counter,
                'global_step': global_step,
                'smooth_loss': smooth_loss,
            }, last_ckpt_path)

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

    loaded_best_ckpt = False
    if args.save_checkpoints and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        loaded_best_ckpt = _load_model_state_with_prefix_fallback(model, ckpt['model_state_dict'], best_ckpt_path)
        if loaded_best_ckpt:
            print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")
    if not loaded_best_ckpt and ema is not None:
        ema.apply(model)
        print("Using EMA weights for final evaluation")
    elif not loaded_best_ckpt:
        print("No compatible best checkpoint available, using current model state")

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

    raw_model = _unwrap_model(model)
    is_cascade = isinstance(raw_model, CASCADEKGModel)
    is_mm = _is_multimodal(raw_model)
    is_mm_cascade = _is_multimodal_cascade(raw_model)

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
            cpu_eval = LinkPredictionEvaluator(dataset, device='cpu')
            result = cpu_eval.evaluate(wrapped, triples, weights, batch_size=batch_size, device='cpu', max_triples=max_triples, num_eval_negatives=num_eval_negatives, return_details=return_details)
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
    el    if args.model == 'cascade':
        ablation = getattr(args, 'ablation', None)
        return CASCADEKGModel(
            num_ents, num_rels, args.embed_dim,
            num_entity_types=len(ENTITY_TYPE_TO_ID),
            num_modalities=len(MODALITY_TO_ID),
            ablation=ablation,
            dropout=args.dropout,
        )
    elif args.model == 'multimodal_complex':
        modalities = MODALITY_SET_MAP.get(args.modalities, {"text"})
        return MultimodalComplExModel(
            num_ents, num_rels, args.embed_dim,
            num_modalities=len(MODALITY_TO_ID),
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
            num_entity_types=len(ENTITY_TYPE_TO_ID),
            num_modalities=len(MODALITY_TO_ID),
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
) -> dict:
    if args.device == 'cpu':
        return {
            'batch_size': getattr(args, 'batch_size', 64),
            'num_negatives': getattr(args, 'num_negatives', 8),
            'grad_accum_steps': getattr(args, 'grad_accum_steps', 4),
            'eval_batch_size': getattr(args, 'eval_batch_size', 64),
        }

    raw_model = _unwrap_model(model)
    is_cascade = isinstance(raw_model, CASCADEKGModel)
    is_mm = _is_multimodal(raw_model)
    is_mm_cascade = _is_multimodal_cascade(raw_model)
    is_complex = isinstance(raw_model, ComplExModel) or is_mm or is_mm_cascade

    nvidia_total_vram, nvidia_used_vram, _ = _nvidia_smi_memory_gb(args.device)
    torch_total_vram = torch.cuda.get_device_properties(args.device).total_memory / 1e9
    total_vram = nvidia_total_vram if nvidia_total_vram is not None else torch_total_vram

    model_vram_gb = torch.cuda.memory_allocated(args.device) / 1e9
    model_params_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    optimizer_gb = model_params_gb * 2
    grad_gb = model_params_gb
    ema_gb = model_params_gb
    num_ents = dataset.num_entities
    neg_sampler_static_kb = (num_ents * 8 * 3) / 1024
    cuda_ctx_gb = max(0.4, model_vram_gb - model_params_gb)
    static_gb = cuda_ctx_gb + model_params_gb + optimizer_gb + grad_gb + ema_gb + neg_sampler_static_kb / 1024 / 1024

    safety_gb = 0.3
    budget_gb = total_vram - safety_gb
    available_gb = budget_gb - static_gb

    embed_dim = getattr(args, 'embed_dim', 256)
    emb_per_ent = embed_dim * (2 if is_complex else 1) * 4
    per_sample_bytes = 3 * 8
    per_neg_bytes = emb_per_ent * 2
    per_sample_total = per_sample_bytes + per_neg_bytes * (1 + args.num_negatives)
    per_sample_total *= 3.5
    per_sample_kb = per_sample_total / 1024

    available_kb = available_gb * 1024 * 1024
    max_bs = max(16, int(available_kb / per_sample_kb)) if per_sample_kb > 0 else 32

    per_neg_kb = per_neg_bytes * 3.5 / 1024
    if available_gb > 0.5:
        max_neg = min(64, int((available_kb * 0.4) / (per_neg_kb * max_bs)))
        max_neg = max(4, max_neg)
        bs_budget_kb = available_kb - max_neg * per_neg_kb * max_bs
        best_bs = max(16, min(int(bs_budget_kb / per_sample_kb), max_bs, 512))
        best_neg = max_neg
    else:
        best_bs = 16
        best_neg = 4

    target_eff = getattr(args, 'batch_size', 64) * getattr(args, 'grad_accum_steps', 4)
    grad_accum = max(2, (target_eff + best_bs - 1) // best_bs)
    effective_bs = best_bs * grad_accum
    eval_bs = min(best_bs, 64)

    print(f"  [AUTO-TUNE] VRAM {total_vram:.1f}GB | static overhead {static_gb:.2f}GB (ctx={cuda_ctx_gb:.2f} model={model_params_gb:.2f} opt={optimizer_gb:.2f} grad={grad_gb:.2f} ema={ema_gb:.2f} neg_samp={neg_sampler_static_kb/1024/1024:.2f})")
    print(f"  [AUTO-TUNE] Available for batch: {available_gb:.2f}GB | per-sample: {per_sample_kb:.0f}KB | max_bs: {max_bs}")
    print(f"  [AUTO-TUNE] bs {getattr(args, 'batch_size', 64)}->{best_bs} | neg {getattr(args, 'num_negatives', 8)}->{best_neg} | ga {getattr(args, 'grad_accum_steps', 4)}->{grad_accum} | eff {effective_bs} | eval_bs {getattr(args, 'eval_batch_size', 64)}->{eval_bs}")
    print(f"  [AUTO-TUNE] Will adapt during training if shared GPU memory detected")

    config = {
        'batch_size': best_bs,
        'num_negatives': best_neg,
        'grad_accum_steps': grad_accum,
        'eval_batch_size': eval_bs,
    }
    return config


def _calibrate_vram(
    model: nn.Module,
    dataset: KGTriplesDataset,
    neg_sampler: 'NegativeSampler',
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Binary-search calibration: find largest batch size that fits within target VRAM, then optimize neg/ga."""
    raw_model = _unwrap_model(model)
    is_cascade = isinstance(raw_model, CASCADEKGModel)
    is_mm = _is_multimodal(raw_model)
    is_mm_cascade = _is_multimodal_cascade(raw_model)

    nvidia_total_vram, _, _ = _nvidia_smi_memory_gb(args.device)
    total_vram = nvidia_total_vram if nvidia_total_vram is not None else torch.cuda.get_device_properties(args.device).total_memory / 1e9
    safety_gb = 0.25
    target_gb = total_vram - safety_gb

    train_triples, _ = dataset.get_train_triples()

    if is_cascade or is_mm_cascade:
        entity_type_ids = dataset.get_entity_type_ids().to(device)
        entity_modality_ids = dataset.get_entity_modality_ids().to(device)

    device_type = 'cuda'

    def _probe_step(bs, num_neg_probed):
        idx = torch.randperm(train_triples.size(0), device='cpu')[:bs]
        batch = train_triples[idx].to(device)
        neg_triples, _ = neg_sampler.sample(batch)
        actual_neg = neg_triples.size(0) // batch.size(0)
        if actual_neg == 0:
            actual_neg = 1
            neg_triples = batch.clone()

        pos_h, pos_r, pos_t = batch[:, 0], batch[:, 1], batch[:, 2]
        neg_h, neg_r, neg_t = neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]

        optimizer.zero_grad(set_to_none=True)
        _safe_empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        with torch.amp.autocast(device_type=device_type, enabled=True):
            if is_mm_cascade:
                pos_s = model.score(pos_h, pos_r, pos_t, entity_type_ids, entity_modality_ids)
                neg_s = model.score(neg_h, neg_r, neg_t, entity_type_ids, entity_modality_ids)
            elif is_mm:
                mod_ids = dataset.get_entity_modality_ids().to(device)
                pos_s = model.score(pos_h, pos_r, pos_t, mod_ids)
                neg_s = model.score(neg_h, neg_r, neg_t, mod_ids)
                del mod_ids
            elif is_cascade:
                pos_s = model.score(pos_h, pos_r, pos_t, entity_type_ids, entity_modality_ids)
                neg_s = model.score(neg_h, neg_r, neg_t, entity_type_ids, entity_modality_ids)
            else:
                pos_s = model.score(pos_h, pos_r, pos_t)
                neg_s = model.score(neg_h, neg_r, neg_t)
            neg_s = neg_s.view(batch.size(0), actual_neg)
            pos_s = pos_s.unsqueeze(1)
            loss = loss_fn(pos_s, neg_s)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        _, nvidia_used, nvidia_shared = _nvidia_smi_memory_gb(args.device)

        del pos_s, neg_s, loss, batch, neg_triples, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t
        optimizer.zero_grad(set_to_none=True)
        _safe_empty_cache()

        nvidia_peak = nvidia_used if nvidia_used is not None else peak_gb
        nvidia_shared_val = nvidia_shared if nvidia_shared is not None else 0.0
        return nvidia_peak, nvidia_shared_val, actual_neg

    neg_for_probe = args.num_negatives
    neg_actual = neg_for_probe

    peak_small, shared_small, neg_actual = _probe_step(64, neg_for_probe)
    print(f"  [CALIBRATE] warmup probe: bs=64 neg={neg_actual} peak={peak_small:.2f}GB shared={shared_small:.2f}GB")

    if shared_small > 0.3 or peak_small > target_gb:
        print(f"  [CALIBRATE] Already at/over budget at bs=64, reducing")
        new_neg = max(4, neg_actual // 2)
        new_bs = max(16, 32)
        new_ga = args.grad_accum_steps
        return {'batch_size': new_bs, 'num_negatives': new_neg, 'grad_accum_steps': new_ga, 'eval_batch_size': min(new_bs, 64)}

    lo_bs = 64
    # Cap the probe ceiling near the auto-tuned batch size. Probing all the way
    # to 2048 on a 6 GB card creates huge transient + "oversize" allocator
    # segments that fragment memory and inflate driver-visible VRAM long after
    # the probe tensors are freed -- a primary cause of the shared-memory spill.
    hi_bs = min(1024, max(lo_bs + 1, getattr(args, 'batch_size', 256) * 2))
    best_bs = lo_bs
    best_peak = peak_small
    max_iters = 6

    for i in range(max_iters):
        mid_bs = (lo_bs + hi_bs) // 2
        if mid_bs <= lo_bs:
            break
        try:
            peak_mid, shared_mid, _ = _probe_step(mid_bs, neg_for_probe)
        except RuntimeError:
            peak_mid = target_gb + 1
            shared_mid = 0

        print(f"  [CALIBRATE] bsearch iter {i+1}: bs={mid_bs} neg={neg_actual} peak={peak_mid:.2f}GB shared={shared_mid:.2f}GB (target={target_gb:.2f}GB)")

        if shared_mid > 0.5 or peak_mid > target_gb:
            hi_bs = mid_bs
        else:
            best_bs = mid_bs
            best_peak = peak_mid
            lo_bs = mid_bs

    headroom_gb = target_gb - best_peak
    print(f"  [CALIBRATE] bsearch result: bs={best_bs} peak={best_peak:.2f}GB headroom={headroom_gb:.2f}GB")

    if headroom_gb > 0.5 and neg_actual < 128:
        extra_neg_budget = headroom_gb * 0.4
        neg_per_gb = 0.0
        if best_bs >= 128:
            try:
                peak_neg_probe, _, _ = _probe_step(best_bs, neg_actual + 8)
            except RuntimeError:
                peak_neg_probe = target_gb + 1
            neg_per_gb = max((peak_neg_probe - best_peak) / 8, 0.001)
        extra_neg = int(extra_neg_budget / (neg_per_gb * best_bs)) if neg_per_gb > 0 else 0
        new_neg = min(neg_actual + extra_neg, 128)
        new_neg = max(4, new_neg)
    else:
        new_neg = neg_actual

    new_bs = best_bs
    target_eff = args.batch_size * args.grad_accum_steps
    new_ga = max(2, (target_eff + new_bs - 1) // new_bs)

    print(f"  [CALIBRATE] Final: bs={new_bs} neg={new_neg} ga={new_ga} eff={new_bs*new_ga} target_vram={target_gb:.2f}GB")
    return {'batch_size': new_bs, 'num_negatives': new_neg, 'grad_accum_steps': new_ga, 'eval_batch_size': min(new_bs, 64)}


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
        model = model.to(args.device)
        mem_trace(f"after model.to({args.device})", args.device)
        tuned = auto_tune_config(model, dataset, args)
        if tuned:
            for k, v in tuned.items():
                setattr(args, k, v)
        mem_trace("after auto_tune", args.device)

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
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability for embeddings (0=disabled)')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
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
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader workers (0=main process, safe on Windows)')
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--max-eval-triples', type=int, default=10000,
                        help='Max triples to evaluate (subsampled)')
    parser.add_argument('--eval-negatives', type=int, default=100,
                        help='Number of type-constrained negative samples for evaluation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=1)
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
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience on val MRR')
    parser.add_argument('--no-ema', action='store_true', default=False,
                        help='Disable Exponential Moving Average of model parameters')
    parser.add_argument('--min-system-ram-gb', type=float, default=0.5,
                        help='Minimum free system RAM in GB before aborting (0 disables)')
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
    parser.add_argument('--skip-calibrate', action='store_true', default=False,
                        help='Skip VRAM calibration probes and auto-tune; use explicit --batch-size/--num-negatives/--grad-accum-steps as-is')
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
    parser.add_argument('--max-vram-gb', type=float, default=5.0,
                        help='Strict CUDA memory budget in GB for auto-tune/calibration/runtime guards (0 disables). '
                             'Lowered from 5.8 -> 5.0: the old value targeted torch "allocated" bytes and ignored the '
                             'CUDA context + fragmentation the driver counts, so the card overshot 6 GB and spilled.')
    parser.add_argument('--cuda-context-reserve-gb', type=float, default=1.0,
                        help='VRAM (GB) held back for the CUDA context + cuDNN/cuBLAS workspaces + allocator '
                             'fragmentation that the Windows driver counts but torch.cuda.memory_allocated() does not. '
                             'The allocator is hard-capped at min(max_vram_gb, total - this). Raise to 1.3-1.5 if you '
                             'still see ANY shared GPU memory in Task Manager.')
    parser.add_argument('--shared-gpu-limit-gb', type=float, default=0.1,
                        help='Abort/retry if NVIDIA-reported shared GPU memory exceeds this many GB (-1 disables)')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        args.device = 'cpu'

    if args.device == 'cuda':
        # Set the hard allocator ceiling once up front; it persists for the
        # whole process and is respected by calibration and every retry.
        _enforce_vram_ceiling(args, torch.device(args.device))

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

    base_output_dir = Path(args.output_dir)
    subdir = 'testkg' if args.test_kg else 'real'
    if base_output_dir.name != subdir:
        base_output_dir = base_output_dir / subdir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(base_output_dir)

    evaluator = LinkPredictionEvaluator(dataset, device=args.device)

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
        _skip = getattr(args, 'skip_calibrate', False)
        max_vram_retries = 5

        for vram_attempt in range(1 if _skip else (max_vram_retries + 1)):
            try:
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

                model = model.to(args.device)
                mem_trace(f"after model.to({args.device})", args.device)

                if getattr(args, 'skip_calibrate', False):
                    print(f"  [AUTO-TUNE] Skipped by --skip-calibrate; using explicit batch_size={args.batch_size} neg={args.num_negatives} ga={args.grad_accum_steps}")
                else:
                    tuned = auto_tune_config(model, dataset, args)
                    if tuned:
                        for k, v in tuned.items():
                            setattr(args, k, v)
                mem_trace("after auto_tune", args.device)

                active_mods = MODALITY_SET_MAP.get(modalities, {"text"})
                print(f"Training {display_name} on {args.device}...")
                results = train_model(model, dataset, evaluator, args, active_modalities=active_mods)
                break
            except VRAMExceeded as ve:
                if _skip:
                    raise
                if vram_attempt >= max_vram_retries:
                    print(f"\n[VRAM] Max retries ({max_vram_retries}) exhausted for {display_name}")
                    if args.benchmark:
                        all_results[display_name] = {
                            "model": display_name,
                            "ablation": ablation,
                            "modalities": modalities,
                            "error": str(ve),
                        }
                        break
                    raise
                old_bs, old_neg = args.batch_size, args.num_negatives
                if args.num_negatives > 4:
                    args.num_negatives = max(4, args.num_negatives // 2)
                    print(f"\n[VRAM] Retry {vram_attempt+1}: neg {old_neg} → {args.num_negatives} (shared was {ve.shared_gb:.2f}GB)")
                elif args.batch_size > 16:
                    args.batch_size = max(16, args.batch_size // 2)
                    print(f"\n[VRAM] Retry {vram_attempt+1}: bs {old_bs} → {args.batch_size} (shared was {ve.shared_gb:.2f}GB)")
                elif args.grad_accum > 1:
                    args.grad_accum = max(1, args.grad_accum // 2)
                    print(f"\n[VRAM] Retry {vram_attempt+1}: ga unchanged (already min bs/neg), ga {args.grad_accum} (shared was {ve.shared_gb:.2f}GB)")
                else:
                    print(f"\n[VRAM] Cannot reduce further — bs={args.batch_size} neg={args.num_negatives} ga={args.grad_accum}")
                    if args.benchmark:
                        all_results[display_name] = {
                            "model": display_name,
                            "ablation": ablation,
                            "modalities": modalities,
                            "error": str(ve),
                        }
                        break
                    raise
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                set_seed(args.seed)
                model = build_model(args, dataset).to(args.device)
                print(f"[VRAM] Rebuilt model with bs={args.batch_size} neg={args.num_negatives} ga={args.grad_accum}")
            except RuntimeError as re_err:
                if 'out of memory' in str(re_err).lower():
                    if vram_attempt >= max_vram_retries:
                        print(f"\n[VRAM] Max retries ({max_vram_retries}) exhausted (CUDA OOM)")
                        if args.benchmark:
                            all_results[display_name] = {
                                "model": display_name,
                                "ablation": ablation,
                                "modalities": modalities,
                                "error": str(re_err),
                            }
                            break
                        raise
                    old_bs, old_neg = args.batch_size, args.num_negatives
                    if args.num_negatives > 4:
                        args.num_negatives = max(4, args.num_negatives // 2)
                    elif args.batch_size > 16:
                        args.batch_size = max(16, args.batch_size // 2)
                    else:
                        print(f"\n[VRAM] CUDA OOM and cannot reduce further")
                        if args.benchmark:
                            all_results[display_name] = {
                                "model": display_name,
                                "ablation": ablation,
                                "modalities": modalities,
                                "error": str(re_err),
                            }
                            break
                        raise
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                    set_seed(args.seed)
                    model = build_model(args, dataset).to(args.device)
                    print(f"\n[VRAM] Retry {vram_attempt+1} (OOM): bs {old_bs} → {args.batch_size}, neg {old_neg} → {args.num_negatives}")
                else:
                    print(f"\nModel {display_name} failed: {re_err}")
                    if args.benchmark:
                        all_results[display_name] = {
                            "model": display_name,
                            "ablation": ablation,
                            "modalities": modalities,
                            "error": str(re_err),
                        }
                        break
                    raise
        else:
            if args.benchmark and display_name in all_results:
                continue

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
