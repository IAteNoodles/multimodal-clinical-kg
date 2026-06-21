from __future__ import annotations
import argparse, torch, gc, json, os, sys, shutil, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:256"

import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from simulation.kg.dataset import (
    ENTITY_TYPE_TO_ID, MODALITY_TO_ID, KGTriplesDataset, NegativeSampler,
    LinkPredictionEvaluator, MODALITY_SET_MAP, CROSS_MODAL_RELATIONS,
)
from simulation.kg.models import (
    MultimodalCASCADEModel, TransEModel, ComplExModel,
)
from simulation.kg.inference import load_entity_features


def build_model(args, dataset):
    num_ents, num_rels = dataset.num_entities, dataset.num_relations
    mods = MODALITY_SET_MAP.get(args.modalities, {"text"})
    if args.model == 'transE':
        return TransEModel(num_ents, num_rels, args.embed_dim, dropout=args.dropout)
    if args.model == 'complex':
        return ComplExModel(num_ents, num_rels, args.embed_dim, dropout=args.dropout)
    if args.model == 'multimodal_cascade':
        cross_rel_ids = {dataset.relation2id[r] for r in CROSS_MODAL_RELATIONS if r in dataset.relation2id}
        model = MultimodalCASCADEModel(
            num_ents, num_rels, args.embed_dim,
            num_entity_types=len(ENTITY_TYPE_TO_ID),
            num_modalities=len(MODALITY_TO_ID), synergy_dim=64, num_heads=4,
            dropout=args.dropout, use_modality_encoders=False,
            use_pretrained_encoders=False, modalities=mods, ablation=args.ablation,
            cross_modal_relations=cross_rel_ids,
        )
        if not args.feature_dir:
            model.precompute_features = False
        return model
    raise ValueError(f"Unsupported model: {args.model}")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, label_smoothing=0.0):
        super().__init__()
        self.t = temperature
        self.ls = label_smoothing

    def forward(self, pos, neg):
        logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1) / self.t
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, target, label_smoothing=self.ls)


def save_checkpoint(path, model, opt, sched, scaler, global_step, ep, metrics=None, skip_full=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not skip_full:
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'global_step': global_step,
            'ep': ep,
            'metrics': metrics or {},
            'rng_python': random.getstate(),
            'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
    # Lightweight model-only and meta files for OOM-safe resume
    if "best" in path.stem:
        torch.save(model.state_dict(), path.with_name("best_model.pt"))
    with open(path.with_name("meta.json"), 'w') as f:
        json.dump({"ep": ep, "global_step": global_step, "metrics": metrics or {}}, f)


def load_checkpoint(path, model, opt, sched, scaler, device):
    model_path = path.with_name("model.pt")
    meta_path = path.with_name("meta.json")
    if model_path.exists():
        sd = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        del sd
    elif path.exists():
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = {k: v.to(device) for k, v in ckpt['model'].items()}
        model.load_state_dict(sd)
        if 'opt' in ckpt:
            opt.load_state_dict(ckpt['opt'])
        if 'sched' in ckpt:
            sched.load_state_dict(ckpt['sched'])
        if scaler and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        if 'rng_python' in ckpt:
            random.setstate(ckpt['rng_python'])
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        if 'rng_torch' in ckpt:
            torch.set_rng_state(ckpt['rng_torch'])
        if 'rng_cuda' in ckpt and ckpt['rng_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
        del ckpt, sd
    gc.collect()
    torch.cuda.empty_cache()
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta["ep"], meta["global_step"], meta.get("metrics", {})
    return 0, 0, {}


def load_best_weights(model, ckpt_dir, device):
    for name in ["best_model.pt", "model.pt"]:
        p = ckpt_dir / name
        if p.exists():
            sd = torch.load(p, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            del sd
            gc.collect()
            torch.cuda.empty_cache()
            return True
    return False


def _needs_modality_ids(model):
    return not isinstance(model, (TransEModel, ComplExModel))


def run_test_eval(model, evaluator, dataset, args, device, entity_type_ids=None, entity_modality_ids=None):
    print(f"\n  running test evaluation...")
    if _needs_modality_ids(model):
        if entity_type_ids is None:
            entity_type_ids = dataset.get_entity_type_ids().to(device)
        if entity_modality_ids is None:
            entity_modality_ids = dataset.get_entity_modality_ids().to(device)
    test_t, test_w = dataset.get_test_triples()
    test_t, test_w = test_t.to(device), test_w.to(device)
    test_metrics = evaluator.evaluate(
        model, test_t, test_w,
        batch_size=min(args.eval_batch_size, 4096),
        max_triples=args.max_eval_triples,
        num_eval_negatives=args.num_eval_negatives,
        full_rank=True,
        entity_type_ids=entity_type_ids if _needs_modality_ids(model) else None,
        entity_modality_ids=entity_modality_ids if _needs_modality_ids(model) else None,
    )
    print(f"  test MRR={test_metrics.get('MRR',0):.4f} H@1={test_metrics.get('Hits@1',0):.4f} H@10={test_metrics.get('Hits@10',0):.4f}")
    return test_metrics


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"device={device}")

    dataset = KGTriplesDataset.from_efficient(Path(args.data_dir), seed=args.seed)
    print(f"entities={dataset.num_entities} relations={dataset.num_relations}")

    train_triples, _ = dataset.get_train_triples()
    train_dataset = TensorDataset(train_triples.clone())
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    model = build_model(args, dataset).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    if args.feature_dir and isinstance(model, MultimodalCASCADEModel):
        entity_features = load_entity_features(args.feature_dir, dataset)
        model.set_precomputed_features(entity_features)
        print(f"  loaded precomputed features from {args.feature_dir}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    neg_sampler = NegativeSampler(dataset, num_negatives=args.num_negatives, device=device)
    loss_fn = InfoNCELoss(temperature=args.temperature, label_smoothing=args.label_smoothing).to(device)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    if _needs_modality_ids(model):
        entity_type_ids = dataset.get_entity_type_ids().to(device)
        entity_modality_ids = dataset.get_entity_modality_ids().to(device)
    else:
        entity_type_ids = entity_modality_ids = None
    n_steps_total = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = args.warmup_steps
    sched = CosineAnnealingLR(opt, T_max=n_steps_total)

    evaluator = None  # lazy init on first eval to save GPU memory during training

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    start_ep, global_step = 0, 0
    best_metric = -float('inf')
    best_ep = 0
    patience_counter = 0

    if ckpt_dir and args.resume:
        latest_path = ckpt_dir / "latest.pt"
        if not latest_path.exists():
            ckpt_files = sorted(ckpt_dir.glob("ep_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
            if ckpt_files:
                latest_path = ckpt_files[-1]
        if not latest_path.exists():
            latest_path = ckpt_dir / "best.pt"
        if latest_path.exists():
            start_ep, global_step, metrics = load_checkpoint(latest_path, model, opt, sched, scaler, device)
            best_metric = metrics.get('best_metric', -float('inf'))
            best_ep = metrics.get('best_ep', 0)
            patience_counter = metrics.get('patience_counter', 0)
            print(f"  resumed from {latest_path} (ep={start_ep}, step={global_step})")
    if args.eval_only:
        if ckpt_dir and load_best_weights(model, ckpt_dir, device):
            evaluator = LinkPredictionEvaluator(dataset, device=device)
            test_metrics = run_test_eval(model, evaluator, dataset, args, device)
            if test_metrics:
                with open(ckpt_dir / "test_results.json", 'w') as f:
                    json.dump(test_metrics, f)
        return

    if start_ep >= args.epochs and not args.eval_only:
        print("  training already complete")
        return

    from tqdm import tqdm
    done = False
    for ep in range(start_ep + 1, args.epochs + 1):
        if done:
            break
        print(f"  ep{ep}/{args.epochs} starting ({len(train_loader)} batches)", flush=True)
        model.train()
        ep_loss_sum = 0.0
        ep_loss_count = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"ep{ep}")
        for bi, (batch,) in pbar:
            batch = batch.to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            neg_all, _ = neg_sampler.sample(batch)
            n_pos = h.size(0)
            K = args.num_negatives
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                if _needs_modality_ids(model):
                    pos = model.score(h, r, t, entity_type_ids, entity_modality_ids)
                    neg = model.score(neg_all[:, 0], neg_all[:, 1], neg_all[:, 2],
                        entity_type_ids, entity_modality_ids).reshape(2, n_pos, K).permute(1, 0, 2).reshape(n_pos, -1)
                else:
                    pos = model.score(h, r, t)
                    neg = model.score(neg_all[:, 0], neg_all[:, 1], neg_all[:, 2]).reshape(2, n_pos, K).permute(1, 0, 2).reshape(n_pos, -1)
                loss = loss_fn(pos, neg)
                if args.n3_weight > 0:
                    n3_pos = model.n3_penalty(h, r, t)
                    n3_neg = model.n3_penalty(neg_all[:, 0], neg_all[:, 1], neg_all[:, 2])
                    loss = loss + args.n3_weight * (n3_pos + n3_neg) / 2

            if args.grad_accum_steps > 0:
                loss = loss / args.grad_accum_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (bi + 1) % args.grad_accum_steps == 0 or bi == len(train_loader) - 1:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if args.lr_warmup and global_step < warmup_steps:
                    for g in opt.param_groups:
                        g['lr'] = args.lr * min(1.0, (global_step + 1) / warmup_steps)

                if args.clamp_norm > 0 and hasattr(model, 'clamp_embed_norm'):
                    model.clamp_embed_norm(max_norm=args.clamp_norm)

            if bi == 0 and ep == 1:
                alloc = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                print(f"\n  first step: alloc={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)
            if bi == 0:
                print(f"  bi0 loss={loss.item():.4f}", flush=True)

            ep_loss_sum += loss.item()
            ep_loss_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0
        ep_metrics = {"train_loss": train_loss}

        if args.eval and ep % args.eval_every_epochs == 0:
            if evaluator is None:
                evaluator = LinkPredictionEvaluator(dataset, device=device)
            val_t, val_w = dataset.get_val_triples()
            val_t, val_w = val_t.to(device), val_w.to(device)
            val_metrics = evaluator.evaluate(
                model, val_t, val_w,
                batch_size=min(args.eval_batch_size, 4096),
                max_triples=args.max_eval_triples,
                num_eval_negatives=args.num_eval_negatives,
                full_rank=True,
                entity_type_ids=entity_type_ids if _needs_modality_ids(model) else None,
                entity_modality_ids=entity_modality_ids if _needs_modality_ids(model) else None,
            )
            ep_metrics.update(val_metrics)
            mrr = val_metrics.get("MRR", 0.0)
            print(f"  val MRR={mrr:.4f} H@1={val_metrics.get('Hits@1',0):.4f} H@10={val_metrics.get('Hits@10',0):.4f}", flush=True)

            if mrr > best_metric:
                best_metric = mrr
                best_ep = ep
                patience_counter = 0
                if ckpt_dir:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(ckpt_dir / "best.pt", model, opt, sched, scaler, global_step, ep, ep_metrics)
                    print(f"  new best: {best_metric:.4f}")
            else:
                patience_counter += 1
                print(f"  no improv ({patience_counter}/{args.patience}) best={best_metric:.4f}@ep{best_ep}")
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"  early stopping at ep{ep}")
                    done = True
            # Free evaluator GPU buffers immediately
            evaluator = None
            gc.collect()
            torch.cuda.empty_cache()
            if done:
                break
        else:
            ep_metrics["MRR"] = best_metric

        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ep_metrics['best_metric'] = best_metric
            ep_metrics['best_ep'] = best_ep
            ep_metrics['patience_counter'] = patience_counter
            save_checkpoint(ckpt_dir / "latest.pt", model, opt, sched, scaler, global_step, ep, ep_metrics)

            if args.keep_last_n != 0:
                ep_path = ckpt_dir / f"ep_{ep}.pt"
                save_checkpoint(ep_path, model, opt, sched, scaler, global_step, ep, ep_metrics)
                if args.keep_last_n > 0:
                    old_eps = sorted(ckpt_dir.glob("ep_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
                    while len(old_eps) > args.keep_last_n:
                        old_eps[0].unlink()
                        old_eps.pop(0)

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  ep{ep} done, loss={train_loss:.4f}", flush=True)

    if args.eval:
        if evaluator is None:
            evaluator = LinkPredictionEvaluator(dataset, device=device)
        print(f"\n  best val MRR={best_metric:.4f} at ep{best_ep}")
        if ckpt_dir and load_best_weights(model, ckpt_dir, device):
            test_metrics = run_test_eval(model, evaluator, dataset, args, device, entity_type_ids, entity_modality_ids)
            with open(ckpt_dir / "test_results.json", 'w') as f:
                json.dump(test_metrics, f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='multimodal_cascade',
                   choices=['transE', 'complex', 'multimodal_cascade'])
    p.add_argument('--ablation', default=None,
                   help='ablation mode: no_pid, no_type, no_modality, all')
    p.add_argument('--data-dir', default='simulation/data/kg/clinical_kg_efficient')
    p.add_argument('--feature-dir', default=None, help='dir with precomputed entity features')
    p.add_argument('--modalities', default='text')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-negatives', type=int, default=4)
    p.add_argument('--grad-accum-steps', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--embed-dim', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--label-smoothing', type=float, default=0.05)
    p.add_argument('--n3-weight', type=float, default=0.01)
    p.add_argument('--clamp-norm', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--lr-warmup', action='store_true', default=True)
    p.add_argument('--checkpoint-dir', default=None, help='save checkpoints to dir')
    p.add_argument('--resume', action='store_true', help='resume from latest.pt')
    p.add_argument('--keep-last-n', type=int, default=3, help='keep N epoch checkpoints (0=all, -1=none)')
    p.add_argument('--eval', action='store_true', help='run val eval after each epoch')
    p.add_argument('--eval-only', action='store_true', help='load best model and run test eval only')
    p.add_argument('--eval-every-epochs', type=int, default=1)
    p.add_argument('--patience', type=int, default=5, help='early stop after N val MRR drops')
    p.add_argument('--eval-batch-size', type=int, default=1024, help='batch size during eval')
    p.add_argument('--max-eval-triples', type=int, default=None, help='cap val triples for speed')
    p.add_argument('--num-eval-negatives', type=int, default=50)
    args = p.parse_args()
    train(args)
