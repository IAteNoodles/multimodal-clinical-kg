from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from simulation.kg.dataset import (
    KGTriplesDataset,
    MultimodalKGTriplesDataset,
    LinkPredictionEvaluator,
    ENTITY_TYPE_TO_ID,
    MODALITY_TO_ID,
    MODALITY_SET_MAP,
    FEATURE_KEY_MAP,
)
from simulation.kg.models import (
    TransEModel,
    ComplExModel,
    CASCADEKGModel,
    MultimodalComplExModel,
    MultimodalCASCADEModel,
)


class CASCADEWrapper(nn.Module):
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


def detect_model_type(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    raw_keys = set(k.replace("_orig_mod.", "") for k in keys)

    if "has_modality_logit" in raw_keys:
        return "multimodal_complex"
    if any("cross_modal_attn" in k for k in raw_keys):
        return "multimodal_cascade"
    if "pid_synergy_raw" in raw_keys:
        return "cascade"
    if "modality_embeddings.weight" in raw_keys:
        return "cascade"

    ent_shape = state_dict.get("entity_embeddings.weight").shape
    rel_shape = state_dict.get("relation_embeddings.weight").shape
    if ent_shape[1] == 2 * rel_shape[1]:
        return "complex"
    return "transe"


def detect_embed_dim(state_dict: Dict[str, torch.Tensor], model_type: str) -> int:
    ent_dim = state_dict["entity_embeddings.weight"].shape[1]
    if model_type in ("cascade", "complex"):
        embed_dim = ent_dim // 2
        if "modality_embeddings.weight" in state_dict:
            embed_dim = state_dict["modality_embeddings.weight"].shape[1]
        return embed_dim
    return ent_dim


def strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
) -> Tuple[nn.Module, str, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    state_dict = strip_orig_mod_prefix(raw_sd)
    model_type = detect_model_type(state_dict)
    embed_dim = detect_embed_dim(state_dict, model_type)

    num_ents = state_dict["entity_embeddings.weight"].shape[0]
    num_rels = state_dict["relation_embeddings.weight"].shape[0]

    if model_type == "transe":
        model = TransEModel(num_ents, num_rels, embed_dim)
    elif model_type == "complex":
        model = ComplExModel(num_ents, num_rels, embed_dim)
    elif model_type == "cascade":
        model = CASCADEKGModel(
            num_ents, num_rels, embed_dim,
            num_entity_types=5,
            num_modalities=4,
        )
    elif model_type == "multimodal_complex":
        model = MultimodalComplExModel(
            num_ents, num_rels, embed_dim,
            num_modalities=4,
            synergy_dim=64,
            num_heads=4,
        )
    elif model_type == "multimodal_cascade":
        model = MultimodalCASCADEModel(
            num_ents, num_rels, embed_dim,
            num_entity_types=5,
            num_modalities=4,
            synergy_dim=64,
            num_heads=4,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, model_type, ckpt


def load_entity_features(
    feature_dir: str,
    dataset: KGTriplesDataset,
) -> Dict[int, Dict[str, FloatTensor]]:
    feature_path = Path(feature_dir)
    entity_features: Dict[int, Dict[str, FloatTensor]] = {}

    for mod_name in ("text", "image", "ecg", "structured"):
        file_stem = FEATURE_KEY_MAP[mod_name]
        feat_file = feature_path / f"{file_stem}_features.npy"
        id_file = feature_path / f"{file_stem}_feature_ids.csv"

        if not feat_file.exists() or not id_file.exists():
            continue

        features_np = np.load(feat_file)
        features_tensor = torch.from_numpy(features_np).float()

        with open(id_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row_idx, row in enumerate(reader):
                if len(row) >= 1 and row_idx < features_tensor.shape[0]:
                    eid = int(row[0])
                    feat = features_tensor[row_idx]
                    if eid not in entity_features:
                        entity_features[eid] = {}
                    entity_features[eid][mod_name] = feat

    return entity_features


def wrap_model(
    model: nn.Module,
    model_type: str,
    dataset: KGTriplesDataset,
    device: str,
) -> nn.Module:
    dev = torch.device(device)
    if model_type in ("cascade", "multimodal_cascade"):
        entity_type_ids = dataset.get_entity_type_ids().to(dev)
        entity_modality_ids = dataset.get_entity_modality_ids().to(dev)
        return CASCADEWrapper(model, entity_type_ids, entity_modality_ids)
    elif model_type in ("multimodal_complex",):
        entity_modality_ids = dataset.get_entity_modality_ids().to(dev)
        return MultimodalWrapper(model, entity_modality_ids)
    return model


def run_evaluate(args: argparse.Namespace) -> None:
    model, model_type, ckpt = build_model_from_checkpoint(args.checkpoint, args.device)

    is_multimodal = model_type in ("multimodal_complex", "multimodal_cascade")
    if is_multimodal:
        dataset = MultimodalKGTriplesDataset.from_efficient(
            Path(args.data_dir),
            seed=args.seed,
            feature_dir=args.feature_dir,
            feature_dim=args.feature_dim,
            modalities=args.modalities,
        )
        entity_features = load_entity_features(args.feature_dir, dataset)
        model.set_precomputed_features(entity_features)
        model.to(args.device)
        model.eval()
    else:
        dataset = KGTriplesDataset.from_efficient(Path(args.data_dir), seed=args.seed)

    wrapped = wrap_model(model, model_type, dataset, args.device)
    evaluator = LinkPredictionEvaluator(dataset, device=args.device)

    result = evaluator.evaluate(
        wrapped,
        *dataset.get_test_triples(),
        batch_size=args.batch_size,
        device=args.device,
        num_eval_negatives=args.num_negatives,
    )

    if args.test_kg:
        data_label = "test_kg"
    else:
        data_label = "clinical_kg"

    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({model_type} on {data_label})")
    print(f"{'='*60}")
    for metric, value in result.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print(f"{'='*60}")


def resolve_entity(entity_arg: str, dataset: KGTriplesDataset) -> int:
    try:
        return int(entity_arg)
    except ValueError:
        if entity_arg in dataset.entity2id:
            return dataset.entity2id[entity_arg]
        raise ValueError(f"Entity '{entity_arg}' not found in dataset")


def resolve_relation(relation_arg: str, dataset: KGTriplesDataset) -> int:
    try:
        return int(relation_arg)
    except ValueError:
        if relation_arg in dataset.relation2id:
            return dataset.relation2id[relation_arg]
        raise ValueError(f"Relation '{relation_arg}' not found in dataset")


def run_predict(args: argparse.Namespace) -> None:
    model, model_type, ckpt = build_model_from_checkpoint(args.checkpoint, args.device)

    is_multimodal = model_type in ("multimodal_complex", "multimodal_cascade")
    if is_multimodal:
        dataset = MultimodalKGTriplesDataset.from_efficient(
            Path(args.data_dir),
            seed=args.seed,
            feature_dir=args.feature_dir,
            feature_dim=args.feature_dim,
            modalities=args.modalities,
        )
        entity_features = load_entity_features(args.feature_dir, dataset)
        model.set_precomputed_features(entity_features)
        model.to(args.device)
        model.eval()
    else:
        dataset = KGTriplesDataset.from_efficient(Path(args.data_dir), seed=args.seed)

    wrapped = wrap_model(model, model_type, dataset, args.device)
    num_entities = dataset.num_entities
    dev = torch.device(args.device)
    chunk_size = 4096

    top_k = args.top_k

    if args.head is not None and args.relation is not None:
        head_id = resolve_entity(args.head, dataset)
        rel_id = resolve_relation(args.relation, dataset)
        direction = "tail"

        head_t = torch.tensor([head_id], dtype=torch.long, device=dev)
        rel_t = torch.tensor([rel_id], dtype=torch.long, device=dev)

        all_scores = []
        all_ids = []
        for start in range(0, num_entities, chunk_size):
            end = min(start + chunk_size, num_entities)
            tail_ids = torch.arange(start, end, dtype=torch.long, device=dev)
            heads = head_t.expand(end - start)
            rels = rel_t.expand(end - start)
            with torch.no_grad():
                scores = wrapped.score(heads, rels, tail_ids)
            all_scores.append(scores.cpu())
            all_ids.append(tail_ids.cpu())

        all_scores = torch.cat(all_scores)
        all_ids = torch.cat(all_ids)
        topk_scores, topk_indices = all_scores.topk(min(top_k, num_entities))
        topk_entity_ids = all_ids[topk_indices]

        print(f"\nTop-{top_k} tail predictions for (head={args.head}, relation={args.relation}):")
        print(f"{'Rank':<6} {'Entity ID':<12} {'Entity Name':<40} {'Score':<12}")
        print("-" * 70)
        for rank, (eid, score) in enumerate(zip(topk_entity_ids.tolist(), topk_scores.tolist()), 1):
            name = dataset.id2entity.get(eid, str(eid))
            print(f"{rank:<6} {eid:<12} {name:<40} {score:<12.4f}")

    elif args.relation is not None and args.tail is not None:
        rel_id = resolve_relation(args.relation, dataset)
        tail_id = resolve_entity(args.tail, dataset)
        direction = "head"

        rel_t = torch.tensor([rel_id], dtype=torch.long, device=dev)
        tail_t = torch.tensor([tail_id], dtype=torch.long, device=dev)

        all_scores = []
        all_ids = []
        for start in range(0, num_entities, chunk_size):
            end = min(start + chunk_size, num_entities)
            head_ids = torch.arange(start, end, dtype=torch.long, device=dev)
            tails = tail_t.expand(end - start)
            rels = rel_t.expand(end - start)
            with torch.no_grad():
                scores = wrapped.score(head_ids, rels, tails)
            all_scores.append(scores.cpu())
            all_ids.append(head_ids.cpu())

        all_scores = torch.cat(all_scores)
        all_ids = torch.cat(all_ids)
        topk_scores, topk_indices = all_scores.topk(min(top_k, num_entities))
        topk_entity_ids = all_ids[topk_indices]

        print(f"\nTop-{top_k} head predictions for (relation={args.relation}, tail={args.tail}):")
        print(f"{'Rank':<6} {'Entity ID':<12} {'Entity Name':<40} {'Score':<12}")
        print("-" * 70)
        for rank, (eid, score) in enumerate(zip(topk_entity_ids.tolist(), topk_scores.tolist()), 1):
            name = dataset.id2entity.get(eid, str(eid))
            print(f"{rank:<6} {eid:<12} {name:<40} {score:<12.4f}")

    else:
        print("Error: provide --head and --relation, or --relation and --tail")
        return


def run_similar(args: argparse.Namespace) -> None:
    model, model_type, ckpt = build_model_from_checkpoint(args.checkpoint, args.device)

    is_multimodal = model_type in ("multimodal_complex", "multimodal_cascade")
    if is_multimodal:
        dataset = MultimodalKGTriplesDataset.from_efficient(
            Path(args.data_dir),
            seed=args.seed,
            feature_dir=args.feature_dir,
            feature_dim=args.feature_dim,
            modalities=args.modalities,
        )
    else:
        dataset = KGTriplesDataset.from_efficient(Path(args.data_dir), seed=args.seed)

    entity_id = resolve_entity(args.entity, dataset)
    dev = torch.device(args.device)

    with torch.no_grad():
        all_embs = model.entity_embeddings.weight.data
        query_emb = all_embs[entity_id].unsqueeze(0)
        query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
        all_norm = torch.nn.functional.normalize(all_embs, p=2, dim=-1)
        similarities = (query_norm @ all_norm.T).squeeze(0)

    top_k = args.top_k + 1
    topk_scores, topk_indices = similarities.topk(min(top_k, dataset.num_entities))

    print(f"\nTop-{args.top_k} most similar entities to '{args.entity}' (id={entity_id}):")
    print(f"{'Rank':<6} {'Entity ID':<12} {'Entity Name':<40} {'Cosine Sim':<12}")
    print("-" * 70)
    rank = 0
    for eid, score in zip(topk_indices.tolist(), topk_scores.tolist()):
        if eid == entity_id:
            continue
        rank += 1
        if rank > args.top_k:
            break
        name = dataset.id2entity.get(eid, str(eid))
        print(f"{rank:<6} {eid:<12} {name:<40} {score:<12.4f}")


def run_export(args: argparse.Namespace) -> None:
    model, model_type, ckpt = build_model_from_checkpoint(args.checkpoint, args.device)

    is_multimodal = model_type in ("multimodal_complex", "multimodal_cascade")
    if is_multimodal:
        dataset = MultimodalKGTriplesDataset.from_efficient(
            Path(args.data_dir),
            seed=args.seed,
            feature_dir=args.feature_dir,
            feature_dim=args.feature_dim,
            modalities=args.modalities,
        )
    else:
        dataset = KGTriplesDataset.from_efficient(Path(args.data_dir), seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        entity_embs = model.entity_embeddings.weight.data.cpu().numpy()
        relation_embs = model.relation_embeddings.weight.data.cpu().numpy()

    entity_path = output_dir / "entity_embeddings.npy"
    relation_path = output_dir / "relation_embeddings.npy"
    np.save(entity_path, entity_embs)
    np.save(relation_path, relation_embs)

    metadata = {}
    metadata_path = Path(args.data_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        metadata["entity_id_map"] = meta.get("entity_id_map", {})
    else:
        metadata["entity_id_map"] = dataset.entity2id

    id_map_path = output_dir / "entity_id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(metadata["entity_id_map"], f, indent=2)

    print(f"Exported entity embeddings: {entity_path} (shape={entity_embs.shape})")
    print(f"Exported relation embeddings: {relation_path} (shape={relation_embs.shape})")
    print(f"Exported entity ID map: {id_map_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KG Embedding Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["evaluate", "predict", "similar", "export"],
                        help="Inference mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu), default: cuda if available")
    parser.add_argument("--data-dir", type=str, default="simulation/data/kg/clinical_kg_efficient",
                        help="Path to efficient KG data directory")
    parser.add_argument("--feature-dir", type=str, default="simulation/data/kg/multimodal/features",
                        help="Path to multimodal feature directory")
    parser.add_argument("--feature-dim", type=int, default=256,
                        help="Feature dimension for multimodal models")
    parser.add_argument("--modalities", type=str, default="text",
                        choices=["text", "text+image", "text+image+ecg", "text+image+ecg+structured"],
                        help="Modality setting for multimodal models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-kg", action="store_true",
                        help="Use test KG data paths instead of full clinical KG")

    # Evaluate mode
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-negatives", type=int, default=50,
                        help="Number of negative samples for evaluation")

    # Predict mode
    parser.add_argument("--head", type=str, default=None,
                        help="Head entity (name or ID) for prediction")
    parser.add_argument("--relation", type=str, default=None,
                        help="Relation (name or ID) for prediction")
    parser.add_argument("--tail", type=str, default=None,
                        help="Tail entity (name or ID) for prediction")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top predictions to return")

    # Similar mode
    parser.add_argument("--entity", type=str, default=None,
                        help="Entity (name or ID) for similarity search")

    # Export mode
    parser.add_argument("--output-dir", type=str, default="simulation/data/kg/results",
                        help="Output directory for exported embeddings")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_kg:
        args.data_dir = "simulation/data/kg/test_kg"
        args.feature_dir = "simulation/data/kg/test_kg/features"

    if args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "similar":
        run_similar(args)
    elif args.mode == "export":
        run_export(args)


if __name__ == "__main__":
    main()