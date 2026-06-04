from __future__ import annotations

import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from simulation.kg.extract_entities import ClinicalKG, Entity, Relation

ENTITY_TYPE_TO_ID = {"Finding": 0, "Anatomy": 1, "Disease": 2, "Patient": 3, "Study": 4}
MODALITY_TO_ID = {"CXR": 0, "ECG": 1, "RAD": 2, None: 3}

ENTITY_TYPE_ORDER = ["Finding", "Anatomy", "Disease", "Patient", "Study"]
CROSS_MODAL_RELATIONS = {"suggestive_of"}
WITHIN_MODAL_RELATIONS = {"located_at", "indicates", "modifies", "subsumes"}
STRUCTURAL_RELATIONS = {"has_finding", "finding_of", "same_patient"}

MODALITY_SET_MAP = {
    "text": {"text"},
    "text+image": {"text", "image"},
    "text+image+ecg": {"text", "image", "ecg"},
    "text+image+ecg+structured": {"text", "image", "ecg", "structured"},
}

FEATURE_KEY_MAP = {
    "text": "text",
    "image": "cxr",
    "ecg": "ecg",
    "structured": "structured",
}


class KGTriplesDataset:
    def __init__(self, kg: ClinicalKG, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42):
        sorted_entities = sorted(
            kg.entities.values(),
            key=lambda e: (ENTITY_TYPE_ORDER.index(e.type) if e.type in ENTITY_TYPE_ORDER else len(ENTITY_TYPE_ORDER), e.id),
        )
        self.entity2id: Dict[str, int] = {e.id: i for i, e in enumerate(sorted_entities)}
        self.id2entity: Dict[int, str] = {i: e.id for i, e in enumerate(sorted_entities)}

        all_relations = sorted(set(r.relation for r in kg.relations))
        self.relation2id: Dict[str, int] = {r: i for i, r in enumerate(all_relations)}
        self.id2relation: Dict[int, str] = {i: r for i, r in enumerate(all_relations)}

        triples_data = [
            (self.entity2id[rel.head], self.relation2id[rel.relation], self.entity2id[rel.tail], rel.weight)
            for rel in kg.relations
            if rel.head in self.entity2id and rel.tail in self.entity2id
        ]

        inverse_rel_map = self._build_inverse_map()

        self._known_triples_set: Set[Tuple[int, int, int]] = {(h, r, t) for h, r, t, _ in triples_data}

        self._hr_to_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self._rt_to_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for h, r, t in self._known_triples_set:
            self._hr_to_tails[(h, r)].add(t)
            self._rt_to_heads[(r, t)].add(h)

        all_triples = torch.tensor(triples_data, dtype=torch.float32)
        indices = torch.randperm(len(all_triples), generator=torch.Generator().manual_seed(seed))
        all_triples = all_triples[indices]

        n = len(all_triples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        self.train_triples = all_triples[:n_train]
        self.val_triples = all_triples[n_train:n_train + n_val]
        self.test_triples = all_triples[n_train + n_val:]

        self._entity_type_by_id: Dict[int, str] = {i: e.type for i, e in enumerate(sorted_entities)}
        self._entity_modality_by_id: Dict[int, Optional[str]] = {
            i: (e.modality if e.modality and e.modality != "None" else None)
            for i, e in enumerate(sorted_entities)
        }
        self._entity_ids_by_type: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(sorted_entities):
            self._entity_ids_by_type[e.type].append(i)

        self._relation_names_by_id: Dict[int, str] = dict(self.id2relation)

    @classmethod
    def from_efficient(cls, directory: Path, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42) -> 'KGTriplesDataset':
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)

        entity_id_map: Dict[str, int] = metadata["entity_id_map"]
        entity_labels: List[str] = metadata["entity_labels"]
        relation_names: List[str] = metadata["relation_names"]

        entity2id: Dict[str, int] = entity_id_map
        id2entity: Dict[int, str] = {v: k for k, v in entity_id_map.items()}
        relation2id: Dict[str, int] = {r: i for i, r in enumerate(relation_names)}
        id2relation: Dict[int, str] = {i: r for i, r in enumerate(relation_names)}

        entities_arr = np.load(directory / "entities.npy")

        entity_type_names = metadata["entity_type_names"]
        modality_names = metadata["modality_names"]
        ENTITY_ID_TO_TYPE = {i: name for i, name in enumerate(entity_type_names)}
        MODALITY_ID_TO_NAME = {i: name for i, name in enumerate(modality_names)}

        type_ids = entities_arr[:, 1].astype(np.int64)
        mod_ids = entities_arr[:, 2].astype(np.int64)

        _entity_type_by_id: Dict[int, str] = {}
        _entity_modality_by_id: Dict[int, Optional[str]] = {}
        _entity_ids_by_type: Dict[str, List[int]] = defaultdict(list)

        unique_type_ids = np.unique(type_ids)
        for tid in unique_type_ids:
            etype = ENTITY_ID_TO_TYPE.get(int(tid), "Unknown")
            mask = type_ids == tid
            indices = np.where(mask)[0]
            _entity_ids_by_type[etype] = indices.tolist()
            for idx in indices:
                _entity_type_by_id[int(idx)] = etype

        unique_mod_ids = np.unique(mod_ids)
        for mid in unique_mod_ids:
            mod_name = MODALITY_ID_TO_NAME.get(int(mid), "None")
            modality = mod_name if mod_name != "None" else None
            mask = mod_ids == mid
            indices = np.where(mask)[0]
            for idx in indices:
                _entity_modality_by_id[int(idx)] = modality

        for i in range(entities_arr.shape[0]):
            if i not in _entity_type_by_id:
                _entity_type_by_id[i] = "Unknown"
            if i not in _entity_modality_by_id:
                _entity_modality_by_id[i] = None

        relations_arr = np.load(directory / "relations.npy")

        heads = relations_arr['head'].astype(np.int64)
        rels = relations_arr['relation'].astype(np.int64)
        tails = relations_arr['tail'].astype(np.int64)
        weights = relations_arr['weight'].astype(np.float32)

        triples_data = np.stack([heads, rels, tails], axis=1)
        _known_triples_set: Set[Tuple[int, int, int]] = set(map(tuple, triples_data.tolist()))

        _hr_to_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        _rt_to_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for h, r, t in _known_triples_set:
            _hr_to_tails[(h, r)].add(t)
            _rt_to_heads[(r, t)].add(h)

        all_triples = torch.tensor(np.column_stack([triples_data, weights]), dtype=torch.float32)
        indices = torch.randperm(len(all_triples), generator=torch.Generator().manual_seed(seed))
        all_triples = all_triples[indices]

        n = len(all_triples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        train_triples = all_triples[:n_train]
        val_triples = all_triples[n_train:n_train + n_val]
        test_triples = all_triples[n_train + n_val:]

        _relation_names_by_id: Dict[int, str] = dict(id2relation)

        obj = cls.__new__(cls)
        obj.entity2id = entity2id
        obj.id2entity = id2entity
        obj.relation2id = relation2id
        obj.id2relation = id2relation
        obj._known_triples_set = _known_triples_set
        obj._hr_to_tails = _hr_to_tails
        obj._rt_to_heads = _rt_to_heads
        obj.train_triples = train_triples
        obj.val_triples = val_triples
        obj.test_triples = test_triples
        obj._entity_type_by_id = _entity_type_by_id
        obj._entity_modality_by_id = _entity_modality_by_id
        obj._entity_ids_by_type = _entity_ids_by_type
        obj._relation_names_by_id = _relation_names_by_id
        return obj

    @staticmethod
    def _build_inverse_map() -> Dict[str, str]:
        return {
            "has_finding": "finding_of",
            "finding_of": "has_finding",
        }

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @property
    def entity_types(self) -> Dict[int, str]:
        return dict(self._entity_type_by_id)

    @property
    def entity_modalities(self) -> Dict[int, Optional[str]]:
        return dict(self._entity_modality_by_id)

    @property
    def relation_names(self) -> Dict[int, str]:
        return dict(self.id2relation)

    def _split_tensors(self, data: torch.Tensor) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        triples = data[:, :3].long()
        weights = data[:, 3].float()
        return triples, weights

    def get_train_triples(self) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        return self._split_tensors(self.train_triples)

    def get_val_triples(self) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        return self._split_tensors(self.val_triples)

    def get_test_triples(self) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        return self._split_tensors(self.test_triples)

    def get_all_triples(self) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        return self._split_tensors(torch.cat([self.train_triples, self.val_triples, self.test_triples], dim=0))

    def _get_relation_mask(self, triples: torch.LongTensor, relation_names: Set[str]) -> torch.BoolTensor:
        rel_ids = [self.relation2id[r] for r in relation_names if r in self.relation2id]
        if not rel_ids:
            return torch.zeros(triples.shape[0], dtype=torch.bool)
        rel_ids_t = torch.tensor(rel_ids, dtype=torch.long)
        return (triples[:, 1].unsqueeze(1) == rel_ids_t.unsqueeze(0)).any(dim=1)

    def get_cross_modal_mask(self, triples: torch.LongTensor) -> torch.BoolTensor:
        return self._get_relation_mask(triples, CROSS_MODAL_RELATIONS)

    def get_within_modal_mask(self, triples: torch.LongTensor) -> torch.BoolTensor:
        return self._get_relation_mask(triples, WITHIN_MODAL_RELATIONS)

    def get_structural_mask(self, triples: torch.LongTensor) -> torch.BoolTensor:
        return self._get_relation_mask(triples, STRUCTURAL_RELATIONS)

    def filter_candidates(self, head: int, relation: int, direction: str = 'tail',
                      true_entity: int = -1) -> torch.LongTensor:
        all_ents = torch.arange(self.num_entities)
        if direction == 'tail':
            known = set(self._hr_to_tails.get((head, relation), set()))
        else:
            known = set(self._rt_to_heads.get((relation, head), set()))

        if true_entity >= 0:
            known.discard(true_entity)

        if not known:
            return all_ents

        mask = torch.ones(self.num_entities, dtype=torch.bool)
        for k in known:
            if 0 <= k < self.num_entities:
                mask[k] = False
        return all_ents[mask]

    def get_entity_type_ids(self) -> torch.LongTensor:
        ids = torch.zeros(self.num_entities, dtype=torch.long)
        for eid, etype in self._entity_type_by_id.items():
            ids[eid] = ENTITY_TYPE_TO_ID.get(etype, 0)
        return ids

    def get_entity_modality_ids(self) -> torch.LongTensor:
        ids = torch.full((self.num_entities,), len(MODALITY_TO_ID) - 1, dtype=torch.long)
        for eid, mod in self._entity_modality_by_id.items():
            ids[eid] = MODALITY_TO_ID.get(mod, len(MODALITY_TO_ID) - 1)
        return ids

    def get_disease_entity_indices(self) -> torch.LongTensor:
        ids = []
        for eid, etype in self._entity_type_by_id.items():
            if etype == "Disease":
                ids.append(eid)
        return torch.tensor(ids, dtype=torch.long)

    def get_entity_to_diseases(self) -> Tuple[torch.LongTensor, torch.LongTensor]:
        all_triples, _ = self.get_all_triples()
        disease_ids = self.get_disease_entity_indices()

        is_disease = torch.zeros(self.num_entities, dtype=torch.bool)
        is_disease[disease_ids] = True

        heads = all_triples[:, 0]
        tails = all_triples[:, 2]

        tail_is_disease = is_disease[tails]
        head_not_disease = ~is_disease[heads]
        mask_t = tail_is_disease & head_not_disease

        head_is_disease = is_disease[heads]
        tail_not_disease = ~is_disease[tails]
        mask_h = head_is_disease & tail_not_disease

        parts = []
        if mask_t.any():
            parts.append(torch.stack([heads[mask_t], tails[mask_t]], dim=1))
        if mask_h.any():
            parts.append(torch.stack([tails[mask_h], heads[mask_h]], dim=1))

        if not parts:
            return torch.zeros(0, 2, dtype=torch.long), torch.zeros(self.num_entities + 1, dtype=torch.long)

        all_pairs = torch.cat(parts, dim=0)
        all_pairs = torch.unique(all_pairs, dim=0)
        sort_idx = torch.argsort(all_pairs[:, 0])
        all_pairs = all_pairs[sort_idx]

        entity_counts = torch.bincount(all_pairs[:, 0], minlength=self.num_entities)
        ptrs = torch.zeros(self.num_entities + 1, dtype=torch.long)
        ptrs[1:] = torch.cumsum(entity_counts, dim=0)

        return all_pairs, ptrs

    def get_edge_data(self) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        triples, weights = self.get_all_triples()
        edge_index = torch.stack([triples[:, 0], triples[:, 2]], dim=0)
        edge_type = triples[:, 1]
        return edge_index, edge_type, weights

    @property
    def suggestive_of_rel_id(self) -> int:
        for rid, rname in self._relation_names_by_id.items():
            if rname == "suggestive_of":
                return rid
        return -1

    def get_modality_features(self, entity_ids: torch.LongTensor, modality: str) -> torch.FloatTensor:
        if not hasattr(self, "_modality_features") or self._modality_features is None:
            return torch.zeros(len(entity_ids), self.modality_feature_dim)

        feature_key = FEATURE_KEY_MAP.get(modality, modality)
        features_dict = self._modality_features.get(feature_key)
        if features_dict is None:
            return torch.zeros(len(entity_ids), self.modality_feature_dim)

        dim = self.modality_feature_dim
        result = torch.zeros(len(entity_ids), dim)
        for i, eid in enumerate(entity_ids.tolist()):
            if eid in features_dict:
                result[i] = features_dict[eid]
        return result

    def get_available_modalities(self, entity_id: int) -> List[str]:
        if not hasattr(self, "_modality_features") or self._modality_features is None:
            return []
        available = []
        for mod_key in ["text", "cxr", "ecg", "structured"]:
            features_dict = self._modality_features.get(mod_key)
            if features_dict is not None and entity_id in features_dict:
                available.append(mod_key)
        return available

    @property
    def modality_feature_dim(self) -> int:
        return getattr(self, "_modality_feature_dim", 256)

    def load_modality_features(self, feature_dir: Path, feature_dim: int = 256) -> None:
        self._modality_feature_dim = feature_dim
        self._modality_features = {}

        if not feature_dir.exists():
            print(f"[WARN] Feature directory not found: {feature_dir}")
            return

        for mod_name, file_stem in [("text", "text"), ("cxr", "cxr"), ("ecg", "ecg"), ("structured", "structured")]:
            feat_path = feature_dir / f"{file_stem}_features.npy"
            id_path = feature_dir / f"{file_stem}_feature_ids.csv"

            if not feat_path.exists() or not id_path.exists():
                print(f"[WARN] Missing feature files for {mod_name}: {feat_path.name}, {id_path.name}")
                continue

            features_np = np.load(feat_path)
            features_tensor = torch.from_numpy(features_np).float()

            id_map: Dict[int, torch.Tensor] = {}
            with open(id_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row_idx, row in enumerate(reader):
                    if len(row) >= 1:
                        eid = int(row[0])
                        if row_idx < features_tensor.shape[0]:
                            id_map[eid] = features_tensor[row_idx]

            self._modality_features[mod_name] = id_map
            print(f"  Loaded {len(id_map)} {mod_name} features (dim={features_tensor.shape[1]})")


class MultimodalKGTriplesDataset(KGTriplesDataset):
    def __init__(
        self,
        kg: ClinicalKG,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        feature_dir: Optional[Path] = None,
        feature_dim: int = 256,
        modalities: str = "text",
    ):
        super().__init__(kg, split_ratio, seed)
        self._modality_feature_dim = feature_dim
        self._modality_features: Dict[str, Dict[int, torch.Tensor]] = {}
        self.active_modalities = MODALITY_SET_MAP.get(modalities, {"text"})

        if feature_dir is not None:
            self.load_modality_features(feature_dir, feature_dim)

    @classmethod
    def from_efficient(
        cls,
        directory: Path,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        feature_dir: Optional[Path] = None,
        feature_dim: int = 256,
        modalities: str = "text",
    ) -> 'MultimodalKGTriplesDataset':
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)

        entity_id_map: Dict[str, int] = metadata["entity_id_map"]
        entity_labels: List[str] = metadata["entity_labels"]
        relation_names: List[str] = metadata["relation_names"]

        entity2id: Dict[str, int] = entity_id_map
        id2entity: Dict[int, str] = {v: k for k, v in entity_id_map.items()}
        relation2id: Dict[str, int] = {r: i for i, r in enumerate(relation_names)}
        id2relation: Dict[int, str] = {i: r for i, r in enumerate(relation_names)}

        entities_arr = np.load(directory / "entities.npy")

        entity_type_names = metadata["entity_type_names"]
        modality_names = metadata["modality_names"]
        ENTITY_ID_TO_TYPE = {i: name for i, name in enumerate(entity_type_names)}
        MODALITY_ID_TO_NAME = {i: name for i, name in enumerate(modality_names)}

        type_ids = entities_arr[:, 1].astype(np.int64)
        mod_ids = entities_arr[:, 2].astype(np.int64)

        _entity_type_by_id: Dict[int, str] = {}
        _entity_modality_by_id: Dict[int, Optional[str]] = {}
        _entity_ids_by_type: Dict[str, List[int]] = defaultdict(list)

        unique_type_ids = np.unique(type_ids)
        for tid in unique_type_ids:
            etype = ENTITY_ID_TO_TYPE.get(int(tid), "Unknown")
            mask = type_ids == tid
            indices = np.where(mask)[0]
            _entity_ids_by_type[etype] = indices.tolist()
            for idx in indices:
                _entity_type_by_id[int(idx)] = etype

        unique_mod_ids = np.unique(mod_ids)
        for mid in unique_mod_ids:
            mod_name = MODALITY_ID_TO_NAME.get(int(mid), "None")
            modality = mod_name if mod_name != "None" else None
            mask = mod_ids == mid
            indices = np.where(mask)[0]
            for idx in indices:
                _entity_modality_by_id[int(idx)] = modality

        for i in range(entities_arr.shape[0]):
            if i not in _entity_type_by_id:
                _entity_type_by_id[i] = "Unknown"
            if i not in _entity_modality_by_id:
                _entity_modality_by_id[i] = None

        relations_arr = np.load(directory / "relations.npy")

        heads = relations_arr['head'].astype(np.int64)
        rels = relations_arr['relation'].astype(np.int64)
        tails = relations_arr['tail'].astype(np.int64)
        weights = relations_arr['weight'].astype(np.float32)

        triples_data = np.stack([heads, rels, tails], axis=1)
        _known_triples_set: Set[Tuple[int, int, int]] = set(map(tuple, triples_data.tolist()))

        _hr_to_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        _rt_to_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for h, r, t in _known_triples_set:
            _hr_to_tails[(h, r)].add(t)
            _rt_to_heads[(r, t)].add(h)

        all_triples = torch.tensor(np.column_stack([triples_data, weights]), dtype=torch.float32)
        indices = torch.randperm(len(all_triples), generator=torch.Generator().manual_seed(seed))
        all_triples = all_triples[indices]

        n = len(all_triples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        train_triples = all_triples[:n_train]
        val_triples = all_triples[n_train:n_train + n_val]
        test_triples = all_triples[n_train + n_val:]

        _relation_names_by_id: Dict[int, str] = dict(id2relation)

        obj = cls.__new__(cls)
        obj.entity2id = entity2id
        obj.id2entity = id2entity
        obj.relation2id = relation2id
        obj.id2relation = id2relation
        obj._known_triples_set = _known_triples_set
        obj._hr_to_tails = _hr_to_tails
        obj._rt_to_heads = _rt_to_heads
        obj.train_triples = train_triples
        obj.val_triples = val_triples
        obj.test_triples = test_triples
        obj._entity_type_by_id = _entity_type_by_id
        obj._entity_modality_by_id = _entity_modality_by_id
        obj._entity_ids_by_type = _entity_ids_by_type
        obj._relation_names_by_id = _relation_names_by_id
        obj._modality_feature_dim = feature_dim
        obj._modality_features = {}
        obj.active_modalities = MODALITY_SET_MAP.get(modalities, {"text"})

        if feature_dir is not None:
            obj.load_modality_features(feature_dir, feature_dim)

        return obj

    def get_batch_with_modalities(
        self,
        batch_triples: torch.LongTensor,
        modalities: Set[str],
    ) -> Dict[str, torch.LongTensor]:
        heads = batch_triples[:, 0]
        tails = batch_triples[:, 2]

        result = {
            "heads": heads,
            "relations": batch_triples[:, 1],
            "tails": tails,
        }

        for mod in modalities:
            feature_key = FEATURE_KEY_MAP.get(mod, mod)
            head_feats = self.get_modality_features(heads, feature_key)
            tail_feats = self.get_modality_features(tails, feature_key)
            result[f"head_{mod}_features"] = head_feats
            result[f"tail_{mod}_features"] = tail_feats

        return result


class LinkPredictionEvaluator:
    def __init__(self, dataset: KGTriplesDataset):
        self.dataset = dataset
        self.known_triples: Set[Tuple[int, int, int]] = set(dataset._known_triples_set)
        self._hr_to_tails: Dict[Tuple[int, int], Set[int]] = dataset._hr_to_tails
        self._rt_to_heads: Dict[Tuple[int, int], Set[int]] = dataset._rt_to_heads

    def _rank_score(self, scores: torch.FloatTensor, true_idx: int, filter_set: Set[int]) -> Tuple[float, int]:
        true_score = scores[true_idx]
        if filter_set:
            filtered_scores = scores.clone()
            filter_idx = torch.tensor(list(filter_set), dtype=torch.long, device=scores.device)
            filtered_scores[filter_idx] = float("-inf")
        else:
            filtered_scores = scores
        rank = (filtered_scores > true_score).sum().item() + 1
        return rank

    @staticmethod
    def _rank_to_metrics(ranks: List[float]) -> Dict[str, float]:
        if not ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        mrr = sum(1.0 / r for r in ranks) / len(ranks)
        h1 = sum(1 for r in ranks if r <= 1) / len(ranks)
        h3 = sum(1 for r in ranks if r <= 3) / len(ranks)
        h10 = sum(1 for r in ranks if r <= 10) / len(ranks)
        return {"MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}

    def evaluate(self, model, triples: torch.LongTensor, weights: torch.FloatTensor,
                 batch_size: int = 256, device: str = "cpu",
                 max_triples: Optional[int] = None, num_eval_negatives: int = 50,
                 return_details: bool = False):
        model.eval()
        if max_triples is not None and triples.shape[0] > max_triples:
            rels_np = triples[:, 1].cpu().numpy() if triples.is_cuda else triples[:, 1].numpy()

            cross_ids = np.array([self.dataset.relation2id[r] for r in CROSS_MODAL_RELATIONS if r in self.dataset.relation2id])
            within_ids = np.array([self.dataset.relation2id[r] for r in WITHIN_MODAL_RELATIONS if r in self.dataset.relation2id])
            struct_ids = np.array([self.dataset.relation2id[r] for r in STRUCTURAL_RELATIONS if r in self.dataset.relation2id])

            cross_mask = np.isin(rels_np, cross_ids)
            within_mask = np.isin(rels_np, within_ids)
            struct_mask = np.isin(rels_np, struct_ids)

            cross_indices = np.where(cross_mask)[0]
            within_indices = np.where(within_mask)[0]
            struct_indices = np.where(struct_mask)[0]

            rng_np = np.random.RandomState(42)

            min_per_category = min(500, max_triples // 4)

            selected = []
            for idx_arr in [cross_indices, within_indices, struct_indices]:
                if len(idx_arr) > 0:
                    n_take = min(len(idx_arr), min_per_category)
                    chosen = rng_np.choice(idx_arr, size=n_take, replace=False)
                    selected.append(chosen)

            selected_arr = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
            remaining_budget = max_triples - len(selected_arr)

            if remaining_budget > 0:
                remaining_indices = np.where(~(cross_mask | within_mask | struct_mask))[0]
                uncat_or_other = np.concatenate([remaining_indices]) if len(remaining_indices) > 0 else np.array([], dtype=np.int64)
                already_selected = set(selected_arr.tolist())
                leftover = np.array([i for i in np.concatenate([cross_indices, within_indices, struct_indices]) if i not in already_selected], dtype=np.int64)
                pool = np.concatenate([uncat_or_other, leftover]) if len(uncat_or_other) > 0 and len(leftover) > 0 else (uncat_or_other if len(uncat_or_other) > 0 else leftover)

                if len(pool) > 0:
                    n_extra = min(len(pool), remaining_budget)
                    extra = rng_np.choice(pool, size=n_extra, replace=False)
                    selected_arr = np.concatenate([selected_arr, extra])

            triples = triples[selected_arr]
            weights = weights[selected_arr]

        all_ranks: List[float] = []
        cross_ranks: List[float] = []
        within_ranks: List[float] = []
        struct_ranks: List[float] = []
        per_triple_data: List[Dict] = []

        rel_id_cross = {self.dataset.relation2id[r] for r in CROSS_MODAL_RELATIONS if r in self.dataset.relation2id}
        rel_id_within = {self.dataset.relation2id[r] for r in WITHIN_MODAL_RELATIONS if r in self.dataset.relation2id}
        rel_id_struct = {self.dataset.relation2id[r] for r in STRUCTURAL_RELATIONS if r in self.dataset.relation2id}

        entity_type_ids = torch.zeros(self.dataset.num_entities, dtype=torch.long)
        for eid, type_name in self.dataset._entity_type_by_id.items():
            entity_type_ids[eid] = ENTITY_TYPE_TO_ID.get(type_name, 0)

        type_id_to_entity_ids_local: Dict[int, List[int]] = defaultdict(list)
        for eid, type_name in self.dataset._entity_type_by_id.items():
            tid = ENTITY_TYPE_TO_ID.get(type_name, 0)
            type_id_to_entity_ids_local[tid].append(eid)
        type_id_to_tensor: Dict[int, torch.LongTensor] = {}
        for tid, ids in type_id_to_entity_ids_local.items():
            type_id_to_tensor[tid] = torch.tensor(ids, dtype=torch.long)

        num_triples = triples.shape[0]
        rng = torch.Generator().manual_seed(42)

        with torch.no_grad():
            pbar = tqdm(range(0, num_triples, batch_size), desc="Evaluating", leave=False,
                        total=(num_triples + batch_size - 1) // batch_size)
            for start in pbar:
                end = min(start + batch_size, num_triples)
                batch = triples[start:end].to(device)
                bsz = batch.shape[0]

                heads = batch[:, 0]
                rels = batch[:, 1]
                tails = batch[:, 2]

                batch_tail_type_ids = entity_type_ids[batch[:, 2].cpu()]
                batch_head_type_ids = entity_type_ids[batch[:, 0].cpu()]

                max_cands = num_eval_negatives + 1
                tail_cands = torch.zeros(bsz, max_cands, dtype=torch.long)
                tail_cand_mask = torch.zeros(bsz, max_cands, dtype=torch.bool)
                tail_true_pos = torch.zeros(bsz, dtype=torch.long)

                for i in range(bsz):
                    t_i = int(tails[i])
                    t_type = int(batch_tail_type_ids[i])
                    cands = type_id_to_tensor[t_type]
                    n_avail = len(cands)
                    if n_avail <= num_eval_negatives:
                        sampled = cands.clone()
                    else:
                        idx = torch.randint(0, n_avail, (num_eval_negatives,), generator=rng)
                        sampled = cands[idx]
                    true_pos = (sampled == t_i).nonzero(as_tuple=True)[0]
                    if true_pos.numel() > 0:
                        tp = int(true_pos[0])
                        nc = sampled.shape[0]
                        tail_cands[i, :nc] = sampled
                        tail_cand_mask[i, :nc] = True
                        tail_true_pos[i] = tp
                    else:
                        nc = sampled.shape[0]
                        tail_cands[i, :nc] = sampled
                        tail_cand_mask[i, :nc] = True
                        tail_cands[i, nc] = t_i
                        tail_cand_mask[i, nc] = True
                        tail_true_pos[i] = nc

                tail_cands_dev = tail_cands.to(device)
                tail_cand_mask_dev = tail_cand_mask.to(device)

                tail_heads_exp = heads.unsqueeze(1).expand(-1, max_cands)
                tail_rels_exp = rels.unsqueeze(1).expand(-1, max_cands)
                tail_scores = model.score(
                    tail_heads_exp.reshape(-1),
                    tail_rels_exp.reshape(-1),
                    tail_cands_dev.reshape(-1),
                ).reshape(bsz, max_cands)
                tail_scores[~tail_cand_mask_dev] = float("-inf")

                head_cands = torch.zeros(bsz, max_cands, dtype=torch.long)
                head_cand_mask = torch.zeros(bsz, max_cands, dtype=torch.bool)
                head_true_pos = torch.zeros(bsz, dtype=torch.long)

                for i in range(bsz):
                    h_i = int(heads[i])
                    h_type = int(batch_head_type_ids[i])
                    cands = type_id_to_tensor[h_type]
                    n_avail = len(cands)
                    if n_avail <= num_eval_negatives:
                        sampled = cands.clone()
                    else:
                        idx = torch.randint(0, n_avail, (num_eval_negatives,), generator=rng)
                        sampled = cands[idx]
                    true_pos = (sampled == h_i).nonzero(as_tuple=True)[0]
                    if true_pos.numel() > 0:
                        tp = int(true_pos[0])
                        nc = sampled.shape[0]
                        head_cands[i, :nc] = sampled
                        head_cand_mask[i, :nc] = True
                        head_true_pos[i] = tp
                    else:
                        nc = sampled.shape[0]
                        head_cands[i, :nc] = sampled
                        head_cand_mask[i, :nc] = True
                        head_cands[i, nc] = h_i
                        head_cand_mask[i, nc] = True
                        head_true_pos[i] = nc

                head_cands_dev = head_cands.to(device)
                head_cand_mask_dev = head_cand_mask.to(device)

                head_rels_exp = rels.unsqueeze(1).expand(-1, max_cands)
                head_tails_exp = tails.unsqueeze(1).expand(-1, max_cands)
                head_scores = model.score(
                    head_cands_dev.reshape(-1),
                    head_rels_exp.reshape(-1),
                    head_tails_exp.reshape(-1),
                ).reshape(bsz, max_cands)
                head_scores[~head_cand_mask_dev] = float("-inf")

                heads_cpu = batch[:, 0].cpu()
                rels_cpu = batch[:, 1].cpu()
                tails_cpu = batch[:, 2].cpu()

                for i in range(bsz):
                    h_i = int(heads_cpu[i])
                    r_i = int(rels_cpu[i])
                    t_i = int(tails_cpu[i])

                    filter_tails = self._hr_to_tails.get((h_i, r_i), set()) - {t_i}
                    if filter_tails:
                        ft_tensor = torch.tensor(list(filter_tails), dtype=torch.long)
                        mask = (tail_cands[i].unsqueeze(0) == ft_tensor.unsqueeze(1)).any(dim=0)
                        tail_scores[i, mask] = float("-inf")

                    filter_heads = self._rt_to_heads.get((r_i, t_i), set()) - {h_i}
                    if filter_heads:
                        fh_tensor = torch.tensor(list(filter_heads), dtype=torch.long)
                        mask = (head_cands[i].unsqueeze(0) == fh_tensor.unsqueeze(1)).any(dim=0)
                        head_scores[i, mask] = float("-inf")

                tail_true_scores = tail_scores[torch.arange(bsz, device=device), tail_true_pos.to(device)]
                tail_ranks = (tail_scores > tail_true_scores.unsqueeze(1)).sum(dim=1) + 1

                head_true_scores = head_scores[torch.arange(bsz, device=device), head_true_pos.to(device)]
                head_ranks = (head_scores > head_true_scores.unsqueeze(1)).sum(dim=1) + 1

                rels_list = rels_cpu.tolist()
                rels_arr = rels_cpu.numpy() if hasattr(rels_cpu, 'numpy') else np.array(rels_list)
                cross_mask = np.isin(rels_arr, list(rel_id_cross))
                within_mask = np.isin(rels_arr, list(rel_id_within))
                struct_mask = np.isin(rels_arr, list(rel_id_struct))

                tail_ranks_cpu = tail_ranks.cpu().tolist()
                head_ranks_cpu = head_ranks.cpu().tolist()

                for i in range(bsz):
                    all_ranks.append(float(tail_ranks_cpu[i]))
                    all_ranks.append(float(head_ranks_cpu[i]))
                    if cross_mask[i]:
                        cross_ranks.append(float(tail_ranks_cpu[i]))
                        cross_ranks.append(float(head_ranks_cpu[i]))
                    elif within_mask[i]:
                        within_ranks.append(float(tail_ranks_cpu[i]))
                        within_ranks.append(float(head_ranks_cpu[i]))
                    elif struct_mask[i]:
                        struct_ranks.append(float(tail_ranks_cpu[i]))
                        struct_ranks.append(float(head_ranks_cpu[i]))
                    if return_details:
                        h_type = int(batch_head_type_ids[i])
                        t_type = int(batch_tail_type_ids[i])
                        r_id = int(rels_cpu[i])
                        per_triple_data.append({'rank': float(tail_ranks_cpu[i]), 'head_type': h_type, 'tail_type': t_type, 'relation_id': r_id, 'direction': 'tail'})
                        per_triple_data.append({'rank': float(head_ranks_cpu[i]), 'head_type': h_type, 'tail_type': t_type, 'relation_id': r_id, 'direction': 'head'})

                del tail_scores, head_scores, tail_cands_dev, head_cands_dev

        overall = self._rank_to_metrics(all_ranks)
        cross = self._rank_to_metrics(cross_ranks)
        within = self._rank_to_metrics(within_ranks)
        struct = self._rank_to_metrics(struct_ranks)

        results: Dict[str, float] = {}
        for k, v in overall.items():
            results[k] = v
        for k, v in cross.items():
            results[f"{k}_cross"] = v
        for k, v in within.items():
            results[f"{k}_within"] = v
        for k, v in struct.items():
            results[f"{k}_struct"] = v

        if return_details:
            return results, per_triple_data
        return results


class NegativeSampler:
    def __init__(self, dataset: KGTriplesDataset, num_negatives: int = 64):
        self.dataset = dataset
        self.num_negatives = num_negatives
        self._entity_ids_by_type: Dict[str, List[int]] = dict(dataset._entity_ids_by_type)
        self._type_to_cands: Dict[str, torch.LongTensor] = {}
        for type_name, ids in self._entity_ids_by_type.items():
            if ids:
                self._type_to_cands[type_name] = torch.tensor(ids, dtype=torch.long)
        self._rng = torch.Generator().manual_seed(42)

    def sample(self, triples: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        n = triples.shape[0]
        num_neg = self.num_negatives

        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]

        head_types = [self.dataset._entity_type_by_id[int(h)] for h in heads]
        tail_types = [self.dataset._entity_type_by_id[int(t)] for t in tails]

        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i in range(n):
            ht = head_types[i]
            tt = tail_types[i]
            if ht not in self._type_to_cands and tt not in self._type_to_cands:
                continue
            groups[(ht, tt)].append(i)

        neg_parts: List[torch.LongTensor] = []

        for (ht, tt), indices in groups.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            batch = triples[idx_tensor]
            n_group = len(indices)

            h_cands = self._type_to_cands.get(ht)
            t_cands = self._type_to_cands.get(tt)

            if h_cands is not None:
                rand_idx = torch.randint(0, len(h_cands), (n_group, num_neg), generator=self._rng)
                neg_h = h_cands[rand_idx]
                neg_r = batch[:, 1].unsqueeze(1).expand(-1, num_neg)
                neg_t = batch[:, 2].unsqueeze(1).expand(-1, num_neg)
                neg_batch = torch.stack([neg_h, neg_r, neg_t], dim=-1).reshape(-1, 3)
                neg_batch = self._filter_known(neg_batch, h_cands, 0)
                neg_parts.append(neg_batch)

            if t_cands is not None:
                rand_idx = torch.randint(0, len(t_cands), (n_group, num_neg), generator=self._rng)
                neg_t_vals = t_cands[rand_idx]
                neg_h_vals = batch[:, 0].unsqueeze(1).expand(-1, num_neg)
                neg_r_vals = batch[:, 1].unsqueeze(1).expand(-1, num_neg)
                neg_batch = torch.stack([neg_h_vals, neg_r_vals, neg_t_vals], dim=-1).reshape(-1, 3)
                neg_batch = self._filter_known(neg_batch, t_cands, 2)
                neg_parts.append(neg_batch)

        if not neg_parts:
            return torch.zeros(0, 3, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        all_neg = torch.cat(neg_parts, dim=0)
        labels = torch.zeros(all_neg.shape[0], dtype=torch.long)
        return all_neg, labels

    def _filter_known(self, neg_triples: torch.LongTensor, cands: torch.LongTensor, replace_col: int) -> torch.LongTensor:
        known = self.dataset._known_triples_set
        if not known:
            return neg_triples
        for _ in range(2):
            is_known = torch.tensor(
                [(int(neg_triples[i, 0]), int(neg_triples[i, 1]), int(neg_triples[i, 2])) in known
                 for i in range(neg_triples.size(0))],
                dtype=torch.bool,
            )
            if not is_known.any():
                break
            idx = is_known.nonzero(as_tuple=True)[0]
            neg_triples[idx, replace_col] = cands[torch.randint(0, len(cands), (idx.size(0),), generator=self._rng)]
        return neg_triples


if __name__ == "__main__":
    efficient_dir = Path(__file__).parent.parent / "data" / "kg" / "clinical_kg_efficient"
    kg_path = Path(__file__).parent.parent / "data" / "kg" / "clinical_kg.pkl"

    if efficient_dir.exists() and (efficient_dir / "metadata.json").exists():
        print(f"Loading from efficient format: {efficient_dir}")
        dataset = KGTriplesDataset.from_efficient(efficient_dir)
    elif kg_path.exists():
        print(f"Loading from pickle: {kg_path}")
        with open(kg_path, "rb") as f:
            kg: ClinicalKG = pickle.load(f)
        dataset = KGTriplesDataset(kg)
    else:
        print(f"KG not found (checked {efficient_dir} and {kg_path})")
        raise SystemExit(1)
    print(f"Entities: {dataset.num_entities}")
    print(f"Relations: {dataset.num_relations}")
    print(f"Relation names: {dataset.relation_names}")

    train_t, train_w = dataset.get_train_triples()
    val_t, val_w = dataset.get_val_triples()
    test_t, test_w = dataset.get_test_triples()
    print(f"Train triples: {train_t.shape[0]}")
    print(f"Val triples: {val_t.shape[0]}")
    print(f"Test triples: {test_t.shape[0]}")

    cross_m = dataset.get_cross_modal_mask(train_t)
    within_m = dataset.get_within_modal_mask(train_t)
    struct_m = dataset.get_structural_mask(train_t)
    print(f"Cross-modal train triples: {cross_m.sum().item()}")
    print(f"Within-modal train triples: {within_m.sum().item()}")
    print(f"Structural train triples: {struct_m.sum().item()}")

    sampler = NegativeSampler(dataset, num_negatives=4)
    neg_t, neg_l = sampler.sample(train_t[:10])
    print(f"Negative samples from 10 triples: {neg_t.shape[0]}")

    print(f"\nEntity types distribution:")
    type_counts: Dict[str, int] = defaultdict(int)
    for eid, etype in dataset.entity_types.items():
        type_counts[etype] += 1
    for etype in ENTITY_TYPE_ORDER:
        print(f"  {etype}: {type_counts.get(etype, 0)}")
