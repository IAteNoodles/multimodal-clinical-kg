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

        entities_arr = np.load(directory / "entities.npy", mmap_mode='r')

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

        relations_arr = np.load(directory / "relations.npy", mmap_mode='r')

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
        known_list = [k for k in known if 0 <= k < self.num_entities]
        if not known_list:
            return all_ents
        mask[torch.tensor(known_list, dtype=torch.long)] = False
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
        if not hasattr(self, "_modality_padded") or self._modality_padded is None:
            return torch.zeros(len(entity_ids), self.modality_feature_dim)

        feature_key = FEATURE_KEY_MAP.get(modality, modality)
        padded = self._modality_padded.get(feature_key)
        if padded is None:
            return torch.zeros(len(entity_ids), self.modality_feature_dim)

        return padded[entity_ids]

    def get_available_modalities(self, entity_id: int) -> List[str]:
        if not hasattr(self, "_modality_mask") or self._modality_mask is None:
            return []
        available = []
        for mod_key in ["text", "cxr", "ecg", "structured"]:
            mask = self._modality_mask.get(mod_key)
            if mask is not None and mask[entity_id]:
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

            features_np = np.load(feat_path, mmap_mode='r')
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

        self._build_modality_padded_tensors()

    def _build_modality_padded_tensors(self) -> None:
        self._modality_padded: Dict[str, torch.Tensor] = {}
        self._modality_mask: Dict[str, torch.BoolTensor] = {}
        if not hasattr(self, "_modality_features") or not self._modality_features:
            return
        num_entities = self.num_entities
        feat_dim = self._modality_feature_dim
        for mod_key, feat_dict in self._modality_features.items():
            if not feat_dict:
                continue
            padded = torch.zeros(num_entities, feat_dim)
            mask = torch.zeros(num_entities, dtype=torch.bool)
            for eid, feat in feat_dict.items():
                padded[eid] = feat
                mask[eid] = True
            self._modality_padded[mod_key] = padded
            self._modality_mask[mod_key] = mask


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

        entities_arr = np.load(directory / "entities.npy", mmap_mode='r')

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

        relations_arr = np.load(directory / "relations.npy", mmap_mode='r')

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
    def __init__(self, dataset: KGTriplesDataset, device: str = "cpu"):
        self.dataset = dataset
        self.device = torch.device(device)
        self.num_entities = dataset.num_entities
        self.num_relations = dataset.num_relations
        self.num_types = len(ENTITY_TYPE_TO_ID)

        # CPU-side filter sets kept for filtered ranking
        self.known_triples: Set[Tuple[int, int, int]] = set(dataset._known_triples_set)
        self._hr_to_tails: Dict[Tuple[int, int], Set[int]] = dataset._hr_to_tails
        self._rt_to_heads: Dict[Tuple[int, int], Set[int]] = dataset._rt_to_heads

        # GPU-resident: entity_id → type_id mapping
        self.entity_type_tensor = torch.zeros(self.num_entities, dtype=torch.long, device=self.device)
        for eid, etype in dataset._entity_type_by_id.items():
            self.entity_type_tensor[eid] = ENTITY_TYPE_TO_ID.get(etype, 0)

        # GPU-resident: padded type entity pools [num_types, max_pool_size]
        all_pool_sizes = [len(ids) for ids in dataset._entity_ids_by_type.values()] if dataset._entity_ids_by_type else [0]
        self.max_pool_size = max(all_pool_sizes)
        self.type_entity_pools_padded = torch.zeros(self.num_types, self.max_pool_size, dtype=torch.long, device=self.device)
        self.type_pool_sizes_tensor = torch.zeros(self.num_types, dtype=torch.long, device=self.device)
        for type_name, ids in dataset._entity_ids_by_type.items():
            if ids:
                tid = ENTITY_TYPE_TO_ID.get(type_name, 0)
                self.type_entity_pools_padded[tid, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                self.type_pool_sizes_tensor[tid] = len(ids)

        # GPU-resident: known triples hash keys for O(1) lookup
        # Key = head * num_relations * num_entities + rel * num_entities + tail
        known_list = list(dataset._known_triples_set)
        if known_list:
            keys = torch.tensor(
                [h * self.num_relations * self.num_entities + r * self.num_entities + t
                 for h, r, t in known_list],
                dtype=torch.long,
            )
            sorted_keys, _ = torch.sort(torch.unique(keys))
            self.known_triples_keys_sorted = sorted_keys.to(self.device)
        else:
            self.known_triples_keys_sorted = torch.zeros(0, dtype=torch.long, device=self.device)

        # GPU-resident: relation category masks for vectorized rank binning
        rel_id_cross = torch.tensor(
            [dataset.relation2id[r] for r in CROSS_MODAL_RELATIONS if r in dataset.relation2id],
            dtype=torch.long, device=self.device,
        )
        rel_id_within = torch.tensor(
            [dataset.relation2id[r] for r in WITHIN_MODAL_RELATIONS if r in dataset.relation2id],
            dtype=torch.long, device=self.device,
        )
        rel_id_struct = torch.tensor(
            [dataset.relation2id[r] for r in STRUCTURAL_RELATIONS if r in dataset.relation2id],
            dtype=torch.long, device=self.device,
        )
        self._rel_id_cross = rel_id_cross
        self._rel_id_within = rel_id_within
        self._rel_id_struct = rel_id_struct

        if self.device.type == 'cuda':
            self._rng = torch.Generator(device=self.device)
            self._rng.manual_seed(42)
        else:
            self._rng = torch.Generator().manual_seed(42)

    def _compute_triple_keys(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return h * (self.num_relations * self.num_entities) + r * self.num_entities + t

    def _sample_from_pools(self, type_ids: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Vectorized sampling from per-type entity pools. type_ids: [n], returns [n, num_samples]."""
        n = type_ids.size(0)
        pool_sizes = self.type_pool_sizes_tensor[type_ids]  # [n]
        rand_idx = torch.randint(0, self.max_pool_size, (n, num_samples), generator=self._rng, device=self.device)
        rand_idx = rand_idx % pool_sizes.unsqueeze(1).clamp(min=1)
        type_idx = type_ids.unsqueeze(1).expand(-1, num_samples)
        return self.type_entity_pools_padded[type_idx, rand_idx]  # [n, num_samples]

    @staticmethod
    def _rank_to_metrics(ranks: List[float]) -> Dict[str, float]:
        if not ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        mrr = sum(1.0 / r for r in ranks) / len(ranks)
        h1 = sum(1 for r in ranks if r <= 1) / len(ranks)
        h3 = sum(1 for r in ranks if r <= 3) / len(ranks)
        h10 = sum(1 for r in ranks if r <= 10) / len(ranks)
        return {"MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}

    def _vectorized_sample_candidates(
        self, true_entities: torch.Tensor, type_ids: torch.Tensor, num_neg: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized candidate sampling for a batch.

        Args:
            true_entities: [bsz] true entity ids (on device)
            type_ids: [bsz] type ids for those entities (on device)
            num_neg: number of negative candidates

        Returns:
            cands: [bsz, max_cands] candidate entity ids on device
            cand_mask: [bsz, max_cands] bool mask on device
            true_pos: [bsz] index of true entity in candidate list (on device)
        """
        bsz = true_entities.size(0)
        pool_sizes = self.type_pool_sizes_tensor[type_ids]  # [bsz]

        # Determine per-row candidate counts: min(pool_size, num_neg) + 1 slot for true entity
        nc_per_row = pool_sizes.clamp(max=num_neg)  # number of sampled negatives per row
        max_cands = int(nc_per_row.max().item()) + 1  # +1 for true entity slot

        # Sample num_neg candidates per row from type pools
        sampled = self._sample_from_pools(type_ids, num_neg)  # [bsz, num_neg]

        # Build candidates tensor on device
        cands = torch.zeros(bsz, max_cands, dtype=torch.long, device=self.device)
        cand_mask = torch.zeros(bsz, max_cands, dtype=torch.bool, device=self.device)

        # Place sampled candidates row by row using vectorized scatter
        # For each row i, place nc_per_row[i] samples into cands[i, :nc_per_row[i]]
        row_idx = torch.arange(bsz, device=self.device)
        for j in range(num_neg):
            valid = j < nc_per_row  # [bsz] bool
            cands[row_idx[valid], j] = sampled[valid, j]
            cand_mask[row_idx[valid], j] = True

        # Check if true entity already in sampled portion
        # sampled: [bsz, num_neg], true_entities: [bsz]
        true_in_sampled = (sampled == true_entities.unsqueeze(1))  # [bsz, num_neg]

        # For rows where true entity is present, find its first position
        # For rows where it's absent, append it at position nc_per_row
        has_true = true_in_sampled.any(dim=1)  # [bsz]

        # Compute true_pos: for rows with true entity, first occurrence index
        # For rows without, nc_per_row (appended position)
        # first_true_idx: [bsz] - index of first match in sampled
        first_true_idx = true_in_sampled.float().argmax(dim=1)  # [bsz]
        true_pos = torch.where(has_true, first_true_idx, nc_per_row)

        # For rows without true entity, append it
        append_row = row_idx[~has_true]
        append_col = nc_per_row[~has_true]
        cands[append_row, append_col] = true_entities[~has_true]
        cand_mask[append_row, append_col] = True

        return cands, cand_mask, true_pos

    def evaluate(self, model, triples: torch.LongTensor, weights: torch.FloatTensor,
                 batch_size: int = 256, device: str = "cpu",
                 max_triples: Optional[int] = None, num_eval_negatives: int = 50,
                 return_details: bool = False):
        model.eval()
        eval_dev = self.device

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

        # Move triples to device once
        triples_dev = triples.to(eval_dev)
        num_triples = triples_dev.shape[0]

        # Pre-compute type ids on device
        heads_all = triples_dev[:, 0]
        rels_all = triples_dev[:, 1]
        tails_all = triples_dev[:, 2]
        head_type_ids_all = self.entity_type_tensor[heads_all]
        tail_type_ids_all = self.entity_type_tensor[tails_all]

        # Pre-compute relation category masks on device
        rels_all_cpu = rels_all.cpu()
        rels_np = rels_all_cpu.numpy()
        cross_mask_all = np.isin(rels_np, self._rel_id_cross.cpu().numpy())
        within_mask_all = np.isin(rels_np, self._rel_id_within.cpu().numpy())
        struct_mask_all = np.isin(rels_np, self._rel_id_struct.cpu().numpy())

        with torch.no_grad():
            pbar = tqdm(range(0, num_triples, batch_size), desc="Evaluating", leave=False,
                        total=(num_triples + batch_size - 1) // batch_size)
            for start in pbar:
                end = min(start + batch_size, num_triples)
                batch_idx = slice(start, end)
                bsz = end - start

                heads = heads_all[batch_idx]
                rels = rels_all[batch_idx]
                tails = tails_all[batch_idx]
                batch_head_type_ids = head_type_ids_all[batch_idx]
                batch_tail_type_ids = tail_type_ids_all[batch_idx]

                # --- Vectorized tail candidate sampling ---
                tail_cands, tail_cand_mask, tail_true_pos = self._vectorized_sample_candidates(
                    tails, batch_tail_type_ids, num_eval_negatives,
                )

                max_cands = tail_cands.size(1)

                # Score tail candidates
                tail_heads_exp = heads.unsqueeze(1).expand(-1, max_cands)
                tail_rels_exp = rels.unsqueeze(1).expand(-1, max_cands)
                tail_scores = model.score(
                    tail_heads_exp.reshape(-1),
                    tail_rels_exp.reshape(-1),
                    tail_cands.reshape(-1),
                ).reshape(bsz, max_cands)
                tail_scores[~tail_cand_mask] = float("-inf")

                # --- Vectorized head candidate sampling ---
                head_cands, head_cand_mask, head_true_pos = self._vectorized_sample_candidates(
                    heads, batch_head_type_ids, num_eval_negatives,
                )

                max_cands_h = head_cands.size(1)
                # Pad tail/head candidates to same width if needed
                if max_cands_h != max_cands:
                    # Re-score with the actual max_cands_h
                    pass
                max_cands = max(max_cands, head_cands.size(1))

                # Ensure tail arrays are padded to max_cands
                if tail_cands.size(1) < max_cands:
                    pad = max_cands - tail_cands.size(1)
                    tail_cands = torch.cat([tail_cands, torch.zeros(bsz, pad, dtype=torch.long, device=self.device)], dim=1)
                    tail_cand_mask = torch.cat([tail_cand_mask, torch.zeros(bsz, pad, dtype=torch.bool, device=self.device)], dim=1)
                    if tail_scores.size(1) < max_cands:
                        tail_scores = torch.cat([tail_scores, torch.full((bsz, pad), float("-inf"), device=self.device)], dim=1)

                # Ensure head arrays are padded to max_cands
                if head_cands.size(1) < max_cands:
                    pad = max_cands - head_cands.size(1)
                    head_cands = torch.cat([head_cands, torch.zeros(bsz, pad, dtype=torch.long, device=self.device)], dim=1)
                    head_cand_mask = torch.cat([head_cand_mask, torch.zeros(bsz, pad, dtype=torch.bool, device=self.device)], dim=1)

                # Score head candidates
                head_rels_exp = rels.unsqueeze(1).expand(-1, head_cands.size(1) if head_cands.size(1) == max_cands else max_cands)
                head_tails_exp = tails.unsqueeze(1).expand(-1, max_cands)
                # Use original head_cands width for scoring if it differs
                hc_width = head_cands.size(1) if head_true_pos.max() < head_cands.size(1) else max_cands
                head_scores = model.score(
                    head_cands[:, :hc_width].reshape(-1),
                    rels.unsqueeze(1).expand(-1, hc_width).reshape(-1),
                    tails.unsqueeze(1).expand(-1, hc_width).reshape(-1),
                ).reshape(bsz, hc_width)
                head_scores[~head_cand_mask[:, :hc_width]] = float("-inf")

                # --- Vectorized filtered ranking using GPU hash keys ---
                # Build candidate triple keys for filtering
                # Tail direction: (head, rel, tail_candidate)
                tc_flat = tail_cands[:, :tail_scores.size(1)].reshape(-1)
                th_flat = heads.unsqueeze(1).expand(-1, tail_scores.size(1)).reshape(-1)
                tr_flat = rels.unsqueeze(1).expand(-1, tail_scores.size(1)).reshape(-1)
                tail_keys = self._compute_triple_keys(th_flat, tr_flat, tc_flat)
                tail_keys = tail_keys.reshape(bsz, -1)

                # Mask out known triples (excluding the true triple) from tail scores
                pos = torch.searchsorted(self.known_triples_keys_sorted, tail_keys)
                pos = pos.clamp(max=self.known_triples_keys_sorted.size(0) - 1)
                is_known_tail = self.known_triples_keys_sorted[pos] == tail_keys
                # Don't filter the true entity position
                true_pos_mask_tail = torch.zeros_like(is_known_tail)
                true_pos_mask_tail[torch.arange(bsz, device=self.device), tail_true_pos] = True
                filter_tail = is_known_tail & ~true_pos_mask_tail & tail_cand_mask[:, :tail_scores.size(1)]
                tail_scores[filter_tail] = float("-inf")

                # Head direction: (head_candidate, rel, tail)
                hc_flat = head_cands[:, :head_scores.size(1)].reshape(-1)
                ht_flat = tails.unsqueeze(1).expand(-1, head_scores.size(1)).reshape(-1)
                hr_flat = rels.unsqueeze(1).expand(-1, head_scores.size(1)).reshape(-1)
                head_keys = self._compute_triple_keys(hc_flat, hr_flat, ht_flat)
                head_keys = head_keys.reshape(bsz, -1)

                pos = torch.searchsorted(self.known_triples_keys_sorted, head_keys)
                pos = pos.clamp(max=self.known_triples_keys_sorted.size(0) - 1)
                is_known_head = self.known_triples_keys_sorted[pos] == head_keys
                true_pos_mask_head = torch.zeros_like(is_known_head)
                true_pos_mask_head[torch.arange(bsz, device=self.device), head_true_pos] = True
                filter_head = is_known_head & ~true_pos_mask_head & head_cand_mask[:, :head_scores.size(1)]
                head_scores[filter_head] = float("-inf")

                # --- Vectorized rank computation ---
                tail_true_scores = tail_scores[torch.arange(bsz, device=self.device), tail_true_pos]
                tail_ranks = (tail_scores > tail_true_scores.unsqueeze(1)).sum(dim=1) + 1

                head_true_scores = head_scores[torch.arange(bsz, device=self.device), head_true_pos]
                head_ranks = (head_scores > head_true_scores.unsqueeze(1)).sum(dim=1) + 1

                # --- Vectorized rank binning ---
                batch_cross = cross_mask_all[start:end]
                batch_within = within_mask_all[start:end]
                batch_struct = struct_mask_all[start:end]

                tail_ranks_cpu = tail_ranks.cpu()
                head_ranks_cpu = head_ranks.cpu()
                tail_ranks_list = tail_ranks_cpu.tolist()
                head_ranks_list = head_ranks_cpu.tolist()

                batch_head_type_cpu = batch_head_type_ids.cpu()
                batch_tail_type_cpu = batch_tail_type_ids.cpu()
                rels_cpu = rels.cpu()

                for i in range(bsz):
                    all_ranks.append(float(tail_ranks_list[i]))
                    all_ranks.append(float(head_ranks_list[i]))
                    if batch_cross[i]:
                        cross_ranks.append(float(tail_ranks_list[i]))
                        cross_ranks.append(float(head_ranks_list[i]))
                    elif batch_within[i]:
                        within_ranks.append(float(tail_ranks_list[i]))
                        within_ranks.append(float(head_ranks_list[i]))
                    elif batch_struct[i]:
                        struct_ranks.append(float(tail_ranks_list[i]))
                        struct_ranks.append(float(head_ranks_list[i]))
                    if return_details:
                        h_type = int(batch_head_type_cpu[i])
                        t_type = int(batch_tail_type_cpu[i])
                        r_id = int(rels_cpu[i])
                        per_triple_data.append({'rank': float(tail_ranks_list[i]), 'head_type': h_type, 'tail_type': t_type, 'relation_id': r_id, 'direction': 'tail'})
                        per_triple_data.append({'rank': float(head_ranks_list[i]), 'head_type': h_type, 'tail_type': t_type, 'relation_id': r_id, 'direction': 'head'})

                del tail_scores, head_scores

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
    def __init__(self, dataset: KGTriplesDataset, num_negatives: int = 64, device: str = "cpu"):
        self.dataset = dataset
        self.num_negatives = num_negatives
        self.device = torch.device(device)
        self.num_entities = dataset.num_entities
        self.num_relations = dataset.num_relations
        self.num_types = len(ENTITY_TYPE_TO_ID)

        # GPU-resident: entity_id → type_id mapping
        self.entity_type_tensor = torch.zeros(self.num_entities, dtype=torch.long, device=self.device)
        for eid, etype in dataset._entity_type_by_id.items():
            self.entity_type_tensor[eid] = ENTITY_TYPE_TO_ID.get(etype, 0)

        # GPU-resident: padded type entity pools [num_types, max_pool_size]
        all_pool_sizes = [len(ids) for ids in dataset._entity_ids_by_type.values()] if dataset._entity_ids_by_type else [0]
        self.max_pool_size = max(all_pool_sizes)
        self.type_entity_pools_padded = torch.zeros(self.num_types, self.max_pool_size, dtype=torch.long, device=self.device)
        self.type_pool_sizes_tensor = torch.zeros(self.num_types, dtype=torch.long, device=self.device)
        for type_name, ids in dataset._entity_ids_by_type.items():
            if ids:
                tid = ENTITY_TYPE_TO_ID.get(type_name, 0)
                self.type_entity_pools_padded[tid, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                self.type_pool_sizes_tensor[tid] = len(ids)

        # GPU-resident: known triples hash keys for O(1) lookup
        # Key = head * num_relations * num_entities + rel * num_entities + tail
        known_list = list(dataset._known_triples_set)
        if known_list:
            keys = torch.tensor(
                [h * self.num_relations * self.num_entities + r * self.num_entities + t
                 for h, r, t in known_list],
                dtype=torch.long,
            )
            sorted_keys, _ = torch.sort(torch.unique(keys))
            self.known_triples_keys_sorted = sorted_keys.to(self.device)
        else:
            self.known_triples_keys_sorted = torch.zeros(0, dtype=torch.long, device=self.device)

        if self.device.type == 'cuda':
            self._rng = torch.Generator(device=self.device)
            self._rng.manual_seed(42)
        else:
            self._rng = torch.Generator().manual_seed(42)

    def _compute_triple_keys(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return h * (self.num_relations * self.num_entities) + r * self.num_entities + t

    def _sample_from_pools(self, type_ids: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Fully vectorized sampling from per-type entity pools. type_ids: [n], returns [n, num_samples]."""
        n = type_ids.size(0)
        pool_sizes = self.type_pool_sizes_tensor[type_ids]  # [n]
        rand_idx = torch.randint(0, self.max_pool_size, (n, num_samples), generator=self._rng, device=self.device)
        rand_idx = rand_idx % pool_sizes.unsqueeze(1).clamp(min=1)  # clamp avoids div-by-0 for empty types
        type_idx = type_ids.unsqueeze(1).expand(-1, num_samples)
        return self.type_entity_pools_padded[type_idx, rand_idx]  # [n, num_samples]

    def sample(self, triples: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        n = triples.shape[0]
        num_neg = self.num_negatives

        triples_dev = triples.to(self.device)
        heads = triples_dev[:, 0]
        rels = triples_dev[:, 1]
        tails = triples_dev[:, 2]

        # Vectorized type lookup — no Python loops
        head_type_ids = self.entity_type_tensor[heads]  # [n]
        tail_type_ids = self.entity_type_tensor[tails]  # [n]

        neg_parts: List[torch.LongTensor] = []

        # Head corruption: replace head with same-type entity
        neg_heads = self._sample_from_pools(head_type_ids, num_neg)  # [n, num_neg]
        neg_rels_h = rels.unsqueeze(1).expand(-1, num_neg)
        neg_tails_h = tails.unsqueeze(1).expand(-1, num_neg)
        neg_batch_h = torch.stack([neg_heads, neg_rels_h, neg_tails_h], dim=-1).reshape(-1, 3)
        neg_batch_h = self._filter_known(neg_batch_h, 0, head_type_ids)
        neg_parts.append(neg_batch_h)

        # Tail corruption: replace tail with same-type entity
        neg_tails = self._sample_from_pools(tail_type_ids, num_neg)  # [n, num_neg]
        neg_rels_t = rels.unsqueeze(1).expand(-1, num_neg)
        neg_heads_t = heads.unsqueeze(1).expand(-1, num_neg)
        neg_batch_t = torch.stack([neg_heads_t, neg_rels_t, neg_tails], dim=-1).reshape(-1, 3)
        neg_batch_t = self._filter_known(neg_batch_t, 2, tail_type_ids)
        neg_parts.append(neg_batch_t)

        all_neg = torch.cat(neg_parts, dim=0)
        labels = torch.zeros(all_neg.shape[0], dtype=torch.long, device=self.device)
        return all_neg, labels

    def _filter_known(self, neg_triples: torch.LongTensor, replace_col: int, type_ids: torch.LongTensor) -> torch.LongTensor:
        """GPU-vectorized known-triple filtering using hash keys and torch.isin."""
        if self.known_triples_keys_sorted.numel() == 0:
            return neg_triples

        # Expand type_ids: each source triple has num_negatives entries
        expanded_type_ids = type_ids.repeat_interleave(self.num_negatives)

        for _ in range(2):
            keys = self._compute_triple_keys(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
            pos = torch.searchsorted(self.known_triples_keys_sorted, keys)
            pos = pos.clamp(max=self.known_triples_keys_sorted.size(0) - 1)
            is_known = self.known_triples_keys_sorted[pos] == keys
            if not is_known.any():
                break
            idx = is_known.nonzero(as_tuple=True)[0]
            # Re-sample from type pools for known triples
            re_type_ids = expanded_type_ids[idx]
            pool_sizes = self.type_pool_sizes_tensor[re_type_ids]
            rand_idx = torch.randint(0, self.max_pool_size, (idx.size(0),), generator=self._rng, device=self.device)
            rand_idx = rand_idx % pool_sizes.clamp(min=1)
            neg_triples[idx, replace_col] = self.type_entity_pools_padded[re_type_ids, rand_idx]

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
