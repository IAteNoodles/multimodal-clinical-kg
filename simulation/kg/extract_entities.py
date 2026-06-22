import ast
import json
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged_Cardiomegaly', 'Fracture', 'Lung_Lesion', 'Lung_Opacity',
    'No_Finding', 'Pleural_Effusion', 'Pleural_Other', 'Pneumonia',
    'Pneumothorax', 'Support_Devices',
]

CHEXPERT_CSV_LABELS = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Enlarged_Cardiomegaly': 'Enlarged Cardiomediastinum',
    'Fracture': 'Fracture',
    'Lung_Lesion': 'Lung Lesion',
    'Lung_Opacity': 'Lung Opacity',
    'No_Finding': 'No Finding',
    'Pleural_Effusion': 'Pleural Effusion',
    'Pleural_Other': 'Pleural Other',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Support_Devices': 'Support Devices',
}

CHEXPERT_TO_DISEASE = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'HeartFailure',
    'Consolidation': 'Pneumonia',
    'Edema': 'PulmonaryEdema',
    'Enlarged_Cardiomegaly': 'Cardiomegaly',
    'Fracture': 'Fracture',
    'Lung_Lesion': 'LungLesion',
    'Lung_Opacity': 'LungOpacity',
    'No_Finding': 'Normal',
    'Pleural_Effusion': 'PleuralEffusion',
    'Pleural_Other': 'PleuralDisease',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Support_Devices': 'SupportDevices',
}

ECG_CLASS_TO_DISEASE = {
    'NORM': 'Normal',
    'MI': 'MyocardialInfarction',
    'STTC': 'STTChanges',
    'CD': 'ConductionDisease',
    'HYP': 'Hypertrophy',
}

ECG_SCP_TO_DISEASE = {
    'IMI': 'MyocardialInfarction',
    'AMI': 'MyocardialInfarction',
    'LMI': 'MyocardialInfarction',
    'PMI': 'MyocardialInfarction',
    'AFIB': 'ConductionDisease',
    'AFLT': 'ConductionDisease',
    'SVTAC': 'ConductionDisease',
    'CRBBB': 'ConductionDisease',
    'CLBBB': 'ConductionDisease',
    'IRBBB': 'ConductionDisease',
    'ILBBB': 'ConductionDisease',
    'LAFB': 'ConductionDisease',
    'LPFB': 'ConductionDisease',
    '1AVB': 'ConductionDisease',
    '2AVB': 'ConductionDisease',
    '3AVB': 'ConductionDisease',
    'WPW': 'ConductionDisease',
    'LVH': 'Hypertrophy',
    'RVH': 'Hypertrophy',
    'LAO/LAE': 'Hypertrophy',
    'RAO/RAE': 'Hypertrophy',
    'STACH': 'ConductionDisease',
    'SBRAD': 'ConductionDisease',
    'SBRI': 'STTChanges',
    'NST_': 'STTChanges',
    'DIG': 'STTChanges',
    'LNGQT': 'ConductionDisease',
    'SR': 'SinusRhythm',
    'SA': 'SinusArrhythmia',
    'JER': 'JunctionalRhythm',
    'AJR': 'JunctionalRhythm',
    'AVNRT': 'SupraventricularTachycardia',
    'AVRT': 'SupraventricularTachycardia',
}

CHEXPERT_TO_ANATOMY = {
    'Atelectasis': ['Lung'],
    'Cardiomegaly': ['Heart'],
    'Consolidation': ['Lung'],
    'Edema': ['Lung'],
    'Enlarged_Cardiomegaly': ['Heart', 'Mediastinum'],
    'Fracture': ['Bone'],
    'Lung_Lesion': ['Lung'],
    'Lung_Opacity': ['Lung'],
    'No_Finding': [],
    'Pleural_Effusion': ['Pleura'],
    'Pleural_Other': ['Pleura'],
    'Pneumonia': ['Lung'],
    'Pneumothorax': ['Pleura', 'Lung'],
    'Support_Devices': [],
}

ECG_ANATOMY = ['Atria', 'Ventricles', 'AV_Node']

SCP_TO_ANATOMY = {
    'IMI': ['InferiorWall', 'Ventricles'],
    'AMI': ['AnteriorWall', 'Ventricles'],
    'LMI': ['LateralWall', 'Ventricles'],
    'PMI': ['PosteriorWall', 'Ventricles'],
    'AFIB': ['Atria'],
    'AFLT': ['Atria'],
    'SVTAC': ['Atria'],
    'CRBBB': ['RightBundle', 'Ventricles'],
    'CLBBB': ['LeftBundle', 'Ventricles'],
    '1AVB': ['AV_Node'],
    '2AVB': ['AV_Node'],
    '3AVB': ['AV_Node'],
    'WPW': ['AccessoryPathway'],
    'LVH': ['LeftVentricle'],
    'RVH': ['RightVentricle'],
    'LAO/LAE': ['LeftAtrium'],
    'RAO/RAE': ['RightAtrium'],
}

DISEASE_SUBSUMES = [
    ('MyocardialInfarction', 'HeartFailure'),
    ('ConductionDisease', 'HeartFailure'),
    ('Hypertrophy', 'HeartFailure'),
    ('PulmonaryEdema', 'HeartFailure'),
    ('Cardiomegaly', 'HeartFailure'),
    ('Pneumonia', 'LungDisease'),
    ('Atelectasis', 'LungDisease'),
    ('Pneumothorax', 'LungDisease'),
    ('LungOpacity', 'LungDisease'),
    ('LungLesion', 'LungDisease'),
    ('PleuralEffusion', 'PleuralDisease'),
    ('STTChanges', 'HeartFailure'),
    ('PleuralDisease', 'LungDisease'),
    # Cross-modal disease equivalences: ECG diseases → CXR-relevant parent diseases
    ('MyocardialInfarction', 'Pneumonia'),          # MI can present with consolidation on CXR
    ('STTChanges', 'PulmonaryEdema'),                # STT changes often accompany pulmonary edema
    ('ConductionDisease', 'Cardiomegaly'),            # Conduction disease → cardiomegaly
    ('Hypertrophy', 'Cardiomegaly'),                  # LVH → cardiomegaly on CXR
    ('Hypertrophy', 'PulmonaryEdema'),                # Hypertrophy → pulmonary edema
    ('MyocardialInfarction', 'PleuralEffusion'),      # MI → pleural effusion
    ('ConductionDisease', 'PulmonaryEdema'),          # Conduction disease → pulmonary edema
]

RADGRAPH_TO_CHEXPERT = {
    'effusion': 'Pleural_Effusion',
    'cardiomegaly': 'Cardiomegaly',
    'atelectasis': 'Atelectasis',
    'consolidation': 'Consolidation',
    'edema': 'Edema',
    'pneumonia': 'Pneumonia',
    'pneumothorax': 'Pneumothorax',
    'fracture': 'Fracture',
    'lung opacity': 'Lung_Opacity',
    'lung lesion': 'Lung_Lesion',
    'pleural effusion': 'Pleural_Effusion',
}


def _normalize_anatomy_name(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(r'\s+', '_', normalized)
    parts = normalized.split('_')
    return '_'.join(p.capitalize() for p in parts if p)


@dataclass
class Entity:
    id: str
    type: str
    modality: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    split: Optional[str] = None


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    weight: float = 1.0
    source: Optional[str] = None
    split: Optional[str] = None


@dataclass
class ClinicalKG:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    relation_type_counts: Dict[str, int] = field(default_factory=dict)
    _relation_set: set = field(default_factory=set)

    def add_entity(self, entity: Entity) -> None:
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
            self.entity_type_counts[entity.type] = self.entity_type_counts.get(entity.type, 0) + 1

    def add_relation(self, relation: Relation) -> None:
        key = (relation.head, relation.relation, relation.tail)
        if key in self._relation_set:
            return
        self._relation_set.add(key)
        self.relations.append(relation)
        self.relation_type_counts[relation.relation] = self.relation_type_counts.get(relation.relation, 0) + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> 'ClinicalKG':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_efficient(self, directory: Path) -> None:
        """Save KG as numpy arrays + JSON metadata to avoid pickle deserialization overhead."""
        directory.mkdir(parents=True, exist_ok=True)

        ENTITY_TYPE_TO_ID = {
            "Finding": 0, "Anatomy": 1, "Disease": 2, "Patient": 3,
            "Study": 4, "Drug": 5, "LabResult": 6, "Procedure": 7, "VitalResult": 8,
            "ECGMeasurement": 9, "ECGRhythm": 10, "Unknown": 11,
        }
        MODALITY_TO_ID = {"CXR": 0, "ECG": 1, "RAD": 2, "STR": 3, "ONTOLOGY": 5, None: 6}
        ENTITY_TYPE_ORDER = ["Finding", "Anatomy", "Disease", "Patient", "Study", "Drug", "LabResult", "Procedure", "VitalResult", "ECGMeasurement", "ECGRhythm", "Unknown"]

        # Deterministic entity ordering: sort by (type_order, id) — same as KGTriplesDataset.__init__
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: (ENTITY_TYPE_ORDER.index(e.type) if e.type in ENTITY_TYPE_ORDER else len(ENTITY_TYPE_ORDER), e.id),
        )

        entity_id_map: Dict[str, int] = {e.id: i for i, e in enumerate(sorted_entities)}
        entity_labels: List[str] = [e.label for e in sorted_entities]
        entity_type_ids = [ENTITY_TYPE_TO_ID.get(e.type, ENTITY_TYPE_TO_ID["Unknown"]) for e in sorted_entities]
        entity_modality_ids = [MODALITY_TO_ID.get(e.modality if e.modality and e.modality != "None" else None, len(MODALITY_TO_ID) - 1) for e in sorted_entities]

        entities_arr = np.array(
            [list(range(len(sorted_entities))), entity_type_ids, entity_modality_ids, list(range(len(sorted_entities)))],
            dtype=np.int32,
        ).T  # shape (N, 4): [entity_id_int, type_id, modality_id, label_idx]

        # Relations
        all_relation_names = sorted(set(r.relation for r in self.relations))
        relation2id = {r: i for i, r in enumerate(all_relation_names)}

        rel_data = []
        for rel in self.relations:
            if rel.head in entity_id_map and rel.tail in entity_id_map:
                rel_data.append((entity_id_map[rel.head], relation2id[rel.relation], entity_id_map[rel.tail], rel.weight))

        if rel_data:
            relations_arr = np.array(rel_data, dtype=np.float32)
            relations_arr[:, :3] = relations_arr[:, :3].astype(np.int32)  # head, rel, tail as int
            # Store as structured: head(int32), relation(int32), tail(int32), weight(float32)
            relations_out = np.zeros(len(rel_data), dtype=[('head', np.int32), ('relation', np.int32), ('tail', np.int32), ('weight', np.float32)])
            relations_out['head'] = relations_arr[:, 0].astype(np.int32)
            relations_out['relation'] = relations_arr[:, 1].astype(np.int32)
            relations_out['tail'] = relations_arr[:, 2].astype(np.int32)
            relations_out['weight'] = relations_arr[:, 3].astype(np.float32)
        else:
            relations_out = np.zeros(0, dtype=[('head', np.int32), ('relation', np.int32), ('tail', np.int32), ('weight', np.float32)])

        # Metadata
        metadata = {
            "entity_id_map": entity_id_map,
            "entity_labels": entity_labels,
            "relation_names": all_relation_names,
            "entity_type_names": [ENTITY_TYPE_ORDER[i] if i < len(ENTITY_TYPE_ORDER) else "Unknown" for i in range(len(ENTITY_TYPE_ORDER))],
            "modality_names": ["CXR", "ECG", "RAD", "STR", "None"],
            "entity_type_counts": self.entity_type_counts,
            "relation_type_counts": self.relation_type_counts,
        }

        np.save(directory / "entities.npy", entities_arr)
        np.save(directory / "relations.npy", relations_out)
        with open(directory / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"  Saved efficient format to {directory}/")
        print(f"    entities.npy: {entities_arr.shape}, relations.npy: {relations_out.shape}")

    @classmethod
    def load_efficient(cls, directory: Path) -> 'ClinicalKG':
        """Reconstruct ClinicalKG from efficient format (for debugging/backward compat)."""
        entities_arr = np.load(directory / "entities.npy")
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)

        entity_id_map = metadata["entity_id_map"]  # str -> int
        id2entity_str = {v: k for k, v in entity_id_map.items()}
        entity_labels = metadata["entity_labels"]
        entity_type_names = metadata["entity_type_names"]
        modality_names = metadata["modality_names"]

        ENTITY_ID_TO_TYPE = {i: name for i, name in enumerate(entity_type_names)}
        MODALITY_ID_TO_NAME = {i: name for i, name in enumerate(modality_names)}

        kg = cls()
        for i in range(entities_arr.shape[0]):
            eid_str = id2entity_str[int(entities_arr[i, 0])]
            type_id = int(entities_arr[i, 1])
            mod_id = int(entities_arr[i, 2])
            label_idx = int(entities_arr[i, 3])
            etype = ENTITY_ID_TO_TYPE.get(type_id, "Unknown")
            modality = MODALITY_ID_TO_NAME.get(mod_id, None)
            if modality == "None":
                modality = None
            label = entity_labels[label_idx]
            entity = Entity(id=eid_str, type=etype, modality=modality, label=label)
            kg.entities[eid_str] = entity
            kg.entity_type_counts[etype] = kg.entity_type_counts.get(etype, 0) + 1

        relations_arr = np.load(directory / "relations.npy")
        relation_names = metadata["relation_names"]
        for i in range(relations_arr.shape[0]):
            head_str = id2entity_str[int(relations_arr[i]['head'])]
            rel_name = relation_names[int(relations_arr[i]['relation'])]
            tail_str = id2entity_str[int(relations_arr[i]['tail'])]
            weight = float(relations_arr[i]['weight'])
            kg.relations.append(Relation(head=head_str, relation=rel_name, tail=tail_str, weight=weight))
            kg.relation_type_counts[rel_name] = kg.relation_type_counts.get(rel_name, 0) + 1

        return kg


def _add_entity_if_new(kg: ClinicalKG, eid: str, etype: str, modality: str, label: str,
                       properties: Optional[Dict[str, Any]] = None, split: Optional[str] = None) -> Entity:
    if eid in kg.entities:
        existing = kg.entities[eid]
        if split and existing.split and existing.split != split:
            existing.split = "shared"
        return existing
    entity = Entity(id=eid, type=etype, modality=modality, label=label, properties=properties or {}, split=split)
    kg.add_entity(entity)
    return entity


def extract_chexpert_findings(kg: ClinicalKG, data_dir: Path) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    chexpert_path = data_dir / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-chexpert.csv"
    if not chexpert_path.exists():
        print("  CheXpert labels not found, skipping CXR finding extraction")
        return pd.DataFrame(), {}

    print("Loading CheXpert labels...")
    df = pd.read_csv(chexpert_path)
    df = df.dropna(subset=['subject_id'])
    df['subject_id'] = df['subject_id'].astype(int)
    if 'study_id' in df.columns:
        df['study_id'] = df['study_id'].where(df['study_id'].notna()).astype('Int64')

    # Melt label columns
    label_cols = {label: csv_col for label, csv_col in zip(CHEXPERT_LABELS, [CHEXPERT_CSV_LABELS[l] for l in CHEXPERT_LABELS]) if csv_col in df.columns}
    if not label_cols:
        print("  No CheXpert label columns found")
        return df, {}

    melt_cols = list(label_cols.values())
    melted = df.melt(id_vars=['subject_id', 'study_id'], value_vars=melt_cols, var_name='csv_col', value_name='val')
    melted = melted[melted['val'].notna() & (melted['val'] != 0.0)]
    melted['weight'] = melted['val'].apply(lambda v: 1.0 if v == 1.0 else 0.5)

    # Reverse map csv_col → label name
    col_to_label = {v: k for k, v in label_cols.items()}
    melted['label'] = melted['csv_col'].map(col_to_label)

    print(f"  CheXpert: {len(melted)} positive labels across {melted['subject_id'].nunique()} patients")

    # Add unique entities
    unique_subjects = melted['subject_id'].unique()
    for sid in unique_subjects:
        _add_entity_if_new(kg, f"PAT_{sid}", "Patient", "CXR", str(sid))

    if 'study_id' in melted.columns:
        valid_studies = melted['study_id'].dropna().unique()
        for sty in valid_studies:
            _add_entity_if_new(kg, f"STY_{int(sty)}", "Study", "CXR", str(int(sty)))

    unique_labels = melted['label'].unique()
    for lbl in unique_labels:
        _add_entity_if_new(kg, f"FND_CXR_{lbl}", "Finding", "CXR", lbl)

    # Add relations — still loop but only over positive labels (much smaller than full df)
    patient_findings: Dict[int, List[str]] = defaultdict(list)
    for _, row in melted.iterrows():
        subject_id = int(row['subject_id'])
        study_id = row.get('study_id')
        label = row['label']
        weight = row['weight']
        fnd_id = f"FND_CXR_{label}"

        kg.add_relation(Relation(head=f"PAT_{subject_id}", relation="has_finding", tail=fnd_id, weight=weight))
        patient_findings[subject_id].append(label)

        if pd.notna(study_id):
            kg.add_relation(Relation(head=fnd_id, relation="finding_of", tail=f"STY_{int(study_id)}", weight=1.0))

        disease_name = CHEXPERT_TO_DISEASE.get(label)
        if disease_name:
            dis_id = f"DIS_{disease_name}"
            _add_entity_if_new(kg, dis_id, "Disease", "CXR", disease_name)

        for anat in CHEXPERT_TO_ANATOMY.get(label, []):
            anat_id = f"ANAT_{_normalize_anatomy_name(anat)}"
            _add_entity_if_new(kg, anat_id, "Anatomy", "CXR", _normalize_anatomy_name(anat))
            kg.add_relation(Relation(head=fnd_id, relation="located_at", tail=anat_id, weight=1.0))

    print(f"  Extracted {kg.entity_type_counts.get('Finding', 0)} CXR findings, "
          f"{len(patient_findings)} patients with findings")
    return df, patient_findings


def extract_ptbxl_findings(kg: ClinicalKG, data_dir: Path) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    ptbxl_db_path = (data_dir / "ptb_xl" /
                     "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" /
                     "ptbxl_database.csv")
    scp_path = (data_dir / "ptb_xl" /
                "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" /
                "scp_statements.csv")

    if not ptbxl_db_path.exists():
        print("  PTB-XL database not found, skipping ECG finding extraction")
        return pd.DataFrame(), {}

    print("Loading PTB-XL metadata...")
    df = pd.read_csv(ptbxl_db_path, index_col=0)

    scp_df = None
    if scp_path.exists():
        scp_df = pd.read_csv(scp_path, index_col=0)

    scp_to_class: Dict[str, str] = {}
    if scp_df is not None and 'diagnostic_class' in scp_df.columns:
        for code, row in scp_df.iterrows():
            cls = row.get('diagnostic_class')
            if pd.notna(cls):
                scp_to_class[str(code)] = str(cls)

    rhythm_codes: set = set()
    if scp_df is not None and 'diagnostic_class' not in scp_df.columns:
        if 'rhythm' in scp_df.columns:
            for code, row in scp_df.iterrows():
                if pd.notna(row.get('rhythm')):
                    rhythm_codes.add(str(code))

    patient_findings: Dict[int, List[str]] = defaultdict(list)

    for i, (idx, row) in enumerate(df.iterrows()):
        if i > 0 and i % 10000 == 0:
            print(f"    PTB-XL: processed {i}/{len(df)} rows...")
        ecg_id = int(idx)
        patient_id = int(row.get('patient_id', ecg_id))

        _add_entity_if_new(kg, f"PAT_PTB{patient_id}", "Patient", "ECG", str(patient_id))
        _add_entity_if_new(kg, f"STY_PTBECG{ecg_id}", "Study", "ECG", str(ecg_id))

        scp_str = row.get('scp_codes', '{}')
        try:
            scp_codes = ast.literal_eval(scp_str) if isinstance(scp_str, str) else {}
        except (ValueError, SyntaxError):
            scp_codes = {}

        for code, prob in scp_codes.items():
            code_str = str(code).strip()
            diag_class = scp_to_class.get(code_str)

            if diag_class:
                fnd_id = f"FND_ECG_{code_str}"
                _add_entity_if_new(kg, fnd_id, "Finding", "ECG", code_str,
                                   properties={'diagnostic_class': diag_class, 'probability': prob})

                kg.add_relation(Relation(head=f"PAT_PTB{patient_id}", relation="has_finding",
                                         tail=fnd_id, weight=float(prob)))
                kg.add_relation(Relation(head=fnd_id, relation="finding_of",
                                         tail=f"STY_PTBECG{ecg_id}", weight=1.0))
                patient_findings[patient_id].append(code_str)

                disease_name = ECG_CLASS_TO_DISEASE.get(diag_class)
                if disease_name:
                    dis_id = f"DIS_{disease_name}"
                    _add_entity_if_new(kg, dis_id, "Disease", "ECG", disease_name)
                    kg.add_relation(Relation(head=fnd_id, relation="indicates",
                                             tail=dis_id, weight=1.0))

                # Also map individual SCP codes to diseases
                scp_disease_name = ECG_SCP_TO_DISEASE.get(code_str)
                if scp_disease_name:
                    scp_dis_id = f"DIS_{scp_disease_name}"
                    _add_entity_if_new(kg, scp_dis_id, "Disease", "ECG", scp_disease_name)

                anats = SCP_TO_ANATOMY.get(code_str, [])
                for anat in anats:
                    anat_id = f"ANAT_{_normalize_anatomy_name(anat)}"
                    _add_entity_if_new(kg, anat_id, "Anatomy", "ECG", _normalize_anatomy_name(anat))
                    kg.add_relation(Relation(head=fnd_id, relation="located_at",
                                             tail=anat_id, weight=1.0))

            elif code_str in ECG_SCP_TO_DISEASE:
                fnd_id = f"FND_ECG_{code_str}"
                _add_entity_if_new(kg, fnd_id, "Finding", "ECG", code_str,
                                   properties={'rhythm': True, 'probability': prob})

                kg.add_relation(Relation(head=f"PAT_PTB{patient_id}", relation="has_finding",
                                         tail=fnd_id, weight=float(prob)))
                kg.add_relation(Relation(head=fnd_id, relation="finding_of",
                                         tail=f"STY_PTBECG{ecg_id}", weight=1.0))
                patient_findings[patient_id].append(code_str)

                anats = SCP_TO_ANATOMY.get(code_str, [])
                for anat in anats:
                    anat_id = f"ANAT_{_normalize_anatomy_name(anat)}"
                    _add_entity_if_new(kg, anat_id, "Anatomy", "ECG", _normalize_anatomy_name(anat))
                    kg.add_relation(Relation(head=fnd_id, relation="located_at",
                                             tail=anat_id, weight=1.0))

                # Map rhythm SCP codes to diseases via ECG_SCP_TO_DISEASE
                scp_disease_name = ECG_SCP_TO_DISEASE.get(code_str)
                if scp_disease_name:
                    scp_dis_id = f"DIS_{scp_disease_name}"
                    _add_entity_if_new(kg, scp_dis_id, "Disease", "ECG", scp_disease_name)

    for anat in ECG_ANATOMY:
        _add_entity_if_new(kg, f"ANAT_{_normalize_anatomy_name(anat)}", "Anatomy", "ECG", _normalize_anatomy_name(anat))

    print(f"  Extracted {sum(1 for e in kg.entities.values() if e.type == 'Finding' and e.modality == 'ECG')} ECG findings, "
          f"{len(patient_findings)} PTB-XL patients with findings")
    return df, patient_findings


def extract_radgraph_findings(kg: ClinicalKG, data_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    radgraph_dir = (data_dir / "radgraph" /
                    "radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0")
    if not radgraph_dir.exists():
        print("  RadGraph data not found, skipping RadGraph extraction")
        return {}

    print("Loading RadGraph data...")
    all_annotations: Dict[str, Dict[str, List[Dict]]] = {}

    for split_file in ['train.json', 'dev.json', 'test.json']:
        fpath = radgraph_dir / split_file
        if not fpath.exists():
            continue
        print(f"  Parsing {split_file}...")
        with open(fpath, 'r') as f:
            data = json.load(f)

        split_name = split_file.replace('.json', '')
        all_annotations.setdefault(split_name, {})

        for doc_key, doc_data in data.items():
            entities_list = []
            relations_list = []

            if split_file == 'test.json' and 'labeler_1' in doc_data:
                doc_data = doc_data['labeler_1']

            doc_entities = doc_data.get('entities', {})
            for ent_id, ent in doc_entities.items():
                entities_list.append({
                    'id': ent_id,
                    'text': ent.get('tokens', '').lower(),
                    'type': ent.get('label', ''),
                    'start': ent.get('start', 0),
                    'end': ent.get('end', 0),
                })

                if ent.get('label') in ('OBS-DP', 'OBS-DA'):
                    fnd_id = f"FND_RAD_{ent['tokens'].lower()}"
                    _add_entity_if_new(kg, fnd_id, "Finding", "RAD", ent['tokens'].lower(), split=split_name)

                elif ent.get('label') == 'ANAT-DP':
                    anat_id = f"ANAT_{_normalize_anatomy_name(ent['tokens'])}"
                    _add_entity_if_new(kg, anat_id, "Anatomy", "RAD", _normalize_anatomy_name(ent['tokens']), split=split_name)

            doc_relations = doc_data.get('relations', {})
            for rel_id, rel in doc_relations.items():
                rel_type = rel.get('type', '')
                obj1 = rel.get('object1', '')
                obj2 = rel.get('object2', '')
                relations_list.append({
                    'type': rel_type,
                    'object1': obj1,
                    'object2': obj2,
                })

                if rel_type == 'located_at' and obj1 in doc_entities and obj2 in doc_entities:
                    src_ent = doc_entities[obj1]
                    tgt_ent = doc_entities[obj2]
                    if src_ent.get('label') in ('OBS-DP', 'OBS-DA') and tgt_ent.get('label') == 'ANAT-DP':
                        fnd_id = f"FND_RAD_{src_ent['tokens'].lower()}"
                        anat_id = f"ANAT_{_normalize_anatomy_name(tgt_ent['tokens'])}"
                        kg.add_relation(Relation(head=fnd_id, relation="located_at",
                                                 tail=anat_id, weight=1.0, split=split_name))

                elif rel_type == 'modify' and obj1 in doc_entities and obj2 in doc_entities:
                    src_ent = doc_entities[obj1]
                    tgt_ent = doc_entities[obj2]
                    if src_ent.get('label') in ('OBS-DP', 'OBS-DA') and tgt_ent.get('label') in ('OBS-DP', 'OBS-DA'):
                        src_id = f"FND_RAD_{src_ent['tokens'].lower()}"
                        tgt_id = f"FND_RAD_{tgt_ent['tokens'].lower()}"
                        kg.add_relation(Relation(head=src_id, relation="modifies",
                                                 tail=tgt_id, weight=1.0, split=split_name))

            all_annotations[split_name][doc_key] = entities_list + relations_list

    for graph_file in ['MIMIC-CXR_graphs.json', 'CheXpert_graphs.json']:
        fpath = radgraph_dir / graph_file
        if not fpath.exists():
            continue
        print(f"  Parsing {graph_file}...")
        with open(fpath, 'r') as f:
            data = json.load(f)

        for doc_key, doc_data in data.items():
            if 'labeler_1' in doc_data:
                doc_data = doc_data['labeler_1']

            doc_entities = doc_data.get('entities', {})
            for ent_id, ent in doc_entities.items():
                if ent.get('label') in ('OBS-DP', 'OBS-DA'):
                    fnd_id = f"FND_RAD_{ent['tokens'].lower()}"
                    _add_entity_if_new(kg, fnd_id, "Finding", "RAD", ent['tokens'].lower())
                elif ent.get('label') == 'ANAT-DP':
                    anat_id = f"ANAT_{_normalize_anatomy_name(ent['tokens'])}"
                    _add_entity_if_new(kg, anat_id, "Anatomy", "RAD", _normalize_anatomy_name(ent['tokens']))

    print(f"  RadGraph: {sum(1 for e in kg.entities.values() if e.type == 'Finding' and e.modality == 'RAD')} findings, "
          f"{sum(1 for e in kg.entities.values() if e.type == 'Anatomy' and e.modality == 'RAD')} anatomy nodes")
    return all_annotations


def link_radgraph_to_cxr(kg: ClinicalKG, data_dir: Path) -> None:
    radgraph_dir = (data_dir / "radgraph" /
                    "radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0")
    mimic_graph_path = radgraph_dir / "MIMIC-CXR_graphs.json"
    if not mimic_graph_path.exists():
        return

    print("Linking RadGraph findings to MIMIC-CXR studies...")
    with open(mimic_graph_path, 'r') as f:
        data = json.load(f)

    linked = 0
    for doc_key, doc_data in data.items():
        if 'labeler_1' in doc_data:
            doc_data = doc_data['labeler_1']

        try:
            parts = doc_key.replace('.txt', '').split('/')
            subject_part = [p for p in parts if p.startswith('p') and p[1:].isdigit()]
            study_part = [p for p in parts if p.startswith('s') and p[1:].isdigit()]
            if not subject_part or not study_part:
                continue
            subject_id = int(subject_part[-1][1:])
            study_id = int(study_part[-1][1:])
        except (ValueError, IndexError):
            continue

        study_id_str = f"STY_{study_id}"
        if study_id_str not in kg.entities:
            continue

        doc_entities = doc_data.get('entities', {})
        for ent_id, ent in doc_entities.items():
            if ent.get('label') in ('OBS-DP', 'OBS-DA'):
                fnd_id = f"FND_RAD_{ent['tokens'].lower()}"
                if fnd_id in kg.entities:
                    kg.add_relation(Relation(head=fnd_id, relation="finding_of",
                                             tail=study_id_str, weight=1.0))

                    lower_text = ent['tokens'].lower()
                    chexpert_match = RADGRAPH_TO_CHEXPERT.get(lower_text)
                    if chexpert_match:
                        cxr_fnd_id = f"FND_CXR_{chexpert_match}"
                        if cxr_fnd_id in kg.entities:
                            kg.add_relation(Relation(head=fnd_id, relation="modifies",
                                                     tail=cxr_fnd_id, weight=1.0))
                    linked += 1

    print(f"  Linked {linked} RadGraph findings to MIMIC-CXR studies")


def build_disease_subsumption(kg: ClinicalKG) -> None:
    print("Building disease subsumption edges...")
    for child, parent in DISEASE_SUBSUMES:
        child_id = f"DIS_{child}"
        parent_id = f"DIS_{parent}"
        _add_entity_if_new(kg, child_id, "Disease", "ONTOLOGY", child)
        _add_entity_if_new(kg, parent_id, "Disease", "ONTOLOGY", parent)
        kg.add_relation(Relation(head=parent_id, relation="subsumes",
                                 tail=child_id, weight=1.0))


def build_same_patient_edges(kg: ClinicalKG, chexpert_df: pd.DataFrame, data_dir: Path) -> None:
    if chexpert_df.empty:
        return

    print("Building same_patient edges from MIMIC-CXR...")
    patient_studies: Dict[int, List[str]] = defaultdict(list)
    for _, row in chexpert_df.iterrows():
        subject_id = row.get('subject_id')
        study_id = row.get('study_id')
        if pd.notna(subject_id) and pd.notna(study_id):
            sty_id = f"STY_{int(study_id)}"
            if sty_id in kg.entities:
                patient_studies[int(subject_id)].append(sty_id)

    count = 0
    for sid, studies in patient_studies.items():
        studies = studies[:20]
        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                kg.add_relation(Relation(head=studies[i], relation="same_patient",
                                         tail=studies[j], weight=1.0))
                count += 1
    print(f"  Added {count} same_patient edges from CXR")

    finding_to_patient: Dict[str, str] = {}
    for rel in kg.relations:
        if rel.relation == "has_finding":
            finding_to_patient[rel.tail] = rel.head

    ecg_patient_studies: Dict[str, List[str]] = defaultdict(list)
    for rel in kg.relations:
        if rel.relation == "finding_of" and rel.tail.startswith("STY_PTBECG"):
            patient = finding_to_patient.get(rel.head)
            if patient is not None:
                ecg_patient_studies[patient].append(rel.tail)

    ecg_count = 0
    for pat_id, studies in ecg_patient_studies.items():
        studies = list(dict.fromkeys(studies))[:20]
        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                kg.add_relation(Relation(head=studies[i], relation="same_patient",
                                         tail=studies[j], weight=1.0))
                ecg_count += 1
    print(f"  Added {ecg_count} same_patient edges from ECG")


CLINICAL_CROSS_MODAL_EDGES = {
    # Original clinically-validated pairs
    ('Cardiomegaly', 'LVH'): 0.8,
    ('Cardiomegaly', 'LAO/LAE'): 0.7,
    ('Cardiomegaly', 'RVH'): 0.6,
    ('Edema', 'AFIB'): 0.6,
    ('Edema', 'STACH'): 0.5,
    ('Pleural_Effusion', 'AFIB'): 0.7,
    ('Pleural_Effusion', 'RVH'): 0.5,
    ('Consolidation', 'STACH'): 0.5,
    ('Pneumonia', 'STACH'): 0.5,
    ('Atelectasis', 'AFIB'): 0.4,
    ('Enlarged_Cardiomegaly', 'LVH'): 0.8,
    ('Enlarged_Cardiomegaly', 'LAFB'): 0.5,
    ('Enlarged_Cardiomegaly', 'CLBBB'): 0.6,
    ('Enlarged_Cardiomegaly', 'CRBBB'): 0.4,
    ('Enlarged_Cardiomegaly', 'LAO/LAE'): 0.7,
    ('No_Finding', 'NORM'): 0.9,
    ('No_Finding', 'SR'): 0.9,
    ('Lung_Opacity', 'STACH'): 0.4,
    ('Pneumothorax', 'STACH'): 0.3,
    ('Fracture', 'AFIB'): 0.3,
    ('Lung_Lesion', 'STACH'): 0.4,
    ('Cardiomegaly', 'CLBBB'): 0.6,
    ('Cardiomegaly', 'IRBBB'): 0.3,
    ('Edema', 'LVH'): 0.6,
    ('Edema', 'CLBBB'): 0.5,
    ('Edema', 'LAFB'): 0.4,
    ('Pleural_Effusion', 'LVH'): 0.6,
    ('Pleural_Effusion', 'CLBBB'): 0.5,
    # Additional clinically-validated pairs
    ('Cardiomegaly', 'LAFB'): 0.4,
    ('Cardiomegaly', 'RAO/RAE'): 0.3,
    ('Cardiomegaly', 'CRBBB'): 0.3,
    ('Cardiomegaly', 'LPFB'): 0.3,
    ('Cardiomegaly', 'AFLT'): 0.3,
    ('Enlarged_Cardiomegaly', 'IRBBB'): 0.2,
    ('Edema', 'STTC'): 0.5,
    ('Edema', 'AFLT'): 0.4,
    ('Edema', 'SBRI'): 0.3,
    ('Pleural_Effusion', 'STTC'): 0.3,
    ('Pleural_Effusion', 'AFLT'): 0.3,
    ('Pleural_Effusion', 'SBRI'): 0.2,
    ('Consolidation', 'MI'): 0.6,
    ('Consolidation', 'STTC'): 0.4,
    ('Consolidation', 'LVH'): 0.2,
    ('Lung_Opacity', 'STTC'): 0.4,
    ('Lung_Opacity', 'MI'): 0.3,
    ('Lung_Opacity', 'AFIB'): 0.2,
    ('Lung_Opacity', 'SBRI'): 0.2,
    ('Lung_Lesion', 'STTC'): 0.2,
    ('Lung_Lesion', 'MI'): 0.2,
    ('Atelectasis', 'STTC'): 0.3,
    ('Atelectasis', 'SBRI'): 0.2,
    ('Pneumonia', 'STTC'): 0.4,
    ('Pneumonia', 'SBRI'): 0.3,
    ('Pneumothorax', 'STTC'): 0.3,
    ('Pneumothorax', 'SBRI'): 0.2,
    ('Pneumothorax', 'AFIB'): 0.2,
    ('Fracture', 'SBRI'): 0.1,
    ('Support_Devices', 'AFIB'): 0.2,
    ('Support_Devices', 'SBRI'): 0.2,
    ('Support_Devices', 'STACH'): 0.2,
    # RadGraph finding ↔ ECG mappings (keys are full RAD entity IDs)
    ('FND_RAD_cardiomegaly', 'LVH'): 0.6,
    ('FND_RAD_cardiomegaly', 'LAFB'): 0.3,
    ('FND_RAD_cardiomegaly', 'LAO/LAE'): 0.4,
    ('FND_RAD_pleural effusion', 'AFIB'): 0.3,
    ('FND_RAD_pleural effusion', 'STTC'): 0.3,
    ('FND_RAD_pulmonary edema', 'AFIB'): 0.4,
    ('FND_RAD_pulmonary edema', 'STTC'): 0.4,
    ('FND_RAD_pulmonary edema', 'LVH'): 0.4,
    ('FND_RAD_consolidation', 'MI'): 0.4,
    ('FND_RAD_consolidation', 'STTC'): 0.3,
    ('FND_RAD_atelectasis', 'STTC'): 0.2,
}


def build_cross_modal_edges(kg: ClinicalKG, chexpert_df: pd.DataFrame, data_dir: Path) -> None:
    print("Building cross-modal edges...")
    ecg_dir = data_dir / "mimic_iv_ecg_demo"

    # --- Extract MIMIC-IV-ECG subject IDs from directory structure and CSV files ---
    ecg_subjects: set = set()
    if ecg_dir.exists():
        import os
        # Walk the full directory tree to find p{subject_id} directories
        # (they live under waveforms/files/, not at the top level)
        for root, dirs, files in os.walk(ecg_dir):
            for d in dirs:
                if d.startswith('p') and d[1:].isdigit():
                    ecg_subjects.add(int(d[1:]))

        # Also check CSV files for subject_id columns
        for candidate in ['record_list.csv', 'machine_measurements.csv', 'metadata.csv']:
            candidate_path = ecg_dir / candidate
            if candidate_path.exists():
                try:
                    ecg_meta = pd.read_csv(candidate_path)
                    if 'subject_id' in ecg_meta.columns:
                        for sid in ecg_meta['subject_id'].unique():
                            ecg_subjects.add(int(sid))
                except Exception:
                    pass

    # --- Build direct lookup from ECG patient entity IDs to their findings ---
    # (replaces O(n^2) scan of all relations)
    ecg_patient_to_findings: Dict[str, List[str]] = defaultdict(list)
    for rel in kg.relations:
        if rel.relation == "has_finding":
            tail_entity = kg.entities.get(rel.tail)
            if tail_entity and tail_entity.type == "Finding" and tail_entity.modality == "ECG":
                head_entity = kg.entities.get(rel.head)
                if head_entity and head_entity.modality == "ECG":
                    ecg_patient_to_findings[rel.head].append(tail_entity.label)

    # --- Determine which MIMIC-IV-ECG subjects have ECG diagnostic labels ---
    # MIMIC-IV-ECG demo data only has waveform files; check for any label file
    mimic_ecg_has_labels = False
    if ecg_dir.exists():
        label_candidates = ['machine_measurements.csv', 'diagnostic_labels.csv',
                            'scp_codes.csv', 'labels.csv']
        for candidate in label_candidates:
            if (ecg_dir / candidate).exists():
                mimic_ecg_has_labels = True
                break

    # --- Check for overlapping patients between CXR and ECG ---
    # PTB-XL patients use entity IDs PAT_PTB{patient_id} — these NEVER overlap with MIMIC subjects
    # MIMIC-IV-ECG patients use entity IDs PAT_{subject_id} — these CAN overlap with MIMIC-CXR
    # We only compute PMI for MIMIC-IV-ECG subjects that overlap with CXR subjects

    if chexpert_df.empty or not ecg_subjects:
        print("  No overlapping patients found for cross-modal edges")
        _add_clinical_knowledge_edges(kg)
        _add_disease_mediated_edges(kg)
        return

    cxr_subjects = set(chexpert_df['subject_id'].dropna().astype(int).unique())
    overlap_subjects = cxr_subjects & ecg_subjects
    print(f"  Found {len(overlap_subjects)} overlapping patients between CXR and ECG")

    if not overlap_subjects:
        _add_clinical_knowledge_edges(kg)
        _add_disease_mediated_edges(kg)
        return

    # --- Collect CXR findings per overlapping patient ---
    cxr_findings_by_patient: Dict[int, List[str]] = defaultdict(list)
    for i, (_, row) in enumerate(chexpert_df.iterrows()):
        if i > 0 and i % 10000 == 0:
            print(f"    CXR findings: processed {i}/{len(chexpert_df)} rows...")
        subject_id = row.get('subject_id')
        if pd.isna(subject_id):
            continue
        sid = int(subject_id)
        if sid not in overlap_subjects:
            continue
        for label in CHEXPERT_LABELS:
            csv_col = CHEXPERT_CSV_LABELS[label]
            if csv_col not in chexpert_df.columns:
                continue
            val = row.get(csv_col)
            if pd.notna(val) and val == 1.0:
                cxr_findings_by_patient[sid].append(label)

    # --- PMI computation for overlapping MIMIC-IV-ECG patients ---
    pmi_edges = 0

    if not mimic_ecg_has_labels:
        # No ECG labels for MIMIC-IV-ECG patients; cannot compute PMI
        print("  No ECG labels for MIMIC-IV-ECG patients; using clinical knowledge fallback only")
    else:
        # Build co-occurrence from overlapping patients
        # For MIMIC-IV-ECG subjects, the entity ID is PAT_{sid}
        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        for sid in overlap_subjects:
            cxr_fnds = set(cxr_findings_by_patient.get(sid, []))
            ecg_fnds = set(ecg_patient_to_findings.get(f"PAT_{sid}", []))
            for cxr_f in cxr_fnds:
                for ecg_f in ecg_fnds:
                    cooccurrence[(f"FND_CXR_{cxr_f}", f"FND_ECG_{ecg_f}")] += 1

        total_patients = len(overlap_subjects)
        for (cxr_id, ecg_id), count in cooccurrence.items():
            if count < 1:
                continue
            p_joint = count / total_patients
            cxr_count = sum(1 for sid in overlap_subjects if cxr_id.replace("FND_CXR_", "") in cxr_findings_by_patient.get(sid, []))
            ecg_count = sum(1 for sid in overlap_subjects if ecg_id.replace("FND_ECG_", "") in ecg_patient_to_findings.get(f"PAT_{sid}", []))
            p_cxr = max(cxr_count / total_patients, 1e-10)
            p_ecg = max(ecg_count / total_patients, 1e-10)
            pmi = math.log(p_joint / (p_cxr * p_ecg)) if p_joint > 0 else 0.0

            if pmi > 0.5:
                norm_pmi = max(0.0, min(1.0, pmi / 3.0))
                if cxr_id in kg.entities and ecg_id in kg.entities:
                    src, tgt = (cxr_id, ecg_id) if cxr_id <= ecg_id else (ecg_id, cxr_id)
                    kg.add_relation(Relation(head=src, relation="suggestive_of",
                                             tail=tgt, weight=norm_pmi))
                    pmi_edges += 1

        print(f"  Added {pmi_edges} suggestive_of edges (PMI-based)")

    # --- Clinical knowledge fallback ---
    _add_clinical_knowledge_edges(kg)
    _add_disease_mediated_edges(kg)


def _add_clinical_knowledge_edges(kg: ClinicalKG) -> None:
    """Add curated CXR/RAD<->ECG finding associations when no PMI-based edge exists."""
    # Pre-compute existing suggestive_of pairs to avoid duplicates
    existing_suggestive: set = set()
    for rel in kg.relations:
        if rel.relation == "suggestive_of":
            existing_suggestive.add((rel.head, rel.tail))

    clinical_edges_added = 0
    for (source_label, ecg_scp), weight in CLINICAL_CROSS_MODAL_EDGES.items():
        # Determine the CXR or RAD entity ID from the key
        if source_label.startswith("FND_RAD_"):
            source_id = source_label
        else:
            source_id = f"FND_CXR_{source_label}"
        ecg_id = f"FND_ECG_{ecg_scp}"
        if source_id not in kg.entities or ecg_id not in kg.entities:
            continue
        # Skip if a suggestive_of edge already exists in either direction
        if (source_id, ecg_id) in existing_suggestive or (ecg_id, source_id) in existing_suggestive:
            continue
        src, tgt = (source_id, ecg_id) if source_id <= ecg_id else (ecg_id, source_id)
        kg.add_relation(Relation(head=src, relation="suggestive_of",
                                 tail=tgt, weight=weight))
        existing_suggestive.add((src, tgt))
        clinical_edges_added += 1

    print(f"  Added {clinical_edges_added} suggestive_of edges (clinical knowledge fallback)")


def _add_disease_mediated_edges(kg: ClinicalKG) -> None:
    """Add suggestive_of edges between findings that share a common disease (or ancestor disease)."""
    # Build disease hierarchy from subsumes relations (child -> parents)
    disease_children: Dict[str, Set[str]] = defaultdict(set)
    disease_parents: Dict[str, Set[str]] = defaultdict(set)
    for rel in kg.relations:
        if rel.relation == "subsumes":
            disease_children[rel.tail].add(rel.head)  # tail is parent, head is child
            disease_parents[rel.head].add(rel.tail)

    # Transitive closure: for each disease, find all ancestors
    def get_ancestors(disease: str, visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set()
        for parent in disease_parents.get(disease, set()):
            if parent not in visited:
                visited.add(parent)
                get_ancestors(parent, visited)
        return visited

    # Build disease -> {modality: [finding_ids]} mapping from indicates relations
    disease_to_findings: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for rel in kg.relations:
        if rel.relation == "indicates":
            head_name = rel.head
            if head_name.startswith("FND_CXR_"):
                disease_to_findings[rel.tail]["CXR"].append(head_name)
            elif head_name.startswith("FND_ECG_"):
                disease_to_findings[rel.tail]["ECG"].append(head_name)
            elif head_name.startswith("FND_RAD_"):
                disease_to_findings[rel.tail]["RAD"].append(head_name)

    # Expand: also map findings to ancestor diseases
    expanded: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for disease, mod_findings in disease_to_findings.items():
        for mod, findings in mod_findings.items():
            expanded[disease][mod].extend(findings)
        # Add findings to all ancestor diseases too
        for ancestor in get_ancestors(disease):
            for mod, findings in mod_findings.items():
                expanded[ancestor][mod].extend(findings)

    # Build existing suggestive_of edge set for dedup
    existing_edges: set = set()
    for rel in kg.relations:
        if rel.relation == "suggestive_of":
            existing_edges.add((rel.head, rel.tail))

    modality_pairs = [("CXR", "ECG"), ("CXR", "RAD"), ("ECG", "RAD")]
    new_edges = 0

    for disease_entity_id, mod_findings in expanded.items():
        for mod1, mod2 in modality_pairs:
            if mod1 in mod_findings and mod2 in mod_findings:
                # Deduplicate findings per modality
                unique_f1 = list(dict.fromkeys(mod_findings[mod1]))
                unique_f2 = list(dict.fromkeys(mod_findings[mod2]))
                for fid1 in unique_f1:
                    for fid2 in unique_f2:
                        if (fid1, fid2) not in existing_edges and (fid2, fid1) not in existing_edges:
                            src, tgt = (fid1, fid2) if fid1 <= fid2 else (fid2, fid1)
                            kg.add_relation(Relation(head=src, relation="suggestive_of",
                                                      tail=tgt, weight=0.5))
                            existing_edges.add((src, tgt))
                            new_edges += 1

    print(f"  Disease-mediated suggestive_of edges added: {new_edges}")


def build_kg(data_dir: Optional[Path] = None) -> ClinicalKG:
    if data_dir is None:
        data_dir = DATA_DIR

    print("Building CASCADE-KG clinical knowledge graph...")
    kg = ClinicalKG()

    chexpert_df, _ = extract_chexpert_findings(kg, data_dir)

    _, ptbxl_findings = extract_ptbxl_findings(kg, data_dir)

    extract_radgraph_findings(kg, data_dir)

    link_radgraph_to_cxr(kg, data_dir)

    build_disease_subsumption(kg)

    build_same_patient_edges(kg, chexpert_df, data_dir)

    build_cross_modal_edges(kg, chexpert_df, data_dir)

    print(f"\nKnowledge graph statistics:")
    print(f"  Entities: {len(kg.entities)}")
    for etype, count in sorted(kg.entity_type_counts.items()):
        print(f"    {etype}: {count}")
    print(f"  Relations: {len(kg.relations)}")
    for rtype, count in sorted(kg.relation_type_counts.items()):
        print(f"    {rtype}: {count}")

    return kg


if __name__ == "__main__":
    kg = build_kg()
    output_path = DATA_DIR / "kg" / "clinical_kg.pkl"
    kg.save(output_path)
    print(f"\nSaved knowledge graph to {output_path}")

    efficient_dir = DATA_DIR / "kg" / "clinical_kg_efficient"
    kg.save_efficient(efficient_dir)