from __future__ import annotations

import ast
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIM_DIR = BASE_DIR / "simulation"
DATA_DIR = SIM_DIR / "data"
OUT_DIR = DATA_DIR / "kg" / "multimodal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RADGRAPH_JSON = (
    DATA_DIR
    / "radgraph"
    / "radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0"
    / "MIMIC-CXR_graphs.json"
)
CHEXPERT_CSV = DATA_DIR / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-chexpert.csv"
CXR_IMAGES_DIR = DATA_DIR / "mimic_cxr_jpg" / "images"
PTBXL_CSV = (
    DATA_DIR
    / "ptb_xl"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "ptbxl_database.csv"
)
PTBXL_RECORDS_DIR = (
    DATA_DIR
    / "ptb_xl"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    / "records100"
)
ECG_DEMO_CSV = DATA_DIR / "mimic_iv_ecg_demo" / "record_list.csv"
ECG_DEMO_WAVEFORMS = DATA_DIR / "mimic_iv_ecg_demo" / "waveforms" / "files"
MIMIC_DEMO_DIR = DATA_DIR / "mimic_iv_demo"


def _parse_key(key: str):
    parts = key.split("/")
    subject_id = None
    study_id = None
    for p in parts:
        if p.startswith("p") and p[1:].isdigit():
            subject_id = int(p[1:])
        elif p.startswith("s") and p[1:].isdigit():
            study_id = int(p[1:].replace(".txt", ""))
    return subject_id, study_id


class RadGraphTextExtractor:
    def __init__(self):
        self.output_text = OUT_DIR / "text_reports.csv.gz"
        self.output_entities = OUT_DIR / "radgraph_entities.csv.gz"
        self.output_relations = OUT_DIR / "radgraph_relations.csv.gz"

    def run(self):
        if all(p.exists() for p in [self.output_text, self.output_entities, self.output_relations]):
            print("[RadGraph] Outputs exist, skipping.")
            return
        print("[RadGraph] Loading MIMIC-CXR_graphs.json (streaming)...")
        text_rows = []
        entity_rows = []
        relation_rows = []
        total_docs = 0
        total_text_len = 0
        total_entities = 0
        total_relations = 0
        with open(RADGRAPH_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, doc in tqdm(data.items(), desc="Processing RadGraph docs"):
            total_docs += 1
            subject_id, study_id = _parse_key(key)
            text = doc.get("text", "")
            total_text_len += len(text)
            text_rows.append({"subject_id": subject_id, "study_id": study_id, "text": text})
            entities = doc.get("entities", {})
            for eid, ent in entities.items():
                total_entities += 1
                entity_rows.append({
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "entity_id": eid,
                    "tokens": ent.get("tokens", ""),
                    "label": ent.get("label", ""),
                    "start_ix": ent.get("start_ix", -1),
                    "end_ix": ent.get("end_ix", -1),
                })
                for rel in ent.get("relations", []):
                    total_relations += 1
                    relation_rows.append({
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "source_entity_id": eid,
                        "relation_type": rel[0] if len(rel) > 0 else "",
                        "target_entity_id": rel[1] if len(rel) > 1 else "",
                    })
        df_text = pd.DataFrame(text_rows)
        df_text.to_csv(self.output_text, index=False, compression="gzip")
        df_entities = pd.DataFrame(entity_rows)
        df_entities.to_csv(self.output_entities, index=False, compression="gzip")
        df_relations = pd.DataFrame(relation_rows)
        df_relations.to_csv(self.output_relations, index=False, compression="gzip")
        avg_len = total_text_len / max(total_docs, 1)
        print(f"[RadGraph] Docs: {total_docs}, Avg text length: {avg_len:.0f}")
        print(f"[RadGraph] Entities: {total_entities}, Relations: {total_relations}")


class CXRDatasetBuilder:
    def __init__(self):
        self.output = OUT_DIR / "cxr_images.csv"

    def run(self):
        if self.output.exists():
            print("[CXR] Output exists, skipping.")
            return
        print("[CXR] Scanning image directory...")
        rows = []
        for jpg_path in tqdm(sorted(CXR_IMAGES_DIR.rglob("*.jpg")), desc="Scanning JPGs"):
            parts = jpg_path.relative_to(CXR_IMAGES_DIR).parts
            subject_id = None
            study_id = None
            for p in parts:
                if p.startswith("p") and p[1:].isdigit():
                    subject_id = int(p[1:])
                elif p.startswith("s") and p[1:].isdigit():
                    study_id = int(p[1:])
            rows.append({
                "subject_id": subject_id,
                "study_id": study_id,
                "image_path": str(jpg_path),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output, index=False)
        n_images = len(df)
        n_patients = df["subject_id"].nunique()
        n_studies = df["study_id"].nunique()
        print(f"[CXR] Images: {n_images}, Patients: {n_patients}, Studies: {n_studies}")


class ECGDatasetBuilder:
    def __init__(self):
        self.output_ptbxl = OUT_DIR / "ecg_records.csv.gz"
        self.output_demo = OUT_DIR / "ecg_demo_records.csv"

    def run(self):
        self._run_ptbxl()
        self._run_demo()

    def _run_ptbxl(self):
        if self.output_ptbxl.exists():
            print("[ECG] PTB-XL output exists, skipping.")
            return
        print("[ECG] Loading PTB-XL database...")
        df = pd.read_csv(PTBXL_CSV)
        cols = [
            "ecg_id", "patient_id", "scp_codes", "report",
            "age", "sex", "heart_axis", "infarction_stadium1", "infarction_stadium2",
        ]
        available_cols = [c for c in cols if c in df.columns]
        df_out = df[available_cols].copy()
        dat_map = {}
        if PTBXL_RECORDS_DIR.exists():
            for dat_file in PTBXL_RECORDS_DIR.rglob("*.dat"):
                stem = dat_file.stem
                if "_" in stem:
                    stem = stem.split("_")[0]
                try:
                    ecg_id = int(stem)
                    dat_map[ecg_id] = str(dat_file)
                except ValueError:
                    pass
        df_out["dat_path"] = df_out["ecg_id"].map(dat_map)
        df_out.to_csv(self.output_ptbxl, index=False, compression="gzip")
        n_records = len(df_out)
        n_patients = df_out["patient_id"].nunique()
        scp_counter = Counter()
        for codes_str in df_out["scp_codes"].dropna():
            try:
                codes = ast.literal_eval(str(codes_str))
                if isinstance(codes, dict):
                    for k in codes:
                        scp_counter[k] += 1
            except (ValueError, SyntaxError):
                pass
        top_scp = scp_counter.most_common(10)
        print(f"[ECG] PTB-XL Records: {n_records}, Patients: {n_patients}")
        print(f"[ECG] Top SCP codes: {top_scp}")

    def _run_demo(self):
        if self.output_demo.exists():
            print("[ECG] Demo output exists, skipping.")
            return
        print("[ECG] Loading MIMIC-IV ECG demo...")
        df = pd.read_csv(ECG_DEMO_CSV)
        df.to_csv(self.output_demo, index=False)
        n_records = len(df)
        n_patients = df["subject_id"].nunique() if "subject_id" in df.columns else 0
        print(f"[ECG] Demo Records: {n_records}, Patients: {n_patients}")


class StructuredDataBuilder:
    def __init__(self):
        self.output_features = OUT_DIR / "structured_features.npz"
        self.output_patient_ids = OUT_DIR / "structured_patient_ids.csv"

    def run(self):
        if self.output_features.exists() and self.output_patient_ids.exists():
            print("[Structured] Outputs exist, skipping.")
            return
        print("[Structured] Loading MIMIC-IV demo tables...")
        patients = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "patients.csv.gz")
        admissions = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "admissions.csv.gz")
        diagnoses = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "diagnoses_icd.csv.gz")
        prescriptions = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "prescriptions.csv.gz")
        labevents = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "labevents.csv.gz")
        d_labitems = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "d_labitems.csv.gz")
        icustays = pd.read_csv(MIMIC_DEMO_DIR / "icu" / "icustays.csv.gz")
        patient_ids = sorted(patients["subject_id"].unique())
        dup_count = int(patients["subject_id"].duplicated(keep="first").sum())
        if dup_count:
            print(f"[Structured] WARNING: {dup_count} duplicate subject_id rows; deduplicating")
        patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
        n_patients = len(patient_ids)
        demo_features = np.zeros((n_patients, 3), dtype=np.float32)
        for _, row in patients.iterrows():
            idx = patient_id_to_idx[row["subject_id"]]
            anchor_age = row.get("anchor_age", None)
            demo_features[idx, 0] = float(anchor_age) if pd.notna(anchor_age) else 0.0
            gender = row.get("gender", None)
            demo_features[idx, 1] = 1.0 if gender == "M" else 0.0
            anchor_year = row.get("anchor_year", None)
            demo_features[idx, 2] = float(anchor_year) if pd.notna(anchor_year) else 0.0
        diag_matrix = np.zeros((n_patients, 50), dtype=np.float32)
        diag_counts = Counter()
        for _, row in diagnoses.iterrows():
            if row["subject_id"] in patient_id_to_idx:
                icd_code = str(row.get("icd_code", ""))
                diag_counts[icd_code] += 1
        top_diags = [code for code, _ in diag_counts.most_common(50)]
        diag_to_col = {code: i for i, code in enumerate(top_diags)}
        for _, row in diagnoses.iterrows():
            if row["subject_id"] in patient_id_to_idx:
                icd_code = str(row.get("icd_code", ""))
                if icd_code in diag_to_col:
                    diag_matrix[patient_id_to_idx[row["subject_id"]], diag_to_col[icd_code]] = 1.0
        lab_features = np.zeros((n_patients, 50), dtype=np.float32)
        labitem_counts = Counter()
        for _, row in labevents.iterrows():
            labitem_counts[row.get("itemid", 0)] += 1
        top_labitems = [itemid for itemid, _ in labitem_counts.most_common(50)]
        labitem_to_col = {itemid: i for i, itemid in enumerate(top_labitems)}
        lab_sums = defaultdict(lambda: [0.0] * 50)
        lab_counts = defaultdict(lambda: [0] * 50)
        for _, row in labevents.iterrows():
            if row["subject_id"] in patient_id_to_idx:
                itemid = row.get("itemid", 0)
                val = row.get("valuenum", None)
                if itemid in labitem_to_col and pd.notna(val):
                    col = labitem_to_col[itemid]
                    lab_sums[row["subject_id"]][col] += float(val)
                    lab_counts[row["subject_id"]][col] += 1
        for pid in patient_ids:
            idx = patient_id_to_idx[pid]
            for col in range(50):
                if lab_counts[pid][col] > 0:
                    lab_features[idx, col] = lab_sums[pid][col] / lab_counts[pid][col]
        rx_features = np.zeros((n_patients, 30), dtype=np.float32)
        rx_counts = Counter()
        for _, row in prescriptions.iterrows():
            drug = str(row.get("drug", ""))
            rx_counts[drug] += 1
        top_rx = [drug for drug, _ in rx_counts.most_common(30)]
        rx_to_col = {drug: i for i, drug in enumerate(top_rx)}
        for _, row in prescriptions.iterrows():
            if row["subject_id"] in patient_id_to_idx:
                drug = str(row.get("drug", ""))
                if drug in rx_to_col:
                    rx_features[patient_id_to_idx[row["subject_id"]], rx_to_col[drug]] = 1.0
        proc_features = np.zeros((n_patients, 20), dtype=np.float32)
        top_procs = []
        if (MIMIC_DEMO_DIR / "hosp" / "procedures_icd.csv.gz").exists():
            procedures = pd.read_csv(MIMIC_DEMO_DIR / "hosp" / "procedures_icd.csv.gz")
            proc_counts = Counter()
            for _, row in procedures.iterrows():
                proc_counts[str(row.get("icd_code", ""))] += 1
            top_procs = [p for p, _ in proc_counts.most_common(20)]
            proc_to_col = {p: i for i, p in enumerate(top_procs)}
            for _, row in procedures.iterrows():
                if row["subject_id"] in patient_id_to_idx:
                    code = str(row.get("icd_code", ""))
                    if code in proc_to_col:
                        proc_features[patient_id_to_idx[row["subject_id"]], proc_to_col[code]] = 1.0
        icu_features = np.zeros((n_patients, 3), dtype=np.float32)
        for _, row in icustays.iterrows():
            if row["subject_id"] in patient_id_to_idx:
                idx = patient_id_to_idx[row["subject_id"]]
                icu_features[idx, 0] += 1
                los = row.get("los", None)
                if pd.notna(los):
                    icu_features[idx, 1] += float(los)
                icu_features[idx, 2] = 1.0
        feature_matrix = np.hstack([demo_features, diag_matrix, lab_features, rx_features, proc_features, icu_features])
        feature_names = (
            ["age", "gender_m", "anchor_year"]
            + [f"diag_{c}" for c in top_diags]
            + [f"lab_{iid}" for iid in top_labitems]
            + [f"rx_{d}" for d in top_rx]
            + [f"proc_{p}" for p in top_procs]
            + ["icu_count", "icu_los_total", "icu_flag"]
        )
        np.savez_compressed(
            self.output_features,
            features=feature_matrix,
            patient_ids=np.array(patient_ids, dtype=np.int64),
            feature_names=np.array(feature_names, dtype=object),
        )
        pd.DataFrame({"subject_id": patient_ids}).to_csv(self.output_patient_ids, index=False)
        print(f"[Structured] Patients: {n_patients}, Features per patient: {feature_matrix.shape[1]}")


class ModalityFeatureStore:
    def __init__(self):
        self.index_path = OUT_DIR / "modality_index.json"
        self._text_df = None
        self._cxr_df = None
        self._ecg_df = None
        self._ecg_demo_df = None
        self._structured_pids = None
        self._structured_features = None
        self._modality_index = {}

    def build_index(self):
        if self.index_path.exists():
            print("[ModalityIndex] Index exists, skipping.")
            return
        print("[ModalityIndex] Building unified modality index...")
        text_path = OUT_DIR / "text_reports.csv.gz"
        if text_path.exists():
            self._text_df = pd.read_csv(text_path)
        cxr_path = OUT_DIR / "cxr_images.csv"
        if cxr_path.exists():
            self._cxr_df = pd.read_csv(cxr_path)
        ecg_path = OUT_DIR / "ecg_records.csv.gz"
        if ecg_path.exists():
            self._ecg_df = pd.read_csv(ecg_path)
        ecg_demo_path = OUT_DIR / "ecg_demo_records.csv"
        if ecg_demo_path.exists():
            self._ecg_demo_df = pd.read_csv(ecg_demo_path)
        structured_path = OUT_DIR / "structured_features.npz"
        if structured_path.exists():
            loaded = np.load(structured_path, allow_pickle=True)
            self._structured_features = loaded["features"]
            self._structured_pids = loaded["patient_ids"]
        self._modality_index = {}
        if self._text_df is not None:
            for _, row in self._text_df.iterrows():
                sid = row["study_id"]
                if pd.notna(sid):
                    key = f"STY_{int(sid)}"
                    if key not in self._modality_index:
                        self._modality_index[key] = []
                    if "text" not in self._modality_index[key]:
                        self._modality_index[key].append("text")
        if self._cxr_df is not None:
            for _, row in self._cxr_df.iterrows():
                sid = row["study_id"]
                if pd.notna(sid):
                    key = f"STY_{int(sid)}"
                    if key not in self._modality_index:
                        self._modality_index[key] = []
                    if "image" not in self._modality_index[key]:
                        self._modality_index[key].append("image")
        if self._ecg_df is not None:
            for _, row in self._ecg_df.iterrows():
                pid = row["patient_id"]
                if pd.notna(pid):
                    key = f"PAT_PTB{int(pid)}"
                    if key not in self._modality_index:
                        self._modality_index[key] = []
                    if "ecg" not in self._modality_index[key]:
                        self._modality_index[key].append("ecg")
        if self._ecg_demo_df is not None:
            for _, row in self._ecg_demo_df.iterrows():
                pid = row["subject_id"]
                if pd.notna(pid):
                    key = "PAT_{}".format(int(pid))
                    if key not in self._modality_index:
                        self._modality_index[key] = []
                    if "ecg" not in self._modality_index[key]:
                        self._modality_index[key].append("ecg")
        if self._structured_pids is not None:
            for pid in self._structured_pids:
                key = "PAT_{}".format(int(pid))
                if key not in self._modality_index:
                    self._modality_index[key] = []
                if "structured" not in self._modality_index[key]:
                    self._modality_index[key].append("structured")
        with open(self.index_path, "w") as f:
            json.dump(self._modality_index, f)
        print(f"[ModalityIndex] Indexed {len(self._modality_index)} entities")

    def get_text(self, study_id):
        if self._text_df is None:
            path = OUT_DIR / "text_reports.csv.gz"
            if path.exists():
                self._text_df = pd.read_csv(path)
            else:
                return None
        row = self._text_df[self._text_df["study_id"] == study_id]
        if len(row) == 0:
            return None
        return row.iloc[0]["text"]

    def get_image_path(self, study_id):
        if self._cxr_df is None:
            path = OUT_DIR / "cxr_images.csv"
            if path.exists():
                self._cxr_df = pd.read_csv(path)
            else:
                return None
        row = self._cxr_df[self._cxr_df["study_id"] == study_id]
        if len(row) == 0:
            return None
        return row.iloc[0]["image_path"]

    def get_ecg_path(self, patient_id):
        if self._ecg_df is None:
            path = OUT_DIR / "ecg_records.csv.gz"
            if path.exists():
                self._ecg_df = pd.read_csv(path)
            else:
                return None
        row = self._ecg_df[self._ecg_df["patient_id"] == patient_id]
        if len(row) == 0:
            return None
        dat_path = row.iloc[0].get("dat_path", None)
        if pd.notna(dat_path):
            return dat_path
        return None

    def get_structured(self, patient_id):
        if self._structured_pids is None:
            path = OUT_DIR / "structured_features.npz"
            if path.exists():
                loaded = np.load(path, allow_pickle=True)
                self._structured_features = loaded["features"]
                self._structured_pids = loaded["patient_ids"]
            else:
                return None
        mask = self._structured_pids == patient_id
        if not mask.any():
            return None
        idx = np.where(mask)[0][0]
        return self._structured_features[idx]

    def has_modality(self, entity_id, modality):
        if not self._modality_index:
            if self.index_path.exists():
                with open(self.index_path) as f:
                    self._modality_index = json.load(f)
            else:
                return False
        return modality in self._modality_index.get(entity_id, [])


def prepare_all():
    print("=" * 60)
    print("Multimodal Data Preparation Pipeline")
    print("=" * 60)
    print()
    print("Step 1/5: RadGraph text extraction")
    RadGraphTextExtractor().run()
    print()
    print("Step 2/5: CXR image catalog")
    CXRDatasetBuilder().run()
    print()
    print("Step 3/5: ECG dataset")
    ECGDatasetBuilder().run()
    print()
    print("Step 4/5: Structured data")
    StructuredDataBuilder().run()
    print()
    print("Step 5/5: Modality index")
    store = ModalityFeatureStore()
    store.build_index()
    print()
    print("=" * 60)
    print("Pipeline complete. Outputs in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    prepare_all()
