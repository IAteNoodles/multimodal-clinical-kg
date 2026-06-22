"""Build enhanced clinical KG with BigQuery MIMIC-IV, MRCONSO crosswalks, SNOMED CT ontology."""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from extract_entities import (
    ClinicalKG, Entity, Relation,
    extract_chexpert_findings, extract_ptbxl_findings,
    extract_radgraph_findings, link_radgraph_to_cxr,
    build_disease_subsumption, build_same_patient_edges,
    build_cross_modal_edges,
    CHEXPERT_TO_DISEASE, ECG_SCP_TO_DISEASE,
)

DATA_DIR = Path(__file__).parent.parent / "data"

MRCONSO_PATH = Path(os.environ.get("MRCONSO_PATH", r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\umls-2026AA-mrconso\2026AA\META\MRCONSO.RRF"))
SNOMED_REL_PATH = Path(os.environ.get("SNOMED_REL_PATH", r"C:\Users\Noodl\Downloads\SnomedCT\SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260601.txt"))
SNOMED_DESC_PATH = Path(os.environ.get("SNOMED_DESC_PATH", r"C:\Users\Noodl\Downloads\SnomedCT\SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260601.txt"))

BQ_TOKEN_PATH = Path(os.environ.get("BQ_TOKEN_PATH", r"C:\Users\Noodl\Projects\Research\Exploration-MJ\data\mimic_5k\token.json"))
BQ_CLIENT_SECRET_PATH = Path(os.environ.get("BQ_CLIENT_SECRET_PATH", r"C:\Users\Noodl\Projects\Research\Exploration-MJ\client_secret.json"))
BQ_PROJECT_ID = "physionet-data-498016"

OUTPUT_DIR = DATA_DIR / "kg" / "data_new"

ENTITY_TYPE_TO_ID = {
    "Finding": 0, "Anatomy": 1, "Disease": 2, "Patient": 3,
    "Study": 4, "Drug": 5, "LabResult": 6, "Procedure": 7, "VitalResult": 8,
    "ECGMeasurement": 9, "ECGRhythm": 10, "Unknown": 11,
}
ENTITY_TYPE_ORDER = ["Finding", "Anatomy", "Disease", "Patient", "Study", "Drug", "LabResult", "Procedure", "VitalResult", "ECGMeasurement", "ECGRhythm", "Unknown"]

MODALITY_TO_ID = {"CXR": 0, "ECG": 1, "RAD": 2, "STR": 3, None: 4}

N_PATIENTS = 2000

CLINICAL_LABS = {
    50802: "Base Excess",
    50804: "Calculated Total CO2",
    50805: "Carboxyhemoglobin",
    50808: "Free Calcium",
    50813: "Lactate",
    50814: "Methemoglobin",
    50817: "Oxygen Saturation",
    50818: "pCO2",
    50820: "pH",
    50821: "pO2",
    50853: "25-OH Vitamin D",
    50856: "Acetaminophen",
    50861: "Alanine Aminotransferase (ALT)",
    50862: "Albumin",
    50863: "Alkaline Phosphatase",
    50867: "Amylase",
    50868: "Anion Gap",
    50882: "Bicarbonate",
    50883: "Bilirubin, Direct",
    50885: "Bilirubin, Total",
    50889: "C-Reactive Protein",
    50893: "Calcium, Total",
    50902: "Chloride",
    50904: "Cholesterol, HDL",
    50905: "Cholesterol, LDL, Calculated",
    50907: "Cholesterol, Total",
    50910: "Creatine Kinase (CK)",
    50911: "Creatine Kinase, MB Isoenzyme",
    50912: "Creatinine",
    50915: "D-Dimer",
    50920: "Estimated GFR (MDRD equation)",
    50922: "Ethanol",
    50924: "Ferritin",
    50925: "Folate",
    50927: "Gamma Glutamyltransferase",
    50930: "Globulin",
    50931: "Glucose",
    50941: "Hepatitis B Surface Antigen",
    50943: "Hepatitis C Virus Antibody",
    50949: "Immunoglobulin A",
    50950: "Immunoglobulin G",
    50951: "Immunoglobulin M",
    50952: "Iron",
    50953: "Iron Binding Capacity, Total",
    50954: "Lactate Dehydrogenase (LD)",
    50956: "Lipase",
    50960: "Magnesium",
    50963: "NTproBNP",
    50970: "Phosphate",
    50971: "Potassium",
    50976: "Protein, Total",
    50981: "Salicylate",
    50983: "Sodium",
    50986: "Tacrolimus",
    50993: "Thyroid Stimulating Hormone",
    50995: "Thyroxine (T4), Free",
    50998: "Transferrin",
    50999: "Tricyclic Antidepressant Screen",
    51000: "Triglycerides",
    51002: "Troponin I",
    51003: "Troponin T",
    51006: "Urea Nitrogen",
    51007: "Uric Acid",
    51008: "Valproic Acid",
    51009: "Vancomycin",
    51010: "Vitamin B12",
    51133: "Absolute Lymphocyte Count",
    51137: "Anisocytosis",
    51143: "Atypical Lymphocytes",
    51144: "Bands",
    51146: "Basophils",
    51200: "Eosinophils",
    51214: "Fibrinogen, Functional",
    51221: "Hematocrit",
    51222: "Hemoglobin",
    51233: "Hypochromia",
    51237: "INR(PT)",
    51244: "Lymphocytes",
    51246: "Macrocytes",
    51248: "MCH",
    51249: "MCHC",
    51250: "MCV",
    51251: "Metamyelocytes",
    51252: "Microcytes",
    51254: "Monocytes",
    51255: "Myelocytes",
    51256: "Neutrophils",
    51257: "Nucleated Red Cells",
    51260: "Ovalocytes",
    51265: "Platelet Count",
    51267: "Poikilocytosis",
    51268: "Polychromasia",
    51274: "PT",
    51275: "PTT",
    51277: "RDW",
    51279: "Red Blood Cells",
    51288: "Sedimentation Rate",
    51296: "Teardrop Cells",
    51301: "White Blood Cells",
    52069: "Absolute Basophil Count",
    52073: "Absolute Eosinophil Count",
    52074: "Absolute Monocyte Count",
    52075: "Absolute Neutrophil Count",
    52135: "Immature Granulocytes",
    52142: "Mean Platelet Volume",
    52172: "RDW-SD",
}

VITAL_SIGNS = {
    "heart_rate": ("HR", 100, 60),
    "resp_rate": ("RR", 20, 12),
    "spo2": ("SpO2", None, 92),
    "temperature": ("Temp", 38.0, 36.0),
    "sbp": ("SBP", 140, 90),
    "dbp": ("DBP", 90, 60),
    "mbp": ("MBP", 110, 70),
    "glucose": ("Gluc", 180, 60),
}

VITAL_COLUMN_ALIASES = {
    "heart_rate": ["heart_rate", "heartrate"],
    "resp_rate": ["resp_rate", "resprate", "respiratory_rate"],
    "spo2": ["spo2", "o2_saturation"],
    "temperature": ["temperature", "temp"],
    "sbp": ["sbp", "systolic_bp"],
    "dbp": ["dbp", "diastolic_bp"],
    "mbp": ["mbp", "mean_bp"],
    "glucose": ["glucose", "blood_glucose"],
}

LAB_TO_DISEASE = {
    "LAB_Creatinine_High": "DIS_KidneyDisease",
    "LAB_Estimated_GFR_MDRD_equation_Low": "DIS_KidneyDisease",
    "LAB_Urea_Nitrogen_High": "DIS_KidneyDisease",
    "LAB_Potassium_High": "DIS_Hyperkalemia",
    "LAB_Potassium_Low": "DIS_Hypokalemia",
    "LAB_Sodium_High": "DIS_Hypernatremia",
    "LAB_Sodium_Low": "DIS_Hyponatremia",
    "LAB_Chloride_High": "DIS_Hyperchloremia",
    "LAB_Chloride_Low": "DIS_Hypochloremia",
    "LAB_Bicarbonate_High": "DIS_MetabolicAlkalosis",
    "LAB_Bicarbonate_Low": "DIS_MetabolicAcidosis",
    "LAB_Anion_Gap_High": "DIS_MetabolicAcidosis",
    "LAB_Glucose_High": "DIS_Hyperglycemia",
    "LAB_Glucose_Low": "DIS_Hypoglycemia",
    "LAB_Calcium_Total_High": "DIS_Hypercalcemia",
    "LAB_Calcium_Total_Low": "DIS_Hypocalcemia",
    "LAB_Magnesium_High": "DIS_Hypermagnesemia",
    "LAB_Magnesium_Low": "DIS_Hypomagnesemia",
    "LAB_Phosphate_High": "DIS_Hyperphosphatemia",
    "LAB_Phosphate_Low": "DIS_Hypophosphatemia",
    "LAB_Alanine_Aminotransferase_ALT_High": "DIS_LiverDisease",
    "LAB_Bilirubin_Total_High": "DIS_LiverDisease",
    "LAB_Alkaline_Phosphatase_High": "DIS_LiverDisease",
    "LAB_Albumin_Low": "DIS_Malnutrition",
    "LAB_Protein_Total_Low": "DIS_Malnutrition",
    "LAB_Globulin_High": "DIS_Inflammation",
    "LAB_Troponin_T_High": "DIS_MyocardialInfarction",
    "LAB_Creatine_Kinase_MB_Isoenzyme_High": "DIS_MyocardialInfarction",
    "LAB_NTproBNP_High": "DIS_HeartFailure",
    "LAB_Lactate_Dehydrogenase_LD_High": "DIS_Hemolysis",
    "LAB_Creatine_Kinase_CK_High": "DIS_Rhabdomyolysis",
    "LAB_Cholesterol_Total_High": "DIS_Hyperlipidemia",
    "LAB_Cholesterol_HDL_Low": "DIS_Hyperlipidemia",
    "LAB_Cholesterol_LDL_Calculated_High": "DIS_Hyperlipidemia",
    "LAB_Triglycerides_High": "DIS_Hypertriglyceridemia",
    "LAB_Thyroid_Stimulating_Hormone_High": "DIS_Hypothyroidism",
    "LAB_Thyroid_Stimulating_Hormone_Low": "DIS_Hyperthyroidism",
    "LAB_Thyroxine_T4_Free_High": "DIS_Hyperthyroidism",
    "LAB_Thyroxine_T4_Free_Low": "DIS_Hypothyroidism",
    "LAB_Iron_Low": "DIS_IronDeficiency",
    "LAB_Ferritin_Low": "DIS_IronDeficiency",
    "LAB_C_Reactive_Protein_High": "DIS_Inflammation",
    "LAB_Lactate_High": "DIS_LacticAcidosis",
    "LAB_Uric_Acid_High": "DIS_Gout",
    "LAB_Vitamin_B12_Low": "DIS_B12Deficiency",
    "LAB_25_OH_Vitamin_D_Low": "DIS_VitaminDDeficiency",
    "LAB_Lipase_High": "DIS_Pancreatitis",
    "LAB_Amylase_High": "DIS_Pancreatitis",
    "LAB_Hematocrit_High": "DIS_Polycythemia",
    "LAB_Hematocrit_Low": "DIS_Anemia",
    "LAB_Hemoglobin_High": "DIS_Polycythemia",
    "LAB_Hemoglobin_Low": "DIS_Anemia",
    "LAB_Red_Blood_Cells_High": "DIS_Polycythemia",
    "LAB_Red_Blood_Cells_Low": "DIS_Anemia",
    "LAB_White_Blood_Cells_High": "DIS_Leukocytosis",
    "LAB_White_Blood_Cells_Low": "DIS_Leukopenia",
    "LAB_Platelet_Count_High": "DIS_Thrombocytosis",
    "LAB_Platelet_Count_Low": "DIS_Thrombocytopenia",
    "LAB_RDW_High": "DIS_Anemia",
    "LAB_MCV_High": "DIS_MacrocyticAnemia",
    "LAB_MCV_Low": "DIS_MicrocyticAnemia",
    "LAB_Neutrophils_High": "DIS_Neutrophilia",
    "LAB_Neutrophils_Low": "DIS_Neutropenia",
    "LAB_Lymphocytes_High": "DIS_Lymphocytosis",
    "LAB_Lymphocytes_Low": "DIS_Lymphocytopenia",
    "LAB_Eosinophils_High": "DIS_Eosinophilia",
    "LAB_INR_PT_High": "DIS_Coagulopathy",
    "LAB_PT_High": "DIS_Coagulopathy",
    "LAB_PTT_High": "DIS_Coagulopathy",
    "LAB_Fibrinogen_Functional_Low": "DIS_Hypofibrinogenemia",
    "LAB_pH_High": "DIS_Alkalosis",
    "LAB_pH_Low": "DIS_Acidosis",
    "LAB_pO2_Low": "DIS_Hypoxemia",
    "LAB_pCO2_High": "DIS_Hypercapnia",
    "LAB_pCO2_Low": "DIS_Hypocapnia",
    "LAB_Base_Excess_High": "DIS_MetabolicAlkalosis",
    "LAB_Base_Excess_Low": "DIS_MetabolicAcidosis",
    "LAB_Absolute_Neutrophil_Count_Low": "DIS_Neutropenia",
    "LAB_Bands_High": "DIS_LeftShift",
}

VITAL_TO_DISEASE = {
    "VIT_HR_High": "DIS_Tachycardia",
    "VIT_HR_Low": "DIS_Bradycardia",
    "VIT_RR_High": "DIS_Tachypnea",
    "VIT_RR_Low": "DIS_Bradypnea",
    "VIT_SpO2_Low": "DIS_Hypoxemia",
    "VIT_Temp_High": "DIS_Fever",
    "VIT_Temp_Low": "DIS_Hypothermia",
    "VIT_SBP_High": "DIS_Hypertension",
    "VIT_SBP_Low": "DIS_Hypotension",
    "VIT_DBP_High": "DIS_Hypertension",
    "VIT_DBP_Low": "DIS_Hypotension",
}

SNOMED_REL_TYPE_IDS = {
    "116680003": "is_a",
    "363702006": "has_focus",
    "42752001": "due_to",
    "246090004": "associated_finding",
    "363703001": "has_intent",
    "363699004": "direct_site",
    "405813007": "procedure_site",
}

ECG_MEASUREMENT_THRESHOLDS = {
    "qrs_duration": {
        "high": (120, "ECGM_Wide_QRS", "Wide QRS Complex (>120ms)"),
        "low": None,
    },
    "pr_interval": {
        "high": (200, "ECGM_PR_Prolonged", "PR Prolonged (>200ms)"),
        "low": (120, "ECGM_PR_Short", "PR Short (<120ms)"),
    },
    "qtc": {
        "high": (450, "ECGM_QTc_Prolonged", "QTc Prolonged (>450ms)"),
        "low": (350, "ECGM_QTc_Short", "QTc Short (<350ms)"),
    },
    "qrs_axis": {
        "high": (90, "ECGM_QRS_Axis_RightDeviation", "QRS Right Axis Deviation (>90deg)"),
        "low": (-30, "ECGM_QRS_Axis_LeftDeviation", "QRS Left Axis Deviation (<-30deg)"),
    },
    "p_axis": {
        "high": (75, "ECGM_P_Axis_Abnormal_High", "P Axis Abnormal (>75deg)"),
        "low": (0, "ECGM_P_Axis_Abnormal_Low", "P Axis Abnormal (<0deg)"),
    },
    "t_axis": {
        "high": (90, "ECGM_T_Axis_Abnormal", "T Axis Abnormal (>90deg)"),
        "low": None,
    },
}

ECG_COMPUTED_COLUMNS = {
    "qrs_duration": {
        "source_cols": ["qrs_onset", "qrs_end"],
        "compute": lambda df: df["qrs_end"] - df["qrs_onset"],
    },
    "pr_interval": {
        "source_cols": ["p_onset", "qrs_onset"],
        "compute": lambda df: df["qrs_onset"] - df["p_onset"],
    },
    "qtc": {
        "source_cols": ["t_end", "qrs_onset", "rr_interval"],
        "compute": lambda df: (df["t_end"] - df["qrs_onset"]) / np.sqrt(df["rr_interval"] / 1000),
    },
}

ECG_MEASUREMENT_TO_DISEASE = {
    "ECGM_Wide_QRS": "DIS_ConductionAbnormality",
    "ECGM_PR_Prolonged": "DIS_FirstDegreeAVBlock",
    "ECGM_PR_Short": "DIS_PreExcitation",
    "ECGM_QTc_Prolonged": "DIS_LongQTSyndrome",
    "ECGM_QTc_Short": "DIS_ShortQTSyndrome",
    "ECGM_QRS_Axis_RightDeviation": "DIS_RightAxisDeviation",
    "ECGM_QRS_Axis_LeftDeviation": "DIS_LeftAxisDeviation",
    "ECGM_P_Axis_Abnormal_High": "DIS_RightAtrialEnlargement",
    "ECGM_P_Axis_Abnormal_Low": "DIS_LeftAtrialEnlargement",
    "ECGM_T_Axis_Abnormal": "DIS_Ischemia",
}

RHYTHM_PATTERNS = {
    "SR": ("ECGR_Sinus_Rhythm", "Sinus Rhythm"),
    "ST": ("ECGR_Sinus_Tachycardia", "Sinus Tachycardia"),
    "SB": ("ECGR_Sinus_Bradycardia", "Sinus Bradycardia"),
    "SA": ("ECGR_Sinus_Arrhythmia", "Sinus Arrhythmia"),
    "AF": ("ECGR_Atrial_Fibrillation", "Atrial Fibrillation"),
    "AFL": ("ECGR_Atrial_Flutter", "Atrial Flutter"),
    "JER": ("ECGR_Junctional_Escape_Rhythm", "Junctional Escape Rhythm"),
    "AJR": ("ECGR_Accelerated_Junctional_Rhythm", "Accelerated Junctional Rhythm"),
    "SVT": ("ECGR_Supraventricular_Tachycardia", "Supraventricular Tachycardia"),
    "AT": ("ECGR_Atrial_Tachycardia", "Atrial Tachycardia"),
    "AVNRT": ("ECGR_AVNRT", "AV Nodal Reentrant Tachycardia"),
    "AVRT": ("ECGR_AVRT", "AV Reentrant Tachycardia"),
    "VT": ("ECGR_Ventricular_Tachycardia", "Ventricular Tachycardia"),
    "VF": ("ECGR_Ventricular_Fibrillation", "Ventricular Fibrillation"),
}

RHYTHM_TO_DISEASE = {
    "ECGR_Sinus_Rhythm": "DIS_SinusRhythm",
    "ECGR_Sinus_Tachycardia": "DIS_Tachycardia",
    "ECGR_Sinus_Bradycardia": "DIS_Bradycardia",
    "ECGR_Sinus_Arrhythmia": "DIS_SinusArrhythmia",
    "ECGR_Atrial_Fibrillation": "DIS_AtrialFibrillation",
    "ECGR_Atrial_Flutter": "DIS_AtrialFlutter",
    "ECGR_Junctional_Escape_Rhythm": "DIS_JunctionalRhythm",
    "ECGR_Accelerated_Junctional_Rhythm": "DIS_JunctionalRhythm",
    "ECGR_Supraventricular_Tachycardia": "DIS_SupraventricularTachycardia",
    "ECGR_Atrial_Tachycardia": "DIS_AtrialTachycardia",
    "ECGR_AVNRT": "DIS_SupraventricularTachycardia",
    "ECGR_AVRT": "DIS_SupraventricularTachycardia",
    "ECGR_Ventricular_Tachycardia": "DIS_VentricularTachycardia",
    "ECGR_Ventricular_Fibrillation": "DIS_VentricularFibrillation",
}


def _add_entity_if_new(kg: ClinicalKG, eid: str, etype: str, modality: str, label: str,
                       properties: Optional[Dict[str, Any]] = None) -> Entity:
    if eid in kg.entities:
        return kg.entities[eid]
    entity = Entity(id=eid, type=etype, modality=modality, label=label, properties=properties or {})
    kg.add_entity(entity)
    return entity


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9]', '_', str(name))
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized


def _normalize_anatomy_name(name: str) -> str:
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    return normalized


def _parse_datetime(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, datetime):
        return val
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            val_str = str(int(val))
        except (ValueError, OverflowError):
            val_str = str(val).strip()
    else:
        val_str = str(val).strip()
    if not val_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(val_str, fmt)
        except ValueError:
            continue
    return None


def _get_kg_patient_ids(kg: ClinicalKG) -> Set[int]:
    patient_ids = set()
    for eid, entity in kg.entities.items():
        if entity.type == "Patient":
            if eid.startswith("PAT_"):
                suffix = eid[4:]
                try:
                    patient_ids.add(int(suffix))
                except ValueError:
                    pass
    return patient_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Crosswalks
# ═══════════════════════════════════════════════════════════════════════════════

def load_crosswalks(output_dir: Path) -> Dict[str, Any]:
    pkl_path = output_dir / "crosswalks.pkl"
    if pkl_path.exists():
        print(f"Loading cached crosswalks from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    print("Building crosswalks from MRCONSO + SNOMED CT (one-time)...")
    crosswalks = _build_crosswalks()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(crosswalks, f)
    print(f"  Cached crosswalks to {pkl_path}")
    return crosswalks


def _build_crosswalks() -> Dict[str, Any]:
    icd_to_cui: Dict[str, str] = {}
    icd10pcs_to_cui: Dict[str, str] = {}
    cui_to_snomed: Dict[str, str] = {}
    rxnorm_to_cui: Dict[str, str] = {}
    cui_to_name: Dict[str, str] = {}
    snomed_to_cui: Dict[str, str] = {}

    if MRCONSO_PATH.exists():
        print(f"  Parsing MRCONSO.RRF ({MRCONSO_PATH})...")
        with open(MRCONSO_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="MRCONSO", mininterval=10):
                parts = line.rstrip('\n').split('|')
                if len(parts) < 15:
                    continue
                cui = parts[0]
                lat = parts[1]
                suppress = parts[16] if len(parts) > 16 else ''
                sab = parts[11]
                tty = parts[12]
                code = parts[13]
                term = parts[14]
                scui = parts[9]

                if lat != 'ENG':
                    continue
                if suppress in ('O', 'D'):
                    continue

                if cui not in cui_to_name:
                    cui_to_name[cui] = term

                if sab == 'ICD10CM' and code:
                    icd_to_cui[f"{sab}:{code}"] = cui
                if sab == 'ICD9CM' and code:
                    icd_to_cui[f"{sab}:{code}"] = cui

                if sab == 'ICD10PCS' and code:
                    icd10pcs_to_cui[code] = cui

                if sab == 'RXNORM' and tty in ('SCD', 'SBD', 'SCDG', 'SBDG', 'GPCK', 'BN', 'IN', 'PIN', 'MIN'):
                    if code:
                        rxnorm_to_cui[code] = cui

                if sab == 'SNOMEDCT_US' and scui:
                    cui_to_snomed[cui] = scui
                    snomed_to_cui[scui] = cui
    else:
        print(f"  MRCONSO.RRF not found at {MRCONSO_PATH}, skipping UMLS crosswalk")

    snomed_id_to_name: Dict[str, str] = {}
    if SNOMED_DESC_PATH.exists():
        print(f"  Parsing SNOMED Description ({SNOMED_DESC_PATH})...")
        with open(SNOMED_DESC_PATH, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in tqdm(f, desc="SNOMED Desc", mininterval=10):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 8:
                    continue
                concept_id = parts[4]
                type_id = parts[6]
                term = parts[7]
                if type_id == '900000000000003001':
                    snomed_id_to_name[concept_id] = term
    else:
        print(f"  SNOMED Description not found at {SNOMED_DESC_PATH}, skipping")

    snomed_rels: Dict[str, List[Tuple[str, str]]] = {name: [] for name in SNOMED_REL_TYPE_IDS.values()}

    if SNOMED_REL_PATH.exists():
        print(f"  Parsing SNOMED Relationships ({SNOMED_REL_PATH})...")
        with open(SNOMED_REL_PATH, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in tqdm(f, desc="SNOMED Rel", mininterval=10):
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 10:
                    continue
                active = parts[2]
                if active != '1':
                    continue
                source_id = parts[4]
                dest_id = parts[5]
                type_id = parts[7]

                rel_name = SNOMED_REL_TYPE_IDS.get(type_id)
                if rel_name:
                    snomed_rels[rel_name].append((source_id, dest_id))
    else:
        print(f"  SNOMED Relationships not found at {SNOMED_REL_PATH}, skipping")

    print(f"  Crosswalks: icd_to_cui={len(icd_to_cui)}, icd10pcs_to_cui={len(icd10pcs_to_cui)}, rxnorm_to_cui={len(rxnorm_to_cui)}, "
          f"cui_to_snomed={len(cui_to_snomed)}, snomed_id_to_name={len(snomed_id_to_name)}, "
          + ", ".join(f"{name}={len(pairs)}" for name, pairs in snomed_rels.items()))

    result = {
        "icd_to_cui": icd_to_cui,
        "icd10pcs_to_cui": icd10pcs_to_cui,
        "cui_to_snomed": cui_to_snomed,
        "rxnorm_to_cui": rxnorm_to_cui,
        "cui_to_name": cui_to_name,
        "snomed_to_cui": snomed_to_cui,
        "snomed_id_to_name": snomed_id_to_name,
    }
    result.update(snomed_rels)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BigQuery helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_bq_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google.cloud import bigquery

    creds = Credentials.from_authorized_user_file(str(BQ_TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(BQ_TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())

    return bigquery.Client(credentials=creds, project=BQ_PROJECT_ID)


_RESOLVED_DATASETS = {}

def _resolve_bq_dataset(client, project: str, candidates: list) -> str:
    for dataset in candidates:
        key = f"{project}.{dataset}"
        if key in _RESOLVED_DATASETS:
            return _RESOLVED_DATASETS[key]
        try:
            sql = f"SELECT 1 FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLES` LIMIT 1"
            client.query(sql).result()
            _RESOLVED_DATASETS[key] = dataset
            print(f"  BQ dataset resolved: {project}.{dataset}")
            return dataset
        except Exception:
            continue
    print(f"  WARNING: No BQ dataset found among: {candidates}")
    return candidates[0]


def _bq_query(client, sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()


def get_top_patient_ids(client, n: int = N_PATIENTS) -> List[int]:
    sql = f"""
    SELECT subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.admissions`
    GROUP BY subject_id
    ORDER BY COUNT(*) DESC
    LIMIT {n}
    """
    df = _bq_query(client, sql)
    patient_ids = df['subject_id'].astype(int).tolist()
    print(f"  Selected top {len(patient_ids)} patients by admission count")
    return patient_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: BigQuery enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def extract_diagnoses_bq(kg: ClinicalKG, client, patient_ids: List[int],
                         crosswalks: Dict[str, Any]) -> pd.DataFrame:
    print("Extracting diagnoses from BigQuery...")
    icd_to_cui = crosswalks["icd_to_cui"]
    cui_to_name = crosswalks["cui_to_name"]

    id_list = ','.join(str(x) for x in patient_ids)
    sql = f"""
    SELECT subject_id, hadm_id, icd_code, icd_version
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE subject_id IN ({id_list})
    """
    df = _bq_query(client, sql)
    if df.empty:
        print("  No diagnosis data returned")
        return df

    df['subject_id'] = df['subject_id'].astype(int)
    df['icd_code'] = df['icd_code'].astype(str)
    df['icd_version'] = df['icd_version'].astype(str)

    sql_d = """
    SELECT icd_code, icd_version, long_title
    FROM `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`
    """
    print("  Querying BigQuery for ICD diagnosis titles...")
    d_icd = _bq_query(client, sql_d)
    icd_lookup: Dict[Tuple[str, str], str] = {}
    if not d_icd.empty:
        d_icd['icd_code'] = d_icd['icd_code'].astype(str)
        d_icd['icd_version'] = d_icd['icd_version'].astype(str)
        d_icd['long_title'] = d_icd['long_title'].astype(str)
        icd_lookup = dict(zip(zip(d_icd['icd_version'], d_icd['icd_code']), d_icd['long_title']))
    print(f"  Loaded {len(icd_lookup)} ICD diagnosis titles")

    df['long_title'] = df.apply(
        lambda r: icd_lookup.get((r['icd_version'], r['icd_code']), r['icd_code']), axis=1
    )
    df['dis_id'] = 'DIS_' + df['long_title'].apply(_sanitize_name)
    df['long_title'] = df['long_title'].str[:100]

    df['cui'] = df.apply(
        lambda r: icd_to_cui.get(f"ICD{r['icd_version']}CM:{r['icd_code']}"), axis=1
    )

    unique_pairs = df[['subject_id', 'dis_id', 'long_title']].drop_duplicates(subset=['subject_id', 'dis_id'])

    new_diseases = 0
    new_edges = 0
    # Vectorized: add entities and relations in batch
    unique_diseases = unique_pairs[['dis_id', 'long_title']].drop_duplicates(subset=['dis_id'])
    for _, row in unique_diseases.iterrows():
        is_new = row['dis_id'] not in kg.entities
        _add_entity_if_new(kg, row['dis_id'], "Disease", "STR", row['long_title'])
        if is_new:
            new_diseases += 1

    # Add patient entities for any missing patients
    missing_patients = unique_pairs['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    # Add all relations at once
    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Diagnoses"):
        pat_id = f"PAT_{row['subject_id']}"
        kg.add_relation(Relation(head=pat_id, relation="diagnosed_with", tail=row['dis_id'], weight=1.0))
        new_edges += 1

    icd_to_dis_name: Dict[str, str] = {}
    for _, row in df[['icd_code', 'icd_version', 'dis_id']].drop_duplicates(subset=['icd_code', 'icd_version']).iterrows():
        key = f"{row['icd_code']}_{row['icd_version']}"
        icd_to_dis_name[key] = row['dis_id']
    crosswalks['icd_to_dis_name'] = icd_to_dis_name

    print(f"  Diagnoses: {unique_pairs['subject_id'].nunique()} patients, {new_diseases} new diseases, {new_edges} edges")
    return df


def extract_medications_bq(kg: ClinicalKG, client, patient_ids: List[int],
                           crosswalks: Dict[str, Any]) -> pd.DataFrame:
    print("Extracting medications from BigQuery...")
    rxnorm_to_cui = crosswalks["rxnorm_to_cui"]

    id_list = ','.join(str(x) for x in patient_ids)
    print("  Querying BigQuery for prescriptions...")
    sql = f"""
    SELECT subject_id, hadm_id, drug, gsn, ndc
    FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
    WHERE subject_id IN ({id_list})
    LIMIT 500000
    """
    df = _bq_query(client, sql)
    print(f"  Got {len(df)} prescription rows from BigQuery")
    if df.empty:
        print("  No prescription data returned")
        return df

    df = df.dropna(subset=['subject_id', 'drug'])
    df['subject_id'] = df['subject_id'].astype(int)
    df['drug_str'] = df['drug'].astype(str).str.strip()
    df = df[df['drug_str'] != '']
    df['sanitized'] = df['drug_str'].apply(_sanitize_name)
    df = df[df['sanitized'] != '']
    df['drg_id'] = 'DRG_' + df['sanitized']

    unique_pairs = df[['subject_id', 'drg_id', 'drug_str']].drop_duplicates(subset=['subject_id', 'drg_id'])

    new_drugs = 0
    new_edges = 0
    print(f"  Processing {len(unique_pairs)} unique patient-drug pairs...")
    # Vectorized: add drug entities in batch
    unique_drugs = unique_pairs[['drg_id', 'drug_str']].drop_duplicates(subset=['drg_id'])
    for _, row in unique_drugs.iterrows():
        is_new = row['drg_id'] not in kg.entities
        _add_entity_if_new(kg, row['drg_id'], "Drug", "STR", row['drug_str'])
        if is_new:
            new_drugs += 1

    # Add patient entities for any missing patients
    missing_patients = unique_pairs['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    # Add all relations at once
    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Medications"):
        pat_id = f"PAT_{row['subject_id']}"
        kg.add_relation(Relation(head=pat_id, relation="prescribed", tail=row['drg_id'], weight=1.0))
        new_edges += 1

    print(f"  Medications: {unique_pairs['subject_id'].nunique()} patients, {new_drugs} drugs, {new_edges} edges")
    return df


def extract_lab_results_bq(kg: ClinicalKG, client, patient_ids: List[int]) -> pd.DataFrame:
    print("Extracting lab results from BigQuery...")
    clinical_itemids = list(CLINICAL_LABS.keys())
    itemid_str = ','.join(str(x) for x in clinical_itemids)
    id_list = ','.join(str(x) for x in patient_ids)

    print("  Querying BigQuery for lab events...")
    sql = f"""
    SELECT subject_id, hadm_id, itemid, charttime, valuenum, ref_range_lower, ref_range_upper
    FROM `physionet-data.mimiciv_3_1_hosp.labevents`
    WHERE subject_id IN ({id_list})
    AND itemid IN ({itemid_str})
    AND valuenum IS NOT NULL
    LIMIT 500000
    """
    df = _bq_query(client, sql)
    print(f"  Got {len(df)} lab event rows from BigQuery")
    if df.empty:
        print("  No lab data returned")
        return df

    df['subject_id'] = df['subject_id'].astype(int)
    df['itemid'] = df['itemid'].astype(int)
    df['valuenum'] = df['valuenum'].astype(float)

    lab_dfs = []
    for itemid, lab_name in CLINICAL_LABS.items():
        sub = df[df['itemid'] == itemid].copy()
        if sub.empty:
            continue
        safe_name = _sanitize_name(lab_name)
        has_ref = sub['ref_range_upper'].notna() | sub['ref_range_lower'].notna()
        sub_ref = sub[has_ref].copy()
        if not sub_ref.empty:
            high_mask = sub_ref['ref_range_upper'].notna() & (sub_ref['valuenum'] > sub_ref['ref_range_upper'])
            if high_mask.any():
                high = sub_ref[high_mask].copy()
                high['lab_id'] = f"LAB_{safe_name}_High"
                high['lab_label'] = f"{lab_name} High"
                lab_dfs.append(high[['subject_id', 'lab_id', 'lab_label']].drop_duplicates())
            low_mask = sub_ref['ref_range_lower'].notna() & (sub_ref['valuenum'] < sub_ref['ref_range_lower'])
            if low_mask.any():
                low = sub_ref[low_mask].copy()
                low['lab_id'] = f"LAB_{safe_name}_Low"
                low['lab_label'] = f"{lab_name} Low"
                lab_dfs.append(low[['subject_id', 'lab_id', 'lab_label']].drop_duplicates())
        else:
            norm = sub[['subject_id']].drop_duplicates().copy()
            norm['lab_id'] = f"LAB_{safe_name}"
            norm['lab_label'] = lab_name
            lab_dfs.append(norm)

    if not lab_dfs:
        print("  Labs: 0 patients, 0 lab result entities, 0 edges")
        return df

    all_labs = pd.concat(lab_dfs, ignore_index=True).drop_duplicates(subset=['subject_id', 'lab_id'])

    new_labs = 0
    new_edges = 0
    print(f"  Processing {len(all_labs)} unique patient-lab pairs...")
    # Vectorized: add lab entities in batch
    unique_lab_entities = all_labs[['lab_id', 'lab_label']].drop_duplicates(subset=['lab_id'])
    for _, row in unique_lab_entities.iterrows():
        is_new = row['lab_id'] not in kg.entities
        _add_entity_if_new(kg, row['lab_id'], "LabResult", "STR", row['lab_label'])
        if is_new:
            new_labs += 1

    # Add patient entities for any missing patients
    missing_patients = all_labs['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    # Add all relations at once
    for _, row in tqdm(all_labs.iterrows(), total=len(all_labs), desc="Lab results"):
        pat_id = f"PAT_{row['subject_id']}"
        kg.add_relation(Relation(head=pat_id, relation="has_lab", tail=row['lab_id'], weight=1.0))
        new_edges += 1

    indicates_edges = 0
    for lab_id, dis_id in LAB_TO_DISEASE.items():
        if lab_id not in kg.entities:
            continue
        if dis_id not in kg.entities:
            _add_entity_if_new(kg, dis_id, "Disease", "STR", dis_id.replace("DIS_", ""))
        kg.add_relation(Relation(head=lab_id, relation="indicates", tail=dis_id, weight=1.0))
        indicates_edges += 1

    print(f"  Labs: {all_labs['subject_id'].nunique()} patients, {new_labs} lab result entities, "
          f"{new_edges} has_lab edges, {indicates_edges} indicates edges")
    return df


def extract_vital_signs_bq(kg: ClinicalKG, client, patient_ids: List[int]) -> pd.DataFrame:
    print("Extracting vital signs from BigQuery...")
    id_list = ','.join(str(x) for x in patient_ids)

    vital_dataset = _resolve_bq_dataset(client, "physionet-data", ["mimiciv_3_1_derived", "mimiciv_derived"])

    canonical_cols = list(VITAL_COLUMN_ALIASES.keys())
    col_str = ', '.join(canonical_cols)
    print("  Querying BigQuery for vital signs...")
    sql = f"""
    SELECT subject_id, charttime, {col_str}
    FROM `physionet-data.{vital_dataset}.vitalsign`
    WHERE subject_id IN ({id_list})
    LIMIT 500000
    """
    df = _bq_query(client, sql)
    print(f"  Got {len(df)} vital sign rows from BigQuery")
    if df.empty:
        print("  No vital sign data returned")
        return df

    df['subject_id'] = df['subject_id'].astype(int)

    for canonical, aliases in VITAL_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                df[canonical] = df[alias]
                break

    vital_dfs = []
    for col, (abbr, high_thresh, low_thresh) in VITAL_SIGNS.items():
        if col not in df.columns:
            continue
        sub = df[['subject_id', col]].dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(float)
        if high_thresh is not None:
            high = sub[sub[col] > high_thresh].copy()
            high['vit_id'] = f"VIT_{abbr}_High"
            high['vit_label'] = f"{abbr} High"
            vital_dfs.append(high[['subject_id', 'vit_id', 'vit_label']].drop_duplicates())
        if low_thresh is not None:
            low = sub[sub[col] < low_thresh].copy()
            low['vit_id'] = f"VIT_{abbr}_Low"
            low['vit_label'] = f"{abbr} Low"
            vital_dfs.append(low[['subject_id', 'vit_id', 'vit_label']].drop_duplicates())

    if not vital_dfs:
        print("  Vitals: 0 patients, 0 vital result entities, 0 edges")
        return df

    all_vitals = pd.concat(vital_dfs, ignore_index=True).drop_duplicates(subset=['subject_id', 'vit_id'])

    new_vitals = 0
    new_edges = 0
    print(f"  Processing {len(all_vitals)} unique patient-vital pairs...")
    # Vectorized: add vital entities in batch
    unique_vital_entities = all_vitals[['vit_id', 'vit_label']].drop_duplicates(subset=['vit_id'])
    for _, row in unique_vital_entities.iterrows():
        is_new = row['vit_id'] not in kg.entities
        _add_entity_if_new(kg, row['vit_id'], "VitalResult", "STR", row['vit_label'])
        if is_new:
            new_vitals += 1

    # Add patient entities for any missing patients
    missing_patients = all_vitals['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    # Add all relations at once
    for _, row in tqdm(all_vitals.iterrows(), total=len(all_vitals), desc="Vital signs"):
        pat_id = f"PAT_{row['subject_id']}"
        kg.add_relation(Relation(head=pat_id, relation="has_vital", tail=row['vit_id'], weight=1.0))
        new_edges += 1

    indicates_edges = 0
    for vit_id, dis_id in VITAL_TO_DISEASE.items():
        if vit_id not in kg.entities:
            continue
        if dis_id not in kg.entities:
            _add_entity_if_new(kg, dis_id, "Disease", "STR", dis_id.replace("DIS_", ""))
        kg.add_relation(Relation(head=vit_id, relation="indicates", tail=dis_id, weight=1.0))
        indicates_edges += 1

    print(f"  Vitals: {all_vitals['subject_id'].nunique()} patients, {new_vitals} vital result entities, "
          f"{new_edges} has_vital edges, {indicates_edges} indicates edges")
    return df


def extract_ecg_measurements_bq(kg: ClinicalKG, client, patient_ids: List[int]) -> pd.DataFrame:
    print("Extracting ECG measurements from BigQuery...")
    id_list = ','.join(str(x) for x in patient_ids)

    ecg_dataset = _resolve_bq_dataset(client, "physionet-data", ["mimiciv_ecg", "mimiciv_3_1_ecg"])

    ECG_BQ_COLUMNS = ["rr_interval", "qrs_onset", "qrs_end", "p_onset", "p_end", "t_end", "p_axis", "qrs_axis", "t_axis"]
    meas_cols = ', '.join(ECG_BQ_COLUMNS)
    print("  Querying BigQuery for ECG measurements...")
    sql = f"""
    SELECT subject_id, study_id, {meas_cols}
    FROM `physionet-data.{ecg_dataset}.machine_measurements`
    WHERE subject_id IN ({id_list})
    LIMIT 500000
    """
    df = _bq_query(client, sql)
    print(f"  Got {len(df)} ECG measurement rows from BigQuery")
    if df.empty:
        print("  No ECG measurement data returned")
        return df

    df['subject_id'] = df['subject_id'].astype(int)
    df['study_id'] = df['study_id'].astype(int)

    for col_name, col_spec in ECG_COMPUTED_COLUMNS.items():
        source_ok = all(c in df.columns for c in col_spec["source_cols"])
        if source_ok:
            df[col_name] = col_spec["compute"](df)

    ecg_dfs = []
    for col, thresholds in ECG_MEASUREMENT_THRESHOLDS.items():
        if col not in df.columns:
            continue
        sub = df[['subject_id', 'study_id', col]].dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(float)
        high_val, high_id, high_label = thresholds['high']
        high = sub[sub[col] > high_val].copy()
        high['ecgm_id'] = high_id
        high['ecgm_label'] = high_label
        ecg_dfs.append(high[['subject_id', 'study_id', 'ecgm_id', 'ecgm_label']].drop_duplicates())
        if thresholds['low'] is not None:
            low_val, low_id, low_label = thresholds['low']
            low = sub[sub[col] < low_val].copy()
            low['ecgm_id'] = low_id
            low['ecgm_label'] = low_label
            ecg_dfs.append(low[['subject_id', 'study_id', 'ecgm_id', 'ecgm_label']].drop_duplicates())

    if not ecg_dfs:
        print("  ECG: 0 patients, 0 measurement entities, 0 edges")
        return df

    all_ecg = pd.concat(ecg_dfs, ignore_index=True).drop_duplicates(subset=['subject_id', 'study_id', 'ecgm_id'])

    new_meas = 0
    new_edges = 0
    unique_meas_entities = all_ecg[['ecgm_id', 'ecgm_label']].drop_duplicates(subset=['ecgm_id'])
    for _, row in unique_meas_entities.iterrows():
        is_new = row['ecgm_id'] not in kg.entities
        _add_entity_if_new(kg, row['ecgm_id'], "ECGMeasurement", "ECG", row['ecgm_label'])
        if is_new:
            new_meas += 1

    missing_patients = all_ecg['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    study_ids_seen = set()
    for _, row in tqdm(all_ecg.iterrows(), total=len(all_ecg), desc="ECG measurements"):
        pat_id = f"PAT_{row['subject_id']}"
        sty_id = f"STY_ECG_{row['study_id']}"
        if sty_id not in study_ids_seen:
            study_ids_seen.add(sty_id)
            _add_entity_if_new(kg, sty_id, "Study", "ECG", f"ECG Study {row['study_id']}")
            kg.add_relation(Relation(head=pat_id, relation="has_study", tail=sty_id, weight=1.0))
        kg.add_relation(Relation(head=pat_id, relation="has_ecg_measurement", tail=row['ecgm_id'], weight=1.0))
        kg.add_relation(Relation(head=row['ecgm_id'], relation="finding_of", tail=sty_id, weight=1.0))
        new_edges += 1

    indicates_edges = 0
    for ecgm_id, dis_id in ECG_MEASUREMENT_TO_DISEASE.items():
        if ecgm_id not in kg.entities:
            continue
        if dis_id not in kg.entities:
            _add_entity_if_new(kg, dis_id, "Disease", "ECG", dis_id.replace("DIS_", ""))
        kg.add_relation(Relation(head=ecgm_id, relation="indicates", tail=dis_id, weight=1.0))
        indicates_edges += 1

    print(f"  ECG measurements: {all_ecg['subject_id'].nunique()} patients, {new_meas} measurement entities, "
          f"{new_edges} has_ecg_measurement edges, {indicates_edges} indicates edges")

    print("  Querying BigQuery for ECG rhythm data (from report text)...")
    report_cols = ', '.join(f'report_{i}' for i in range(18))
    sql_rhythm = f"""
    SELECT subject_id, study_id, {report_cols}
    FROM `physionet-data.{ecg_dataset}.machine_measurements`
    WHERE subject_id IN ({id_list})
    """
    rhythm_df = _bq_query(client, sql_rhythm)
    print(f"  Got {len(rhythm_df)} ECG report rows from BigQuery")
    if not rhythm_df.empty:
        rhythm_df['subject_id'] = rhythm_df['subject_id'].astype(int)
        rhythm_df['study_id'] = rhythm_df['study_id'].astype(int)
        import re
        report_col_list = [f'report_{i}' for i in range(18)]
        rhythm_rows = []
        rhythm_patterns_re = {code: re.compile(r'\b' + re.escape(code) + r'\b') for code in RHYTHM_PATTERNS}
        for _, row in rhythm_df.iterrows():
            text = ' '.join(str(row[c]) for c in report_col_list if pd.notna(row.get(c))).upper()
            for rhy_code, pat in rhythm_patterns_re.items():
                if pat.search(text):
                    ecgr_id, ecgr_label = RHYTHM_PATTERNS[rhy_code]
                    rhythm_rows.append({
                        'subject_id': row['subject_id'],
                        'study_id': row['study_id'],
                        'ecgr_id': ecgr_id,
                        'ecgr_label': ecgr_label,
                    })
        if rhythm_rows:
            rhythm_entities = pd.DataFrame(rhythm_rows).drop_duplicates(subset=['subject_id', 'study_id', 'ecgr_id'])
            new_rhythm = 0
            rhythm_edges = 0
            unique_rhythm = rhythm_entities[['ecgr_id', 'ecgr_label']].drop_duplicates(subset=['ecgr_id'])
            for _, row in unique_rhythm.iterrows():
                is_new = row['ecgr_id'] not in kg.entities
                _add_entity_if_new(kg, row['ecgr_id'], "ECGRhythm", "ECG", row['ecgr_label'])
                if is_new:
                    new_rhythm += 1
            for _, row in tqdm(rhythm_entities.iterrows(), total=len(rhythm_entities), desc="ECG rhythm"):
                pat_id = f"PAT_{row['subject_id']}"
                sty_id = f"STY_ECG_{row['study_id']}"
                if sty_id not in kg.entities:
                    _add_entity_if_new(kg, sty_id, "Study", "ECG", f"ECG Study {row['study_id']}")
                    kg.add_relation(Relation(head=pat_id, relation="has_study", tail=sty_id, weight=1.0))
                kg.add_relation(Relation(head=pat_id, relation="has_ecg_measurement", tail=row['ecgr_id'], weight=1.0))
                kg.add_relation(Relation(head=row['ecgr_id'], relation="finding_of", tail=sty_id, weight=1.0))
                rhythm_edges += 1
            rhythm_indicates = 0
            for ecgr_id, dis_id in RHYTHM_TO_DISEASE.items():
                if ecgr_id not in kg.entities:
                    continue
                if dis_id not in kg.entities:
                    _add_entity_if_new(kg, dis_id, "Disease", "ECG", dis_id.replace("DIS_", ""))
                kg.add_relation(Relation(head=ecgr_id, relation="indicates", tail=dis_id, weight=1.0))
                rhythm_indicates += 1
            print(f"  ECG rhythm: {rhythm_entities['subject_id'].nunique()} patients, {new_rhythm} rhythm entities, "
                  f"{rhythm_edges} has_ecg_measurement edges, {rhythm_indicates} indicates edges")
        else:
            print("  ECG rhythm: 0 rhythm entities (no matching patterns)")
    else:
        print("  No ECG rhythm data returned")

    ecg_patient_studies: Dict[int, List[str]] = defaultdict(list)
    for rel in kg.relations:
        if rel.relation == "has_study" and rel.tail.startswith("STY_ECG_"):
            pat_id = rel.head
            if pat_id.startswith("PAT_"):
                ecg_patient_studies[int(pat_id[4:])].append(rel.tail)

    ecg_same_patient = 0
    for sid, studies in ecg_patient_studies.items():
        studies = studies[:20]
        for i in range(len(studies)):
            for j in range(i + 1, len(studies)):
                kg.add_relation(Relation(head=studies[i], relation="same_patient", tail=studies[j], weight=1.0))
                ecg_same_patient += 1
    if ecg_same_patient > 0:
        print(f"  Added {ecg_same_patient} ECG same_patient edges")

    return df


def extract_procedures_bq(kg: ClinicalKG, client, patient_ids: List[int],
                          crosswalks: Dict[str, Any]) -> pd.DataFrame:
    print("Extracting procedures from BigQuery...")
    icd_to_cui = crosswalks["icd_to_cui"]
    icd10pcs_to_cui = crosswalks.get("icd10pcs_to_cui", {})
    cui_to_snomed = crosswalks["cui_to_snomed"]
    snomed_id_to_name = crosswalks["snomed_id_to_name"]

    snomed_rel_types = ["has_focus", "has_intent", "direct_site", "procedure_site"]
    snomed_rels_map: Dict[str, List[Tuple[str, str]]] = {}
    for rel_name in snomed_rel_types:
        snomed_rels_map[rel_name] = crosswalks.get(rel_name, [])

    print("  Querying BigQuery for procedures ICD...")
    id_list = ','.join(str(x) for x in patient_ids)
    sql = f"""
    SELECT subject_id, hadm_id, icd_code, icd_version
    FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
    WHERE subject_id IN ({id_list})
    """
    df = _bq_query(client, sql)
    print(f"  Got {len(df)} procedure rows from BigQuery")
    if df.empty:
        print("  No procedure data returned")
        return df

    print("  Querying BigQuery for procedure ICD titles...")
    sql_d = """
    SELECT icd_code, icd_version, long_title
    FROM `physionet-data.mimiciv_3_1_hosp.d_icd_procedures`
    """
    d_icd = _bq_query(client, sql_d)
    icd_lookup: Dict[Tuple[str, str], str] = {}
    if not d_icd.empty:
        d_icd['icd_code'] = d_icd['icd_code'].astype(str)
        d_icd['icd_version'] = d_icd['icd_version'].astype(str)
        d_icd['long_title'] = d_icd['long_title'].astype(str)
        icd_lookup = dict(zip(zip(d_icd['icd_version'], d_icd['icd_code']), d_icd['long_title']))
    print(f"  Loaded {len(icd_lookup)} ICD procedure titles")

    df['subject_id'] = df['subject_id'].astype(int)
    df['icd_code'] = df['icd_code'].astype(str)
    df['icd_version'] = df['icd_version'].astype(str)
    df['proc_name'] = df.apply(
        lambda r: icd_lookup.get((r['icd_version'], r['icd_code']), f"ICD_Proc_{r['icd_code']}"), axis=1
    )
    df['sanitized'] = df['proc_name'].apply(_sanitize_name)
    df = df[df['sanitized'] != '']
    df['prc_id'] = 'PRC_' + df['sanitized']

    unique_pairs = df[['subject_id', 'prc_id', 'proc_name']].drop_duplicates(subset=['subject_id', 'prc_id'])

    new_procedures = 0
    new_edges = 0
    print(f"  Processing {len(unique_pairs)} unique patient-procedure pairs...")
    # Vectorized: add procedure entities in batch
    unique_procedure_entities = unique_pairs[['prc_id', 'proc_name']].drop_duplicates(subset=['prc_id'])
    for _, row in unique_procedure_entities.iterrows():
        is_new = row['prc_id'] not in kg.entities
        _add_entity_if_new(kg, row['prc_id'], "Procedure", "STR", row['proc_name'])
        if is_new:
            new_procedures += 1

    # Add patient entities for any missing patients
    missing_patients = unique_pairs['subject_id'].unique()
    for sid in missing_patients:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    # Add all relations at once
    for _, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Procedures"):
        pat_id = f"PAT_{row['subject_id']}"
        kg.add_relation(Relation(head=pat_id, relation="underwent", tail=row['prc_id'], weight=1.0))
        new_edges += 1

    # SNOMED relationships: procedure→disease for each rel type
    total_snomed_edges = 0
    for rel_name in snomed_rel_types:
        rel_pairs = snomed_rels_map[rel_name]
        if not rel_pairs:
            continue
        src_to_dsts: Dict[str, Set[str]] = defaultdict(set)
        for src, dst in rel_pairs:
            src_to_dsts[src].add(dst)

        icd10_codes = df[df['icd_version'] == '10'][['icd_code']].drop_duplicates()
        rel_edges = 0
        for _, row in icd10_codes.iterrows():
            cui = icd_to_cui.get(f"ICD10CM:{row['icd_code']}") or icd10pcs_to_cui.get(row['icd_code'])
            if not cui:
                continue
            snomed_id = cui_to_snomed.get(cui)
            if not snomed_id or snomed_id not in src_to_dsts:
                continue
            proc_name = snomed_id_to_name.get(snomed_id, snomed_id)
            prc_id = 'PRC_' + _sanitize_name(proc_name)
            if prc_id not in kg.entities:
                continue
            for dst in src_to_dsts[snomed_id]:
                dis_name = snomed_id_to_name.get(dst, dst)
                dis_id = 'DIS_' + _sanitize_name(dis_name)
                if dis_id not in kg.entities:
                    _add_entity_if_new(kg, dis_id, "Disease", "STR", dis_name)
                kg.add_relation(Relation(head=prc_id, relation=rel_name, tail=dis_id, weight=1.0))
                rel_edges += 1
        total_snomed_edges += rel_edges
        print(f"    {rel_name}: {rel_edges} edges")

    print(f"  Procedures: {unique_pairs['subject_id'].nunique()} patients, {new_procedures} procedures, "
          f"{new_edges} underwent edges, {total_snomed_edges} SNOMED edges")
    return df


def build_drug_disease_edges(kg: ClinicalKG, client, patient_ids: List[int],
                             crosswalks: Dict[str, Any], diag_df: pd.DataFrame,
                             rx_df: pd.DataFrame) -> None:
    print("Building Drug→Disease edges...")

    RELA_TO_RELATION = {
        "may_treat": "treats",
        "may_prevent": "prevents",
        "contraindicated_with_disease": "contraindicated",
    }

    csv_path = Path(os.environ.get("MRREL_DRUG_DISEASE_CSV", r"C:\Users\Noodl\Projects\Research\MultiModal\bq_results\mrrel_mimic_drug_disease_filtered.csv"))

    icd_to_cui = crosswalks.get("icd_to_cui", {})
    cui_to_name = crosswalks.get("cui_to_name", {})

    drug_entities = {eid: entity for eid, entity in kg.entities.items() if entity.type == "Drug"}
    cui_to_drug_entity: Dict[str, str] = {}

    drug_name_lower_to_eid: Dict[str, str] = {}
    drug_word_index: Dict[str, List[str]] = defaultdict(list)
    for eid, entity in drug_entities.items():
        name_lower = entity.label.lower().strip()
        drug_name_lower_to_eid[name_lower] = eid
        for word in name_lower.split():
            if len(word) >= 2:
                drug_word_index[word].append(eid)

    csv_drug_cuis = set()
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                csv_drug_cuis.add(row['CUI1'])

    for cui in csv_drug_cuis:
        if cui not in cui_to_name:
            continue
        umls_name = cui_to_name[cui]
        sanitized = 'DRG_' + _sanitize_name(umls_name)

        if sanitized in drug_entities:
            cui_to_drug_entity[cui] = sanitized
            continue

        umls_lower = umls_name.lower().strip()
        matched = False

        prefix_candidates = set(drug_word_index.get(umls_lower.split()[0], [])) if umls_lower.split() else set()
        for eid in prefix_candidates:
            label_lower = drug_entities[eid].label.lower().strip()
            if label_lower.startswith(umls_lower) or label_lower == umls_lower:
                cui_to_drug_entity[cui] = eid
                matched = True
                break

        if not matched:
            umls_words = set(umls_lower.split())
            umls_words = {w for w in umls_words if len(w) >= 2}
            if umls_words:
                candidate_eids: Dict[str, int] = defaultdict(int)
                for w in umls_words:
                    for eid in drug_word_index.get(w, []):
                        candidate_eids[eid] += 1
                for eid, count in sorted(candidate_eids.items(), key=lambda x: -x[1]):
                    label_words = {w for w in drug_entities[eid].label.lower().split() if len(w) >= 2}
                    if umls_words.issubset(label_words):
                        cui_to_drug_entity[cui] = eid
                        matched = True
                        break

        if not matched:
            cui_to_drug_entity[cui] = _add_entity_if_new(kg, sanitized, "Drug", "STR", umls_name).id

    cui_to_name = crosswalks.get("cui_to_name", {})
    dis_name_to_eid: Dict[str, str] = {}
    dis_word_index: Dict[str, List[str]] = defaultdict(list)
    dis_eid_to_label: Dict[str, str] = {}
    _STOP = {'of', 'the', 'and', 'in', 'with', 'by', 'from', 'to', 'a', 'an', 'or', 'nos', 'unspecified'}
    for eid, ent in kg.entities.items():
        if ent.type == "Disease":
            dis_name_to_eid[ent.label.lower()] = eid
            dis_eid_to_label[eid] = ent.label.lower()
            for word in ent.label.lower().split():
                if len(word) >= 2 and word not in _STOP:
                    dis_word_index[word].append(eid)

    icd_to_dis_name = crosswalks.get("icd_to_dis_name", {})
    cui_to_dis_entity: Dict[str, str] = {}
    for icd_key, dis_id in icd_to_dis_name.items():
        if dis_id in kg.entities:
            icd_code, icd_version = icd_key.rsplit('_', 1)
            cui = icd_to_cui.get(f"ICD{icd_version}CM:{icd_code}")
            if cui:
                cui_to_dis_entity[cui] = dis_id

    for cui, umls_name in cui_to_name.items():
        if cui in cui_to_dis_entity:
            continue
        sanitized = _sanitize_name(umls_name)
        eid = f"DIS_{sanitized}"
        if eid in kg.entities:
            cui_to_dis_entity[cui] = eid
        else:
            lower_name = umls_name.lower()
            if lower_name in dis_name_to_eid:
                cui_to_dis_entity[cui] = dis_name_to_eid[lower_name]
            else:
                query_words = {w for w in lower_name.split() if len(w) >= 2 and w not in _STOP}
                if query_words:
                    candidate_eids: Dict[str, int] = defaultdict(int)
                    for w in query_words:
                        for ceid in dis_word_index.get(w, []):
                            candidate_eids[ceid] += 1
                    best_eid, best_score = None, 0.0
                    for ceid, overlap in candidate_eids.items():
                        cand_words = {w for w in dis_eid_to_label[ceid].split() if len(w) >= 2 and w not in _STOP}
                        jaccard = len(query_words & cand_words) / len(query_words | cand_words)
                        if jaccard > best_score:
                            best_score = jaccard
                            best_eid = ceid
                    if best_eid and best_score >= 0.5:
                        cui_to_dis_entity[cui] = best_eid

    ontology_edges = 0
    edge_counts: Dict[str, int] = defaultdict(int)

    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                cui1 = row['CUI1']
                cui2 = row['CUI2']
                rela = row['RELA']

                drug_eid = cui_to_drug_entity.get(cui1)
                dis_eid = cui_to_dis_entity.get(cui2)

                if not drug_eid or not dis_eid:
                    continue
                if drug_eid not in kg.entities or dis_eid not in kg.entities:
                    continue

                rel_name = RELA_TO_RELATION.get(rela, "treats")
                kg.add_relation(Relation(head=drug_eid, relation=rel_name, tail=dis_eid, weight=1.0, source="umls"))
                ontology_edges += 1
                edge_counts[rel_name] += 1

        print(f"  Ontology Drug→Disease (source=umls): {ontology_edges} edges")
        for rel, cnt in sorted(edge_counts.items()):
            print(f"    {rel}: {cnt}")
    else:
        print(f"  WARNING: {csv_path} not found")


def build_snomed_disease_hierarchy(kg: ClinicalKG, crosswalks: Dict[str, Any]) -> None:
    print("Building SNOMED CT disease hierarchy...")
    is_a_rels = crosswalks["is_a"]
    snomed_id_to_name = crosswalks["snomed_id_to_name"]
    snomed_to_cui = crosswalks["snomed_to_cui"]

    kg_disease_names: Set[str] = set()
    for eid, entity in kg.entities.items():
        if entity.type == "Disease" and eid.startswith("DIS_"):
            kg_disease_names.add(eid)

    kg_disease_snomed_ids: Set[str] = set()
    name_to_snomed: Dict[str, str] = {}
    for snomed_id, name in snomed_id_to_name.items():
        sanitized = 'DIS_' + _sanitize_name(name)
        if sanitized in kg_disease_names:
            kg_disease_snomed_ids.add(snomed_id)
            name_to_snomed[sanitized] = snomed_id

    subsumes_edges = 0
    for src, dst in tqdm(is_a_rels, desc="SNOMED is_a"):
        if src in kg_disease_snomed_ids and dst in kg_disease_snomed_ids:
            src_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(src, src))
            dst_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(dst, dst))
            if src_name in kg.entities and dst_name in kg.entities:
                kg.add_relation(Relation(head=dst_name, relation="subsumes", tail=src_name, weight=1.0))
                subsumes_edges += 1

    due_to_rels = crosswalks["due_to"]
    due_to_edges = 0
    for src, dst in due_to_rels:
        if src in kg_disease_snomed_ids and dst in kg_disease_snomed_ids:
            src_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(src, src))
            dst_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(dst, dst))
            if src_name in kg.entities and dst_name in kg.entities:
                kg.add_relation(Relation(head=src_name, relation="due_to", tail=dst_name, weight=1.0))
                due_to_edges += 1

    assoc_rels = crosswalks["associated_finding"]
    assoc_edges = 0
    for src, dst in assoc_rels:
        if src in kg_disease_snomed_ids and dst in kg_disease_snomed_ids:
            src_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(src, src))
            dst_name = 'DIS_' + _sanitize_name(snomed_id_to_name.get(dst, dst))
            if src_name in kg.entities and dst_name in kg.entities:
                kg.add_relation(Relation(head=src_name, relation="associated_with", tail=dst_name, weight=1.0))
                assoc_edges += 1

    print(f"  SNOMED hierarchy: {subsumes_edges} subsumes, {due_to_edges} due_to, {assoc_edges} associated_with")


def build_temporal_edges_bq(kg: ClinicalKG, client, patient_ids: List[int],
                            diag_df: Optional[pd.DataFrame] = None,
                            admissions_df: Optional[pd.DataFrame] = None,
                            lab_df: Optional[pd.DataFrame] = None,
                            vit_df: Optional[pd.DataFrame] = None,
                            rx_df: Optional[pd.DataFrame] = None,
                            proc_df: Optional[pd.DataFrame] = None) -> None:
    print("Building temporal edges from BigQuery...")
    id_list = ','.join(str(x) for x in patient_ids)

    if admissions_df is not None and not admissions_df.empty:
        print("  Using passed admissions DataFrame for temporal edges...")
        admissions = admissions_df.copy()
        admissions['subject_id'] = admissions['subject_id'].astype(int)
        admissions['hadm_id'] = admissions['hadm_id'].astype(int)
        admissions['admittime'] = admissions['admittime'].apply(_parse_datetime)
        admissions['dischtime'] = admissions['dischtime'].apply(_parse_datetime)
    else:
        print("  Querying BigQuery for admissions...")
        sql_adm = f"""
        SELECT subject_id, hadm_id, admittime, dischtime
        FROM `physionet-data.mimiciv_3_1_hosp.admissions`
        WHERE subject_id IN ({id_list})
        """
        admissions = _bq_query(client, sql_adm)
        print(f"  Got {len(admissions)} admission rows from BigQuery")
        if admissions.empty:
            print("  No admission data returned")
            return

        admissions['subject_id'] = admissions['subject_id'].astype(int)
        admissions['hadm_id'] = admissions['hadm_id'].astype(int)
        admissions['admittime'] = admissions['admittime'].apply(_parse_datetime)
        admissions['dischtime'] = admissions['dischtime'].apply(_parse_datetime)

    # Vectorized: build hadm_times lookup
    hadm_times: Dict[int, Tuple[Optional[datetime], Optional[datetime]]] = dict(zip(
        admissions['hadm_id'],
        zip(admissions['admittime'], admissions['dischtime'])
    ))

    event_frames = []

    # Diagnoses — use passed diag_df instead of re-querying
    if diag_df is not None and not diag_df.empty:
        print("  Using passed diagnosis DataFrame for temporal edges...")
        temporal_diag = diag_df[['subject_id', 'hadm_id', 'dis_id']].copy()
        temporal_diag = temporal_diag.rename(columns={'dis_id': 'entity_id'})
        temporal_diag = temporal_diag[temporal_diag['entity_id'].isin(kg.entities)]
        temporal_diag['timestamp'] = temporal_diag['hadm_id'].map(lambda h: hadm_times.get(h, (None, None))[0])
        temporal_diag = temporal_diag.dropna(subset=['timestamp'])
        if not temporal_diag.empty:
            event_frames.append(temporal_diag[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])
    else:
        print("  Querying BigQuery for diagnoses (no diag_df passed)...")
        sql_diag = f"""
        SELECT subject_id, hadm_id, icd_code
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
        WHERE subject_id IN ({id_list})
        """
        diag_bq = _bq_query(client, sql_diag)
        print(f"  Got {len(diag_bq)} diagnosis rows from BigQuery")
        if not diag_bq.empty:
            diag_bq['subject_id'] = diag_bq['subject_id'].astype(int)
            diag_bq['hadm_id'] = diag_bq['hadm_id'].astype(int)
            diag_bq['icd_code'] = diag_bq['icd_code'].astype(str)
            diag_bq['entity_id'] = 'DIS_ICD_' + diag_bq['icd_code']
            diag_bq = diag_bq[diag_bq['entity_id'].isin(kg.entities)]
            diag_bq['timestamp'] = diag_bq['hadm_id'].map(lambda h: hadm_times.get(h, (None, None))[0])
            diag_bq = diag_bq.dropna(subset=['timestamp'])
            if not diag_bq.empty:
                event_frames.append(diag_bq[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    # Lab events
    clinical_itemids = list(CLINICAL_LABS.keys())
    itemid_str = ','.join(str(x) for x in clinical_itemids)
    if lab_df is not None and not lab_df.empty:
        print("  Using passed lab events DataFrame for temporal edges...")
        lab_df = lab_df.copy()
    else:
        print("  Querying BigQuery for lab events (temporal)...")
        sql_lab = f"""
        SELECT subject_id, hadm_id, itemid, charttime, valuenum, ref_range_lower, ref_range_upper
        FROM `physionet-data.mimiciv_3_1_hosp.labevents`
        WHERE subject_id IN ({id_list})
        AND itemid IN ({itemid_str})
        AND valuenum IS NOT NULL
        LIMIT 500000
        """
        lab_df = _bq_query(client, sql_lab)
        print(f"  Got {len(lab_df)} lab event rows from BigQuery (temporal)")
    if not lab_df.empty:
        lab_df['subject_id'] = lab_df['subject_id'].astype(int)
        lab_df['itemid'] = lab_df['itemid'].astype(int)
        lab_df['valuenum'] = lab_df['valuenum'].astype(float)
        lab_parts = []
        for itemid, lab_name in CLINICAL_LABS.items():
            sub = lab_df[lab_df['itemid'] == itemid]
            if sub.empty:
                continue
            safe_name = _sanitize_name(lab_name)
            has_ref = sub['ref_range_upper'].notna() | sub['ref_range_lower'].notna()
            sub_ref = sub[has_ref]
            if not sub_ref.empty:
                high_mask = sub_ref['ref_range_upper'].notna() & (sub_ref['valuenum'] > sub_ref['ref_range_upper'])
                if high_mask.any():
                    high = sub_ref[high_mask].copy()
                    high['entity_id'] = f"LAB_{safe_name}_High"
                    lab_parts.append(high)
                low_mask = sub_ref['ref_range_lower'].notna() & (sub_ref['valuenum'] < sub_ref['ref_range_lower'])
                if low_mask.any():
                    low = sub_ref[low_mask].copy()
                    low['entity_id'] = f"LAB_{safe_name}_Low"
                    lab_parts.append(low)
            else:
                norm = sub.copy()
                norm['entity_id'] = f"LAB_{safe_name}"
                lab_parts.append(norm)
        if lab_parts:
            lab_all = pd.concat(lab_parts, ignore_index=True)
            lab_all = lab_all[lab_all['entity_id'].isin(kg.entities)]
            lab_all['timestamp'] = lab_all['charttime'].apply(_parse_datetime)
            lab_all = lab_all.dropna(subset=['timestamp'])
            lab_all['hadm_id'] = lab_all['hadm_id'].fillna(0).astype(int)
            if not lab_all.empty:
                event_frames.append(lab_all[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    # Vital signs
    if vit_df is not None and not vit_df.empty:
        print("  Using passed vital signs DataFrame for temporal edges...")
        vit_df = vit_df.copy()
    else:
        vital_dataset = _resolve_bq_dataset(client, "physionet-data", ["mimiciv_3_1_derived", "mimiciv_derived"])
        real_vital_cols = {"heart_rate", "sbp", "dbp", "mbp", "sbp_ni", "dbp_ni", "mbp_ni",
                           "resp_rate", "temperature", "spo2", "glucose"}
        all_vital_cols = set()
        for aliases in VITAL_COLUMN_ALIASES.values():
            for a in aliases:
                if a in real_vital_cols:
                    all_vital_cols.add(a)
        vital_col_str = ', '.join(sorted(all_vital_cols))
        print("  Querying BigQuery for vital signs (temporal)...")
        sql_vit = f"""
        SELECT v.subject_id, i.hadm_id, v.charttime, {vital_col_str}
        FROM `physionet-data.{vital_dataset}.vitalsign` v
        LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
        ON v.stay_id = i.stay_id
        WHERE v.subject_id IN ({id_list})
        LIMIT 500000
        """
        vit_df = _bq_query(client, sql_vit)
        print(f"  Got {len(vit_df)} vital sign rows from BigQuery (temporal)")
    if not vit_df.empty:
        vit_df['subject_id'] = vit_df['subject_id'].astype(int)
        if 'hadm_id' not in vit_df.columns:
            vit_df['hadm_id'] = pd.NA

        for canonical, aliases in VITAL_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in vit_df.columns and canonical not in vit_df.columns:
                    vit_df[canonical] = vit_df[alias]
                    break
        vit_parts = []
        for col, (abbr, high_thresh, low_thresh) in VITAL_SIGNS.items():
            if col not in vit_df.columns:
                continue
            sub = vit_df[['subject_id', 'hadm_id', 'charttime', col]].dropna(subset=[col]).copy()
            sub[col] = sub[col].astype(float)
            if high_thresh is not None:
                high = sub[sub[col] > high_thresh].copy()
                high['entity_id'] = f"VIT_{abbr}_High"
                vit_parts.append(high)
            if low_thresh is not None:
                low = sub[sub[col] < low_thresh].copy()
                low['entity_id'] = f"VIT_{abbr}_Low"
                vit_parts.append(low)
        if vit_parts:
            vit_all = pd.concat(vit_parts, ignore_index=True)
            vit_all = vit_all[vit_all['entity_id'].isin(kg.entities)]
            vit_all['timestamp'] = vit_all['charttime'].apply(_parse_datetime)
            vit_all = vit_all.dropna(subset=['timestamp'])
            vit_all['hadm_id'] = vit_all['hadm_id'].fillna(0).astype(int)
            if not vit_all.empty:
                event_frames.append(vit_all[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    # Medications
    if rx_df is not None and not rx_df.empty:
        print("  Using passed prescriptions DataFrame for temporal edges...")
        rx_df = rx_df.copy()
    else:
        print("  Querying BigQuery for prescriptions (temporal)...")
        sql_rx = f"""
        SELECT subject_id, hadm_id, drug, starttime
        FROM `physionet-data.mimiciv_3_1_hosp.prescriptions`
        WHERE subject_id IN ({id_list})
        AND drug IS NOT NULL
        LIMIT 500000
        """
        rx_df = _bq_query(client, sql_rx)
        print(f"  Got {len(rx_df)} prescription rows from BigQuery (temporal)")
    if not rx_df.empty:
        rx_df['subject_id'] = rx_df['subject_id'].astype(int)
        if 'drug_str' not in rx_df.columns:
            rx_df['drug_str'] = rx_df['drug'].astype(str).str.strip()
            rx_df = rx_df[rx_df['drug_str'] != '']
        if 'entity_id' not in rx_df.columns:
            if 'drg_id' in rx_df.columns:
                rx_df['entity_id'] = rx_df['drg_id']
            else:
                rx_df['entity_id'] = 'DRG_' + rx_df['drug_str'].apply(_sanitize_name)
        rx_df = rx_df[rx_df['entity_id'].isin(kg.entities)]
        if 'starttime' in rx_df.columns:
            rx_df['timestamp'] = rx_df['starttime'].apply(_parse_datetime)
        else:
            rx_df['timestamp'] = rx_df['hadm_id'].map(lambda h: hadm_times.get(h, (None, None))[0])
        rx_df = rx_df.dropna(subset=['timestamp'])
        rx_df['hadm_id'] = rx_df['hadm_id'].fillna(0).astype(int)
        if not rx_df.empty:
            event_frames.append(rx_df[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    # Procedures
    if proc_df is not None and not proc_df.empty:
        print("  Using passed procedures DataFrame for temporal edges...")
        proc_df = proc_df.copy()
    else:
        print("  Querying BigQuery for procedures (temporal)...")
        sql_proc = f"""
        SELECT p.subject_id, p.hadm_id, p.icd_code, p.icd_version
        FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` p
        WHERE p.subject_id IN ({id_list})
        """
        proc_df = _bq_query(client, sql_proc)
        print(f"  Got {len(proc_df)} procedure rows from BigQuery (temporal)")
    if not proc_df.empty:
        proc_df['subject_id'] = proc_df['subject_id'].astype(int)
        proc_df['icd_code'] = proc_df['icd_code'].astype(str)
        proc_df['icd_version'] = proc_df['icd_version'].astype(str)
        if 'proc_name' not in proc_df.columns:
            print("  Querying BigQuery for procedure ICD titles (temporal)...")
            sql_d = "SELECT icd_code, icd_version, long_title FROM `physionet-data.mimiciv_3_1_hosp.d_icd_procedures`"
            d_icd = _bq_query(client, sql_d)
            icd_lookup: Dict[Tuple[str, str], str] = {}
            if not d_icd.empty:
                d_icd['icd_code'] = d_icd['icd_code'].astype(str)
                d_icd['icd_version'] = d_icd['icd_version'].astype(str)
                d_icd['long_title'] = d_icd['long_title'].astype(str)
                icd_lookup = dict(zip(zip(d_icd['icd_version'], d_icd['icd_code']), d_icd['long_title']))
            proc_df['proc_name'] = proc_df.apply(
                lambda r: icd_lookup.get((r['icd_version'], r['icd_code']), f"ICD_Proc_{r['icd_code']}"), axis=1
            )
        if 'entity_id' not in proc_df.columns:
            if 'prc_id' in proc_df.columns:
                proc_df['entity_id'] = proc_df['prc_id']
            else:
                proc_df['entity_id'] = 'PRC_' + proc_df['proc_name'].apply(_sanitize_name)
        proc_df = proc_df[proc_df['entity_id'].isin(kg.entities)]
        proc_df['timestamp'] = proc_df['hadm_id'].map(lambda h: hadm_times.get(h, (None, None))[0])
        proc_df = proc_df.dropna(subset=['timestamp'])
        proc_df['hadm_id'] = proc_df['hadm_id'].astype(int)
        if not proc_df.empty:
            event_frames.append(proc_df[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    print("  Querying BigQuery for ECG studies (temporal)...")
    ecg_dataset_temp = _resolve_bq_dataset(client, "physionet-data", ["mimiciv_ecg", "mimiciv_3_1_ecg"])
    sql_ecg_rec = f"""
    SELECT r.subject_id, i.hadm_id, r.study_id, r.ecg_time
    FROM `physionet-data.{ecg_dataset_temp}.record_list` r
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
    ON r.subject_id = i.subject_id
    WHERE r.subject_id IN ({id_list})
    """
    ecg_rec_df = _bq_query(client, sql_ecg_rec)
    print(f"  Got {len(ecg_rec_df)} ECG record rows from BigQuery (temporal)")
    if not ecg_rec_df.empty:
        ecg_rec_df['subject_id'] = ecg_rec_df['subject_id'].astype(int)
        ecg_rec_df['hadm_id'] = ecg_rec_df['hadm_id'].fillna(0).astype(int)
        ecg_rec_df['timestamp'] = ecg_rec_df['ecg_time'].apply(_parse_datetime)
        ecg_rec_df = ecg_rec_df.dropna(subset=['timestamp'])
        ecg_entity_rows = []
        ecg_entity_types = {"ECGMeasurement", "ECGRhythm"}
        for rel in kg.relations:
            if rel.relation == "finding_of" and rel.head in kg.entities and kg.entities[rel.head].type in ecg_entity_types:
                ecg_entity_rows.append({'entity_id': rel.head, 'study_key': rel.tail})
        if ecg_entity_rows:
            ecg_ent_df = pd.DataFrame(ecg_entity_rows)
            ecg_rec_df['study_key'] = 'STY_ECG_' + ecg_rec_df['study_id'].astype(str)
            ecg_merged = ecg_ent_df.merge(ecg_rec_df[['subject_id', 'hadm_id', 'timestamp', 'study_key']], on='study_key', how='left')
            ecg_merged = ecg_merged.dropna(subset=['timestamp'])
            ecg_merged = ecg_merged[ecg_merged['entity_id'].isin(kg.entities)]
            if not ecg_merged.empty:
                event_frames.append(ecg_merged[['subject_id', 'hadm_id', 'timestamp', 'entity_id']])

    if not event_frames:
        print("  Temporal: 0 admissions with temporal edges")
        return

    all_events = pd.concat(event_frames, ignore_index=True)
    all_events = all_events[all_events['hadm_id'] != 0]

    total_before = 0
    admissions_with_temporal = 0

    for (_, hadm_id), group in tqdm(all_events.groupby(['subject_id', 'hadm_id']), desc="Temporal edges"):
        if len(group) < 2:
            continue
        sorted_events = group.sort_values('timestamp')
        entities = sorted_events['entity_id'].tolist()
        timestamps = sorted_events['timestamp'].tolist()
        edge_count = 0
        for i in range(len(entities) - 1):
            if edge_count >= 10:
                break
            if timestamps[i] < timestamps[i + 1]:
                kg.add_relation(Relation(head=entities[i], relation="before", tail=entities[i + 1], weight=1.0))
                total_before += 1
                edge_count += 1
        admissions_with_temporal += 1

    print(f"  Temporal: {admissions_with_temporal} admissions, {total_before} before edges")


def ensure_patient_finding_edges(kg: ClinicalKG, data_dir: Path) -> None:
    print("Ensuring patient-finding edges...")

    study_to_findings: Dict[str, Set[str]] = defaultdict(set)
    existing_edges: Set[Tuple[str, str]] = set()

    patient_entities = {eid for eid, e in kg.entities.items() if e.type == "Patient"}

    for rel in kg.relations:
        if rel.relation == "finding_of":
            study_to_findings[rel.tail].add(rel.head)
        elif rel.relation == "has_finding":
            existing_edges.add((rel.head, rel.tail))

    study_to_patient: Dict[str, str] = {}

    for cxr_path in [
        data_dir / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-chexpert.csv",
        data_dir / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-chexpert.csv.gz",
        data_dir / "mimic-cxr-2.0.0-metadata.csv",
    ]:
        if cxr_path.exists():
            try:
                cxr_df = pd.read_csv(cxr_path, compression='gzip' if str(cxr_path).endswith('.gz') else None)
                if 'subject_id' in cxr_df.columns and 'study_id' in cxr_df.columns:
                    for _, row in cxr_df.dropna(subset=['subject_id', 'study_id']).iterrows():
                        sty_id = f"STY_{int(row['study_id'])}"
                        pat_id = f"PAT_{int(row['subject_id'])}"
                        if sty_id in study_to_findings and pat_id in patient_entities:
                            study_to_patient[sty_id] = pat_id
                break
            except Exception:
                pass

    for rel in kg.relations:
        if rel.relation == "has_study" and rel.tail.startswith("STY_ECG_"):
            pat_id = rel.head
            sty_id = rel.tail
            if sty_id in study_to_findings and pat_id in patient_entities:
                study_to_patient[sty_id] = pat_id

    new_edges = 0
    for sty_id, pat_id in study_to_patient.items():
        for fnd_id in study_to_findings.get(sty_id, set()):
            if (pat_id, fnd_id) not in existing_edges:
                kg.add_relation(Relation(head=pat_id, relation="has_finding", tail=fnd_id, weight=1.0))
                existing_edges.add((pat_id, fnd_id))
                new_edges += 1

    print(f"  Added {new_edges} new has_finding edges")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Save
# ═══════════════════════════════════════════════════════════════════════════════

def save_efficient_extended(kg: ClinicalKG, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    sorted_entities = sorted(
        kg.entities.values(),
        key=lambda e: (ENTITY_TYPE_ORDER.index(e.type) if e.type in ENTITY_TYPE_ORDER else len(ENTITY_TYPE_ORDER), e.id),
    )

    entity_id_map: Dict[str, int] = {e.id: i for i, e in enumerate(sorted_entities)}
    entity_labels: List[str] = [e.label for e in sorted_entities]
    entity_type_ids = [ENTITY_TYPE_TO_ID.get(e.type, 0) for e in sorted_entities]
    entity_modality_ids = [
        MODALITY_TO_ID.get(e.modality if e.modality and e.modality != "None" else None, len(MODALITY_TO_ID) - 1)
        for e in sorted_entities
    ]

    entities_arr = np.array(
        [list(range(len(sorted_entities))), entity_type_ids, entity_modality_ids, list(range(len(sorted_entities)))],
        dtype=np.int32,
    ).T

    all_relation_names = sorted(set(r.relation for r in kg.relations))
    relation2id = {r: i for i, r in enumerate(all_relation_names)}

    rel_data = []
    for rel in kg.relations:
        if rel.head in entity_id_map and rel.tail in entity_id_map:
            rel_data.append((entity_id_map[rel.head], relation2id[rel.relation], entity_id_map[rel.tail], rel.weight))

    if rel_data:
        relations_arr = np.array(rel_data, dtype=np.float32)
        relations_out = np.zeros(len(rel_data), dtype=[('head', np.int32), ('relation', np.int32), ('tail', np.int32), ('weight', np.float32)])
        relations_out['head'] = relations_arr[:, 0].astype(np.int32)
        relations_out['relation'] = relations_arr[:, 1].astype(np.int32)
        relations_out['tail'] = relations_arr[:, 2].astype(np.int32)
        relations_out['weight'] = relations_arr[:, 3].astype(np.float32)
    else:
        relations_out = np.zeros(0, dtype=[('head', np.int32), ('relation', np.int32), ('tail', np.int32), ('weight', np.float32)])

    metadata = {
        "entity_id_map": entity_id_map,
        "entity_labels": entity_labels,
        "relation_names": all_relation_names,
        "entity_type_names": ENTITY_TYPE_ORDER,
        "modality_names": ["CXR", "ECG", "RAD", "STR", "None"],
        "entity_type_counts": kg.entity_type_counts,
        "relation_type_counts": kg.relation_type_counts,
        "enhanced": True,
        "new_relation_types": ["diagnosed_with", "prescribed", "has_lab", "has_vital", "underwent",
                                "before", "treats", "has_focus", "due_to", "associated_with"],
    }

    np.save(directory / "entities.npy", entities_arr)
    np.save(directory / "relations.npy", relations_out)
    with open(directory / "metadata.json", 'w') as f:
        json.dump(metadata, f)

    print(f"  Saved efficient format to {directory}/")
    print(f"    entities.npy: {entities_arr.shape}, relations.npy: {relations_out.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_enhanced_kg(data_dir: Optional[Path] = None, output_dir: Optional[Path] = None,
                      n_patients: int = N_PATIENTS) -> ClinicalKG:
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("Building enhanced clinical KG (BigQuery + SNOMED CT)...")

    # Phase 0: Crosswalks
    print("\n=== Phase 0: Loading crosswalks ===")
    crosswalks = load_crosswalks(output_dir)

    # Phase 1: Base KG from existing pipeline
    print("\n=== Phase 1: Base KG extraction ===")
    kg = ClinicalKG()

    chexpert_df, _ = extract_chexpert_findings(kg, data_dir)
    _, ptbxl_findings = extract_ptbxl_findings(kg, data_dir)
    extract_radgraph_findings(kg, data_dir)
    link_radgraph_to_cxr(kg, data_dir)
    build_disease_subsumption(kg)
    build_same_patient_edges(kg, chexpert_df, data_dir)
    build_cross_modal_edges(kg, chexpert_df, data_dir)

    print(f"  Base KG: {len(kg.entities)} entities, {len(kg.relations)} relations")

    # Phase 2: BigQuery enrichment
    print("\n=== Phase 2: BigQuery enrichment ===")
    client = get_bq_client()
    patient_ids = get_top_patient_ids(client, n_patients)

    # Ensure all BQ patients exist in KG
    for sid in patient_ids:
        pat_id = f"PAT_{sid}"
        if pat_id not in kg.entities:
            _add_entity_if_new(kg, pat_id, "Patient", "STR", str(sid))

    diag_df = extract_diagnoses_bq(kg, client, patient_ids, crosswalks)
    rx_df = extract_medications_bq(kg, client, patient_ids, crosswalks)
    lab_df = extract_lab_results_bq(kg, client, patient_ids)
    vit_df = extract_vital_signs_bq(kg, client, patient_ids)
    ecg_df = extract_ecg_measurements_bq(kg, client, patient_ids)
    proc_df = extract_procedures_bq(kg, client, patient_ids, crosswalks)
    build_drug_disease_edges(kg, client, patient_ids, crosswalks, diag_df, rx_df)
    build_snomed_disease_hierarchy(kg, crosswalks)
    build_temporal_edges_bq(kg, client, patient_ids, diag_df=diag_df, rx_df=rx_df,
                            lab_df=lab_df, vit_df=vit_df, proc_df=proc_df)
    ensure_patient_finding_edges(kg, data_dir)

    # Phase 3: Save
    print("\n=== Phase 3: Saving ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "clinical_kg_enhanced.pkl"
    kg.save(pkl_path)
    print(f"  Saved pickle to {pkl_path}")

    save_efficient_extended(kg, output_dir)

    print(f"\n=== Enhanced KG Statistics ===")
    print(f"  Entities: {len(kg.entities)}")
    for etype, count in sorted(kg.entity_type_counts.items()):
        print(f"    {etype}: {count}")
    print(f"  Relations: {len(kg.relations)}")
    for rtype, count in sorted(kg.relation_type_counts.items()):
        print(f"    {rtype}: {count}")

    return kg


if __name__ == "__main__":
    kg = build_enhanced_kg()