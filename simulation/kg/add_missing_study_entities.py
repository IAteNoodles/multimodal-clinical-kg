"""Add missing Study entities from text_feature_ids.csv to the enhanced KG."""
from pathlib import Path
import sys
import pandas as pd
from extract_entities import ClinicalKG, Entity
from build_enhanced_kg import save_efficient_extended

KG_PICKLE = Path(r"C:\Users\Noodl\Projects\Research\MultiModal\simulation\data\kg\data_new\clinical_kg_enhanced.pkl")
CSV_PATH = Path(r"C:\Users\Noodl\Projects\Research\MultiModal\simulation\data\kg\multimodal\features\text_feature_ids.csv")
EFFICIENT_DIR = Path(r"C:\Users\Noodl\Projects\Research\MultiModal\simulation\data\kg\data_new")
EFFICIENT_COPY_DIR = Path(r"C:\Users\Noodl\Projects\Research\MultiModal\simulation\data\kg\clinical_kg_efficient")

kg = ClinicalKG.load(KG_PICKLE)
initial_count = len(kg.entities)

text_ids_df = pd.read_csv(CSV_PATH)
unique_text_ids = set(text_ids_df.entity_id.unique())
existing_ids = set(kg.entities.keys())
missing_ids = unique_text_ids - existing_ids

added = 0
for eid in sorted(missing_ids):
    study_id = eid.replace("STY_", "")
    entity = Entity(id=eid, label=f"Study {study_id}", type="Study", modality="RAD")
    kg.add_entity(entity)
    added += 1

kg.save(KG_PICKLE)
save_efficient_extended(kg, EFFICIENT_DIR)
save_efficient_extended(kg, EFFICIENT_COPY_DIR)

print(f"Missing text feature IDs: {len(missing_ids)}")
print(f"Entities added: {added}")
print(f"New total entity count: {len(kg.entities)}")
