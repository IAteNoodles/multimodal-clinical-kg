import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection

npz = np.load('simulation/data/kg/features/remote_cxr_features.npz')
study_ids = npz['study_ids']
features = npz['features'].astype(np.float32)
print(f'Features: {features.shape}')

chexpert = pd.read_csv('simulation/data/mimic_cxr_jpg/mimic-cxr-2.0.0-chexpert.csv')

with open('simulation/data/kg/clinical_kg_efficient/metadata.json') as f:
    md = json.load(f)
entity_labels = md['entity_labels']
entity_id_map = md['entity_id_map']
id2entity = {v: k for k, v in entity_id_map.items()}

# Map chexpert label -> entity name
chex2ename = {
    'Atelectasis': 'FND_CXR_Atelectasis',
    'Cardiomegaly': 'FND_CXR_Cardiomegaly',
    'Consolidation': 'FND_CXR_Consolidation',
    'Edema': 'FND_CXR_Edema',
    'Enlarged Cardiomediastinum': 'FND_CXR_Enlarged_Cardiomegaly',
    'Fracture': 'FND_CXR_Fracture',
    'Lung Lesion': 'FND_CXR_Lung_Lesion',
    'Lung Opacity': 'FND_CXR_Lung_Opacity',
    'No Finding': 'FND_CXR_No_Finding',
    'Pleural Effusion': 'FND_CXR_Pleural_Effusion',
    'Pleural Other': 'FND_CXR_Pleural_Other',
    'Pneumonia': 'FND_CXR_Pneumonia',
    'Pneumothorax': 'FND_CXR_Pneumothorax',
    'Support Devices': 'FND_CXR_Support_Devices',
}

study_ids_set = set(study_ids)
chex = chexpert[chexpert['study_id'].isin(study_ids_set)].copy()
print(f'Matched studies: {len(chex)}')

entity_features = {}
entity_matched = {}
for chex_label, ename in chex2ename.items():
    if ename not in entity_id_map:
        continue
    pos = chex[chex[chex_label] == 1]
    if len(pos) == 0:
        continue
    pos_study_ids = set(pos['study_id'])
    pos_mask = np.isin(study_ids, list(pos_study_ids))
    feat = features[pos_mask].mean(axis=0)
    entity_features[ename] = feat
    entity_matched[ename] = len(pos_study_ids)

print(f'Entities with features: {len(entity_features)}')
for ename, cnt in sorted(entity_matched.items(), key=lambda x: -x[1]):
    print(f'  {ename}: {cnt} studies')

feat_matrix = np.stack(list(entity_features.values()))
rp = GaussianRandomProjection(n_components=256, random_state=42)
feat_256 = rp.fit_transform(feat_matrix)
print(f'Projected: {feat_256.shape}')

out_dir = Path('simulation/data/kg/cxr_features')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'cxr_features.npy', feat_256.astype(np.float32))
with open(out_dir / 'cxr_feature_ids.csv', 'w') as f:
    f.write('entity_name\n')
    for ename in entity_features.keys():
        f.write(f'{ename}\n')
print(f'Saved {len(entity_features)} features to {out_dir}/')
for p in sorted(out_dir.iterdir()):
    print(f'  {p.name} ({p.stat().st_size} bytes)')
