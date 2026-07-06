import json, tempfile, shutil, os, subprocess, sys
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
shutil.copy('simulation/data/kg/cxr_features/cxr_features.npy', tmp / 'cxr_features.npy')
shutil.copy('simulation/data/kg/cxr_features/cxr_feature_ids.csv', tmp / 'cxr_feature_ids.csv')

meta = {
    'id': 'abhijitkumarsingh007/cxr-features',
    'title': 'cxr-features',
    'licenses': [{'name': 'MIT'}],
}
with open(tmp / 'dataset-metadata.json', 'w') as f:
    json.dump(meta, f)
print('Files:', [p.name for p in sorted(tmp.iterdir())], flush=True)
print('Uploading...', flush=True)
result = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(tmp), '--dir-mode', 'zip'], capture_output=True, text=True)
if result.returncode != 0:
    result = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(tmp), '--dir-mode', 'zip', '-m', 'CXR DenseNet features 14 entities 256-dim'], capture_output=True, text=True)
    print('version stdout:', result.stdout, flush=True)
    print('version stderr:', result.stderr, flush=True)
else:
    print('create stdout:', result.stdout, flush=True)
shutil.rmtree(tmp, ignore_errors=True)
print('Return code:', result.returncode, flush=True)
sys.exit(result.returncode)
