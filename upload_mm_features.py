import json, tempfile, shutil, subprocess, sys
from pathlib import Path

feat_dir = Path('simulation/data/kg/multimodal/features')
tmp = Path(tempfile.mkdtemp())
for f in feat_dir.iterdir():
    shutil.copy(f, tmp / f.name)

meta = {
    'id': 'abhijitkumarsingh007/multimodal-features',
    'title': 'multimodal-features',
    'licenses': [{'name': 'MIT'}],
}
with open(tmp / 'dataset-metadata.json', 'w') as f:
    json.dump(meta, f)

print('Files:', [p.name for p in sorted(tmp.iterdir())], flush=True)
print('Uploading...', flush=True)
result = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(tmp), '--dir-mode', 'zip'], capture_output=True, text=True)
if result.returncode != 0:
    result = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(tmp), '--dir-mode', 'zip', '-m', 'Multimodal features ECG text structured CXR 256-dim'], capture_output=True, text=True)
    print('version stdout:', result.stdout, flush=True)
    print('version stderr:', result.stderr, flush=True)
else:
    print('create stdout:', result.stdout, flush=True)
shutil.rmtree(tmp, ignore_errors=True)
print('Return code:', result.returncode, flush=True)
sys.exit(result.returncode)
