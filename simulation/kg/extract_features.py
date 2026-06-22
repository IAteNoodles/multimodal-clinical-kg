from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset

from simulation.kg.models import ModalityEncoder, FeatureUnifier


def _resolve_data_dir() -> Path:
    candidates = [
        Path("simulation/data/kg/multimodal"),
        Path(__file__).parent.parent / "data" / "kg" / "multimodal",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_output_dir(data_dir: Path) -> Path:
    out = data_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_features(features_np: np.ndarray, ids: list, output_dir: Path, stem: str) -> None:
    feat_path = output_dir / f"{stem}_features.npy"
    id_path = output_dir / f"{stem}_feature_ids.csv"
    np.save(feat_path, features_np)
    with open(id_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id"])
        for eid in ids:
            writer.writerow([eid])
    print(f"  Saved {stem}: {features_np.shape[0]} entities, dim={features_np.shape[1]}")


def load_modality_index(data_dir: Path) -> dict:
    index_path = data_dir / "modality_index.json"
    if not index_path.exists():
        print("  [WARN] modality_index.json not found, building minimal index")
        return {}
    with open(index_path) as f:
        return json.load(f)


def load_text_reports(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "text_reports.csv.gz"
    if not path.exists():
        path = data_dir / "text_reports.csv"
    if not path.exists():
        print("  [SKIP] No text_reports file found")
        return pd.DataFrame()
    if path.suffix == '.gz':
        return pd.read_csv(path, compression='gzip')
    return pd.read_csv(path)


def load_cxr_images(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "cxr_images.csv"
    if not path.exists():
        print("  [SKIP] No cxr_images.csv found")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_ecg_records(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "ecg_records.csv.gz"
    if not path.exists():
        path = data_dir / "ecg_records.csv"
    if not path.exists():
        print("  [SKIP] No ecg_records file found")
        return pd.DataFrame()
    if path.suffix == '.gz':
        return pd.read_csv(path, compression='gzip')
    return pd.read_csv(path)


def load_structured_features(data_dir: Path):
    feat_path = data_dir / "structured_features.npz"
    id_path = data_dir / "structured_patient_ids.csv"
    if not feat_path.exists() or not id_path.exists():
        print("  [SKIP] No structured features found")
        return None, None
    data = np.load(feat_path)
    features = data['features'] if 'features' in data else data[data.files[0]]
    ids_df = pd.read_csv(id_path)
    ids = ids_df['patient_id'].tolist() if 'patient_id' in ids_df.columns else ids_df.iloc[:, 0].tolist()
    return ids, features


def tokenize_texts_generator(texts: list[str], batch_size: int = 256, max_length: int = 256):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        yield encoded['input_ids'], encoded['attention_mask'], i


def load_images(image_paths: list[str], target_size: int = 224):
    try:
        from torchvision import transforms
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        tensors = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('L')
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.zeros(1, target_size, target_size))
        return torch.stack(tensors)
    except Exception as e:
        print(f"  [WARN] Could not load images: {e}")
        n = len(image_paths)
        return torch.zeros(n, 1, target_size, target_size)


def load_waveforms(waveform_paths: list[str], target_length: int = 5000):
    try:
        import wfdb
        tensors = []
        for p in waveform_paths:
            try:
                p_str = str(p)
                record_path = p_str.replace('.dat', '').replace('.hea', '')
                record = wfdb.rdrecord(record_path)
                sig = record.p_signal
                if sig.shape[0] < target_length:
                    pad = np.zeros((target_length - sig.shape[0], sig.shape[1]))
                    sig = np.vstack([sig, pad])
                else:
                    sig = sig[:target_length]
                if sig.shape[1] < 12:
                    pad = np.zeros((sig.shape[0], 12 - sig.shape[1]))
                    sig = np.hstack([sig, pad])
                elif sig.shape[1] > 12:
                    sig = sig[:, :12]
                tensors.append(torch.from_numpy(sig.T).float())
            except Exception:
                tensors.append(torch.zeros(12, target_length))
        return torch.stack(tensors)
    except ImportError:
        print("  [WARN] wfdb not installed, using random waveforms as placeholder")
        n = len(waveform_paths)
        return torch.randn(n, 12, target_length) * 0.1


def precompute_text_features(encoder: ModalityEncoder, unifier: FeatureUnifier,
                              entity_ids: list, texts: list[str],
                              output_dir: Path, batch_size: int = 32,
                              device: torch.device = torch.device("cpu"),
                              max_length: int = 256):
    print(f"[Text] Computing text features for {len(texts)} entities on {device}...")
    encoder = encoder.to(device)
    unifier = unifier.to(device)
    encoder.eval()
    unifier.eval()

    all_features = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    batch_count = 0

    for input_ids, attention_mask, offset in tokenize_texts_generator(texts, batch_size=batch_size, max_length=max_length):
        with torch.no_grad():
            ids_batch = input_ids.to(device)
            mask_batch = attention_mask.to(device)
            feats = encoder.encode_text(ids_batch, mask_batch)
            unified = unifier.unify_single("text", feats)
            all_features.append(unified.cpu().numpy())
            del feats, unified, ids_batch, mask_batch, input_ids, attention_mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"  batch {batch_count}/{total_batches} ({batch_count*batch_size}/{len(texts)})")

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, entity_ids, output_dir, "text")
    del encoder, unifier, all_features, features_np
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def precompute_cxr_features(encoder: ModalityEncoder, unifier: FeatureUnifier,
                             entity_ids: list, image_paths: list[str],
                             output_dir: Path, batch_size: int = 32,
                             device: torch.device = torch.device("cpu")):
    print(f"[CXR] Computing CXR features for {len(image_paths)} images on {device}...")
    encoder = encoder.to(device)
    unifier = unifier.to(device)
    encoder.eval()
    unifier.eval()

    images = load_images(image_paths)
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            imgs = batch[0].to(device)
            feats = encoder.encode_cxr(imgs)
            unified = unifier.unify_single("cxr", feats)
            all_features.append(unified.cpu().numpy())
            del feats, unified, imgs
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, entity_ids, output_dir, "cxr")
    del encoder, unifier, all_features, features_np
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def precompute_ecg_features(encoder: ModalityEncoder, unifier: FeatureUnifier,
                             entity_ids: list, waveform_paths: list[str],
                             output_dir: Path, batch_size: int = 64,
                             device: torch.device = torch.device("cpu")):
    print(f"[ECG] Computing ECG features for {len(waveform_paths)} waveforms on {device}...")
    encoder = encoder.to(device)
    unifier = unifier.to(device)
    encoder.eval()
    unifier.eval()

    waveforms = load_waveforms(waveform_paths)
    dataset = TensorDataset(waveforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            waves = batch[0].to(device)
            feats = encoder.encode_ecg(waves)
            unified = unifier.unify_single("ecg", feats)
            all_features.append(unified.cpu().numpy())
            del feats, unified, waves
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, entity_ids, output_dir, "ecg")
    del encoder, unifier, all_features, features_np
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def precompute_structured_features(encoder: ModalityEncoder, unifier: FeatureUnifier,
                                    entity_ids: list, features_np: np.ndarray,
                                    output_dir: Path, batch_size: int = 128,
                                    device: torch.device = torch.device("cpu")):
    print(f"[Structured] Computing structured features for {len(entity_ids)} entities on {device}...")
    encoder = encoder.to(device)
    unifier = unifier.to(device)
    encoder.eval()
    unifier.eval()

    features = torch.from_numpy(features_np).float()
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch in loader:
            feats_in = batch[0].to(device)
            feats = encoder.encode_structured(feats_in)
            unified = unifier.unify_single("structured", feats)
            all_features.append(unified.cpu().numpy())
            del feats, unified, feats_in

    features_out = np.concatenate(all_features, axis=0)
    _save_features(features_out, entity_ids, output_dir, "structured")
    del encoder, unifier, all_features, features_out
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def build_entity_feature_map(output_dir: Path) -> dict[int, np.ndarray]:
    feature_map = {}
    for feat_file in output_dir.glob("*_features.npy"):
        stem = feat_file.stem.replace("_features", "")
        id_file = output_dir / f"{stem}_feature_ids.csv"
        if not id_file.exists():
            continue
        features = np.load(feat_file)
        with open(id_file) as f:
            reader = csv.DictReader(f)
            ids = []
            for row in reader:
                try:
                    ids.append(row['entity_id'])
                except (ValueError, TypeError):
                    continue
        for i, eid in enumerate(ids):
            if eid not in feature_map:
                feature_map[eid] = {}
            feature_map[eid][stem] = features[i]
    return feature_map


def save_entity_feature_map(feature_map: dict, output_dir: Path):
    MOD_KEYS = ["cxr", "ecg", "text", "structured"]
    MOD_DIMS = {"cxr": 256, "ecg": 256, "text": 256, "structured": 256}
    save_dict = {}
    for eid, mod_feats in feature_map.items():
        parts = []
        for k in MOD_KEYS:
            if k in mod_feats:
                parts.append(mod_feats[k].astype(np.float16))
            else:
                parts.append(np.zeros(MOD_DIMS[k], dtype=np.float16))
        combined = np.concatenate(parts, axis=0)
        save_dict[str(eid)] = combined
    all_ids = sorted(save_dict.keys())
    all_features = np.stack([save_dict[k] for k in all_ids], axis=0)
    np.save(output_dir / "entity_features.npy", all_features)
    with open(output_dir / "entity_feature_ids.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["entity_id"])
        for eid in all_ids:
            writer.writerow([eid])
    print(f"  Saved combined entity features: {all_features.shape}")


def main():
    parser = argparse.ArgumentParser(description="Precompute multimodal features for KG entities")
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--modalities', type=str, default='all',
                        help='Comma-separated: text,cxr,ecg,structured')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Max token length for text encoding')
    parser.add_argument('--use-pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use pretrained encoder weights (default: True)')
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else _resolve_data_dir()
    output_dir = Path(args.output_dir) if args.output_dir else _resolve_output_dir(data_dir)

    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Feature dim: {args.feature_dim}")

    encoder = ModalityEncoder(
        unified_dim=args.feature_dim,
        use_pretrained=args.use_pretrained,
        precompute=False,
        device=device,
    )
    encoder.eval()

    unifier = FeatureUnifier(unified_dim=args.feature_dim)
    unifier.eval()

    all_modalities = ['text', 'cxr', 'ecg', 'structured']
    if args.modalities == 'all':
        modalities_to_run = all_modalities
    else:
        modalities_to_run = [m.strip() for m in args.modalities.split(',')]

    for modality in modalities_to_run:
        print(f"\n{'='*50}")
        print(f"Processing: {modality}")
        print('='*50)

        if modality == 'text':
            df = load_text_reports(data_dir)
            if not df.empty and 'text' in df.columns:
                if 'study_id' in df.columns:
                    df = df[df['study_id'].notna()].copy()
                    df['study_id'] = df['study_id'].astype(int)
                entity_ids = df['study_id'].tolist() if 'study_id' in df.columns else list(range(len(df)))
                texts = df['text'].fillna('').tolist()
                precompute_text_features(encoder, unifier, entity_ids, texts, output_dir, args.batch_size, device, max_length=args.max_length)
            else:
                print("  Skipping text: no data")

        elif modality == 'cxr':
            df = load_cxr_images(data_dir)
            if not df.empty and 'image_path' in df.columns:
                if 'study_id' in df.columns:
                    df = df[df['study_id'].notna()].copy()
                    df['study_id'] = df['study_id'].astype(int)
                entity_ids = df['study_id'].tolist() if 'study_id' in df.columns else list(range(len(df)))
                paths = df['image_path'].tolist()
                precompute_cxr_features(encoder, unifier, entity_ids, paths, output_dir, max(args.batch_size // 2, 8), device)
            else:
                print("  Skipping CXR: no data")

        elif modality == 'ecg':
            df = load_ecg_records(data_dir)
            if not df.empty:
                entity_ids = [f"STY_PTBECG{int(eid)}" for eid in df['ecg_id'].tolist()] if 'ecg_id' in df.columns else list(range(len(df)))
                if 'waveform_path' in df.columns:
                    paths = df['waveform_path'].tolist()
                else:
                    print("  [WARN] No waveform_path column, constructing from file naming convention")
                    base = data_dir.parent / "ptb_xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    paths = [str(base / f"HR{str(eid).zfill(5)}.dat") for eid in entity_ids]
                precompute_ecg_features(encoder, unifier, entity_ids, paths, output_dir, args.batch_size, device)
            else:
                print("  Skipping ECG: no data")

        elif modality == 'structured':
            ids, features = load_structured_features(data_dir)
            if ids is not None and features is not None:
                precompute_structured_features(encoder, unifier, ids, features, output_dir, args.batch_size * 2, device)
            else:
                print("  Skipping structured: no data")

    print(f"\n{'='*50}")
    print("Building combined entity feature map...")
    feature_map = build_entity_feature_map(output_dir)
    if feature_map:
        save_entity_feature_map(feature_map, output_dir)
        print(f"  Total entities with features: {len(feature_map)}")
    else:
        print("  No features computed!")

    print(f"\nFeature files:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
