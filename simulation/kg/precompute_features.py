from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset

from simulation.kg.models import ModalityEncoder


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
    print(f"  Saved {stem}: {features_np.shape[0]} entities, dim={features_np.shape[1]} -> {feat_path}")


def precompute_text_features(
    encoder: ModalityEncoder,
    entity_ids: list,
    input_ids: LongTensor,
    attention_mask: Optional[LongTensor],
    output_dir: Path,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> None:
    print("[Text] Computing text features...")
    encoder = encoder.to(device)
    encoder.eval()

    all_features = []
    all_ids = entity_ids

    dataset = TensorDataset(input_ids, attention_mask) if attention_mask is not None else TensorDataset(input_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            ids_batch = batch[0].to(device)
            mask_batch = batch[1].to(device) if len(batch) > 1 else None
            feats = encoder.encode_text(ids_batch, mask_batch)
            all_features.append(feats.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, all_ids, output_dir, "text")


def precompute_cxr_features(
    encoder: ModalityEncoder,
    entity_ids: list,
    images: FloatTensor,
    output_dir: Path,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
) -> None:
    print("[CXR] Computing CXR features...")
    encoder = encoder.to(device)
    encoder.eval()

    all_features = []
    all_ids = entity_ids

    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            feats = encoder.encode_cxr(imgs)
            all_features.append(feats.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, all_ids, output_dir, "cxr")


def precompute_ecg_features(
    encoder: ModalityEncoder,
    entity_ids: list,
    waveforms: FloatTensor,
    output_dir: Path,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> None:
    print("[ECG] Computing ECG features...")
    encoder = encoder.to(device)
    encoder.eval()

    all_features = []
    all_ids = entity_ids

    dataset = TensorDataset(waveforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            waves = batch[0].to(device)
            feats = encoder.encode_ecg(waves)
            all_features.append(feats.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, all_ids, output_dir, "ecg")


def precompute_structured_features(
    encoder: ModalityEncoder,
    entity_ids: list,
    features: FloatTensor,
    output_dir: Path,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> None:
    print("[Structured] Computing structured features...")
    encoder = encoder.to(device)
    encoder.eval()

    all_features = []
    all_ids = entity_ids

    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            feats_in = batch[0].to(device)
            feats = encoder.encode_structured(feats_in)
            all_features.append(feats.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    _save_features(features_np, all_ids, output_dir, "structured")


def load_modality_data(data_dir: Path, modality: str):
    raw_dir = data_dir / "raw" / modality
    if not raw_dir.exists():
        raw_dir = data_dir / modality
    if not raw_dir.exists():
        print(f"  [SKIP] No data directory for {modality}: checked {raw_dir}")
        return None, None

    id_file = raw_dir / "entity_ids.npy"
    data_file = raw_dir / "data.npy"

    if id_file.exists() and data_file.exists():
        ids = np.load(id_file).tolist()
        data = torch.from_numpy(np.load(data_file)).float()
        print(f"  Loaded {modality}: {len(ids)} entities, shape={data.shape}")
        return ids, data

    for ext in [".pt", ".pth"]:
        pt_file = raw_dir / f"data{ext}"
        if pt_file.exists() and id_file.exists():
            ids = np.load(id_file).tolist()
            data = torch.load(pt_file, weights_only=True).float()
            print(f"  Loaded {modality}: {len(ids)} entities, shape={data.shape}")
            return ids, data

    pt_file = raw_dir / "data.pt"
    if pt_file.exists():
        payload = torch.load(pt_file, weights_only=True)
        if isinstance(payload, dict):
            data = payload["data"].float()
            if "entity_ids" not in payload:
                print(f"  [WARN] {modality}: no entity_ids in payload, falling back to row indices")
                ids = list(range(data.shape[0]))
            else:
                ids = payload["entity_ids"]
            assert len(ids) == data.shape[0], f"{modality}: len(entity_ids)={len(ids)} != data.shape[0]={data.shape[0]}"
            print(f"  Loaded {modality}: {len(ids)} entities, shape={data.shape}")
            return ids, data

    print(f"  [SKIP] No recognizable data files for {modality} in {raw_dir}")
    return None, None


def load_text_data(data_dir: Path):
    raw_dir = data_dir / "raw" / "text"
    if not raw_dir.exists():
        raw_dir = data_dir / "text"
    if not raw_dir.exists():
        raw_dir = data_dir / "raw" / "rad"
        if not raw_dir.exists():
            raw_dir = data_dir / "rad"

    if not raw_dir.exists():
        print(f"  [SKIP] No text data directory found")
        return None, None, None

    id_file = raw_dir / "entity_ids.npy"
    input_ids_file = raw_dir / "input_ids.npy"
    mask_file = raw_dir / "attention_mask.npy"

    if id_file.exists() and input_ids_file.exists():
        ids = np.load(id_file).tolist()
        input_ids = torch.from_numpy(np.load(input_ids_file)).long()
        attention_mask = torch.from_numpy(np.load(mask_file)).long() if mask_file.exists() else None
        print(f"  Loaded text: {len(ids)} entities, input_ids shape={input_ids.shape}")
        return ids, input_ids, attention_mask

    data_file = raw_dir / "data.npy"
    if id_file.exists() and data_file.exists():
        ids = np.load(id_file).tolist()
        input_ids = torch.from_numpy(np.load(data_file)).long()
        print(f"  Loaded text (single tensor): {len(ids)} entities, shape={input_ids.shape}")
        return ids, input_ids, None

    for ext in [".pt", ".pth"]:
        pt_file = raw_dir / f"data{ext}"
        if pt_file.exists() and id_file.exists():
            ids = np.load(id_file).tolist()
            data = torch.load(pt_file, weights_only=True)
            if isinstance(data, dict):
                input_ids = data.get("input_ids", data.get("data", None))
                attention_mask = data.get("attention_mask", None)
                if input_ids is not None:
                    input_ids = input_ids.long()
                    if attention_mask is not None:
                        attention_mask = attention_mask.long()
                    print(f"  Loaded text (pt): {len(ids)} entities, input_ids shape={input_ids.shape}")
                    return ids, input_ids, attention_mask
            else:
                input_ids = data.long()
                print(f"  Loaded text (pt single): {len(ids)} entities, shape={input_ids.shape}")
                return ids, input_ids, None

    print(f"  [SKIP] No recognizable text data files in {raw_dir}")
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Precompute multimodal features for KG entities")
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to multimodal data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to save precomputed features')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoder inference')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for encoder inference (default: CPU for RTX 4050 6GB)')
    parser.add_argument('--feature-dim', type=int, default=256,
                        help='Unified feature dimension')
    parser.add_argument('--modalities', type=str, default='all',
                        choices=['all', 'text', 'cxr', 'ecg', 'structured'],
                        help='Which modalities to precompute')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='Use pretrained encoder weights')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = _resolve_data_dir()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = _resolve_output_dir(data_dir)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Feature dim: {args.feature_dim}")
    print()

    encoder = ModalityEncoder(
        unified_dim=args.feature_dim,
        use_pretrained=args.use_pretrained,
        precompute=False,
        device=device,
    )
    encoder.eval()

    modalities_to_run = ['text', 'cxr', 'ecg', 'structured'] if args.modalities == 'all' else [args.modalities]

    for modality in modalities_to_run:
        print(f"\n{'='*50}")
        print(f"Processing modality: {modality}")
        print('='*50)

        if modality == 'text':
            ids, input_ids, attention_mask = load_text_data(data_dir)
            if ids is not None and input_ids is not None:
                precompute_text_features(
                    encoder, ids, input_ids, attention_mask,
                    output_dir, batch_size=max(args.batch_size, 32), device=device,
                )
            else:
                print("  Skipping text: no data found")
        elif modality == 'cxr':
            ids, images = load_modality_data(data_dir, "cxr")
            if ids is None:
                ids, images = load_modality_data(data_dir, "image")
            if ids is not None and images is not None:
                precompute_cxr_features(
                    encoder, ids, images,
                    output_dir, batch_size=max(args.batch_size // 2, 8), device=device,
                )
            else:
                print("  Skipping CXR: no data found")
        elif modality == 'ecg':
            ids, waveforms = load_modality_data(data_dir, "ecg")
            if ids is not None and waveforms is not None:
                precompute_ecg_features(
                    encoder, ids, waveforms,
                    output_dir, batch_size=args.batch_size, device=device,
                )
            else:
                print("  Skipping ECG: no data found")
        elif modality == 'structured':
            ids, features = load_modality_data(data_dir, "structured")
            if ids is None:
                ids, features = load_modality_data(data_dir, "mimic")
            if ids is not None and features is not None:
                precompute_structured_features(
                    encoder, ids, features,
                    output_dir, batch_size=args.batch_size * 2, device=device,
                )
            else:
                print("  Skipping structured: no data found")

    print(f"\n{'='*50}")
    print("Precompute complete. Feature files:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
