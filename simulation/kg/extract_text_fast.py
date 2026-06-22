"""Fast text feature extraction using ClinicalBERT. Accumulates in memory, saves once."""
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


def extract_text_features_fast(texts, entity_ids, output_dir, batch_size=256, max_length=128, device='cuda', feature_dim=256, precision='float32'):
    from transformers import AutoTokenizer, AutoModel

    print(f"[Text] Loading ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = model.to(device).eval()

    proj_weights_path = output_dir / "text_projection.pt"
    proj = nn.Linear(768, feature_dim).to(device)
    if proj_weights_path.exists():
        proj.load_state_dict(torch.load(proj_weights_path, map_location=device))
    else:
        torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(proj.weight)
        torch.nn.init.zeros_(proj.bias)
        torch.save(proj.state_dict(), proj_weights_path)

    feat_file = output_dir / "text_features.npy"
    id_file = output_dir / "text_feature_ids.csv"

    n = len(texts)
    total_batches = (n + batch_size - 1) // batch_size

    all_features = []
    all_ids = []

    print(f"[Text] Processing {n} texts in {total_batches} batches of {batch_size}...")
    start = time.time()

    with torch.no_grad():
        for batch_idx in range(total_batches):
            i0 = batch_idx * batch_size
            i1 = min(i0 + batch_size, n)
            batch_texts = texts[i0:i1]

            encoded = tokenizer(batch_texts, padding='max_length', truncation=True,
                              max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.amp.autocast('cuda', enabled=(torch.device(device).type == 'cuda')):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                mask_exp = attention_mask.unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
                feats = proj(pooled).cpu()
                if precision == 'float16':
                    feats = feats.half()
                else:
                    feats = feats.float()

            all_features.append(feats.numpy())
            all_ids.extend(entity_ids[i0:i1])

            del input_ids, attention_mask, out, pooled, feats
            torch.cuda.empty_cache()

            if batch_idx % 100 == 0:
                elapsed = time.time() - start
                done = i1
                pct = done / n * 100
                rate = done / elapsed
                eta = (n - done) / rate
                print(f"  batch {batch_idx}/{total_batches} ({pct:.0f}%, {rate:.0f} texts/s, ETA {eta:.0f}s)")

    dtype = np.float16 if precision == 'float16' else np.float32
    features_np = np.concatenate(all_features, axis=0).astype(dtype)
    np.save(feat_file, features_np)

    with open(id_file, 'w', newline='') as id_f:
        w = csv.writer(id_f)
        w.writerow(["entity_id"])
        for eid in all_ids:
            w.writerow([eid])

    elapsed = time.time() - start
    print(f"[Text] Done: {n} texts in {elapsed:.1f}s ({n/elapsed:.0f} texts/s)")
    print(f"[Text] Saved: {features_np.shape} -> {feat_file}")

    del model, proj, tokenizer, all_features, features_np
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='simulation/data/kg/multimodal')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16'])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(data_dir / "text_reports.csv.gz", compression='gzip')
    print(f"Loaded {len(df)} text reports")

    entity_ids = df['study_id'].tolist() if 'study_id' in df.columns else list(range(len(df)))
    texts = df['text'].fillna('').str.strip().tolist()
    texts = [t if t else "no finding" for t in texts]

    extract_text_features_fast(
        texts, entity_ids, output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        feature_dim=args.feature_dim,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
