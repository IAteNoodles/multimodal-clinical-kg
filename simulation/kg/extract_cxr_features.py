import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_IMAGE_DIR = PROJECT_ROOT / "simulation" / "data" / "mimic_cxr_jpg" / "images"
META_CSV = PROJECT_ROOT / "simulation" / "data" / "mimic_cxr_jpg" / "mimic-cxr-2.0.0-metadata.csv"
OUT_NPZ = PROJECT_ROOT / "simulation" / "data" / "kg" / "data_new" / "cxr_features.npz"
PROGRESS_FILE = PROJECT_ROOT / "simulation" / "data" / "kg" / "data_new" / "cxr_extract_progress.json"
COOKIES_FILE = PROJECT_ROOT / "simulation" / "data" / "kg" / "data_new" / "cookies.txt"


def physionet_login(username: str, password: str) -> dict[str, str]:
    sess = requests.Session()
    r = sess.get("https://physionet.org/login/")
    csrf = sess.cookies.get("csrftoken", "")
    r = sess.post(
        "https://physionet.org/login/",
        data={"username": username, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": "https://physionet.org/login/"},
        allow_redirects=True,
    )
    if "sessionid" not in sess.cookies:
        raise RuntimeError(f"PhysioNet login failed: status={r.status_code}")
    log.info("PhysioNet login ok")
    return {c.name: c.value for c in sess.cookies}


def write_cookies_txt(cookies: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Netscape HTTP Cookie File"]
    for name in ("sessionid", "csrftoken"):
        if name in cookies:
            lines.append(f".physionet.org\tTRUE\t/\tFALSE\t0\t{name}\t{cookies[name]}")
    path.write_text("\n".join(lines) + "\n")


def get_local_image_path(subject_id: int, study_id: int, dicom_id: str) -> Path | None:
    pfx = f"p{str(subject_id)[:2]}"
    path = LOCAL_IMAGE_DIR / pfx / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    if path.exists():
        return path
    flat = LOCAL_IMAGE_DIR / f"{dicom_id}.jpg"
    return flat if flat.exists() else None


def build_url(subject_id: int, study_id: int, dicom_id: str) -> str:
    pfx = f"p{str(subject_id)[:2]}"
    subj = f"p{subject_id}"
    study = f"s{study_id}"
    fname = f"{dicom_id}.jpg"
    return f"{BASE_URL}{pfx}/{subj}/{study}/{fname}"


def load_metadata(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df = df[df["ViewPosition"].isin(["AP", "PA"])]
    df = df.sort_values("ViewPosition", ascending=False)
    df = df.drop_duplicates(subset=["study_id"], keep="first")
    log.info(f"Filtered to {len(df)} frontal studies")
    return df


def download_batch(urls_with_names: list[tuple[str, str]], cookies: dict[str, str], tmp_dir: Path, username: str, password: str) -> Path:
    url_file = tmp_dir / "urls.txt"
    lines = []
    for url, out_name in urls_with_names:
        lines.append(url)
        lines.append(f"\tout={out_name}")
    url_file.write_text("\n".join(lines) + "\n", newline="\n")

    cookies = _run_aria2c(url_file, cookies, tmp_dir, username, password)
    return cookies


def _run_aria2c(url_file: Path, cookies: dict[str, str], tmp_dir: Path, username: str, password: str, retries: int = 2) -> dict[str, str]:
    for attempt in range(retries + 1):
        write_cookies_txt(cookies, COOKIES_FILE)
        cmd = [
            "aria2c",
            f"--load-cookies={COOKIES_FILE}",
            f"-d={tmp_dir}",
            "-x", "16", "-s", "16", "-j", "8",
            "--allow-overwrite=true",
            "--max-tries=3",
            "--retry-wait=5",
            "--timeout=30",
            "-i", str(url_file),
        ]
        log.info(f"aria2c attempt {attempt + 1}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return cookies
        output = (result.stdout + result.stderr).lower()
        if "403" in output or "forbidden" in output or attempt < retries:
            log.warning(f"aria2c rc={result.returncode}, re-logging in")
            cookies = physionet_login(username, password)
        else:
            log.error(f"aria2c failed rc={result.returncode}: {result.stdout[:500]} {result.stderr[:500]}")
            raise RuntimeError(f"aria2c failed with return code {result.returncode}")
    return cookies


class DenseNetExtractor:
    def __init__(self, device: str = "cuda"):
        torch.cuda.empty_cache()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        all_feats = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = []
            valid_idx = []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_idx.append(j)
                except Exception:
                    log.warning(f"Corrupt image: {p.name}")
            if not tensors:
                batch_feats = np.zeros((len(batch_paths), 1024), dtype=np.float16)
                all_feats.append(batch_feats)
                continue
            batch = torch.stack(tensors).to(self.device)
            out = self.model(batch).cpu().numpy().astype(np.float16)
            batch_feats = np.zeros((len(batch_paths), 1024), dtype=np.float16)
            for k, vi in enumerate(valid_idx):
                batch_feats[vi] = out[k]
            all_feats.append(batch_feats)
        return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 1024), dtype=np.float16)


def load_progress() -> tuple[set[int], int]:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("completed_studies", [])), data.get("batch_index", 0)
    return set(), 0


def save_progress(completed: set[int], batch_idx: int) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({
        "completed_studies": sorted(completed),
        "batch_index": batch_idx,
    }))


def save_output(study_ids: list[int], features: np.ndarray) -> None:
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    if OUT_NPZ.exists():
        bak = OUT_NPZ.with_name(f"{OUT_NPZ.stem}.bak.npz")
        for attempt in range(50):
            try:
                bak.unlink(missing_ok=True)
                OUT_NPZ.rename(bak)
                break
            except OSError:
                time.sleep(0.2)
    np.savez_compressed(OUT_NPZ, study_ids=np.array(study_ids, dtype=np.int64), features=features)
    log.info(f"Saved {len(study_ids)} features to {OUT_NPZ}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of studies to process (0=all)")
    parser.add_argument("--local-only", action="store_true", default=False, help="Only use locally available images; skip PhysioNet login and remote downloads")
    args = parser.parse_args()

    username = os.environ.get("PHYSIONET_USERNAME", "")
    password = os.environ.get("PHYSIONET_PASSWORD", "")

    df = load_metadata(META_CSV)
    rows = list(df.itertuples(index=False))
    if args.limit > 0:
        rows = rows[:args.limit]
        log.info(f"Limited to {args.limit} studies")

    completed_studies, start_batch = load_progress()
    log.info(f"Resume: {len(completed_studies)} done, starting batch {start_batch}")

    extractor = DenseNetExtractor(device=args.device)
    all_study_ids: list[int] = []
    all_features: list[np.ndarray] = []

    existing_ids = set()
    if OUT_NPZ.exists():
        try:
            with np.load(OUT_NPZ) as existing:
                existing["study_ids"]
                existing["features"]
        except Exception:
            bak = OUT_NPZ.with_name(f"{OUT_NPZ.stem}.bak.npz")
            if bak.exists():
                log.warning(f"Main NPZ corrupt, restoring from {bak}")
                bak.rename(OUT_NPZ)
            else:
                log.warning(f"Main NPZ corrupt, no .bak to restore from")
    if OUT_NPZ.exists():
        try:
            with np.load(OUT_NPZ) as existing:
                existing_ids = set(int(s) for s in existing["study_ids"])
                all_study_ids = existing["study_ids"].tolist()
                all_features.append(existing["features"].copy())
            log.info(f"Loaded {len(existing_ids)} existing features from {OUT_NPZ}")
        except Exception as e:
            log.warning(f"Could not load NPZ (starting fresh): {e}")

    completed_studies = completed_studies | existing_ids

    cookies: dict[str, str] = {}
    if not args.local_only:
        if not username or not password:
            parser.error("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD env vars")
        cookies = physionet_login(username, password)

    num_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    log.info(f"Total: {len(rows)} studies in {num_batches} batches")

    for batch_idx in range(num_batches):
        if batch_idx < start_batch:
            continue
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(rows))
        batch_rows = rows[start:end]

        batch_rows = [r for r in batch_rows if r.study_id not in completed_studies]
        if not batch_rows:
            log.info(f"Batch {batch_idx + 1}/{num_batches}: all skipped")
            continue

        local_paths: list[tuple[Path, int]] = []  # (image_path, study_id)
        remote_urls: list[tuple[str, str, int]] = []  # (url, out_name, study_id)

        for r in batch_rows:
            local = get_local_image_path(r.subject_id, r.study_id, r.dicom_id)
            if local is not None:
                local_paths.append((local, r.study_id))
            elif not args.local_only:
                url = build_url(r.subject_id, r.study_id, r.dicom_id)
                out_name = f"{r.study_id}_{r.dicom_id}.jpg"
                remote_urls.append((url, out_name, r.study_id))

        batch_study_ids: list[int] = []
        batch_feats: list[np.ndarray] = []

        if local_paths:
            t0 = time.time()
            paths_only = [p for p, _ in local_paths]
            sids_only = [s for _, s in local_paths]
            feats = extractor.extract(paths_only, batch_size=32)
            valid = 0
            for i, sid in enumerate(sids_only):
                if not np.all(feats[i] == 0):
                    batch_study_ids.append(sid)
                    batch_feats.append(feats[i])
                    valid += 1
            log.info(f"Batch {batch_idx + 1}/{num_batches}: {valid}/{len(local_paths)} local features in {time.time()-t0:.1f}s")

        if remote_urls:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                urls_with_names = [(u, n) for u, n, _ in remote_urls]
                study_id_map = {n: s for _, n, s in remote_urls}

                t0 = time.time()
                cookies = download_batch(urls_with_names, cookies, tmp_path, username, password)
                dl_time = time.time() - t0

                jpg_files = sorted(tmp_path.glob("*.jpg"))
                log.info(f"Batch {batch_idx + 1}/{num_batches}: downloaded {len(jpg_files)} images in {dl_time:.1f}s")

                name_to_idx = {f.name: i for i, f in enumerate(jpg_files)}

                t0 = time.time()
                if jpg_files:
                    features = extractor.extract(jpg_files, batch_size=32)
                    ext_time = time.time() - t0
                    for out_name, sid in study_id_map.items():
                        if out_name in name_to_idx:
                            batch_study_ids.append(sid)
                            batch_feats.append(features[name_to_idx[out_name]])
                    log.info(f"Batch {batch_idx + 1}/{num_batches}: extracted {len(jpg_files)} remote features in {ext_time:.1f}s")

        if batch_study_ids:
            batch_feats_arr = np.stack(batch_feats)
            all_study_ids.extend(batch_study_ids)
            all_features.append(batch_feats_arr)
            completed_studies.update(batch_study_ids)

        log.info(
            f"Batch {batch_idx + 1}/{num_batches}: total={len(all_study_ids)}"
        )

        save_progress(completed_studies, batch_idx + 1)
        if all_features:
            save_output(all_study_ids, np.concatenate(all_features))

    if all_features:
        save_output(all_study_ids, np.concatenate(all_features))
    log.info(f"Done. {len(all_study_ids)} studies processed.")


if __name__ == "__main__":
    main()