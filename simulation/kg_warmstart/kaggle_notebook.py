# %% [markdown]
# Multimodal Clinical KG - Warm-Start Cascade Training
# Run on Kaggle (GPU T4 x2)
#
# Dataset: abhijitkumarsingh007/multimodal-clinical-kg-data
# Code: https://github.com/IAteNoodles/multimodal-clinical-kg/tree/feat/warmstart-cascade
#
# Strategy:
#   1. Train ComplEx (51 epochs, seed 42) → saves entity_embeddings.pt
#   2. Build cascade model, freeze entity_embeddings from ComplEx, train heads only
#   3. Run 3 seeds of warm-started cascade
#   4. Evaluate on test set

# %% [markdown]
# ## Setup

# %%
import os, sys, json, math, gc, shutil, copy, zipfile, tarfile
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

KAGGLE = "KAGGLE_URL_BASE" in os.environ

if KAGGLE:
    IN_DIR = Path("/kaggle/input/multimodal-clinical-kg-data")
    WORK_DIR = Path("/kaggle/working")
    if not (WORK_DIR / "multimodal-clinical-kg").exists():
        os.system("git clone -b feat/warmstart-cascade --depth 1 https://github.com/IAteNoodles/multimodal-clinical-kg.git " + str(WORK_DIR / "multimodal-clinical-kg"))
    CODE_DIR = WORK_DIR / "multimodal-clinical-kg"
    os.chdir(str(CODE_DIR))
    sys.path.insert(0, str(CODE_DIR))
    # Discover data: try direct subdirs, flat files, or extract archives
    print("  input files:", flush=True)
    for p in sorted(IN_DIR.iterdir()):
        tag = "dir" if p.is_dir() else f"sz={p.stat().st_size}"
        print(f"    {p.name}  ({tag})", flush=True)
    if (IN_DIR / "clinical_kg_efficient").is_dir():
        DATA_DIR = IN_DIR
        print(f"  using direct subdirs: {DATA_DIR}", flush=True)
    elif (IN_DIR / "metadata.json").is_file():
        DATA_DIR = IN_DIR
        print(f"  using flat files at root: {DATA_DIR}", flush=True)
    else:
        print("  extracting archives ...", flush=True)
        for f in sorted(IN_DIR.iterdir()):
            name = f.name.lower()
            if name.endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    z.extractall(WORK_DIR)
                print(f"    extracted {f.name}", flush=True)
            elif name.endswith(".tar") or name.endswith(".tar.gz"):
                with tarfile.open(f) as t:
                    t.extractall(WORK_DIR)
                print(f"    extracted {f.name}", flush=True)
        DATA_DIR = WORK_DIR
        print(f"  extracted to: {DATA_DIR}", flush=True)
else:
    DATA_DIR = Path("simulation/data/kg")
    WORK_DIR = Path(".")
    CODE_DIR = Path(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)

# %% [markdown]
# ## Install Dependencies

# %%
if KAGGLE:
    os.system("pip install -q torch_optimizer")

# %% [markdown]
# ## Imports

# %%
from simulation.kg.dataset import (
    KGTriplesDataset, NegativeSampler,
    LinkPredictionEvaluator,
)

# %% [markdown]
# ## Config

# %%
SEEDS = [42, 123, 456]
BATCH_SIZE = 4096
LR = 1e-3
EMBED_DIM = 256
EPOCHS = 51
CKPT_DIR = WORK_DIR / "ckpts"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Training Functions

# %%
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    def forward(self, pos, neg):
        B = pos.size(0)
        neg = neg.view(B, -1)
        logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1) / self.t
        target = torch.zeros(B, dtype=torch.long, device=pos.device)
        return torch.nn.functional.cross_entropy(logits, target)

def build_complex_model(dataset):
    from simulation.kg_warmstart.models import ComplExModel
    return ComplExModel(dataset.num_entities, dataset.num_relations, EMBED_DIM, dropout=0.0)

def build_cascade_model(dataset):
    from simulation.kg_warmstart.models import MultimodalCASCADEModel
    from simulation.kg.dataset import ENTITY_TYPE_TO_ID, MODALITY_TO_ID, CROSS_MODAL_RELATIONS
    cross_rel_ids = {dataset.relation2id[r] for r in CROSS_MODAL_RELATIONS if r in dataset.relation2id}
    model = MultimodalCASCADEModel(
        dataset.num_entities, dataset.num_relations, EMBED_DIM,
        num_entity_types=len(ENTITY_TYPE_TO_ID),
        num_modalities=len(MODALITY_TO_ID), synergy_dim=64, num_heads=4,
        dropout=0.1, use_modality_encoders=False,
        use_pretrained_encoders=False, modalities={"text", "image", "ecg", "structured"},
        ablation=None, cross_modal_relations=cross_rel_ids,
    )
    model.precompute_features = False
    return model

def get_loaders(dataset, seed, batch_size):
    torch.manual_seed(seed)
    train_triples, _ = dataset.get_train_triples()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_triples.clone()),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    return loader

def train_complex(dataset, seed, epochs=EPOCHS, start_ep=0):
    print(f"\n{'='*60}\nComplEx seed {seed} (resume from ep{start_ep})\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_complex_model(dataset).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    loader = get_loaders(dataset, seed, BATCH_SIZE)
    neg_sampler = NegativeSampler(dataset, num_negatives=1, device=device)
    loss_fn = InfoNCELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    n_steps_total = epochs * len(loader)
    global_step = start_ep * len(loader)
    warmup = 200
    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        p = (s - warmup) / max(1, n_steps_total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    complex_ckpt = CKPT_DIR / "complex_latest.pt"
    if start_ep > 0 and complex_ckpt.exists():
        sd = torch.load(complex_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        if scaler and "scaler" in sd and sd["scaler"]:
            scaler.load_state_dict(sd["scaler"])
        global_step = sd.get("global_step", start_ep * len(loader))
        print(f"  loaded checkpoint ep{start_ep}", flush=True)

    for ep in range(start_ep + 1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for bi, (batch,) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            neg_all, _ = neg_sampler.sample(batch)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                pos = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
                neg = model.score(neg_all[:, 0], neg_all[:, 1], neg_all[:, 2])
                loss = loss_fn(pos, neg)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            global_step += 1
        avg_loss = loss_sum / len(loader)
        print(f"  ep{ep}/{epochs} loss={avg_loss:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict() if scaler else None, "ep": ep,
                     "global_step": global_step}, complex_ckpt)

    # Save entity embeddings for warm-start
    ew_path = CKPT_DIR / "complex_entity_embeddings.pt"
    if not ew_path.exists():
        torch.save(model.entity_embeddings.weight.data.cpu(), ew_path)
        print(f"  saved entity_embeddings", flush=True)

    # Evaluate ComplEx
    model.eval()
    evaluator = LinkPredictionEvaluator(dataset, device=device)
    test_t, test_w = dataset.get_test_triples()
    test_metrics = evaluator.evaluate(
        model, test_t.to(device), test_w.to(device),
        batch_size=256, max_triples=5000,
        num_eval_negatives=500, full_rank=False,
    )
    print(f"  ComplEx test MRR={test_metrics.get('MRR',0):.4f}", flush=True)
    # Save metrics in checkpoint
    sd = torch.load(complex_ckpt, map_location="cpu", weights_only=False)
    sd["metrics"] = test_metrics
    torch.save(sd, complex_ckpt)
    return test_metrics

def train_cascade_warmstart(dataset, seed, epochs=EPOCHS):
    print(f"\n{'='*60}\nCascade warm-start seed {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_cascade_model(dataset).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Warm-start entity_embeddings from pretrained ComplEx
    ew_path = CKPT_DIR / "complex_entity_embeddings.pt"
    if ew_path.exists():
        ew = torch.load(ew_path, map_location="cpu", weights_only=True)
        if ew.shape == model.entity_embeddings.weight.shape:
            model.entity_embeddings.weight.data.copy_(ew)
            model.entity_embeddings.weight.requires_grad_(False)
            print(f"  warm-started entity_embeddings (frozen)", flush=True)
        else:
            print(f"  WARNING: shape mismatch {ew.shape} vs {model.entity_embeddings.weight.shape}", flush=True)
    else:
        print(f"  WARNING: {ew_path} not found, using random init", flush=True)

    loader = get_loaders(dataset, seed, BATCH_SIZE)
    neg_sampler = NegativeSampler(dataset, num_negatives=1, device=device)
    loss_fn = InfoNCELoss().to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable params={sum(p.numel() for p in trainable_params):,}", flush=True)
    opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    n_steps_total = epochs * len(loader)
    warmup = 200
    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        p = (s - warmup) / max(1, n_steps_total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    entity_type_ids = dataset.get_entity_type_ids().to(device)
    entity_modality_ids = dataset.get_entity_modality_ids().to(device)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for bi, (batch,) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            neg_all, _ = neg_sampler.sample(batch)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                pos = model.score(batch[:, 0], batch[:, 1], batch[:, 2], entity_type_ids, entity_modality_ids)
                neg = model.score(neg_all[:, 0], neg_all[:, 1], neg_all[:, 2], entity_type_ids, entity_modality_ids)
                loss = loss_fn(pos, neg)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            loss_sum += loss.item()
        gate_val = model.cascade_gate.item() if hasattr(model, "cascade_gate") else None
        gate_str = f" gate={gate_val:.4f}" if gate_val is not None else ""
        print(f"  ep{ep}/{epochs} loss={loss_sum/len(loader):.4f}{gate_str}", flush=True)

        if ep % 10 == 0:
            torch.save(model.state_dict(), CKPT_DIR / f"cascade_ws_seed{seed}_ep{ep}.pt")

    torch.save(model.state_dict(), CKPT_DIR / f"cascade_ws_seed{seed}_final.pt")
    print(f"  saved checkpoint", flush=True)

    # Evaluate
    model.eval()
    evaluator = LinkPredictionEvaluator(dataset, device=device)
    test_t, test_w = dataset.get_test_triples()
    test_metrics = evaluator.evaluate(
        model, test_t.to(device), test_w.to(device),
        batch_size=256, max_triples=5000,
        num_eval_negatives=500, full_rank=False,
        entity_type_ids=entity_type_ids,
        entity_modality_ids=entity_modality_ids,
    )
    print(f"  cascade test MRR={test_metrics.get('MRR',0):.4f}", flush=True)
    return test_metrics

# %% [markdown]
# ## Step 1: Train ComplEx (seed 42, shared embeddings)

# %%
# Find where metadata.json lives
kg_dir = DATA_DIR / "clinical_kg_efficient"
if not kg_dir.is_dir():
    for cand in [DATA_DIR, DATA_DIR / "kg", DATA_DIR / "data"]:
        if (cand / "metadata.json").is_file():
            kg_dir = cand
            break
    if not (kg_dir / "metadata.json").is_file():
        for sub in DATA_DIR.iterdir():
            if sub.is_dir() and (sub / "metadata.json").is_file():
                kg_dir = sub
                break
    if not (kg_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"metadata.json not found under {DATA_DIR}")
print(f"  kg_dir = {kg_dir}", flush=True)

dataset = KGTriplesDataset.from_efficient(kg_dir, seed=42)
print(f"entities={dataset.num_entities} relations={dataset.num_relations}", flush=True)

# Resume ComplEx if checkpoint exists
complex_ckpt = CKPT_DIR / "complex_latest.pt"
complex_start_ep = 0
if complex_ckpt.exists():
    sd = torch.load(complex_ckpt, map_location="cpu", weights_only=False)
    complex_start_ep = sd.get("ep", 0)
    print(f"  ComplEx checkpoint found, resuming from ep{complex_start_ep}", flush=True)
if complex_start_ep >= 51:
    print("  ComplEx already complete, skipping", flush=True)
    complex_metrics = sd.get("metrics", {})
else:
    complex_metrics = train_complex(dataset, seed=42, start_ep=complex_start_ep)
    print(f"ComplEx results: {json.dumps(complex_metrics)}", flush=True)

# %% [markdown]
# ## Step 2: Train Warm-Start Cascade (3 seeds)

# %%
cascade_results = {}
for seed in SEEDS:
    try:
        metrics = train_cascade_warmstart(dataset, seed)
        cascade_results[str(seed)] = metrics
    except Exception as e:
        print(f"Seed {seed} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

print(f"\nResults:\n  ComplEx: {json.dumps(complex_metrics)}\n  Cascade: {json.dumps(cascade_results)}", flush=True)

if KAGGLE:
    with open(WORK_DIR / "results.json", "w") as f:
        json.dump({"complex": complex_metrics, "cascade": cascade_results}, f)
    print("Saved results.json. Copy before session ends.", flush=True)
