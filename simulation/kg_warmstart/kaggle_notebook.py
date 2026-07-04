# %% [markdown]
# Multimodal Clinical KG - Warm-Start Cascade Training
# Run this on Kaggle (GPU T4 x2) or locally
#
# Dataset: abhijitkumarsingh007/multimodal-clinical-kg-data
# Code: https://github.com/IAteNoodles/multimodal-clinical-kg/tree/feat/warmstart-cascade

# %% [markdown]
# ## Setup

# %%
import os, sys, json, math, gc, shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

KAGGLE = "KAGGLE_URL_BASE" in os.environ

if KAGGLE:
    DATA_DIR = Path("/kaggle/input/multimodal-clinical-kg-data")
    WORK_DIR = Path("/kaggle/working")
    # Clone code
    if not (WORK_DIR / "multimodal-clinical-kg").exists():
        os.system("git clone -b feat/warmstart-cascade --depth 1 https://github.com/IAteNoodles/multimodal-clinical-kg.git " + str(WORK_DIR / "multimodal-clinical-kg"))
    CODE_DIR = WORK_DIR / "multimodal-clinical-kg"
    os.chdir(str(CODE_DIR))
    sys.path.insert(0, str(CODE_DIR))
else:
    DATA_DIR = Path("simulation/data/kg")
    WORK_DIR = Path(".")
    CODE_DIR = Path(".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# %% [markdown]
# ## Install Dependencies

# %%
if KAGGLE:
    os.system("pip install -q torch_optimizer")

# %% [markdown]
# ## Imports

# %%
from simulation.kg.dataset import KGTriplesDataset, MultimodalKGTriplesDataset, LinkPredictionEvaluator
from simulation.kg_warmstart.train_manual import build_model, save_checkpoint, load_checkpoint, load_best_weights
from simulation.kg_warmstart.models import MultimodalCASCADEModel

# %% [markdown]
# ## Training Config

# %%
SEEDS = [42, 123, 456]
BATCH_SIZE = 4096
LR = 1e-3
EMBED_DIM = 256
EPOCHS = 51
CKPT_DIR = WORK_DIR / "ckpts"

def train_seed(seed, resume_from=None):
    """Train one seed with warm-start from ComplEx, resume if checkpoint exists."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_ckpt = CKPT_DIR / f"cascade_warmstart_seed{seed}"
    seed_ckpt.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    dataset = KGTriplesDataset.from_efficient(DATA_DIR / "clinical_kg_efficient", seed=seed)
    print(f"entities={dataset.num_entities} relations={dataset.num_relations}")

    # Build model
    model = build_model({
        "model": "multimodal_cascade",
        "embed_dim": EMBED_DIM,
        "num_entity_types": dataset.num_entity_types,
        "num_modalities": dataset.num_modalities,
        "dropout": 0.1,
        "ablation": None,
        "cross_modal_relations": None,
    }, dataset).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    # Warm-start entity embeddings from pretrained ComplEx
    warmstart_path = DATA_DIR / ".." / ".." / "ckpts" / "complex_seed42" / "latest.pt"
    if KAGGLE:
        warmstart_path = WORK_DIR / "multimodal-clinical-kg" / "ckpts" / "complex_seed42" / "latest.pt"
    if warmstart_path:
        sd = torch.load(warmstart_path, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        elif "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        if "entity_embeddings.weight" in sd:
            ew = sd["entity_embeddings.weight"]
            model.entity_embeddings.weight.data.copy_(ew)
            model.entity_embeddings.weight.requires_grad_(False)
            print(f"  warm-started entity_embeddings from {warmstart_path} (frozen)")

    # Optimizer (only trainable params)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": LR},
    ], lr=LR, weight_decay=0)

    # Data
    train_triples, _ = dataset.get_train_triples()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_triples.clone()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )

    neg_sampler = NegativeSampler(dataset, num_negatives=1, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    if model.needs_modality_ids():
        entity_type_ids = dataset.get_entity_type_ids().to(device)
        entity_modality_ids = dataset.get_entity_modality_ids().to(device)
    else:
        entity_type_ids = entity_modality_ids = None

    # Scheduler
    n_steps_total = EPOCHS * len(train_loader)
    warmup_steps = 1000

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, n_steps_total - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Resume
    start_ep, global_step = 0, 0
    latest_path = seed_ckpt / "latest.pt"
    if latest_path.exists():
        start_ep, global_step, _ = load_checkpoint(latest_path, model, opt, sched, scaler, device)
        print(f"  resumed from ep{start_ep}")

    # Training loop
    for ep in range(start_ep + 1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bi, (batch,) in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            neg_heads, neg_rels, neg_tails = neg_sampler.sample(batch)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2], entity_type_ids, entity_modality_ids)
                neg_score = model(neg_heads, neg_rels, neg_tails, entity_type_ids, entity_modality_ids)
                loss = loss_fn(torch.stack([pos_score, neg_score], dim=-1), torch.zeros(pos_score.size(0), dtype=torch.long, device=device))

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            train_loss += loss.item()
            if bi == 0:
                print(f"  ep{ep} bi0 loss={loss.item():.4f}")

        train_loss /= len(train_loader)

        gate_val = model.cascade_gate.item() if hasattr(model, "cascade_gate") else None
        gate_str = f" gate={gate_val:.4f}" if gate_val is not None else ""
        print(f"  ep{ep}/{EPOCHS} loss={train_loss:.4f}{gate_str}", flush=True)

        # Save checkpoint
        save_checkpoint(seed_ckpt / "latest.pt", model, opt, sched, scaler, global_step, ep)
        if ep % 10 == 0:
            save_checkpoint(seed_ckpt / f"ep_{ep}.pt", model, opt, sched, scaler, global_step, ep)

    # Final validation
    model.eval()
    evaluator = LinkPredictionEvaluator(dataset, device=device)
    val_t, val_w = dataset.get_val_triples()
    val_metrics = evaluator.evaluate(
        model, val_t.to(device), val_w.to(device),
        batch_size=256, max_triples=500,
        num_eval_negatives=500, full_rank=False,
        entity_type_ids=entity_type_ids,
        entity_modality_ids=entity_modality_ids,
    )
    print(f"  val MRR={val_metrics.get('MRR',0):.4f}")

    # Test evaluation
    test_metrics = run_test_eval(model, evaluator, dataset, device,
                                 entity_type_ids=entity_type_ids,
                                 entity_modality_ids=entity_modality_ids)
    print(f"  test MRR={test_metrics.get('MRR',0):.4f}")

    return test_metrics

def run_test_eval(model, evaluator, dataset, device, entity_type_ids=None, entity_modality_ids=None):
    model.eval()
    test_t, test_w = dataset.get_test_triples()
    test_t, test_w = test_t.to(device), test_w.to(device)
    metrics = evaluator.evaluate(
        model, test_t, test_w,
        batch_size=256, max_triples=5000,
        num_eval_negatives=500, full_rank=False,
        entity_type_ids=entity_type_ids,
        entity_modality_ids=entity_modality_ids,
    )
    return metrics

class NegativeSampler:
    def __init__(self, dataset, num_negatives=1, device="cpu"):
        self.num_entities = dataset.num_entities
        self.num_negatives = num_negatives
        self.device = device

    def sample(self, triples):
        neg_heads = torch.randint(0, self.num_entities, triples[:, 0].shape, device=self.device)
        neg_rels = triples[:, 1].clone()
        neg_tails = torch.randint(0, self.num_entities, triples[:, 2].shape, device=self.device)
        return neg_heads, neg_rels, neg_tails

# %% [markdown]
# ## Run Training

# %%
results = {}
for seed in SEEDS:
    try:
        metrics = train_seed(seed)
        results[str(seed)] = metrics
    except Exception as e:
        print(f"Seed {seed} failed: {e}")
        import traceback
        traceback.print_exc()

print("\nFinal Results:", json.dumps(results, indent=2))

if KAGGLE:
    with open(WORK_DIR / "results.json", "w") as f:
        json.dump(results, f)
    print("Results saved. Copy /kaggle/working/results.json before session ends.")
