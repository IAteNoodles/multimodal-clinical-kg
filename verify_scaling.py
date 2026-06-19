import torch
import torch.nn as nn
import math

torch.manual_seed(42)


def xavier_uniform_bound(weight):
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    return math.sqrt(6.0 / (fan_in + fan_out))


def row_norms(weight):
    return weight.norm(dim=1)


print("=" * 70)
print("CASCADE Initialization Scaling Verification")
print("=" * 70)

# 1. entity_embeddings: nn.Embedding(421216, 512)
ent = nn.Embedding(421216, 512)
nn.init.xavier_uniform_(ent.weight)
ent_norms = row_norms(ent.weight)
ent_re = ent.weight[:, :256].norm(dim=1)
ent_im = ent.weight[:, 256:].norm(dim=1)

print("\n[entity_embeddings] nn.Embedding(421216, 512)")
print(f"  shape: {tuple(ent.weight.shape)}")
print(f"  xavier_uniform_ bound: {xavier_uniform_bound(ent.weight):.6f}")
print(f"  row norm  mean: {ent_norms.mean().item():.6f}  std: {ent_norms.std().item():.6f}")
print(f"  re (first 256)  norm mean: {ent_re.mean().item():.6f}  std: {ent_re.std().item():.6f}")
print(f"  im (last 256)   norm mean: {ent_im.mean().item():.6f}  std: {ent_im.std().item():.6f}")

# 2. modality_embeddings: nn.Embedding(5, 256)
mod = nn.Embedding(5, 256)
nn.init.xavier_uniform_(mod.weight)
mod_norms = row_norms(mod.weight)

print("\n[modality_embeddings] nn.Embedding(5, 256)")
print(f"  shape: {tuple(mod.weight.shape)}")
print(f"  xavier_uniform_ bound: {xavier_uniform_bound(mod.weight):.6f}")
print(f"  row norm  mean: {mod_norms.mean().item():.6f}  std: {mod_norms.std().item():.6f}")

# 3. entity_type_embeddings: nn.Embedding(12, 256)
etyp = nn.Embedding(12, 256)
nn.init.xavier_uniform_(etyp.weight)
etyp_norms = row_norms(etyp.weight)

print("\n[entity_type_embeddings] nn.Embedding(12, 256)")
print(f"  shape: {tuple(etyp.weight.shape)}")
print(f"  xavier_uniform_ bound: {xavier_uniform_bound(etyp.weight):.6f}")
print(f"  row norm  mean: {etyp_norms.mean().item():.6f}  std: {etyp_norms.std().item():.6f}")

# 4. Ratios
base_re = ent_re.mean().item()
print("\n" + "=" * 70)
print("Scaling Ratios (auxiliary_norm / base_re_norm)")
print("=" * 70)
print(f"  base_re_norm (entity re part):       {base_re:.6f}")
print(f"  modality / base_re:    {mod_norms.mean().item()/base_re:.2f}x")
print(f"  entity_type / base_re: {etyp_norms.mean().item()/base_re:.2f}x")
print(f"  entity full / base_re: {ent_norms.mean().item()/base_re:.2f}x")
print("=" * 70)