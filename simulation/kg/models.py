from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor


class KGModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embed_dim: int = 256):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(
        self, heads: LongTensor, relations: LongTensor, tails: LongTensor
    ) -> FloatTensor:
        raise NotImplementedError

    def forward(self, triples: LongTensor) -> FloatTensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2])


class TransEModel(KGModel):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        p_norm: int = 2,
        margin: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__(num_entities, num_relations, embed_dim)
        self.p_norm = p_norm
        self.margin = margin
        self.dropout = nn.Dropout(p=dropout)

    def score(
        self, heads: LongTensor, relations: LongTensor, tails: LongTensor
    ) -> FloatTensor:
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        if self.training:
            h = self.dropout(h)
            r = self.dropout(r)
            t = self.dropout(t)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def n3_penalty(self, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        return (h.norm(p=3, dim=-1).mean() + r.norm(p=3, dim=-1).mean() + t.norm(p=3, dim=-1).mean())

    def clamp_embed_norm(self, max_norm: float = 1.0):
        with torch.no_grad():
            nn.functional.normalize(self.entity_embeddings.weight, p=2, dim=-1, out=self.entity_embeddings.weight)
            self.entity_embeddings.weight.mul_(max_norm)
            nn.functional.normalize(self.relation_embeddings.weight, p=2, dim=-1, out=self.relation_embeddings.weight)
            self.relation_embeddings.weight.mul_(max_norm)


class ComplExModel(KGModel):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(num_entities, num_relations, embed_dim * 2)
        self.real_embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def score(
        self, heads: LongTensor, relations: LongTensor, tails: LongTensor
    ) -> FloatTensor:
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        if self.training and self.dropout is not None:
            h = self.dropout(h)
            r = self.dropout(r)
            t = self.dropout(t)
        d = self.real_embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        return (
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def n3_penalty(self, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        d = self.real_embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        return (h_re.norm(p=3, dim=-1).mean() + h_im.norm(p=3, dim=-1).mean()
                + r_re.norm(p=3, dim=-1).mean() + r_im.norm(p=3, dim=-1).mean()
                + t_re.norm(p=3, dim=-1).mean() + t_im.norm(p=3, dim=-1).mean())

    def clamp_embed_norm(self, max_norm: float = 1.0):
        with torch.no_grad():
            nn.functional.normalize(self.entity_embeddings.weight, p=2, dim=-1, out=self.entity_embeddings.weight)
            self.entity_embeddings.weight.mul_(max_norm)
            nn.functional.normalize(self.relation_embeddings.weight, p=2, dim=-1, out=self.relation_embeddings.weight)
            self.relation_embeddings.weight.mul_(max_norm)


class ModalityEncoder(nn.Module):
    """Produces fixed-dimension feature vectors from raw modality data.

    CXR: Pretrained DenseNet-121 backbone -> 512-d visual features.
         Pass pretrained=False for random init (testing / no download).
    ECG: 1D-CNN with 4 conv blocks -> 256-d temporal features.
    Text: ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) -> 768-d, projected to 256-d.
          Set use_pretrained=False to use a random 768-d projection stub.
    Structured: 2-layer MLP -> 128-d clinical features.

    All encoders are lightweight enough for RTX 4050 6GB when features
    are precomputed. Set precompute=True to store features in a buffer
    after first forward and skip encoder thereafter.
    """

    MODALITY_DIM = {"cxr": 1024, "ecg": 256, "text": 256, "structured": 128}

    def __init__(
        self,
        unified_dim: int = 256,
        use_pretrained: bool = True,
        precompute: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.unified_dim = unified_dim
        self.precompute = precompute
        self.device_param = nn.Parameter(torch.zeros(1, device=device))
        self._precomputed: dict[str, torch.Tensor] = {}

        # CXR: DenseNet-121 backbone
        self.cxr_encoder = self._build_cxr_encoder(use_pretrained)

        # ECG: 1D-CNN
        self.ecg_encoder = self._build_ecg_encoder()

        # Text: ClinicalBERT + projection
        self.text_encoder, self.text_proj = self._build_text_encoder(use_pretrained)

        # Structured: 2-layer MLP
        self.structured_encoder = self._build_structured_encoder()

    def _build_cxr_encoder(self, use_pretrained: bool) -> nn.Module:
        try:
            import torchvision.models as models
            densenet = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT if use_pretrained else None
            )
            backbone = nn.Sequential(*list(densenet.children())[:-1], nn.AdaptiveAvgPool2d(1))
            return backbone
        except Exception:
            return nn.Sequential(
                nn.Conv2d(1, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 512),
            )

    def _build_ecg_encoder(self) -> nn.Module:
        in_ch = 12
        blocks = []
        for out_ch in [32, 64, 128, 256]:
            blocks.extend([
                nn.Conv1d(in_ch, out_ch, 7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        blocks.append(nn.AdaptiveAvgPool1d(1))
        blocks.append(nn.Flatten())
        return nn.Sequential(*blocks)

    def _build_text_encoder(self, use_pretrained: bool):
        if use_pretrained:
            try:
                from transformers import AutoModel
                bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                bert.gradient_checkpointing_enable()
                proj = nn.Linear(768, 256)
                return bert, proj
            except Exception:
                pass
        stub = nn.Sequential(nn.Linear(768, 768), nn.ReLU(inplace=True))
        proj = nn.Linear(768, 256)
        return stub, proj

    def _build_structured_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def encode_cxr(self, images: FloatTensor) -> FloatTensor:
        if self.precompute and "cxr" in self._precomputed:
            return self._precomputed["cxr"]
        x = images
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3]
        feat = self.cxr_encoder(x).flatten(1)
        if self.precompute:
            self._precomputed["cxr"] = feat.detach()
        return feat

    def encode_ecg(self, waveforms: FloatTensor) -> FloatTensor:
        if self.precompute and "ecg" in self._precomputed:
            return self._precomputed["ecg"]
        x = waveforms
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.ecg_encoder(x).flatten(1)
        if self.precompute:
            self._precomputed["ecg"] = feat.detach()
        return feat

    def encode_text(self, input_ids: LongTensor, attention_mask: LongTensor | None = None) -> FloatTensor:
        if self.precompute and "text" in self._precomputed:
            return self._precomputed["text"]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        feat = self.text_proj(cls)
        if self.precompute:
            self._precomputed["text"] = feat.detach()
        return feat

    def encode_structured(self, features: FloatTensor) -> FloatTensor:
        if self.precompute and "structured" in self._precomputed:
            return self._precomputed["structured"]
        feat = self.structured_encoder(features)
        if self.precompute:
            self._precomputed["structured"] = feat.detach()
        return feat

    def precompute_all(
        self,
        cxr_images: FloatTensor | None = None,
        ecg_waveforms: FloatTensor | None = None,
        text_input_ids: LongTensor | None = None,
        text_attention_mask: LongTensor | None = None,
        structured_features: FloatTensor | None = None,
    ) -> dict[str, FloatTensor]:
        self.precompute = True
        self._precomputed = {}
        results = {}
        if cxr_images is not None:
            results["cxr"] = self.encode_cxr(cxr_images)
        if ecg_waveforms is not None:
            results["ecg"] = self.encode_ecg(ecg_waveforms)
        if text_input_ids is not None:
            results["text"] = self.encode_text(text_input_ids, text_attention_mask)
        if structured_features is not None:
            results["structured"] = self.encode_structured(structured_features)
        return results

    def clear_precomputed(self):
        self._precomputed = {}
        self.precompute = False


class FeatureUnifier(nn.Module):
    """Projects modality-specific features to a common dimension."""

    MODALITY_KEYS = ["cxr", "ecg", "text", "structured"]
    MODALITY_DIM = {"cxr": 1024, "ecg": 256, "text": 256, "structured": 128}

    def __init__(self, unified_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.unified_dim = unified_dim
        self.projections = nn.ModuleDict({
            key: nn.Linear(self.MODALITY_DIM[key], unified_dim)
            for key in self.MODALITY_KEYS
        })
        self.norm = nn.LayerNorm(unified_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, modality_features: dict[str, FloatTensor]) -> dict[str, FloatTensor]:
        unified = {}
        for key, feat in modality_features.items():
            if key in self.projections:
                proj = self.projections[key](feat)
                proj = self.norm(proj)
                if self.training:
                    proj = self.dropout(proj)
                unified[key] = proj
        return unified

    def unify_single(self, key: str, feat: FloatTensor) -> FloatTensor:
        proj = self.norm(self.projections[key](feat))
        if self.training:
            proj = self.dropout(proj)
        return proj


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention between head and tail modality features.

    For cross-modal triples (head and tail have different modalities),
    computes attention over modality feature sets to produce context vectors.
    """

    def __init__(self, unified_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = unified_dim // num_heads
        assert self.head_dim * num_heads == unified_dim

        self.norm_q = nn.LayerNorm(unified_dim)
        self.norm_k = nn.LayerNorm(unified_dim)
        self.q_proj = nn.Linear(unified_dim, unified_dim)
        self.k_proj = nn.Linear(unified_dim, unified_dim)
        self.v_proj = nn.Linear(unified_dim, unified_dim)
        self.out_proj = nn.Linear(unified_dim, unified_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        head_features: FloatTensor,
        tail_features: FloatTensor,
        mask: FloatTensor | None = None,
    ) -> tuple[FloatTensor, FloatTensor]:
        B = head_features.size(0)
        Q = self.q_proj(self.norm_q(head_features)).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.norm_k(tail_features)).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(self.norm_k(tail_features)).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1)
        output = head_features + self.dropout(self.out_proj(ctx))
        return output, attn.mean(dim=1)


class PIDSynergy(nn.Module):
    """PID-inspired synergy that modulates cross-attention between modality pairs.

    Returns a modulation vector (not scalar) that augments the ComplEx score.
    Uses a hypernetwork-style approach: modality pair -> synergy vector.
    """

    def __init__(self, num_modalities: int, num_relations: int = 8, synergy_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_relations = num_relations
        self.synergy_dim = synergy_dim

        self.pair_encoder = nn.Sequential(
            nn.Embedding(num_modalities * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.relation_encoder = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_encoder.weight)
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.synergy_head = nn.Linear(hidden_dim, synergy_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, synergy_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, head_mod_ids: LongTensor, tail_mod_ids: LongTensor, relation_ids: LongTensor) -> FloatTensor:
        pair_ids = head_mod_ids * self.num_modalities + tail_mod_ids
        pair_h = self.pair_encoder(pair_ids)
        rel_h = self.relation_encoder(relation_ids)
        h = self.combiner(torch.cat([pair_h, rel_h], dim=-1))
        h = self.dropout(h)
        synergy = self.synergy_head(h) * self.gate_head(h)
        return synergy

    def get_synergy_matrix(self) -> FloatTensor:
        with torch.no_grad():
            mods = torch.arange(self.num_modalities, device=next(self.parameters()).device)
            rels = torch.arange(self.num_relations, device=next(self.parameters()).device)
            h_mod, t_mod = torch.meshgrid(mods, mods, indexing="ij")
            pair_ids = h_mod * self.num_modalities + t_mod
            pair_h = self.pair_encoder(pair_ids.flatten())
            rel_h = self.relation_encoder(rels).mean(dim=0, keepdim=True)
            h = self.combiner(torch.cat([pair_h, rel_h.expand(self.num_modalities * self.num_modalities, -1)], dim=-1))
            synergy = self.synergy_head(h) * self.gate_head(h)
            return synergy.norm(dim=-1).view(self.num_modalities, self.num_modalities)


class MultimodalComplExModel(nn.Module):
    """ComplEx base scoring augmented with cross-modal attention and PID synergy.

    For entities without modality data, falls back to pure ComplEx embeddings.
    Supports modality settings via the `modalities` argument.
    """

    MODALITY_NAME_TO_KEY = {"CXR": "cxr", "ECG": "ecg", "RAD": "text", None: "none"}
    MODALITY_NAME_TO_IDX = {"CXR": 0, "ECG": 1, "RAD": 2, None: 3}

    precompute_features = True

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        num_modalities: int = 4,
        synergy_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_modality_encoders: bool = False,
        use_pretrained_encoders: bool = False,
        modalities: set[str] | None = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.real_embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.synergy_dim = synergy_dim
        self.use_modality_encoders = use_modality_encoders
        self.modalities = modalities or {"text"}

        self.entity_embeddings = nn.Embedding(num_entities, embed_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim * 2)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        nn.init.xavier_uniform_(self.modality_embeddings.weight)

        self.has_modality_logit = nn.Parameter(torch.zeros(num_entities, 1))

        if use_modality_encoders:
            self.modality_encoder = ModalityEncoder(
                unified_dim=embed_dim,
                use_pretrained=use_pretrained_encoders,
                precompute=True,
            )
            self.feature_unifier = FeatureUnifier(unified_dim=embed_dim, dropout=dropout)
        else:
            self.modality_encoder = None
            self.feature_unifier = None

        self.cross_modal_attn = CrossModalAttention(
            unified_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_modal_norm = nn.LayerNorm(embed_dim)

        self.pid_synergy = PIDSynergy(
            num_modalities=num_modalities,
            num_relations=num_relations,
            synergy_dim=synergy_dim,
            dropout=dropout,
        )

        self.synergy_proj = nn.Linear(synergy_dim, embed_dim)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.modulation_residual_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def set_precomputed_features(self, entity_features: dict[int, dict[str, FloatTensor]]):
        self._entity_features = entity_features
        self._feature_tensor = None
        self._feature_mask = None
        if entity_features:
            max_id = max(entity_features.keys()) + 1
            self._feature_tensor = torch.zeros(max_id, self.embed_dim)
            self._feature_mask = torch.zeros(max_id, dtype=torch.bool)
            indices = []
            values = []
            for eid, feat_dict in entity_features.items():
                if feat_dict:
                    indices.append(eid)
                    feats = list(feat_dict.values())
                    if len(feats) == 1:
                        val = feats[0]
                        if val.dim() > 1:
                            val = val.squeeze(0)
                    else:
                        stacked = torch.stack([f.squeeze(0) if f.dim() > 1 else f for f in feats])
                        val = stacked.mean(dim=0)
                    values.append(val)
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                val_tensor = torch.stack(values)
                self._feature_tensor.index_copy_(0, idx_tensor, val_tensor)
                self._feature_mask[idx_tensor] = True
            dev = self.entity_embeddings.weight.device
            self._feature_tensor = self._feature_tensor.to(dev)
            self._feature_mask = self._feature_mask.to(dev)

    def _get_entity_emb(
        self,
        entity_ids: LongTensor,
        entity_modality_ids: LongTensor,
    ) -> tuple[FloatTensor, FloatTensor, FloatTensor]:
        emb = self.entity_embeddings(entity_ids)
        d = self.real_embed_dim
        re, im = emb[:, :d], emb[:, d:]

        has_mod = torch.sigmoid(self.has_modality_logit[entity_ids]).squeeze(-1)
        mod_emb = self.modality_embeddings(entity_modality_ids[entity_ids])
        re = re + mod_emb * has_mod.unsqueeze(-1)

        return re, im, has_mod

    def _compute_cross_modal_context(
        self,
        head_ids: LongTensor,
        tail_ids: LongTensor,
        head_mod_ids: LongTensor,
        tail_mod_ids: LongTensor,
    ) -> FloatTensor:
        B = head_ids.size(0)
        ctx = torch.zeros(B, self.embed_dim, device=head_ids.device)

        if not hasattr(self, "_entity_features") or self._entity_features is None:
            return ctx

        cross_mask = head_mod_ids != tail_mod_ids
        if not cross_mask.any():
            return ctx

        head_idx = head_ids[cross_mask]
        tail_idx = tail_ids[cross_mask]

        if not hasattr(self, "_feature_tensor") or self._feature_tensor is None:
            return ctx

        max_id = self._feature_tensor.size(0)
        h_valid = (head_idx < max_id) & self._feature_mask[head_idx]
        t_valid = (tail_idx < max_id) & self._feature_mask[tail_idx]
        both_valid = h_valid & t_valid

        if not both_valid.any():
            return ctx

        valid_head = head_idx[both_valid]
        valid_tail = tail_idx[both_valid]

        h_feats = self._feature_tensor[valid_head]
        t_feats = self._feature_tensor[valid_tail]
        h_feats = F.normalize(h_feats, p=2, dim=-1)
        t_feats = F.normalize(t_feats, p=2, dim=-1)

        h_ctx, _ = self.cross_modal_attn(h_feats, t_feats)
        h_ctx = self.cross_modal_norm(h_ctx)

        cross_indices = cross_mask.nonzero(as_tuple=True)[0]
        ctx[cross_indices[both_valid]] = h_ctx

        return ctx

    def _complEx_score(
        self,
        h_re: FloatTensor, h_im: FloatTensor,
        r_re: FloatTensor, r_im: FloatTensor,
        t_re: FloatTensor, t_im: FloatTensor,
    ) -> FloatTensor:
        return (
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def n3_penalty(self, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        d = self.real_embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        return (h_re.norm(p=3, dim=-1).mean() + h_im.norm(p=3, dim=-1).mean()
                + r_re.norm(p=3, dim=-1).mean() + r_im.norm(p=3, dim=-1).mean()
                + t_re.norm(p=3, dim=-1).mean() + t_im.norm(p=3, dim=-1).mean())

    def clamp_embed_norm(self, max_norm: float = 1.0):
        with torch.no_grad():
            nn.functional.normalize(self.entity_embeddings.weight, p=2, dim=-1, out=self.entity_embeddings.weight)
            self.entity_embeddings.weight.mul_(max_norm)
            nn.functional.normalize(self.relation_embeddings.weight, p=2, dim=-1, out=self.relation_embeddings.weight)
            self.relation_embeddings.weight.mul_(max_norm)
            if hasattr(self, 'has_modality_logit'):
                self.has_modality_logit.clamp_(-5.0, 5.0)

    def score(
        self,
        heads: LongTensor,
        relations: LongTensor,
        tails: LongTensor,
        entity_modality_ids: LongTensor,
    ) -> FloatTensor:
        h_re, h_im, h_has = self._get_entity_emb(heads, entity_modality_ids)
        t_re, t_im, t_has = self._get_entity_emb(tails, entity_modality_ids)

        r = self.relation_embeddings(relations)
        d = self.real_embed_dim
        r_re, r_im = r[:, :d], r[:, d:]

        if self.training and self.dropout is not None:
            h_re = self.dropout(h_re)
            h_im = self.dropout(h_im)
            t_re = self.dropout(t_re)
            t_im = self.dropout(t_im)
            r_re = self.dropout(r_re)
            r_im = self.dropout(r_im)

        base_score = self._complEx_score(h_re, h_im, r_re, r_im, t_re, t_im)

        h_mod = entity_modality_ids[heads]
        t_mod = entity_modality_ids[tails]
        cross_modal_mask = h_mod != t_mod

        if cross_modal_mask.any():
            synergy = self.pid_synergy(h_mod[cross_modal_mask], t_mod[cross_modal_mask], relations[cross_modal_mask])
            synergy_vec = self.synergy_proj(synergy)

            ctx = self._compute_cross_modal_context(heads, tails, h_mod, t_mod)
            ctx_cross = ctx[cross_modal_mask]

            combined = torch.cat([synergy_vec, ctx_cross], dim=-1)
            modulation_cross = self.modulation_mlp(combined) + self.modulation_residual_proj(combined)

            modulation = torch.zeros_like(h_re)
            modulation[cross_modal_mask] = modulation_cross if modulation_cross.dtype == modulation.dtype else modulation_cross.to(modulation.dtype)

            augmented_re = h_re + modulation
            augmented_im = h_im + modulation
            aug_score = self._complEx_score(augmented_re, augmented_im, r_re, r_im, t_re, t_im)
            base_score = torch.where(cross_modal_mask, aug_score, base_score)

        return base_score

    def forward(
        self, triples: LongTensor, entity_modality_ids: LongTensor
    ) -> FloatTensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2], entity_modality_ids)

    def get_pid_synergy_matrix(self) -> FloatTensor:
        return self.pid_synergy.get_synergy_matrix()


class MultimodalCASCADEModel(nn.Module):
    """Full multimodal CASCADE model: MultimodalComplEx + PID synergy + cross-modal attention.

    This is the main contribution model. Supports 4 modality settings:
      - {'text'}: text-only (radiology reports)
      - {'text', 'image'}: text + CXR images
      - {'text', 'image', 'ecg'}: text + CXR + ECG waveforms
      - {'text', 'image', 'ecg', 'structured'}: all modalities

    For entities without modality data, falls back to pure ComplEx embeddings.
    """

    precompute_features = True

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        num_entity_types: int = 5,
        num_modalities: int = 4,
        synergy_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_modality_encoders: bool = False,
        use_pretrained_encoders: bool = False,
        modalities: set[str] | None = None,
        ablation: str | None = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.real_embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.ablation = ablation
        self.modalities = modalities or {"text"}

        self.entity_embeddings = nn.Embedding(num_entities, embed_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim * 2)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Scale auxiliary embeddings to match entity embedding per-element magnitude.
        # xavier_uniform_ bound depends on num_embeddings (fan_out), so tiny tables
        # (5 modalities, 12 types) get ~40x larger norms than the 421K entity table.
        # Using the entity table's bound for all auxiliary embeddings fixes this.
        _ent_bound = (6.0 / (num_entities + embed_dim * 2)) ** 0.5

        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        nn.init.uniform_(self.modality_embeddings.weight, -_ent_bound, _ent_bound)

        self.entity_type_embeddings = nn.Embedding(num_entity_types, embed_dim)
        nn.init.uniform_(self.entity_type_embeddings.weight, -_ent_bound, _ent_bound)

        self.has_modality_logit = nn.Parameter(torch.zeros(num_entities, 1))

        if use_modality_encoders:
            self.modality_encoder = ModalityEncoder(
                unified_dim=embed_dim,
                use_pretrained=use_pretrained_encoders,
                precompute=True,
            )
            self.feature_unifier = FeatureUnifier(unified_dim=embed_dim, dropout=dropout)
        else:
            self.modality_encoder = None
            self.feature_unifier = None

        self.cross_modal_attn = CrossModalAttention(
            unified_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_modal_norm = nn.LayerNorm(embed_dim)

        self.pid_synergy = PIDSynergy(
            num_modalities=num_modalities,
            num_relations=num_relations,
            synergy_dim=synergy_dim,
            dropout=dropout,
        )

        self.synergy_proj = nn.Linear(synergy_dim, embed_dim)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.modulation_residual_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def set_precomputed_features(self, entity_features: dict[int, dict[str, FloatTensor]]):
        self._entity_features = entity_features
        self._feature_tensor = None
        self._feature_mask = None
        if entity_features:
            max_id = max(entity_features.keys()) + 1
            self._feature_tensor = torch.zeros(max_id, self.embed_dim)
            self._feature_mask = torch.zeros(max_id, dtype=torch.bool)
            indices = []
            values = []
            for eid, feat_dict in entity_features.items():
                if feat_dict:
                    indices.append(eid)
                    feats = list(feat_dict.values())
                    if len(feats) == 1:
                        val = feats[0]
                        if val.dim() > 1:
                            val = val.squeeze(0)
                    else:
                        stacked = torch.stack([f.squeeze(0) if f.dim() > 1 else f for f in feats])
                        val = stacked.mean(dim=0)
                    values.append(val)
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                val_tensor = torch.stack(values)
                self._feature_tensor.index_copy_(0, idx_tensor, val_tensor)
                self._feature_mask[idx_tensor] = True
            dev = self.entity_embeddings.weight.device
            self._feature_tensor = self._feature_tensor.to(dev)
            self._feature_mask = self._feature_mask.to(dev)

    def _get_entity_emb(
        self,
        entity_ids: LongTensor,
        entity_type_ids: LongTensor,
        entity_modality_ids: LongTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
        emb = self.entity_embeddings(entity_ids)
        d = self.real_embed_dim
        re, im = emb[:, :d], emb[:, d:]

        has_mod = torch.sigmoid(self.has_modality_logit[entity_ids]).squeeze(-1)

        if self.ablation not in ("no_modality", "all"):
            mod_emb = self.modality_embeddings(entity_modality_ids[entity_ids])
            re = re + mod_emb * has_mod.unsqueeze(-1)

        if self.ablation not in ("no_type", "all"):
            type_emb = self.entity_type_embeddings(entity_type_ids[entity_ids])
            re = re + type_emb

        return re, im

    def _compute_cross_modal_context(
        self,
        head_ids: LongTensor,
        tail_ids: LongTensor,
        head_mod_ids: LongTensor,
        tail_mod_ids: LongTensor,
    ) -> FloatTensor:
        B = head_ids.size(0)
        ctx = torch.zeros(B, self.embed_dim, device=head_ids.device)

        if not hasattr(self, "_entity_features") or self._entity_features is None:
            return ctx

        cross_mask = head_mod_ids != tail_mod_ids
        if not cross_mask.any():
            return ctx

        head_idx = head_ids[cross_mask]
        tail_idx = tail_ids[cross_mask]

        if not hasattr(self, "_feature_tensor") or self._feature_tensor is None:
            return ctx

        max_id = self._feature_tensor.size(0)
        h_valid = (head_idx < max_id) & self._feature_mask[head_idx]
        t_valid = (tail_idx < max_id) & self._feature_mask[tail_idx]
        both_valid = h_valid & t_valid

        if not both_valid.any():
            return ctx

        valid_head = head_idx[both_valid]
        valid_tail = tail_idx[both_valid]

        h_feats = self._feature_tensor[valid_head]
        t_feats = self._feature_tensor[valid_tail]
        h_feats = F.normalize(h_feats, p=2, dim=-1)
        t_feats = F.normalize(t_feats, p=2, dim=-1)

        h_ctx, _ = self.cross_modal_attn(h_feats, t_feats)
        h_ctx = self.cross_modal_norm(h_ctx)

        cross_indices = cross_mask.nonzero(as_tuple=True)[0]
        ctx[cross_indices[both_valid]] = h_ctx

        return ctx

    def _complEx_score(
        self,
        h_re: FloatTensor, h_im: FloatTensor,
        r_re: FloatTensor, r_im: FloatTensor,
        t_re: FloatTensor, t_im: FloatTensor,
    ) -> FloatTensor:
        return (
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def n3_penalty(self, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        d = self.real_embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        return (h_re.norm(p=3, dim=-1).mean() + h_im.norm(p=3, dim=-1).mean()
                + r_re.norm(p=3, dim=-1).mean() + r_im.norm(p=3, dim=-1).mean()
                + t_re.norm(p=3, dim=-1).mean() + t_im.norm(p=3, dim=-1).mean())

    def clamp_embed_norm(self, max_norm: float = 1.0):
        with torch.no_grad():
            nn.functional.normalize(self.entity_embeddings.weight, p=2, dim=-1, out=self.entity_embeddings.weight)
            self.entity_embeddings.weight.mul_(max_norm)
            nn.functional.normalize(self.relation_embeddings.weight, p=2, dim=-1, out=self.relation_embeddings.weight)
            self.relation_embeddings.weight.mul_(max_norm)
            if hasattr(self, 'has_modality_logit'):
                self.has_modality_logit.clamp_(-5.0, 5.0)

    def score(
        self,
        heads: LongTensor,
        relations: LongTensor,
        tails: LongTensor,
        entity_type_ids: LongTensor,
        entity_modality_ids: LongTensor,
    ) -> FloatTensor:
        h_re, h_im = self._get_entity_emb(heads, entity_type_ids, entity_modality_ids)
        t_re, t_im = self._get_entity_emb(tails, entity_type_ids, entity_modality_ids)

        r = self.relation_embeddings(relations)
        d = self.real_embed_dim
        r_re, r_im = r[:, :d], r[:, d:]

        if self.training and self.dropout is not None:
            h_re = self.dropout(h_re)
            h_im = self.dropout(h_im)
            t_re = self.dropout(t_re)
            t_im = self.dropout(t_im)
            r_re = self.dropout(r_re)
            r_im = self.dropout(r_im)

        base_score = self._complEx_score(h_re, h_im, r_re, r_im, t_re, t_im)

        if self.ablation not in ("no_pid", "all"):
            h_mod = entity_modality_ids[heads]
            t_mod = entity_modality_ids[tails]
            cross_modal_mask = h_mod != t_mod

            if cross_modal_mask.any():
                synergy = self.pid_synergy(h_mod[cross_modal_mask], t_mod[cross_modal_mask], relations[cross_modal_mask])
                synergy_vec = self.synergy_proj(synergy)

                ctx = self._compute_cross_modal_context(heads, tails, h_mod, t_mod)
                ctx_cross = ctx[cross_modal_mask]

                combined = torch.cat([synergy_vec, ctx_cross], dim=-1)
                modulation_cross = self.modulation_mlp(combined) + self.modulation_residual_proj(combined)

                modulation = torch.zeros_like(h_re)
                modulation[cross_modal_mask] = modulation_cross if modulation_cross.dtype == modulation.dtype else modulation_cross.to(modulation.dtype)

                augmented_re = h_re + modulation
                augmented_im = h_im + modulation
                aug_score = self._complEx_score(augmented_re, augmented_im, r_re, r_im, t_re, t_im)
                base_score = torch.where(cross_modal_mask, aug_score, base_score)

        return base_score

    def forward(
        self, triples: LongTensor, entity_type_ids: LongTensor, entity_modality_ids: LongTensor
    ) -> FloatTensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2], entity_type_ids, entity_modality_ids)

    def get_pid_synergy_matrix(self) -> FloatTensor:
        return self.pid_synergy.get_synergy_matrix()


class CASCADEKGModel(nn.Module):
    """PID-weighted ComplEx: ComplEx scoring with modality/type augmentation and
    learnable PID synergy weights that amplify cross-modal triples."""

    precompute_features = False

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 256,
        num_entity_types: int = 5,
        num_modalities: int = 4,
        ablation: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.real_embed_dim = embed_dim
        self.ablation = ablation

        self.entity_embeddings = nn.Embedding(num_entities, embed_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim * 2)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        nn.init.xavier_uniform_(self.modality_embeddings.weight)

        self.entity_type_embeddings = nn.Embedding(num_entity_types, embed_dim)
        nn.init.xavier_uniform_(self.entity_type_embeddings.weight)

        self.pid_synergy_raw = nn.Parameter(torch.zeros(num_modalities, num_modalities))
        self._softplus_zero = F.softplus(torch.tensor(0.0))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def _get_entity_emb(
        self, entity_ids: LongTensor, entity_type_ids: LongTensor, entity_modality_ids: LongTensor
    ) -> tuple[FloatTensor, FloatTensor]:
        emb = self.entity_embeddings(entity_ids)
        d = self.real_embed_dim
        re, im = emb[:, :d], emb[:, d:]

        if self.ablation != "no_modality":
            mod_emb = self.modality_embeddings(entity_modality_ids[entity_ids])
            re = re + mod_emb
        if self.ablation != "no_type":
            type_emb = self.entity_type_embeddings(entity_type_ids[entity_ids])
            re = re + type_emb

        return re, im

    def _complEx_score(
        self,
        h_re: FloatTensor, h_im: FloatTensor,
        r_re: FloatTensor, r_im: FloatTensor,
        t_re: FloatTensor, t_im: FloatTensor,
    ) -> FloatTensor:
        return (
            h_re * r_re * t_re
            + h_re * r_im * t_im
            + h_im * r_re * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def score(
        self,
        heads: LongTensor,
        relations: LongTensor,
        tails: LongTensor,
        entity_type_ids: LongTensor,
        entity_modality_ids: LongTensor,
    ) -> FloatTensor:
        h_re, h_im = self._get_entity_emb(heads, entity_type_ids, entity_modality_ids)
        t_re, t_im = self._get_entity_emb(tails, entity_type_ids, entity_modality_ids)

        r = self.relation_embeddings(relations)
        d = self.real_embed_dim
        r_re, r_im = r[:, :d], r[:, d:]

        if self.training and self.dropout is not None:
            h_re = self.dropout(h_re)
            h_im = self.dropout(h_im)
            t_re = self.dropout(t_re)
            t_im = self.dropout(t_im)
            r_re = self.dropout(r_re)
            r_im = self.dropout(r_im)

        base_score = self._complEx_score(h_re, h_im, r_re, r_im, t_re, t_im)

        if self.ablation not in ("no_pid", "all"):
            h_mod = entity_modality_ids[heads]
            t_mod = entity_modality_ids[tails]
            cross_modal_mask = h_mod != t_mod

            if cross_modal_mask.any():
                sp0 = self._softplus_zero.to(base_score.device)
                pid_weight = 1.0 + F.softplus(self.pid_synergy_raw[h_mod[cross_modal_mask], t_mod[cross_modal_mask]]) - sp0
                base_score[cross_modal_mask] = base_score[cross_modal_mask] * pid_weight

        return base_score

    def forward(
        self, triples: LongTensor, entity_type_ids: LongTensor, entity_modality_ids: LongTensor
    ) -> FloatTensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2], entity_type_ids, entity_modality_ids)

    def get_pid_synergy_matrix(self) -> FloatTensor:
        sp0 = self._softplus_zero.to(self.pid_synergy_raw.device)
        return 1.0 + F.softplus(self.pid_synergy_raw) - sp0

    def n3_penalty(self, head_ids: LongTensor, rel_ids: LongTensor, tail_ids: LongTensor) -> FloatTensor:
        h = self.entity_embeddings(head_ids)
        r = self.relation_embeddings(rel_ids)
        t = self.entity_embeddings(tail_ids)
        d = self.real_embed_dim
        h_re, h_im = h[:, :d], h[:, d:]
        r_re, r_im = r[:, :d], r[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        return (h_re.norm(p=3, dim=-1).mean() + h_im.norm(p=3, dim=-1).mean()
                + r_re.norm(p=3, dim=-1).mean() + r_im.norm(p=3, dim=-1).mean()
                + t_re.norm(p=3, dim=-1).mean() + t_im.norm(p=3, dim=-1).mean())

    def clamp_embed_norm(self, max_norm: float = 1.0):
        with torch.no_grad():
            nn.functional.normalize(self.entity_embeddings.weight, p=2, dim=-1, out=self.entity_embeddings.weight)
            self.entity_embeddings.weight.mul_(max_norm)
            nn.functional.normalize(self.relation_embeddings.weight, p=2, dim=-1, out=self.relation_embeddings.weight)
            self.relation_embeddings.weight.mul_(max_norm)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1, label_smoothing: float = 0.0, learnable: bool = True):
        super().__init__()
        self.label_smoothing = label_smoothing
        if learnable:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> FloatTensor:
        return self.log_temperature.exp()

    def forward(
        self, pos_scores: FloatTensor, neg_scores: FloatTensor
    ) -> FloatTensor:
        temp = self.temperature.clamp(min=0.01, max=2.0)
        pos_scaled = pos_scores / temp
        neg_scaled = neg_scores / temp
        logits = torch.cat([pos_scaled, neg_scaled], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing if self.training else 0.0)
        with torch.no_grad():
            self.log_temperature.clamp_(min=math.log(0.01), max=math.log(2.0))
        return loss
