from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypergraph import HypergraphEncoder


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(z)


class SequenceMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class TransformerExpert(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = AttentionPool(input_dim, hidden_dim=max(64, input_dim), dropout=dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        pooled, _ = self.pool(z)
        return self.norm(pooled)


class SparseTMoE(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 3, top_k: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.gate_pool = AttentionPool(input_dim, hidden_dim=max(64, input_dim), dropout=dropout)
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            TransformerExpert(input_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_input, gate_attn = self.gate_pool(x)
        gate_logits = self.gate(gate_input)
        top_vals, top_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        sparse_logits = torch.full_like(gate_logits, fill_value=-1e9)
        sparse_logits.scatter_(1, top_idx, top_vals)
        gate_probs = torch.softmax(sparse_logits, dim=-1)
        gate_mask = torch.zeros_like(gate_logits)
        gate_mask.scatter_(1, top_idx, 1.0)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)
        mixed = self.norm(mixed)

        importance = gate_probs.mean(dim=0)
        load = gate_mask.mean(dim=0)
        aux = {
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "gate_mask": gate_mask,
            "importance": importance,
            "load": load,
            "gate_attn": gate_attn,
            "expert_outputs": expert_outputs,
        }
        return mixed, aux


class TokenLevelDynamicWeighting(nn.Module):
    """
    目标从“温和但几乎被 text-private 长期占优”改成：
    1) shared hypergraph token 作为主导锚点；
    2) private token 作为补充，不允许被压死；
    3) 最终 token 融合既保留动态性，又带有 shared-dominant 的结构先验。
    """
    def __init__(
        self,
        shared_dim: int,
        private_dim: int,
        token_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.05,
        flatten_power: float = 0.95,
        prior_mix: float = 0.35,
        shared_min_weight: float = 0.28,
        private_min_weight: float = 0.10,
        gate_scale: float = 0.55,
        shared_to_private_scale: float = 0.12,
    ):
        super().__init__()
        self.temperature = max(1e-3, float(temperature))
        self.flatten_power = float(flatten_power)
        self.prior_mix = float(prior_mix)
        self.shared_min_weight = float(shared_min_weight)
        self.private_min_weight = float(private_min_weight)
        self.gate_scale = float(gate_scale)
        self.shared_to_private_scale = float(shared_to_private_scale)
        self.num_tokens = 4

        self.shared_proj = nn.Sequential(
            nn.Linear(shared_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.private_t_proj = nn.Sequential(
            nn.Linear(private_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.private_v_proj = nn.Sequential(
            nn.Linear(private_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.private_a_proj = nn.Sequential(
            nn.Linear(private_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

        self.token_type_embed = nn.Parameter(torch.zeros(1, self.num_tokens, token_dim))
        nn.init.trunc_normal_(self.token_type_embed, std=0.02)

        self.context_proj = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_proj = nn.Linear(token_dim, token_dim)
        self.key_proj = nn.Linear(token_dim, token_dim)
        self.local_score = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 1),
        )
        self.shared_affinity_proj = nn.Linear(token_dim, token_dim)
        self.base_prior_logits = nn.Parameter(torch.tensor([1.10, 0.0, 0.0, 0.0], dtype=torch.float32))
        self.token_bias = nn.Parameter(torch.tensor([0.35, 0.0, 0.0, 0.0], dtype=torch.float32))
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        shared_repr: torch.Tensor,
        private_t: torch.Tensor,
        private_v: torch.Tensor,
        private_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shared_token = self.shared_proj(shared_repr)
        t_token = self.private_t_proj(private_t)
        v_token = self.private_v_proj(private_v)
        a_token = self.private_a_proj(private_a)

        raw_tokens = torch.stack([shared_token, t_token, v_token, a_token], dim=1)
        typed_tokens = raw_tokens + self.token_type_embed

        shared_anchor = typed_tokens[:, 0, :]
        private_tokens = typed_tokens[:, 1:, :]
        private_mean = private_tokens.mean(dim=1)
        private_max = private_tokens.amax(dim=1)
        global_ctx = self.context_proj(torch.cat([shared_anchor, private_mean, private_max], dim=-1))

        query = F.normalize(self.query_proj(global_ctx), dim=-1)
        keys = F.normalize(self.key_proj(typed_tokens), dim=-1)
        compat_score = torch.sum(keys * query.unsqueeze(1), dim=-1)

        ctx_expand = global_ctx.unsqueeze(1).expand_as(typed_tokens)
        delta = typed_tokens - ctx_expand
        local_score = self.local_score(torch.cat([typed_tokens, delta], dim=-1)).squeeze(-1)

        shared_ref = F.normalize(self.shared_affinity_proj(shared_anchor), dim=-1)
        shared_affinity = torch.sum(F.normalize(typed_tokens, dim=-1) * shared_ref.unsqueeze(1), dim=-1)

        private_consistency = F.cosine_similarity(
            private_tokens,
            private_mean.unsqueeze(1).expand_as(private_tokens),
            dim=-1,
        )
        private_bonus = torch.cat([
            torch.zeros_like(shared_affinity[:, :1]),
            0.10 * private_consistency,
        ], dim=1)

        prior_logits = self.base_prior_logits.view(1, -1)
        token_scores = (
            compat_score
            + 0.25 * local_score
            + 0.35 * shared_affinity
            + private_bonus
            + self.token_bias.view(1, -1)
            + prior_logits
        )
        token_weights = torch.softmax(token_scores / self.temperature, dim=1)

        if abs(self.flatten_power - 1.0) > 1e-6:
            token_weights = token_weights.pow(self.flatten_power)
            token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        prior_dist = torch.softmax(self.base_prior_logits, dim=0).view(1, -1)
        if self.prior_mix > 0.0:
            token_weights = (1.0 - self.prior_mix) * token_weights + self.prior_mix * prior_dist

        min_floor = torch.tensor(
            [self.shared_min_weight, self.private_min_weight, self.private_min_weight, self.private_min_weight],
            device=token_weights.device,
            dtype=token_weights.dtype,
        ).view(1, -1)
        token_weights = torch.maximum(token_weights, min_floor)
        token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        guided_tokens = typed_tokens.clone()
        guided_tokens[:, 1:, :] = guided_tokens[:, 1:, :] + self.shared_to_private_scale * shared_anchor.unsqueeze(1)

        token_scale = 1.0 + self.gate_scale * (token_weights - prior_dist)
        token_scale = token_scale.clamp(min=0.85, max=1.30)
        weighted_tokens = self.out_norm(guided_tokens * token_scale.unsqueeze(-1))

        dominance_margin = token_weights[:, 0] - token_weights[:, 1:].max(dim=1).values
        aux = {
            "raw_tokens": raw_tokens,
            "typed_tokens": typed_tokens,
            "guided_tokens": guided_tokens,
            "weighted_tokens": weighted_tokens,
            "token_scores": token_scores,
            "token_weights": token_weights,
            "token_scale": token_scale,
            "prior_dist": prior_dist.expand(token_weights.size(0), -1),
            "dominance_margin": dominance_margin,
            "shared_token": shared_token,
            "text_token": t_token,
            "vision_token": v_token,
            "audio_token": a_token,
            "global_context": global_ctx,
        }
        return weighted_tokens, aux


class TokenTransformerBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 4, token_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights


class TransformerGuidedIB(nn.Module):
    def __init__(self, token_dim: int, num_tokens: int = 4, num_heads: int = 4, dropout: float = 0.1, latent_dim: int = None):
        super().__init__()
        latent_dim = latent_dim or token_dim
        self.token_embed = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        nn.init.trunc_normal_(self.token_embed, std=0.02)
        self.block = TokenTransformerBlock(token_dim=token_dim, num_heads=num_heads, dropout=dropout)
        self.pre_norm = nn.LayerNorm(token_dim)
        self.mu_proj = nn.Linear(token_dim, latent_dim)
        self.logvar_proj = nn.Linear(token_dim, latent_dim)
        self.post = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, token_weights: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = tokens + self.token_embed[:, :tokens.size(1), :]
        x, attn_weights = self.block(x)
        if token_weights is None:
            pool_weights = torch.full(
                (x.size(0), x.size(1)),
                1.0 / float(x.size(1)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            pool_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = self.pre_norm(torch.sum(x * pool_weights.unsqueeze(-1), dim=1))
        mu = self.mu_proj(pooled)
        logvar = self.logvar_proj(pooled).clamp(min=-8.0, max=8.0)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std)
            z = mu
        z = self.post(z)

        aux = {
            "transformer_tokens": x,
            "transformer_attn": attn_weights,
            "pre_bottleneck": pooled,
            "pool_weights": pool_weights,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "eps": eps,
            "post_bottleneck": z,
        }
        return z, aux


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DHMModel(nn.Module):
    """
    Conv1D -> Factorized(shared/private) -> HGL(shared) + TMoEs(private)
    -> 4-token moderate dynamic weighting -> Transformer-guided IB -> Prediction
    """
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        conv_hidden: int = 128,
        shared_dim: int = 64,
        private_dim: int = 64,
        hyper_hidden: int = 96,
        fusion_dim: int = 128,
        num_experts: int = 3,
        top_k: int = 1,
        num_heads: int = 4,
        hg_layers: int = 3,
        intra_k: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.text_conv = TemporalConvEncoder(text_dim, conv_hidden, kernel_size=3, dropout=dropout)
        self.vision_conv = TemporalConvEncoder(vision_dim, conv_hidden, kernel_size=3, dropout=dropout)
        self.audio_conv = TemporalConvEncoder(audio_dim, conv_hidden, kernel_size=3, dropout=dropout)

        self.shared_t = SequenceMLPEncoder(conv_hidden, shared_dim, dropout=dropout)
        self.shared_v = SequenceMLPEncoder(conv_hidden, shared_dim, dropout=dropout)
        self.shared_a = SequenceMLPEncoder(conv_hidden, shared_dim, dropout=dropout)

        self.private_t = SequenceMLPEncoder(conv_hidden, private_dim, dropout=dropout)
        self.private_v = SequenceMLPEncoder(conv_hidden, private_dim, dropout=dropout)
        self.private_a = SequenceMLPEncoder(conv_hidden, private_dim, dropout=dropout)

        self.decoder_t = SequenceDecoder(shared_dim + private_dim, conv_hidden, dropout=dropout)
        self.decoder_v = SequenceDecoder(shared_dim + private_dim, conv_hidden, dropout=dropout)
        self.decoder_a = SequenceDecoder(shared_dim + private_dim, conv_hidden, dropout=dropout)

        self.shared_pool_t = AttentionPool(shared_dim, hidden_dim=max(64, shared_dim), dropout=dropout)
        self.shared_pool_v = AttentionPool(shared_dim, hidden_dim=max(64, shared_dim), dropout=dropout)
        self.shared_pool_a = AttentionPool(shared_dim, hidden_dim=max(64, shared_dim), dropout=dropout)

        self.hypergraph = HypergraphEncoder(
            node_dim=shared_dim,
            hidden_dim=hyper_hidden,
            num_layers=hg_layers,
            intra_k=intra_k,
            dropout=dropout,
            max_batch_size=32,
            max_seq_len=64,
        )

        self.tmoe_t = SparseTMoE(private_dim, num_experts=num_experts, top_k=top_k, num_heads=num_heads, dropout=dropout)
        self.tmoe_v = SparseTMoE(private_dim, num_experts=num_experts, top_k=top_k, num_heads=num_heads, dropout=dropout)
        self.tmoe_a = SparseTMoE(private_dim, num_experts=num_experts, top_k=top_k, num_heads=num_heads, dropout=dropout)

        self.token_weighting = TokenLevelDynamicWeighting(
            shared_dim=hyper_hidden,
            private_dim=private_dim,
            token_dim=fusion_dim,
            dropout=dropout,
            temperature=1.05,
            flatten_power=0.95,
            prior_mix=0.35,
            shared_min_weight=0.28,
            private_min_weight=0.10,
            gate_scale=0.55,
            shared_to_private_scale=0.12,
        )
        self.tgib = TransformerGuidedIB(
            token_dim=fusion_dim,
            num_tokens=4,
            num_heads=num_heads,
            dropout=dropout,
            latent_dim=fusion_dim,
        )
        self.regressor = RegressionHead(fusion_dim, hidden_dim=fusion_dim, dropout=dropout)

    def forward(self, text: torch.Tensor, vision: torch.Tensor, audio: torch.Tensor):
        c_t = self.text_conv(text)
        c_v = self.vision_conv(vision)
        c_a = self.audio_conv(audio)

        e_irr_t = self.shared_t(c_t)
        e_irr_v = self.shared_v(c_v)
        e_irr_a = self.shared_a(c_a)
        e_spe_t = self.private_t(c_t)
        e_spe_v = self.private_v(c_v)
        e_spe_a = self.private_a(c_a)

        rec_t = self.decoder_t(torch.cat([e_irr_t, e_spe_t], dim=-1))
        rec_v = self.decoder_v(torch.cat([e_irr_v, e_spe_v], dim=-1))
        rec_a = self.decoder_a(torch.cat([e_irr_a, e_spe_a], dim=-1))

        # 论文中的 HGL 先保留 shared 序列级节点，再在当前 batch 内构造超图，
        # 而不是先把每个模态池化成单个节点。
        shared_sequences = torch.stack([e_irr_t, e_irr_v, e_irr_a], dim=1)
        hyper_repr, hyper_aux = self.hypergraph(shared_sequences)

        p_t, tmoe_t_aux = self.tmoe_t(e_spe_t)
        p_v, tmoe_v_aux = self.tmoe_v(e_spe_v)
        p_a, tmoe_a_aux = self.tmoe_a(e_spe_a)

        weighted_tokens, token_fusion_aux = self.token_weighting(hyper_repr, p_t, p_v, p_a)
        z, tgib_aux = self.tgib(weighted_tokens, token_fusion_aux["token_weights"])
        pred = self.regressor(z)

        aux = {
            "c_t": c_t,
            "c_v": c_v,
            "c_a": c_a,
            "e_irr_t": e_irr_t,
            "e_irr_v": e_irr_v,
            "e_irr_a": e_irr_a,
            "e_spe_t": e_spe_t,
            "e_spe_v": e_spe_v,
            "e_spe_a": e_spe_a,
            "rec_t": rec_t,
            "rec_v": rec_v,
            "rec_a": rec_a,
            "shared_sequences": shared_sequences,
            "hyper_repr": hyper_repr,
            "private_t": p_t,
            "private_v": p_v,
            "private_a": p_a,
            "token_inputs": token_fusion_aux["raw_tokens"],
            "weighted_tokens": weighted_tokens,
            "fused_repr": torch.sum(token_fusion_aux["weighted_tokens"] * token_fusion_aux["token_weights"].unsqueeze(-1), dim=1),
            "filtered_repr": z,
            "hyper_aux": hyper_aux,
            "tmoe_t_aux": tmoe_t_aux,
            "tmoe_v_aux": tmoe_v_aux,
            "tmoe_a_aux": tmoe_a_aux,
            "token_fusion_aux": token_fusion_aux,
            "tgib_aux": tgib_aux,
        }
        return pred, aux
