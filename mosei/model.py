import math
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
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.mean(dim=1)
        return self.norm(z)


class SparseTMoE(nn.Module):
    """
    Paper-aligned sparse gated Transformer MoE for one modality.
    Gate(e_spe_m) = Softmax(KeepTopK(W_g e_spe_m, k))
    """
    def __init__(self, input_dim: int, num_experts: int = 3, top_k: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            TransformerExpert(input_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_input = x.mean(dim=1)
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

        aux = {
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "gate_mask": gate_mask,
            "importance": gate_probs.mean(dim=0),
            "load": gate_mask.mean(dim=0),
            "expert_outputs": expert_outputs,
        }
        return mixed, aux


class DynamicWeightedFusion(nn.Module):
    """
    Paper-aligned DWF (Eq. 15-16).

    N: shared representation from HGL, shape [B, 3, d_n]
    T: modality-specific representation from TMoEs, shape [B, 3, d_t]
    U = [N ⊕ T] along feature dimension -> [B, 3, d_n + d_t]
    Q = W_Q U, K = W_K U, V = W_V U
    F = Concat(Softmax(Q K^T / sqrt(d_k)) V)
    """
    def __init__(self, shared_dim: int, private_dim: int, fusion_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        input_dim = shared_dim + private_dim
        self.q_proj = nn.Linear(input_dim, fusion_dim)
        self.k_proj = nn.Linear(input_dim, fusion_dim)
        self.v_proj = nn.Linear(input_dim, fusion_dim)
        self.out_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_dim = fusion_dim

    def forward(self, shared_repr: torch.Tensor, specific_repr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        u = torch.cat([shared_repr, specific_repr], dim=-1)
        q = self.q_proj(u)
        k = self.k_proj(u)
        v = self.v_proj(u)

        attn_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        attn = torch.softmax(attn_logits, dim=-1)
        fused_tokens = torch.matmul(attn, v)
        fused_tokens = self.out_norm(self.dropout(fused_tokens))
        fused_repr = fused_tokens.reshape(fused_tokens.size(0), -1)

        aux = {
            "u": u,
            "q": q,
            "k": k,
            "v": v,
            "attn": attn,
            "fused_tokens": fused_tokens,
            "fused_repr": fused_repr,
        }
        return fused_repr, aux


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
    Paper-aligned DHM baseline for regression setting:
    Conv1D -> feature decoupling -> HGL(shared) + TMoEs(private) -> DWF -> prediction

    DTF and TGIB are removed and replaced by the paper's DWF module.
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

        self.dwf = DynamicWeightedFusion(
            shared_dim=hyper_hidden,
            private_dim=private_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        self.regressor = RegressionHead(input_dim=3 * fusion_dim, hidden_dim=fusion_dim, dropout=dropout)

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

        shared_sequences = torch.stack([e_irr_t, e_irr_v, e_irr_a], dim=1)
        shared_repr, hyper_aux = self.hypergraph(shared_sequences)  # [B, 3, hyper_hidden]

        p_t, tmoe_t_aux = self.tmoe_t(e_spe_t)
        p_v, tmoe_v_aux = self.tmoe_v(e_spe_v)
        p_a, tmoe_a_aux = self.tmoe_a(e_spe_a)
        specific_repr = torch.stack([p_t, p_v, p_a], dim=1)  # [B, 3, private_dim]

        fused_repr, dwf_aux = self.dwf(shared_repr, specific_repr)
        pred = self.regressor(fused_repr)

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
            "shared_repr": shared_repr,
            "specific_repr": specific_repr,
            "fused_repr": fused_repr,
            "hyper_aux": hyper_aux,
            "tmoe_t_aux": tmoe_t_aux,
            "tmoe_v_aux": tmoe_v_aux,
            "tmoe_a_aux": tmoe_a_aux,
            "dwf_aux": dwf_aux,
        }
        return pred, aux
