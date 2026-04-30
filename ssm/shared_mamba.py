from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba
except Exception:
    _OfficialMamba = None


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


class GlobalLocalContextExtractor(nn.Module):
    """ISM-style global-local context extraction."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_ctx = self.global_proj(x)
        local_ctx = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(self.norm(global_ctx + local_ctx))


class OfficialMambaBlock(nn.Module):
    """Pre-norm residual wrapper around the official mamba_ssm.Mamba operator."""

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
        require_official_mamba: bool = True,
    ):
        super().__init__()
        if _OfficialMamba is None and require_official_mamba:
            raise RuntimeError(
                "mamba_ssm is not installed. Install official Mamba first, for example:\n"
                "pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.2\n"
                "Then rerun training on a CUDA-enabled PyTorch environment."
            )
        if _OfficialMamba is None:
            raise RuntimeError("Official Mamba is required in this version; fallback implementation is intentionally disabled.")

        self.norm = nn.LayerNorm(dim)
        self.mamba = _OfficialMamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=expand,
        )
        self.select_probe = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.norm(x)
        y = self.mamba(z)
        out = self.out_norm(x + self.dropout(y))

        # Official Mamba does not expose its internal B/C/Delta gates. This probe
        # is only used as a stable diagnostic for selectivity/anti-collapse logs.
        select = torch.sigmoid(self.select_probe(z.detach()).float())
        stats = {
            "select_mean": select.mean(),
            "select_std": select.std(unbiased=False) if select.numel() > 1 else select.new_zeros(()),
            "select_spread": select.amax() - select.amin() if select.numel() > 1 else select.new_zeros(()),
        }
        return out, stats


class BiDirectionalMamba(nn.Module):
    """Forward + backward official Mamba scanning, similar to BSSM in MSAmba."""

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
        require_official_mamba: bool = True,
    ):
        super().__init__()
        self.forward_mamba = OfficialMambaBlock(
            dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )
        self.backward_mamba = OfficialMambaBlock(
            dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )
        self.merge = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_f, stats_f = self.forward_mamba(x)
        y_b_rev, stats_b = self.backward_mamba(torch.flip(x, dims=[1]))
        y_b = torch.flip(y_b_rev, dims=[1])
        y = self.merge(torch.cat([y_f, y_b], dim=-1))
        out = self.out_norm(x + y)
        stats = {
            "select_mean": 0.5 * (stats_f["select_mean"] + stats_b["select_mean"]),
            "select_std": 0.5 * (stats_f["select_std"] + stats_b["select_std"]),
            "select_spread": 0.5 * (stats_f["select_spread"] + stats_b["select_spread"]),
        }
        return out, stats


class SharedSelectiveStateMixerLayer(nn.Module):
    """
    One shared branch layer.

    Intra path: each modality's shared sequence is scanned independently.
    Cross path: text-centered language guidance is used to scan [vision, text]
    and [audio, text] pairs, replacing explicit cross-modal hyperedges.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        require_official_mamba: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = bool(use_cross_attention)

        self.intra_glce = GlobalLocalContextExtractor(hidden_dim, dropout=dropout)
        self.intra_mamba = BiDirectionalMamba(
            hidden_dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )
        self.text_glce = GlobalLocalContextExtractor(hidden_dim, dropout=dropout)
        self.text_mamba = BiDirectionalMamba(
            hidden_dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )
        self.vt_glce = GlobalLocalContextExtractor(hidden_dim, dropout=dropout)
        self.at_glce = GlobalLocalContextExtractor(hidden_dim, dropout=dropout)
        self.vt_mamba = BiDirectionalMamba(
            hidden_dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )
        self.at_mamba = BiDirectionalMamba(
            hidden_dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
        )

        self.text_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.intra_cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_pair_sequence(other: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # [B, T, H], [B, T, H] -> [B, 2T, H] as [other_1, text_1, other_2, text_2, ...]
        return torch.stack([other, text], dim=2).reshape(other.size(0), other.size(1) * 2, other.size(2))

    @staticmethod
    def _split_pair_sequence(pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, 2T, H] -> other, text components, each [B, T, H]
        reshaped = pair.reshape(pair.size(0), pair.size(1) // 2, 2, pair.size(2))
        return reshaped[:, :, 0, :], reshaped[:, :, 1, :]

    def forward(self, x: torch.Tensor, text_context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_modalities, seq_len, hidden_dim = x.shape

        # Intra-modal shared scan.
        intra_in = x.reshape(batch_size * num_modalities, seq_len, hidden_dim)
        intra_in = self.intra_glce(intra_in)
        intra_out, intra_stats = self.intra_mamba(intra_in)
        intra_out = intra_out.view(batch_size, num_modalities, seq_len, hidden_dim)

        t = intra_out[:, 0]
        v = intra_out[:, 1]
        a = intra_out[:, 2]
        t_ctx = self.text_context_proj(text_context).unsqueeze(1)

        # Central text scan and text-guided cross-modal pair scans.
        t_scan, t_stats = self.text_mamba(self.text_glce(t + t_ctx))
        vt_in = self.vt_glce(self._make_pair_sequence(v, t + t_ctx))
        at_in = self.at_glce(self._make_pair_sequence(a, t + t_ctx))
        vt_out, vt_stats = self.vt_mamba(vt_in)
        at_out, at_stats = self.at_mamba(at_in)
        v_cross, t_from_v = self._split_pair_sequence(vt_out)
        a_cross, t_from_a = self._split_pair_sequence(at_out)
        t_cross = self.layer_norm(t_scan + 0.50 * (t_from_v + t_from_a))

        cross_out = torch.stack([t_cross, v_cross, a_cross], dim=1)

        if self.use_cross_attention:
            flat = cross_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len * num_modalities, hidden_dim)
            attn_out, _ = self.cross_attn(flat, flat, flat, need_weights=False)
            flat = self.cross_attn_norm(flat + self.dropout(attn_out))
            cross_out = flat.view(batch_size, seq_len, num_modalities, hidden_dim).permute(0, 2, 1, 3)

        gate = self.intra_cross_gate(torch.cat([intra_out, cross_out], dim=-1))
        out = self.layer_norm(x + self.dropout(gate * cross_out + (1.0 - gate) * intra_out))

        cross_mean = (t_stats["select_mean"] + vt_stats["select_mean"] + at_stats["select_mean"]) / 3.0
        cross_std = (t_stats["select_std"] + vt_stats["select_std"] + at_stats["select_std"]) / 3.0
        cross_spread = (t_stats["select_spread"] + vt_stats["select_spread"] + at_stats["select_spread"]) / 3.0
        stats = {
            "intra_select_mean": intra_stats["select_mean"],
            "intra_select_std": intra_stats["select_std"],
            "intra_select_spread": intra_stats["select_spread"],
            "cross_select_mean": cross_mean,
            "cross_select_std": cross_std,
            "cross_select_spread": cross_spread,
            "intra_cross_gate_mean": gate.detach().float().mean(),
        }
        return out, stats


class SharedSelectiveStateMixer(nn.Module):
    """
    Official-Mamba shared branch replacement for HypergraphEncoder.

    Input:  shared_seq [B, 3, T, node_dim]
    Output: refined shared token [B, hidden_dim]
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        state_dim: int = 16,
        conv_kernel: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_cross_attention: bool = True,
        require_official_mamba: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.modality_embed = nn.Parameter(torch.zeros(1, 3, 1, hidden_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.layers = nn.ModuleList([
            SharedSelectiveStateMixerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                expand=expand,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
                require_official_mamba=require_official_mamba,
            )
            for _ in range(num_layers)
        ])
        self.text_pool = AttentionPool(hidden_dim, hidden_dim=max(64, hidden_dim), dropout=dropout)
        self.modality_readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.token_readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.intra_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _drop_mask(length: int, drop_rate: float, device, dtype, deterministic: bool) -> torch.Tensor:
        if drop_rate <= 0.0:
            return torch.ones(length, device=device, dtype=dtype)
        if deterministic:
            interval = max(2, int(round(1.0 / max(drop_rate, 1e-6))))
            keep = ((torch.arange(length, device=device) + 1) % interval != 0).to(dtype=dtype)
        else:
            keep = (torch.rand(length, device=device) > drop_rate).to(dtype=dtype)
        if keep.sum() == 0:
            keep[0] = 1.0
        return keep

    def _apply_token_drop(self, x: torch.Tensor, drop_rate: float, deterministic: bool) -> torch.Tensor:
        if drop_rate <= 0.0:
            return x
        _, num_modalities, seq_len, _ = x.shape
        mask = self._drop_mask(num_modalities * seq_len, drop_rate, x.device, x.dtype, deterministic)
        mask = mask.view(num_modalities, seq_len).unsqueeze(0).unsqueeze(-1)
        return x * mask

    def forward(
        self,
        shared_seq: torch.Tensor,
        drop_rate: float = 0.0,
        deterministic_drop: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_modalities, seq_len, _ = shared_seq.shape
        if num_modalities != 3:
            raise ValueError("SharedSelectiveStateMixer expects [B, 3, T, D] shared features.")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")

        x = self.input_proj(shared_seq)
        x = x + self.modality_embed + self.time_embed[:, :, :seq_len]
        x = self._apply_token_drop(x, drop_rate=drop_rate, deterministic=deterministic_drop)

        cross_means: List[torch.Tensor] = []
        intra_means: List[torch.Tensor] = []
        cross_stds: List[torch.Tensor] = []
        intra_stds: List[torch.Tensor] = []
        select_stds: List[torch.Tensor] = []
        gaps: List[torch.Tensor] = []
        spreads: List[torch.Tensor] = []
        layer_gates: List[torch.Tensor] = []

        text_context, _ = self.text_pool(x[:, 0])
        for layer in self.layers:
            x, layer_stats = layer(x, text_context=text_context)
            text_context, _ = self.text_pool(x[:, 0])

            cross_mean = layer_stats["cross_select_mean"]
            intra_mean = layer_stats["intra_select_mean"]
            cross_std = layer_stats["cross_select_std"]
            intra_std = layer_stats["intra_select_std"]
            select_std = 0.5 * (cross_std + intra_std)
            gap = cross_mean - intra_mean
            spread = 0.5 * (layer_stats["cross_select_spread"] + layer_stats["intra_select_spread"])

            cross_means.append(cross_mean)
            intra_means.append(intra_mean)
            cross_stds.append(cross_std)
            intra_stds.append(intra_std)
            select_stds.append(select_std)
            gaps.append(gap)
            spreads.append(spread)
            layer_gates.append(layer_stats["intra_cross_gate_mean"])

        modality_repr = x.mean(dim=2)
        modality_score = self.modality_readout(modality_repr).squeeze(-1)
        modality_attn = torch.softmax(modality_score, dim=-1)
        intra_summary = torch.sum(modality_attn.unsqueeze(-1) * modality_repr, dim=1)

        cross_tokens = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len * 3, self.hidden_dim)
        cross_token_attn = torch.softmax(self.token_readout(cross_tokens).squeeze(-1), dim=1)
        cross_summary = torch.sum(cross_token_attn.unsqueeze(-1) * cross_tokens, dim=1)

        intra_summary = self.intra_proj(intra_summary)
        cross_summary = self.cross_proj(cross_summary)
        gate = self.fuse_gate(torch.cat([intra_summary, cross_summary], dim=-1))
        shared_repr = self.out_norm(gate * cross_summary + (1.0 - gate) * intra_summary)
        shared_repr = self.dropout(shared_repr)

        per_layer_cross_means = torch.stack(cross_means)
        per_layer_intra_means = torch.stack(intra_means)
        per_layer_cross_stds = torch.stack(cross_stds)
        per_layer_intra_stds = torch.stack(intra_stds)
        per_layer_select_stds = torch.stack(select_stds)
        per_layer_gaps = torch.stack(gaps)
        per_layer_spreads = torch.stack(spreads)
        per_layer_gates = torch.stack(layer_gates)

        aux = {
            "refined_sequence": x,
            "cross_select_mean": per_layer_cross_means.mean(),
            "intra_select_mean": per_layer_intra_means.mean(),
            "cross_select_std": per_layer_cross_stds.mean(),
            "intra_select_std": per_layer_intra_stds.mean(),
            "select_std": per_layer_select_stds.mean(),
            "cross_intra_gap": per_layer_gaps.mean(),
            "select_spread": per_layer_spreads.mean(),
            "per_layer_cross_select_mean": per_layer_cross_means,
            "per_layer_intra_select_mean": per_layer_intra_means,
            "per_layer_cross_select_std": per_layer_cross_stds,
            "per_layer_intra_select_std": per_layer_intra_stds,
            "per_layer_select_std": per_layer_select_stds,
            "per_layer_cross_intra_gap": per_layer_gaps,
            "per_layer_select_spread": per_layer_spreads,
            "per_layer_intra_cross_gate": per_layer_gates,
            "modality_attn": modality_attn,
            "cross_token_attn": cross_token_attn,
            "shared_gate_mean": gate.detach().float().mean(),
            "intra_cross_gate_mean": per_layer_gates.mean(),
        }
        return shared_repr, aux
