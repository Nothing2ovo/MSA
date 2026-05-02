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
    """ISM-style global-local context extraction without residual passthrough."""

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
        return self.dropout(self.norm(global_ctx + local_ctx))


class OfficialMambaBlock(nn.Module):
    """Pre-norm wrapper around the official mamba_ssm.Mamba operator."""

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
        out = self.out_norm(self.dropout(y))

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
        out = self.out_norm(y)
        stats = {
            "select_mean": 0.5 * (stats_f["select_mean"] + stats_b["select_mean"]),
            "select_std": 0.5 * (stats_f["select_std"] + stats_b["select_std"]),
            "select_spread": 0.5 * (stats_f["select_spread"] + stats_b["select_spread"]),
        }
        return out, stats


class SharedSelectiveStateMixerLayer(nn.Module):
    """
    One intra-modal shared branch layer.

    Each modality's shared sequence is scanned independently with bidirectional
    Mamba. Cross-modal interaction is intentionally deferred to the downstream
    token-level fusion stage.
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
        self.intra_glce = GlobalLocalContextExtractor(hidden_dim, dropout=dropout)
        self.intra_mamba = BiDirectionalMamba(
            hidden_dim, state_dim=state_dim, conv_kernel=conv_kernel, expand=expand,
            dropout=dropout, require_official_mamba=require_official_mamba,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_modalities, seq_len, hidden_dim = x.shape

        # Intra-modal shared scan.
        intra_in = x.reshape(batch_size * num_modalities, seq_len, hidden_dim)
        intra_in = self.intra_glce(intra_in)
        intra_out, intra_stats = self.intra_mamba(intra_in)
        intra_out = intra_out.view(batch_size, num_modalities, seq_len, hidden_dim)
        out = self.layer_norm(self.dropout(intra_out))

        zero = intra_stats["select_mean"].new_zeros(())
        stats = {
            "intra_select_mean": intra_stats["select_mean"],
            "intra_select_std": intra_stats["select_std"],
            "intra_select_spread": intra_stats["select_spread"],
            "cross_select_mean": zero,
            "cross_select_std": zero,
            "cross_select_spread": zero,
            "intra_cross_gate_mean": zero,
        }
        return out, stats


class SharedSelectiveStateMixer(nn.Module):
    """
    Official-Mamba shared branch replacement for HypergraphEncoder.

    Input:  shared_seq [B, 3, T, node_dim]
    Output: refined per-modality shared tokens [B, 3, hidden_dim]
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
        self.modality_readout = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.token_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.summary_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
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

        intra_means: List[torch.Tensor] = []
        intra_stds: List[torch.Tensor] = []
        select_stds: List[torch.Tensor] = []
        spreads: List[torch.Tensor] = []
        zero_refs: List[torch.Tensor] = []

        for layer in self.layers:
            x, layer_stats = layer(x)
            zero = layer_stats["cross_select_mean"]
            intra_mean = layer_stats["intra_select_mean"]
            intra_std = layer_stats["intra_select_std"]
            select_std = intra_std
            spread = layer_stats["intra_select_spread"]

            intra_means.append(intra_mean)
            intra_stds.append(intra_std)
            select_stds.append(select_std)
            spreads.append(spread)
            zero_refs.append(zero)

        modality_repr = x.mean(dim=2)
        shared_tokens = self.out_norm(self.token_proj(modality_repr))
        modality_score = self.modality_readout(modality_repr).squeeze(-1)
        modality_attn = torch.softmax(modality_score, dim=-1)
        shared_summary = torch.sum(modality_attn.unsqueeze(-1) * shared_tokens, dim=1)
        shared_summary = self.dropout(self.out_norm(self.summary_proj(shared_summary)))

        per_layer_zeros = torch.stack(zero_refs)
        per_layer_intra_means = torch.stack(intra_means)
        per_layer_intra_stds = torch.stack(intra_stds)
        per_layer_select_stds = torch.stack(select_stds)
        per_layer_spreads = torch.stack(spreads)

        aux = {
            "cross_select_mean": per_layer_zeros.mean(),
            "intra_select_mean": per_layer_intra_means.mean(),
            "cross_select_std": per_layer_zeros.mean(),
            "intra_select_std": per_layer_intra_stds.mean(),
            "select_std": per_layer_select_stds.mean(),
            "cross_intra_gap": per_layer_zeros.mean(),
            "select_spread": per_layer_spreads.mean(),
            "per_layer_cross_select_mean": per_layer_zeros,
            "per_layer_intra_select_mean": per_layer_intra_means,
            "per_layer_cross_select_std": per_layer_zeros,
            "per_layer_intra_select_std": per_layer_intra_stds,
            "per_layer_select_std": per_layer_select_stds,
            "per_layer_cross_intra_gap": per_layer_zeros,
            "per_layer_select_spread": per_layer_spreads,
            "per_layer_intra_cross_gate": per_layer_zeros,
            "modality_attn": modality_attn,
            "shared_summary": shared_summary,
            "shared_tokens": shared_tokens,
            "shared_gate_mean": per_layer_zeros.mean(),
            "intra_cross_gate_mean": per_layer_zeros.mean(),
        }
        return self.dropout(shared_tokens), aux
