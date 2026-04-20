from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperBatchHypergraphBuilder(nn.Module):
    """
    Paper-aligned sequence-level hypergraph construction.
    Input: shared_seq [B, 3, T, D]
    Output: incidence matrix H [N, E], where:
      N = B * 3 * T
      E = B * T + 3 * B * T
    """
    def __init__(self, intra_k: int = 3):
        super().__init__()
        self.intra_k = intra_k

    @staticmethod
    def _node_index(sample_idx: int, modality_idx: int, time_idx: int, seq_len: int) -> int:
        return sample_idx * 3 * seq_len + modality_idx * seq_len + time_idx

    @staticmethod
    def _flat_to_bt(flat_idx: int, seq_len: int) -> Tuple[int, int]:
        return flat_idx // seq_len, flat_idx % seq_len

    def forward(self, shared_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_modalities, seq_len, feat_dim = shared_seq.shape
        assert num_modalities == 3, "Only text/vision/audio are supported."
        device = shared_seq.device
        dtype = shared_seq.dtype

        num_nodes = batch_size * num_modalities * seq_len
        cross_edges = batch_size * seq_len
        intra_edges = batch_size * num_modalities * seq_len
        num_edges = cross_edges + intra_edges

        H = torch.zeros(num_nodes, num_edges, device=device, dtype=dtype)

        # cross-modal hyperedges: three modality nodes at the same sample/time step
        e = 0
        for b in range(batch_size):
            for t in range(seq_len):
                for m in range(num_modalities):
                    H[self._node_index(b, m, t, seq_len), e] = 1.0
                e += 1

        # intra-modal hyperedges: current node + top-k neighbors in the same modality inside the batch
        for m in range(num_modalities):
            feats = shared_seq[:, m, :, :].reshape(batch_size * seq_len, feat_dim)
            feats = F.normalize(feats, dim=-1)
            sim = torch.matmul(feats, feats.t())
            sim.fill_diagonal_(-1e4)
            topk = min(self.intra_k, max(1, batch_size * seq_len - 1))
            nn_idx = torch.topk(sim, k=topk, dim=-1).indices

            for p in range(batch_size * seq_len):
                edge_id = cross_edges + m * (batch_size * seq_len) + p
                center_b, center_t = self._flat_to_bt(p, seq_len)
                H[self._node_index(center_b, m, center_t, seq_len), edge_id] = 1.0
                for q in nn_idx[p].tolist():
                    nb_b, nb_t = self._flat_to_bt(q, seq_len)
                    H[self._node_index(nb_b, m, nb_t, seq_len), edge_id] = 1.0

        aux = {
            "num_nodes": torch.tensor(float(num_nodes), device=device),
            "num_edges": torch.tensor(float(num_edges), device=device),
            "cross_edges": torch.tensor(float(cross_edges), device=device),
            "intra_edges": torch.tensor(float(intra_edges), device=device),
            "batch_size": torch.tensor(float(batch_size), device=device),
            "seq_len": torch.tensor(float(seq_len), device=device),
        }
        return H, aux


class PaperHypergraphConv(nn.Module):
    """
    Paper-aligned normalized hypergraph aggregation (Eq. 11):
        N^{l+1} = rho(D_n^{-1/2} H W D_e^{-1} H^T D_n^{-1/2} N^{l} theta)

    W is implemented as a learnable diagonal vector over hyperedges.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_batch_size: int = 32,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.max_edges = 4 * self.max_batch_size * self.max_seq_len

        self.theta = nn.Linear(input_dim, output_dim)
        self.edge_logits = nn.Parameter(torch.zeros(self.max_edges))
        nn.init.normal_(self.edge_logits, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def _slice_edge_weights(self, num_edges: int, device, dtype) -> torch.Tensor:
        if num_edges > self.max_edges:
            raise ValueError(
                f"Current number of edges {num_edges} exceeds configured maximum {self.max_edges}."
            )
        logits = self.edge_logits[:num_edges].to(device=device, dtype=dtype)
        return F.softplus(logits).clamp_min(1e-6)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_edges = H.size(1)
        edge_w = self._slice_edge_weights(num_edges, x.device, x.dtype)

        de = H.sum(dim=0).clamp_min(1.0)
        dv = torch.matmul(H, edge_w.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        de_inv = 1.0 / de
        dv_inv_sqrt = torch.rsqrt(dv)

        x_theta = self.theta(x)
        x_norm = dv_inv_sqrt.unsqueeze(-1) * x_theta
        edge_repr = torch.matmul(H.transpose(0, 1), x_norm)
        edge_msg = de_inv.unsqueeze(-1) * edge_w.unsqueeze(-1) * edge_repr
        node_msg = torch.matmul(H, edge_msg)
        out = F.gelu(dv_inv_sqrt.unsqueeze(-1) * node_msg)
        out = self.dropout(out)
        return out, edge_w, edge_repr


class HypergraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        intra_k: int = 3,
        dropout: float = 0.1,
        max_batch_size: int = 32,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.builder = PaperBatchHypergraphBuilder(intra_k=intra_k)
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            PaperHypergraphConv(
                hidden_dim,
                hidden_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    @staticmethod
    def _compute_edge_stats(edge_w: torch.Tensor, num_cross: int) -> Dict[str, torch.Tensor]:
        cross_w = edge_w[:num_cross]
        intra_w = edge_w[num_cross:]
        zero = torch.zeros((), device=edge_w.device, dtype=edge_w.dtype)
        cross_mean = cross_w.mean() if cross_w.numel() > 0 else zero
        intra_mean = intra_w.mean() if intra_w.numel() > 0 else zero
        cross_std = cross_w.std(unbiased=False) if cross_w.numel() > 1 else zero
        intra_std = intra_w.std(unbiased=False) if intra_w.numel() > 1 else zero
        edge_std = edge_w.std(unbiased=False) if edge_w.numel() > 1 else zero
        cross_intra_gap = torch.abs(cross_mean - intra_mean)
        if edge_w.numel() >= 4:
            k = max(1, edge_w.numel() // 10)
            top_mean = torch.topk(edge_w, k=k).values.mean()
            bottom_mean = torch.topk(edge_w, k=k, largest=False).values.mean()
            edge_spread = top_mean - bottom_mean
        else:
            edge_spread = zero
        return {
            "cross_edge_weight_mean": cross_mean,
            "intra_edge_weight_mean": intra_mean,
            "cross_edge_weight_std": cross_std,
            "intra_edge_weight_std": intra_std,
            "edge_weight_std": edge_std,
            "cross_intra_gap": cross_intra_gap,
            "edge_spread": edge_spread,
        }

    def forward(self, shared_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, seq_len, _ = shared_seq.shape
        H, graph_meta = self.builder(shared_seq)
        x = self.input_proj(shared_seq.reshape(batch_size * 3 * seq_len, -1))

        edge_weights_per_layer: List[torch.Tensor] = []
        edge_repr_per_layer: List[torch.Tensor] = []
        layer_stats: List[Dict[str, torch.Tensor]] = []

        num_cross = batch_size * seq_len
        for layer in self.layers:
            x, edge_w, edge_repr = layer(x, H)
            edge_weights_per_layer.append(edge_w)
            edge_repr_per_layer.append(edge_repr)
            layer_stats.append(self._compute_edge_stats(edge_w, num_cross=num_cross))

        node_out = x.view(batch_size, 3, seq_len, -1)
        modality_repr = node_out.mean(dim=2)  # [B, 3, hidden]

        aux = {
            "incidence_matrix": H,
            "node_output": node_out,
            "modality_output": modality_repr,
            "edge_weights_per_layer": edge_weights_per_layer,
            "edge_repr_per_layer": edge_repr_per_layer,
            "cross_edge_weight_mean": layer_stats[-1]["cross_edge_weight_mean"],
            "intra_edge_weight_mean": layer_stats[-1]["intra_edge_weight_mean"],
            "cross_edge_weight_std": layer_stats[-1]["cross_edge_weight_std"],
            "intra_edge_weight_std": layer_stats[-1]["intra_edge_weight_std"],
            "edge_weight_std": layer_stats[-1]["edge_weight_std"],
            "cross_intra_gap": layer_stats[-1]["cross_intra_gap"],
            "edge_spread": layer_stats[-1]["edge_spread"],
            **graph_meta,
        }
        return modality_repr, aux
