from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperBatchHypergraphBuilder(nn.Module):
    """
    Shared-branch unified multimodal hypergraph builder.

    Nodes:
        shared_seq [B, 3, T, D] -> N = B * 3 * T nodes

    Hyperedges:
        1) tri-modal edge   : (t, v, a) at the same timestep
        2) bi-modal edges   : (t, v), (t, a), (v, a) at the same timestep
        3) intra-modal edge : each node + top-k same-modality neighbors in the batch
    """

    CROSS_EDGE_TYPES = ("tva", "tv", "ta", "va")

    def __init__(self, intra_k: int = 3):
        super().__init__()
        self.intra_k = int(intra_k)

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
        num_cross_types = len(self.CROSS_EDGE_TYPES)
        cross_edges = num_cross_types * batch_size * seq_len
        intra_edges = batch_size * num_modalities * seq_len
        num_edges = cross_edges + intra_edges

        H = torch.zeros(num_nodes, num_edges, device=device, dtype=dtype)

        # cross-modal hyperedges: tri-modal + three bi-modal pairs
        edge_id = 0
        for b in range(batch_size):
            for t in range(seq_len):
                # tri-modal: text / vision / audio
                for m in (0, 1, 2):
                    H[self._node_index(b, m, t, seq_len), edge_id] = 1.0
                edge_id += 1

                # bi-modal: tv / ta / va
                for pair in ((0, 1), (0, 2), (1, 2)):
                    for m in pair:
                        H[self._node_index(b, m, t, seq_len), edge_id] = 1.0
                    edge_id += 1

        # intra-modal hyperedges: batch-wise top-k same-modality neighbors
        for m in range(num_modalities):
            feats = shared_seq[:, m, :, :].reshape(batch_size * seq_len, feat_dim)
            feats = F.normalize(feats, dim=-1)
            sim = torch.matmul(feats, feats.t())
            sim.fill_diagonal_(-1e4)
            topk = min(self.intra_k, max(1, batch_size * seq_len - 1))
            nn_idx = torch.topk(sim, k=topk, dim=-1).indices

            for p in range(batch_size * seq_len):
                eid = cross_edges + m * (batch_size * seq_len) + p
                center_b, center_t = self._flat_to_bt(p, seq_len)
                H[self._node_index(center_b, m, center_t, seq_len), eid] = 1.0
                for q in nn_idx[p].tolist():
                    nb_b, nb_t = self._flat_to_bt(q, seq_len)
                    H[self._node_index(nb_b, m, nb_t, seq_len), eid] = 1.0

        aux = {
            "num_nodes": torch.tensor(float(num_nodes), device=device),
            "num_edges": torch.tensor(float(num_edges), device=device),
            "cross_edges": torch.tensor(float(cross_edges), device=device),
            "intra_edges": torch.tensor(float(intra_edges), device=device),
            "num_cross_types": torch.tensor(float(num_cross_types), device=device),
            "batch_size": torch.tensor(float(batch_size), device=device),
            "seq_len": torch.tensor(float(seq_len), device=device),
        }
        return H, aux


class AntiCollapseHypergraphConv(nn.Module):
    """
    Node -> hyperedge -> node propagation with separate trainable priors for
    cross-modal and intra-modal hyperedges.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_batch_size: int = 32,
        max_seq_len: int = 64,
        num_cross_types: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.num_cross_types = int(num_cross_types)
        self.theta = nn.Linear(input_dim, output_dim)

        self.cross_edge_logits = nn.Parameter(
            torch.empty(self.num_cross_types, self.max_batch_size, self.max_seq_len)
        )
        self.intra_edge_logits = nn.Parameter(torch.empty(3, self.max_batch_size, self.max_seq_len))
        nn.init.normal_(self.cross_edge_logits, mean=0.12, std=0.08)
        nn.init.normal_(self.intra_edge_logits, mean=-0.03, std=0.08)

        self.cross_edge_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )
        self.intra_edge_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )
        self.cross_res_scale = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.intra_res_scale = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.cross_bias = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))
        self.intra_bias = nn.Parameter(torch.tensor(-0.01, dtype=torch.float32))

        self.out_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def _slice_base_logits(self, batch_size: int, seq_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size > self.max_batch_size or seq_len > self.max_seq_len:
            raise ValueError(
                f"Current batch/seq ({batch_size}, {seq_len}) exceeds configured max "
                f"({self.max_batch_size}, {self.max_seq_len})."
            )
        cross_logits = self.cross_edge_logits[:, :batch_size, :seq_len].reshape(-1)
        intra_logits = self.intra_edge_logits[:, :batch_size, :seq_len].reshape(-1)
        return (
            cross_logits.to(device=device, dtype=dtype),
            intra_logits.to(device=device, dtype=dtype),
        )

    @staticmethod
    def _normalize_score(x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= 1:
            return torch.zeros_like(x)
        x = x - x.mean()
        return x / x.std(unbiased=False).clamp_min(1e-4)

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ht = H.transpose(0, 1)
        de = H.sum(dim=0).clamp_min(1.0)
        num_cross = self.num_cross_types * batch_size * seq_len

        edge_repr = torch.matmul(Ht, x) / de.unsqueeze(-1)
        cross_repr = edge_repr[:num_cross]
        intra_repr = edge_repr[num_cross:]

        cross_base, intra_base = self._slice_base_logits(batch_size, seq_len, x.device, x.dtype)
        cross_res = self._normalize_score(self.cross_edge_mlp(cross_repr).squeeze(-1))
        intra_res = self._normalize_score(self.intra_edge_mlp(intra_repr).squeeze(-1))

        cross_logits = cross_base + self.cross_bias + self.cross_res_scale * cross_res
        intra_logits = intra_base + self.intra_bias + self.intra_res_scale * intra_res
        edge_logits = torch.cat([cross_logits, intra_logits], dim=0)
        edge_w = F.softplus(edge_logits).clamp_min(1e-6)

        dv = torch.matmul(H, edge_w.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        dv_inv_sqrt = torch.rsqrt(dv)
        de_inv = 1.0 / de

        x_theta = self.theta(x)
        x_norm = dv_inv_sqrt.unsqueeze(-1) * x_theta
        edge_msg = torch.matmul(Ht, x_norm)
        edge_msg = de_inv.unsqueeze(-1) * edge_w.unsqueeze(-1) * edge_msg
        node_msg = torch.matmul(H, edge_msg)
        node_msg = dv_inv_sqrt.unsqueeze(-1) * node_msg

        out = self.out_norm(x_theta + self.dropout(F.gelu(node_msg)))
        return out, edge_w, edge_repr


class HypergraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        intra_k: int = 3,
        dropout: float = 0.1,
        max_batch_size: int = 32,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.num_cross_types = 4
        self.builder = PaperBatchHypergraphBuilder(intra_k=intra_k)
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            AntiCollapseHypergraphConv(
                hidden_dim,
                hidden_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                num_cross_types=self.num_cross_types,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.modality_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_token_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_token_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

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
        cross_intra_gap = cross_mean - intra_mean
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

    @staticmethod
    def _drop_hyperedges(
        H: torch.Tensor,
        drop_rate: float,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if drop_rate <= 0.0:
            mask = torch.ones(H.size(1), device=H.device, dtype=H.dtype)
            return H, mask

        num_edges = H.size(1)
        if deterministic:
            interval = max(2, int(round(1.0 / max(drop_rate, 1e-6))))
            keep_mask = ((torch.arange(num_edges, device=H.device) + 1) % interval != 0)
        else:
            keep_mask = torch.rand(num_edges, device=H.device) > drop_rate

        if keep_mask.sum() == 0:
            keep_mask[0] = True
        mask = keep_mask.to(dtype=H.dtype)
        return H * mask.unsqueeze(0), mask

    def forward(
        self,
        shared_seq: torch.Tensor,
        edge_drop_rate: float = 0.0,
        deterministic_drop: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, seq_len, _ = shared_seq.shape
        H, graph_meta = self.builder(shared_seq)
        H_used, edge_keep_mask = self._drop_hyperedges(H, edge_drop_rate, deterministic=deterministic_drop)
        x = self.input_proj(shared_seq.reshape(batch_size * 3 * seq_len, -1))

        edge_weights_per_layer: List[torch.Tensor] = []
        edge_repr_per_layer: List[torch.Tensor] = []
        layer_cross_means: List[torch.Tensor] = []
        layer_intra_means: List[torch.Tensor] = []
        layer_cross_stds: List[torch.Tensor] = []
        layer_intra_stds: List[torch.Tensor] = []
        layer_edge_stds: List[torch.Tensor] = []
        layer_gaps: List[torch.Tensor] = []
        layer_spreads: List[torch.Tensor] = []

        num_cross = self.num_cross_types * batch_size * seq_len
        for layer in self.layers:
            x, edge_w, edge_repr = layer(x, H_used, batch_size=batch_size, seq_len=seq_len)
            edge_weights_per_layer.append(edge_w)
            edge_repr_per_layer.append(edge_repr)
            stats = self._compute_edge_stats(edge_w, num_cross)
            layer_cross_means.append(stats["cross_edge_weight_mean"])
            layer_intra_means.append(stats["intra_edge_weight_mean"])
            layer_cross_stds.append(stats["cross_edge_weight_std"])
            layer_intra_stds.append(stats["intra_edge_weight_std"])
            layer_edge_stds.append(stats["edge_weight_std"])
            layer_gaps.append(stats["cross_intra_gap"])
            layer_spreads.append(stats["edge_spread"])

        node_out = x.view(batch_size, 3, seq_len, -1)
        modality_repr = node_out.mean(dim=2)
        modality_score = self.modality_readout(modality_repr).squeeze(-1)
        modality_attn = torch.softmax(modality_score, dim=-1)

        node_tokens = node_out.reshape(batch_size, 3 * seq_len, -1)
        node_token_attn = torch.softmax(self.node_token_readout(node_tokens).squeeze(-1), dim=1)
        node_summary = torch.sum(node_token_attn.unsqueeze(-1) * node_tokens, dim=1)

        last_edge_repr = edge_repr_per_layer[-1]
        cross_repr = last_edge_repr[:num_cross].reshape(batch_size, self.num_cross_types * seq_len, -1)
        intra_repr = last_edge_repr[num_cross:].reshape(3, batch_size, seq_len, -1).permute(1, 0, 2, 3)
        intra_repr = intra_repr.reshape(batch_size, 3 * seq_len, -1)
        edge_tokens = torch.cat([cross_repr, intra_repr], dim=1)
        edge_attn = torch.softmax(self.edge_token_readout(edge_tokens).squeeze(-1), dim=1)
        edge_summary = torch.sum(edge_attn.unsqueeze(-1) * edge_tokens, dim=1)

        node_summary = self.node_proj(node_summary)
        edge_summary = self.edge_proj(edge_summary)
        gate = self.fuse_gate(torch.cat([node_summary, edge_summary], dim=-1))
        graph_repr = self.out_norm(gate * node_summary + (1.0 - gate) * edge_summary)
        graph_repr = self.dropout(graph_repr)

        per_layer_cross_means = torch.stack(layer_cross_means)
        per_layer_intra_means = torch.stack(layer_intra_means)
        per_layer_cross_stds = torch.stack(layer_cross_stds)
        per_layer_intra_stds = torch.stack(layer_intra_stds)
        per_layer_edge_stds = torch.stack(layer_edge_stds)
        per_layer_gaps = torch.stack(layer_gaps)
        per_layer_spreads = torch.stack(layer_spreads)

        aux = {
            "incidence_matrix": H_used,
            "raw_incidence_matrix": H,
            "edge_keep_mask": edge_keep_mask,
            "node_output": node_out,
            "modality_output": modality_repr,
            "graph_repr": graph_repr,
            "node_attn": modality_attn,
            "node_token_attn": node_token_attn,
            "edge_attn": edge_attn,
            "node_summary": node_summary,
            "edge_summary": edge_summary,
            "node_edge_gate": gate,
            "edge_weights_per_layer": edge_weights_per_layer,
            "edge_repr_per_layer": edge_repr_per_layer,
            "per_layer_cross_edge_weight_mean": per_layer_cross_means,
            "per_layer_intra_edge_weight_mean": per_layer_intra_means,
            "per_layer_cross_edge_weight_std": per_layer_cross_stds,
            "per_layer_intra_edge_weight_std": per_layer_intra_stds,
            "per_layer_edge_weight_std": per_layer_edge_stds,
            "per_layer_cross_intra_gap": per_layer_gaps,
            "per_layer_edge_spread": per_layer_spreads,
            "cross_edge_weight_mean": per_layer_cross_means[-1],
            "intra_edge_weight_mean": per_layer_intra_means[-1],
            "cross_edge_weight_std": per_layer_cross_stds[-1],
            "intra_edge_weight_std": per_layer_intra_stds[-1],
            "edge_weight_std": per_layer_edge_stds[-1],
            "cross_intra_gap": per_layer_gaps[-1],
            "edge_spread": per_layer_spreads[-1],
            "multi_layer_cross_edge_weight_mean": per_layer_cross_means.mean(),
            "multi_layer_intra_edge_weight_mean": per_layer_intra_means.mean(),
            "multi_layer_edge_weight_std": per_layer_edge_stds.mean(),
            "multi_layer_cross_intra_gap": per_layer_gaps.mean(),
            "multi_layer_edge_spread": per_layer_spreads.mean(),
            **graph_meta,
        }
        return graph_repr, aux
