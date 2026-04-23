from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperBatchHypergraphBuilder(nn.Module):
    """
    Unified multimodal hypergraph builder on the shared branch.

    Speed-up changes:
    1) keep the same tri-/bi-/intra-modal hyperedge design;
    2) keep soft membership rather than reverting to hard binary H;
    3) replace the previous Python-heavy per-edge writes with batched scatter/index_put.
    """

    CROSS_EDGE_TYPES = ("tva", "tv", "ta", "va")
    PAIRS = ((0, 1), (0, 2), (1, 2))

    def __init__(self, intra_k: int = 3, soft_tau: float = 0.85):
        super().__init__()
        self.intra_k = int(intra_k)
        self.soft_tau = float(max(1e-4, soft_tau))

    @staticmethod
    def _flat_to_global_node(flat_idx: torch.Tensor, modality_idx: int, seq_len: int) -> torch.Tensor:
        sample_idx = torch.div(flat_idx, seq_len, rounding_mode="floor")
        time_idx = flat_idx - sample_idx * seq_len
        return sample_idx * (3 * seq_len) + modality_idx * seq_len + time_idx

    def _soft_membership(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores / self.soft_tau, dim=-1)

    def _build_cross_modal_edges(self, shared_norm: torch.Tensor, H: torch.Tensor) -> None:
        batch_size, _, seq_len, feat_dim = shared_norm.shape
        device = shared_norm.device
        dest_dtype = H.dtype
        bt = batch_size * seq_len

        # [B*T, 3, D]
        nodes_bt = shared_norm.permute(0, 2, 1, 3).reshape(bt, 3, feat_dim)

        bt_idx = torch.arange(bt, device=device)
        sample_idx = torch.div(bt_idx, seq_len, rounding_mode="floor")
        time_idx = bt_idx - sample_idx * seq_len
        base = sample_idx * (3 * seq_len) + time_idx
        node_idx_all = torch.stack(
            [base + 0 * seq_len, base + 1 * seq_len, base + 2 * seq_len],
            dim=1,
        )  # [BT, 3]

        # Tri-modal edges: edge ids [0, BT)
        tri_center = F.normalize(nodes_bt.mean(dim=1), dim=-1)
        tri_scores = torch.sum(nodes_bt * tri_center.unsqueeze(1), dim=-1)
        tri_weights = self._soft_membership(tri_scores).to(dest_dtype)
        tri_edge_ids = bt_idx.unsqueeze(1).expand(-1, 3)
        H.index_put_(
            (node_idx_all.reshape(-1), tri_edge_ids.reshape(-1)),
            tri_weights.reshape(-1),
            accumulate=False,
        )

        # Bi-modal edges: tv, ta, va
        for pair_offset, pair in enumerate(self.PAIRS, start=1):
            pair_nodes = nodes_bt[:, list(pair), :]  # [BT, 2, D]
            pair_center = F.normalize(pair_nodes.mean(dim=1), dim=-1)
            pair_scores = torch.sum(pair_nodes * pair_center.unsqueeze(1), dim=-1)
            pair_weights = self._soft_membership(pair_scores).to(dest_dtype)
            pair_node_idx = node_idx_all[:, list(pair)]
            edge_ids = (pair_offset * bt + bt_idx).unsqueeze(1).expand(-1, 2)
            H.index_put_(
                (pair_node_idx.reshape(-1), edge_ids.reshape(-1)),
                pair_weights.reshape(-1),
                accumulate=False,
            )

    def _build_intra_modal_edges(self, shared_norm: torch.Tensor, H: torch.Tensor, cross_edges: int) -> None:
        batch_size, num_modalities, seq_len, feat_dim = shared_norm.shape
        device = shared_norm.device
        dest_dtype = H.dtype
        bt = batch_size * seq_len
        topk = min(self.intra_k, max(1, bt - 1))
        flat_idx = torch.arange(bt, device=device)

        for m in range(num_modalities):
            feats = shared_norm[:, m, :, :].reshape(bt, feat_dim)
            sim_raw = torch.matmul(feats, feats.t())
            sim_for_topk = sim_raw.clone()
            sim_for_topk.fill_diagonal_(-1e4)
            nn_idx = torch.topk(sim_for_topk, k=topk, dim=-1).indices  # [BT, topk]

            members = torch.cat([flat_idx.unsqueeze(1), nn_idx], dim=1)  # [BT, topk+1]
            scores = sim_raw.gather(dim=1, index=members)
            weights = self._soft_membership(scores).to(dest_dtype)

            global_member_idx = self._flat_to_global_node(members, modality_idx=m, seq_len=seq_len)
            edge_ids = (cross_edges + m * bt + flat_idx).unsqueeze(1).expand_as(global_member_idx)
            H.index_put_(
                (global_member_idx.reshape(-1), edge_ids.reshape(-1)),
                weights.reshape(-1),
                accumulate=False,
            )

    def forward(self, shared_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_modalities, seq_len, _ = shared_seq.shape
        assert num_modalities == 3, "Only text/vision/audio are supported."
        device = shared_seq.device
        dtype = shared_seq.dtype

        shared_norm = F.normalize(shared_seq, dim=-1)
        bt = batch_size * seq_len
        num_nodes = batch_size * num_modalities * seq_len
        num_cross_types = len(self.CROSS_EDGE_TYPES)
        cross_edges = num_cross_types * bt
        intra_edges = num_modalities * bt
        num_edges = cross_edges + intra_edges

        H = torch.zeros(num_nodes, num_edges, device=device, dtype=dtype)
        self._build_cross_modal_edges(shared_norm, H)
        self._build_intra_modal_edges(shared_norm, H, cross_edges=cross_edges)

        aux = {
            "num_nodes": torch.tensor(float(num_nodes), device=device),
            "num_edges": torch.tensor(float(num_edges), device=device),
            "cross_edges": torch.tensor(float(cross_edges), device=device),
            "intra_edges": torch.tensor(float(intra_edges), device=device),
            "num_cross_types": torch.tensor(float(num_cross_types), device=device),
            "batch_size": torch.tensor(float(batch_size), device=device),
            "seq_len": torch.tensor(float(seq_len), device=device),
            "soft_tau": torch.tensor(float(self.soft_tau), device=device),
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
        de = H.sum(dim=0).clamp_min(1e-6)
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
        soft_tau: float = 0.85,
        initial_residual_alpha: float = 0.25,
    ):
        super().__init__()
        self.num_cross_types = 4
        self.initial_residual_alpha = float(initial_residual_alpha)
        self.builder = PaperBatchHypergraphBuilder(intra_k=intra_k, soft_tau=soft_tau)
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
        x0 = self.input_proj(shared_seq.reshape(batch_size * 3 * seq_len, -1))
        x = x0

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
            layer_out, edge_w, edge_repr = layer(x, H_used, batch_size=batch_size, seq_len=seq_len)
            x = self.initial_residual_alpha * x0 + (1.0 - self.initial_residual_alpha) * layer_out
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
            "x0": x0,
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
