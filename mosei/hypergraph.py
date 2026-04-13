
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperBatchHypergraphBuilder(nn.Module):
    """
    论文对齐的序列级超图构造：
    1) 只对 shared(modality-irrelevant) 序列建图；
    2) cross-modal hyperedge: 同一样本、同一时间位置上的 t/v/a 三个节点；
    3) intra-modal hyperedge: 当前 batch 内，同一模态节点与其 top-k 近邻节点构成超边。

    输入: shared_seq [B, 3, T, D]
    输出: incidence matrix H [N, E]
    其中 N = B * 3 * T, E = B * T + 3 * B * T
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
        assert num_modalities == 3, 'Only text/vision/audio are supported.'
        device = shared_seq.device
        dtype = shared_seq.dtype

        num_nodes = batch_size * num_modalities * seq_len
        cross_edges = batch_size * seq_len
        intra_edges = batch_size * num_modalities * seq_len
        num_edges = cross_edges + intra_edges

        H = torch.zeros(num_nodes, num_edges, device=device, dtype=dtype)

        # cross-modal hyperedges
        e = 0
        for b in range(batch_size):
            for t in range(seq_len):
                for m in range(num_modalities):
                    H[self._node_index(b, m, t, seq_len), e] = 1.0
                e += 1

        # intra-modal hyperedges: batch 内同模态 top-k 近邻
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
            'num_nodes': torch.tensor(float(num_nodes), device=device),
            'num_edges': torch.tensor(float(num_edges), device=device),
            'cross_edges': torch.tensor(float(cross_edges), device=device),
            'intra_edges': torch.tensor(float(intra_edges), device=device),
            'batch_size': torch.tensor(float(batch_size), device=device),
            'seq_len': torch.tensor(float(seq_len), device=device),
        }
        return H, aux


class AntiCollapseHypergraphConv(nn.Module):
    """
    防塌缩版超图卷积：
    1) 保留论文 Eq.(11) 的标准归一化传播；
    2) 边权 = 可学习 base logit + 内容感知 residual，不再只靠纯 slot 参数；
    3) cross / intra 分开建权，打破两类边天然收敛到同一常数的趋势。
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
        self.theta = nn.Linear(input_dim, output_dim)

        # slot-based base logits, but with non-zero / asymmetric initialization
        self.cross_edge_logits = nn.Parameter(torch.empty(self.max_batch_size, self.max_seq_len))
        self.intra_edge_logits = nn.Parameter(torch.empty(3, self.max_batch_size, self.max_seq_len))
        nn.init.normal_(self.cross_edge_logits, mean=0.18, std=0.08)
        nn.init.normal_(self.intra_edge_logits, mean=-0.05, std=0.08)

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
        self.cross_res_scale = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.intra_res_scale = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.cross_bias = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.intra_bias = nn.Parameter(torch.tensor(-0.02, dtype=torch.float32))

        self.out_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def _slice_base_logits(self, batch_size: int, seq_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size > self.max_batch_size or seq_len > self.max_seq_len:
            raise ValueError(
                f'Current batch/seq ({batch_size}, {seq_len}) exceeds configured max '
                f'({self.max_batch_size}, {self.max_seq_len}).'
            )
        cross_logits = self.cross_edge_logits[:batch_size, :seq_len].reshape(-1)
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
        Ht = H.transpose(0, 1)  # [E, N]
        de = H.sum(dim=0).clamp_min(1.0)
        num_cross = batch_size * seq_len

        # edge representation from current node features (breaks symmetry)
        edge_repr = torch.matmul(Ht, x) / de.unsqueeze(-1)
        cross_repr = edge_repr[:num_cross]
        intra_repr = edge_repr[num_cross:]

        cross_base, intra_base = self._slice_base_logits(batch_size, seq_len, x.device, x.dtype)
        cross_res = self.cross_edge_mlp(cross_repr).squeeze(-1)
        intra_res = self.intra_edge_mlp(intra_repr).squeeze(-1)
        cross_res = self._normalize_score(cross_res)
        intra_res = self._normalize_score(intra_res)

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
            AntiCollapseHypergraphConv(
                hidden_dim,
                hidden_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, shared_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, seq_len, _ = shared_seq.shape
        H, graph_meta = self.builder(shared_seq)
        x = self.input_proj(shared_seq.reshape(batch_size * 3 * seq_len, -1))

        edge_weights_per_layer: List[torch.Tensor] = []
        edge_repr_per_layer: List[torch.Tensor] = []
        for layer in self.layers:
            x, edge_w, edge_repr = layer(x, H, batch_size=batch_size, seq_len=seq_len)
            edge_weights_per_layer.append(edge_w)
            edge_repr_per_layer.append(edge_repr)

        node_out = x.view(batch_size, 3, seq_len, -1)
        modality_repr = node_out.mean(dim=2)
        score = self.node_readout(modality_repr).squeeze(-1)
        node_attn = torch.softmax(score, dim=-1)
        graph_repr = torch.sum(node_attn.unsqueeze(-1) * modality_repr, dim=1)
        graph_repr = self.dropout(graph_repr)

        last_edge_w = edge_weights_per_layer[-1]
        num_cross = batch_size * seq_len
        cross_w = last_edge_w[:num_cross]
        intra_w = last_edge_w[num_cross:]
        cross_mean = cross_w.mean()
        intra_mean = intra_w.mean()
        cross_std = cross_w.std(unbiased=False) if cross_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
        intra_std = intra_w.std(unbiased=False) if intra_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
        edge_std = last_edge_w.std(unbiased=False) if last_edge_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
        cross_intra_gap = cross_mean - intra_mean
        if last_edge_w.numel() >= 4:
            k = max(1, last_edge_w.numel() // 10)
            top_mean = torch.topk(last_edge_w, k=k).values.mean()
            bottom_mean = torch.topk(last_edge_w, k=k, largest=False).values.mean()
            edge_spread = top_mean - bottom_mean
        else:
            edge_spread = torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)

        aux = {
            'incidence_matrix': H,
            'node_output': node_out,
            'modality_output': modality_repr,
            'graph_repr': graph_repr,
            'node_attn': node_attn,
            'edge_weights_per_layer': edge_weights_per_layer,
            'edge_repr_per_layer': edge_repr_per_layer,
            'cross_edge_weight_mean': cross_mean,
            'intra_edge_weight_mean': intra_mean,
            'cross_edge_weight_std': cross_std,
            'intra_edge_weight_std': intra_std,
            'edge_weight_std': edge_std,
            'cross_intra_gap': cross_intra_gap,
            'edge_spread': edge_spread,
            **graph_meta,
        }
        return graph_repr, aux
