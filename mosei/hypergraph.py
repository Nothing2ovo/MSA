from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperBatchHypergraphBuilder(nn.Module):
    """
    严格贴近论文 3.3.1 的超图构造思路：
    1) 只对 modality-irrelevant(shared) 序列建图；
    2) 节点保留为序列级节点，而不是先池化成“每模态一个节点”；
    3) cross-modal hyperedge: 同一样本、同一时间位置上的 t/v/a 三个节点；
    4) intra-modal hyperedge: 当前 batch 内，同一模态节点与其 top-k 近邻节点构成超边。

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
        # shared_seq: [B, 3, T, D]
        batch_size, num_modalities, seq_len, feat_dim = shared_seq.shape
        assert num_modalities == 3, 'Only text/vision/audio are supported.'
        device = shared_seq.device
        dtype = shared_seq.dtype

        num_nodes = batch_size * num_modalities * seq_len
        cross_edges = batch_size * seq_len
        intra_edges = batch_size * num_modalities * seq_len
        num_edges = cross_edges + intra_edges

        H = torch.zeros(num_nodes, num_edges, device=device, dtype=dtype)

        # ---------- cross-modal hyperedges ----------
        e = 0
        for b in range(batch_size):
            for t in range(seq_len):
                for m in range(num_modalities):
                    H[self._node_index(b, m, t, seq_len), e] = 1.0
                e += 1

        # ---------- intra-modal hyperedges ----------
        # 论文写法是：当前 batch 内，同一模态不同 utterance 的节点建立超边。
        # 这里按 shared 特征的余弦相似度，为每个节点寻找同模态 top-k 近邻。
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


class PaperHypergraphConv(nn.Module):
    """
    贴近论文 3.3.2 / Eq.(10)(11)：
        N^(l+1) = rho(Dn^{-1/2} H W De^{-1} H^T Dn^{-1/2} N^(l) Theta)

    这里 W 采用“可学习对角权重向量”的实现，不再使用内容驱动的 edge MLP。
    为了适配 batch 内动态图，使用固定最大 batch / seq 长度的可学习 edge slots，
    前向时按当前 B,T 截取对应长度。
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

        # learnable diagonal W: one weight per edge slot
        self.cross_edge_logits = nn.Parameter(torch.zeros(self.max_batch_size, self.max_seq_len))
        self.intra_edge_logits = nn.Parameter(torch.zeros(3, self.max_batch_size, self.max_seq_len))
        self.out_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def _slice_edge_weights(self, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if batch_size > self.max_batch_size or seq_len > self.max_seq_len:
            raise ValueError(
                f'Current batch/seq ({batch_size}, {seq_len}) exceeds configured max '
                f'({self.max_batch_size}, {self.max_seq_len}).'
            )
        cross_w = F.softplus(self.cross_edge_logits[:batch_size, :seq_len].reshape(-1)) + 1e-6
        intra_w = F.softplus(self.intra_edge_logits[:, :batch_size, :seq_len].reshape(-1)) + 1e-6
        edge_w = torch.cat([cross_w, intra_w], dim=0)
        return edge_w.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [N, D], H: [N, E]
        edge_w = self._slice_edge_weights(batch_size, seq_len, x.device, x.dtype)
        Ht = H.transpose(0, 1)  # [E, N]
        de = H.sum(dim=0).clamp_min(1.0)  # [E]
        dv = torch.matmul(H, edge_w.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)  # [N]
        dv_inv_sqrt = torch.rsqrt(dv)
        de_inv = 1.0 / de

        x_theta = self.theta(x)
        x_norm = dv_inv_sqrt.unsqueeze(-1) * x_theta
        edge_msg = torch.matmul(Ht, x_norm)
        edge_msg = de_inv.unsqueeze(-1) * edge_w.unsqueeze(-1) * edge_msg
        node_msg = torch.matmul(H, edge_msg)
        node_msg = dv_inv_sqrt.unsqueeze(-1) * node_msg

        out = self.out_norm(self.dropout(F.gelu(node_msg)))
        return out, edge_w, edge_msg


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
        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, shared_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shared_seq: [B, 3, T, D]
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
        modality_repr = node_out.mean(dim=2)  # [B, 3, H]
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
            **graph_meta,
        }
        return graph_repr, aux
