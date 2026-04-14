import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

DEFAULT_MOE_BALANCE_WEIGHT = 1e-2
DEFAULT_RENYI_ALPHA = 1.9
DEFAULT_RENYI_RANK = 10
DEFAULT_TOKEN_REG_WEIGHT = 2e-2
DEFAULT_HYPERGRAPH_REG_WEIGHT = 8e-2
DEFAULT_TOKEN_TARGET_ENTROPY = 0.80
DEFAULT_PRIVATE_MIN_WEIGHT = 0.08
DEFAULT_SHARED_TARGET_WEIGHT = 0.34
DEFAULT_SHARED_DOMINANCE_MARGIN = 0.02
DEFAULT_TOKEN_MAX_WEIGHT = 0.70
DEFAULT_EDGE_TARGET_STD = 0.03
DEFAULT_EDGE_MIN_GAP = 0.02
DEFAULT_CROSS_EDGE_TARGET_STD = 0.015
DEFAULT_INTRA_EDGE_TARGET_STD = 0.015
DEFAULT_EDGE_SPREAD_MARGIN = 0.05


def compute_mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds.view(-1) - labels.view(-1))).item()


def compute_corr(preds: torch.Tensor, labels: torch.Tensor) -> float:
    x = preds.view(-1)
    y = labels.view(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()) + 1e-8
    return ((x * y).sum() / denom).item()


def compute_acc7(preds: torch.Tensor, labels: torch.Tensor) -> float:
    p = torch.clamp(torch.round(preds.view(-1)), min=-3, max=3)
    y = torch.clamp(torch.round(labels.view(-1)), min=-3, max=3)
    return (p == y).float().mean().item()


def _binary_by_zero(
    preds: torch.Tensor,
    labels: torch.Tensor,
    include_zero_as_positive: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = preds.view(-1)
    labels = labels.view(-1)
    if include_zero_as_positive:
        y = (labels >= 0).long()
        p = (preds >= 0).long()
    else:
        mask = labels != 0
        if mask.sum() == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        y = (labels[mask] > 0).long()
        p = (preds[mask] > 0).long()
    return p, y


def _f1_binary(p: torch.Tensor, y: torch.Tensor) -> float:
    if p.numel() == 0:
        return 0.0
    tp = ((p == 1) & (y == 1)).sum().item()
    fp = ((p == 1) & (y == 0)).sum().item()
    fn = ((p == 0) & (y == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-8)


def compute_acc2_f1(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    p1, y1 = _binary_by_zero(preds, labels, include_zero_as_positive=True)
    p2, y2 = _binary_by_zero(preds, labels, include_zero_as_positive=False)
    acc_nonneg = (p1 == y1).float().mean().item() if p1.numel() > 0 else 0.0
    acc_posneg = (p2 == y2).float().mean().item() if p2.numel() > 0 else 0.0
    return {
        "Acc2_nonneg": acc_nonneg,
        "F1_nonneg": _f1_binary(p1, y1),
        "Acc2_posneg": acc_posneg,
        "F1_posneg": _f1_binary(p2, y2),
    }


def labels_to_classes(labels: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(labels.view(-1)), min=-3, max=3).long() + 3


def _cross_modal_triplet(anchor: torch.Tensor, positive: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    batch_size = anchor.size(0)
    cls = labels_to_classes(labels)
    anchor_n = F.normalize(anchor, dim=-1)
    positive_n = F.normalize(positive, dim=-1)
    pos_sim = F.cosine_similarity(anchor_n, positive_n, dim=-1)

    sim_mat = torch.matmul(anchor_n, anchor_n.t())
    diff_mask = cls.unsqueeze(1) != cls.unsqueeze(0)
    valid_mask = diff_mask & (~torch.eye(batch_size, device=anchor.device, dtype=torch.bool))
    masked = sim_mat.masked_fill(~valid_mask, -1e4)
    hardest_neg = masked.max(dim=1).values
    no_neg = hardest_neg < -1e3
    hardest_neg = torch.where(no_neg, torch.zeros_like(hardest_neg), hardest_neg)
    loss = F.relu(margin + hardest_neg - pos_sim)
    return loss.mean()


def similarity_loss(aux: Dict[str, torch.Tensor], labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    e_irr_t = aux["e_irr_t"].mean(dim=1)
    e_irr_v = aux["e_irr_v"].mean(dim=1)
    e_irr_a = aux["e_irr_a"].mean(dim=1)
    losses = [
        _cross_modal_triplet(e_irr_t, e_irr_v, labels, margin),
        _cross_modal_triplet(e_irr_t, e_irr_a, labels, margin),
        _cross_modal_triplet(e_irr_v, e_irr_t, labels, margin),
        _cross_modal_triplet(e_irr_v, e_irr_a, labels, margin),
        _cross_modal_triplet(e_irr_a, e_irr_t, labels, margin),
        _cross_modal_triplet(e_irr_a, e_irr_v, labels, margin),
    ]
    return sum(losses) / len(losses)


def reconstruction_loss(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        F.mse_loss(aux["rec_t"], aux["c_t"])
        + F.mse_loss(aux["rec_v"], aux["c_v"])
        + F.mse_loss(aux["rec_a"], aux["c_a"])
    ) / 3.0


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    if mean.abs().item() < 1e-8:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.var(unbiased=False) / (mean ** 2 + 1e-8)


def moe_load_loss(aux: Dict[str, torch.Tensor], balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT) -> torch.Tensor:
    losses = []
    for key in ["tmoe_t_aux", "tmoe_v_aux", "tmoe_a_aux"]:
        importance = aux[key]["importance"]
        load = aux[key]["load"]
        losses.append(balance_weight * (cv_squared(importance) + cv_squared(load)))
    return sum(losses) / len(losses)



def hypergraph_structure_loss(
    aux: Dict[str, torch.Tensor],
    target_edge_std: float = DEFAULT_EDGE_TARGET_STD,
    min_cross_intra_gap: float = DEFAULT_EDGE_MIN_GAP,
    target_cross_std: float = DEFAULT_CROSS_EDGE_TARGET_STD,
    target_intra_std: float = DEFAULT_INTRA_EDGE_TARGET_STD,
    min_edge_spread: float = DEFAULT_EDGE_SPREAD_MARGIN,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    防塌缩超图正则：
    1) 整体边权方差不能太小；
    2) cross / intra 两类边不能完全重合；
    3) 强边与弱边之间要有可见间隔。
    """
    hyper_aux = aux["hyper_aux"]
    last_edge_w = hyper_aux["edge_weights_per_layer"][-1]
    num_cross = int(hyper_aux["cross_edges"].item())
    cross_w = last_edge_w[:num_cross]
    intra_w = last_edge_w[num_cross:]

    edge_std = last_edge_w.std(unbiased=False) if last_edge_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
    cross_std = cross_w.std(unbiased=False) if cross_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
    intra_std = intra_w.std(unbiased=False) if intra_w.numel() > 1 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)
    gap = torch.abs(cross_w.mean() - intra_w.mean()) if cross_w.numel() > 0 and intra_w.numel() > 0 else torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)

    if last_edge_w.numel() >= 4:
        k = max(1, last_edge_w.numel() // 10)
        top_mean = torch.topk(last_edge_w, k=k).values.mean()
        bottom_mean = torch.topk(last_edge_w, k=k, largest=False).values.mean()
        spread = top_mean - bottom_mean
    else:
        spread = torch.zeros((), device=last_edge_w.device, dtype=last_edge_w.dtype)

    loss = (
        F.relu(target_edge_std - edge_std)
        + 0.5 * F.relu(target_cross_std - cross_std)
        + 0.5 * F.relu(target_intra_std - intra_std)
        + 1.0 * F.relu(min_cross_intra_gap - gap)
        + 1.0 * F.relu(min_edge_spread - spread)
    )
    stats = {
        "hypergraph_reg_loss": float(loss.item()),
        "edge_weight_std": float(edge_std.item()),
        "cross_edge_weight_std": float(cross_std.item()),
        "intra_edge_weight_std": float(intra_std.item()),
        "cross_intra_gap": float(gap.item()),
        "edge_spread": float(spread.item()),
    }
    return loss, stats


def pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4).contiguous()
    dist = torch.cdist(x, x, p=2)
    return dist.pow(2)


def _estimate_sigma2_from_batch(dist2: torch.Tensor) -> torch.Tensor:
    n = dist2.size(0)
    if n <= 1:
        return torch.tensor(1.0, device=dist2.device, dtype=dist2.dtype)
    dist = torch.sqrt(dist2.clamp_min(1e-12))
    masked = dist + torch.eye(n, device=dist.device, dtype=dist.dtype) * 1e9
    k = min(5, n - 1)
    knn = torch.topk(masked, k=k, largest=False, dim=-1).values
    sigma = torch.nan_to_num(knn.mean(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-4)
    return sigma.pow(2)


def normalized_gaussian_gram(x: torch.Tensor, sigma2: torch.Tensor = None) -> torch.Tensor:
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
    n = x.size(0)
    if n == 1:
        return torch.ones(1, 1, device=x.device, dtype=torch.float32)

    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    dist2 = pairwise_sq_dist(x)
    if sigma2 is None:
        sigma2 = _estimate_sigma2_from_batch(dist2)

    gram = torch.exp(-(dist2 / (sigma2 + 1e-8)).clamp(max=50.0))
    gram = torch.nan_to_num(gram, nan=0.0, posinf=1.0, neginf=0.0)
    gram = 0.5 * (gram + gram.t())
    gram = gram + 1e-6 * torch.eye(n, device=gram.device, dtype=gram.dtype)
    gram = gram / torch.trace(gram).clamp_min(1e-8)
    gram = 0.5 * (gram + gram.t())
    return gram


def low_rank_renyi_entropy(gram: torch.Tensor, alpha: float = DEFAULT_RENYI_ALPHA, rank_k: int = DEFAULT_RENYI_RANK) -> torch.Tensor:
    n = gram.size(0)
    if n <= 1:
        return torch.zeros((), device=gram.device, dtype=gram.dtype)

    gram_safe = torch.nan_to_num(gram.float(), nan=0.0, posinf=1.0, neginf=0.0)
    gram_safe = 0.5 * (gram_safe + gram_safe.t())
    gram_safe = gram_safe + 1e-6 * torch.eye(n, device=gram_safe.device, dtype=gram_safe.dtype)
    gram_safe = gram_safe / torch.trace(gram_safe).clamp_min(1e-8)

    evals = torch.linalg.eigvalsh(gram_safe.to(device="cpu", dtype=torch.float64))
    evals = torch.flip(evals, dims=[0]).clamp_min(1e-12)
    evals = evals.to(device=gram.device, dtype=gram.dtype)

    if rank_k >= n:
        val = torch.sum(evals.pow(alpha))
        return torch.log2(val.clamp_min(1e-12)) / (1.0 - alpha)

    top = evals[:rank_k]
    remain_mass = (1.0 - top.sum()).clamp_min(1e-12)
    lambda_r = remain_mass / max(1, n - rank_k)
    val = top.pow(alpha).sum() + max(1, n - rank_k) * lambda_r.pow(alpha)
    return torch.log2(val.clamp_min(1e-12)) / (1.0 - alpha)


def low_rank_renyi_mutual_information(
    x: torch.Tensor,
    z: torch.Tensor,
    alpha: float = DEFAULT_RENYI_ALPHA,
    rank_k: int = DEFAULT_RENYI_RANK,
) -> torch.Tensor:
    if x.size(0) <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    a = normalized_gaussian_gram(x)
    b = normalized_gaussian_gram(z)
    joint = a * b
    joint = joint / torch.trace(joint).clamp_min(1e-8)
    joint = 0.5 * (joint + joint.t())

    h_a = low_rank_renyi_entropy(a, alpha=alpha, rank_k=rank_k)
    h_b = low_rank_renyi_entropy(b, alpha=alpha, rank_k=rank_k)
    h_joint = low_rank_renyi_entropy(joint, alpha=alpha, rank_k=rank_k)
    mi = h_a + h_b - h_joint
    return mi.clamp_min(0.0)


def transformer_guided_ib_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    alpha: float = DEFAULT_RENYI_ALPHA,
    rank_k: int = DEFAULT_RENYI_RANK,
    mae_weight: float = 1.0,
    kl_weight: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pre_ib = aux["fused_repr"]
    z = aux["filtered_repr"]
    mi_term = low_rank_renyi_mutual_information(pre_ib, z, alpha=alpha, rank_k=rank_k)

    mu = aux["tgib_aux"]["mu"]
    logvar = aux["tgib_aux"]["logvar"]
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    mae = F.l1_loss(preds.view(-1), labels.view(-1))

    loss = mi_term + mae_weight * mae + kl_weight * kl
    stats = {
        "mmib_mi": float(mi_term.item()),
        "mmib_mae": float(mae.item()),
        "mmib_kl": float(kl.item()),
        "mmib_loss": float(loss.item()),
    }
    return loss, stats


def token_regularization_loss(
    token_weights: torch.Tensor,
    target_entropy: float = DEFAULT_TOKEN_TARGET_ENTROPY,
    private_min_weight: float = DEFAULT_PRIVATE_MIN_WEIGHT,
    shared_target_weight: float = DEFAULT_SHARED_TARGET_WEIGHT,
    shared_margin: float = DEFAULT_SHARED_DOMINANCE_MARGIN,
    max_weight: float = DEFAULT_TOKEN_MAX_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    目标从“接近均匀”改成“shared 主导、private 补充”：
    - shared 至少要站住主导位；
    - 3 个 private 不能被压死；
    - 但也不能出现单 token 长期一边倒独占。
    """
    eps = 1e-8
    num_tokens = token_weights.size(1)
    entropy = -(token_weights * torch.log(token_weights.clamp_min(eps))).sum(dim=1)
    entropy = entropy / math.log(num_tokens)
    entropy_penalty = (entropy - target_entropy).pow(2).mean()

    mean_w = token_weights.mean(dim=0)
    target_prior = torch.tensor([0.40, 0.20, 0.20, 0.20], device=token_weights.device, dtype=token_weights.dtype)
    balance = F.mse_loss(mean_w, target_prior)

    shared_w = token_weights[:, 0]
    private_w = token_weights[:, 1:]
    max_private = private_w.max(dim=1).values

    shared_floor_penalty = F.relu(shared_target_weight - shared_w).mean()
    shared_margin_penalty = F.relu(max_private + shared_margin - shared_w).mean()
    private_floor_penalty = F.relu(private_min_weight - private_w).mean()
    peak_penalty = F.relu(token_weights.max(dim=1).values - max_weight).mean()

    loss = (
        entropy_penalty
        + 0.35 * balance
        + 1.25 * shared_floor_penalty
        + 1.50 * shared_margin_penalty
        + 1.00 * private_floor_penalty
        + 1.00 * peak_penalty
    )
    stats = {
        "token_reg_loss": float(loss.item()),
        "token_entropy": float(entropy.mean().item()),
        "token_balance": float(balance.item()),
        "token_max_weight": float(token_weights.max(dim=1).values.mean().item()),
        "token_floor_penalty": float(private_floor_penalty.item()),
        "token_peak_penalty": float(peak_penalty.item()),
        "token_shared_mean": float(shared_w.mean().item()),
        "token_private_max_mean": float(max_private.mean().item()),
        "token_dominance_margin": float((shared_w - max_private).mean().item()),
    }
    return loss, stats


def task_loss_regression(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds.view(-1), labels.view(-1))


def total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    beta: float = 0.05,
    gamma: float = 0.10,
    delta: float = 0.05,
    sim_margin: float = 0.2,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    renyi_alpha: float = DEFAULT_RENYI_ALPHA,
    renyi_rank_k: int = DEFAULT_RENYI_RANK,
    mmib_mae_weight: float = 1.0,
    mmib_kl_weight: float = 1e-4,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    hypergraph_reg_weight: float = DEFAULT_HYPERGRAPH_REG_WEIGHT,
):
    l_task = task_loss_regression(preds, labels)
    l_s = similarity_loss(aux, labels, margin=sim_margin)
    l_r = reconstruction_loss(aux)
    l_m = moe_load_loss(aux, balance_weight=moe_balance_weight)
    l_mmib, mmib_stats = transformer_guided_ib_loss(
        preds,
        labels,
        aux,
        alpha=renyi_alpha,
        rank_k=renyi_rank_k,
        mae_weight=mmib_mae_weight,
        kl_weight=mmib_kl_weight,
    )
    token_weights = aux["token_fusion_aux"]["token_weights"]
    l_token_reg, token_stats = token_regularization_loss(token_weights)
    l_hg, hg_stats = hypergraph_structure_loss(aux)

    total = l_task + alpha * l_s + beta * l_r + gamma * l_m + delta * l_mmib + token_reg_weight * l_token_reg + hypergraph_reg_weight * l_hg

    transformer_attn = aux["tgib_aux"]["transformer_attn"]
    if transformer_attn.dim() == 4:
        transformer_attn = transformer_attn.mean(dim=1)
    hyper_aux = aux["hyper_aux"]
    tgib_aux = aux["tgib_aux"]

    stats = {
        "task_loss": float(l_task.item()),
        "sim_loss": float(l_s.item()),
        "recon_loss": float(l_r.item()),
        "moe_loss": float(l_m.item()),
        "mmib_mi": mmib_stats["mmib_mi"],
        "mmib_mae": mmib_stats["mmib_mae"],
        "mmib_kl": mmib_stats["mmib_kl"],
        "mmib_loss": mmib_stats["mmib_loss"],
        "hypergraph_reg_loss": hg_stats["hypergraph_reg_loss"],
        "total_loss": float(total.item()),
        **token_stats,
        "cross_edge_weight_mean": float(hyper_aux["cross_edge_weight_mean"].item()),
        "intra_edge_weight_mean": float(hyper_aux["intra_edge_weight_mean"].item()),
        "cross_edge_weight_std": float(hyper_aux["cross_edge_weight_std"].item()),
        "intra_edge_weight_std": float(hyper_aux["intra_edge_weight_std"].item()),
        "edge_weight_std": hg_stats["edge_weight_std"],
        "cross_intra_gap": hg_stats["cross_intra_gap"],
        "edge_spread": hg_stats["edge_spread"],
        "node_attn_t": float(hyper_aux["node_attn"][:, 0].mean().item()),
        "node_attn_v": float(hyper_aux["node_attn"][:, 1].mean().item()),
        "node_attn_a": float(hyper_aux["node_attn"][:, 2].mean().item()),
        "token_weight_shared": float(token_weights[:, 0].mean().item()),
        "token_weight_text": float(token_weights[:, 1].mean().item()),
        "token_weight_vision": float(token_weights[:, 2].mean().item()),
        "token_weight_audio": float(token_weights[:, 3].mean().item()),
        "token_dominance_margin": token_stats["token_dominance_margin"],
        "attn_shared_to_shared": float(transformer_attn[:, 0, 0].mean().item()),
        "attn_shared_to_text": float(transformer_attn[:, 0, 1].mean().item()),
        "attn_shared_to_vision": float(transformer_attn[:, 0, 2].mean().item()),
        "attn_shared_to_audio": float(transformer_attn[:, 0, 3].mean().item()),
        "gate_t_mean": float(aux["tmoe_t_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "gate_v_mean": float(aux["tmoe_v_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "gate_a_mean": float(aux["tmoe_a_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "filtered_std_mean": float(tgib_aux["std"].mean().item()),
        "filter_shift": float(torch.mean(torch.abs(aux["filtered_repr"] - aux["fused_repr"])).item()),
    }
    return total, stats


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    sim_margin: float,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    renyi_alpha: float = DEFAULT_RENYI_ALPHA,
    renyi_rank_k: int = DEFAULT_RENYI_RANK,
    mmib_mae_weight: float = 1.0,
    mmib_kl_weight: float = 1e-4,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    hypergraph_reg_weight: float = DEFAULT_HYPERGRAPH_REG_WEIGHT,
):
    model.eval()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
        "mmib": 0.0,
        "mmib_mi": 0.0,
        "mmib_mae": 0.0,
        "mmib_kl": 0.0,
        "token_reg_loss": 0.0,
        "hypergraph_reg_loss": 0.0,
        "token_entropy": 0.0,
        "token_balance": 0.0,
        "token_max_weight": 0.0,
        "token_floor_penalty": 0.0,
        "token_peak_penalty": 0.0,
        "cross_edge_weight_mean": 0.0,
        "intra_edge_weight_mean": 0.0,
        "cross_edge_weight_std": 0.0,
        "intra_edge_weight_std": 0.0,
        "edge_weight_std": 0.0,
        "cross_intra_gap": 0.0,
        "edge_spread": 0.0,
        "node_attn_t": 0.0,
        "node_attn_v": 0.0,
        "node_attn_a": 0.0,
        "token_weight_shared": 0.0,
        "token_weight_text": 0.0,
        "token_weight_vision": 0.0,
        "token_weight_audio": 0.0,
        "token_dominance_margin": 0.0,
        "attn_shared_to_shared": 0.0,
        "attn_shared_to_text": 0.0,
        "attn_shared_to_vision": 0.0,
        "attn_shared_to_audio": 0.0,
        "gate_t_mean": 0.0,
        "gate_v_mean": 0.0,
        "gate_a_mean": 0.0,
        "filtered_std_mean": 0.0,
        "filter_shift": 0.0,
    }
    all_preds = []
    all_labels = []

    for batch in dataloader:
        text = batch["text"].to(device).float()
        vision = batch["vision"].to(device).float()
        audio = batch["audio"].to(device).float()
        labels = batch["label"].to(device).float().view(-1)

        preds, aux = model(text, vision, audio)
        _, stats = total_loss(
            preds,
            labels,
            aux,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            sim_margin=sim_margin,
            moe_balance_weight=moe_balance_weight,
            renyi_alpha=renyi_alpha,
            renyi_rank_k=renyi_rank_k,
            mmib_mae_weight=mmib_mae_weight,
            mmib_kl_weight=mmib_kl_weight,
            token_reg_weight=token_reg_weight,
            hypergraph_reg_weight=hypergraph_reg_weight,
        )
        batch_size = labels.size(0)
        total_samples += batch_size

        totals["loss"] += stats["total_loss"] * batch_size
        totals["task"] += stats["task_loss"] * batch_size
        totals["sim"] += stats["sim_loss"] * batch_size
        totals["recon"] += stats["recon_loss"] * batch_size
        totals["moe"] += stats["moe_loss"] * batch_size
        totals["mmib"] += stats["mmib_loss"] * batch_size
        totals["mmib_mi"] += stats["mmib_mi"] * batch_size
        totals["mmib_mae"] += stats["mmib_mae"] * batch_size
        totals["mmib_kl"] += stats["mmib_kl"] * batch_size
        totals["token_reg_loss"] += stats["token_reg_loss"] * batch_size
        totals["hypergraph_reg_loss"] += stats["hypergraph_reg_loss"] * batch_size
        totals["token_entropy"] += stats["token_entropy"] * batch_size
        totals["token_balance"] += stats["token_balance"] * batch_size
        totals["token_max_weight"] += stats["token_max_weight"] * batch_size
        totals["token_floor_penalty"] += stats["token_floor_penalty"] * batch_size
        totals["token_peak_penalty"] += stats["token_peak_penalty"] * batch_size

        for key in [
            "cross_edge_weight_mean", "intra_edge_weight_mean",
            "cross_edge_weight_std", "intra_edge_weight_std", "edge_weight_std", "cross_intra_gap", "edge_spread",
            "node_attn_t", "node_attn_v", "node_attn_a",
            "token_weight_shared", "token_weight_text", "token_weight_vision", "token_weight_audio", "token_dominance_margin",
            "attn_shared_to_shared", "attn_shared_to_text", "attn_shared_to_vision", "attn_shared_to_audio",
            "gate_t_mean", "gate_v_mean", "gate_a_mean",
            "filtered_std_mean", "filter_shift",
        ]:
            totals[key] += stats[key] * batch_size

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = {
        "MAE": compute_mae(preds, labels),
        "Corr": compute_corr(preds, labels),
        "Acc7": compute_acc7(preds, labels),
        **compute_acc2_f1(preds, labels),
        "total_loss": totals["loss"] / max(1, total_samples),
        "task_loss": totals["task"] / max(1, total_samples),
        "sim_loss": totals["sim"] / max(1, total_samples),
        "recon_loss": totals["recon"] / max(1, total_samples),
        "moe_loss": totals["moe"] / max(1, total_samples),
        "mmib_loss": totals["mmib"] / max(1, total_samples),
        "mmib_mi": totals["mmib_mi"] / max(1, total_samples),
        "mmib_mae": totals["mmib_mae"] / max(1, total_samples),
        "mmib_kl": totals["mmib_kl"] / max(1, total_samples),
        "token_reg_loss": totals["token_reg_loss"] / max(1, total_samples),
        "hypergraph_reg_loss": totals["hypergraph_reg_loss"] / max(1, total_samples),
        "token_entropy": totals["token_entropy"] / max(1, total_samples),
        "token_balance": totals["token_balance"] / max(1, total_samples),
        "token_max_weight": totals["token_max_weight"] / max(1, total_samples),
        "analysis": {
            key: value / max(1, total_samples)
            for key, value in totals.items()
            if key not in {
                "loss", "task", "sim", "recon", "moe", "mmib", "mmib_mi", "mmib_mae", "mmib_kl",
                "token_reg_loss", "hypergraph_reg_loss", "token_entropy", "token_balance", "token_max_weight"
            }
        },
    }
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    sim_margin: float,
    grad_clip: float = 1.0,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    renyi_alpha: float = DEFAULT_RENYI_ALPHA,
    renyi_rank_k: int = DEFAULT_RENYI_RANK,
    mmib_mae_weight: float = 1.0,
    mmib_kl_weight: float = 1e-4,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    hypergraph_reg_weight: float = DEFAULT_HYPERGRAPH_REG_WEIGHT,
):
    model.train()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
        "mmib": 0.0,
        "mmib_mi": 0.0,
        "mmib_mae": 0.0,
        "mmib_kl": 0.0,
        "token_reg": 0.0,
        "hypergraph_reg": 0.0,
        "token_entropy": 0.0,
        "token_balance": 0.0,
        "token_max_weight": 0.0,
        "token_floor_penalty": 0.0,
        "token_peak_penalty": 0.0,
    }

    for batch in dataloader:
        text = batch["text"].to(device).float()
        vision = batch["vision"].to(device).float()
        audio = batch["audio"].to(device).float()
        labels = batch["label"].to(device).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        preds, aux = model(text, vision, audio)
        loss, stats = total_loss(
            preds,
            labels,
            aux,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            sim_margin=sim_margin,
            moe_balance_weight=moe_balance_weight,
            renyi_alpha=renyi_alpha,
            renyi_rank_k=renyi_rank_k,
            mmib_mae_weight=mmib_mae_weight,
            mmib_kl_weight=mmib_kl_weight,
            token_reg_weight=token_reg_weight,
            hypergraph_reg_weight=hypergraph_reg_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        totals["loss"] += stats["total_loss"] * batch_size
        totals["task"] += stats["task_loss"] * batch_size
        totals["sim"] += stats["sim_loss"] * batch_size
        totals["recon"] += stats["recon_loss"] * batch_size
        totals["moe"] += stats["moe_loss"] * batch_size
        totals["mmib"] += stats["mmib_loss"] * batch_size
        totals["mmib_mi"] += stats["mmib_mi"] * batch_size
        totals["mmib_mae"] += stats["mmib_mae"] * batch_size
        totals["mmib_kl"] += stats["mmib_kl"] * batch_size
        totals["token_reg"] += stats["token_reg_loss"] * batch_size
        totals["hypergraph_reg"] += stats["hypergraph_reg_loss"] * batch_size
        totals["token_entropy"] += stats["token_entropy"] * batch_size
        totals["token_balance"] += stats["token_balance"] * batch_size
        totals["token_max_weight"] += stats["token_max_weight"] * batch_size
        totals["token_floor_penalty"] += stats["token_floor_penalty"] * batch_size
        totals["token_peak_penalty"] += stats["token_peak_penalty"] * batch_size

    return {
        "train_total_loss": totals["loss"] / max(1, total_samples),
        "train_task_loss": totals["task"] / max(1, total_samples),
        "train_sim_loss": totals["sim"] / max(1, total_samples),
        "train_recon_loss": totals["recon"] / max(1, total_samples),
        "train_moe_loss": totals["moe"] / max(1, total_samples),
        "train_mmib_loss": totals["mmib"] / max(1, total_samples),
        "train_mmib_mi": totals["mmib_mi"] / max(1, total_samples),
        "train_mmib_mae": totals["mmib_mae"] / max(1, total_samples),
        "train_mmib_kl": totals["mmib_kl"] / max(1, total_samples),
        "train_token_reg_loss": totals["token_reg"] / max(1, total_samples),
        "train_hypergraph_reg_loss": totals["hypergraph_reg"] / max(1, total_samples),
        "train_token_entropy": totals["token_entropy"] / max(1, total_samples),
        "train_token_balance": totals["token_balance"] / max(1, total_samples),
        "train_token_max_weight": totals["token_max_weight"] / max(1, total_samples),
        "train_token_floor_penalty": totals["token_floor_penalty"] / max(1, total_samples),
        "train_token_peak_penalty": totals["token_peak_penalty"] / max(1, total_samples),
    }
