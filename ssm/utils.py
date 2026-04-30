import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

DEFAULT_MOE_BALANCE_WEIGHT = 1e-2
DEFAULT_TOKEN_REG_WEIGHT = 2e-2
DEFAULT_SHARED_MIXER_REG_WEIGHT = 5.5e-2
DEFAULT_TOKEN_TARGET_ENTROPY = 0.87
DEFAULT_PRIVATE_MIN_WEIGHT = 0.055
DEFAULT_SHARED_TARGET_WEIGHT = 0.27
DEFAULT_SHARED_DOMINANCE_MARGIN = 0.005
DEFAULT_TOKEN_MAX_WEIGHT = 0.68
DEFAULT_SELECT_TARGET_STD = 0.034
DEFAULT_SELECT_MIN_GAP = 0.013
DEFAULT_CROSS_SELECT_TARGET_STD = 0.018
DEFAULT_INTRA_SELECT_TARGET_STD = 0.018
DEFAULT_SELECT_SPREAD_MARGIN = 0.050
DEFAULT_ACC5_LOSS_WEIGHT = 0.10
DEFAULT_ACC7_LOSS_WEIGHT = 0.06
DEFAULT_SUPCON_TEMPERATURE = 0.20
DEFAULT_UNSUPCON_TEMPERATURE = 0.20
DEFAULT_SHARED_AUX_WEIGHT = 0.10


def _safe_neg_fill_value(x: torch.Tensor) -> float:
    if not x.is_floating_point():
        return -1e4
    info = torch.finfo(x.dtype)
    # use a large finite negative instead of raw min to keep logsumexp stable
    return max(info.min, -1e4)


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def _dp_reduce_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.float().mean().item())
    return float(x)


def _dp_reduce_lastdim_series(x: torch.Tensor) -> torch.Tensor:
    x = _as_float_tensor(x)
    while x.dim() > 1:
        x = x.mean(dim=0)
    return x


def _dp_reduce_matrix_lastdim(x: torch.Tensor) -> torch.Tensor:
    x = _as_float_tensor(x)
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x.reshape(-1, x.shape[-1])


def compute_mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds.view(-1) - labels.view(-1))).item()


def compute_corr(preds: torch.Tensor, labels: torch.Tensor) -> float:
    x = preds.view(-1)
    y = labels.view(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()) + 1e-8
    return ((x * y).sum() / denom).item()


def compute_acc5(preds: torch.Tensor, labels: torch.Tensor) -> float:
    p = torch.clamp(torch.round(preds.view(-1)), min=-2, max=2)
    y = torch.clamp(torch.round(labels.view(-1)), min=-2, max=2)
    return (p == y).float().mean().item()


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


def supervised_shared_contrastive_loss(
    aux: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    temperature: float = DEFAULT_SUPCON_TEMPERATURE,
) -> torch.Tensor:
    z = F.normalize(aux["shared_proj"].float(), dim=-1)
    cls = labels_to_classes(labels)
    batch_size = z.size(0)
    if batch_size <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    logits_mask = ~torch.eye(batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(~logits_mask, _safe_neg_fill_value(sim))

    positive_mask = (cls.unsqueeze(1) == cls.unsqueeze(0)) & logits_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    loss = -(positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1).float()
    return loss[valid].mean()


def unsupervised_shared_contrastive_loss(
    aux: Dict[str, torch.Tensor],
    temperature: float = DEFAULT_UNSUPCON_TEMPERATURE,
) -> torch.Tensor:
    z1 = F.normalize(aux["shared_proj"].float(), dim=-1)
    z2 = F.normalize(aux["shared_proj_aug"].float(), dim=-1)
    batch_size = z1.size(0)
    if batch_size <= 1:
        return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    logits_mask = ~torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(~logits_mask, _safe_neg_fill_value(sim))

    pos_idx = torch.arange(2 * batch_size, device=z.device)
    pos_idx = (pos_idx + batch_size) % (2 * batch_size)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -log_prob[torch.arange(2 * batch_size, device=z.device), pos_idx]
    return loss.mean()


def shared_mixer_structure_loss(
    aux: Dict[str, torch.Tensor],
    target_select_std: float = DEFAULT_SELECT_TARGET_STD,
    min_cross_intra_gap: float = DEFAULT_SELECT_MIN_GAP,
    target_cross_std: float = DEFAULT_CROSS_SELECT_TARGET_STD,
    target_intra_std: float = DEFAULT_INTRA_SELECT_TARGET_STD,
    min_select_spread: float = DEFAULT_SELECT_SPREAD_MARGIN,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    shared_mixer_aux = aux["shared_mixer_aux"]

    per_layer_select_std = _dp_reduce_lastdim_series(shared_mixer_aux["per_layer_select_std"])
    per_layer_cross_std = _dp_reduce_lastdim_series(shared_mixer_aux["per_layer_cross_select_std"])
    per_layer_intra_std = _dp_reduce_lastdim_series(shared_mixer_aux["per_layer_intra_select_std"])
    per_layer_gap = _dp_reduce_lastdim_series(shared_mixer_aux["per_layer_cross_intra_gap"].abs())
    per_layer_spread = _dp_reduce_lastdim_series(shared_mixer_aux["per_layer_select_spread"])

    num_layers = int(per_layer_select_std.numel())
    layer_weights = torch.linspace(
        0.90,
        1.10,
        steps=max(1, num_layers),
        device=per_layer_select_std.device,
        dtype=per_layer_select_std.dtype,
    )
    layer_weights = layer_weights / layer_weights.sum().clamp_min(1e-8)

    def weighted_mean(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(layer_weights * x)

    select_std_pen = weighted_mean(F.relu(target_select_std - per_layer_select_std))
    cross_std_pen = weighted_mean(F.relu(target_cross_std - per_layer_cross_std))
    intra_std_pen = weighted_mean(F.relu(target_intra_std - per_layer_intra_std))
    gap_pen = weighted_mean(F.relu(min_cross_intra_gap - per_layer_gap))
    spread_pen = weighted_mean(F.relu(min_select_spread - per_layer_spread))

    if num_layers > 1:
        next_select_std = per_layer_select_std[1:]
        prev_select_std = per_layer_select_std[:-1]
        next_spread = per_layer_spread[1:]
        prev_spread = per_layer_spread[:-1]
        late_std_preserve_pen = F.relu(0.60 * prev_select_std - next_select_std).mean()
        late_spread_preserve_pen = F.relu(0.60 * prev_spread - next_spread).mean()
    else:
        late_std_preserve_pen = torch.zeros((), device=per_layer_select_std.device, dtype=per_layer_select_std.dtype)
        late_spread_preserve_pen = torch.zeros((), device=per_layer_select_std.device, dtype=per_layer_select_std.dtype)

    loss = (
        1.15 * select_std_pen
        + 0.55 * cross_std_pen
        + 0.55 * intra_std_pen
        + 1.10 * spread_pen
        + 0.18 * gap_pen
        + 0.18 * late_std_preserve_pen
        + 0.18 * late_spread_preserve_pen
    )

    stats = {
        "shared_mixer_reg_loss": float(loss.item()),
        "select_std": float(per_layer_select_std.mean().item()),
        "cross_select_std": float(per_layer_cross_std.mean().item()),
        "intra_select_std": float(per_layer_intra_std.mean().item()),
        "cross_intra_gap": float(per_layer_gap.mean().item()),
        "select_spread": float(per_layer_spread.mean().item()),
        "multi_layer_select_std": float(per_layer_select_std.mean().item()),
        "multi_layer_cross_select_std": float(per_layer_cross_std.mean().item()),
        "multi_layer_intra_select_std": float(per_layer_intra_std.mean().item()),
        "multi_layer_cross_intra_gap": float(per_layer_gap.mean().item()),
        "multi_layer_select_spread": float(per_layer_spread.mean().item()),
        "late_std_preserve_penalty": float(late_std_preserve_pen.item()),
        "late_spread_preserve_penalty": float(late_spread_preserve_pen.item()),
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
    eps = 1e-8
    num_tokens = token_weights.size(1)
    entropy = -(token_weights * torch.log(token_weights.clamp_min(eps))).sum(dim=1)
    entropy = entropy / math.log(num_tokens)
    entropy_penalty = (entropy - target_entropy).pow(2).mean()

    mean_w = token_weights.mean(dim=0)
    target_prior = torch.tensor([0.34, 0.26, 0.21, 0.19], device=token_weights.device, dtype=token_weights.dtype)
    balance = F.mse_loss(mean_w, target_prior)

    shared_w = token_weights[:, 0]
    text_w = token_weights[:, 1]
    private_w = token_weights[:, 1:]
    max_private = private_w.max(dim=1).values
    other_private_max = private_w[:, 1:].max(dim=1).values

    shared_floor_penalty = F.relu(shared_target_weight - shared_w).mean()
    shared_margin_penalty = F.relu(max_private + shared_margin - shared_w).mean()
    private_floor_penalty = F.relu(private_min_weight - private_w).mean()
    text_soft_rank_penalty = F.relu(other_private_max + 0.01 - text_w).mean()
    peak_penalty = F.relu(token_weights.max(dim=1).values - max_weight).mean()

    loss = (
        entropy_penalty
        + 0.18 * balance
        + 0.80 * shared_floor_penalty
        + 0.65 * shared_margin_penalty
        + 0.65 * private_floor_penalty
        + 0.25 * text_soft_rank_penalty
        + 0.75 * peak_penalty
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



def shared_auxiliary_loss(aux: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    shared_pred = aux.get("shared_pred", None)
    if shared_pred is None:
        return torch.zeros((), device=labels.device, dtype=labels.dtype)
    return F.smooth_l1_loss(shared_pred.view(-1), labels.view(-1))


def task_loss_regression(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds.view(-1), labels.view(-1))


def ordinal_threshold_loss(preds: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    preds = preds.view(-1)
    labels = labels.view(-1)
    logits = preds.unsqueeze(1) - thresholds.view(1, -1)
    targets = (labels.unsqueeze(1) > thresholds.view(1, -1)).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def classification_aux_losses(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = preds.device
    dtype = preds.dtype
    thresholds_5 = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=device, dtype=dtype)
    thresholds_7 = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], device=device, dtype=dtype)
    loss5 = ordinal_threshold_loss(preds, labels.clamp(min=-2.0, max=2.0), thresholds_5)
    loss7 = ordinal_threshold_loss(preds, labels.clamp(min=-3.0, max=3.0), thresholds_7)
    return loss5, loss7


def total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    sim_weight: float = 0.05,
    recon_weight: float = 0.05,
    moe_weight: float = 0.10,
    supcon_weight: float = 0.05,
    unsupcon_weight: float = 0.05,
    sim_margin: float = 0.2,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    shared_mixer_reg_weight: float = DEFAULT_SHARED_MIXER_REG_WEIGHT,
    shared_aux_weight: float = DEFAULT_SHARED_AUX_WEIGHT,
    acc5_loss_weight: float = DEFAULT_ACC5_LOSS_WEIGHT,
    acc7_loss_weight: float = DEFAULT_ACC7_LOSS_WEIGHT,
):
    l_task = task_loss_regression(preds, labels)
    l_s = similarity_loss(aux, labels, margin=sim_margin)
    l_r = reconstruction_loss(aux)
    l_m = moe_load_loss(aux, balance_weight=moe_balance_weight)
    l_sup = supervised_shared_contrastive_loss(aux, labels)
    l_unsup = unsupervised_shared_contrastive_loss(aux)
    token_weights = aux["token_fusion_aux"]["token_weights"]
    l_token_reg, token_stats = token_regularization_loss(token_weights)
    l_shared_mixer, shared_mixer_stats = shared_mixer_structure_loss(aux)
    l_shared_aux = shared_auxiliary_loss(aux, labels)
    l_acc5, l_acc7 = classification_aux_losses(preds, labels)

    total = (
        l_task
        + sim_weight * l_s
        + recon_weight * l_r
        + moe_weight * l_m
        + supcon_weight * l_sup
        + unsupcon_weight * l_unsup
        + token_reg_weight * l_token_reg
        + shared_mixer_reg_weight * l_shared_mixer
        + shared_aux_weight * l_shared_aux
        + acc5_loss_weight * l_acc5
        + acc7_loss_weight * l_acc7
    )

    fusion_attn = aux["fusion_attn"]
    if fusion_attn.dim() == 4:
        fusion_attn = fusion_attn.mean(dim=1)
    fusion_attn = _dp_reduce_matrix_lastdim(fusion_attn.reshape(-1, fusion_attn.shape[-2], fusion_attn.shape[-1])).reshape(-1, fusion_attn.shape[-2], fusion_attn.shape[-1])
    shared_mixer_aux = aux["shared_mixer_aux"]
    modality_attn = _dp_reduce_matrix_lastdim(shared_mixer_aux["modality_attn"])
    token_weights = _dp_reduce_matrix_lastdim(token_weights)
    gate_t_probs = _as_float_tensor(aux["tmoe_t_aux"]["gate_probs"])
    gate_v_probs = _as_float_tensor(aux["tmoe_v_aux"]["gate_probs"])
    gate_a_probs = _as_float_tensor(aux["tmoe_a_aux"]["gate_probs"])
    if gate_t_probs.dim() > 2:
        gate_t_probs = gate_t_probs.reshape(-1, gate_t_probs.shape[-1])
    if gate_v_probs.dim() > 2:
        gate_v_probs = gate_v_probs.reshape(-1, gate_v_probs.shape[-1])
    if gate_a_probs.dim() > 2:
        gate_a_probs = gate_a_probs.reshape(-1, gate_a_probs.shape[-1])
    shared_repr = _as_float_tensor(aux["shared_repr"])
    shared_repr_aug = _as_float_tensor(aux["shared_repr_aug"])
    if shared_repr.dim() > 2:
        shared_repr = shared_repr.reshape(-1, shared_repr.shape[-1])
    if shared_repr_aug.dim() > 2:
        shared_repr_aug = shared_repr_aug.reshape(-1, shared_repr_aug.shape[-1])
    fused_repr = _as_float_tensor(aux.get("final_repr", aux["fused_repr"]))
    if fused_repr.dim() > 2:
        fused_repr = fused_repr.reshape(-1, fused_repr.shape[-1])

    stats = {
        "task_loss": _dp_reduce_scalar(l_task),
        "sim_loss": _dp_reduce_scalar(l_s),
        "recon_loss": _dp_reduce_scalar(l_r),
        "moe_loss": _dp_reduce_scalar(l_m),
        "supcon_loss": _dp_reduce_scalar(l_sup),
        "unsupcon_loss": _dp_reduce_scalar(l_unsup),
        "shared_mixer_reg_loss": shared_mixer_stats["shared_mixer_reg_loss"],
        "shared_aux_loss": _dp_reduce_scalar(l_shared_aux),
        "acc5_loss": _dp_reduce_scalar(l_acc5),
        "acc7_loss": _dp_reduce_scalar(l_acc7),
        "total_loss": _dp_reduce_scalar(total),
        **token_stats,
        "cross_select_mean": _dp_reduce_scalar(shared_mixer_aux["cross_select_mean"]),
        "intra_select_mean": _dp_reduce_scalar(shared_mixer_aux["intra_select_mean"]),
        "cross_select_std": _dp_reduce_scalar(shared_mixer_aux["cross_select_std"]),
        "intra_select_std": _dp_reduce_scalar(shared_mixer_aux["intra_select_std"]),
        "select_std": shared_mixer_stats["select_std"],
        "cross_intra_gap": shared_mixer_stats["cross_intra_gap"],
        "select_spread": shared_mixer_stats["select_spread"],
        "multi_layer_select_std": shared_mixer_stats["multi_layer_select_std"],
        "multi_layer_cross_intra_gap": shared_mixer_stats["multi_layer_cross_intra_gap"],
        "multi_layer_select_spread": shared_mixer_stats["multi_layer_select_spread"],
        "late_std_preserve_penalty": shared_mixer_stats["late_std_preserve_penalty"],
        "late_spread_preserve_penalty": shared_mixer_stats["late_spread_preserve_penalty"],
        "modality_attn_t": _dp_reduce_scalar(modality_attn[:, 0].mean()),
        "modality_attn_v": _dp_reduce_scalar(modality_attn[:, 1].mean()),
        "modality_attn_a": _dp_reduce_scalar(modality_attn[:, 2].mean()),
        "token_weight_shared": _dp_reduce_scalar(token_weights[:, 0].mean()),
        "token_weight_text": _dp_reduce_scalar(token_weights[:, 1].mean()),
        "token_weight_vision": _dp_reduce_scalar(token_weights[:, 2].mean()),
        "token_weight_audio": _dp_reduce_scalar(token_weights[:, 3].mean()),
        "token_dominance_margin": token_stats["token_dominance_margin"],
        "attn_shared_to_shared": _dp_reduce_scalar(fusion_attn[:, 0, 0].mean()),
        "attn_shared_to_text": _dp_reduce_scalar(fusion_attn[:, 0, 1].mean()),
        "attn_shared_to_vision": _dp_reduce_scalar(fusion_attn[:, 0, 2].mean()),
        "attn_shared_to_audio": _dp_reduce_scalar(fusion_attn[:, 0, 3].mean()),
        "gate_t_mean": _dp_reduce_scalar(gate_t_probs.max(dim=-1).values.mean()),
        "gate_v_mean": _dp_reduce_scalar(gate_v_probs.max(dim=-1).values.mean()),
        "gate_a_mean": _dp_reduce_scalar(gate_a_probs.max(dim=-1).values.mean()),
        "shared_view_gap": _dp_reduce_scalar(torch.mean(torch.abs(shared_repr - shared_repr_aug))),
        "fused_repr_norm": _dp_reduce_scalar(fused_repr.norm(dim=-1).mean()),
        "direct_gate_mean": _dp_reduce_scalar(aux.get("direct_gate", torch.zeros((), device=preds.device)).mean()),
    }
    return total, stats


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    sim_weight: float,
    recon_weight: float,
    moe_weight: float,
    supcon_weight: float,
    unsupcon_weight: float,
    sim_margin: float,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    shared_mixer_reg_weight: float = DEFAULT_SHARED_MIXER_REG_WEIGHT,
    shared_aux_weight: float = DEFAULT_SHARED_AUX_WEIGHT,
    acc5_loss_weight: float = DEFAULT_ACC5_LOSS_WEIGHT,
    acc7_loss_weight: float = DEFAULT_ACC7_LOSS_WEIGHT,
    use_amp: bool = False,
):
    model.eval()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
        "supcon": 0.0,
        "unsupcon": 0.0,
        "token_reg_loss": 0.0,
        "shared_mixer_reg_loss": 0.0,
        "shared_aux_loss": 0.0,
        "acc5_loss": 0.0,
        "acc7_loss": 0.0,
        "token_entropy": 0.0,
        "token_balance": 0.0,
        "token_max_weight": 0.0,
        "token_floor_penalty": 0.0,
        "token_peak_penalty": 0.0,
        "cross_select_mean": 0.0,
        "intra_select_mean": 0.0,
        "cross_select_std": 0.0,
        "intra_select_std": 0.0,
        "select_std": 0.0,
        "cross_intra_gap": 0.0,
        "select_spread": 0.0,
        "multi_layer_select_std": 0.0,
        "multi_layer_cross_intra_gap": 0.0,
        "multi_layer_select_spread": 0.0,
        "late_std_preserve_penalty": 0.0,
        "late_spread_preserve_penalty": 0.0,
        "modality_attn_t": 0.0,
        "modality_attn_v": 0.0,
        "modality_attn_a": 0.0,
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
        "shared_view_gap": 0.0,
        "fused_repr_norm": 0.0,
    }
    all_preds = []
    all_labels = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in dataloader:
        text = batch["text"].to(device, non_blocking=True).float()
        vision = batch["vision"].to(device, non_blocking=True).float()
        audio = batch["audio"].to(device, non_blocking=True).float()
        labels = batch["label"].to(device, non_blocking=True).float().view(-1)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            preds, aux = model(text, vision, audio)
            _, stats = total_loss(
                preds,
                labels,
                aux,
                sim_weight=sim_weight,
                recon_weight=recon_weight,
                moe_weight=moe_weight,
                supcon_weight=supcon_weight,
                unsupcon_weight=unsupcon_weight,
                sim_margin=sim_margin,
                moe_balance_weight=moe_balance_weight,
                token_reg_weight=token_reg_weight,
                shared_mixer_reg_weight=shared_mixer_reg_weight,
                shared_aux_weight=shared_aux_weight,
                acc5_loss_weight=acc5_loss_weight,
                acc7_loss_weight=acc7_loss_weight,
            )

        batch_size = labels.size(0)
        total_samples += batch_size

        totals["loss"] += stats["total_loss"] * batch_size
        totals["task"] += stats["task_loss"] * batch_size
        totals["sim"] += stats["sim_loss"] * batch_size
        totals["recon"] += stats["recon_loss"] * batch_size
        totals["moe"] += stats["moe_loss"] * batch_size
        totals["supcon"] += stats["supcon_loss"] * batch_size
        totals["unsupcon"] += stats["unsupcon_loss"] * batch_size
        totals["token_reg_loss"] += stats["token_reg_loss"] * batch_size
        totals["shared_mixer_reg_loss"] += stats["shared_mixer_reg_loss"] * batch_size
        totals["shared_aux_loss"] += stats["shared_aux_loss"] * batch_size
        totals["acc5_loss"] += stats["acc5_loss"] * batch_size
        totals["acc7_loss"] += stats["acc7_loss"] * batch_size
        totals["token_entropy"] += stats["token_entropy"] * batch_size
        totals["token_balance"] += stats["token_balance"] * batch_size
        totals["token_max_weight"] += stats["token_max_weight"] * batch_size
        totals["token_floor_penalty"] += stats["token_floor_penalty"] * batch_size
        totals["token_peak_penalty"] += stats["token_peak_penalty"] * batch_size

        for key in [
            "cross_select_mean", "intra_select_mean",
            "cross_select_std", "intra_select_std", "select_std", "cross_intra_gap", "select_spread",
            "multi_layer_select_std", "multi_layer_cross_intra_gap", "multi_layer_select_spread",
            "late_std_preserve_penalty", "late_spread_preserve_penalty",
            "modality_attn_t", "modality_attn_v", "modality_attn_a",
            "token_weight_shared", "token_weight_text", "token_weight_vision", "token_weight_audio", "token_dominance_margin",
            "attn_shared_to_shared", "attn_shared_to_text", "attn_shared_to_vision", "attn_shared_to_audio",
            "gate_t_mean", "gate_v_mean", "gate_a_mean",
            "shared_view_gap", "fused_repr_norm",
        ]:
            totals[key] += stats[key] * batch_size

        all_preds.append(preds.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = {
        "MAE": compute_mae(preds, labels),
        "Corr": compute_corr(preds, labels),
        "Acc5": compute_acc5(preds, labels),
        "Acc7": compute_acc7(preds, labels),
        **compute_acc2_f1(preds, labels),
        "total_loss": totals["loss"] / max(1, total_samples),
        "task_loss": totals["task"] / max(1, total_samples),
        "sim_loss": totals["sim"] / max(1, total_samples),
        "recon_loss": totals["recon"] / max(1, total_samples),
        "moe_loss": totals["moe"] / max(1, total_samples),
        "supcon_loss": totals["supcon"] / max(1, total_samples),
        "unsupcon_loss": totals["unsupcon"] / max(1, total_samples),
        "token_reg_loss": totals["token_reg_loss"] / max(1, total_samples),
        "shared_mixer_reg_loss": totals["shared_mixer_reg_loss"] / max(1, total_samples),
        "shared_aux_loss": totals["shared_aux_loss"] / max(1, total_samples),
        "acc5_loss": totals["acc5_loss"] / max(1, total_samples),
        "acc7_loss": totals["acc7_loss"] / max(1, total_samples),
        "token_entropy": totals["token_entropy"] / max(1, total_samples),
        "token_balance": totals["token_balance"] / max(1, total_samples),
        "token_max_weight": totals["token_max_weight"] / max(1, total_samples),
        "analysis": {
            key: value / max(1, total_samples)
            for key, value in totals.items()
            if key not in {
                "loss", "task", "sim", "recon", "moe", "supcon", "unsupcon",
                "token_reg_loss", "shared_mixer_reg_loss", "shared_aux_loss", "acc5_loss", "acc7_loss", "token_entropy", "token_balance", "token_max_weight"
            }
        },
    }
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    sim_weight: float,
    recon_weight: float,
    moe_weight: float,
    supcon_weight: float,
    unsupcon_weight: float,
    sim_margin: float,
    grad_clip: float = 1.0,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
    token_reg_weight: float = DEFAULT_TOKEN_REG_WEIGHT,
    shared_mixer_reg_weight: float = DEFAULT_SHARED_MIXER_REG_WEIGHT,
    shared_aux_weight: float = DEFAULT_SHARED_AUX_WEIGHT,
    acc5_loss_weight: float = DEFAULT_ACC5_LOSS_WEIGHT,
    acc7_loss_weight: float = DEFAULT_ACC7_LOSS_WEIGHT,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
):
    model.train()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
        "supcon": 0.0,
        "unsupcon": 0.0,
        "token_reg": 0.0,
        "shared_mixer_reg": 0.0,
        "shared_aux_loss": 0.0,
        "acc5_loss": 0.0,
        "acc7_loss": 0.0,
        "token_entropy": 0.0,
        "token_balance": 0.0,
        "token_max_weight": 0.0,
        "token_floor_penalty": 0.0,
        "token_peak_penalty": 0.0,
    }

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in dataloader:
        text = batch["text"].to(device, non_blocking=True).float()
        vision = batch["vision"].to(device, non_blocking=True).float()
        audio = batch["audio"].to(device, non_blocking=True).float()
        labels = batch["label"].to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            preds, aux = model(text, vision, audio)
            loss, stats = total_loss(
                preds,
                labels,
                aux,
                sim_weight=sim_weight,
                recon_weight=recon_weight,
                moe_weight=moe_weight,
                supcon_weight=supcon_weight,
                unsupcon_weight=unsupcon_weight,
                sim_margin=sim_margin,
                moe_balance_weight=moe_balance_weight,
                token_reg_weight=token_reg_weight,
                shared_mixer_reg_weight=shared_mixer_reg_weight,
                shared_aux_weight=shared_aux_weight,
                acc5_loss_weight=acc5_loss_weight,
                acc7_loss_weight=acc7_loss_weight,
            )

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        totals["supcon"] += stats["supcon_loss"] * batch_size
        totals["unsupcon"] += stats["unsupcon_loss"] * batch_size
        totals["token_reg"] += stats["token_reg_loss"] * batch_size
        totals["shared_mixer_reg"] += stats["shared_mixer_reg_loss"] * batch_size
        totals["shared_aux_loss"] += stats["shared_aux_loss"] * batch_size
        totals["acc5_loss"] += stats["acc5_loss"] * batch_size
        totals["acc7_loss"] += stats["acc7_loss"] * batch_size
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
        "train_supcon_loss": totals["supcon"] / max(1, total_samples),
        "train_unsupcon_loss": totals["unsupcon"] / max(1, total_samples),
        "train_token_reg_loss": totals["token_reg"] / max(1, total_samples),
        "train_shared_mixer_reg_loss": totals["shared_mixer_reg"] / max(1, total_samples),
        "train_shared_aux_loss": totals["shared_aux_loss"] / max(1, total_samples),
        "train_acc5_loss": totals["acc5_loss"] / max(1, total_samples),
        "train_acc7_loss": totals["acc7_loss"] / max(1, total_samples),
        "train_token_entropy": totals["token_entropy"] / max(1, total_samples),
        "train_token_balance": totals["token_balance"] / max(1, total_samples),
        "train_token_max_weight": totals["token_max_weight"] / max(1, total_samples),
        "train_token_floor_penalty": totals["token_floor_penalty"] / max(1, total_samples),
        "train_token_peak_penalty": totals["token_peak_penalty"] / max(1, total_samples),
    }
