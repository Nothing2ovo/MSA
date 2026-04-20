import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

DEFAULT_MOE_BALANCE_WEIGHT = 1e-2


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
    # pool each sequence to one utterance-level representation for the triplet loss
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
    # Paper Eq. (14): sum over modalities rather than averaging.
    losses = []
    for key in ["tmoe_t_aux", "tmoe_v_aux", "tmoe_a_aux"]:
        importance = aux[key]["importance"]
        load = aux[key]["load"]
        losses.append(balance_weight * (cv_squared(importance) + cv_squared(load)))
    return sum(losses)


def task_loss_regression(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds.view(-1), labels.view(-1))


def total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    beta: float = 0.05,
    gamma: float = 0.10,
    sim_margin: float = 0.2,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
):
    l_task = task_loss_regression(preds, labels)
    l_s = similarity_loss(aux, labels, margin=sim_margin)
    l_r = reconstruction_loss(aux)
    l_m = moe_load_loss(aux, balance_weight=moe_balance_weight)
    total = l_task + alpha * l_s + beta * l_r + gamma * l_m

    hyper_aux = aux["hyper_aux"]
    dwf_aux = aux["dwf_aux"]
    attn = dwf_aux["attn"]
    attn_entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)
    attn_entropy = attn_entropy / math.log(attn.size(-1))

    stats = {
        "task_loss": float(l_task.item()),
        "sim_loss": float(l_s.item()),
        "recon_loss": float(l_r.item()),
        "moe_loss": float(l_m.item()),
        "total_loss": float(total.item()),
        "cross_edge_weight_mean": float(hyper_aux["cross_edge_weight_mean"].item()),
        "intra_edge_weight_mean": float(hyper_aux["intra_edge_weight_mean"].item()),
        "cross_edge_weight_std": float(hyper_aux["cross_edge_weight_std"].item()),
        "intra_edge_weight_std": float(hyper_aux["intra_edge_weight_std"].item()),
        "edge_weight_std": float(hyper_aux["edge_weight_std"].item()),
        "cross_intra_gap": float(hyper_aux["cross_intra_gap"].item()),
        "edge_spread": float(hyper_aux["edge_spread"].item()),
        "gate_t_mean": float(aux["tmoe_t_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "gate_v_mean": float(aux["tmoe_v_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "gate_a_mean": float(aux["tmoe_a_aux"]["gate_probs"].max(dim=-1).values.mean().item()),
        "dwf_attn_entropy": float(attn_entropy.mean().item()),
        "dwf_attn_trace": float(attn.diagonal(dim1=-2, dim2=-1).mean().item()),
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
    sim_margin: float,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
):
    model.eval()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
        "cross_edge_weight_mean": 0.0,
        "intra_edge_weight_mean": 0.0,
        "cross_edge_weight_std": 0.0,
        "intra_edge_weight_std": 0.0,
        "edge_weight_std": 0.0,
        "cross_intra_gap": 0.0,
        "edge_spread": 0.0,
        "gate_t_mean": 0.0,
        "gate_v_mean": 0.0,
        "gate_a_mean": 0.0,
        "dwf_attn_entropy": 0.0,
        "dwf_attn_trace": 0.0,
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
            sim_margin=sim_margin,
            moe_balance_weight=moe_balance_weight,
        )
        batch_size = labels.size(0)
        total_samples += batch_size

        totals["loss"] += stats["total_loss"] * batch_size
        totals["task"] += stats["task_loss"] * batch_size
        totals["sim"] += stats["sim_loss"] * batch_size
        totals["recon"] += stats["recon_loss"] * batch_size
        totals["moe"] += stats["moe_loss"] * batch_size
        for key in [
            "cross_edge_weight_mean", "intra_edge_weight_mean",
            "cross_edge_weight_std", "intra_edge_weight_std", "edge_weight_std",
            "cross_intra_gap", "edge_spread",
            "gate_t_mean", "gate_v_mean", "gate_a_mean",
            "dwf_attn_entropy", "dwf_attn_trace",
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
        "analysis": {
            key: value / max(1, total_samples)
            for key, value in totals.items()
            if key not in {"loss", "task", "sim", "recon", "moe"}
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
    sim_margin: float,
    grad_clip: float = 1.0,
    moe_balance_weight: float = DEFAULT_MOE_BALANCE_WEIGHT,
):
    model.train()
    total_samples = 0
    totals = {
        "loss": 0.0,
        "task": 0.0,
        "sim": 0.0,
        "recon": 0.0,
        "moe": 0.0,
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
            sim_margin=sim_margin,
            moe_balance_weight=moe_balance_weight,
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

    return {
        "train_total_loss": totals["loss"] / max(1, total_samples),
        "train_task_loss": totals["task"] / max(1, total_samples),
        "train_sim_loss": totals["sim"] / max(1, total_samples),
        "train_recon_loss": totals["recon"] / max(1, total_samples),
        "train_moe_loss": totals["moe"] / max(1, total_samples),
    }
