import os
import sys
import random
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_mosei_from_pkl
from model import DHMModel
from utils import evaluate, train_one_epoch

BASE_DIR = PROJECT_ROOT
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "final_test_results.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model file")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_dhm_mosei_dwf_baseline.pt")
DATA_FILE = os.path.join(BASE_DIR, "data", "aligned_50e.pkl")


def ensure_output_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)


def set_seed(seed: int = 3407, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def model_selection_score(metrics: Dict[str, float]) -> float:
    return metrics["MAE"] - 0.10 * metrics["Corr"] - 0.01 * metrics["Acc2_posneg"]


def plot_training_curves(history: Dict[str, list], save_dir: str = PLOTS_DIR) -> None:
    epochs = range(1, len(history["train_total_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_total_loss"], label="train total loss")
    plt.plot(epochs, history["valid_total_loss"], label="valid total loss")
    plt.plot(epochs, history["valid_mae"], label="valid MAE")
    plt.plot(epochs, history["valid_corr"], label="valid Corr")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Core Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_core_metrics.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["valid_acc2_posneg"], label="valid Acc2 pos/neg")
    plt.plot(epochs, history["valid_acc7"], label="valid Acc7")
    plt.plot(epochs, history["valid_f1_posneg"], label="valid F1 pos/neg")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Classification Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "validation_cls_metrics.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["cross_edge_weight_mean"], label="cross edge weight")
    plt.plot(epochs, history["intra_edge_weight_mean"], label="intra edge weight")
    plt.plot(epochs, history["edge_weight_std"], label="edge std")
    plt.plot(epochs, history["cross_intra_gap"], label="cross/intra gap")
    plt.plot(epochs, history["dwf_attn_entropy"], label="DWF attn entropy")
    plt.plot(epochs, history["dwf_attn_trace"], label="DWF attn trace")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Structure and DWF Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "structure_dwf_metrics.png"))
    plt.close()


def save_final_test_results(file_path: str, metrics: Dict[str, float]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis = metrics["analysis"]
    lines = [
        f"[{timestamp}]",
        "========== Final Test (mosei dhm + paper DWF baseline) ==========" ,
        f"Test total loss: {metrics['total_loss']:.4f}",
        f"Test task loss : {metrics['task_loss']:.4f}",
        f"Test sim loss  : {metrics['sim_loss']:.4f}",
        f"Test recon loss: {metrics['recon_loss']:.4f}",
        f"Test moe loss  : {metrics['moe_loss']:.4f}",
        f"Test MAE       : {metrics['MAE']:.4f}",
        f"Test Corr      : {metrics['Corr']:.4f}",
        f"Test Acc2      : {metrics['Acc2_nonneg']:.4f} / {metrics['Acc2_posneg']:.4f}",
        f"Test F1        : {metrics['F1_nonneg']:.4f} / {metrics['F1_posneg']:.4f}",
        f"Test Acc7      : {metrics['Acc7']:.4f}",
        "Analysis:",
        f"  cross edge mean   : {analysis['cross_edge_weight_mean']:.4f}",
        f"  intra edge mean   : {analysis['intra_edge_weight_mean']:.4f}",
        f"  cross edge std    : {analysis['cross_edge_weight_std']:.4f}",
        f"  intra edge std    : {analysis['intra_edge_weight_std']:.4f}",
        f"  edge std/gap/spread: {analysis['edge_weight_std']:.4f} / {analysis['cross_intra_gap']:.4f} / {analysis['edge_spread']:.4f}",
        f"  gate t/v/a        : {analysis['gate_t_mean']:.4f} / {analysis['gate_v_mean']:.4f} / {analysis['gate_a_mean']:.4f}",
        f"  dwf ent/trace     : {analysis['dwf_attn_entropy']:.4f} / {analysis['dwf_attn_trace']:.4f}",
        "",
    ]
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    lr: float,
    train_stats: Dict[str, float],
    valid_metrics: Dict[str, float],
    score: float,
    improved: bool,
    wait: int,
    patience: int,
) -> None:
    print(f"\n[Epoch {epoch + 1:02d}/{num_epochs}]")
    print(f"  lr              = {lr:.6f}")
    print(
        f"  train total/task= {train_stats['train_total_loss']:.4f} / {train_stats['train_task_loss']:.4f} | "
        f"sim={train_stats['train_sim_loss']:.4f} recon={train_stats['train_recon_loss']:.4f} "
        f"moe={train_stats['train_moe_loss']:.4f}"
    )
    print(
        f"  valid MAE/Corr  = {valid_metrics['MAE']:.4f} / {valid_metrics['Corr']:.4f} | "
        f"Acc2={valid_metrics['Acc2_nonneg']:.4f}/{valid_metrics['Acc2_posneg']:.4f} | "
        f"F1={valid_metrics['F1_nonneg']:.4f}/{valid_metrics['F1_posneg']:.4f} | Acc7={valid_metrics['Acc7']:.4f}"
    )
    print(
        f"  valid losses    = total {valid_metrics['total_loss']:.4f} | task {valid_metrics['task_loss']:.4f} | "
        f"sim {valid_metrics['sim_loss']:.4f} | recon {valid_metrics['recon_loss']:.4f} | moe {valid_metrics['moe_loss']:.4f}"
    )
    print(
        f"  structure       = cross/intra {valid_metrics['analysis']['cross_edge_weight_mean']:.4f} / {valid_metrics['analysis']['intra_edge_weight_mean']:.4f} | "
        f"edge_std {valid_metrics['analysis']['edge_weight_std']:.4f} | gap {valid_metrics['analysis']['cross_intra_gap']:.4f} | "
        f"spread {valid_metrics['analysis']['edge_spread']:.4f}"
    )
    print(
        f"  dwf detail      = attn entropy {valid_metrics['analysis']['dwf_attn_entropy']:.4f} | "
        f"attn trace {valid_metrics['analysis']['dwf_attn_trace']:.4f} | "
        f"gate t/v/a {valid_metrics['analysis']['gate_t_mean']:.4f} / {valid_metrics['analysis']['gate_v_mean']:.4f} / {valid_metrics['analysis']['gate_a_mean']:.4f}"
    )
    print(f"  selection score = {score:.4f}")
    if improved:
        print("  status          = improved, best model saved")
    else:
        print(f"  status          = no improvement, early stop counter {wait}/{patience}")


def main() -> None:
    ensure_output_dirs()
    seed = int(os.environ.get("RUN_SEED", str(random.SystemRandom().randint(1, 2**31 - 1))))
    deterministic = os.environ.get("DET_TRAIN", "0") == "1"
    set_seed(seed, deterministic=deterministic)

    # Paper-aligned settings reported in Section 4.2.
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 10
    grad_clip = 1.0

    alpha = 0.05
    beta = 0.05
    gamma = 0.10
    sim_margin = 0.20

    dropout = 0.50
    conv_hidden = 128
    shared_dim = 64
    private_dim = 64
    hyper_hidden = 96
    fusion_dim = 128
    num_experts = 3
    top_k = 1
    num_heads = 4
    hg_layers = 3
    intra_k = 3

    print("[Config] paper-aligned DHM baseline with DWF (remove DTF and TGIB)")
    print(f"[Config] alpha={alpha:.4f} | beta={beta:.4f} | gamma={gamma:.4f} | batch={batch_size} | lr={learning_rate:.1e} | dropout={dropout:.1f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed: {seed} | deterministic: {deterministic}")
    print("device:", device)

    train_dataset, valid_dataset, test_dataset, meta = load_mosei_from_pkl(DATA_FILE)
    print("dataset: MOSEI")
    print(f"train/valid/test: {meta['train_size']}/{meta['valid_size']}/{meta['test_size']}")
    print(f"dims: text={meta['text_dim']}, vision={meta['vision_dim']}, audio={meta['audio_dim']}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = DHMModel(
        text_dim=meta["text_dim"],
        vision_dim=meta["vision_dim"],
        audio_dim=meta["audio_dim"],
        conv_hidden=conv_hidden,
        shared_dim=shared_dim,
        private_dim=private_dim,
        hyper_hidden=hyper_hidden,
        fusion_dim=fusion_dim,
        num_experts=num_experts,
        top_k=top_k,
        num_heads=num_heads,
        hg_layers=hg_layers,
        intra_k=intra_k,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_score = float("inf")
    best_mae = float("inf")
    wait = 0

    history = {
        "train_total_loss": [],
        "valid_total_loss": [],
        "valid_mae": [],
        "valid_corr": [],
        "valid_acc2_posneg": [],
        "valid_f1_posneg": [],
        "valid_acc7": [],
        "cross_edge_weight_mean": [],
        "intra_edge_weight_mean": [],
        "edge_weight_std": [],
        "cross_intra_gap": [],
        "dwf_attn_entropy": [],
        "dwf_attn_trace": [],
    }

    for epoch in range(num_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sim_margin=sim_margin,
            grad_clip=grad_clip,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sim_margin=sim_margin,
        )
        score = model_selection_score(valid_metrics)
        scheduler.step(valid_metrics["MAE"])

        history["train_total_loss"].append(train_stats["train_total_loss"])
        history["valid_total_loss"].append(valid_metrics["total_loss"])
        history["valid_mae"].append(valid_metrics["MAE"])
        history["valid_corr"].append(valid_metrics["Corr"])
        history["valid_acc2_posneg"].append(valid_metrics["Acc2_posneg"])
        history["valid_f1_posneg"].append(valid_metrics["F1_posneg"])
        history["valid_acc7"].append(valid_metrics["Acc7"])
        history["cross_edge_weight_mean"].append(valid_metrics["analysis"]["cross_edge_weight_mean"])
        history["intra_edge_weight_mean"].append(valid_metrics["analysis"]["intra_edge_weight_mean"])
        history["edge_weight_std"].append(valid_metrics["analysis"]["edge_weight_std"])
        history["cross_intra_gap"].append(valid_metrics["analysis"]["cross_intra_gap"])
        history["dwf_attn_entropy"].append(valid_metrics["analysis"]["dwf_attn_entropy"])
        history["dwf_attn_trace"].append(valid_metrics["analysis"]["dwf_attn_trace"])

        improved = score < best_score - 1e-4 or valid_metrics["MAE"] < best_mae - 1e-4
        if improved:
            best_score = score
            best_mae = valid_metrics["MAE"]
            wait = 0
            torch.save(model.state_dict(), BEST_MODEL_FILE)
        else:
            wait += 1

        print_epoch_summary(
            epoch=epoch,
            num_epochs=num_epochs,
            lr=optimizer.param_groups[0]["lr"],
            train_stats=train_stats,
            valid_metrics=valid_metrics,
            score=score,
            improved=improved,
            wait=wait,
            patience=patience,
        )

        if wait >= patience:
            print("early stopping triggered")
            break

    print("\nloading best DHM + DWF baseline model...")
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))
    final_valid = evaluate(model, valid_loader, device, alpha=alpha, beta=beta, gamma=gamma, sim_margin=sim_margin)
    final_test = evaluate(model, test_loader, device, alpha=alpha, beta=beta, gamma=gamma, sim_margin=sim_margin)

    print("\n========== Final Validation ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc7", "total_loss", "task_loss", "sim_loss", "recon_loss", "moe_loss"]:
        print(f"{k:<14}: {final_valid[k]:.4f}")

    print("\n========== Final Test ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc7", "total_loss", "task_loss", "sim_loss", "recon_loss", "moe_loss"]:
        print(f"{k:<14}: {final_test[k]:.4f}")

    save_final_test_results(RESULTS_FILE, final_test)
    plot_training_curves(history, save_dir=PLOTS_DIR)


if __name__ == "__main__":
    main()
