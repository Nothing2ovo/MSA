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
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_dhm_mosei_tgib.pt")
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


def compute_ib_weight_schedule(epoch: int, warmup_epochs: int, ramp_epochs: int, max_delta: float) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_delta
    progress = min(1.0, (epoch - warmup_epochs + 1) / float(ramp_epochs))
    return max_delta * progress


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
    plt.plot(epochs, history["valid_mmib_loss"], label="valid TGIB loss")
    plt.plot(epochs, history["valid_mmib_mi"], label="valid TGIB MI")
    plt.plot(epochs, history["valid_filter_shift"], label="valid TGIB shift")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Transformer-guided IB Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tgib_metrics.png"))
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
    plt.plot(epochs, history["token_weight_shared"], label="token weight shared")
    plt.plot(epochs, history["token_weight_text"], label="token weight text")
    plt.plot(epochs, history["token_weight_vision"], label="token weight vision")
    plt.plot(epochs, history["token_weight_audio"], label="token weight audio")
    plt.plot(epochs, history["token_entropy"], label="token entropy")
    plt.plot(epochs, history["token_max_weight"], label="token max weight")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Structure and Token Weight Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "structure_token_metrics.png"))
    plt.close()


def save_final_test_results(file_path: str, metrics: Dict[str, float]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis = metrics["analysis"]
    lines = [
        f"[{timestamp}]",
        "========== Final Test (mosei dhm + balanced hg / relaxed 4-token / mild tgib+) ==========",
        f"Test total loss: {metrics['total_loss']:.4f}",
        f"Test task loss : {metrics['task_loss']:.4f}",
        f"Test sim loss  : {metrics['sim_loss']:.4f}",
        f"Test recon loss: {metrics['recon_loss']:.4f}",
        f"Test moe loss  : {metrics['moe_loss']:.4f}",
        f"Test tgib loss : {metrics['mmib_loss']:.4f}",
        f"  tgib mi      : {metrics['mmib_mi']:.4f}",
        f"  tgib mae     : {metrics['mmib_mae']:.4f}",
        f"  tgib kl      : {metrics['mmib_kl']:.6f}",
        f"  token reg    : {metrics['token_reg_loss']:.4f}",
        f"  hyper reg    : {metrics['hypergraph_reg_loss']:.4f}",
        f"  token entropy: {metrics['token_entropy']:.4f}",
        f"  token maxw   : {metrics['token_max_weight']:.4f}",
        f"Test MAE       : {metrics['MAE']:.4f}",
        f"Test Corr      : {metrics['Corr']:.4f}",
        f"Test Acc2      : {metrics['Acc2_nonneg']:.4f} / {metrics['Acc2_posneg']:.4f}",
        f"Test F1        : {metrics['F1_nonneg']:.4f} / {metrics['F1_posneg']:.4f}",
        f"Test Acc7      : {metrics['Acc7']:.4f}",
        "Analysis:",
        f"  cross edge mean     : {analysis['cross_edge_weight_mean']:.4f}",
        f"  intra edge mean     : {analysis['intra_edge_weight_mean']:.4f}",
        f"  node attn t/v/a     : {analysis['node_attn_t']:.4f} / {analysis['node_attn_v']:.4f} / {analysis['node_attn_a']:.4f}",
        f"  token w s/t/v/a     : {analysis['token_weight_shared']:.4f} / {analysis['token_weight_text']:.4f} / {analysis['token_weight_vision']:.4f} / {analysis['token_weight_audio']:.4f}",
        f"  edge std/gap/spread : {analysis['edge_weight_std']:.4f} / {analysis['cross_intra_gap']:.4f} / {analysis['edge_spread']:.4f}",
        f"  dominance margin    : {analysis['token_dominance_margin']:.4f}",
        f"  attn s->s/t/v/a     : {analysis['attn_shared_to_shared']:.4f} / {analysis['attn_shared_to_text']:.4f} / {analysis['attn_shared_to_vision']:.4f} / {analysis['attn_shared_to_audio']:.4f}",
        f"  gate t/v/a          : {analysis['gate_t_mean']:.4f} / {analysis['gate_v_mean']:.4f} / {analysis['gate_a_mean']:.4f}",
        f"  token floor/peak    : {analysis['token_floor_penalty']:.4f} / {analysis['token_peak_penalty']:.4f}",
        f"  latent std mean     : {analysis['filtered_std_mean']:.4f}",
        f"  tgib shift          : {analysis['filter_shift']:.4f}",
        "",
    ]
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    lr: float,
    ib_delta: float,
    train_stats: Dict[str, float],
    valid_metrics: Dict[str, float],
    score: float,
    improved: bool,
    wait: int,
    patience: int,
) -> None:
    print(f"\n[Epoch {epoch + 1:02d}/{num_epochs}]")
    print(f"  lr              = {lr:.6f}")
    print(f"  tgib delta      = {ib_delta:.4f}")
    print(
        f"  train total/task= {train_stats['train_total_loss']:.4f} / {train_stats['train_task_loss']:.4f} | "
        f"sim={train_stats['train_sim_loss']:.4f} recon={train_stats['train_recon_loss']:.4f} "
        f"moe={train_stats['train_moe_loss']:.4f} token_reg={train_stats['train_token_reg_loss']:.4f} "
        f"hyper_reg={train_stats['train_hypergraph_reg_loss']:.4f} tgib={train_stats['train_mmib_loss']:.4f}"
    )
    print(
        f"  valid MAE/Corr  = {valid_metrics['MAE']:.4f} / {valid_metrics['Corr']:.4f} | "
        f"Acc2={valid_metrics['Acc2_nonneg']:.4f}/{valid_metrics['Acc2_posneg']:.4f} | "
        f"F1={valid_metrics['F1_nonneg']:.4f}/{valid_metrics['F1_posneg']:.4f} | Acc7={valid_metrics['Acc7']:.4f}"
    )
    print(
        f"  valid losses    = total {valid_metrics['total_loss']:.4f} | task {valid_metrics['task_loss']:.4f} | "
        f"sim {valid_metrics['sim_loss']:.4f} | recon {valid_metrics['recon_loss']:.4f} | "
        f"moe {valid_metrics['moe_loss']:.4f} | hg {valid_metrics['hypergraph_reg_loss']:.4f} | tgib {valid_metrics['mmib_loss']:.4f}"
    )
    print(
        f"  tgib detail     = mi {valid_metrics['mmib_mi']:.4f} | mae {valid_metrics['mmib_mae']:.4f} | "
        f"kl {valid_metrics['mmib_kl']:.6f} | shift {valid_metrics['analysis']['filter_shift']:.4f}"
    )
    print(
        f"  token detail    = entropy {valid_metrics['token_entropy']:.4f} | prior-fit {valid_metrics['token_balance']:.4f} | "
        f"maxw {valid_metrics['token_max_weight']:.4f} | dom {valid_metrics['analysis']['token_dominance_margin']:.4f} | "
        f"floor {valid_metrics['analysis']['token_floor_penalty']:.4f} | peak {valid_metrics['analysis']['token_peak_penalty']:.4f}"
    )
    print(
        f"  structure       = cross/intra {valid_metrics['analysis']['cross_edge_weight_mean']:.4f} / {valid_metrics['analysis']['intra_edge_weight_mean']:.4f} | "
        f"edge_std {valid_metrics['analysis']['edge_weight_std']:.4f} | gap {valid_metrics['analysis']['cross_intra_gap']:.4f} | spread {valid_metrics['analysis']['edge_spread']:.4f} | "
        f"token s/t/v/a {valid_metrics['analysis']['token_weight_shared']:.4f} / {valid_metrics['analysis']['token_weight_text']:.4f} / {valid_metrics['analysis']['token_weight_vision']:.4f} / {valid_metrics['analysis']['token_weight_audio']:.4f}"
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

    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 10
    grad_clip = 1.0

    alpha = 0.05
    beta = 0.05
    gamma = 0.10
    delta = 0.0080
    sim_margin = 0.20

    renyi_alpha = 1.9
    renyi_rank_k = 10
    mmib_mae_weight = 0.5
    mmib_kl_weight = 7.5e-5
    token_reg_weight = 0.04
    hypergraph_reg_weight = 0.055

    ib_warmup_epochs = 6
    ib_ramp_epochs = 8

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

    print("[Config] stronger-hypergraph / slightly-relaxed-4token / mildly-raised-TGIB")
    print(f"[Config] delta={delta:.4f} | mmib_kl={mmib_kl_weight:.6f} | hyper_reg_w={hypergraph_reg_weight:.4f}")

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
        "valid_mmib_loss": [],
        "valid_mmib_mi": [],
        "valid_filter_shift": [],
        "valid_acc2_posneg": [],
        "valid_f1_posneg": [],
        "valid_acc7": [],
        "cross_edge_weight_mean": [],
        "intra_edge_weight_mean": [],
        "token_weight_shared": [],
        "token_weight_text": [],
        "token_weight_vision": [],
        "token_weight_audio": [],
        "edge_weight_std": [],
        "cross_intra_gap": [],
        "token_entropy": [],
        "token_max_weight": [],
    }

    for epoch in range(num_epochs):
        current_delta = compute_ib_weight_schedule(epoch=epoch, warmup_epochs=ib_warmup_epochs, ramp_epochs=ib_ramp_epochs, max_delta=delta)

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=current_delta,
            sim_margin=sim_margin,
            grad_clip=grad_clip,
            renyi_alpha=renyi_alpha,
            renyi_rank_k=renyi_rank_k,
            mmib_mae_weight=mmib_mae_weight,
            mmib_kl_weight=mmib_kl_weight,
            token_reg_weight=token_reg_weight,
            hypergraph_reg_weight=hypergraph_reg_weight,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=current_delta,
            sim_margin=sim_margin,
            renyi_alpha=renyi_alpha,
            renyi_rank_k=renyi_rank_k,
            mmib_mae_weight=mmib_mae_weight,
            mmib_kl_weight=mmib_kl_weight,
            token_reg_weight=token_reg_weight,
            hypergraph_reg_weight=hypergraph_reg_weight,
        )
        score = model_selection_score(valid_metrics)
        scheduler.step(valid_metrics["MAE"])

        history["train_total_loss"].append(train_stats["train_total_loss"])
        history["valid_total_loss"].append(valid_metrics["total_loss"])
        history["valid_mae"].append(valid_metrics["MAE"])
        history["valid_corr"].append(valid_metrics["Corr"])
        history["valid_mmib_loss"].append(valid_metrics["mmib_loss"])
        history["valid_mmib_mi"].append(valid_metrics["mmib_mi"])
        history["valid_filter_shift"].append(valid_metrics["analysis"]["filter_shift"])
        history["valid_acc2_posneg"].append(valid_metrics["Acc2_posneg"])
        history["valid_f1_posneg"].append(valid_metrics["F1_posneg"])
        history["valid_acc7"].append(valid_metrics["Acc7"])
        history["cross_edge_weight_mean"].append(valid_metrics["analysis"]["cross_edge_weight_mean"])
        history["intra_edge_weight_mean"].append(valid_metrics["analysis"]["intra_edge_weight_mean"])
        history["token_weight_shared"].append(valid_metrics["analysis"]["token_weight_shared"])
        history["token_weight_text"].append(valid_metrics["analysis"]["token_weight_text"])
        history["token_weight_vision"].append(valid_metrics["analysis"]["token_weight_vision"])
        history["token_weight_audio"].append(valid_metrics["analysis"]["token_weight_audio"])
        history["edge_weight_std"].append(valid_metrics["analysis"]["edge_weight_std"])
        history["cross_intra_gap"].append(valid_metrics["analysis"]["cross_intra_gap"])
        history["token_entropy"].append(valid_metrics["token_entropy"])
        history["token_max_weight"].append(valid_metrics["token_max_weight"])

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
            ib_delta=current_delta,
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

    print("\nloading best DHM + TGIB model...")
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))
    final_valid = evaluate(
        model, valid_loader, device,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        sim_margin=sim_margin, renyi_alpha=renyi_alpha, renyi_rank_k=renyi_rank_k,
        mmib_mae_weight=mmib_mae_weight, mmib_kl_weight=mmib_kl_weight,
        token_reg_weight=token_reg_weight,
        hypergraph_reg_weight=hypergraph_reg_weight,
    )
    final_test = evaluate(
        model, test_loader, device,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        sim_margin=sim_margin, renyi_alpha=renyi_alpha, renyi_rank_k=renyi_rank_k,
        mmib_mae_weight=mmib_mae_weight, mmib_kl_weight=mmib_kl_weight,
        token_reg_weight=token_reg_weight,
        hypergraph_reg_weight=hypergraph_reg_weight,
    )

    print("\n========== Final Validation ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc7", "total_loss", "mmib_loss", "token_reg_loss", "hypergraph_reg_loss"]:
        print(f"{k:<14}: {final_valid[k]:.4f}")

    print("\n========== Final Test ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc7", "total_loss", "mmib_loss", "token_reg_loss", "hypergraph_reg_loss"]:
        print(f"{k:<14}: {final_test[k]:.4f}")

    save_final_test_results(RESULTS_FILE, final_test)
    plot_training_curves(history, save_dir=PLOTS_DIR)


if __name__ == "__main__":
    main()
