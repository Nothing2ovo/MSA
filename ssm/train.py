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
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_dhib_4token_direct_pred.pt")
BEST_MAE_MODEL_FILE = os.path.join(MODEL_DIR, "best_dhib_4token_mae_ref.pt")
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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_model_state(model: torch.nn.Module, path: str) -> None:
    torch.save(unwrap_model(model).state_dict(), path)


def build_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size: int) -> tuple:
    # Kaggle notebooks have a small RAM budget; many persistent workers with
    # deep prefetch queues can keep several large batches resident at once.
    default_workers = 0 if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else 2
    num_workers = max(0, int(os.environ.get("DATALOADER_WORKERS", str(default_workers))))
    pin_memory = torch.cuda.is_available() and os.environ.get("PIN_MEMORY", "1") != "0"
    common_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = os.environ.get("PERSISTENT_WORKERS", "0") == "1"
        common_kwargs["prefetch_factor"] = max(1, int(os.environ.get("PREFETCH_FACTOR", "2")))

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, valid_loader, test_loader, num_workers


def maybe_wrap_dataparallel(model: torch.nn.Module, device: torch.device) -> tuple[torch.nn.Module, str]:
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        return model, f"single GPU cuda:0 (DataParallel disabled; visible GPUs: {gpu_count})"
    return model, "single device"


def model_selection_score(metrics: Dict[str, float]) -> float:
    """
    Regression-oriented plateau score. Smaller is better.
    Primary objective: lower MAE
    Secondary objective: higher Corr
    """
    return float(metrics["MAE"]) - 0.10 * float(metrics["Corr"])


def regression_priority_tuple(metrics: Dict[str, float]) -> tuple:
    """
    Larger tuple is better. Order: -MAE > Corr
    """
    return (
        round(float(-metrics["MAE"]), 6),
        round(float(metrics["Corr"]), 6),
    )


def is_better_checkpoint(current: Dict[str, float], best: Dict[str, float] | None) -> bool:
    if best is None:
        return True

    mae_gap = float(best["MAE"] - current["MAE"])
    corr_gap = float(current["Corr"] - best["Corr"])

    if mae_gap >= 0.0010:
        return True
    if abs(float(current["MAE"] - best["MAE"])) <= 0.0010 and corr_gap >= 0.0010:
        return True

    return regression_priority_tuple(current) > regression_priority_tuple(best)


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
    plt.plot(epochs, history["valid_supcon_loss"], label="valid shared supcon")
    plt.plot(epochs, history["valid_unsupcon_loss"], label="valid shared unsupcon")
    plt.plot(epochs, history["valid_shared_view_gap"], label="valid shared view gap")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Shared Mamba Contrastive Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shared_mamba_contrastive_metrics.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["valid_acc2_posneg"], label="valid Acc2 pos/neg")
    plt.plot(epochs, history["valid_acc5"], label="valid Acc5")
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
    plt.plot(epochs, history["cross_select_mean"], label="cross select mean")
    plt.plot(epochs, history["intra_select_mean"], label="intra select mean")
    plt.plot(epochs, history["select_std"], label="select std")
    plt.plot(epochs, history["cross_intra_gap"], label="cross/intra gap")
    plt.plot(epochs, history["token_weight_shared"], label="token weight shared")
    plt.plot(epochs, history["token_weight_text"], label="token weight text")
    plt.plot(epochs, history["token_weight_vision"], label="token weight vision")
    plt.plot(epochs, history["token_weight_audio"], label="token weight audio")
    plt.plot(epochs, history["token_entropy"], label="token entropy")
    plt.plot(epochs, history["token_max_weight"], label="token max weight")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Structure and 4-Token Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "structure_token_metrics.png"))
    plt.close()


def save_final_test_results(file_path: str, metrics: Dict[str, float]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis = metrics["analysis"]
    lines = [
        f"[{timestamp}]",
        "========== Final Test (shared-residual decoupling + official-Mamba Shared Selective State Mixer + TMoEs + 4-token direct prediction) ==========",
        f"Test total loss: {metrics['total_loss']:.4f}",
        f"Test task loss : {metrics['task_loss']:.4f}",
        f"Test sim loss  : {metrics['sim_loss']:.4f}",
        f"Test recon loss: {metrics['recon_loss']:.4f}",
        f"Test moe loss  : {metrics['moe_loss']:.4f}",
        f"Test supcon    : {metrics['supcon_loss']:.4f}",
        f"Test unsupcon  : {metrics['unsupcon_loss']:.4f}",
        f"  token reg    : {metrics['token_reg_loss']:.4f}",
        f"  mixer reg    : {metrics['shared_mixer_reg_loss']:.4f}",
        f"  shared aux   : {metrics['shared_aux_loss']:.4f}",
        f"  token entropy: {metrics['token_entropy']:.4f}",
        f"  token maxw   : {metrics['token_max_weight']:.4f}",
        f"  cls5/cls7    : {metrics['acc5_loss']:.4f} / {metrics['acc7_loss']:.4f}",
        f"Test MAE       : {metrics['MAE']:.4f}",
        f"Test Corr      : {metrics['Corr']:.4f}",
        f"Test Acc2      : {metrics['Acc2_nonneg']:.4f} / {metrics['Acc2_posneg']:.4f}",
        f"Test F1        : {metrics['F1_nonneg']:.4f} / {metrics['F1_posneg']:.4f}",
        f"Test Acc5      : {metrics['Acc5']:.4f}",
        f"Test Acc7      : {metrics['Acc7']:.4f}",
        "Analysis:",
        f"  cross select mean     : {analysis['cross_select_mean']:.4f}",
        f"  intra select mean     : {analysis['intra_select_mean']:.4f}",
        f"  modality attn t/v/a     : {analysis['modality_attn_t']:.4f} / {analysis['modality_attn_v']:.4f} / {analysis['modality_attn_a']:.4f}",
        f"  token w s/t/v/a     : {analysis['token_weight_shared']:.4f} / {analysis['token_weight_text']:.4f} / {analysis['token_weight_vision']:.4f} / {analysis['token_weight_audio']:.4f}",
        f"  select std/gap/spread : {analysis['select_std']:.4f} / {analysis['cross_intra_gap']:.4f} / {analysis['select_spread']:.4f}",
        f"  dominance margin    : {analysis['token_dominance_margin']:.4f}",
        f"  attn s->s/t/v/a     : {analysis['attn_shared_to_shared']:.4f} / {analysis['attn_shared_to_text']:.4f} / {analysis['attn_shared_to_vision']:.4f} / {analysis['attn_shared_to_audio']:.4f}",
        f"  gate t/v/a          : {analysis['gate_t_mean']:.4f} / {analysis['gate_v_mean']:.4f} / {analysis['gate_a_mean']:.4f}",
        f"  token floor/peak    : {analysis['token_floor_penalty']:.4f} / {analysis['token_peak_penalty']:.4f}",
        f"  shared view gap     : {analysis['shared_view_gap']:.4f}",
        f"  fused repr norm     : {analysis['fused_repr_norm']:.4f}",
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
        f"moe={train_stats['train_moe_loss']:.4f} supcon={train_stats['train_supcon_loss']:.4f} "
        f"unsupcon={train_stats['train_unsupcon_loss']:.4f} token_reg={train_stats['train_token_reg_loss']:.4f} "
        f"mixer_reg={train_stats['train_shared_mixer_reg_loss']:.4f} cls5={train_stats['train_acc5_loss']:.4f} "
        f"shared_aux={train_stats['train_shared_aux_loss']:.4f} cls7={train_stats['train_acc7_loss']:.4f}"
    )
    print(
        f"  valid MAE/Corr  = {valid_metrics['MAE']:.4f} / {valid_metrics['Corr']:.4f} | "
        f"Acc2={valid_metrics['Acc2_nonneg']:.4f}/{valid_metrics['Acc2_posneg']:.4f} | "
        f"F1={valid_metrics['F1_nonneg']:.4f}/{valid_metrics['F1_posneg']:.4f} | "
        f"Acc5={valid_metrics['Acc5']:.4f} | Acc7={valid_metrics['Acc7']:.4f}"
    )
    print(
        f"  valid losses    = total {valid_metrics['total_loss']:.4f} | task {valid_metrics['task_loss']:.4f} | "
        f"sim {valid_metrics['sim_loss']:.4f} | recon {valid_metrics['recon_loss']:.4f} | "
        f"moe {valid_metrics['moe_loss']:.4f} | supcon {valid_metrics['supcon_loss']:.4f} | "
        f"unsupcon {valid_metrics['unsupcon_loss']:.4f} | mixer {valid_metrics['shared_mixer_reg_loss']:.4f} | shared_aux {valid_metrics['shared_aux_loss']:.4f}"
    )
    print(
        f"  4-token detail  = entropy {valid_metrics['token_entropy']:.4f} | prior-fit {valid_metrics['token_balance']:.4f} | "
        f"maxw {valid_metrics['token_max_weight']:.4f} | dom {valid_metrics['analysis']['token_dominance_margin']:.4f} | "
        f"floor {valid_metrics['analysis']['token_floor_penalty']:.4f} | peak {valid_metrics['analysis']['token_peak_penalty']:.4f}"
    )
    print(
        f"  mixer          = cross/intra {valid_metrics['analysis']['cross_select_mean']:.4f} / {valid_metrics['analysis']['intra_select_mean']:.4f} | "
        f"select_std {valid_metrics['analysis']['select_std']:.4f} | gap {valid_metrics['analysis']['cross_intra_gap']:.4f} | spread {valid_metrics['analysis']['select_spread']:.4f} | "
        f"token s/t/v/a {valid_metrics['analysis']['token_weight_shared']:.4f} / {valid_metrics['analysis']['token_weight_text']:.4f} / {valid_metrics['analysis']['token_weight_vision']:.4f} / {valid_metrics['analysis']['token_weight_audio']:.4f}"
    )
    print(
        f"  shared CL       = sup {valid_metrics['supcon_loss']:.4f} | unsup {valid_metrics['unsupcon_loss']:.4f} | "
        f"view gap {valid_metrics['analysis']['shared_view_gap']:.4f}"
    )
    print(f"  selection score = {score:.4f}")
    if improved:
        print("  status          = improved, regression-priority best model saved")
    else:
        print(f"  status          = no improvement, early stop counter {wait}/{patience}")


def main() -> None:
    ensure_output_dirs()
    seed = int(os.environ.get("RUN_SEED", str(random.SystemRandom().randint(1, 2**31 - 1))))
    deterministic = os.environ.get("DET_TRAIN", "0") == "1"
    set_seed(seed, deterministic=deterministic)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    num_epochs = int(os.environ.get("EPOCHS", "50"))
    learning_rate = float(os.environ.get("LR", "1e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    patience = int(os.environ.get("PATIENCE", "10"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

    sim_weight = float(os.environ.get("SIM_WEIGHT", "0.02"))
    recon_weight = float(os.environ.get("RECON_WEIGHT", "0.05"))
    moe_weight = float(os.environ.get("MOE_WEIGHT", "0.10"))
    supcon_weight = float(os.environ.get("SUPCON_WEIGHT", "0.02"))
    unsupcon_weight = float(os.environ.get("UNSUPCON_WEIGHT", "0.01"))
    sim_margin = float(os.environ.get("SIM_MARGIN", "0.20"))

    token_reg_weight = float(os.environ.get("TOKEN_REG_WEIGHT", "0.04"))
    shared_mixer_reg_weight = float(os.environ.get("MIXER_REG_WEIGHT", "0.030"))
    shared_aux_weight = float(os.environ.get("SHARED_AUX_WEIGHT", "0.10"))
    acc5_loss_weight = float(os.environ.get("ACC5_LOSS_WEIGHT", "0.10"))
    acc7_loss_weight = float(os.environ.get("ACC7_LOSS_WEIGHT", "0.06"))

    dropout = 0.50
    conv_hidden = 128
    shared_dim = 64
    private_dim = 64
    shared_mixer_hidden = 96
    fusion_dim = 128
    num_experts = 3
    top_k = 1
    num_heads = 4
    mixer_layers = 2
    mamba_state_dim = 16
    shared_drop_rate = 0.15

    print("[Config] shared-residual decoupling / Shared Selective State Mixer(official Mamba) / TMoEs / 4-token direct prediction")
    print(
        f"[Config] sim={sim_weight:.3f} recon={recon_weight:.3f} moe={moe_weight:.3f} "
        f"supcon={supcon_weight:.3f} unsupcon={unsupcon_weight:.3f} "
        f"token_reg={token_reg_weight:.3f} mixer_reg={shared_mixer_reg_weight:.3f} shared_aux={shared_aux_weight:.3f} "
        f"shared_drop={shared_drop_rate:.2f} mamba_state={mamba_state_dim} | ckpt=MAE>Corr"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_amp = device.type == "cuda"
    print(f"seed: {seed} | deterministic: {deterministic}")
    print("device:", device)
    print("gpu_count:", gpu_count)
    print("amp:", use_amp)

    train_dataset, valid_dataset, test_dataset, meta = load_mosei_from_pkl(DATA_FILE)
    print("dataset: MOSEI")
    print(f"train/valid/test: {meta['train_size']}/{meta['valid_size']}/{meta['test_size']}")
    print(f"dims: text={meta['text_dim']}, vision={meta['vision_dim']}, audio={meta['audio_dim']}")

    train_loader, valid_loader, test_loader, num_workers = build_dataloaders(
        train_dataset, valid_dataset, test_dataset, batch_size=batch_size
    )
    print("num_workers:", num_workers)

    model = DHMModel(
        text_dim=meta["text_dim"],
        vision_dim=meta["vision_dim"],
        audio_dim=meta["audio_dim"],
        conv_hidden=conv_hidden,
        shared_dim=shared_dim,
        private_dim=private_dim,
        shared_mixer_hidden=shared_mixer_hidden,
        fusion_dim=fusion_dim,
        num_experts=num_experts,
        top_k=top_k,
        num_heads=num_heads,
        mixer_layers=mixer_layers,
        mamba_state_dim=mamba_state_dim,
        mamba_conv_kernel=3,
        mamba_expand=2,
        dropout=dropout,
        shared_drop_rate=shared_drop_rate,
        shared_residual_scale=0.20,
        use_shared_cross_attention=True,
        require_official_mamba=True,
    ).to(device)
    model, parallel_mode = maybe_wrap_dataparallel(model, device)
    print("parallel_mode:", parallel_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    best_score = float("inf")
    best_mae_ref = float("inf")
    best_reg_metrics = None
    wait = 0

    history = {
        "train_total_loss": [],
        "valid_total_loss": [],
        "valid_mae": [],
        "valid_corr": [],
        "valid_supcon_loss": [],
        "valid_unsupcon_loss": [],
        "valid_shared_view_gap": [],
        "valid_acc2_posneg": [],
        "valid_f1_posneg": [],
        "valid_acc5": [],
        "valid_acc7": [],
        "cross_select_mean": [],
        "intra_select_mean": [],
        "token_weight_shared": [],
        "token_weight_text": [],
        "token_weight_vision": [],
        "token_weight_audio": [],
        "select_std": [],
        "cross_intra_gap": [],
        "token_entropy": [],
        "token_max_weight": [],
    }

    for epoch in range(num_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sim_weight=sim_weight,
            recon_weight=recon_weight,
            moe_weight=moe_weight,
            supcon_weight=supcon_weight,
            unsupcon_weight=unsupcon_weight,
            sim_margin=sim_margin,
            grad_clip=grad_clip,
            token_reg_weight=token_reg_weight,
            shared_mixer_reg_weight=shared_mixer_reg_weight,
            shared_aux_weight=shared_aux_weight,
            acc5_loss_weight=acc5_loss_weight,
            acc7_loss_weight=acc7_loss_weight,
            scaler=scaler,
            use_amp=use_amp,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            sim_weight=sim_weight,
            recon_weight=recon_weight,
            moe_weight=moe_weight,
            supcon_weight=supcon_weight,
            unsupcon_weight=unsupcon_weight,
            sim_margin=sim_margin,
            token_reg_weight=token_reg_weight,
            shared_mixer_reg_weight=shared_mixer_reg_weight,
            shared_aux_weight=shared_aux_weight,
            acc5_loss_weight=acc5_loss_weight,
            acc7_loss_weight=acc7_loss_weight,
            use_amp=use_amp,
        )
        score = model_selection_score(valid_metrics)
        scheduler.step(score)

        history["train_total_loss"].append(train_stats["train_total_loss"])
        history["valid_total_loss"].append(valid_metrics["total_loss"])
        history["valid_mae"].append(valid_metrics["MAE"])
        history["valid_corr"].append(valid_metrics["Corr"])
        history["valid_supcon_loss"].append(valid_metrics["supcon_loss"])
        history["valid_unsupcon_loss"].append(valid_metrics["unsupcon_loss"])
        history["valid_shared_view_gap"].append(valid_metrics["analysis"]["shared_view_gap"])
        history["valid_acc2_posneg"].append(valid_metrics["Acc2_posneg"])
        history["valid_f1_posneg"].append(valid_metrics["F1_posneg"])
        history["valid_acc5"].append(valid_metrics["Acc5"])
        history["valid_acc7"].append(valid_metrics["Acc7"])
        history["cross_select_mean"].append(valid_metrics["analysis"]["cross_select_mean"])
        history["intra_select_mean"].append(valid_metrics["analysis"]["intra_select_mean"])
        history["token_weight_shared"].append(valid_metrics["analysis"]["token_weight_shared"])
        history["token_weight_text"].append(valid_metrics["analysis"]["token_weight_text"])
        history["token_weight_vision"].append(valid_metrics["analysis"]["token_weight_vision"])
        history["token_weight_audio"].append(valid_metrics["analysis"]["token_weight_audio"])
        history["select_std"].append(valid_metrics["analysis"]["select_std"])
        history["cross_intra_gap"].append(valid_metrics["analysis"]["cross_intra_gap"])
        history["token_entropy"].append(valid_metrics["token_entropy"])
        history["token_max_weight"].append(valid_metrics["token_max_weight"])

        improved = is_better_checkpoint(valid_metrics, best_reg_metrics)
        if improved:
            best_reg_metrics = {
                "MAE": float(valid_metrics["MAE"]),
                "Corr": float(valid_metrics["Corr"]),
            }
            best_score = score
            wait = 0
            save_model_state(model, BEST_MODEL_FILE)
        else:
            wait += 1

        if valid_metrics["MAE"] < best_mae_ref - 1e-4:
            best_mae_ref = float(valid_metrics["MAE"])
            save_model_state(model, BEST_MAE_MODEL_FILE)

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

    print("\nloading best regression-priority model...")
    unwrap_model(model).load_state_dict(torch.load(BEST_MODEL_FILE, map_location=device))
    final_valid = evaluate(
        model, valid_loader, device,
        sim_weight=sim_weight, recon_weight=recon_weight, moe_weight=moe_weight,
        supcon_weight=supcon_weight, unsupcon_weight=unsupcon_weight,
        sim_margin=sim_margin,
        token_reg_weight=token_reg_weight,
        shared_mixer_reg_weight=shared_mixer_reg_weight,
        shared_aux_weight=shared_aux_weight,
        acc5_loss_weight=acc5_loss_weight,
        acc7_loss_weight=acc7_loss_weight,
        use_amp=use_amp,
    )
    final_test = evaluate(
        model, test_loader, device,
        sim_weight=sim_weight, recon_weight=recon_weight, moe_weight=moe_weight,
        supcon_weight=supcon_weight, unsupcon_weight=unsupcon_weight,
        sim_margin=sim_margin,
        token_reg_weight=token_reg_weight,
        shared_mixer_reg_weight=shared_mixer_reg_weight,
        shared_aux_weight=shared_aux_weight,
        acc5_loss_weight=acc5_loss_weight,
        acc7_loss_weight=acc7_loss_weight,
        use_amp=use_amp,
    )

    print("\n========== Final Validation ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc5", "Acc7", "total_loss", "supcon_loss", "unsupcon_loss", "token_reg_loss", "shared_mixer_reg_loss", "shared_aux_loss", "acc5_loss", "acc7_loss"]:
        print(f"{k:<14}: {final_valid[k]:.4f}")

    print("\n========== Final Test ==========")
    for k in ["MAE", "Corr", "Acc2_nonneg", "F1_nonneg", "Acc2_posneg", "F1_posneg", "Acc5", "Acc7", "total_loss", "supcon_loss", "unsupcon_loss", "token_reg_loss", "shared_mixer_reg_loss", "shared_aux_loss", "acc5_loss", "acc7_loss"]:
        print(f"{k:<14}: {final_test[k]:.4f}")

    save_final_test_results(RESULTS_FILE, final_test)
    plot_training_curves(history, save_dir=PLOTS_DIR)


if __name__ == "__main__":
    main()
