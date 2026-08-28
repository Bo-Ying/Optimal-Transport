"""Train lightweight CoordCNN autoencoders for normalized football heatmaps."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


HEIGHT = 72
WIDTH = 112
EPS = 1e-8
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "DATA" / "dataset_player_KDE_clean_enriched.pkl"
SPLIT_YEARS = {
    "train": tuple(range(2015, 2023)),
    "dev": (2023,),
    "val": (2024,),
    "test": (2025,),
}
STAT_NAMES = (
    "centroid_x",
    "centroid_y",
    "spread_x",
    "spread_y",
    "covariance_xy",
    "entropy",
    "left_lane_share",
    "center_lane_share",
    "right_lane_share",
    "defensive_third_share",
    "middle_third_share",
    "attacking_third_share",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2), encoding="utf-8")


def make_coords(height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def season_start_year(dates: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="raise")
    return (parsed.dt.year - (parsed.dt.month < 7).astype(np.int16)).to_numpy(np.int16)


class HeatmapDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        row_indices: np.ndarray,
        target: str,
        player_ids: np.ndarray | None = None,
        team_sides: np.ndarray | None = None,
    ) -> None:
        self.frame = frame
        self.row_indices = row_indices
        self.target = target
        self.player_ids = player_ids
        self.team_sides = team_sides

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_index = int(self.row_indices[index])
        if self.target == "player":
            player_id = int(self.player_ids[index])
            heatmap = self.frame.at[row_index, "Player_Heatmap"][player_id]
        else:
            side = int(self.team_sides[index])
            column = "Team_Heatmap(Home)" if side == 0 else "Team_Heatmap(Away)"
            heatmap = self.frame.at[row_index, column]

        array = np.asarray(heatmap, dtype=np.float32)
        mass = float(array.sum(dtype=np.float64))
        density = array / max(mass, EPS)
        return torch.from_numpy(density).unsqueeze(0), torch.tensor(mass, dtype=torch.float32)


def build_datasets(frame: pd.DataFrame, target: str) -> tuple[dict[str, HeatmapDataset], dict[str, Any]]:
    years = season_start_year(frame["Match_Date"])
    entries: dict[str, dict[str, list[int]]] = {
        split: {"rows": [], "keys": []} for split in SPLIT_YEARS
    }
    skipped = defaultdict(int)
    masses = defaultdict(list)

    year_to_split = {year: split for split, year_values in SPLIT_YEARS.items() for year in year_values}
    iterator = tqdm(range(len(frame)), desc=f"Indexing {target} heatmaps", unit="match")
    for row_index in iterator:
        split = year_to_split.get(int(years[row_index]))
        if split is None:
            continue
        if target == "player":
            values = frame.at[row_index, "Player_Heatmap"]
            for player_id, heatmap in values.items():
                mass = float(np.asarray(heatmap, dtype=np.float32).sum(dtype=np.float64))
                if not np.isfinite(mass) or mass <= EPS:
                    skipped[split] += 1
                    continue
                entries[split]["rows"].append(row_index)
                entries[split]["keys"].append(int(player_id))
                masses[split].append(mass)
        else:
            for side, column in enumerate(("Team_Heatmap(Home)", "Team_Heatmap(Away)")):
                heatmap = frame.at[row_index, column]
                mass = float(np.asarray(heatmap, dtype=np.float32).sum(dtype=np.float64))
                if not np.isfinite(mass) or mass <= EPS:
                    skipped[split] += 1
                    continue
                entries[split]["rows"].append(row_index)
                entries[split]["keys"].append(side)
                masses[split].append(mass)

    datasets: dict[str, HeatmapDataset] = {}
    summary: dict[str, Any] = {"target": target, "splits": {}}
    for split in SPLIT_YEARS:
        rows = np.asarray(entries[split]["rows"], dtype=np.int32)
        keys = np.asarray(entries[split]["keys"], dtype=np.int64 if target == "player" else np.int8)
        kwargs = {"player_ids": keys} if target == "player" else {"team_sides": keys}
        datasets[split] = HeatmapDataset(frame, rows, target, **kwargs)
        split_masses = np.asarray(masses[split], dtype=np.float32)
        summary["splits"][split] = {
            "season_start_years": list(SPLIT_YEARS[split]),
            "num_samples": int(len(rows)),
            "num_zero_or_invalid_mass_skipped": int(skipped[split]),
            "mean_original_mass": float(split_masses.mean()) if len(split_masses) else None,
            "median_original_mass": float(np.median(split_masses)) if len(split_masses) else None,
        }
    return datasets, summary


class CoordConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, 3, stride=2, padding=1)
        groups = 8 if out_channels % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = make_coords(x.shape[-2], x.shape[-1], x.device).expand(x.shape[0], -1, -1, -1)
        return F.silu(self.norm(self.conv(torch.cat((x, coords), dim=1))))


class CoordCNNAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            CoordConvBlock(1, 16),
            CoordConvBlock(16, 24),
            CoordConvBlock(24, 32),
            CoordConvBlock(32, 48),
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.stats_head = nn.Linear(latent_dim, len(STAT_NAMES))

        # The first decoder layer is algebraically identical to an MLP on [z, x, y],
        # but avoids explicitly materializing z at every pitch cell.
        self.z_to_hidden = nn.Linear(latent_dim, 64)
        self.coord_to_hidden = nn.Conv2d(2, 64, 1)
        self.decoder_hidden = nn.Conv2d(64, 32, 1)
        self.decoder_logits = nn.Conv2d(32, 1, 1)
        self.register_buffer("decoder_coords", make_coords(HEIGHT, WIDTH), persistent=False)

    def freeze_for_prediction(self) -> "CoordCNNAutoencoder":
        """Freeze the pretrained spatial representation for downstream prediction."""
        self.requires_grad_(False)
        self.eval()
        return self

    def encode(self, density: torch.Tensor) -> torch.Tensor:
        features = self.encoder_blocks(density)
        pooled = torch.cat(
            (F.adaptive_avg_pool2d(features, 1), F.adaptive_max_pool2d(features, 1)), dim=1
        ).flatten(1)
        return self.encoder_head(pooled)

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        coords = self.decoder_coords.expand(latent.shape[0], -1, -1, -1)
        hidden = F.silu(self.z_to_hidden(latent)[:, :, None, None] + self.coord_to_hidden(coords))
        return self.decoder_logits(F.silu(self.decoder_hidden(hidden)))

    def forward(self, density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(density)
        logits = self.decode_logits(latent)
        prediction = torch.softmax(logits.float().flatten(1), dim=1).view_as(logits)
        return prediction, self.stats_head(latent), latent


def density_stats(density: torch.Tensor) -> torch.Tensor:
    batch = density.shape[0]
    coords = make_coords(HEIGHT, WIDTH, density.device)
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    cx = (density * x).sum((1, 2, 3))
    cy = (density * y).sum((1, 2, 3))
    dx = x - cx.view(batch, 1, 1, 1)
    dy = y - cy.view(batch, 1, 1, 1)
    var_x = (density * dx.square()).sum((1, 2, 3)).clamp_min(0)
    var_y = (density * dy.square()).sum((1, 2, 3)).clamp_min(0)
    cov = (density * dx * dy).sum((1, 2, 3))
    entropy = -(density.clamp_min(EPS) * density.clamp_min(EPS).log()).sum((1, 2, 3)) / math.log(HEIGHT * WIDTH)

    x1, x2 = WIDTH // 3, 2 * WIDTH // 3
    y1, y2 = HEIGHT // 3, 2 * HEIGHT // 3
    lanes = (
        density[:, :, :, :x1].sum((1, 2, 3)),
        density[:, :, :, x1:x2].sum((1, 2, 3)),
        density[:, :, :, x2:].sum((1, 2, 3)),
    )
    thirds = (
        density[:, :, :y1, :].sum((1, 2, 3)),
        density[:, :, y1:y2, :].sum((1, 2, 3)),
        density[:, :, y2:, :].sum((1, 2, 3)),
    )
    return torch.stack((cx, cy, var_x.sqrt(), var_y.sqrt(), cov, entropy, *lanes, *thirds), dim=1)


def marginal_wasserstein(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target_x = target.sum(dim=2).squeeze(1)
    pred_x = prediction.sum(dim=2).squeeze(1)
    target_y = target.sum(dim=3).squeeze(1)
    pred_y = prediction.sum(dim=3).squeeze(1)
    wx = (target_x.cumsum(1)[:, :-1] - pred_x.cumsum(1)[:, :-1]).abs().sum(1) * (2.0 / (WIDTH - 1))
    wy = (target_y.cumsum(1)[:, :-1] - pred_y.cumsum(1)[:, :-1]).abs().sum(1) * (2.0 / (HEIGHT - 1))
    return 0.5 * (wx + wy)


def batch_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    target_flat = target.flatten(1).float().clamp_min(EPS)
    pred_flat = prediction.flatten(1).float().clamp_min(EPS)
    log_target = target_flat.log()
    log_pred = pred_flat.log()
    midpoint = 0.5 * (target_flat + pred_flat)
    log_midpoint = midpoint.log()
    ce = -(target_flat * log_pred).sum(1)
    kl = (target_flat * (log_target - log_pred)).sum(1)
    js = 0.5 * (
        (target_flat * (log_target - log_midpoint)).sum(1)
        + (pred_flat * (log_pred - log_midpoint)).sum(1)
    )
    diff = target_flat - pred_flat
    cosine = F.cosine_similarity(target_flat, pred_flat, dim=1)

    target_stats = density_stats(target.float())
    pred_stats = density_stats(prediction.float())
    stat_diff = (target_stats - pred_stats).abs()
    dx = stat_diff[:, 0]
    dy = stat_diff[:, 1]
    centroid_pitch = torch.sqrt(dx.square() + dy.square())
    centroid_cells = torch.sqrt((dx * (WIDTH - 1) / 2).square() + (dy * (HEIGHT - 1) / 2).square())
    spread = 0.5 * (stat_diff[:, 2] + stat_diff[:, 3])
    zone = stat_diff[:, 6:12].mean(1)
    entropy = stat_diff[:, 5]
    composite = js + 0.10 * centroid_pitch + 0.10 * spread + 0.10 * zone + 0.05 * entropy
    return {
        "cross_entropy_density": ce,
        "kl_divergence": kl,
        "jensen_shannon_divergence": js,
        "pixel_mae_scaled": diff.abs().mean(1) * (HEIGHT * WIDTH),
        "pixel_rmse_scaled": torch.sqrt(diff.square().sum(1)),
        "cosine_similarity": cosine,
        "marginal_wasserstein_1": marginal_wasserstein(target.float(), prediction.float()),
        "centroid_error_x": dx,
        "centroid_error_y": dy,
        "centroid_error_l2_cells": centroid_cells,
        "centroid_error_l2_pitch_norm": centroid_pitch,
        "spread_x_abs_error": stat_diff[:, 2],
        "spread_y_abs_error": stat_diff[:, 3],
        "covariance_xy_abs_error": stat_diff[:, 4],
        "entropy_abs_error": entropy,
        "zone_share_mae": zone,
        "composite_score": composite,
    }


def reconstruction_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    predicted_stats: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_flat = target.flatten(1).float().clamp_min(EPS)
    pred_flat = prediction.flatten(1).float().clamp_min(EPS)
    ce = -(target_flat * pred_flat.log()).sum(1).mean()
    midpoint = 0.5 * (target_flat + pred_flat)
    js = 0.5 * (
        (target_flat * (target_flat.log() - midpoint.log())).sum(1)
        + (pred_flat * (pred_flat.log() - midpoint.log())).sum(1)
    ).mean()
    l1 = (target_flat - pred_flat).abs().mean() * (HEIGHT * WIDTH)
    target_stats = density_stats(target.float())
    stats = F.smooth_l1_loss(predicted_stats.float(), target_stats)
    wasserstein = marginal_wasserstein(target.float(), prediction.float()).mean()
    loss = (
        weights["ce"] * ce
        + weights["js"] * js
        + weights["l1"] * l1
        + weights["stats"] * stats
        + weights.get("wasserstein", 0.0) * wasserstein
    )
    parts = {"loss": loss, "ce": ce, "js": js, "l1": l1, "stats": stats, "wasserstein": wasserstein}
    return loss, {name: float(value.detach()) for name, value in parts.items()}


def make_loader(dataset: Dataset, config: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for density, _ in tqdm(loader, desc=description, leave=False, unit="batch"):
        density = density.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, _, _ = model(density)
        metrics = batch_metrics(density, prediction)
        batch_size = density.shape[0]
        count += batch_size
        for name, values in metrics.items():
            totals[name] += float(values.sum())
    return {name: total / count for name, total in totals.items()} | {"num_samples": count}


def save_reconstructions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_path: Path,
    seed: int,
    count: int = 4,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    batch = torch.stack([dataset[int(index)][0] for index in indices]).to(device)
    model.eval()
    with torch.inference_mode():
        prediction, _, _ = model(batch)
    originals = batch[:, 0].cpu().numpy()
    reconstructions = prediction[:, 0].cpu().numpy()
    fig, axes = plt.subplots(len(originals), 3, figsize=(12, 3 * len(originals)), squeeze=False)
    for row, (original, reconstruction) in enumerate(zip(originals, reconstructions)):
        vmax = max(float(original.max()), float(reconstruction.max()))
        axes[row, 0].imshow(original, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axes[row, 0].set_title("Target density")
        axes[row, 1].imshow(reconstruction, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axes[row, 1].set_title("Reconstruction")
        axes[row, 2].imshow(np.abs(original - reconstruction), origin="lower", cmap="viridis")
        axes[row, 2].set_title("Absolute error")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_target(
    frame: pd.DataFrame,
    config: dict[str, Any],
    data_path: Path,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    target = str(config["target"])
    seed = int(config["seed"])
    seed_everything(seed)
    datasets, split_summary = build_datasets(frame, target)

    if smoke:
        for split, limit in (("train", 512), ("dev", 256), ("val", 256), ("test", 256)):
            datasets[split] = torch.utils.data.Subset(
                datasets[split], range(min(limit, len(datasets[split])))
            )
        config = dict(config)
        config["max_epochs"] = 1

    write_json(ROOT / "artifacts" / f"{target}_split_summary.json", split_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("mixed_precision", True) and device.type == "cuda")
    model = CoordCNNAutoencoder(int(config["latent_dim"])).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    train_loader = make_loader(datasets["train"], config, shuffle=True, seed=seed)
    dev_loader = make_loader(datasets["dev"], config, shuffle=False, seed=seed)

    best_score = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    checkpoint_path = ROOT / "checkpoints" / f"{target}_coordcnn_ae_best.pt"
    history_path = ROOT / "reports" / f"{target}_training_history.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_dev_metrics"]["composite_score"])
        start_epoch = best_epoch + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [record for record in history if int(record["epoch"]) <= best_epoch]
        print(
            f"[{target}] resuming from epoch={best_epoch} dev_score={best_score:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.1e}",
            flush=True,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    prior_seconds = float(history[-1]["elapsed_seconds"]) if history else 0.0
    started = time.time() - prior_seconds

    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        model.train()
        running = defaultdict(float)
        seen = 0
        progress = tqdm(train_loader, desc=f"{target} epoch {epoch}", unit="batch")
        for density, _ in progress:
            density = density.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction, predicted_stats, _ = model(density)
                loss, parts = reconstruction_loss(density, prediction, predicted_stats, config["weights"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            batch_size = density.shape[0]
            seen += batch_size
            for name, value in parts.items():
                running[name] += value * batch_size
            progress.set_postfix(loss=f"{running['loss'] / seen:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        train_losses = {name: value / seen for name, value in running.items()}
        dev_metrics = evaluate(model, dev_loader, device, f"{target} dev epoch {epoch}", use_amp)
        score = float(dev_metrics["composite_score"])
        scheduler.step(score)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_losses": train_losses,
            "dev_metrics": dev_metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        write_json(history_path, history)

        improved = score < best_score - float(config["min_delta"])
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "config": config,
                    "parameter_count_total": total_params,
                    "parameter_count_trainable": trainable_params,
                    "best_dev_metrics": dev_metrics,
                    "data_path": str(data_path),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"[{target}] epoch={epoch} dev_score={score:.6f} best={best_score:.6f} "
            f"patience={epochs_without_improvement}/{config['early_stopping_patience']}",
            flush=True,
        )
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_metrics: dict[str, dict[str, float]] = {}
    for split in ("train", "dev", "val", "test"):
        loader = make_loader(datasets[split], config, shuffle=False, seed=seed)
        split_metrics[split] = evaluate(model, loader, device, f"{target} final {split}", use_amp)

    sample_path = ROOT / "artifacts" / "sample_reconstructions" / f"{target}_val.png"
    save_reconstructions(model, datasets["val"], device, sample_path, seed)
    result = {
        "target": target,
        "seed": seed,
        "data_path": str(data_path),
        "device": str(device),
        "config": config,
        "parameter_count_total": total_params,
        "parameter_count_trainable": trainable_params,
        "best_epoch_by_dev": best_epoch,
        "last_epoch_trained": int(history[-1]["epoch"]),
        "best_dev_score_during_training": best_score,
        "training_seconds": time.time() - started,
        "split_summary": split_summary,
        "metrics": split_metrics,
        "checkpoint": str(checkpoint_path),
        "sample_reconstructions": str(sample_path),
        "smoke_run": smoke,
    }
    write_json(ROOT / "reports" / f"{target}_coordcnn_ae_metrics.json", result)
    checkpoint["val_metrics"] = split_metrics["val"]
    checkpoint["test_metrics"] = split_metrics["test"]
    torch.save(checkpoint, checkpoint_path)
    return result


def write_summary(results: list[dict[str, Any]]) -> None:
    def last_epoch_trained(result: dict[str, Any]) -> int:
        if "last_epoch_trained" in result:
            return int(result["last_epoch_trained"])
        history_path = ROOT / "reports" / f"{result['target']}_training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        return int(history[-1]["epoch"])

    lines = [
        "# H1 CoordCNN Autoencoder Results",
        "",
        "Seed: `42`. Heatmaps are normalized to densities; original mass is excluded from the autoencoder.",
        "The single baseline variant is stopped on dev composite score. Validation is reported for model-selection comparison, and test is evaluated only after the checkpoint is fixed.",
        "",
        "| Target | Parameters | Best epoch | Last epoch | Dev score | Val score | Test score | Test JS | Test W1 marginal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['target']} | {result['parameter_count_trainable']:,} | {result['best_epoch_by_dev']} | "
            f"{last_epoch_trained(result)} | "
            f"{metrics['dev']['composite_score']:.6f} | {metrics['val']['composite_score']:.6f} | "
            f"{metrics['test']['composite_score']:.6f} | {metrics['test']['jensen_shannon_divergence']:.6f} | "
            f"{metrics['test']['marginal_wasserstein_1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The maximum epoch is a safety cap. Dev patience and `min_delta` determine stopping, while only the best dev checkpoint is retained.",
            "",
            "## Data",
            "",
            "| Target | Split | Samples used | Zero/invalid mass skipped |",
            "|---|---|---:|---:|",
        ]
    )
    for result in results:
        for split in ("train", "dev", "val", "test"):
            summary = result["split_summary"]["splits"][split]
            lines.append(
                f"| {result['target']} | {split} | {summary['num_samples']:,} | "
                f"{summary['num_zero_or_invalid_mass_skipped']:,} |"
            )
    lines.extend(
        [
            "",
            "## Full Metrics",
            "",
            "| Target | Split | Composite | JS | KL | Cosine | Marginal W1 | Centroid error (cells) | Zone MAE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        for split in ("train", "dev", "val", "test"):
            metrics = result["metrics"][split]
            lines.append(
                f"| {result['target']} | {split} | {metrics['composite_score']:.6f} | "
                f"{metrics['jensen_shannon_divergence']:.6f} | {metrics['kl_divergence']:.6f} | "
                f"{metrics['cosine_similarity']:.6f} | {metrics['marginal_wasserstein_1']:.6f} | "
                f"{metrics['centroid_error_l2_cells']:.4f} | {metrics['zone_share_mae']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Loss",
            "",
            "The baseline uses density cross-entropy + Jensen-Shannon + scaled L1 + SmoothL1 on interpretable spatial statistics. Marginal Wasserstein-1 is reported but has training weight 0 in version 1.",
            "",
            "## Configuration",
            "",
        ]
    )
    for result in results:
        config = result["config"]
        weights = config["weights"]
        lines.append(
            f"- `{result['target']}`: latent_dim={config['latent_dim']}, batch_size={config['batch_size']}, "
            f"lr={config['learning_rate']}, weight_decay={config['weight_decay']}, max_epochs={config['max_epochs']}, "
            f"patience={config['early_stopping_patience']}, loss_weights={weights}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 39,957-parameter baseline preserves global location, centroid, spread, and broad occupied regions. The reconstruction samples also show that the implicit coordinate decoder acts as a strong spatial smoother: it does not preserve every local mode or sharp KDE peak, especially for team heatmaps. A follow-up model comparison should therefore keep the same encoder and test a small upsampling or residual-basis decoder on validation data.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result['target']}` checkpoint: `{result['checkpoint']}`")
        lines.append(f"- `{result['target']}` reconstruction samples: `{result['sample_reconstructions']}`")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "h1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("player", "team", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on a tiny subset")
    parser.add_argument("--resume", action="store_true", help="Resume each target from its best checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = ("team", "player") if args.target == "all" else (args.target,)
    print(f"Loading {args.data} ...", flush=True)
    load_started = time.time()
    frame = pd.read_pickle(args.data)
    print(f"Loaded {frame.shape} in {time.time() - load_started:.1f}s", flush=True)
    results = []
    for target in targets:
        config_path = ROOT / "configs" / f"{target}_ae.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        results.append(
            train_target(frame, config, args.data.resolve(), smoke=args.smoke, resume=args.resume)
        )
    if not args.smoke:
        existing = []
        for target in ("team", "player"):
            path = ROOT / "reports" / f"{target}_coordcnn_ae_metrics.json"
            if path.exists():
                existing.append(json.loads(path.read_text(encoding="utf-8")))
        write_summary(existing)
    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
