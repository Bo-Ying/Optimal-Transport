"""Evaluate substitute appearance classification without decoding heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from train_coordcnn_ae import EPS
from train_player_prediction_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    ROOT,
    SPLIT_YEARS,
    PredictionV1,
    load_frozen_autoencoder,
    load_or_build_cache,
    seed_everything,
    to_device,
    write_json,
)


class AppearanceDataset(Dataset):
    """Match-side inputs needed by PredictionV1 when spatial decoding is disabled."""

    def __init__(self, arrays: dict[str, np.ndarray], side_indices: np.ndarray) -> None:
        self.a = arrays
        self.side_indices = side_indices.astype(np.int32)
        self.roster_size = arrays["roster_valid"].shape[1] // 2
        self.total_slots = 2 * self.roster_size

    def __len__(self) -> int:
        return len(self.side_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        global_side = int(self.side_indices[index])
        match_index, side = divmod(global_side, 2)
        slots = np.arange(side * self.roster_size, (side + 1) * self.roster_size)
        history = self.a["player_history"][match_index, slots]
        valid = history >= 0
        hist_match, hist_slot = np.divmod(np.maximum(history, 0), self.total_slots)
        current_day = int(self.a["date_days"][match_index])

        team_history = self.a["team_history"][match_index, side]
        team_valid = team_history >= 0
        team_hist_match, team_hist_side = np.divmod(np.maximum(team_history, 0), 2)

        return {
            "side": torch.tensor(side),
            "league": torch.tensor(int(self.a["league_ids"][match_index])),
            "roster_valid": torch.from_numpy(self.a["roster_valid"][match_index, slots]),
            "player_hist_token": torch.from_numpy(self.a["player_tokens"][hist_match, hist_slot].astype(np.float32)),
            "player_hist_mass": torch.from_numpy(self.a["player_mass"][hist_match, hist_slot]),
            "player_hist_rating": torch.from_numpy(self.a["player_rating"][hist_match, hist_slot]),
            "player_hist_minutes": torch.from_numpy(self.a["player_minutes"][hist_match, hist_slot]),
            "player_hist_position": torch.from_numpy(self.a["player_position"][hist_match, hist_slot].astype(np.int64)),
            "player_hist_starter": torch.from_numpy(self.a["player_starter"][hist_match, hist_slot].astype(np.int64)),
            "player_hist_side": torch.from_numpy(self.a["player_side"][hist_match, hist_slot].astype(np.int64)),
            "player_hist_league": torch.from_numpy(self.a["league_ids"][hist_match].astype(np.int64)),
            "player_hist_recency": torch.from_numpy(
                np.log1p(np.maximum(current_day - self.a["date_days"][hist_match], 0)).astype(np.float32)
            ),
            "player_hist_valid": torch.from_numpy(valid),
            "current_position": torch.from_numpy(self.a["player_position"][match_index, slots].astype(np.int64)),
            "current_starter": torch.from_numpy(self.a["player_starter"][match_index, slots].astype(np.int64)),
            "current_age": torch.from_numpy(self.a["player_age"][match_index, slots]),
            "current_height": torch.from_numpy(self.a["player_height"][match_index, slots]),
            "current_weight": torch.from_numpy(self.a["player_weight"][match_index, slots]),
            "target_player_mass": torch.from_numpy(self.a["player_mass"][match_index, slots]),
            "team_hist_token": torch.from_numpy(
                self.a["team_tokens"][team_hist_match, team_hist_side].astype(np.float32)
            ),
            "team_hist_mass": torch.from_numpy(self.a["team_mass"][team_hist_match, team_hist_side]),
            "team_hist_side": torch.from_numpy(team_hist_side.astype(np.int64)),
            "team_hist_league": torch.from_numpy(self.a["league_ids"][team_hist_match].astype(np.int64)),
            "team_hist_recency": torch.from_numpy(
                np.log1p(np.maximum(current_day - self.a["date_days"][team_hist_match], 0)).astype(np.float32)
            ),
            "team_hist_valid": torch.from_numpy(team_valid),
        }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "threshold": 0.5,
        "num_substitute_candidates": int(len(labels)),
        "num_appeared": int(labels.sum()),
        "appearance_rate": float(labels.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1),
        "brier": float(brier_score_loss(labels, probabilities)),
        "binary_cross_entropy": float(log_loss(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    frame = pd.read_pickle(DEFAULT_DATA)
    frame, arrays, metadata = load_or_build_cache(frame, config, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player_ae, _ = load_frozen_autoencoder("player", device)
    team_ae, _ = load_frozen_autoencoder("team", device)
    model = PredictionV1(config, metadata, player_ae, team_ae).to(device)
    checkpoint_path = ROOT / "checkpoints" / "player_prediction_v1_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    side_years = np.repeat(arrays["season_start_year"], 2)
    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": int(config["seed"]),
        "metrics": {},
    }
    for split in ("dev", "val", "test"):
        indices = np.flatnonzero(np.isin(side_years, SPLIT_YEARS[split]))
        loader = DataLoader(AppearanceDataset(arrays, indices), batch_size=128, shuffle=False, num_workers=0)
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                batch = to_device(batch, device)
                output = model(batch, decode_spatial=False)
                substitute = batch["roster_valid"].bool() & ~batch["current_starter"].bool()
                labels.append((batch["target_player_mass"] > EPS)[substitute].cpu().numpy().astype(np.int8))
                probabilities.append(output["substitute_probability"][substitute].cpu().numpy())
        results["metrics"][split] = classification_metrics(np.concatenate(labels), np.concatenate(probabilities))
        print(split, json.dumps(results["metrics"][split], indent=2), flush=True)

    output_path = ROOT / "reports" / "substitute_classification_metrics.json"
    write_json(output_path, results)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
