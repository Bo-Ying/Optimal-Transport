"""Train frozen-CoordCNN player heatmap forecasting with soft team consistency."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from train_coordcnn_ae import (
    EPS,
    HEIGHT,
    WIDTH,
    CoordCNNAutoencoder,
    batch_metrics,
    density_stats,
    marginal_wasserstein,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "DATA" / "dataset_player_KDE_clean_enriched.pkl"
DEFAULT_CONFIG = ROOT / "configs" / "player_prediction_v1.yaml"
SPLIT_YEARS = {
    "train": tuple(range(2015, 2023)),
    "dev": (2023,),
    "val": (2024,),
    "test": (2025,),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def season_start_year(values: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(values, errors="raise")
    return np.where(dates.dt.month.to_numpy() >= 7, dates.dt.year, dates.dt.year - 1).astype(np.int16)


def normalized_density(value: Any) -> tuple[np.ndarray, float]:
    array = np.asarray(value, dtype=np.float32)
    mass = float(array.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= EPS:
        return np.zeros((HEIGHT, WIDTH), dtype=np.float32), 0.0
    return array / mass, mass


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def load_frozen_autoencoder(target: str, device: torch.device) -> tuple[CoordCNNAutoencoder, dict[str, Any]]:
    path = ROOT / "checkpoints" / f"{target}_coordcnn_ae_best.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CoordCNNAutoencoder(int(checkpoint["config"]["latent_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze_for_prediction()
    return model, checkpoint


def encode_densities(
    model: CoordCNNAutoencoder,
    values: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    if not values:
        return np.empty((0, model.encoder_head[-1].out_features), dtype=np.float16)
    tensor = torch.from_numpy(np.stack(values)).unsqueeze(1).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        latent = model.encode(tensor)
    return latent.float().cpu().numpy().astype(np.float16)


def build_token_cache(
    frame: pd.DataFrame,
    config: dict[str, Any],
    cache_path: Path,
    meta_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    frame = frame.assign(_date=pd.to_datetime(frame["Match_Date"])).sort_values("_date", kind="stable").reset_index(drop=True)
    n_matches = len(frame)
    history_length = int(config["history_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player_ae, player_checkpoint = load_frozen_autoencoder("player", device)
    team_ae, team_checkpoint = load_frozen_autoencoder("team", device)
    latent_dim = int(config["latent_dim"])
    roster_size = max(
        len(row[f"Starting_11({side})"]) + len(row[f"Substitute({side})"])
        for _, row in frame.iterrows()
        for side in ("Home", "Away")
    )
    total_slots = 2 * roster_size

    leagues = sorted(str(value) for value in frame["League"].dropna().unique())
    positions = sorted(
        {str(position) for values in frame["Player_Position"] for position in values.values() if position is not None}
    )
    league_to_id = {value: index + 1 for index, value in enumerate(leagues)}
    position_to_id = {value: index + 1 for index, value in enumerate(positions)}

    shape = (n_matches, total_slots)
    player_ids = np.full(shape, -1, dtype=np.int64)
    roster_valid = np.zeros(shape, dtype=np.bool_)
    player_tokens = np.zeros((*shape, latent_dim), dtype=np.float16)
    player_mass = np.zeros(shape, dtype=np.float32)
    player_rating = np.full(shape, np.nan, dtype=np.float32)
    player_minutes = np.zeros(shape, dtype=np.float32)
    player_age = np.full(shape, np.nan, dtype=np.float32)
    player_height = np.full(shape, np.nan, dtype=np.float32)
    player_weight = np.full(shape, np.nan, dtype=np.float32)
    player_position = np.zeros(shape, dtype=np.int16)
    player_starter = np.zeros(shape, dtype=np.int8)
    player_side = np.zeros(shape, dtype=np.int8)
    player_history = np.full((*shape, history_length), -1, dtype=np.int32)
    team_tokens = np.zeros((n_matches, 2, latent_dim), dtype=np.float16)
    team_mass = np.zeros((n_matches, 2), dtype=np.float32)
    team_ids = np.zeros((n_matches, 2), dtype=np.int64)
    team_history = np.full((n_matches, 2, history_length), -1, dtype=np.int32)
    league_ids = np.asarray([league_to_id[str(value)] for value in frame["League"]], dtype=np.int16)
    date_days = (frame["_date"].astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int32)
    years = season_start_year(frame["Match_Date"])

    player_batch_values: list[np.ndarray] = []
    player_batch_indices: list[tuple[int, int]] = []
    team_batch_values: list[np.ndarray] = []
    team_batch_indices: list[tuple[int, int]] = []
    encode_batch = int(config.get("cache_encode_batch_size", 512))

    def flush_player() -> None:
        nonlocal player_batch_values, player_batch_indices
        encoded = encode_densities(player_ae, player_batch_values, device)
        for (match_index, slot), token in zip(player_batch_indices, encoded, strict=True):
            player_tokens[match_index, slot] = token
        player_batch_values, player_batch_indices = [], []

    def flush_team() -> None:
        nonlocal team_batch_values, team_batch_indices
        encoded = encode_densities(team_ae, team_batch_values, device)
        for (match_index, side), token in zip(team_batch_indices, encoded, strict=True):
            team_tokens[match_index, side] = token
        team_batch_values, team_batch_indices = [], []

    iterator = tqdm(
        frame.iterrows(),
        total=n_matches,
        desc="Building frozen token cache",
        unit="match",
        disable=not bool(config.get("show_progress", False)),
    )
    for match_index, row in iterator:
        for side, side_name in enumerate(("Home", "Away")):
            start = side * roster_size
            roster = [int(value) for value in row[f"Starting_11({side_name})"] + row[f"Substitute({side_name})"]]
            team_ids[match_index, side] = int(row[f"{side_name}_Team_ID"])
            team_density, current_team_mass = normalized_density(row[f"Team_Heatmap({side_name})"])
            team_mass[match_index, side] = current_team_mass
            team_batch_values.append(team_density)
            team_batch_indices.append((match_index, side))
            if len(team_batch_values) >= encode_batch:
                flush_team()

            for local_slot, player_id in enumerate(roster):
                slot = start + local_slot
                player_ids[match_index, slot] = player_id
                roster_valid[match_index, slot] = True
                density, mass = normalized_density(row["Player_Heatmap"][player_id])
                player_mass[match_index, slot] = mass
                player_minutes[match_index, slot] = float(row["Player_Minutes"].get(player_id, 0) or 0)
                player_rating[match_index, slot] = finite_float(row["Player_Rating"].get(player_id))
                player_age[match_index, slot] = finite_float(row["Player_Age"].get(player_id))
                player_height[match_index, slot] = finite_float(row["Player_Height"].get(player_id))
                player_weight[match_index, slot] = finite_float(row["Player_Weight"].get(player_id))
                player_position[match_index, slot] = position_to_id.get(
                    str(row["Player_Position"].get(player_id)), 0
                )
                player_starter[match_index, slot] = int(local_slot < 11)
                player_side[match_index, slot] = side
                if mass > EPS:
                    player_batch_values.append(density)
                    player_batch_indices.append((match_index, slot))
                    if len(player_batch_values) >= encode_batch:
                        flush_player()
    flush_player()
    flush_team()

    player_seen: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=history_length))
    team_seen: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=history_length))
    last_known_position: dict[int, int] = {}
    substitute_position_id = position_to_id.get("Sub", -1)
    for _, group in frame.groupby("_date", sort=False):
        group_indices = group.index.to_list()
        for match_index in group_indices:
            for slot in range(total_slots):
                if not roster_valid[match_index, slot]:
                    continue
                player_id = int(player_ids[match_index, slot])
                if player_position[match_index, slot] == substitute_position_id and player_id in last_known_position:
                    player_position[match_index, slot] = last_known_position[player_id]
                previous = list(player_seen[player_id])
                player_history[match_index, slot, : len(previous)] = previous
            for side in range(2):
                previous = list(team_seen[int(team_ids[match_index, side])])
                team_history[match_index, side, : len(previous)] = previous
        for match_index in group_indices:
            for slot in range(total_slots):
                if player_mass[match_index, slot] > EPS:
                    player_seen[int(player_ids[match_index, slot])].append(match_index * total_slots + slot)
                position_id = int(player_position[match_index, slot])
                if roster_valid[match_index, slot] and position_id not in (0, substitute_position_id):
                    last_known_position[int(player_ids[match_index, slot])] = position_id
            for side in range(2):
                team_seen[int(team_ids[match_index, side])].append(match_index * 2 + side)

    train_match = np.isin(years, SPLIT_YEARS["train"])
    train_appeared = train_match[:, None] & (player_mass > EPS)

    def moments(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        selected = values[mask & np.isfinite(values)].astype(np.float64)
        return float(selected.mean()), float(max(selected.std(), 1e-6))

    normalizers = {
        "player_log_mass": moments(np.log1p(player_mass), train_appeared),
        "team_log_mass": moments(np.log1p(team_mass), np.repeat(train_match[:, None], 2, axis=1)),
        "rating": moments(player_rating, train_appeared),
        "age": moments(player_age, train_appeared),
        "height": moments(player_height, train_appeared),
        "weight": moments(player_weight, train_appeared),
    }
    arrays = {
        "player_ids": player_ids,
        "roster_valid": roster_valid,
        "player_tokens": player_tokens,
        "player_mass": player_mass,
        "player_rating": player_rating,
        "player_minutes": player_minutes,
        "player_age": player_age,
        "player_height": player_height,
        "player_weight": player_weight,
        "player_position": player_position,
        "player_starter": player_starter,
        "player_side": player_side,
        "player_history": player_history,
        "team_tokens": team_tokens,
        "team_mass": team_mass,
        "team_ids": team_ids,
        "team_history": team_history,
        "league_ids": league_ids,
        "date_days": date_days,
        "season_start_year": years,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    metadata = {
        "num_matches": n_matches,
        "history_length": history_length,
        "latent_dim": latent_dim,
        "roster_size": roster_size,
        "leagues": leagues,
        "positions": positions,
        "normalizers": normalizers,
        "player_ae_epoch": int(player_checkpoint["epoch"]),
        "team_ae_epoch": int(team_checkpoint["epoch"]),
        "strict_history_rule": "Match_Date < target Match_Date",
        "substitute_position_rule": "Most recent non-Sub position before target date; Sub only for cold start",
    }
    write_json(meta_path, metadata)
    return frame, arrays, metadata


def load_or_build_cache(
    frame: pd.DataFrame,
    config: dict[str, Any],
    rebuild: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    cache_path = ROOT / "artifacts" / "prediction_v1_token_cache.npz"
    meta_path = ROOT / "artifacts" / "prediction_v1_token_cache_meta.json"
    frame = frame.assign(_date=pd.to_datetime(frame["Match_Date"])).sort_values("_date", kind="stable").reset_index(drop=True)
    if rebuild or not cache_path.exists() or not meta_path.exists():
        return build_token_cache(frame.drop(columns="_date"), config, cache_path, meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["num_matches"] != len(frame) or metadata["history_length"] != int(config["history_length"]):
        raise ValueError("Prediction cache does not match data/config; rerun with --rebuild-cache")
    archive = np.load(cache_path)
    return frame, {key: archive[key] for key in archive.files}, metadata


class MatchSideDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, arrays: dict[str, np.ndarray], side_indices: np.ndarray) -> None:
        self.frame = frame
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
        safe = np.maximum(history, 0)
        hist_match, hist_slot = np.divmod(safe, self.total_slots)
        current_day = int(self.a["date_days"][match_index])
        hist_days = self.a["date_days"][hist_match]

        team_history = self.a["team_history"][match_index, side]
        team_valid = team_history >= 0
        safe_team = np.maximum(team_history, 0)
        team_hist_match, team_hist_side = np.divmod(safe_team, 2)

        row = self.frame.iloc[match_index]
        side_name = "Home" if side == 0 else "Away"
        densities = []
        for slot in slots:
            player_id = int(self.a["player_ids"][match_index, slot])
            density, _ = (
                normalized_density(row["Player_Heatmap"][player_id])
                if player_id >= 0
                else (np.zeros((HEIGHT, WIDTH), dtype=np.float32), 0.0)
            )
            densities.append(density.astype(np.float16))
        team_density, _ = normalized_density(row[f"Team_Heatmap({side_name})"])

        return {
            "match_index": torch.tensor(match_index),
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
            "player_hist_recency": torch.from_numpy(np.log1p(np.maximum(current_day - hist_days, 0)).astype(np.float32)),
            "player_hist_valid": torch.from_numpy(valid),
            "current_position": torch.from_numpy(self.a["player_position"][match_index, slots].astype(np.int64)),
            "current_starter": torch.from_numpy(self.a["player_starter"][match_index, slots].astype(np.int64)),
            "current_age": torch.from_numpy(self.a["player_age"][match_index, slots]),
            "current_height": torch.from_numpy(self.a["player_height"][match_index, slots]),
            "current_weight": torch.from_numpy(self.a["player_weight"][match_index, slots]),
            "target_player_token": torch.from_numpy(self.a["player_tokens"][match_index, slots].astype(np.float32)),
            "target_player_mass": torch.from_numpy(self.a["player_mass"][match_index, slots]),
            "target_player_density": torch.from_numpy(np.stack(densities)),
            "team_hist_token": torch.from_numpy(self.a["team_tokens"][team_hist_match, team_hist_side].astype(np.float32)),
            "team_hist_mass": torch.from_numpy(self.a["team_mass"][team_hist_match, team_hist_side]),
            "team_hist_side": torch.from_numpy(team_hist_side.astype(np.int64)),
            "team_hist_league": torch.from_numpy(self.a["league_ids"][team_hist_match].astype(np.int64)),
            "team_hist_recency": torch.from_numpy(np.log1p(np.maximum(current_day - self.a["date_days"][team_hist_match], 0)).astype(np.float32)),
            "team_hist_valid": torch.from_numpy(team_valid),
            "target_team_token": torch.from_numpy(self.a["team_tokens"][match_index, side].astype(np.float32)),
            "target_team_mass": torch.tensor(float(self.a["team_mass"][match_index, side])),
            "target_team_density": torch.from_numpy(team_density.astype(np.float16)),
        }


class PredictionV1(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        metadata: dict[str, Any],
        player_ae: CoordCNNAutoencoder,
        team_ae: CoordCNNAutoencoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.norm = metadata["normalizers"]
        latent_dim = int(config["latent_dim"])
        hidden = int(config["hidden_dim"])
        self.player_ae = player_ae
        self.team_ae = team_ae
        self.league_embedding = nn.Embedding(len(metadata["leagues"]) + 1, 4, padding_idx=0)
        self.position_embedding = nn.Embedding(len(metadata["positions"]) + 1, 8, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 2)
        self.starter_embedding = nn.Embedding(2, 2)

        player_history_dim = latent_dim + 4 + 4 + 8 + 2 + 2
        self.player_history_projection = nn.Sequential(
            nn.Linear(player_history_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.player_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.player_cold = nn.Parameter(torch.zeros(hidden))
        self.player_cold_latent = nn.Parameter(torch.zeros(latent_dim))
        current_dim = hidden + 4 + 8 + 2 + 2 + 3
        self.player_fusion = nn.Sequential(
            nn.Linear(current_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.player_delta = nn.Linear(hidden, latent_dim)
        self.appearance_head = nn.Linear(hidden, 1)
        self.player_mass_head = nn.Linear(hidden, 1)

        team_history_dim = latent_dim + 2 + 4 + 2
        self.team_history_projection = nn.Sequential(
            nn.Linear(team_history_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.team_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.team_cold = nn.Parameter(torch.zeros(hidden))
        self.team_cold_latent = nn.Parameter(torch.zeros(latent_dim))
        self.team_fusion = nn.Sequential(
            nn.Linear(hidden + 4 + 2, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.team_delta = nn.Linear(hidden, latent_dim)
        self.team_mass_head = nn.Linear(hidden, 1)

    def train(self, mode: bool = True) -> "PredictionV1":
        super().train(mode)
        self.player_ae.eval()
        self.team_ae.eval()
        return self

    def standardize(self, value: torch.Tensor, name: str) -> torch.Tensor:
        mean, std = self.norm[name]
        return (torch.nan_to_num(value.float(), nan=mean) - mean) / std

    @staticmethod
    def temporal_summary(
        projected: torch.Tensor,
        valid: torch.Tensor,
        gru: nn.GRU,
        cold: torch.Tensor,
    ) -> torch.Tensor:
        batch_shape = projected.shape[:-2]
        sequence_length = projected.shape[-2]
        flat = projected.reshape(-1, sequence_length, projected.shape[-1])
        flat_valid = valid.reshape(-1, sequence_length)
        output, _ = gru(flat * flat_valid.unsqueeze(-1))
        lengths = flat_valid.sum(1)
        gathered = output[torch.arange(len(output), device=output.device), (lengths - 1).clamp_min(0)]
        gathered = torch.where(lengths.unsqueeze(1) > 0, gathered, cold.unsqueeze(0))
        return gathered.reshape(*batch_shape, -1)

    def decode(self, model: CoordCNNAutoencoder, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        logits = model.decode_logits(latent.reshape(-1, latent.shape[-1]))
        density = torch.softmax(logits.float().flatten(1), dim=1).view_as(logits)
        return density.reshape(*original_shape, 1, HEIGHT, WIDTH)

    def decode_valid_players(self, latent: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        flat_latent = latent.reshape(-1, latent.shape[-1])
        flat_valid = valid.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(1)
        decoded = self.decode(self.player_ae, flat_latent[valid_indices]).squeeze(1)
        flat_density = decoded.new_zeros((len(flat_latent), HEIGHT, WIDTH))
        flat_density = flat_density.index_copy(0, valid_indices, decoded)
        return flat_density.reshape(*latent.shape[:-1], HEIGHT, WIDTH)

    def forward(self, batch: dict[str, torch.Tensor], decode_spatial: bool = True) -> dict[str, torch.Tensor]:
        valid = batch["player_hist_valid"]
        history_continuous = torch.stack(
            (
                self.standardize(torch.log1p(batch["player_hist_mass"]), "player_log_mass"),
                self.standardize(batch["player_hist_rating"], "rating"),
                batch["player_hist_minutes"].float() / 90.0,
                batch["player_hist_recency"].float() / math.log1p(365.0),
            ),
            dim=-1,
        )
        history_features = torch.cat(
            (
                batch["player_hist_token"].float(),
                history_continuous,
                self.league_embedding(batch["player_hist_league"]),
                self.position_embedding(batch["player_hist_position"]),
                self.side_embedding(batch["player_hist_side"]),
                self.starter_embedding(batch["player_hist_starter"]),
            ),
            dim=-1,
        )
        player_history = self.temporal_summary(
            self.player_history_projection(history_features), valid, self.player_gru, self.player_cold
        )
        lengths = valid.sum(-1)
        last_index = (lengths - 1).clamp_min(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, batch["player_hist_token"].shape[-1])
        base_latent = batch["player_hist_token"].gather(2, last_index).squeeze(2).float()
        base_latent = torch.where(lengths.unsqueeze(-1) > 0, base_latent, self.player_cold_latent)
        batch_size = base_latent.shape[0]
        roster_size = base_latent.shape[1]
        league = batch["league"].unsqueeze(1).expand(-1, roster_size)
        side = batch["side"].unsqueeze(1).expand(-1, roster_size)
        static = torch.stack(
            (
                self.standardize(batch["current_age"], "age"),
                self.standardize(batch["current_height"], "height"),
                self.standardize(batch["current_weight"], "weight"),
            ),
            dim=-1,
        )
        current = torch.cat(
            (
                player_history,
                self.league_embedding(league),
                self.position_embedding(batch["current_position"]),
                self.side_embedding(side),
                self.starter_embedding(batch["current_starter"]),
                static,
            ),
            dim=-1,
        )
        fused = self.player_fusion(current)
        player_latent = base_latent + 0.5 * torch.tanh(self.player_delta(fused))
        appearance_logits = self.appearance_head(fused).squeeze(-1)
        substitute_probability = torch.sigmoid(appearance_logits)
        appearance = torch.where(batch["current_starter"].bool(), torch.ones_like(substitute_probability), substitute_probability)
        appearance = appearance * batch["roster_valid"].float()
        player_mass_z = 4.0 * torch.tanh(self.player_mass_head(fused).squeeze(-1) / 4.0)
        player_mean, player_std = self.norm["player_log_mass"]
        player_mass = torch.expm1(player_mass_z * player_std + player_mean).clamp_min(0)

        team_valid = batch["team_hist_valid"]
        team_continuous = torch.stack(
            (
                self.standardize(torch.log1p(batch["team_hist_mass"]), "team_log_mass"),
                batch["team_hist_recency"].float() / math.log1p(365.0),
            ),
            dim=-1,
        )
        team_features = torch.cat(
            (
                batch["team_hist_token"].float(),
                team_continuous,
                self.league_embedding(batch["team_hist_league"]),
                self.side_embedding(batch["team_hist_side"]),
            ),
            dim=-1,
        )
        team_history = self.temporal_summary(
            self.team_history_projection(team_features), team_valid, self.team_gru, self.team_cold
        )
        team_lengths = team_valid.sum(-1)
        team_last_index = (team_lengths - 1).clamp_min(0).view(batch_size, 1, 1).expand(-1, 1, batch["team_hist_token"].shape[-1])
        team_base = batch["team_hist_token"].gather(1, team_last_index).squeeze(1).float()
        team_base = torch.where(team_lengths.unsqueeze(-1) > 0, team_base, self.team_cold_latent)
        team_current = torch.cat(
            (team_history, self.league_embedding(batch["league"]), self.side_embedding(batch["side"])), dim=-1
        )
        team_fused = self.team_fusion(team_current)
        team_latent = team_base + 0.5 * torch.tanh(self.team_delta(team_fused))
        team_mass_z = 4.0 * torch.tanh(self.team_mass_head(team_fused).squeeze(-1) / 4.0)
        team_mean, team_std = self.norm["team_log_mass"]
        team_mass = torch.expm1(team_mass_z * team_std + team_mean).clamp_min(0)
        result = {
            "player_latent": player_latent,
            "appearance": appearance,
            "appearance_logits": appearance_logits,
            "substitute_probability": substitute_probability,
            "player_mass": player_mass,
            "player_mass_z": player_mass_z,
            "history_length": lengths,
            "team_latent": team_latent,
            "team_mass": team_mass,
            "team_mass_z": team_mass_z,
        }
        if decode_spatial:
            result["player_density"] = self.decode_valid_players(player_latent, batch["roster_valid"])
            result["team_density"] = self.decode(self.team_ae, team_latent).squeeze(1)
        return result


def distribution_loss(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    if target.ndim == 3:
        target = target.unsqueeze(1)
        prediction = prediction.unsqueeze(1)
    target_flat = target.float().flatten(1).clamp_min(EPS)
    prediction_flat = prediction.float().flatten(1).clamp_min(EPS)
    midpoint = 0.5 * (target_flat + prediction_flat)
    kl = (target_flat * (target_flat.log() - prediction_flat.log())).sum(1).mean()
    js = 0.5 * (
        (target_flat * (target_flat.log() - midpoint.log())).sum(1)
        + (prediction_flat * (prediction_flat.log() - midpoint.log())).sum(1)
    ).mean()
    l1 = (target_flat - prediction_flat).abs().sum(1).mean()
    stats = F.smooth_l1_loss(density_stats(prediction.float()), density_stats(target.float()))
    wasserstein = marginal_wasserstein(target.float(), prediction.float()).mean()
    return {"kl": kl, "js": js, "l1": l1, "stats": stats, "wasserstein": wasserstein}


def compute_loss(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    model: PredictionV1,
    weights: dict[str, float],
    consistency_factor: float,
    spatial_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    appeared = batch["target_player_mass"] > EPS
    substitute = batch["roster_valid"].bool() & ~batch["current_starter"].bool()
    latent = F.smooth_l1_loss(output["player_latent"][appeared], batch["target_player_token"][appeared].float())
    target_player_mass_z = model.standardize(torch.log1p(batch["target_player_mass"]), "player_log_mass")
    mass = F.smooth_l1_loss(output["player_mass_z"][appeared], target_player_mass_z[appeared])
    appearance = F.binary_cross_entropy_with_logits(
        output["appearance_logits"][substitute], appeared.float()[substitute]
    )
    team_latent = F.smooth_l1_loss(output["team_latent"], batch["target_team_token"].float())
    target_team_mass_z = model.standardize(torch.log1p(batch["target_team_mass"]), "team_log_mass")
    team_mass = F.smooth_l1_loss(output["team_mass_z"], target_team_mass_z)

    zero = latent.new_zeros(())
    player_distribution = {name: zero for name in ("kl", "js", "l1", "stats", "wasserstein")}
    team_distribution = {name: zero for name in ("kl", "js", "l1", "stats", "wasserstein")}
    consistency_shape = {"js": zero, "wasserstein": zero}
    consistency_mass = zero
    spatial_objective = zero
    if "player_density" in output:
        player_distribution = distribution_loss(
            batch["target_player_density"][appeared], output["player_density"][appeared]
        )
        team_distribution = distribution_loss(batch["target_team_density"], output["team_density"])
        expected = output["appearance"] * output["player_mass"]
        aggregate = (expected[:, :, None, None] * output["player_density"]).sum(1)
        aggregate_mass = aggregate.sum((1, 2)).clamp_min(EPS)
        aggregate_density = aggregate / aggregate_mass[:, None, None]
        consistency_shape = distribution_loss(output["team_density"], aggregate_density)
        consistency_mass = F.smooth_l1_loss(
            torch.log1p(aggregate_mass), torch.log1p(output["team_mass"].clamp_min(EPS))
        )
        player_density_loss = sum(
            weights[f"density_{name}"] * player_distribution[name]
            for name in ("kl", "js", "l1", "stats", "wasserstein")
        )
        team_density_loss = sum(
            weights[f"density_{name}"] * team_distribution[name]
            for name in ("kl", "js", "l1", "stats", "wasserstein")
        )
        consistency = consistency_shape["js"] + 0.1 * consistency_shape["wasserstein"] + consistency_mass
        spatial_objective = spatial_scale * (
            player_density_loss
            + weights["team_supervision"] * team_density_loss
            + consistency_factor * weights["team_consistency"] * consistency
        )
    total = (
        spatial_objective
        + weights["latent"] * latent
        + weights["appearance"] * appearance
        + weights["mass"] * mass
        + weights["team_supervision"] * (team_latent + team_mass)
    )
    parts = {
        "loss": float(total.detach()),
        "player_js": float(player_distribution["js"].detach()),
        "player_w1": float(player_distribution["wasserstein"].detach()),
        "latent": float(latent.detach()),
        "appearance": float(appearance.detach()),
        "mass": float(mass.detach()),
        "team_js": float(team_distribution["js"].detach()),
        "consistency_shape_js": float(consistency_shape["js"].detach()),
        "consistency_mass": float(consistency_mass.detach()),
    }
    return total, parts


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: PredictionV1,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for batch in tqdm(loader, desc=description, unit="batch", disable=True):
        batch = to_device(batch, device)
        output = model(batch)
        appeared = batch["target_player_mass"] > EPS
        substitute = batch["roster_valid"].bool() & ~batch["current_starter"].bool()
        metrics = batch_metrics(
            batch["target_player_density"][appeared].unsqueeze(1).float(),
            output["player_density"][appeared].unsqueeze(1).float(),
        )
        n_player = int(appeared.sum())
        for key, value in metrics.items():
            totals[f"player_{key}"] += float(value.sum())
        counts["player"] += n_player
        target_log_mass = torch.log1p(batch["target_player_mass"][appeared])
        pred_log_mass = torch.log1p(output["player_mass"][appeared])
        totals["player_log_mass_mae"] += float((target_log_mass - pred_log_mass).abs().sum())
        totals["player_mass_mae"] += float((batch["target_player_mass"][appeared] - output["player_mass"][appeared]).abs().sum())
        sub_target = appeared[substitute].float()
        sub_pred = output["substitute_probability"][substitute]
        totals["substitute_brier"] += float((sub_target - sub_pred).square().sum())
        totals["substitute_bce"] += float(F.binary_cross_entropy(sub_pred, sub_target, reduction="sum"))
        counts["substitute"] += len(sub_target)

        team_metrics = batch_metrics(
            batch["target_team_density"][:, None].float(), output["team_density"][:, None].float()
        )
        n_team = len(batch["target_team_mass"])
        for key, value in team_metrics.items():
            totals[f"team_{key}"] += float(value.sum())
        totals["team_log_mass_mae"] += float(
            (torch.log1p(batch["target_team_mass"]) - torch.log1p(output["team_mass"])).abs().sum()
        )
        counts["team"] += n_team

        expected = output["appearance"] * output["player_mass"]
        aggregate = (expected[:, :, None, None] * output["player_density"]).sum(1)
        aggregate_mass = aggregate.sum((1, 2)).clamp_min(EPS)
        aggregate_density = aggregate / aggregate_mass[:, None, None]
        aggregate_metrics = batch_metrics(
            batch["target_team_density"][:, None].float(), aggregate_density[:, None].float()
        )
        for key, value in aggregate_metrics.items():
            totals[f"aggregate_{key}"] += float(value.sum())
        totals["aggregate_mass_log_mae"] += float(
            (torch.log1p(batch["target_team_mass"]) - torch.log1p(aggregate_mass)).abs().sum()
        )
    result: dict[str, float] = {}
    for key, value in totals.items():
        denominator = counts["substitute"] if key.startswith("substitute") else counts["player"] if key.startswith("player") else counts["team"]
        result[key] = value / max(denominator, 1)
    result["num_player_appearances"] = counts["player"]
    result["num_substitute_candidates"] = counts["substitute"]
    result["num_team_sides"] = counts["team"]
    result["composite_score"] = (
        result["player_jensen_shannon_divergence"]
        + 0.1 * result["player_marginal_wasserstein_1"]
        + 0.05 * result["player_centroid_error_l2_pitch_norm"]
        + 0.02 * result["player_log_mass_mae"]
        + 0.05 * result["substitute_brier"]
        + 0.1 * result["aggregate_jensen_shannon_divergence"]
    )
    return result


def make_loader(dataset: Dataset, config: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: dict[str, Any],
    smoke: bool,
    resume: bool,
    evaluate_only: bool,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    side_years = np.repeat(arrays["season_start_year"], 2)
    datasets: dict[str, Dataset] = {}
    split_counts: dict[str, int] = {}
    for split, years in SPLIT_YEARS.items():
        indices = np.flatnonzero(np.isin(side_years, years))
        base = MatchSideDataset(frame, arrays, indices)
        if smoke:
            base = Subset(base, range(min(len(base), 32 if split == "train" else 16)))
        datasets[split] = base
        split_counts[split] = len(base)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player_ae, _ = load_frozen_autoencoder("player", device)
    team_ae, _ = load_frozen_autoencoder("team", device)
    model = PredictionV1(config, metadata, player_ae, team_ae).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["mixed_precision"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_loader = make_loader(datasets["train"], config, True, seed)
    dev_loader = make_loader(datasets["dev"], config, False, seed)
    run_suffix = "_smoke" if smoke else ""
    checkpoint_path = ROOT / "checkpoints" / f"player_prediction_v1_best{run_suffix}.pt"
    history_path = ROOT / "reports" / f"player_prediction_v1_history{run_suffix}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    start_epoch = 1
    patience_count = 0
    max_epochs = 1 if smoke else int(config["max_epochs"])
    started = time.time()

    if (resume or evaluate_only) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not evaluate_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_dev_metrics"]["composite_score"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if evaluate_only:
            start_epoch = max_epochs + 1
            print(f"Evaluating prediction V1 checkpoint epoch={best_epoch} dev={best_score:.6f}", flush=True)
        else:
            start_epoch = best_epoch + 1
            history = [record for record in history if int(record["epoch"]) <= best_epoch]
            print(f"Resuming prediction V1 from epoch={best_epoch} dev={best_score:.6f}", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running: defaultdict[str, float] = defaultdict(float)
        seen = 0
        consistency_factor = min(1.0, max(0.0, (epoch - int(config["consistency_warmup_epochs"])) / 3.0))
        spatial_interval = int(config.get("spatial_loss_every", 1))
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"prediction v1 epoch {epoch}", unit="batch", disable=True)
        ):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            decode_spatial = batch_index % spatial_interval == 0
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                output = model(batch, decode_spatial=decode_spatial)
                loss, parts = compute_loss(
                    batch,
                    output,
                    model,
                    config["weights"],
                    consistency_factor,
                    float(spatial_interval),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                float(config["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(batch["side"])
            seen += batch_size
            for key, value in parts.items():
                running[key] += value * batch_size
        dev = evaluate(model, dev_loader, device, f"prediction v1 dev epoch {epoch}")
        score = float(dev["composite_score"])
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "consistency_factor": consistency_factor,
            "train_losses": {key: value / seen for key, value in running.items()},
            "dev_metrics": dev,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json(history_path, history)
        improved = score < best_score - float(config["min_delta"])
        if improved:
            best_score = score
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metadata": metadata,
                    "best_dev_metrics": dev,
                    "trainable_parameter_count": trainable,
                    "frozen_parameter_count": frozen,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
        print(f"epoch={epoch} dev={score:.6f} best={best_score:.6f} patience={patience_count}/{config['early_stopping_patience']}", flush=True)
        if (
            not smoke
            and epoch >= int(config.get("min_epochs", 1))
            and patience_count >= int(config["early_stopping_patience"])
        ):
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    for split in ("train", "dev", "val", "test"):
        metrics[split] = evaluate(
            model, make_loader(datasets[split], config, False, seed), device, f"prediction v1 final {split}"
        )
    result = {
        "seed": seed,
        "best_epoch_by_dev": best_epoch,
        "last_epoch_trained": history[-1]["epoch"],
        "best_dev_score_during_training": best_score,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": frozen,
        "split_team_side_counts": split_counts,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "config": config,
        "smoke_run": smoke,
    }
    write_json(ROOT / "reports" / f"player_prediction_v1_metrics{run_suffix}.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(f"Loading {args.data} ...", flush=True)
    frame = pd.read_pickle(args.data)
    frame, arrays, metadata = load_or_build_cache(frame, config, args.rebuild_cache)
    if args.cache_only:
        print("Token cache ready.", flush=True)
        return
    result = train(frame, arrays, metadata, config, args.smoke, args.resume, args.evaluate_only)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
