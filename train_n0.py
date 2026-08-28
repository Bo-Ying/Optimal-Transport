from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
H1_DIR = ROOT / "H1"
R7_DIR = ROOT / "R7"
if str(H1_DIR) not in sys.path:
    sys.path.insert(0, str(H1_DIR))
if str(R7_DIR) not in sys.path:
    sys.path.insert(0, str(R7_DIR))

from train_player_prediction_v1 import (  # noqa: E402
    PredictionV1,
    load_frozen_autoencoder,
)
from train_coordcnn_ae import EPS, HEIGHT, WIDTH  # noqa: E402
from train_r7 import (  # noqa: E402
    DEFAULT_RATING_PRIOR,
    NROWS,
    POSITION_VALUES,
    R7Model,
    batch_from_indices as r7_batch_from_indices,
    code_maps as r7_code_maps,
    extract_match_metadata_and_lineups,
    extract_positions_and_ratings,
    infer_context_from_position_order,
    make_tensors as r7_make_tensors,
    predict as r7_predict,
    split_id_from_date,
)


SPLIT_IDS = {"train": 0, "dev": 1, "val": 2, "test": 3}
SPLIT_YEARS = {
    "train": tuple(range(2015, 2023)),
    "dev": (2023,),
    "val": (2024,),
    "test": (2025,),
}
WDL_NAMES = ("home_win", "draw", "away_win")
POSITION_GROUPS = {
    "GK": "GK",
    "DC": "DEF",
    "DL": "DEF",
    "DR": "DEF",
    "DMC": "MID",
    "DML": "MID",
    "DMR": "MID",
    "MC": "MID",
    "ML": "MID",
    "MR": "MID",
    "AMC": "ATT",
    "AML": "ATT",
    "AMR": "ATT",
    "FW": "ATT",
    "FWL": "ATT",
    "FWR": "ATT",
    "Sub": "SUB",
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def total_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def season_split_from_year(year: int) -> int:
    for split, years in SPLIT_YEARS.items():
        if int(year) in years:
            return SPLIT_IDS[split]
    return -1


def load_h1_model(device: torch.device) -> tuple[PredictionV1, dict[str, Any]]:
    checkpoint = torch.load(
        H1_DIR / "checkpoints" / "player_prediction_v1_best.pt",
        map_location=device,
        weights_only=False,
    )
    player_ae, _ = load_frozen_autoencoder("player", device)
    team_ae, _ = load_frozen_autoencoder("team", device)
    model = PredictionV1(checkpoint["config"], checkpoint["metadata"], player_ae, team_ae).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def load_r7_model(device: torch.device) -> tuple[R7Model, dict[str, Any]]:
    checkpoint = torch.load(R7_DIR / "best_model_seed42.pt", map_location=device, weights_only=False)
    model = R7Model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, checkpoint


def h1_side_batch(arrays: dict[str, np.ndarray], match_indices: np.ndarray, side: int, device: torch.device) -> dict[str, torch.Tensor]:
    roster_size = arrays["roster_valid"].shape[1] // 2
    total_slots = 2 * roster_size
    slots = np.arange(side * roster_size, (side + 1) * roster_size)
    history = arrays["player_history"][match_indices[:, None], slots[None, :]]
    valid = history >= 0
    safe = np.maximum(history, 0)
    hist_match, hist_slot = np.divmod(safe, total_slots)
    current_day = arrays["date_days"][match_indices]
    hist_days = arrays["date_days"][hist_match]

    team_history = arrays["team_history"][match_indices, side]
    team_valid = team_history >= 0
    safe_team = np.maximum(team_history, 0)
    team_hist_match, team_hist_side = np.divmod(safe_team, 2)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).to(device)

    batch = {
        "side": torch.full((len(match_indices),), side, dtype=torch.long, device=device),
        "league": tensor(arrays["league_ids"][match_indices].astype(np.int64)),
        "roster_valid": tensor(arrays["roster_valid"][match_indices[:, None], slots[None, :]]),
        "player_hist_token": tensor(arrays["player_tokens"][hist_match, hist_slot].astype(np.float32)),
        "player_hist_mass": tensor(arrays["player_mass"][hist_match, hist_slot]),
        "player_hist_rating": tensor(arrays["player_rating"][hist_match, hist_slot]),
        "player_hist_minutes": tensor(arrays["player_minutes"][hist_match, hist_slot]),
        "player_hist_position": tensor(arrays["player_position"][hist_match, hist_slot].astype(np.int64)),
        "player_hist_starter": tensor(arrays["player_starter"][hist_match, hist_slot].astype(np.int64)),
        "player_hist_side": tensor(arrays["player_side"][hist_match, hist_slot].astype(np.int64)),
        "player_hist_league": tensor(arrays["league_ids"][hist_match].astype(np.int64)),
        "player_hist_recency": tensor(np.log1p(np.maximum(current_day[:, None, None] - hist_days, 0)).astype(np.float32)),
        "player_hist_valid": tensor(valid),
        "current_position": tensor(arrays["player_position"][match_indices[:, None], slots[None, :]].astype(np.int64)),
        "current_starter": tensor(arrays["player_starter"][match_indices[:, None], slots[None, :]].astype(np.int64)),
        "current_age": tensor(arrays["player_age"][match_indices[:, None], slots[None, :]]),
        "current_height": tensor(arrays["player_height"][match_indices[:, None], slots[None, :]]),
        "current_weight": tensor(arrays["player_weight"][match_indices[:, None], slots[None, :]]),
        "team_hist_token": tensor(arrays["team_tokens"][team_hist_match, team_hist_side].astype(np.float32)),
        "team_hist_mass": tensor(arrays["team_mass"][team_hist_match, team_hist_side]),
        "team_hist_side": tensor(team_hist_side.astype(np.int64)),
        "team_hist_league": tensor(arrays["league_ids"][team_hist_match].astype(np.int64)),
        "team_hist_recency": tensor(np.log1p(np.maximum(current_day[:, None] - arrays["date_days"][team_hist_match], 0)).astype(np.float32)),
        "team_hist_valid": tensor(team_valid),
    }
    return batch


def heatmap_shape_features(density: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    batch, players, height, width = density.shape
    device = density.device
    ys = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    xs = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    safe = density.clamp_min(0.0)
    denom = safe.sum((2, 3)).clamp_min(EPS)
    cx = (safe * xs).sum((2, 3)) / denom
    cy = (safe * ys).sum((2, 3)) / denom
    dx = xs - cx[:, :, None, None]
    dy = ys - cy[:, :, None, None]
    var_x = (safe * dx.square()).sum((2, 3)) / denom
    var_y = (safe * dy.square()).sum((2, 3)) / denom
    cov_xy = (safe * dx * dy).sum((2, 3)) / denom
    entropy = -(safe.clamp_min(EPS) * safe.clamp_min(EPS).log()).sum((2, 3)) / math.log(height * width)
    pooled = F.avg_pool2d(safe.reshape(batch * players, 1, height, width), kernel_size=4, stride=4).reshape(batch, players, -1)
    top2 = torch.topk(pooled, k=2, dim=-1).values
    peak_mass = top2[:, :, 0]
    peak_ratio = top2[:, :, 1] / top2[:, :, 0].clamp_min(EPS)

    flat_idx = pooled.argmax(dim=-1)
    coarse_h = height // 4
    coarse_w = width // 4
    top1_y = (flat_idx // coarse_w).float() / max(coarse_h - 1, 1)
    top1_x = (flat_idx % coarse_w).float() / max(coarse_w - 1, 1)
    masked = pooled.scatter(2, flat_idx.unsqueeze(-1), -1.0)
    flat_idx2 = masked.argmax(dim=-1)
    top2_y = (flat_idx2 // coarse_w).float() / max(coarse_h - 1, 1)
    top2_x = (flat_idx2 % coarse_w).float() / max(coarse_w - 1, 1)
    peak_separation = torch.sqrt((top1_x - top2_x).square() + (top1_y - top2_y).square())

    zones = []
    for y_part in torch.chunk(safe, 3, dim=2):
        for zone in torch.chunk(y_part, 5, dim=3):
            zones.append(zone.sum((2, 3)) / denom)
    zone_occupancy = torch.stack(zones, dim=-1)
    return torch.cat(
        (
            torch.log1p(mass).unsqueeze(-1),
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
            torch.sqrt(var_x.clamp_min(0.0)).unsqueeze(-1),
            torch.sqrt(var_y.clamp_min(0.0)).unsqueeze(-1),
            cov_xy.unsqueeze(-1),
            entropy.unsqueeze(-1),
            peak_mass.unsqueeze(-1),
            peak_ratio.unsqueeze(-1),
            peak_separation.unsqueeze(-1),
            zone_occupancy,
        ),
        dim=-1,
    )


def pairwise_relations(density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_marginal = density.sum(dim=2)
    y_marginal = density.sum(dim=3)
    x_cdf = x_marginal.cumsum(dim=-1)
    y_cdf = y_marginal.cumsum(dim=-1)
    wx = (x_cdf[:, :, None, :-1] - x_cdf[:, None, :, :-1]).abs().sum(-1) * (2.0 / (WIDTH - 1))
    wy = (y_cdf[:, :, None, :-1] - y_cdf[:, None, :, :-1]).abs().sum(-1) * (2.0 / (HEIGHT - 1))
    wasserstein = 0.5 * (wx + wy)
    coarse = F.avg_pool2d(
        density.reshape(-1, 1, HEIGHT, WIDTH),
        kernel_size=4,
        stride=4,
    ).reshape(density.shape[0], density.shape[1], -1) * 16.0
    overlap = torch.minimum(coarse[:, :, None, :], coarse[:, None, :, :]).sum(-1)
    return wasserstein, overlap


def build_r7_match_predictions(
    data_path: Path,
    h1_arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    r7_arrays_archive = np.load(R7_DIR / "r7_player_history_features.npz")
    r7_arrays = {key: r7_arrays_archive[key] for key in r7_arrays_archive.files}
    tensors, indices = r7_make_tensors(r7_arrays)
    all_indices = np.arange(len(r7_arrays["target"]))
    r7_model, _ = load_r7_model(device)
    preds = r7_predict(r7_model, tensors, all_indices, device, 65536)

    leagues, dates, _, after_lineups_pos = extract_match_metadata_and_lineups(data_path)
    positions, ratings = extract_positions_and_ratings(data_path, after_lineups_pos)
    league_map = r7_code_maps(leagues)
    position_map = r7_code_maps(POSITION_VALUES)
    sorted_original = sorted(range(NROWS), key=lambda idx: (dates[idx], idx))
    sorted_index = {original: sorted_pos for sorted_pos, original in enumerate(sorted_original)}
    histories: dict[int, list[tuple[float, int, int, int, int]]] = defaultdict(list)
    pred_by_match_player: dict[tuple[int, int], float] = {}
    cursor = 0
    for match_idx in sorted_original:
        context = infer_context_from_position_order(positions[match_idx], position_map)
        row_items = []
        for player_id, rating in ratings[match_idx].items():
            if rating is None or player_id not in context:
                continue
            split = split_id_from_date(dates[match_idx])
            if split < 0:
                continue
            is_home, is_sub, position = context[player_id]
            row_items.append((int(player_id), float(rating), league_map[leagues[match_idx]], is_home, is_sub, position))
        row_items.sort(key=lambda item: item[0])
        for player_id, rating, league, is_home, is_sub, position in row_items:
            history = histories[player_id]
            if history:
                pred_by_match_player[(sorted_index[match_idx], player_id)] = float(preds[cursor])
                cursor += 1
            history.append((rating, league, is_home, is_sub, position))
            if len(history) > 60:
                del history[:-60]
    if cursor != len(preds):
        raise RuntimeError(f"R7 row reconstruction mismatch: mapped {cursor}, predictions {len(preds)}")

    n_matches, total_slots = h1_arrays["player_ids"].shape
    rating_pred = np.full((n_matches, total_slots), DEFAULT_RATING_PRIOR, dtype=np.float32)
    rating_known = np.zeros((n_matches, total_slots), dtype=np.float32)
    for match_idx in range(n_matches):
        for slot in range(total_slots):
            player_id = int(h1_arrays["player_ids"][match_idx, slot])
            if player_id < 0:
                continue
            value = pred_by_match_player.get((match_idx, player_id))
            if value is not None:
                rating_pred[match_idx, slot] = value
                rating_known[match_idx, slot] = 1.0
    return rating_pred, rating_known


def build_match_targets(data_path: Path, h1_arrays: dict[str, np.ndarray], out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = out_dir / "match_targets_cache.npz"
    if cache_path.exists():
        archive = np.load(cache_path)
        return archive["target"], archive["home_score"], archive["away_score"]

    import pandas as pd

    frame = pd.read_pickle(data_path)
    frame = frame.assign(_date=pd.to_datetime(frame["Match_Date"])).sort_values("_date", kind="stable").reset_index(drop=True)
    home_score = frame["Home_Score"].to_numpy(np.int16)
    away_score = frame["Away_Score"].to_numpy(np.int16)
    if len(home_score) != len(h1_arrays["season_start_year"]):
        raise RuntimeError("Target cache length does not match H1 cache length.")
    target = np.where(home_score > away_score, 0, np.where(home_score == away_score, 1, 2)).astype(np.int64)
    np.savez(cache_path, target=target, home_score=home_score, away_score=away_score)
    return target, home_score, away_score


@torch.no_grad()
def build_frozen_feature_cache(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out_dir = Path(args.output_dir)
    feature_path = out_dir / "n0_frozen_features.npz"
    meta_path = out_dir / "n0_frozen_features.json"
    if feature_path.exists() and meta_path.exists() and not args.rebuild_cache:
        archive = np.load(feature_path)
        return {key: archive[key] for key in archive.files}, json.loads(meta_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    h1_archive = np.load(H1_DIR / "artifacts" / "prediction_v1_token_cache.npz")
    h1_arrays = {key: h1_archive[key] for key in h1_archive.files}
    h1_meta = json.loads((H1_DIR / "artifacts" / "prediction_v1_token_cache_meta.json").read_text(encoding="utf-8"))
    data_path = Path(args.data)
    targets, home_score, away_score = build_match_targets(data_path, h1_arrays, out_dir)
    r7_rating, r7_known = build_r7_match_predictions(data_path, h1_arrays, device)
    h1_model, h1_checkpoint = load_h1_model(device)

    n_matches, total_slots = h1_arrays["player_ids"].shape
    roster_size = total_slots // 2
    shape_dim = 25
    heatmap_features = np.zeros((n_matches, total_slots, shape_dim), dtype=np.float32)
    h1_pred_mass = np.zeros((n_matches, total_slots), dtype=np.float32)
    wasserstein = np.zeros((n_matches, total_slots, total_slots), dtype=np.float16)
    overlap = np.zeros((n_matches, total_slots, total_slots), dtype=np.float16)

    for start in range(0, n_matches, args.precompute_batch_size):
        match_indices = np.arange(start, min(start + args.precompute_batch_size, n_matches))
        side_outputs = []
        for side in (0, 1):
            batch = h1_side_batch(h1_arrays, match_indices, side, device)
            output = h1_model(batch, decode_spatial=True)
            density = output["player_density"].float()
            mass = output["player_mass"].float() * batch["roster_valid"].float()
            features = heatmap_shape_features(density, mass)
            side_outputs.append((density, mass, features))

        density_full = torch.cat((side_outputs[0][0], side_outputs[1][0]), dim=1)
        mass_full = torch.cat((side_outputs[0][1], side_outputs[1][1]), dim=1)
        feature_full = torch.cat((side_outputs[0][2], side_outputs[1][2]), dim=1)
        relational_density = density_full.clone()
        relational_density[:, roster_size:] = torch.flip(relational_density[:, roster_size:], dims=(-2, -1))
        w, o = pairwise_relations(relational_density)
        valid = torch.from_numpy(h1_arrays["roster_valid"][match_indices]).to(device)
        pair_mask = valid[:, :, None] & valid[:, None, :]
        w = torch.where(pair_mask, w, torch.zeros_like(w))
        o = torch.where(pair_mask, o, torch.zeros_like(o))

        heatmap_features[match_indices] = feature_full.cpu().numpy().astype(np.float32)
        h1_pred_mass[match_indices] = mass_full.cpu().numpy().astype(np.float32)
        wasserstein[match_indices] = w.cpu().numpy().astype(np.float16)
        overlap[match_indices] = o.cpu().numpy().astype(np.float16)
        done = int(match_indices[-1] + 1)
        if done == n_matches or done % 512 < args.precompute_batch_size:
            print(f"precomputed H1 features {done}/{n_matches}", flush=True)

    split = np.asarray([season_split_from_year(year) for year in h1_arrays["season_start_year"]], dtype=np.int8)
    position_group_names = ["GK", "DEF", "MID", "ATT", "SUB"]
    position_group_map = {name: idx for idx, name in enumerate(position_group_names)}
    position_id_to_name = {idx + 1: name for idx, name in enumerate(h1_meta["positions"])}
    position_group = np.zeros_like(h1_arrays["player_position"], dtype=np.int64)
    for pos_id, name in position_id_to_name.items():
        position_group[h1_arrays["player_position"] == pos_id] = position_group_map[POSITION_GROUPS.get(name, "SUB")]

    arrays = {
        "valid": h1_arrays["roster_valid"].astype(np.bool_),
        "league": h1_arrays["league_ids"].astype(np.int64),
        "side": h1_arrays["player_side"].astype(np.int64),
        "starter": h1_arrays["player_starter"].astype(np.int64),
        "position": h1_arrays["player_position"].astype(np.int64),
        "position_group": position_group.astype(np.int64),
        "r7_rating": r7_rating.astype(np.float32),
        "r7_known": r7_known.astype(np.float32),
        "h1_mass": h1_pred_mass.astype(np.float32),
        "heatmap_features": heatmap_features.astype(np.float32),
        "wasserstein": wasserstein,
        "overlap": overlap,
        "target": targets.astype(np.int64),
        "split": split,
        "home_score": home_score,
        "away_score": away_score,
    }
    np.savez(feature_path, **arrays)
    metadata = {
        "source_data": str(data_path),
        "h1_checkpoint": str(H1_DIR / "checkpoints" / "player_prediction_v1_best.pt"),
        "r7_checkpoint": str(R7_DIR / "best_model_seed42.pt"),
        "h1_epoch": int(h1_checkpoint["epoch"]),
        "h1_trainable_parameter_count": int(h1_checkpoint["trainable_parameter_count"]),
        "h1_frozen_parameter_count": int(h1_checkpoint["frozen_parameter_count"]),
        "feature_names": {
            "heatmap_features": [
                "log1p_h1_mass",
                "centroid_x",
                "centroid_y",
                "spread_x",
                "spread_y",
                "covariance_xy",
                "entropy",
                "peak_mass",
                "peak_ratio",
                "peak_separation",
            ]
            + [f"zone_{row}_{col}" for row in range(3) for col in range(5)],
            "wdl_order": list(WDL_NAMES),
            "position_groups": position_group_names,
        },
        "split_ids": SPLIT_IDS,
        "away_rotation_rule": "Only away heatmaps used for relational Wasserstein/overlap are rotated 180 degrees.",
        "single_player_feature_rotation": "No rotation.",
    }
    write_json(meta_path, metadata)
    return arrays, metadata


class N0Dataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "valid": torch.from_numpy(self.arrays["valid"][i]),
            "league": torch.tensor(int(self.arrays["league"][i]), dtype=torch.long),
            "side": torch.from_numpy(self.arrays["side"][i]),
            "starter": torch.from_numpy(self.arrays["starter"][i]),
            "position": torch.from_numpy(self.arrays["position"][i]),
            "position_group": torch.from_numpy(self.arrays["position_group"][i]),
            "r7_rating": torch.from_numpy(self.arrays["r7_rating"][i]),
            "r7_known": torch.from_numpy(self.arrays["r7_known"][i]),
            "h1_mass": torch.from_numpy(self.arrays["h1_mass"][i]),
            "heatmap_features": torch.from_numpy(self.arrays["heatmap_features"][i]),
            "wasserstein": torch.from_numpy(self.arrays["wasserstein"][i].astype(np.float32)),
            "overlap": torch.from_numpy(self.arrays["overlap"][i].astype(np.float32)),
            "target": torch.tensor(int(self.arrays["target"][i]), dtype=torch.long),
        }


class N0Model(nn.Module):
    def __init__(self, n_leagues: int, n_positions: int, n_position_groups: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.rating_mean = DEFAULT_RATING_PRIOR
        self.rating_std = 0.7
        self.league_embedding = nn.Embedding(n_leagues + 1, 8, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 4)
        self.starter_embedding = nn.Embedding(2, 4)
        self.position_embedding = nn.Embedding(n_positions + 1, 12, padding_idx=0)
        self.group_embedding = nn.Embedding(n_position_groups, 8)
        base_dim = 25 + 4
        self.base_projection = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        context_dim = hidden_dim + 8 + 4 + 4 + 12 + 8
        self.player_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch["valid"].bool()
        rating_z = ((batch["r7_rating"].float() - self.rating_mean) / self.rating_std).unsqueeze(-1)
        rating_known = batch["r7_known"].float().unsqueeze(-1)
        mass_z = torch.log1p(batch["h1_mass"].float()).unsqueeze(-1) / 8.0
        side_sign = (batch["side"].float() * 2.0 - 1.0).unsqueeze(-1)
        base = torch.cat(
            (
                batch["heatmap_features"].float(),
                rating_z,
                rating_known,
                mass_z,
                side_sign,
            ),
            dim=-1,
        )
        base_token = self.base_projection(base)
        q = self.query(base_token)
        k = self.key(base_token)
        v = self.value(base_token)
        content = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        relation = torch.stack((batch["wasserstein"].float(), batch["overlap"].float()), dim=-1)
        logits = content + self.relation_bias(relation).squeeze(-1)
        logits = logits.masked_fill(~valid[:, None, :], -1e9)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, v)

        league = batch["league"].long().unsqueeze(1).expand(-1, valid.shape[1])
        context = torch.cat(
            (
                attended,
                self.league_embedding(league),
                self.side_embedding(batch["side"].long()),
                self.starter_embedding(batch["starter"].long()),
                self.position_embedding(batch["position"].long()),
                self.group_embedding(batch["position_group"].long()),
            ),
            dim=-1,
        )
        player_repr = self.player_projection(context)
        pooled = []
        for side in (0, 1):
            side_mask = valid & (batch["side"].long() == side)
            group_emb = self.group_embedding(batch["position_group"].long())
            scores = self.pool_score(torch.cat((player_repr, group_emb), dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~side_mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled.append(torch.sum(weights.unsqueeze(-1) * player_repr, dim=1))
        home, away = pooled
        league_context = self.league_embedding(batch["league"].long())
        return self.out(torch.cat((home, away, home - away, home * away, league_context), dim=-1))


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def make_loaders(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, DataLoader]:
    loaders = {}
    generator = torch.Generator().manual_seed(args.seed)
    for split_name, split_id in SPLIT_IDS.items():
        indices = np.flatnonzero(arrays["split"] == split_id)
        dataset = N0Dataset(arrays, indices)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            generator=generator if split_name == "train" else None,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
        )
    return loaders


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def brier_score(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(target, num_classes=3).float()
    return (prob - one_hot).square().sum(dim=1)


@torch.no_grad()
def evaluate(model: N0Model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_brier = 0.0
    total_ce = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)
        target = batch["target"]
        total_brier += float(brier_score(prob, target).sum().cpu())
        total_ce += float(F.cross_entropy(logits, target, reduction="sum").cpu())
        total_correct += int((logits.argmax(dim=1) == target).sum().cpu())
        total += int(target.numel())
    return {
        "brier": total_brier / max(total, 1),
        "cross_entropy": total_ce / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "n": total,
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    arrays, feature_metadata = build_frozen_feature_cache(args, device)
    loaders = make_loaders(arrays, args)
    n_leagues = int(arrays["league"].max())
    n_positions = int(arrays["position"].max())
    n_position_groups = int(arrays["position_group"].max()) + 1
    model = N0Model(n_leagues, n_positions, n_position_groups, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_dev = float("inf")
    best_val_epoch = None
    best_dev_epoch = None
    best_val_state = None
    stale_dev_epochs = 0
    log_path = out_dir / f"training_log_seed{args.seed}.csv"
    best_path = out_dir / f"best_model_seed{args.seed}.pt"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_brier",
        "dev_brier",
        "val_brier",
        "train_cross_entropy",
        "dev_cross_entropy",
        "val_cross_entropy",
        "seconds",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for batch in loaders["train"]:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["target"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = {
                split: evaluate(model, loaders[split], device)
                for split in ("train", "dev", "val")
            }
            elapsed = time.time() - start
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_brier": metrics["train"]["brier"],
                    "dev_brier": metrics["dev"]["brier"],
                    "val_brier": metrics["val"]["brier"],
                    "train_cross_entropy": metrics["train"]["cross_entropy"],
                    "dev_cross_entropy": metrics["dev"]["cross_entropy"],
                    "val_cross_entropy": metrics["val"]["cross_entropy"],
                    "seconds": elapsed,
                }
            )
            handle.flush()
            print(
                f"epoch {epoch:03d} loss={np.mean(losses):.5f} "
                f"train_brier={metrics['train']['brier']:.5f} "
                f"dev_brier={metrics['dev']['brier']:.5f} "
                f"val_brier={metrics['val']['brier']:.5f} seconds={elapsed:.1f}",
                flush=True,
            )

            if metrics["dev"]["brier"] + args.min_delta < best_dev:
                best_dev = metrics["dev"]["brier"]
                best_dev_epoch = epoch
                stale_dev_epochs = 0
            else:
                stale_dev_epochs += 1

            if metrics["val"]["brier"] < best_val:
                best_val = metrics["val"]["brier"]
                best_val_epoch = epoch
                best_val_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(
                    {
                        "model_state_dict": best_val_state,
                        "seed": args.seed,
                        "epoch": epoch,
                        "best_val_brier": best_val,
                        "best_dev_brier_seen": best_dev,
                        "trainable_parameter_count": parameter_count(model),
                        "feature_metadata": feature_metadata,
                        "model_config": {
                            "hidden_dim": args.hidden_dim,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "batch_size": args.batch_size,
                        },
                    },
                    best_path,
                )

            if epoch >= args.min_epochs and stale_dev_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}; best dev epoch {best_dev_epoch}", flush=True)
                break

    if best_val_state is not None:
        model.load_state_dict(best_val_state)
    final_metrics = {split: evaluate(model, loaders[split], device) for split in SPLIT_IDS}
    result = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "selected_epoch_by_val_brier": best_val_epoch,
        "best_val_brier": best_val,
        "best_dev_epoch_for_early_stopping": best_dev_epoch,
        "best_dev_brier": best_dev,
        "metrics": final_metrics,
        "test_accuracy": final_metrics["test"]["accuracy"],
        "test_cross_entropy": final_metrics["test"]["cross_entropy"],
        "parameter_count": {
            "n0_trainable": parameter_count(model),
            "n0_total": total_parameter_count(model),
            "h1_trainable_when_originally_trained": int(feature_metadata.get("h1_trainable_parameter_count", 0)),
            "h1_frozen_when_originally_trained": int(feature_metadata.get("h1_frozen_parameter_count", 0)),
            "r7_trainable_when_originally_trained": 110,
            "frozen_upstream_total": int(feature_metadata.get("h1_trainable_parameter_count", 0))
            + int(feature_metadata.get("h1_frozen_parameter_count", 0))
            + 110,
        },
        "artifacts": {
            "feature_cache": str(out_dir / "n0_frozen_features.npz"),
            "training_log": str(log_path),
            "best_model": str(best_path),
        },
        "feature_metadata": feature_metadata,
    }
    write_json(out_dir / f"metrics_seed{args.seed}.json", final_metrics)
    write_json(out_dir / f"report_seed{args.seed}.json", result)
    print("FINAL_RESULT_JSON_START", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("FINAL_RESULT_JSON_END", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "DATA" / "dataset_player_KDE_clean_enriched.pkl"))
    parser.add_argument("--output-dir", default=str(ROOT / "N0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--precompute-batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
