"""Evaluate player-history heatmap forecasting baselines with H1 metrics."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from train_coordcnn_ae import EPS, HEIGHT, WIDTH, batch_metrics
from train_player_prediction_v1 import SPLIT_YEARS, normalized_density, season_start_year


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "DATA" / "dataset_player_KDE_clean_enriched.pkl"
DEFAULT_OUTPUT = ROOT / "reports" / "player_history_baselines_metrics.json"
DEFAULT_SUMMARY = ROOT / "reports" / "player_history_baselines_summary.md"
BASELINES = {
    "last_player_heatmap": 1,
    "avg10_player_heatmap": 10,
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n", encoding="utf-8")


@dataclass
class RollingDensityHistory:
    maxlen: int
    densities: deque[np.ndarray] = field(default_factory=deque)
    masses: deque[float] = field(default_factory=deque)
    density_sum: np.ndarray | None = None
    mass_sum: float = 0.0

    def add(self, density: np.ndarray, mass: float) -> None:
        stored = density.astype(np.float16, copy=True)
        if len(self.densities) == self.maxlen:
            removed_density = self.densities.popleft()
            removed_mass = self.masses.popleft()
            self.density_sum -= removed_density.astype(np.float32, copy=False)
            self.mass_sum -= float(removed_mass)
        self.densities.append(stored)
        self.masses.append(float(mass))
        if self.density_sum is None:
            self.density_sum = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.density_sum += stored.astype(np.float32, copy=False)
        self.mass_sum += float(mass)

    def has_history(self) -> bool:
        return bool(self.densities)

    def predict(self, window: int, fallback_density: np.ndarray, fallback_mass: float) -> tuple[np.ndarray, float, bool]:
        if not self.densities:
            return fallback_density, fallback_mass, True
        if window == 1:
            return self.densities[-1].astype(np.float32, copy=False), float(self.masses[-1]), False
        count = min(window, len(self.densities))
        if count == len(self.densities) and self.density_sum is not None:
            density = self.density_sum / float(count)
            mass = self.mass_sum / float(count)
            return density, mass, False
        density = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        mass = 0.0
        for hist_density, hist_mass in list(zip(self.densities, self.masses))[-count:]:
            density += hist_density.astype(np.float32, copy=False)
            mass += float(hist_mass)
        return density / float(count), mass / float(count), False


class MetricAccumulator:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.totals: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self._player_target: list[np.ndarray] = []
        self._player_prediction: list[np.ndarray] = []
        self._team_target: list[np.ndarray] = []
        self._team_prediction: list[np.ndarray] = []
        self._aggregate_target: list[np.ndarray] = []
        self._aggregate_prediction: list[np.ndarray] = []
        self.cold_start_player_appearances = 0
        self.cold_start_roster_slots = 0
        self.cold_start_team_sides = 0

    def add_player_density(self, target: np.ndarray, prediction: np.ndarray) -> None:
        self._player_target.append(target)
        self._player_prediction.append(prediction)
        if len(self._player_target) >= self.batch_size:
            self.flush_player()

    def add_team_density(self, target: np.ndarray, prediction: np.ndarray) -> None:
        self._team_target.append(target)
        self._team_prediction.append(prediction)
        if len(self._team_target) >= self.batch_size:
            self.flush_team()

    def add_aggregate_density(self, target: np.ndarray, prediction: np.ndarray) -> None:
        self._aggregate_target.append(target)
        self._aggregate_prediction.append(prediction)
        if len(self._aggregate_target) >= self.batch_size:
            self.flush_aggregate()

    def add_player_mass(self, target_mass: float, prediction_mass: float) -> None:
        self.totals["player_log_mass_mae"] += abs(math.log1p(target_mass) - math.log1p(max(prediction_mass, 0.0)))
        self.totals["player_mass_mae"] += abs(target_mass - prediction_mass)
        self.counts["player"] += 1

    def add_substitute(self, target_appeared: bool, prediction_probability: float) -> None:
        target = 1.0 if target_appeared else 0.0
        probability = min(max(float(prediction_probability), EPS), 1.0 - EPS)
        self.totals["substitute_brier"] += (target - probability) ** 2
        self.totals["substitute_bce"] += -(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))
        self.counts["substitute"] += 1

    def add_team_mass(self, target_mass: float, prediction_mass: float) -> None:
        self.totals["team_log_mass_mae"] += abs(math.log1p(target_mass) - math.log1p(max(prediction_mass, 0.0)))
        self.counts["team"] += 1

    def add_aggregate_mass(self, target_mass: float, prediction_mass: float) -> None:
        self.totals["aggregate_mass_log_mae"] += abs(math.log1p(target_mass) - math.log1p(max(prediction_mass, EPS)))

    def flush_player(self) -> None:
        self._flush_density("player", self._player_target, self._player_prediction)

    def flush_team(self) -> None:
        self._flush_density("team", self._team_target, self._team_prediction)

    def flush_aggregate(self) -> None:
        self._flush_density("aggregate", self._aggregate_target, self._aggregate_prediction)

    def _flush_density(self, prefix: str, targets: list[np.ndarray], predictions: list[np.ndarray]) -> None:
        if not targets:
            return
        target = torch.from_numpy(np.stack(targets)).unsqueeze(1).float()
        prediction = torch.from_numpy(np.stack(predictions)).unsqueeze(1).float()
        metrics = batch_metrics(target, prediction)
        for key, values in metrics.items():
            self.totals[f"{prefix}_{key}"] += float(values.sum())
        targets.clear()
        predictions.clear()

    def finalize(self) -> dict[str, float]:
        self.flush_player()
        self.flush_team()
        self.flush_aggregate()
        result: dict[str, float] = {}
        for key, value in self.totals.items():
            if key.startswith("substitute"):
                denominator = self.counts["substitute"]
            elif key.startswith("player"):
                denominator = self.counts["player"]
            else:
                denominator = self.counts["team"]
            result[key] = value / max(denominator, 1)
        result["num_player_appearances"] = self.counts["player"]
        result["num_substitute_candidates"] = self.counts["substitute"]
        result["num_team_sides"] = self.counts["team"]
        result["cold_start_player_appearances"] = self.cold_start_player_appearances
        result["cold_start_roster_slots"] = self.cold_start_roster_slots
        result["cold_start_team_sides"] = self.cold_start_team_sides
        required = (
            "player_jensen_shannon_divergence",
            "player_marginal_wasserstein_1",
            "player_centroid_error_l2_pitch_norm",
            "player_log_mass_mae",
            "substitute_brier",
            "aggregate_jensen_shannon_divergence",
        )
        if all(key in result for key in required):
            result["composite_score"] = (
                result["player_jensen_shannon_divergence"]
                + 0.1 * result["player_marginal_wasserstein_1"]
                + 0.05 * result["player_centroid_error_l2_pitch_norm"]
                + 0.02 * result["player_log_mass_mae"]
                + 0.05 * result["substitute_brier"]
                + 0.1 * result["aggregate_jensen_shannon_divergence"]
            )
        else:
            result["composite_score"] = float("nan")
        return result


def split_for_year(year: int) -> str | None:
    for split, years in SPLIT_YEARS.items():
        if year in years:
            return split
    return None


def roster_for_side(row: pd.Series, side: str) -> tuple[list[int], int]:
    starters = [int(player_id) for player_id in row[f"Starting_11({side})"]]
    substitutes = [int(player_id) for player_id in row[f"Substitute({side})"]]
    return starters + substitutes, len(starters)


def compute_train_priors(frame: pd.DataFrame) -> dict[str, Any]:
    years = season_start_year(frame["Match_Date"])
    train_mask = np.isin(years, SPLIT_YEARS["train"])
    player_density_sum = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    team_density_sum = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    player_mass_sum = 0.0
    team_mass_sum = 0.0
    player_count = 0
    team_count = 0
    substitute_count = 0
    substitute_appeared = 0

    for _, row in tqdm(frame.loc[train_mask].iterrows(), total=int(train_mask.sum()), desc="Fitting train priors", unit="match"):
        player_heatmaps = row["Player_Heatmap"]
        for heatmap in player_heatmaps.values():
            density, mass = normalized_density(heatmap)
            if mass <= EPS:
                continue
            player_density_sum += density
            player_mass_sum += mass
            player_count += 1
        for side in ("Home", "Away"):
            team_density, team_mass = normalized_density(row[f"Team_Heatmap({side})"])
            if team_mass > EPS:
                team_density_sum += team_density
                team_mass_sum += team_mass
                team_count += 1
            for player_id in row[f"Substitute({side})"]:
                substitute_count += 1
                _, mass = normalized_density(player_heatmaps[int(player_id)])
                substitute_appeared += int(mass > EPS)

    player_density = (player_density_sum / max(player_count, 1)).astype(np.float32)
    player_density /= max(float(player_density.sum(dtype=np.float64)), EPS)
    team_density = (team_density_sum / max(team_count, 1)).astype(np.float32)
    team_density /= max(float(team_density.sum(dtype=np.float64)), EPS)
    return {
        "player_density": player_density,
        "player_mass": player_mass_sum / max(player_count, 1),
        "team_density": team_density,
        "team_mass": team_mass_sum / max(team_count, 1),
        "substitute_probability": substitute_appeared / max(substitute_count, 1),
        "num_train_player_appearances": player_count,
        "num_train_team_sides": team_count,
        "num_train_substitute_candidates": substitute_count,
        "num_train_substitute_appearances": substitute_appeared,
    }


def evaluate_baselines(frame: pd.DataFrame, batch_size: int, smoke: bool) -> dict[str, Any]:
    started = time.time()
    frame = frame.assign(_date=pd.to_datetime(frame["Match_Date"]))
    frame = frame.sort_values("_date", kind="stable").reset_index(drop=True)
    frame["season_start_year"] = season_start_year(frame["Match_Date"])
    if smoke:
        keep_dates = frame["_date"].drop_duplicates().head(80)
        frame = frame[frame["_date"].isin(keep_dates)].reset_index(drop=True)

    priors = compute_train_priors(frame)
    split_counts = {split: 0 for split in SPLIT_YEARS}
    metrics = {
        name: {split: MetricAccumulator(batch_size=batch_size) for split in SPLIT_YEARS}
        for name in BASELINES
    }
    player_history: dict[int, RollingDensityHistory] = {}
    team_history: dict[int, RollingDensityHistory] = {}

    grouped = frame.groupby("_date", sort=False)
    for _, group in tqdm(grouped, total=frame["_date"].nunique(), desc="Evaluating history baselines", unit="date"):
        for row_index, row in group.iterrows():
            split = split_for_year(int(row["season_start_year"]))
            if split is None:
                continue
            player_heatmaps = row["Player_Heatmap"]
            for side_index, side in enumerate(("Home", "Away")):
                split_counts[split] += 1
                roster, num_starters = roster_for_side(row, side)
                target_team_density, target_team_mass = normalized_density(row[f"Team_Heatmap({side})"])
                team_id = int(row[f"{side}_Team_ID"])
                current_team_history = team_history.get(team_id)

                side_predictions: dict[str, list[tuple[np.ndarray, float, float]]] = {name: [] for name in BASELINES}
                for local_slot, player_id in enumerate(roster):
                    target_density, target_mass = normalized_density(player_heatmaps[player_id])
                    current_player_history = player_history.get(player_id)
                    target_appeared = target_mass > EPS
                    is_starter = local_slot < num_starters

                    for baseline_name, window in BASELINES.items():
                        accumulator = metrics[baseline_name][split]
                        if current_player_history is None:
                            pred_density, pred_mass, cold_start = priors["player_density"], priors["player_mass"], True
                        else:
                            pred_density, pred_mass, cold_start = current_player_history.predict(
                                window,
                                priors["player_density"],
                                priors["player_mass"],
                            )
                        appearance_probability = 1.0 if is_starter else float(priors["substitute_probability"])
                        side_predictions[baseline_name].append((pred_density, pred_mass, appearance_probability))
                        if cold_start:
                            accumulator.cold_start_roster_slots += 1
                        if target_appeared:
                            accumulator.add_player_density(target_density, pred_density)
                            accumulator.add_player_mass(target_mass, pred_mass)
                            if cold_start:
                                accumulator.cold_start_player_appearances += 1
                        if not is_starter:
                            accumulator.add_substitute(target_appeared, appearance_probability)

                for baseline_name, window in BASELINES.items():
                    accumulator = metrics[baseline_name][split]
                    if current_team_history is None:
                        pred_team_density, pred_team_mass, cold_team = priors["team_density"], priors["team_mass"], True
                    else:
                        pred_team_density, pred_team_mass, cold_team = current_team_history.predict(
                            window,
                            priors["team_density"],
                            priors["team_mass"],
                        )
                    if cold_team:
                        accumulator.cold_start_team_sides += 1
                    accumulator.add_team_density(target_team_density, pred_team_density)
                    accumulator.add_team_mass(target_team_mass, pred_team_mass)

                    aggregate = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                    for pred_density, pred_mass, appearance_probability in side_predictions[baseline_name]:
                        aggregate += float(appearance_probability) * float(pred_mass) * pred_density
                    aggregate_mass = float(aggregate.sum(dtype=np.float64))
                    if aggregate_mass <= EPS:
                        aggregate_density = priors["team_density"]
                    else:
                        aggregate_density = aggregate / aggregate_mass
                    accumulator.add_aggregate_density(target_team_density, aggregate_density)
                    accumulator.add_aggregate_mass(target_team_mass, aggregate_mass)

        for _, row in group.iterrows():
            player_heatmaps = row["Player_Heatmap"]
            for side in ("Home", "Away"):
                roster, _ = roster_for_side(row, side)
                for player_id in roster:
                    density, mass = normalized_density(player_heatmaps[player_id])
                    if mass <= EPS:
                        continue
                    history = player_history.setdefault(player_id, RollingDensityHistory(maxlen=10))
                    history.add(density, mass)
                team_id = int(row[f"{side}_Team_ID"])
                team_density, team_mass = normalized_density(row[f"Team_Heatmap({side})"])
                if team_mass > EPS:
                    history = team_history.setdefault(team_id, RollingDensityHistory(maxlen=10))
                    history.add(team_density, team_mass)

    return {
        "baselines": {
            baseline_name: {
                split: accumulator.finalize()
                for split, accumulator in split_metrics.items()
            }
            for baseline_name, split_metrics in metrics.items()
        },
        "split_team_side_counts": split_counts,
        "train_priors": {
            key: value
            for key, value in priors.items()
            if key not in {"player_density", "team_density"}
        },
        "strict_history_rule": "Match_Date < target Match_Date; all matches on the same date are predicted before history update",
        "fallback_rule": "Cold-start player/team predictions use the training-split average density and mass",
        "substitute_probability_rule": "Substitute appearance probability is the training-split substitute appearance prior",
        "elapsed_seconds": time.time() - started,
        "smoke_run": smoke,
    }


def metric_value(metrics: dict[str, Any], baseline: str, split: str, key: str) -> float:
    return float(metrics["baselines"][baseline][split][key])


def write_summary(path: Path, result: dict[str, Any]) -> None:
    def fmt(split_metrics: dict[str, Any], key: str, decimals: int = 6) -> str:
        value = split_metrics.get(key, float("nan"))
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return f"{float(value):.{decimals}f}"

    lines = [
        "# Player History Baselines",
        "",
        "Two non-parametric baselines are evaluated with the same split and composite formula used by `train_player_prediction_v1.py`.",
        "",
        "- `last_player_heatmap`: predict each player's next heatmap with that player's previous positive-appearance heatmap.",
        "- `avg10_player_heatmap`: predict each player's next heatmap with the mean of up to the previous 10 positive-appearance heatmaps.",
        "- Cold starts use the train-split average player/team density and mass.",
        "- Substitute Brier/BCE use the train-split substitute appearance prior.",
        "",
        "Lower is better for composite, JS, Wasserstein, centroid error, mass errors, Brier, and BCE. Higher is better for cosine similarity.",
        "",
        "## Main Metrics",
        "",
        "| Baseline | Split | Composite | Player JS | Player W1 | Player cosine | Centroid cells | Log-mass MAE | Sub Brier | Aggregate JS | Team JS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline in BASELINES:
        for split in ("train", "dev", "val", "test"):
            split_metrics = result["baselines"][baseline][split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        baseline,
                        split,
                        fmt(split_metrics, "composite_score"),
                        fmt(split_metrics, "player_jensen_shannon_divergence"),
                        fmt(split_metrics, "player_marginal_wasserstein_1"),
                        fmt(split_metrics, "player_cosine_similarity"),
                        fmt(split_metrics, "player_centroid_error_l2_cells", 4),
                        fmt(split_metrics, "player_log_mass_mae"),
                        fmt(split_metrics, "substitute_brier"),
                        fmt(split_metrics, "aggregate_jensen_shannon_divergence"),
                        fmt(split_metrics, "team_jensen_shannon_divergence"),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| Baseline | Split | Player appearances | Cold-start appearances | Cold-start roster slots | Team sides | Cold-start team sides |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in BASELINES:
        for split in ("train", "dev", "val", "test"):
            split_metrics = result["baselines"][baseline][split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        baseline,
                        split,
                        str(int(split_metrics["num_player_appearances"])),
                        str(int(split_metrics["cold_start_player_appearances"])),
                        str(int(split_metrics["cold_start_roster_slots"])),
                        str(int(split_metrics["num_team_sides"])),
                        str(int(split_metrics["cold_start_team_sides"])),
                    ]
                )
                + " |"
            )
    prior = result["train_priors"]["substitute_probability"]
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Train substitute appearance prior: `{prior:.6f}`.",
            f"- Elapsed seconds: `{float(result['elapsed_seconds']):.1f}`.",
            f"- Full metrics: `{DEFAULT_OUTPUT}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.data} ...", flush=True)
    frame = pd.read_pickle(args.data)
    result = evaluate_baselines(frame, batch_size=args.batch_size, smoke=args.smoke)
    write_json(args.output, result)
    write_summary(args.summary, result)
    print(json.dumps(json_safe(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
