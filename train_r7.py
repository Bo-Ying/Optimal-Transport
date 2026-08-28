import argparse
import csv
import json
import math
import pickletools
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


NROWS = 19683
LEAGUES = {
    "England_Premier_League",
    "France_Ligue1",
    "Germany_Bundesliga",
    "Italy_Serie_A",
    "Spain_Laliga",
}
WINDOWS = (60, 10, 3)
STABILITY_WINDOWS = (60, 10)
TEAM_CONTEXT_WINDOW = 10
DEFAULT_RATING_PRIOR = 6.8
DEFAULT_TEAM_GOALS_PRIOR = 1.35
DEFAULT_TEAM_POINTS_PRIOR = 1.0
SHRINKAGE_K_INIT = (20.0, 10.0, 3.0)
TEAM_FEATURE_NAMES = (
    "team_rating_ewma_60",
    "team_rating_ewma_10",
    "team_rating_ewma_3",
    "team_goal_diff_ewma",
    "team_goals_for_ewma",
    "team_goals_against_ewma",
    "team_points_ewma",
)
TEAM_CONTEXT_FEATURE_NAMES = TEAM_FEATURE_NAMES[3:]
POSITION_VALUES = {
    "GK",
    "DC",
    "MC",
    "FW",
    "DR",
    "DL",
    "DMC",
    "AMC",
    "MR",
    "ML",
    "AMR",
    "AML",
    "DMR",
    "DML",
    "FWR",
    "FWL",
    "Sub",
}


def season_start_year(date_text):
    year = int(date_text[:4])
    month = int(date_text[5:7])
    return year if month >= 7 else year - 1


def split_id_from_date(date_text):
    start_year = season_start_year(date_text)
    if 2015 <= start_year <= 2022:
        return 0
    if start_year == 2023:
        return 1
    if start_year == 2024:
        return 2
    if start_year == 2025:
        return 3
    return -1


def find_first_league_offset(pkl_path):
    with pkl_path.open("rb") as handle:
        for op, arg, pos in pickletools.genops(handle):
            if isinstance(arg, str) and arg in LEAGUES:
                return pos
    raise RuntimeError("Could not find the League column stream in the pickle.")


def extract_match_metadata_and_lineups(pkl_path):
    start_pos = find_first_league_offset(pkl_path)
    strings = []
    lineups = []
    list_values = None
    collecting_lists = False
    after_lineups_pos = None

    with pkl_path.open("rb") as handle:
        for op, arg, pos in pickletools.genops(handle):
            if pos < start_pos:
                continue

            if len(strings) < NROWS * 4:
                if isinstance(arg, str):
                    strings.append(arg)
                continue

            collecting_lists = True
            if op.name == "EMPTY_LIST":
                list_values = []
            elif collecting_lists and op.name in ("BININT", "BININT1", "BININT2"):
                if list_values is not None:
                    list_values.append(int(arg))
            elif collecting_lists and op.name == "APPENDS":
                lineups.append(list_values or [])
                list_values = None
                if len(lineups) == NROWS * 4:
                    after_lineups_pos = pos
                    break

    if len(strings) != NROWS * 4 or len(lineups) != NROWS * 4:
        raise RuntimeError(
            f"Metadata extraction failed: strings={len(strings)}, lineups={len(lineups)}"
        )

    leagues = strings[:NROWS]
    dates = strings[NROWS : NROWS * 2]
    lineup_cols = {
        "starting_home": lineups[:NROWS],
        "sub_home": lineups[NROWS : NROWS * 2],
        "starting_away": lineups[NROWS * 2 : NROWS * 3],
        "sub_away": lineups[NROWS * 3 : NROWS * 4],
    }
    return leagues, dates, lineup_cols, after_lineups_pos


def extract_positions_and_ratings(pkl_path, after_lineups_pos):
    positions = []
    ratings = []
    current = None
    pending_key = None
    mode = "search_position"

    with pkl_path.open("rb") as handle:
        for op, arg, pos in pickletools.genops(handle):
            if pos <= after_lineups_pos:
                continue

            name = op.name
            if name == "EMPTY_DICT":
                current = {}
                pending_key = None
            elif current is not None and name in ("BININT", "BININT1", "BININT2"):
                pending_key = int(arg)
            elif current is not None and pending_key is not None:
                if name in ("SHORT_BINUNICODE", "BINUNICODE"):
                    current[pending_key] = str(arg)
                    pending_key = None
                elif name == "BINFLOAT":
                    current[pending_key] = float(arg)
                    pending_key = None
                elif name == "NONE":
                    current[pending_key] = None
                    pending_key = None
            elif name == "SETITEMS" and current is not None:
                if mode == "search_position":
                    values = list(current.values())
                    if (
                        len(values) >= 20
                        and any(value == "Sub" for value in values)
                        and all(value in POSITION_VALUES for value in values)
                    ):
                        positions.append(current)
                        mode = "collect_position"
                elif mode == "collect_position":
                    positions.append(current)
                    if len(positions) == NROWS:
                        mode = "collect_rating"
                elif mode == "collect_rating":
                    ratings.append(current)
                    if len(ratings) == NROWS:
                        break
                current = None
                pending_key = None

    if len(positions) != NROWS or len(ratings) != NROWS:
        raise RuntimeError(
            f"Position/rating extraction failed: positions={len(positions)}, ratings={len(ratings)}"
        )
    return positions, ratings


def code_maps(values):
    ordered = sorted(set(values))
    return {value: idx for idx, value in enumerate(ordered)}


def inverse_softplus(value):
    return math.log(math.exp(value) - 1.0)


def infer_context_from_position_order(position_dict, position_map):
    context = {}
    phase = 0
    for player_id, position in position_dict.items():
        if position not in position_map:
            continue

        if phase == 0 and position == "Sub":
            phase = 1
        elif phase == 1 and position != "Sub":
            phase = 2
        elif phase == 2 and position == "Sub":
            phase = 3

        is_home = 1 if phase in (0, 1) else 0
        is_sub = 1 if position == "Sub" else 0
        context[int(player_id)] = (is_home, is_sub, position_map[position])
    return context


def ewma_from_history(history, value_name, window, default):
    values = [
        float(item[value_name])
        for item in history[-window:]
        if item.get(value_name) is not None and np.isfinite(item[value_name])
    ]
    if not values:
        return float(default)
    alpha = 2.0 / (window + 1.0)
    ages = np.arange(len(values) - 1, -1, -1, dtype=np.float32)
    weights = np.power(1.0 - alpha, ages)
    return float(np.dot(np.asarray(values, dtype=np.float32), weights) / weights.sum())


def summarize_team_history(history):
    return np.asarray(
        [
            ewma_from_history(history, "rating", 60, DEFAULT_RATING_PRIOR),
            ewma_from_history(history, "rating", 10, DEFAULT_RATING_PRIOR),
            ewma_from_history(history, "rating", 3, DEFAULT_RATING_PRIOR),
            ewma_from_history(history, "goal_diff", TEAM_CONTEXT_WINDOW, 0.0),
            ewma_from_history(
                history, "goals_for", TEAM_CONTEXT_WINDOW, DEFAULT_TEAM_GOALS_PRIOR
            ),
            ewma_from_history(
                history,
                "goals_against",
                TEAM_CONTEXT_WINDOW,
                DEFAULT_TEAM_GOALS_PRIOR,
            ),
            ewma_from_history(
                history, "points", TEAM_CONTEXT_WINDOW, DEFAULT_TEAM_POINTS_PRIOR
            ),
        ],
        dtype=np.float32,
    )


def team_key(team_id, team_name):
    if team_id is not None and not (isinstance(team_id, float) and np.isnan(team_id)):
        return f"id:{int(team_id)}"
    return f"name:{team_name}"


def side_average_rating(rating_dict, player_ids):
    values = [
        float(rating_dict[player_id])
        for player_id in player_ids
        if player_id in rating_dict and rating_dict[player_id] is not None
    ]
    return float(np.mean(values)) if values else DEFAULT_RATING_PRIOR


def build_match_team_features(pkl_path, dates, lineup_cols, ratings):
    import pandas as pd

    print("Building chronological team/opponent features...", flush=True)
    df = pd.read_pickle(pkl_path)
    required = [
        "Home_Team",
        "Away_Team",
        "Home_Team_ID",
        "Away_Team_ID",
        "Home_Score",
        "Away_Score",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing team feature source columns: {missing}")

    home_features = np.zeros((NROWS, len(TEAM_FEATURE_NAMES)), dtype=np.float32)
    away_features = np.zeros((NROWS, len(TEAM_FEATURE_NAMES)), dtype=np.float32)
    histories = defaultdict(list)

    match_order = sorted(range(NROWS), key=lambda idx: (dates[idx], idx))
    for match_idx in match_order:
        row = df.iloc[match_idx]
        home_key = team_key(row["Home_Team_ID"], row["Home_Team"])
        away_key = team_key(row["Away_Team_ID"], row["Away_Team"])

        home_features[match_idx] = summarize_team_history(histories[home_key])
        away_features[match_idx] = summarize_team_history(histories[away_key])

        home_score = int(row["Home_Score"])
        away_score = int(row["Away_Score"])
        home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
        home_players = set(lineup_cols["starting_home"][match_idx]) | set(
            lineup_cols["sub_home"][match_idx]
        )
        away_players = set(lineup_cols["starting_away"][match_idx]) | set(
            lineup_cols["sub_away"][match_idx]
        )
        home_rating = side_average_rating(ratings[match_idx], home_players)
        away_rating = side_average_rating(ratings[match_idx], away_players)

        histories[home_key].append(
            {
                "rating": home_rating,
                "goal_diff": home_score - away_score,
                "goals_for": home_score,
                "goals_against": away_score,
                "points": home_points,
            }
        )
        histories[away_key].append(
            {
                "rating": away_rating,
                "goal_diff": away_score - home_score,
                "goals_for": away_score,
                "goals_against": home_score,
                "points": away_points,
            }
        )

    return home_features, away_features


def build_feature_cache(pkl_path, cache_path):
    print("Extracting metadata and lineups...", flush=True)
    leagues, dates, lineup_cols, after_lineups_pos = extract_match_metadata_and_lineups(
        pkl_path
    )
    print("Extracting Player_Position and Player_Rating...", flush=True)
    positions, ratings = extract_positions_and_ratings(pkl_path, after_lineups_pos)
    home_team_features, away_team_features = build_match_team_features(
        pkl_path, dates, lineup_cols, ratings
    )

    league_map = code_maps(leagues)
    position_map = code_maps(POSITION_VALUES)
    rows = []

    print("Building chronological player-match rows...", flush=True)
    for match_idx in range(NROWS):
        context = infer_context_from_position_order(positions[match_idx], position_map)

        for player_id, rating in ratings[match_idx].items():
            if rating is None or player_id not in context:
                continue
            is_home, is_sub, position = context[player_id]
            split_id = split_id_from_date(dates[match_idx])
            if split_id < 0:
                continue
            rows.append(
                (
                    dates[match_idx],
                    match_idx,
                    int(player_id),
                    float(rating),
                    split_id,
                    league_map[leagues[match_idx]],
                    is_home,
                    is_sub,
                    position,
                )
            )

    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    max_w = max(WINDOWS)
    hist_rating = []
    hist_age = []
    hist_league_windows = []
    hist_home_windows = []
    hist_sub_windows = []
    hist_position_windows = []
    same_position = []
    same_sub = []
    same_league = []
    same_home = []
    hist_mask = []
    current_league = []
    current_is_home = []
    current_is_sub = []
    current_position = []
    current_team_features = []
    opponent_team_features = []
    targets = []
    splits = []
    histories = defaultdict(list)
    skipped_no_history = 0

    print("Constructing history windows...", flush=True)
    for date, match_idx, player_id, rating, split, league, is_home, is_sub, position in rows:
        history = histories[player_id]
        if history:
            ratings_row = np.zeros(max_w, dtype=np.float32)
            ages_row = np.zeros(max_w, dtype=np.float32)
            hist_league_row = np.zeros(max_w, dtype=np.int64)
            hist_home_row = np.zeros(max_w, dtype=np.int64)
            hist_sub_row = np.zeros(max_w, dtype=np.int64)
            hist_position_row = np.zeros(max_w, dtype=np.int64)
            mask_row = np.zeros(max_w, dtype=np.bool_)
            same_pos_row = np.zeros(max_w, dtype=np.bool_)
            same_sub_row = np.zeros(max_w, dtype=np.bool_)
            same_league_row = np.zeros(max_w, dtype=np.bool_)
            same_home_row = np.zeros(max_w, dtype=np.bool_)

            recent_history = history[-max_w:]
            for lag, hist in enumerate(reversed(recent_history), start=1):
                idx = lag - 1
                (
                    hist_rating_value,
                    hist_league,
                    hist_is_home,
                    hist_is_sub,
                    hist_position,
                ) = hist
                ratings_row[idx] = hist_rating_value
                ages_row[idx] = lag
                hist_league_row[idx] = hist_league
                hist_home_row[idx] = hist_is_home
                hist_sub_row[idx] = hist_is_sub
                hist_position_row[idx] = hist_position
                mask_row[idx] = True
                same_pos_row[idx] = (
                    is_sub == 0 and hist_is_sub == 0 and position == hist_position
                )
                same_sub_row[idx] = is_sub == hist_is_sub
                same_league_row[idx] = league == hist_league
                same_home_row[idx] = is_home == hist_is_home

            hist_rating.append(ratings_row)
            hist_age.append(ages_row)
            hist_league_windows.append(hist_league_row)
            hist_home_windows.append(hist_home_row)
            hist_sub_windows.append(hist_sub_row)
            hist_position_windows.append(hist_position_row)
            hist_mask.append(mask_row)
            same_position.append(same_pos_row)
            same_sub.append(same_sub_row)
            same_league.append(same_league_row)
            same_home.append(same_home_row)
            current_league.append(league)
            current_is_home.append(is_home)
            current_is_sub.append(is_sub)
            current_position.append(position)
            if is_home:
                current_team_features.append(home_team_features[match_idx])
                opponent_team_features.append(away_team_features[match_idx])
            else:
                current_team_features.append(away_team_features[match_idx])
                opponent_team_features.append(home_team_features[match_idx])
            targets.append(rating)
            splits.append(split)
        else:
            skipped_no_history += 1

        history.append((rating, league, is_home, is_sub, position))
        if len(history) > max_w:
            del history[:-max_w]

    arrays = {
        "hist_rating": np.stack(hist_rating).astype(np.float32),
        "hist_age": np.stack(hist_age).astype(np.float32),
        "hist_league": np.stack(hist_league_windows).astype(np.int64),
        "hist_is_home": np.stack(hist_home_windows).astype(np.int64),
        "hist_is_sub": np.stack(hist_sub_windows).astype(np.int64),
        "hist_position": np.stack(hist_position_windows).astype(np.int64),
        "hist_mask": np.stack(hist_mask),
        "same_position": np.stack(same_position),
        "same_sub": np.stack(same_sub),
        "same_league": np.stack(same_league),
        "same_home": np.stack(same_home),
        "current_league": np.asarray(current_league, dtype=np.int64),
        "current_is_home": np.asarray(current_is_home, dtype=np.int64),
        "current_is_sub": np.asarray(current_is_sub, dtype=np.int64),
        "current_position": np.asarray(current_position, dtype=np.int64),
        "current_team_features": np.stack(current_team_features).astype(np.float32),
        "opponent_team_features": np.stack(opponent_team_features).astype(np.float32),
        "target": np.asarray(targets, dtype=np.float32),
        "split": np.asarray(splits, dtype=np.int8),
    }
    metadata = {
        "source": str(pkl_path),
        "n_match_rows": NROWS,
        "n_rated_rows": len(rows),
        "n_samples_with_history": int(len(targets)),
        "skipped_no_history": int(skipped_no_history),
        "league_map": league_map,
        "position_map": position_map,
        "windows": list(WINDOWS),
        "team_feature_names": list(TEAM_FEATURE_NAMES),
        "team_context_window": TEAM_CONTEXT_WINDOW,
        "team_feature_priors": {
            "rating": DEFAULT_RATING_PRIOR,
            "goal_diff": 0.0,
            "goals_for": DEFAULT_TEAM_GOALS_PRIOR,
            "goals_against": DEFAULT_TEAM_GOALS_PRIOR,
            "points": DEFAULT_TEAM_POINTS_PRIOR,
        },
        "split_ids": {"train": 0, "dev": 1, "val": 2, "test": 3},
    }

    np.savez_compressed(cache_path, **arrays)
    cache_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return arrays, metadata


class R7Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # softplus(-2.25) ~= 0.10; softplus(-4) gives small positive log-bonuses.
        self.raw_decay = torch.nn.Parameter(torch.full((3,), -2.25))
        self.raw_bonus = torch.nn.Parameter(torch.full((3, 4), -4.0))
        self.raw_mix = torch.nn.Parameter(torch.zeros(3))
        self.raw_shrinkage_k = torch.nn.Parameter(
            torch.tensor([inverse_softplus(value) for value in SHRINKAGE_K_INIT])
        )
        self.raw_count_gate = torch.nn.Parameter(torch.zeros(3))
        self.trend_weight = torch.nn.Parameter(torch.zeros(3))
        self.momentum_weight = torch.nn.Parameter(torch.zeros(2))
        self.volatility_weight = torch.nn.Parameter(torch.zeros(2))
        self.robust_volatility_weight = torch.nn.Parameter(torch.zeros(2))
        self.range_weight = torch.nn.Parameter(torch.zeros(3))
        self.autocorr_weight = torch.nn.Parameter(torch.zeros(2))
        self.mean_reversion_weight = torch.nn.Parameter(torch.zeros(2))
        self.streak_weight = torch.nn.Parameter(torch.zeros(2))
        self.changepoint_weight = torch.nn.Parameter(torch.zeros(2))
        self.raw_changepoint_threshold = torch.nn.Parameter(torch.tensor(0.0))
        self.bias = torch.nn.Parameter(torch.tensor(DEFAULT_RATING_PRIOR))
        self.raw_position_effect = torch.nn.Parameter(torch.zeros(len(POSITION_VALUES)))
        self.raw_league_effect = torch.nn.Parameter(torch.zeros(len(LEAGUES)))
        self.raw_home_away_effect = torch.nn.Parameter(torch.zeros(2))
        self.raw_starter_sub_effect = torch.nn.Parameter(torch.zeros(2))
        self.own_team_strength_weight = torch.nn.Parameter(torch.zeros(3))
        self.opponent_strength_weight = torch.nn.Parameter(torch.zeros(3))
        self.strength_delta_weight = torch.nn.Parameter(torch.zeros(3))
        self.own_team_context_weight = torch.nn.Parameter(
            torch.zeros(len(TEAM_CONTEXT_FEATURE_NAMES))
        )
        self.opponent_team_context_weight = torch.nn.Parameter(
            torch.zeros(len(TEAM_CONTEXT_FEATURE_NAMES))
        )
        self.form_delta_weight = torch.nn.Parameter(torch.zeros(2))
        self.attack_defense_delta_weight = torch.nn.Parameter(torch.zeros(1))
        self.home_adjusted_strength_delta_weight = torch.nn.Parameter(torch.zeros(1))
        self.raw_position_strength_delta_effect = torch.nn.Parameter(
            torch.zeros(len(POSITION_VALUES))
        )

    def transformed_params(self):
        position_effect = self.raw_position_effect - self.raw_position_effect.mean()
        league_effect = self.raw_league_effect - self.raw_league_effect.mean()
        home_away_effect = self.raw_home_away_effect - self.raw_home_away_effect.mean()
        starter_sub_effect = (
            self.raw_starter_sub_effect - self.raw_starter_sub_effect.mean()
        )
        position_strength_delta_effect = (
            self.raw_position_strength_delta_effect
            - self.raw_position_strength_delta_effect.mean()
        )
        return {
            "decay": torch.nn.functional.softplus(self.raw_decay) + 1e-4,
            "bonus": torch.nn.functional.softplus(self.raw_bonus),
            "mix": torch.softmax(self.raw_mix, dim=0),
            "mix_logits": self.raw_mix,
            "shrinkage_k": torch.nn.functional.softplus(self.raw_shrinkage_k) + 1e-4,
            "count_gate": torch.nn.functional.softplus(self.raw_count_gate),
            "trend_weight": self.trend_weight,
            "momentum_weight": self.momentum_weight,
            "volatility_weight": self.volatility_weight,
            "robust_volatility_weight": self.robust_volatility_weight,
            "range_weight": self.range_weight,
            "autocorr_weight": self.autocorr_weight,
            "mean_reversion_weight": self.mean_reversion_weight,
            "streak_weight": self.streak_weight,
            "changepoint_weight": self.changepoint_weight,
            "changepoint_threshold": (
                torch.nn.functional.softplus(self.raw_changepoint_threshold) + 1e-4
            ),
            "bias": self.bias,
            "position_effect": position_effect,
            "league_effect": league_effect,
            "home_away_effect": home_away_effect,
            "starter_sub_effect": starter_sub_effect,
            "own_team_strength_weight": self.own_team_strength_weight,
            "opponent_strength_weight": self.opponent_strength_weight,
            "strength_delta_weight": self.strength_delta_weight,
            "own_team_context_weight": self.own_team_context_weight,
            "opponent_team_context_weight": self.opponent_team_context_weight,
            "form_delta_weight": self.form_delta_weight,
            "attack_defense_delta_weight": self.attack_defense_delta_weight,
            "home_adjusted_strength_delta_weight": (
                self.home_adjusted_strength_delta_weight
            ),
            "position_strength_delta_effect": position_strength_delta_effect,
        }

    def context_effect(self, params, league, is_home, is_sub, position):
        return (
            params["position_effect"][position.long()]
            + params["league_effect"][league.long()]
            + params["home_away_effect"][is_home.long()]
            + params["starter_sub_effect"][is_sub.long()]
        )

    def team_effect(
        self, params, current_team_features, opponent_team_features, is_home, position
    ):
        own_strength = current_team_features[:, :3] - DEFAULT_RATING_PRIOR
        opponent_strength = opponent_team_features[:, :3] - DEFAULT_RATING_PRIOR
        strength_delta = own_strength - opponent_strength
        own_context = current_team_features[:, 3:].clone()
        opponent_context = opponent_team_features[:, 3:].clone()
        own_context[:, 1] = own_context[:, 1] - DEFAULT_TEAM_GOALS_PRIOR
        own_context[:, 2] = own_context[:, 2] - DEFAULT_TEAM_GOALS_PRIOR
        own_context[:, 3] = own_context[:, 3] - DEFAULT_TEAM_POINTS_PRIOR
        opponent_context[:, 1] = opponent_context[:, 1] - DEFAULT_TEAM_GOALS_PRIOR
        opponent_context[:, 2] = opponent_context[:, 2] - DEFAULT_TEAM_GOALS_PRIOR
        opponent_context[:, 3] = opponent_context[:, 3] - DEFAULT_TEAM_POINTS_PRIOR
        form_delta = torch.stack(
            [
                current_team_features[:, 3] - opponent_team_features[:, 3],
                current_team_features[:, 6] - opponent_team_features[:, 6],
            ],
            dim=1,
        )
        attack_defense_delta = (
            current_team_features[:, 4] - opponent_team_features[:, 5]
        )
        home_sign = is_home.float() * 2.0 - 1.0
        home_adjusted_strength_delta = strength_delta[:, 1] * home_sign
        position_strength_delta = (
            params["position_strength_delta_effect"][position.long()]
            * strength_delta[:, 1]
        )
        return (
            (own_strength * params["own_team_strength_weight"].view(1, 3)).sum(dim=1)
            + (
                opponent_strength
                * params["opponent_strength_weight"].view(1, 3)
            ).sum(dim=1)
            + (
                strength_delta * params["strength_delta_weight"].view(1, 3)
            ).sum(dim=1)
            + (
                own_context
                * params["own_team_context_weight"].view(
                    1, len(TEAM_CONTEXT_FEATURE_NAMES)
                )
            ).sum(dim=1)
            + (
                opponent_context
                * params["opponent_team_context_weight"].view(
                    1, len(TEAM_CONTEXT_FEATURE_NAMES)
                )
            ).sum(dim=1)
            + (form_delta * params["form_delta_weight"].view(1, 2)).sum(dim=1)
            + attack_defense_delta * params["attack_defense_delta_weight"][0]
            + home_adjusted_strength_delta
            * params["home_adjusted_strength_delta_weight"][0]
            + position_strength_delta
        )

    @staticmethod
    def weighted_median(values, weights):
        sorted_values, order = torch.sort(values, dim=1)
        sorted_weights = torch.gather(weights, 1, order)
        cumulative = sorted_weights.cumsum(dim=1)
        cutoff = 0.5 * sorted_weights.sum(dim=1, keepdim=True)
        median_idx = (cumulative >= cutoff).float().argmax(dim=1)
        return sorted_values.gather(1, median_idx.unsqueeze(1)).squeeze(1)

    def forward(self, batch):
        (
            hist_rating,
            hist_age,
            hist_mask,
            same_position,
            same_sub,
            same_league,
            same_home,
            hist_league,
            hist_is_home,
            hist_is_sub,
            hist_position,
            current_league,
            current_is_home,
            current_is_sub,
            current_position,
            current_team_features,
            opponent_team_features,
        ) = batch
        params = self.transformed_params()
        features = []
        trends = []
        volatilities = []
        robust_volatilities = []
        ranges = []
        autocorrelations = []
        window_counts = []
        flags = torch.stack(
            [
                same_position.float(),
                same_sub.float(),
                same_league.float(),
                same_home.float(),
            ],
            dim=-1,
        )
        adjusted_hist_rating = hist_rating - self.context_effect(
            params, hist_league, hist_is_home, hist_is_sub, hist_position
        ) - params["bias"]

        for window_idx, window in enumerate(WINDOWS):
            window_slice = slice(0, window)
            mask = hist_mask[:, window_slice].float()
            ages = hist_age[:, window_slice]
            log_weights = -params["decay"][window_idx] * (ages - 1.0)
            flag_terms = (
                flags[:, window_slice, :] * params["bonus"][window_idx].view(1, 1, 4)
            ).sum(dim=-1)
            weights = torch.exp(log_weights + flag_terms) * mask
            denom = weights.sum(dim=1).clamp_min(1e-8)
            count = mask.sum(dim=1)
            window_counts.append(count)
            window_rating = adjusted_hist_rating[:, window_slice]
            ewma = (weights * window_rating).sum(dim=1) / denom
            mean_age = (weights * ages).sum(dim=1) / denom
            centered_age = ages - mean_age.unsqueeze(1)
            centered_rating = window_rating - ewma.unsqueeze(1)
            slope_denom = (weights * centered_age.square()).sum(dim=1).clamp_min(1e-8)
            slope_by_age = (weights * centered_age * centered_rating).sum(dim=1) / slope_denom
            variance = (weights * centered_rating.square()).sum(dim=1) / denom
            large = torch.full_like(window_rating, 1_000_000.0)
            range_min = torch.where(mask.bool(), window_rating, large).min(dim=1).values
            range_max = torch.where(mask.bool(), window_rating, -large).max(dim=1).values
            features.append(ewma)
            trends.append(-slope_by_age)
            ranges.append(range_max - range_min)
            if window in STABILITY_WINDOWS:
                pair_mask = (hist_mask[:, window_slice][:, :-1] & hist_mask[:, window_slice][:, 1:]).float()
                pair_weights = weights[:, :-1] * pair_mask
                pair_denom = pair_weights.sum(dim=1).clamp_min(1e-8)
                recent_rating = window_rating[:, :-1]
                previous_rating = window_rating[:, 1:]
                recent_mean = (pair_weights * recent_rating).sum(dim=1) / pair_denom
                previous_mean = (pair_weights * previous_rating).sum(dim=1) / pair_denom
                recent_centered = recent_rating - recent_mean.unsqueeze(1)
                previous_centered = previous_rating - previous_mean.unsqueeze(1)
                covariance = (
                    pair_weights * recent_centered * previous_centered
                ).sum(dim=1) / pair_denom
                recent_variance = (
                    pair_weights * recent_centered.square()
                ).sum(dim=1) / pair_denom
                previous_variance = (
                    pair_weights * previous_centered.square()
                ).sum(dim=1) / pair_denom
                autocorr = covariance / torch.sqrt(
                    recent_variance * previous_variance + 1e-8
                )
                pair_count = pair_mask.sum(dim=1)
                autocorrelations.append(
                    torch.where(pair_count >= 2.0, autocorr, torch.zeros_like(autocorr))
                )
                median = self.weighted_median(window_rating, weights)
                abs_deviation = (window_rating - median.unsqueeze(1)).abs()
                mad = self.weighted_median(abs_deviation, weights)
                volatilities.append(torch.sqrt(variance.clamp_min(0.0) + 1e-8))
                robust_volatilities.append(1.4826 * mad)

        raw_feature_matrix = torch.stack(features, dim=1)
        count_matrix = torch.stack(window_counts, dim=1)
        reliability_matrix = count_matrix / (
            count_matrix + params["shrinkage_k"].view(1, 3)
        )
        feature_matrix = raw_feature_matrix * reliability_matrix
        trend_matrix = torch.stack(trends, dim=1)
        volatility_matrix = torch.stack(volatilities, dim=1)
        robust_volatility_matrix = torch.stack(robust_volatilities, dim=1)
        range_matrix = torch.stack(ranges, dim=1)
        autocorr_matrix = torch.stack(autocorrelations, dim=1)
        trend_matrix = trend_matrix * reliability_matrix
        range_matrix = range_matrix * reliability_matrix
        stability_reliability = reliability_matrix[:, :2]
        volatility_matrix = volatility_matrix * stability_reliability
        robust_volatility_matrix = robust_volatility_matrix * stability_reliability
        autocorr_matrix = autocorr_matrix * stability_reliability
        last_adjusted = adjusted_hist_rating[:, 0]
        momentum_matrix = torch.stack(
            [
                feature_matrix[:, 1] - feature_matrix[:, 0],
                feature_matrix[:, 2] - feature_matrix[:, 1],
            ],
            dim=1,
        )
        autocorr_signal_matrix = autocorr_matrix * (
            last_adjusted - feature_matrix[:, 0]
        ).unsqueeze(1)
        mean_reversion_matrix = torch.stack(
            [
                feature_matrix[:, 0] - feature_matrix[:, 2],
                feature_matrix[:, 0] - last_adjusted,
            ],
            dim=1,
        )
        recent_deviation = last_adjusted - feature_matrix[:, 0]
        streak_deviation = adjusted_hist_rating[:, :10] - feature_matrix[:, 0].unsqueeze(1)
        streak_mask = (
            hist_mask[:, :10] & (streak_deviation * recent_deviation.unsqueeze(1) > 0.0)
        ).float()
        streak_prefix = streak_mask.cumprod(dim=1)
        streak_length = streak_prefix.sum(dim=1)
        streak_matrix = torch.stack(
            [
                torch.sign(recent_deviation) * torch.log1p(streak_length),
                (streak_deviation * streak_prefix).sum(dim=1)
                / streak_length.clamp_min(1.0),
            ],
            dim=1,
        )
        regime_gap = feature_matrix[:, 2] - feature_matrix[:, 0]
        long_robust_volatility = robust_volatility_matrix[:, 0].clamp_min(0.05)
        regime_z = regime_gap / long_robust_volatility
        changepoint_strength = torch.relu(
            regime_z.abs() - params["changepoint_threshold"]
        )
        changepoint_matrix = torch.stack(
            [
                regime_gap * changepoint_strength,
                torch.sign(regime_gap) * changepoint_strength,
            ],
            dim=1,
        )
        coverage = (count_matrix + 1.0) / (
            torch.tensor(WINDOWS, device=count_matrix.device, dtype=count_matrix.dtype).view(1, 3)
            + 1.0
        )
        sample_mix_logits = (
            params["mix_logits"].view(1, 3)
            + params["count_gate"].view(1, 3) * torch.log(coverage.clamp_min(1e-4))
        )
        sample_mix = torch.softmax(sample_mix_logits, dim=1)
        base_pred = (
            self.bias
            + (feature_matrix * sample_mix).sum(dim=1)
            + (trend_matrix * params["trend_weight"].view(1, 3)).sum(dim=1)
            + (momentum_matrix * params["momentum_weight"].view(1, 2)).sum(dim=1)
            + (volatility_matrix * params["volatility_weight"].view(1, 2)).sum(dim=1)
            + (
                robust_volatility_matrix
                * params["robust_volatility_weight"].view(1, 2)
            ).sum(dim=1)
            + (range_matrix * params["range_weight"].view(1, 3)).sum(dim=1)
            + (
                autocorr_signal_matrix * params["autocorr_weight"].view(1, 2)
            ).sum(dim=1)
            + (
                mean_reversion_matrix
                * params["mean_reversion_weight"].view(1, 2)
            ).sum(dim=1)
            + (streak_matrix * params["streak_weight"].view(1, 2)).sum(dim=1)
            + (
                changepoint_matrix * params["changepoint_weight"].view(1, 2)
            ).sum(dim=1)
        )
        return (
            base_pred
            + self.context_effect(
                params,
                current_league,
                current_is_home,
                current_is_sub,
                current_position,
            )
            + self.team_effect(
                params,
                current_team_features,
                opponent_team_features,
                current_is_home,
                current_position,
            )
        )


def parameter_count(model):
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def make_tensors(arrays):
    keys = [
        "hist_rating",
        "hist_age",
        "hist_mask",
        "same_position",
        "same_sub",
        "same_league",
        "same_home",
        "hist_league",
        "hist_is_home",
        "hist_is_sub",
        "hist_position",
        "current_league",
        "current_is_home",
        "current_is_sub",
        "current_position",
        "current_team_features",
        "opponent_team_features",
    ]
    tensors = {key: torch.from_numpy(arrays[key]) for key in keys}
    tensors["target"] = torch.from_numpy(arrays["target"])
    split = arrays["split"]
    indices = {
        "train": np.flatnonzero(split == 0),
        "dev": np.flatnonzero(split == 1),
        "val": np.flatnonzero(split == 2),
        "test": np.flatnonzero(split == 3),
    }
    return tensors, indices


def batch_from_indices(tensors, indices, device):
    return tuple(
        tensors[key][indices].to(device)
        for key in [
            "hist_rating",
            "hist_age",
            "hist_mask",
            "same_position",
            "same_sub",
            "same_league",
            "same_home",
            "hist_league",
            "hist_is_home",
            "hist_is_sub",
            "hist_position",
            "current_league",
            "current_is_home",
            "current_is_sub",
            "current_position",
            "current_team_features",
            "opponent_team_features",
        ]
    )


@torch.no_grad()
def predict(model, tensors, indices, device, batch_size):
    model.eval()
    preds = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = batch_from_indices(tensors, batch_indices, device)
        preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def metric_bundle(y_true, y_pred):
    errors = y_pred - y_true
    out = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }
    if len(y_true) > 1:
        rho = spearmanr(y_true, y_pred).statistic
        out["spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    else:
        out["spearman"] = float("nan")
    return out


def evaluate_splits(model, tensors, indices, device, batch_size):
    metrics = {}
    for split_name, split_indices in indices.items():
        y_true = tensors["target"][split_indices].numpy()
        y_pred = predict(model, tensors, split_indices, device, batch_size)
        metrics[split_name] = metric_bundle(y_true, y_pred)
        metrics[split_name]["n"] = int(len(split_indices))
    return metrics


def export_params(model, metadata):
    params = model.transformed_params()
    bonus_names = ["position", "substitute", "league", "home_away"]
    position_names = {
        idx: name for name, idx in metadata["position_map"].items()
    }
    league_names = {
        idx: name for name, idx in metadata["league_map"].items()
    }
    out = {
        "parameter_count": parameter_count(model),
        "bias": float(params["bias"].detach().cpu()),
        "mix": {
            str(window): float(params["mix"][idx].detach().cpu())
            for idx, window in enumerate(WINDOWS)
        },
        "shrinkage": {
            "k": {
                str(window): float(params["shrinkage_k"][idx].detach().cpu())
                for idx, window in enumerate(WINDOWS)
            },
            "count_gate": {
                str(window): float(params["count_gate"][idx].detach().cpu())
                for idx, window in enumerate(WINDOWS)
            },
        },
        "trend_weight": {
            str(window): float(params["trend_weight"][idx].detach().cpu())
            for idx, window in enumerate(WINDOWS)
        },
        "momentum_weight": {
            "10_minus_60": float(params["momentum_weight"][0].detach().cpu()),
            "3_minus_10": float(params["momentum_weight"][1].detach().cpu()),
        },
        "volatility_weight": {
            str(window): float(params["volatility_weight"][idx].detach().cpu())
            for idx, window in enumerate(STABILITY_WINDOWS)
        },
        "robust_volatility_weight": {
            str(window): float(
                params["robust_volatility_weight"][idx].detach().cpu()
            )
            for idx, window in enumerate(STABILITY_WINDOWS)
        },
        "range_weight": {
            str(window): float(params["range_weight"][idx].detach().cpu())
            for idx, window in enumerate(WINDOWS)
        },
        "autocorr_weight": {
            str(window): float(params["autocorr_weight"][idx].detach().cpu())
            for idx, window in enumerate(STABILITY_WINDOWS)
        },
        "mean_reversion_weight": {
            "60_minus_3": float(params["mean_reversion_weight"][0].detach().cpu()),
            "60_minus_last": float(params["mean_reversion_weight"][1].detach().cpu()),
        },
        "streak_weight": {
            "signed_log_length": float(params["streak_weight"][0].detach().cpu()),
            "signed_intensity": float(params["streak_weight"][1].detach().cpu()),
        },
        "changepoint_weight": {
            "signed_gap_times_strength": float(
                params["changepoint_weight"][0].detach().cpu()
            ),
            "signed_strength": float(params["changepoint_weight"][1].detach().cpu()),
        },
        "changepoint_threshold": float(
            params["changepoint_threshold"].detach().cpu()
        ),
        "windows": {},
        "context_effect": {
            "position": {
                position_names[idx]: float(params["position_effect"][idx].detach().cpu())
                for idx in sorted(position_names)
            },
            "league": {
                league_names[idx]: float(params["league_effect"][idx].detach().cpu())
                for idx in sorted(league_names)
            },
            "home_away": {
                "away": float(params["home_away_effect"][0].detach().cpu()),
                "home": float(params["home_away_effect"][1].detach().cpu()),
            },
            "starter_sub": {
                "starter": float(params["starter_sub_effect"][0].detach().cpu()),
                "sub": float(params["starter_sub_effect"][1].detach().cpu()),
            },
        },
        "team_effect": {
            "feature_names": {
                "strength": list(TEAM_FEATURE_NAMES[:3]),
                "context": list(TEAM_CONTEXT_FEATURE_NAMES),
            },
            "own_team_strength_weight": {
                str(window): float(
                    params["own_team_strength_weight"][idx].detach().cpu()
                )
                for idx, window in enumerate(WINDOWS)
            },
            "opponent_strength_weight": {
                str(window): float(
                    params["opponent_strength_weight"][idx].detach().cpu()
                )
                for idx, window in enumerate(WINDOWS)
            },
            "strength_delta_weight": {
                str(window): float(
                    params["strength_delta_weight"][idx].detach().cpu()
                )
                for idx, window in enumerate(WINDOWS)
            },
            "own_team_context_weight": {
                TEAM_CONTEXT_FEATURE_NAMES[idx]: float(
                    params["own_team_context_weight"][idx].detach().cpu()
                )
                for idx in range(len(TEAM_CONTEXT_FEATURE_NAMES))
            },
            "opponent_team_context_weight": {
                TEAM_CONTEXT_FEATURE_NAMES[idx]: float(
                    params["opponent_team_context_weight"][idx].detach().cpu()
                )
                for idx in range(len(TEAM_CONTEXT_FEATURE_NAMES))
            },
            "form_delta_weight": {
                "goal_diff_delta": float(
                    params["form_delta_weight"][0].detach().cpu()
                ),
                "points_delta": float(params["form_delta_weight"][1].detach().cpu()),
            },
            "attack_defense_delta_weight": float(
                params["attack_defense_delta_weight"][0].detach().cpu()
            ),
            "home_adjusted_strength_delta_weight": float(
                params["home_adjusted_strength_delta_weight"][0].detach().cpu()
            ),
            "position_strength_delta_effect": {
                position_names[idx]: float(
                    params["position_strength_delta_effect"][idx].detach().cpu()
                )
                for idx in sorted(position_names)
            },
        },
    }
    for idx, window in enumerate(WINDOWS):
        out["windows"][str(window)] = {
            "decay": float(params["decay"][idx].detach().cpu()),
            "bonus_log_multiplier": {
                bonus_names[j]: float(params["bonus"][idx, j].detach().cpu())
                for j in range(4)
            },
        }
    return out


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = Path(args.data)
    cache_path = out_dir / "r7_player_history_features.npz"

    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...", flush=True)
        loaded = np.load(cache_path)
        arrays = {key: loaded[key] for key in loaded.files}
        metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    else:
        arrays, metadata = build_feature_cache(pkl_path, cache_path)

    tensors, indices = make_tensors(arrays)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = R7Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()
    train_indices = indices["train"].copy()

    log_path = out_dir / f"training_log_seed{args.seed}.csv"
    best_path = out_dir / f"best_model_seed{args.seed}.pt"
    best_dev = float("inf")
    best_epoch = None
    best_state = None

    print(
        f"Training on {device} for {args.epochs} epochs; train/dev/val/test samples = "
        f"{len(indices['train'])}/{len(indices['dev'])}/{len(indices['val'])}/{len(indices['test'])}",
        flush=True,
    )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_mae", "dev_mae", "seconds"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            np.random.shuffle(train_indices)
            losses = []
            for start in range(0, len(train_indices), args.batch_size):
                batch_indices = train_indices[start : start + args.batch_size]
                batch = batch_from_indices(tensors, batch_indices, device)
                target = tensors["target"][batch_indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch)
                if not torch.isfinite(pred).all():
                    raise RuntimeError("Non-finite predictions detected during training.")
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_eval_idx = indices["train"]
            train_pred = predict(model, tensors, train_eval_idx, device, args.eval_batch_size)
            train_true = tensors["target"][train_eval_idx].numpy()
            train_mae = float(np.mean(np.abs(train_pred - train_true)))
            dev_pred = predict(model, tensors, indices["dev"], device, args.eval_batch_size)
            dev_true = tensors["target"][indices["dev"]].numpy()
            dev_mae = float(np.mean(np.abs(dev_pred - dev_true)))

            elapsed = time.time() - start_time
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_mae": train_mae,
                    "dev_mae": dev_mae,
                    "seconds": elapsed,
                }
            )
            handle.flush()
            print(
                f"epoch {epoch:03d} train_loss={np.mean(losses):.5f} "
                f"train_mae={train_mae:.5f} dev_mae={dev_mae:.5f} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )

            if dev_mae < best_dev:
                best_dev = dev_mae
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state": best_state,
                        "best_epoch": best_epoch,
                        "best_dev_mae": best_dev,
                        "seed": args.seed,
                        "parameter_count": parameter_count(model),
                    },
                    best_path,
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    metrics = evaluate_splits(model, tensors, indices, device, args.eval_batch_size)
    params = export_params(model, metadata)

    result = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "selected_epoch_by_dev_mae": best_epoch,
        "best_dev_mae": best_dev,
        "metrics": metrics,
        "parameters": params,
        "metadata": metadata,
        "artifacts": {
            "feature_cache": str(cache_path),
            "training_log": str(log_path),
            "best_model": str(best_path),
        },
    }
    metrics_path = out_dir / f"metrics_seed{args.seed}.json"
    params_path = out_dir / f"best_params_seed{args.seed}.json"
    report_path = out_dir / f"report_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("FINAL_RESULT_JSON_START", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("FINAL_RESULT_JSON_END", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path("DATA") / "dataset_player_KDE_clean_enriched.pkl"),
    )
    parser.add_argument("--output-dir", default="R7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
