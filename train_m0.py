import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from R7.train_r7 import (  # noqa: E402
    NROWS,
    POSITION_VALUES,
    R7Model,
    batch_from_indices as r7_batch_from_indices,
    code_maps,
    extract_match_metadata_and_lineups,
    extract_positions_and_ratings,
    make_tensors as make_r7_tensors,
    split_id_from_date,
)


LABEL_NAMES = ("home_win", "draw", "away_win")


def parameter_count(model, trainable_only=True):
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return int(sum(param.numel() for param in params))


def match_label(home_score, away_score):
    if int(home_score) > int(away_score):
        return 0
    if int(home_score) == int(away_score):
        return 1
    return 2


def reconstruct_r7_row_identity(pkl_path, arrays):
    leagues, dates, lineup_cols, after_lineups_pos = extract_match_metadata_and_lineups(
        pkl_path
    )
    positions, ratings = extract_positions_and_ratings(pkl_path, after_lineups_pos)

    league_map = code_maps(leagues)
    position_map = code_maps(POSITION_VALUES)
    rows = []
    for match_idx in range(NROWS):
        context = infer_context_or_raise(positions[match_idx], position_map)
        split_id = split_id_from_date(dates[match_idx])
        if split_id < 0:
            continue
        for player_id, rating in ratings[match_idx].items():
            if rating is None or player_id not in context:
                continue
            is_home, is_sub, position = context[player_id]
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
    histories = defaultdict(list)
    kept = []
    for date, match_idx, player_id, rating, split, league, is_home, is_sub, position in rows:
        history = histories[player_id]
        if history:
            kept.append(
                {
                    "date": date,
                    "match_idx": match_idx,
                    "player_id": player_id,
                    "rating": rating,
                    "split": split,
                    "league": league,
                    "is_home": is_home,
                    "is_sub": is_sub,
                    "position": position,
                }
            )
        history.append((rating, league, is_home, is_sub, position))
        if len(history) > 60:
            del history[:-60]

    if len(kept) != int(arrays["target"].shape[0]):
        raise RuntimeError(
            f"R7 row reconstruction mismatch: reconstructed={len(kept)} "
            f"cache={arrays['target'].shape[0]}"
        )

    target = arrays["target"].astype(np.float32)
    reconstructed_target = np.asarray([item["rating"] for item in kept], dtype=np.float32)
    if not np.allclose(target, reconstructed_target, atol=1e-5):
        max_diff = float(np.max(np.abs(target - reconstructed_target)))
        raise RuntimeError(f"R7 cache alignment check failed; max target diff={max_diff}")

    return kept, leagues, dates, lineup_cols


def infer_context_or_raise(position_dict, position_map):
    from R7.train_r7 import infer_context_from_position_order

    return infer_context_from_position_order(position_dict, position_map)


@torch.no_grad()
def predict_r7(model, tensors, row_indices, device, batch_size):
    model.eval()
    preds = []
    for start in range(0, len(row_indices), batch_size):
        batch_indices = row_indices[start : start + batch_size]
        batch = r7_batch_from_indices(tensors, batch_indices, device)
        preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def load_frozen_r7(args, arrays, device):
    model = R7Model().to(device)
    checkpoint = torch.load(args.r7_model, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    tensors, _ = make_r7_tensors(arrays)
    all_rows = np.arange(arrays["target"].shape[0])
    pred = predict_r7(model, tensors, all_rows, device, args.r7_eval_batch_size)
    return pred, int(checkpoint.get("parameter_count", parameter_count(model, False)))


def standardize_by_train(values, train_mask):
    mean = float(values[train_mask].mean())
    std = float(values[train_mask].std())
    if std < 1e-6:
        std = 1.0
    return ((values - mean) / std).astype(np.float32), mean, std


def build_m0_feature_cache(args, device):
    r7_cache_path = Path(args.r7_cache)
    loaded = np.load(r7_cache_path)
    arrays = {key: loaded[key] for key in loaded.files}
    r7_metadata = json.loads(r7_cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    r7_pred, r7_parameter_count = load_frozen_r7(args, arrays, device)

    pkl_path = Path(args.data)
    row_info, leagues, dates, lineup_cols = reconstruct_r7_row_identity(pkl_path, arrays)
    df = pd.read_pickle(pkl_path)

    by_match = defaultdict(lambda: {"home": [], "away": []})
    for row_idx, item in enumerate(row_info):
        side = "home" if item["is_home"] else "away"
        by_match[item["match_idx"]][side].append(
            (
                int(item["is_sub"]),
                int(item["player_id"]),
                float(r7_pred[row_idx]),
                int(item["league"]),
                int(item["is_home"]),
                int(item["is_sub"]),
                int(item["position"]),
            )
        )

    selected_matches = []
    labels = []
    splits = []
    for match_idx in range(NROWS):
        split = split_id_from_date(dates[match_idx])
        if split < 0:
            continue
        groups = by_match[match_idx]
        if not groups["home"] or not groups["away"]:
            continue
        row = df.iloc[match_idx]
        selected_matches.append(match_idx)
        labels.append(match_label(row["Home_Score"], row["Away_Score"]))
        splits.append(split)

    if args.max_players > 0:
        max_players = int(args.max_players)
    else:
        max_players = max(
            max(len(by_match[match_idx]["home"]), len(by_match[match_idx]["away"]))
            for match_idx in selected_matches
        )

    n_matches = len(selected_matches)
    home_rating = np.zeros((n_matches, max_players), dtype=np.float32)
    away_rating = np.zeros((n_matches, max_players), dtype=np.float32)
    home_league = np.zeros((n_matches, max_players), dtype=np.int64)
    away_league = np.zeros((n_matches, max_players), dtype=np.int64)
    home_side = np.zeros((n_matches, max_players), dtype=np.int64)
    away_side = np.zeros((n_matches, max_players), dtype=np.int64)
    home_sub = np.zeros((n_matches, max_players), dtype=np.int64)
    away_sub = np.zeros((n_matches, max_players), dtype=np.int64)
    home_position = np.zeros((n_matches, max_players), dtype=np.int64)
    away_position = np.zeros((n_matches, max_players), dtype=np.int64)
    home_mask = np.zeros((n_matches, max_players), dtype=np.bool_)
    away_mask = np.zeros((n_matches, max_players), dtype=np.bool_)
    home_count = np.zeros(n_matches, dtype=np.int64)
    away_count = np.zeros(n_matches, dtype=np.int64)
    truncated_groups = 0

    for out_idx, match_idx in enumerate(selected_matches):
        for prefix, records in (("home", by_match[match_idx]["home"]), ("away", by_match[match_idx]["away"])):
            records = sorted(records, key=lambda item: (item[0], item[1]))
            if len(records) > max_players:
                truncated_groups += 1
                records = records[:max_players]
            count = len(records)
            rating, league, side, sub, position, mask, count_arr = (
                (home_rating, home_league, home_side, home_sub, home_position, home_mask, home_count)
                if prefix == "home"
                else (away_rating, away_league, away_side, away_sub, away_position, away_mask, away_count)
            )
            count_arr[out_idx] = count
            for player_idx, record in enumerate(records):
                _, _, pred, league_id, side_id, sub_id, position_id = record
                rating[out_idx, player_idx] = pred
                league[out_idx, player_idx] = league_id
                side[out_idx, player_idx] = side_id
                sub[out_idx, player_idx] = sub_id
                position[out_idx, player_idx] = position_id
                mask[out_idx, player_idx] = True

    split_array = np.asarray(splits, dtype=np.int8)
    train_mask = split_array == 0
    all_ratings = np.concatenate([home_rating[home_mask], away_rating[away_mask]])
    all_rating_split = np.concatenate(
        [
            np.repeat(split_array, max_players)[home_mask.reshape(-1)],
            np.repeat(split_array, max_players)[away_mask.reshape(-1)],
        ]
    )
    _, rating_mean, rating_std = standardize_by_train(all_ratings, all_rating_split == 0)
    home_rating = ((home_rating - rating_mean) / rating_std).astype(np.float32)
    away_rating = ((away_rating - rating_mean) / rating_std).astype(np.float32)

    features = {
        "home_rating": home_rating,
        "away_rating": away_rating,
        "home_league": home_league,
        "away_league": away_league,
        "home_side": home_side,
        "away_side": away_side,
        "home_sub": home_sub,
        "away_sub": away_sub,
        "home_position": home_position,
        "away_position": away_position,
        "home_mask": home_mask,
        "away_mask": away_mask,
        "label": np.asarray(labels, dtype=np.int64),
        "split": split_array,
        "match_idx": np.asarray(selected_matches, dtype=np.int64),
    }
    metadata = {
        "model": "M0",
        "source_data": str(pkl_path),
        "source_r7_cache": str(r7_cache_path),
        "source_r7_model": str(args.r7_model),
        "source_r7_metadata": r7_metadata,
        "r7_parameter_count": r7_parameter_count,
        "label_names": list(LABEL_NAMES),
        "rating_standardization": {
            "mean": rating_mean,
            "std": rating_std,
            "source": "frozen_r7_predictions_train_players",
        },
        "max_players": int(max_players),
        "truncated_groups": int(truncated_groups),
        "n_matches": int(n_matches),
        "n_matches_by_split": {
            "train": int(np.sum(split_array == 0)),
            "dev": int(np.sum(split_array == 1)),
            "val": int(np.sum(split_array == 2)),
            "test": int(np.sum(split_array == 3)),
        },
        "home_player_count": {
            "min": int(home_count.min()),
            "mean": float(home_count.mean()),
            "max": int(home_count.max()),
        },
        "away_player_count": {
            "min": int(away_count.min()),
            "mean": float(away_count.mean()),
            "max": int(away_count.max()),
        },
        "position_map": r7_metadata["position_map"],
        "league_map": r7_metadata["league_map"],
        "split_ids": r7_metadata["split_ids"],
    }

    cache_path = Path(args.feature_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **features)
    cache_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return features, metadata


def load_or_build_features(args, device):
    cache_path = Path(args.feature_cache)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached M0 features from {cache_path}...", flush=True)
        loaded = np.load(cache_path)
        features = {key: loaded[key] for key in loaded.files}
        metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        return features, metadata
    print("Building M0 match-level features...", flush=True)
    return build_m0_feature_cache(args, device)


class M0Model(torch.nn.Module):
    def __init__(
        self,
        n_positions,
        n_leagues,
        rating_dim,
        position_dim,
        league_dim,
        side_dim,
        sub_dim,
        d_model,
        num_heads,
        dropout,
        head_hidden,
    ):
        super().__init__()
        self.position_embedding = torch.nn.Embedding(n_positions, position_dim)
        self.league_embedding = torch.nn.Embedding(n_leagues, league_dim)
        self.side_embedding = torch.nn.Embedding(2, side_dim)
        self.sub_embedding = torch.nn.Embedding(2, sub_dim)
        player_input_dim = rating_dim + position_dim + league_dim + side_dim + sub_dim
        self.player_mlp = torch.nn.Sequential(
            torch.nn.Linear(player_input_dim, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
        )
        self.home_to_away = torch.nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.away_to_home = torch.nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.home_norm = torch.nn.LayerNorm(d_model)
        self.away_norm = torch.nn.LayerNorm(d_model)
        self.home_ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 2, d_model),
        )
        self.away_ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 2, d_model),
        )
        self.home_ffn_norm = torch.nn.LayerNorm(d_model)
        self.away_ffn_norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, head_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden, head_hidden // 2),
            torch.nn.GELU(),
            torch.nn.Linear(head_hidden // 2, len(LABEL_NAMES)),
        )

    def encode_players(self, rating, league, side, sub, position):
        pieces = [
            rating.unsqueeze(-1),
            self.league_embedding(league.long()),
            self.side_embedding(side.long()),
            self.sub_embedding(sub.long()),
            self.position_embedding(position.long()),
        ]
        return self.player_mlp(torch.cat(pieces, dim=-1))

    @staticmethod
    def masked_mean(values, mask):
        weights = mask.float().unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        home_rating,
        away_rating,
        home_league,
        away_league,
        home_side,
        away_side,
        home_sub,
        away_sub,
        home_position,
        away_position,
        home_mask,
        away_mask,
    ):
        home = self.encode_players(
            home_rating, home_league, home_side, home_sub, home_position
        )
        away = self.encode_players(
            away_rating, away_league, away_side, away_sub, away_position
        )
        home_context, _ = self.home_to_away(
            query=home,
            key=away,
            value=away,
            key_padding_mask=~away_mask.bool(),
            need_weights=False,
        )
        away_context, _ = self.away_to_home(
            query=away,
            key=home,
            value=home,
            key_padding_mask=~home_mask.bool(),
            need_weights=False,
        )
        home = self.home_norm(home + home_context)
        away = self.away_norm(away + away_context)
        home = self.home_ffn_norm(home + self.home_ffn(home))
        away = self.away_ffn_norm(away + self.away_ffn(away))
        home_pool = self.masked_mean(home, home_mask)
        away_pool = self.masked_mean(away, away_mask)
        return self.head(torch.cat([home_pool, away_pool], dim=1))


def make_tensors(features):
    tensor_keys = [
        "home_rating",
        "away_rating",
        "home_league",
        "away_league",
        "home_side",
        "away_side",
        "home_sub",
        "away_sub",
        "home_position",
        "away_position",
        "home_mask",
        "away_mask",
        "label",
    ]
    tensors = {key: torch.from_numpy(features[key]) for key in tensor_keys}
    split = features["split"]
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
            "home_rating",
            "away_rating",
            "home_league",
            "away_league",
            "home_side",
            "away_side",
            "home_sub",
            "away_sub",
            "home_position",
            "away_position",
            "home_mask",
            "away_mask",
        ]
    )


@torch.no_grad()
def predict_probs(model, tensors, indices, device, batch_size):
    model.eval()
    probs = []
    logits_chunks = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = batch_from_indices(tensors, batch_indices, device)
        logits = model(*batch)
        logits_chunks.append(logits.detach().cpu())
        probs.append(torch.softmax(logits, dim=1).detach().cpu())
    if not probs:
        return np.zeros((0, len(LABEL_NAMES)), dtype=np.float32), np.zeros(
            (0, len(LABEL_NAMES)), dtype=np.float32
        )
    return (
        torch.cat(probs, dim=0).numpy().astype(np.float32),
        torch.cat(logits_chunks, dim=0).numpy().astype(np.float32),
    )


def metric_bundle(labels, probs, logits):
    one_hot = np.eye(len(LABEL_NAMES), dtype=np.float32)[labels]
    pred = probs.argmax(axis=1)
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    return {
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(pred == labels)),
        "cross_entropy": float(
            torch.nn.functional.cross_entropy(logits_tensor, labels_tensor).item()
        ),
        "n": int(labels.shape[0]),
    }


def evaluate_splits(model, tensors, indices, device, batch_size):
    metrics = {}
    for split_name, split_indices in indices.items():
        labels = tensors["label"][split_indices].numpy()
        probs, logits = predict_probs(model, tensors, split_indices, device, batch_size)
        metrics[split_name] = metric_bundle(labels, probs, logits)
    return metrics


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache = str(out_dir / "m0_match_features.npz")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    features, metadata = load_or_build_features(args, device)
    tensors, indices = make_tensors(features)

    model = M0Model(
        n_positions=len(metadata["position_map"]),
        n_leagues=len(metadata["league_map"]),
        rating_dim=1,
        position_dim=args.position_dim,
        league_dim=args.league_dim,
        side_dim=args.side_dim,
        sub_dim=args.sub_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    train_indices = indices["train"].copy()
    m0_param_count = parameter_count(model)
    r7_param_count = int(metadata["r7_parameter_count"])

    log_path = out_dir / f"training_log_seed{args.seed}.csv"
    best_path = out_dir / f"best_model_seed{args.seed}.pt"
    best_dev = float("inf")
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0

    print(
        f"Training M0 on {device} for {args.epochs} epochs; "
        f"train/dev/val/test matches = {len(indices['train'])}/{len(indices['dev'])}/"
        f"{len(indices['val'])}/{len(indices['test'])}; "
        f"M0 trainable parameters = {m0_param_count}; frozen R7 parameters = {r7_param_count}",
        flush=True,
    )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_brier",
                "dev_brier",
                "train_cross_entropy",
                "dev_cross_entropy",
                "train_accuracy",
                "dev_accuracy",
                "seconds",
            ],
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
                target = tensors["label"][batch_indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits detected during training.")
                loss = loss_fn(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_probs, train_logits = predict_probs(
                model, tensors, indices["train"], device, args.eval_batch_size
            )
            train_metrics = metric_bundle(
                tensors["label"][indices["train"]].numpy(), train_probs, train_logits
            )
            dev_probs, dev_logits = predict_probs(
                model, tensors, indices["dev"], device, args.eval_batch_size
            )
            dev_metrics = metric_bundle(
                tensors["label"][indices["dev"]].numpy(), dev_probs, dev_logits
            )
            elapsed = time.time() - start_time
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_brier": train_metrics["brier"],
                "dev_brier": dev_metrics["brier"],
                "train_cross_entropy": train_metrics["cross_entropy"],
                "dev_cross_entropy": dev_metrics["cross_entropy"],
                "train_accuracy": train_metrics["accuracy"],
                "dev_accuracy": dev_metrics["accuracy"],
                "seconds": elapsed,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch:03d} train_loss={row['train_loss']:.5f} "
                f"train_brier={row['train_brier']:.5f} dev_brier={row['dev_brier']:.5f} "
                f"dev_acc={row['dev_accuracy']:.4f} seconds={elapsed:.1f}",
                flush=True,
            )

            if dev_metrics["brier"] < best_dev - args.min_delta:
                best_dev = dev_metrics["brier"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state": best_state,
                        "best_epoch": best_epoch,
                        "best_dev_brier": best_dev,
                        "seed": args.seed,
                        "m0_trainable_parameter_count": m0_param_count,
                        "r7_frozen_parameter_count": r7_param_count,
                        "total_parameter_count": m0_param_count + r7_param_count,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs; best dev Brier at epoch {best_epoch}.",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    metrics = evaluate_splits(model, tensors, indices, device, args.eval_batch_size)

    params = {
        "model": "M0",
        "checkpoint_metric": "dev_brier",
        "m0_trainable_parameter_count": m0_param_count,
        "r7_frozen_parameter_count": r7_param_count,
        "total_parameter_count": m0_param_count + r7_param_count,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "head_hidden": args.head_hidden,
        "position_dim": args.position_dim,
        "league_dim": args.league_dim,
        "side_dim": args.side_dim,
        "sub_dim": args.sub_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    result = {
        "seed": args.seed,
        "model": "M0",
        "epochs_requested": args.epochs,
        "selected_epoch_by_dev_brier": best_epoch,
        "best_dev_brier": best_dev,
        "metrics": metrics,
        "parameters": params,
        "metadata": metadata,
        "artifacts": {
            "feature_cache": str(out_dir / "m0_match_features.npz"),
            "training_log": str(log_path),
            "best_model": str(best_path),
            "best_params": str(out_dir / f"best_params_seed{args.seed}.json"),
            "metrics": str(out_dir / f"metrics_seed{args.seed}.json"),
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
    parser.add_argument(
        "--r7-cache", default=str(Path("R7") / "r7_player_history_features.npz")
    )
    parser.add_argument("--r7-model", default=str(Path("R7") / "best_model_seed42.pt"))
    parser.add_argument("--output-dir", default="M0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--r7-eval-batch-size", type=int, default=65536)
    parser.add_argument("--max-players", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--position-dim", type=int, default=8)
    parser.add_argument("--league-dim", type=int, default=4)
    parser.add_argument("--side-dim", type=int, default=2)
    parser.add_argument("--sub-dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
