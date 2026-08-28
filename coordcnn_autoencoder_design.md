# H1 CoordCNN Autoencoder Design

## Scope

This H1 stage only builds and evaluates the lightweight CoordCNN autoencoders. It does not train the full player prediction model yet.

Two autoencoders will be trained with the same architecture but separate weights:

- `player_coordcnn_ae`: trained on normalized player heatmap densities.
- `team_coordcnn_ae`: trained on normalized team heatmap densities.

The autoencoder must not learn raw heatmap mass. For every raw heatmap `H`:

```text
mass = sum(H)
density = H / max(mass, eps)
```

The CoordCNN receives `density`, not `H`. Historical mass is kept as a separate scalar for later stages, but it is not part of the AE reconstruction target.

## Data Split

Use `Match_Date` and derive season by season start year:

```text
season_start_year = year(Match_Date) if month(Match_Date) >= 7 else year(Match_Date) - 1
```

Splits:

```text
train: 2015/16 - 2022/23  -> season_start_year 2015..2022
dev:   2023/24            -> season_start_year 2023
val:   2024/25            -> season_start_year 2024
test:  2025/26            -> season_start_year 2025
```

Usage:

- `train`: optimize model weights.
- `dev`: early stopping.
- `val`: compare model variants and select the best model.
- `test`: report final held-out result only after model selection.

Seed: `42` for Python, NumPy, PyTorch, dataloader shuffling, and CUDA determinism where available.

## Heatmap Handling

Input heatmaps are `72 x 112`.

For player AE:

- Source: `Player_Heatmap[player_id]`.
- Skip zero-mass heatmaps for density reconstruction because density is undefined.
- Record skipped zero-mass count in the report.

For team AE:

- Source: `Team_Heatmap(Home)` and `Team_Heatmap(Away)`.
- Normalize each team heatmap to density.
- Keep original team mass only for summary/reporting and later model stages.

## Model Design

The model is intentionally small. It should encode spatial shape, not memorize heatmaps with large capacity.

### Input

CoordConv input:

```text
[density, x_coord, y_coord]
shape = 3 x 72 x 112
```

Coordinates are normalized to `[-1, 1]`.

### Encoder

Use CoordConv blocks. Each block appends fresh `x/y` coordinate maps before convolution.

```text
Input: density, 1 x 72 x 112

CoordConvBlock 1: (1 + 2) -> 16, kernel 3, stride 2  -> 16 x 36 x 56
CoordConvBlock 2: (16 + 2) -> 24, kernel 3, stride 2 -> 24 x 18 x 28
CoordConvBlock 3: (24 + 2) -> 32, kernel 3, stride 2 -> 32 x 9 x 14
CoordConvBlock 4: (32 + 2) -> 48, kernel 3, stride 2 -> 48 x 5 x 7

GlobalAvgPool + GlobalMaxPool -> 96
MLP: 96 -> 64 -> latent_dim
```

Default:

```text
latent_dim = 32
activation = SiLU
normalization = GroupNorm
dropout = 0.0 for first version
```

### Decoder

Use a low-parameter coordinate-conditioned implicit decoder instead of a large deconvolutional decoder.

For every pitch cell `(x, y)`:

```text
decoder_input(x, y) = [z, x_coord, y_coord]
```

Then a shared `1x1` network produces logits:

```text
1x1 Conv / pixel MLP: (latent_dim + 2) -> 64 -> 32 -> 1
spatial_softmax over 72 * 112 cells
```

The output is always a normalized density:

```text
sum(D_hat) = 1
D_hat >= 0
```

This decoder keeps parameter count low and forces the latent vector to encode global spatial shape.

### Stats Head

The encoder also predicts interpretable density statistics from `z`:

```text
StatsHead: latent_dim -> stats_dim
```

Stats target:

```text
centroid_x
centroid_y
spread_x
spread_y
covariance_xy
entropy
left_lane_share
center_lane_share
right_lane_share
defensive_third_share
middle_third_share
attacking_third_share
```

No raw mass is included in this stats head.

## Training Loss

The AE training loss works only on normalized densities:

```text
L_ae =
  w_ce    * CrossEntropyDensity(D, D_hat)
+ w_js    * JensenShannon(D, D_hat)
+ w_l1    * MeanAbsoluteError(D, D_hat) * H * W
+ w_stats * SmoothL1(stats_hat, stats(D))
+ w_wass  * MarginalWasserstein(D, D_hat)
```

Default weights:

```text
w_ce = 1.0
w_js = 0.5
w_l1 = 0.25
w_stats = 0.5
w_wass = 0.0
```

Rationale:

- Cross entropy / KL gives a proper density-learning signal.
- Jensen-Shannon is symmetric and more stable as a reported shape metric.
- Scaled L1 keeps pixel-level reconstruction honest.
- Stats loss pushes `z` to retain centroid, spread, entropy, and pitch-zone structure.
- Marginal Wasserstein is a cheap spatial transport diagnostic. It is reported in
  version 1 but has zero training weight by default, so it can be enabled later as
  a controlled ablation without changing the pipeline.

`MarginalWasserstein` computes exact one-dimensional Wasserstein-1 distances for
the x and y marginals using cumulative distribution functions, then averages the
two axes. Full 2D Sinkhorn on all `72 x 112` cells is intentionally excluded from
the first training loss because its memory and compute cost are disproportionate.
If the baseline needs a stronger transport term, the next comparison should use
Sinkhorn on `18 x 28` downsampled densities and select it only by validation score.

## Metrics

Report metrics separately for player AE and team AE on train/dev/val/test.

### Primary Early Stopping Metric

Use dev composite score:

```text
dev_score =
  JS
+ 0.10 * centroid_error_norm
+ 0.10 * spread_error_norm
+ 0.10 * zone_share_mae
+ 0.05 * entropy_abs_error
```

Lower is better.

`JS` is the main term because this is a density reconstruction task. The extra terms protect against visually wrong heatmaps that still have acceptable pixel divergence.

### Model Selection Metric

Use the same composite score on `val`.

Process:

1. Each model variant stops by best `dev_score`.
2. Compare variants by `val_score`.
3. Select the best variant.
4. Run final report on `test` once.

### Full Report Metrics

Density metrics:

```text
cross_entropy_density
kl_divergence
jensen_shannon_divergence
pixel_mae_scaled
pixel_rmse_scaled
cosine_similarity
marginal_wasserstein_1
```

Spatial statistics metrics:

```text
centroid_error_x
centroid_error_y
centroid_error_l2_cells
centroid_error_l2_pitch_norm
spread_x_abs_error
spread_y_abs_error
covariance_xy_abs_error
entropy_abs_error
zone_share_mae
```

Quality/control metrics:

```text
num_samples
num_zero_mass_skipped
mean_original_mass
median_original_mass
parameter_count_total
parameter_count_trainable
best_epoch_by_dev
best_dev_score
val_score_at_best_dev
test_score_for_selected_model
```

No AE mass reconstruction metric is used because mass is intentionally outside the autoencoder.

## Outputs

All generated files go under `H1`.

Planned output layout:

```text
H1/
  coordcnn_autoencoder_design.md
  configs/
    player_ae.yaml
    team_ae.yaml
  checkpoints/
    player_coordcnn_ae_best.pt
    team_coordcnn_ae_best.pt
  reports/
    player_coordcnn_ae_metrics.json
    team_coordcnn_ae_metrics.json
    h1_summary.md
  artifacts/
    split_summary.json
    sample_reconstructions/
```

Each checkpoint will include:

```text
model_state_dict
optimizer_state_dict
epoch
seed
config
parameter_count_total
parameter_count_trainable
best_dev_metrics
val_metrics
test_metrics
```

## First Version Defaults

```text
seed: 42
latent_dim: 32
batch_size: 256
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
max_epochs: 100
early_stopping_patience: 10
min_delta: 1e-4
gradient_clip_norm: 1.0
mixed_precision: true if CUDA is available
```

Player and team AEs can use the same defaults, but their checkpoints and reports remain separate.
