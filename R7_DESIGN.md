# R7 design

R7 starts from the R5m player-history model and adds pre-match team context.

## Data split

- train: seasons starting 2015 through 2022
- dev: season starting 2023
- val: season starting 2024
- test: season starting 2025

The training script keeps the R5m behavior: train on train, select the best epoch by dev MAE, use val for model selection, and report test without using it for training.

## Added team features

For every match, R7 builds home and away team features from matches that happened before that match only.

- `team_rating_ewma_60`
- `team_rating_ewma_10`
- `team_rating_ewma_3`
- `team_goal_diff_ewma`
- `team_goals_for_ewma`
- `team_goals_against_ewma`
- `team_points_ewma`

Each player row receives these as own-team and opponent-team features according to the player's side.

## Added model terms

R7 adds absolute and relative team terms:

- own team strength weights for the three rating EWMAs
- opponent strength weights for the three rating EWMAs
- strength-delta weights for the three rating EWMAs
- own and opponent context weights for goal difference, goals for, goals against, and points
- form-delta weights for goal-difference delta and points delta
- attack-vs-defense delta weight
- home-adjusted strength-delta interaction weight
- position-specific `position x strength_delta` effects

The existing R5m `(3, 4)` EWMA bonus parameters are preserved, so the 60/10/3 EWMA bonuses are not shared.

## Run command

```powershell
python .\R7\train_r7.py --seed 42 --epochs 50
```
