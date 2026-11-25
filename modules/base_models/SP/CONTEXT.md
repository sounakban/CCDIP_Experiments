# SP — the subtask policy (code name LLP, low-level policy)

The goal-conditioned driving policy that the subtask-selection policy commands. Trained with
PPO against a hardcoded goal sampler; the trained SSP (`modules/base_models/SSP/`) later
replaces that sampler and drives this policy frozen.

## Files

| file | what it is |
|---|---|
| `hrl_llp_core.py` | shared building blocks: goal sampling, the sub-episode reward and its reach/miss flags, the augmented observation (`llp_augment_obs`), the `GoalConditionedExtractor` encoder and the `Agent` network |
| `train_llp_PPO.py` | the PPO trainer (tyro `Args`); writes `hrl_llp_metadata.json` and `model_final.pt` into `out_path/<run>` |

## Observation layout (what the policy sees)

`base(959) | goal_dx | goal_dy | ts_norm | lat_prev | lon_prev | env_reward_coef`: the raw
environment observation with the ego's absolute position zeroed, then the goal offset in
metres, the sub-episode clock, the previous action's lateral and longitudinal components and
the per-sub-episode environment-reward coefficient.

## Consumers

`SSP` (training, simulation, evaluation) and `human_modeling/inference` build SP observations
through `llp_augment_obs` and `GoalConditionedExtractor`.
