# SSP — the subtask-selection policy (code name HLP, high-level policy)

Chooses, once per block (one SP sub-episode), a goal (lane + forward-distance bin) and the
scope (which cars the frozen SP gets to see), conditioned on a per-episode cognitive-profile
vector θ. Trained with PPO on the full driving task with the SP frozen inside the rollout.

## Files

| file | what it is |
|---|---|
| `hlp_core.py` | `HLPAgent` (goal heads + mask head), `build_hlp_input` (the SSP input vector), `hlp_collect_rollout` (block-level rollout that steps the frozen SP inside), the θ-weighted reward terms, θ sampling and normalisation, `HLP_PARAM_NAMES` |
| `hlp_policy.py` | loads an SSP checkpoint for inference (`load_hlp_policy`); clamps θ to the trained ranges; the `grad_enabled` switch lets the inference package differentiate through it |
| `train_hlp_PPO.py` | the PPO trainer (tyro `Args`); SP-side settings are read from the SP checkpoint's `hrl_llp_metadata.json`, never from the CLI; writes `hlp_metadata.json`, `hlp_model_final.pt`, copies of the SP files, training plots and one rollout |
| `hlp_eval.py` | the paired evaluation harness: `run` rolls the hierarchy out on a set of maps for a batch of θ and returns per-map metrics (cars shown per step, crashes per episode and per 1,000 steps, flags, goal statistics); `sweep` varies one coefficient over values on the SAME maps (construction seeds `0..sims-1`) with the others fixed at `GAME_THETA` |
| `plot_cost_performance.py` | paper Figure 2b: sweeps one coefficient over its trained range via `hlp_eval.sweep` and plots crashes per episode against cars attended per step, one dot per value coloured by the coefficient, with a +-1 standard-error band; a short pilot sweep picks the values so the dots sit at evenly spaced attention levels; writes the pilot and sweep tables as CSV and the figure as PNG and PDF |
| `simulate_hlp.py` | rollout of a checkpoint directory to video and a JSONL trace, masked cars dimmed (paper Figure 1c) |
| `tests/` | `test_hlp_core.py`, `test_hlp_policy.py`, `test_hlp_eval.py` (seed pairing and determinism of the harness) |

Run the tests with `uv run python -m pytest modules/base_models/SSP/tests`.

## Input layout

`processed obs | K car slots | nearest-flag (3) | θ (P, scaled to -1..1)`; `hlp_metadata.json`
records `hlp_input_dim` (6587 at K = 6).

## θ

Six coefficients in `HLP_PARAM_NAMES` order: collision penalty, cognitive (attentional) cost
per attended car per step, goal proximity cost, flag-miss penalty, flag-collect reward,
goal-flag distance penalty. Each is drawn uniformly per episode from `[min, max]` in
`hlp_metadata.json`'s `hlp_param_config`; a zero-width range (the two flag terms in the
paper's checkpoint) is a fixed coefficient that the fit does not estimate. `hlp_policy`
clamps queries to the trained range.

## Rollout

`hlp_collect_rollout` runs B environments in lockstep: at each block boundary the SSP picks a
goal and a mask, the SP then drives for up to `sub_episode_horizon` steps on the masked
observation until the goal is reached, missed or the horizon ends, and the block reward is the
environment reward plus the θ-weighted terms. θ is redrawn per env-episode at block init.
The mask head is
permutation-equivariant over car slots (a pooled, per-slot head), so slot order does not
change the policy's output.
