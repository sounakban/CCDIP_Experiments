"""Unit tests for hlp_core.py — run with: python -m pytest test_hlp_core.py -v"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.core.maps.objects import objectID_dict
from modules.base_models.SSP.hlp_core import (
    OBS_SIZE, OBS_FEATURES, CAR_KIND_IDS, EXPANDED_FEATURES, NUM_CLASSES,
    HLP_PARAM_NAMES, HLP_NUM_PARAMS, FLAG_FEAT_DIM,
    apply_car_mask, build_hlp_input, hlp_extract_car_rows,
    hlp_goal_proximity_cost, hlp_sample_params, HLPAgent, PooledMaskHead,
)

DEVICE = torch.device("cpu")
B = 4      # batch size used in most tests
K = 4      # car slots
CAR_ID = CAR_KIND_IDS[0]   # first enemy-car kind ID (e.g. 5)

# [HLP-goal] goal-head fixture params (arbitrary but self-consistent)
NUM_LANES    = 3
NUM_DY_BINS  = 18
DY_MIN       = 1.0
DY_WIDTH     = 0.5                       # → range [1.0, 10.0]
DY_MAX       = DY_MIN + NUM_DY_BINS * DY_WIDTH
LANE_CENTERS = torch.tensor([1.5, 4.5, 7.5])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(B: int, car_rows: list[int] | None = None, car_id: int = CAR_ID) -> torch.Tensor:
    """Create a (B, OBS_SIZE*OBS_FEATURES) zero obs with optional car rows."""
    obs = torch.zeros(B, OBS_SIZE * OBS_FEATURES, device=DEVICE)
    obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)
    if car_rows:
        for r in car_rows:
            obs_3d[:, r, 6] = float(car_id)   # set kind_id column
            obs_3d[:, r, 0] = 1.0              # non-zero position so row is non-trivial
    return obs_3d.view(B, -1)


def _valid_slots(B: int, n_valid: int) -> torch.Tensor:
    """(B, K) bool mask with the first n_valid slots True."""
    return (torch.arange(K, device=DEVICE).unsqueeze(0)
            < torch.full((B, 1), n_valid, device=DEVICE))


# [HLP-maskhead] the pooled head slices car_feats out of base_in at the real layout offset,
# so every base_in must be full-width.
POOLED_INPUT_DIM = OBS_SIZE * EXPANDED_FEATURES + K * EXPANDED_FEATURES + 3 + HLP_NUM_PARAMS


def _goal_agent(input_dim: int = POOLED_INPUT_DIM, hidden: int = 32) -> HLPAgent:
    """HLPAgent (fixture params above)."""
    return HLPAgent(
        input_dim=input_dim, K=K, hidden_size=hidden,
        num_lanes=NUM_LANES, num_dy_bins=NUM_DY_BINS,
        goal_dy_min=DY_MIN, goal_dy_bin_width=DY_WIDTH, lane_centers=LANE_CENTERS,
    )


def _mask_logits(agent: HLPAgent, base_in: torch.Tensor, goal_dxdy: torch.Tensor) -> torch.Tensor:
    """The pooled mask head evaluated directly, as HLPAgent.get_action_and_value does it."""
    c0 = OBS_SIZE * EXPANDED_FEATURES
    c1 = c0 + K * EXPANDED_FEATURES
    car_feats = base_in[:, c0:c1].view(-1, K, EXPANDED_FEATURES)
    context = torch.cat([base_in[:, :c0], base_in[:, c1:], goal_dxdy], dim=1)
    return agent.pooled_actor(car_feats, context)


def _pooled_base(B_: int = B) -> torch.Tensor:
    """(B, POOLED_INPUT_DIM) base_in with distinguishable, VALID car slots."""
    base = torch.randn(B_, POOLED_INPUT_DIM)
    c0 = OBS_SIZE * EXPANDED_FEATURES
    cars = torch.randn(B_, K, EXPANDED_FEATURES)
    base[:, c0:c0 + K * EXPANDED_FEATURES] = cars.reshape(B_, -1)
    return base


# ---------------------------------------------------------------------------
# Test 1: apply_car_mask — correct rows zeroed, row 0 never touched
# ---------------------------------------------------------------------------

class TestApplyCarMask:
    def test_mask_zeros_correct_row(self):
        """Car slot with mask=0 should zero that obs row."""
        car_row = 10
        obs = _make_obs(B, car_rows=[car_row])
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)

        # 1 valid slot; slots 1-3 point to a different (empty) row to avoid
        # scatter_ last-write-wins overwriting slot 0's hide=True
        car_row_indices = torch.zeros(B, K, dtype=torch.long, device=DEVICE)
        car_row_indices[:, 0] = car_row   # slot 0 → car row
        car_row_indices[:, 1:] = 1        # slots 1-3 → inert row (won't be hidden either)
        car_mask = torch.ones(B, K, device=DEVICE)
        car_mask[:, 0] = 0.0   # hide slot 0 (the only valid car)

        result = apply_car_mask(obs_3d, car_row_indices, car_mask, _valid_slots(B, 1))

        # Row 10 should be zeroed for all envs
        assert result[:, car_row, :].abs().max().item() == 0.0, "Masked row should be all-zeros"

    def test_mask_1_preserves_row(self):
        """Car slot with mask=1 should leave that obs row unchanged."""
        car_row = 15
        obs = _make_obs(B, car_rows=[car_row])
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)
        original_val = obs_3d[0, car_row, 0].item()

        car_row_indices = torch.full((B, K), car_row, dtype=torch.long, device=DEVICE)
        car_mask = torch.ones(B, K, device=DEVICE)  # all visible

        result = apply_car_mask(obs_3d, car_row_indices, car_mask, _valid_slots(B, K))

        assert result[0, car_row, 0].item() == original_val, "Unmasked row should be preserved"

    def test_invalid_slot_not_zeroed(self):
        """Invalid slots (no valid cars) should never zero row 0 even if mask=0."""
        obs = _make_obs(B)
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)
        obs_3d[:, 0, 0] = 99.0  # mark ego row with distinctive value

        # All slots point to row 0 (invalid clamped), no valid slots
        car_row_indices = torch.zeros(B, K, dtype=torch.long, device=DEVICE)
        car_mask = torch.zeros(B, K, device=DEVICE)   # all mask=0 (hide)

        result = apply_car_mask(obs_3d, car_row_indices, car_mask, _valid_slots(B, 0))

        assert (result[:, 0, 0] == 99.0).all(), "Ego row (row 0) must never be zeroed"

    def test_input_not_modified(self):
        """apply_car_mask must not modify the input tensor."""
        car_row = 20
        obs = _make_obs(B, car_rows=[car_row])
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)
        original = obs_3d.clone()

        car_row_indices = torch.full((B, K), car_row, dtype=torch.long, device=DEVICE)
        car_mask = torch.zeros(B, K, device=DEVICE)

        apply_car_mask(obs_3d, car_row_indices, car_mask, _valid_slots(B, K))

        assert torch.equal(obs_3d, original), "Input tensor must not be modified in-place"

    def test_multiple_cars_selective_zeroing(self):
        """With 2 real cars, hiding slot 0 but keeping slot 1 visible."""
        row_a, row_b = 5, 12
        obs = _make_obs(B, car_rows=[row_a, row_b])
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)

        car_row_indices = torch.zeros(B, K, dtype=torch.long, device=DEVICE)
        car_row_indices[:, 0] = row_a
        car_row_indices[:, 1] = row_b
        car_row_indices[:, 2:] = 1        # inert row for the invalid slots
        car_mask = torch.ones(B, K, device=DEVICE)
        car_mask[:, 0] = 0.0    # hide car A, keep car B

        result = apply_car_mask(obs_3d, car_row_indices, car_mask, _valid_slots(B, 2))

        assert result[:, row_a, 0].abs().max().item() == 0.0, "Car A row should be zeroed"
        assert (result[:, row_b, 0] > 0).all(), "Car B row should be visible"


# ---------------------------------------------------------------------------
# Test 2: hlp_extract_car_rows — car detection and slot clamping
# ---------------------------------------------------------------------------

class TestHlpExtractCarRows:
    def test_finds_car_rows(self):
        """Should return the correct row indices for present cars."""
        car_row = 30
        obs = _make_obs(B, car_rows=[car_row])
        row_indices, num_cars = hlp_extract_car_rows(obs, K)

        assert num_cars[0].item() == 1, f"Expected 1 car, got {num_cars[0].item()}"
        # One of the K slots should contain car_row
        found = (row_indices[0] == car_row).any().item()
        assert found, f"Car row {car_row} not found in {row_indices[0].tolist()}"

    def test_no_cars_gives_zero_num_cars(self):
        """With no enemy cars, num_cars should be 0 and indices clamped to 0."""
        obs = _make_obs(B)  # no cars
        row_indices, num_cars = hlp_extract_car_rows(obs, K)

        assert (num_cars == 0).all(), "Should report 0 cars when none present"
        assert (row_indices == 0).all(), "Invalid indices should be clamped to 0"

    def test_k_cars_fills_slots(self):
        """K or more cars should fill all K slots."""
        # Place K cars in consecutive rows (avoid row 0 — ego)
        car_rows = list(range(2, 2 + K))
        obs = _make_obs(1, car_rows=car_rows)
        row_indices, num_cars = hlp_extract_car_rows(obs, K)

        assert num_cars[0].item() == K, f"Expected {K} cars, got {num_cars[0].item()}"
        found_rows = set(row_indices[0].tolist())
        for r in car_rows:
            assert r in found_rows, f"Expected car row {r} in slots"

    def test_more_cars_than_k(self):
        """With more than K cars, should still return exactly K slots."""
        car_rows = list(range(2, 2 + K + 3))  # K+3 cars
        obs = _make_obs(1, car_rows=car_rows)
        row_indices, num_cars = hlp_extract_car_rows(obs, K)

        assert num_cars[0].item() == K, "num_cars should be capped at K"
        assert row_indices.shape == (1, K)

    def test_different_envs_independent(self):
        """Each env's cars should be detected independently."""
        obs = _make_obs(B)
        obs_3d = obs.view(B, OBS_SIZE, OBS_FEATURES)
        # Env 0: car at row 5; Env 1: car at row 50; others: no cars
        obs_3d[0, 5, 6] = float(CAR_ID)
        obs_3d[1, 50, 6] = float(CAR_ID)

        row_indices, num_cars = hlp_extract_car_rows(obs_3d.view(B, -1), K)

        assert num_cars[0].item() == 1 and (row_indices[0] == 5).any()
        assert num_cars[1].item() == 1 and (row_indices[1] == 50).any()
        assert num_cars[2].item() == 0
        assert num_cars[3].item() == 0

    def test_real_cars_form_a_prefix(self):
        """The slots callers mark valid (`arange(K) < num_cars`) must be exactly the real cars."""
        obs = _make_obs(1, car_rows=[30, 45])
        row_indices, num_cars = hlp_extract_car_rows(obs, K)
        n = int(num_cars[0])
        assert n == 2
        assert set(row_indices[0, :n].tolist()) == {30, 45}
        assert (row_indices[0, n:] == 0).all(), "filler slots must be clamped to row 0"

    def test_cars_first_even_when_topk_returns_filler_first(self, monkeypatch):
        """Force the order sorted=False is allowed to return and that broke the cluster's CPU."""
        real_topk = torch.topk

        def filler_first_topk(*args, **kwargs):
            vals, idx = real_topk(*args, **kwargs)
            if kwargs.get("sorted", True) or vals.dim() != 2:
                return vals, idx
            order = torch.argsort(vals.isfinite().to(torch.int8), dim=1, stable=True)
            return vals.gather(1, order), idx.gather(1, order)

        obs = _make_obs(1, car_rows=[30, 45])
        expected_rows, expected_n = hlp_extract_car_rows(obs, K)
        monkeypatch.setattr(torch, "topk", filler_first_topk)
        row_indices, num_cars = hlp_extract_car_rows(obs, K)
        assert torch.equal(num_cars, expected_n)
        assert torch.equal(row_indices, expected_rows), (
            "a filler-first topk order must not change which slots hold the cars")


# ---------------------------------------------------------------------------
# Test 3: block accumulation logic — episode_done freezes accumulation
# ---------------------------------------------------------------------------

class TestBlockAccumulation:
    """Simulate the inner-loop accumulation logic in isolation."""

    def _run_accumulation(self, env_done_steps: list[int | None], total_steps: int = 10):
        """
        Simulate accumulation for B=len(env_done_steps) envs.
        Each env optionally has an env_done event at the given step index.
        Returns (accum_env_reward, collision_count) after total_steps steps.
        """
        B_local = len(env_done_steps)
        reward_per_step = 1.0   # constant reward each step
        collision_per_step = 1.0

        accum_env_reward         = torch.zeros(B_local)
        collision_count          = torch.zeros(B_local)
        episode_done_this_block  = torch.zeros(B_local, dtype=torch.bool)

        for step in range(total_steps):
            # Simulate env_done
            env_done = torch.zeros(B_local, dtype=torch.bool)
            for b, done_step in enumerate(env_done_steps):
                if done_step is not None and step == done_step:
                    env_done[b] = True

            rewards_t    = torch.full((B_local,), reward_per_step)
            is_collision = torch.full((B_local,), collision_per_step)

            active = ~episode_done_this_block
            accum_env_reward += active.float() * rewards_t
            collision_count  += active.float() * is_collision

            episode_done_this_block |= env_done

        return accum_env_reward, collision_count

    def test_no_done_accumulates_all_steps(self):
        rewards, collisions = self._run_accumulation([None], total_steps=10)
        assert rewards[0].item() == 10.0
        assert collisions[0].item() == 10.0

    def test_done_at_step_3_freezes_after(self):
        """env_done at step 3 → accumulates steps 0-3 inclusive (4 steps), then frozen.
        The terminal step itself is accumulated before the done flag is latched."""
        rewards, collisions = self._run_accumulation([3], total_steps=10)
        assert rewards[0].item() == 4.0, f"Expected 4.0, got {rewards[0].item()}"
        assert collisions[0].item() == 4.0

    def test_done_at_step_0_zero_accumulation(self):
        """env_done at step 0 → active=True for step 0 (freeze sets on step 0), then frozen."""
        # active = ~episode_done_this_block; at step 0 episode_done is still False
        # so step 0 still accumulates, THEN episode_done is latched
        rewards, _ = self._run_accumulation([0], total_steps=5)
        assert rewards[0].item() == 1.0, f"Expected 1.0 (step 0 active), got {rewards[0].item()}"

    def test_multiple_envs_independent_freeze(self):
        """Each env freezes independently at its own env_done step."""
        rewards, _ = self._run_accumulation(
            env_done_steps=[None, 2, 7, 0], total_steps=10
        )
        assert rewards[0].item() == 10.0  # never done: all 10 steps
        assert rewards[1].item() == 3.0   # done at step 2: steps 0,1,2 active (terminal included)
        assert rewards[2].item() == 8.0   # done at step 7: steps 0-7 active (terminal included)
        assert rewards[3].item() == 1.0   # done at step 0: step 0 active

    def test_prev_actions_reset_on_env_done_not_subepisode(self):
        """prev_actions should reset to 4 on env_done, not on sub-episode boundaries."""
        prev_actions = torch.full((B,), 2, dtype=torch.long, device=DEVICE)   # action=2
        env_done     = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        env_done[0]  = True   # only env 0 hit done

        # Simulate the update from inner loop
        prev_actions[env_done] = 4   # IDLE on env reset

        assert prev_actions[0].item() == 4, "env_done env should have prev_action=4 (IDLE)"
        assert prev_actions[1].item() == 2, "non-done env should keep its action"


# ---------------------------------------------------------------------------
# Test 3b: flag-outcome accumulation — same active gate as collisions
# ---------------------------------------------------------------------------

class TestFlagAccumulation:
    """[HLP-flag] Flag miss/collect counts accumulate under active_for_accum, then freeze
    once the env fires its first sub_done (sub_done_ever) or an env_done in the block."""

    def _run(self, miss_events, collect_events, sub_done_step=None, total_steps=6):
        """Accumulate flag counts for a single env.

        :param miss_events: step indices where a flag miss fires.
        :param collect_events: step indices where a flag collect fires.
        :param sub_done_step: step index of the env's first sub_done (freezes accumulation).
        """
        flag_miss_count    = torch.zeros(1)
        flag_collect_count = torch.zeros(1)
        sub_done_ever      = torch.zeros(1, dtype=torch.bool)
        episode_done       = torch.zeros(1, dtype=torch.bool)

        for step in range(total_steps):
            is_flag_miss    = torch.tensor([1.0 if step in miss_events else 0.0])
            is_flag_collect = torch.tensor([1.0 if step in collect_events else 0.0])

            active = ~episode_done & ~sub_done_ever
            flag_miss_count    += active.float() * is_flag_miss
            flag_collect_count += active.float() * is_flag_collect

            # sub_done latches AFTER accumulation (mirrors inner-loop ordering)
            if sub_done_step is not None and step == sub_done_step:
                sub_done_ever |= True

        return flag_miss_count.item(), flag_collect_count.item()

    def test_counts_all_events_without_sub_done(self):
        miss, collect = self._run(miss_events=[1, 4], collect_events=[2], total_steps=6)
        assert miss == 2.0 and collect == 1.0

    def test_freezes_after_first_sub_done(self):
        """Events after the first sub_done are not attributed to this block's decision."""
        # sub_done at step 2 → steps 0,1,2 active; misses at 1 (counted) and 4 (frozen)
        miss, collect = self._run(
            miss_events=[1, 4], collect_events=[2, 5], sub_done_step=2, total_steps=6
        )
        assert miss == 1.0, f"only the pre-freeze miss counts, got {miss}"
        assert collect == 1.0, f"collect at step 2 (still active) counts; step 5 frozen, got {collect}"

    def test_no_events_zero(self):
        miss, collect = self._run(miss_events=[], collect_events=[], total_steps=6)
        assert miss == 0.0 and collect == 0.0

    def test_block_reward_sign(self):
        """Miss subtracts, collect adds — verify the block-close composition signs."""
        flag_miss_count    = torch.tensor([3.0])
        flag_collect_count = torch.tensor([2.0])
        miss_pen, collect_rew = 0.1, 0.05
        reward = -miss_pen * flag_miss_count + collect_rew * flag_collect_count
        assert reward.item() == pytest.approx(-0.3 + 0.1), "miss penalises, collect rewards"


# ---------------------------------------------------------------------------
# Test 4: HLPAgent — legacy mode (external goal, mask-only action)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 5: HLPAgent — goal selection (lane + dy-bin + mask joint action)
# ---------------------------------------------------------------------------

class TestHLPAgentGoalSelection:
    def _rollout(self, agent: HLPAgent, input_dim: int = POOLED_INPUT_DIM, deterministic: bool = False):
        base_in = torch.randn(B, input_dim)
        x_m = torch.rand(B) * 8.0
        y_m = torch.rand(B) * 100.0 + 50.0     # far from track start (no clamp)
        actions, logp, ent, value, goal_info = agent.get_action_and_value(
            base_in, x_m=x_m, y_m=y_m, deterministic=deterministic
        )
        return base_in, x_m, y_m, actions, logp, ent, value, goal_info

    def test_rollout_shapes_and_goal_info(self):
        agent = _goal_agent()
        _, _, _, actions, logp, ent, value, goal_info = self._rollout(agent)

        assert actions["mask"].shape == (B, K)
        assert actions["lane"].shape == (B,)
        assert actions["dy_bin"].shape == (B,)
        assert logp.shape == (B,) and logp.isfinite().all()
        assert ent.shape == (B,) and (ent >= 0).all()
        assert value.shape == (B,)
        assert goal_info is not None
        for key in ("goal_x", "goal_y", "dy", "goal_dxdy"):
            assert key in goal_info, f"goal_info missing {key}"
        assert goal_info["goal_dxdy"].shape == (B, 2)

    def test_goal_within_llp_support(self):
        """goal_x must be a lane centre; dy must be a bin centre inside [min, max]."""
        agent = _goal_agent()
        _, x_m, y_m, actions, _, _, _, goal_info = self._rollout(agent)

        # goal_x ∈ lane centres
        gx = goal_info["goal_x"].unsqueeze(1)                      # (B, 1)
        assert (gx == LANE_CENTERS.unsqueeze(0)).any(dim=1).all(), "goal_x must be a lane centre"
        # dy ∈ (min, max), at a bin centre
        dy = goal_info["dy"]
        assert (dy > DY_MIN).all() and (dy < DY_MAX).all(), "dy must lie inside the goal range"
        expected_dy = DY_MIN + (actions["dy_bin"].float() + 0.5) * DY_WIDTH
        assert torch.allclose(dy, expected_dy, atol=1e-6), "dy must be the chosen bin centre"
        # goal_y = clamp(y_m − dy, 0) — y_m chosen large so no clamp here
        assert torch.allclose(goal_info["goal_y"], y_m - dy, atol=1e-5)

    def test_goal_y_clamped_at_track_start(self):
        """goal_y must never go below 0 (same clamp as llp_sample_goals)."""
        agent = _goal_agent()
        base_in = torch.randn(B, POOLED_INPUT_DIM)
        x_m = torch.rand(B) * 8.0
        y_m = torch.zeros(B)                                       # at track start → clamp active
        _, _, _, _, goal_info = agent.get_action_and_value(base_in, x_m=x_m, y_m=y_m)
        assert (goal_info["goal_y"] >= 0.0).all(), "goal_y must be clamped to ≥ 0"

    def test_joint_logprob_composition(self):
        """Joint logp must equal lane + dy_bin + mask component logprobs."""
        agent = _goal_agent()
        base_in, _, _, actions, logp, _, _, goal_info = self._rollout(agent)

        # Update-mode call with the stored samples must reproduce the rollout logp
        _, logp_update, _, _, _ = agent.get_action_and_value(
            base_in,
            goal_dxdy=goal_info["goal_dxdy"],
            given_samples=(actions["lane"], actions["dy_bin"], actions["mask"]),
        )
        assert torch.allclose(logp, logp_update, atol=1e-5), \
            "update-mode logp must match rollout logp for the same samples"

        # Manual composition from the heads
        gl = agent.goal_actor(base_in)
        lane_lp = torch.distributions.Categorical(logits=gl[:, :NUM_LANES]).log_prob(actions["lane"])
        bin_lp  = torch.distributions.Categorical(logits=gl[:, NUM_LANES:]).log_prob(actions["dy_bin"])
        mask_logits = _mask_logits(agent, base_in, goal_info["goal_dxdy"])
        mask_lp = torch.distributions.Bernoulli(logits=mask_logits).log_prob(actions["mask"]).sum(-1)
        assert torch.allclose(logp, lane_lp + bin_lp + mask_lp, atol=1e-5), \
            "joint logp must be the sum of lane + dy_bin + mask logprobs"

    def test_deterministic_is_repeatable(self):
        """deterministic=True must return identical actions on repeated calls."""
        agent = _goal_agent()
        base_in = torch.randn(B, POOLED_INPUT_DIM)
        x_m = torch.rand(B) * 8.0
        y_m = torch.rand(B) * 100.0 + 50.0
        a1, _, _, _, g1 = agent.get_action_and_value(base_in, x_m=x_m, y_m=y_m, deterministic=True)
        a2, _, _, _, g2 = agent.get_action_and_value(base_in, x_m=x_m, y_m=y_m, deterministic=True)
        assert torch.equal(a1["mask"], a2["mask"])
        assert torch.equal(a1["lane"], a2["lane"])
        assert torch.equal(a1["dy_bin"], a2["dy_bin"])
        assert torch.equal(g1["goal_y"], g2["goal_y"])

    def test_mask_conditioned_on_goal(self):
        """Different goal offsets must (generically) change the mask logits."""
        agent = _goal_agent()
        base_in = torch.randn(1, POOLED_INPUT_DIM)
        logits_a = _mask_logits(agent, base_in, torch.tensor([[0.0, -2.0]]))
        logits_b = _mask_logits(agent, base_in, torch.tensor([[3.0, -9.0]]))
        assert not torch.allclose(logits_a, logits_b), \
            "mask logits must depend on the goal conditioning input"


# ---------------------------------------------------------------------------
# Test 6: hlp_goal_proximity_cost — linear shape, boundaries, monotonicity
# ---------------------------------------------------------------------------

class TestGoalProximityCost:
    COEF = 0.05

    def test_boundaries(self):
        """Cost = coef at dy_min, 0 at dy_max."""
        dy = torch.tensor([DY_MIN, DY_MAX])
        cost = hlp_goal_proximity_cost(dy, DY_MIN, DY_MAX, self.COEF)
        assert cost[0].item() == pytest.approx(self.COEF), "max cost at closest goal"
        assert cost[1].item() == pytest.approx(0.0), "zero cost at farthest goal"

    def test_monotonically_decreasing(self):
        """Closer goals must always cost more."""
        dy = torch.linspace(DY_MIN, DY_MAX, steps=20)
        cost = hlp_goal_proximity_cost(dy, DY_MIN, DY_MAX, self.COEF)
        assert (cost[:-1] > cost[1:]).all(), "cost must strictly decrease with distance"

    def test_linear_midpoint(self):
        """Linear shape: midpoint dy → coef/2."""
        dy = torch.tensor([(DY_MIN + DY_MAX) / 2.0])
        cost = hlp_goal_proximity_cost(dy, DY_MIN, DY_MAX, self.COEF)
        assert cost[0].item() == pytest.approx(self.COEF / 2.0)

    def test_out_of_range_clamped(self):
        """dy outside [min, max] must clamp to the [0, coef] cost range."""
        dy = torch.tensor([DY_MIN - 5.0, DY_MAX + 5.0])
        cost = hlp_goal_proximity_cost(dy, DY_MIN, DY_MAX, self.COEF)
        assert cost[0].item() == pytest.approx(self.COEF)
        assert cost[1].item() == pytest.approx(0.0)

    def test_per_env_tensor_coef(self):
        """[HLP-param] (B,) θ column as coef must broadcast elementwise per env."""
        dy = torch.tensor([DY_MIN, DY_MIN])           # both at max-cost distance
        coef = torch.tensor([0.1, 0.4])               # per-env θ values
        cost = hlp_goal_proximity_cost(dy, DY_MIN, DY_MAX, coef)
        assert cost[0].item() == pytest.approx(0.1)
        assert cost[1].item() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Test 7: [HLP-param] θ sampling, obs layout, lazy resample, per-env reward
# ---------------------------------------------------------------------------

# (P, 2) ranges fixture — mixed degenerate and wide ranges
PARAM_RANGES = torch.tensor([
    [0.1, 0.5],    # hlp_collision_penalty
    [0.0, 0.04],   # hlp_cognitive_cost_coef
    [0.05, 0.05],  # hlp_goal_proximity_cost_coef (degenerate — fixed)
    [0.0, 0.3],    # hlp_flag_miss_penalty
    [0.0, 0.0],    # hlp_flag_collect_reward (degenerate — fixed)
    [0.0, 0.05],   # hlp_goal_flag_dist_penalty (measured-good range — see Design Log 11)
])


class TestHlpSampleParams:
    def test_shape_and_bounds(self):
        theta = hlp_sample_params(64, PARAM_RANGES, DEVICE)
        assert theta.shape == (64, HLP_NUM_PARAMS)
        assert (theta >= PARAM_RANGES[:, 0]).all(), "samples below min"
        assert (theta <= PARAM_RANGES[:, 1]).all(), "samples above max"

    def test_degenerate_range_is_constant(self):
        """min == max must yield exactly that value (fixed-coefficient training)."""
        theta = hlp_sample_params(32, PARAM_RANGES, DEVICE)
        assert (theta[:, 2] == 0.05).all(), "degenerate range must be constant"
        assert (theta[:, 4] == 0.0).all()

    def test_wide_range_actually_varies(self):
        theta = hlp_sample_params(256, PARAM_RANGES, DEVICE)
        assert theta[:, 0].std().item() > 0.0, "non-degenerate range must vary"


class TestBuildHlpInputTheta:
    def _args_ns(self):
        from types import SimpleNamespace
        return SimpleNamespace(visible_width=8.0, visible_height=11.0, max_speed=30.0)

    def test_theta_normalized_when_ranges_given(self):
        """[HLP-param-norm] with param_ranges, the last P features are θ mapped to [-1, 1]."""
        obs = _make_obs(B, car_rows=[10])
        # ranges: dim d has [d, d+2]; one degenerate dim to check the 0 fallback
        ranges = torch.tensor([[float(d), float(d) + 2.0] for d in range(HLP_NUM_PARAMS)])
        ranges[-1] = torch.tensor([3.0, 3.0])  # degenerate → feature must be 0
        # raw = midpoint of each free range → normalized 0; low edge → -1; high edge → +1
        raw_mid = torch.stack([ranges[:, 0] + 1.0 for _ in range(B)])  # midpoints
        hlp_in, _, _ = build_hlp_input(obs, K, self._args_ns(), DEVICE, raw_mid, param_ranges=ranges)
        feats = hlp_in[:, -HLP_NUM_PARAMS:]
        assert torch.allclose(feats, torch.zeros_like(feats), atol=1e-6), "midpoints → 0"

        raw_edges = torch.stack([ranges[:, 0], ranges[:, 1]] + [ranges[:, 0]] * (B - 2))
        hlp_in2, _, _ = build_hlp_input(obs, K, self._args_ns(), DEVICE, raw_edges, param_ranges=ranges)
        f = hlp_in2[:, -HLP_NUM_PARAMS:]
        assert torch.allclose(f[0, :-1], -torch.ones(HLP_NUM_PARAMS - 1), atol=1e-6), "low edge → -1"
        assert torch.allclose(f[1, :-1], torch.ones(HLP_NUM_PARAMS - 1), atol=1e-6), "high edge → +1"
        assert f[0, -1] == 0.0 and f[1, -1] == 0.0, "degenerate dim → 0 regardless of raw value"

    def test_normalize_theta_bounds(self):
        """[HLP-param-norm] normalize_theta maps [min,max] onto [-1,1], degenerate→0."""
        from modules.base_models.SSP.hlp_core import normalize_theta
        ranges = torch.tensor([[0.0, 0.1], [0.01, 0.025], [0.05, 0.05]])
        raw = torch.tensor([[0.0, 0.025, 0.05], [0.1, 0.01, 0.05], [0.05, 0.0175, 0.05]])
        out = normalize_theta(raw, ranges)
        assert torch.allclose(out[0, :2], torch.tensor([-1.0, 1.0]), atol=1e-6)
        assert torch.allclose(out[1, :2], torch.tensor([1.0, -1.0]), atol=1e-6)
        assert torch.allclose(out[2, :2], torch.tensor([0.0, 0.0]), atol=1e-6)
        assert (out[:, 2] == 0.0).all(), "degenerate range → 0 for all rows"


class TestLazyParamResample:
    """[HLP-param] θ redraw happens only at block init, only for flagged envs."""

    def test_only_flagged_envs_resampled(self):
        torch.manual_seed(0)
        theta = hlp_sample_params(B, PARAM_RANGES, DEVICE)
        before = theta.clone()
        pending = torch.tensor([True, False, True, False])

        # Mirror of block-init step 0
        if pending.any():
            n_reset = int(pending.sum().item())
            theta[pending] = hlp_sample_params(n_reset, PARAM_RANGES, DEVICE)
            pending = torch.zeros_like(pending)

        # Unflagged rows untouched; flagged rows redrawn (wide columns differ a.s.)
        assert torch.equal(theta[1], before[1]) and torch.equal(theta[3], before[3])
        assert not torch.equal(theta[0], before[0]) or not torch.equal(theta[2], before[2])
        assert not pending.any(), "flag must clear after resample"

    def test_pending_latch_mirrors_block_close(self):
        pending = torch.tensor([False, True, False, False])
        episode_done_this_block = torch.tensor([True, False, False, True])
        pending = pending | episode_done_this_block
        assert pending.tolist() == [True, True, False, True]


class TestPerEnvRewardComposition:
    """[HLP-param] identical events, different θ → different block rewards."""

    def test_reward_uses_per_env_columns(self):
        theta = torch.tensor([
            [0.1, 0.00, 0.05, 0.0, 0.0],   # env 0: collision penalty 0.1
            [0.5, 0.02, 0.05, 0.2, 0.1],   # env 1: harsher everywhere
        ])
        collision_count    = torch.tensor([2.0, 2.0])
        cars_shown         = torch.tensor([3.0, 3.0])
        flag_miss_count    = torch.tensor([1.0, 1.0])
        flag_collect_count = torch.tensor([1.0, 1.0])
        goal_cost          = torch.tensor([0.0, 0.0])

        # Mirror of the block-close composition (env/goal terms zeroed for clarity)
        reward = (
            - theta[:, 0] * collision_count
            - theta[:, 1] * cars_shown
            - goal_cost
            - theta[:, 3] * flag_miss_count
            + theta[:, 4] * flag_collect_count
        )
        assert reward[0].item() == pytest.approx(-0.2)                    # -0.1*2
        assert reward[1].item() == pytest.approx(-1.0 - 0.06 - 0.2 + 0.1) # per-env values
        assert reward[0].item() != reward[1].item(), "same events, different θ → different reward"


class TestParamNamesContract:
    def test_canonical_order_stable(self):
        """The θ layout contract — changing this breaks every trained checkpoint."""
        assert HLP_PARAM_NAMES == [
            "hlp_collision_penalty",
            "hlp_cognitive_cost_coef",
            "hlp_goal_proximity_cost_coef",
            "hlp_flag_miss_penalty",
            "hlp_flag_collect_reward",
            "hlp_goal_flag_dist_penalty",   # [HLP-flag] 6th coef, added 2026-07-13 (Design Log 8)
        ]
        assert HLP_NUM_PARAMS == 6


# ---------------------------------------------------------------------------
# [HLP-maskhead] PooledMaskHead — the properties the global head provably lacks
# ---------------------------------------------------------------------------

class TestPooledMaskHead:
    def test_equivariant_to_slot_permutation(self):
        """THE point of the change: permuting the car slots must permute the logits
        identically. The global head cannot do this (it must learn K routings; measured
        2026-07-15 to have collapsed to one shared logit)."""
        torch.manual_seed(0)
        head = PooledMaskHead(K=K, feat_dim=EXPANDED_FEATURES, context_dim=8, hidden_size=16).eval()
        cars = torch.randn(2, K, EXPANDED_FEATURES)
        ctx = torch.randn(2, 8)
        perm = torch.randperm(K)
        with torch.no_grad():
            base = head(cars, ctx)
            permuted = head(cars[:, perm, :], ctx)
        assert torch.allclose(base[:, perm], permuted, atol=1e-6), \
            "logits must follow the slots — equivariance is the whole point"

    def test_logits_differ_across_slots(self):
        """Distinct cars must get distinct logits — the failure mode being fixed is
        all-K-logits-identical (which forces all-ON/all-OFF masks)."""
        torch.manual_seed(0)
        head = PooledMaskHead(K=K, feat_dim=EXPANDED_FEATURES, context_dim=8, hidden_size=16).eval()
        cars = torch.randn(1, K, EXPANDED_FEATURES) * 3.0
        with torch.no_grad():
            logits = head(cars, torch.randn(1, 8))
        assert logits.std().item() > 1e-3, "per-slot logits collapsed to a constant"

    def test_slot_scored_RELATIVE_to_the_field(self):
        """The objection this head answers: a per-slot scorer must COMPARE cars, not price
        each in isolation. So car A's logit must CHANGE when car B enters the scene —
        that shift is the pooled summary doing the comparison."""
        torch.manual_seed(0)
        head = PooledMaskHead(K=K, feat_dim=EXPANDED_FEATURES, context_dim=8, hidden_size=16).eval()
        ctx = torch.randn(1, 8)
        alone = torch.zeros(1, K, EXPANDED_FEATURES)
        alone[0, 0] = torch.randn(EXPANDED_FEATURES) * 3.0      # car A only
        with_b = alone.clone()
        with_b[0, 1] = torch.randn(EXPANDED_FEATURES) * 3.0     # car A + car B
        with torch.no_grad():
            la, lb = head(alone, ctx), head(with_b, ctx)
        assert not torch.allclose(la[:, 0], lb[:, 0], atol=1e-4), \
            "car A's logit must react to car B — otherwise cars are priced in isolation"

    def test_empty_slots_do_not_pollute_or_NaN(self):
        """Empty slots are all-zero (build_hlp_input convention). Two guards: pooling
        divides by VALID count (not K), and the masked max must not leak -inf."""
        torch.manual_seed(0)
        head = PooledMaskHead(K=K, feat_dim=EXPANDED_FEATURES, context_dim=8, hidden_size=16).eval()
        ctx = torch.randn(1, 8)
        # all slots empty -> clamp(min=1) on the divisor and nan_to_num on the max must hold
        with torch.no_grad():
            empty = head(torch.zeros(1, K, EXPANDED_FEATURES), ctx)
        assert torch.isfinite(empty).all(), "all-empty scene produced NaN/inf logits"
        # mean must be over VALID slots only: one car in a K-slot scene pools to that car,
        # so moving it between empty slots leaves ITS logit unchanged (equivariance + masking)
        one = torch.zeros(1, K, EXPANDED_FEATURES); one[0, 0] = torch.randn(EXPANDED_FEATURES)
        moved = torch.zeros(1, K, EXPANDED_FEATURES); moved[0, 2] = one[0, 0]
        with torch.no_grad():
            a, b = head(one, ctx), head(moved, ctx)
        assert torch.allclose(a[:, 0], b[:, 2], atol=1e-6), \
            "a lone car's logit must not depend on WHICH slot holds it"

    def test_agent_forward_shapes(self):
        """Action, logprob, entropy, value and goal shapes."""
        torch.manual_seed(0)
        agent = _goal_agent()
        base = _pooled_base()
        actions, logp, ent, value, ginfo = agent.get_action_and_value(
            base, x_m=torch.full((B,), 4.0), y_m=torch.full((B,), 50.0),
        )
        assert actions["mask"].shape == (B, K)
        assert logp.shape == (B,) and ent.shape == (B,) and value.shape == (B,)
        assert torch.isfinite(logp).all() and torch.isfinite(value).all()
        assert ginfo is not None and ginfo["goal_dxdy"].shape == (B, 2)

    def test_deterministic_mask_is_repeatable(self):
        torch.manual_seed(0)
        agent = _goal_agent()
        base = _pooled_base()
        kw = dict(x_m=torch.full((B,), 4.0), y_m=torch.full((B,), 50.0), deterministic=True)
        a1, *_ = agent.get_action_and_value(base, **kw)
        a2, *_ = agent.get_action_and_value(base, **kw)
        assert torch.equal(a1["mask"], a2["mask"])


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
