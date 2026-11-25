"""Checkpoint-free tests of hlp_policy.py: loading, theta construction, and the batched
all-goals mask logits against a straight per-goal evaluation of the pooled head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.base_models.SP.hrl_llp_core import LLP_LANE_X_CENTERS
from modules.base_models.SSP.hlp_core import (
    OBS_SIZE, OBS_FEATURES, CAR_KIND_IDS, EXPANDED_FEATURES,
    HLP_PARAM_NAMES, HLP_NUM_PARAMS, FLAG_FEAT_DIM, HLPAgent,
)
from modules.base_models.SSP.hlp_policy import (
    _prepare, hlp_all_goals_and_mask_logits, hlp_params_tensor, load_hlp_policy,
    pooled_mask_logits_all_goals,
)

K = 4
HIDDEN = 16
Y_MAX = 1000.0
INPUT_DIM = (OBS_SIZE + K) * EXPANDED_FEATURES + FLAG_FEAT_DIM + HLP_NUM_PARAMS
CAR_ID = CAR_KIND_IDS[0]

PARAM_CONFIG = [
    {"name": "hlp_collision_penalty",        "min": 0.1,  "max": 0.5},
    {"name": "hlp_cognitive_cost_coef",      "min": 0.0,  "max": 0.04},
    {"name": "hlp_goal_proximity_cost_coef", "min": 0.05, "max": 0.05},
    {"name": "hlp_flag_miss_penalty",        "min": 0.0,  "max": 0.3},
    {"name": "hlp_flag_collect_reward",      "min": 0.0,  "max": 0.0},
    {"name": "hlp_goal_flag_dist_penalty",   "min": 0.0,  "max": 0.2},
]


def _make_meta(input_dim: int = INPUT_DIM) -> dict:
    """Checkpoint metadata fixture with every key the loader and the input builder read."""
    return {
        "hlp_param_config": PARAM_CONFIG,
        "hlp_input_dim": input_dim,
        "hlp_hidden_size": HIDDEN,
        "hlp_num_car_slots": K,
        "hlp_critic_hidden_size_multiplier": 1,
        "num_goal_dy_bins": 18,
        "goal_dy_bin_width_eff": 0.5,
        "goal_forward_m_min": 1.0,
        "visible_width": 8.0,
        "visible_height": 11.0,
        "max_speed": 30.0,
        "y_max": Y_MAX,
    }


def _make_run_dir(tmp_path: Path, meta: dict | None = None) -> Path:
    """Write a loadable run dir: metadata plus the weights of a freshly initialised agent."""
    torch.manual_seed(7)
    agent = HLPAgent(
        input_dim=INPUT_DIM, K=K, hidden_size=HIDDEN,
        num_lanes=LLP_LANE_X_CENTERS.shape[0],
        num_dy_bins=18, goal_dy_min=1.0, goal_dy_bin_width=0.5,
        lane_centers=LLP_LANE_X_CENTERS,
    )
    (tmp_path / "hlp_metadata.json").write_text(json.dumps(meta or _make_meta()), encoding="utf-8")
    torch.save(agent.state_dict(), tmp_path / "hlp_model_final.pt")
    return tmp_path


def _make_raw_obs(car_rows: list[int] | None = None) -> torch.Tensor:
    """(obs_flat,) raw obs: ego at x=3.0, y_norm=0.5, optional car rows."""
    obs = torch.zeros(OBS_SIZE, OBS_FEATURES)
    obs[0, 0] = 3.0    # ego x (m)
    obs[0, 1] = 0.5    # ego y (normalised)
    for r in (car_rows or []):
        obs[r, 6] = float(CAR_ID)
        obs[r, 0] = 1.0
    return obs.view(-1)


def _mask_logits_one_goal(agent: HLPAgent, base_in: torch.Tensor, goal_dxdy: torch.Tensor) -> torch.Tensor:
    """The reference: the pooled head evaluated directly for ONE goal per row, exactly as
    ``HLPAgent.get_action_and_value`` does it."""
    c0 = OBS_SIZE * EXPANDED_FEATURES
    c1 = c0 + agent.K * EXPANDED_FEATURES
    car_feats = base_in[:, c0:c1].view(-1, agent.K, EXPANDED_FEATURES)
    context = torch.cat([base_in[:, :c0], base_in[:, c1:], goal_dxdy], dim=1)
    return agent.pooled_actor(car_feats, context)


class TestLoadHlpPolicy:
    def test_load_and_eval(self, tmp_path):
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        assert isinstance(agent, HLPAgent) and not agent.training
        assert meta["hlp_param_config"] == PARAM_CONFIG

    def test_rejects_pre_param_checkpoint(self, tmp_path):
        meta = _make_meta()
        del meta["hlp_param_config"]
        with pytest.raises(RuntimeError, match="parameterized"):
            load_hlp_policy(_make_run_dir(tmp_path, meta))

    def test_era_guard_on_dim_mismatch(self, tmp_path):
        """Metadata claiming a different input_dim must fail with a readable error."""
        run_dir = _make_run_dir(tmp_path, _make_meta(input_dim=INPUT_DIM + 5))
        with pytest.raises(AssertionError, match="incompatible checkpoint era"):
            load_hlp_policy(run_dir)


class TestHlpParamsTensor:
    def test_midpoint_defaults_and_order(self):
        theta = hlp_params_tensor(_make_meta(), None, torch.device("cpu"), B=3)
        assert theta.shape == (3, HLP_NUM_PARAMS)
        expected = [0.5 * (p["min"] + p["max"]) for p in PARAM_CONFIG]
        assert torch.allclose(theta[0], torch.tensor(expected))

    def test_override_applied(self):
        theta = hlp_params_tensor(_make_meta(), {"hlp_flag_miss_penalty": 0.25}, torch.device("cpu"))
        assert theta[0, HLP_PARAM_NAMES.index("hlp_flag_miss_penalty")].item() == pytest.approx(0.25)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown HLP parameters"):
            hlp_params_tensor(_make_meta(), {"not_a_param": 1.0}, torch.device("cpu"))

    def test_out_of_range_warns_and_clamps(self):
        with pytest.warns(UserWarning, match="outside training range"):
            theta = hlp_params_tensor(_make_meta(), {"hlp_collision_penalty": 9.0}, torch.device("cpu"))
        assert theta[0, 0].item() == pytest.approx(0.5), "must clamp to range max"


class TestPooledMaskLogitsAllGoals:
    """pooled_mask_logits_all_goals is an EXACT reformulation of the head's forward pass
    (the decomposition is linear-algebra-exact), so every test checks agreement to float
    rounding, not just that it runs."""

    def test_single_goal_matches_direct_evaluation(self, tmp_path):
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        _, base_in, *_ = _prepare(agent, meta, _make_raw_obs(car_rows=[100, 110, 120]), None)
        goal_dxdy = torch.tensor([[1.5, -2.0]])                      # (1, 2)
        expected = _mask_logits_one_goal(agent, base_in, goal_dxdy)  # (1, K)
        actual = pooled_mask_logits_all_goals(agent, base_in, goal_dxdy.unsqueeze(1))  # (1, 1, K)
        assert torch.allclose(actual.squeeze(1), expected, atol=1e-5)

    def test_multiple_goals_each_match_individual_calls(self, tmp_path):
        """The batched N>1 path must agree with one direct evaluation per candidate."""
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        _, base_in, *_ = _prepare(agent, meta, _make_raw_obs(car_rows=[100, 110]), None)
        candidates = torch.tensor([[0.0, 0.0], [3.0, -1.0], [-2.5, 4.0], [1.0, 1.0]])  # (4, 2)
        actual = pooled_mask_logits_all_goals(agent, base_in, candidates.unsqueeze(0))  # (1, 4, K)
        for i in range(candidates.shape[0]):
            expected_i = _mask_logits_one_goal(agent, base_in, candidates[i : i + 1])  # (1, K)
            assert torch.allclose(actual[0, i], expected_i[0], atol=1e-5), f"candidate {i} mismatch"

    def test_multiple_rows_each_match_individual_calls(self, tmp_path):
        """Same cross-check varying the ROW (timestep): rows must stay independent."""
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        _, base_a, *_ = _prepare(agent, meta, _make_raw_obs(car_rows=[100]), None)
        _, base_b, *_ = _prepare(agent, meta, _make_raw_obs(car_rows=[100, 110, 120]), None)
        base_in = torch.cat([base_a, base_b], dim=0)                 # (2, input_dim)
        candidates = torch.tensor([[1.0, -1.0], [2.0, 2.0]]).unsqueeze(0).expand(2, -1, -1)  # (2, 2, 2)
        actual = pooled_mask_logits_all_goals(agent, base_in, candidates)  # (2, 2, K)
        for row in range(2):
            for i in range(2):
                expected = _mask_logits_one_goal(agent, base_in[row : row + 1], candidates[row, i : i + 1])
                assert torch.allclose(actual[row, i], expected[0], atol=1e-5), f"row={row} candidate={i}"


class TestHlpAllGoalsAndMaskLogits:
    """hlp_all_goals_and_mask_logits must agree with the agent's own hard-argmax decision at
    the argmax candidate: restricting exact marginalisation to the single highest-probability
    goal is a literal special case of it."""

    def test_goal_logp_normalized(self, tmp_path):
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        out = hlp_all_goals_and_mask_logits(agent, meta, _make_raw_obs(car_rows=[100, 110]))
        assert out["goal_logp"].shape == (1, out["num_lanes"] * out["num_dy_bins"])
        assert torch.allclose(out["goal_logp"].exp().sum(dim=1), torch.ones(1), atol=1e-5)

    def test_argmax_candidate_matches_the_agents_greedy_decision(self, tmp_path):
        agent, meta = load_hlp_policy(_make_run_dir(tmp_path))
        obs = _make_raw_obs(car_rows=[100, 110, 120])
        _, base_in, _, _, x_m, y_m = _prepare(agent, meta, obs, None)
        with torch.no_grad():
            actions, _, _, _, goal_info = agent.get_action_and_value(base_in, x_m=x_m, y_m=y_m, deterministic=True)
            gl = agent.goal_actor(base_in)
            lane_logp = torch.log_softmax(gl[:, : agent.num_lanes], dim=-1)
            bin_logp = torch.log_softmax(gl[:, agent.num_lanes :], dim=-1)
            hard_logp = lane_logp[0, actions["lane"][0]] + bin_logp[0, actions["dy_bin"][0]]
            hard_mask_logits = _mask_logits_one_goal(agent, base_in, goal_info["goal_dxdy"])
        all_g = hlp_all_goals_and_mask_logits(agent, meta, obs)
        argmax_n = (actions["lane"] * all_g["num_dy_bins"] + actions["dy_bin"]).item()
        assert torch.allclose(all_g["goal_dxdy"][0, argmax_n], goal_info["goal_dxdy"][0], atol=1e-5)
        assert torch.allclose(all_g["mask_logits"][0, argmax_n], hard_mask_logits[0], atol=1e-5)
        assert torch.allclose(all_g["goal_x"][0, argmax_n], goal_info["goal_x"][0], atol=1e-5)
        assert torch.allclose(all_g["goal_y"][0, argmax_n], goal_info["goal_y"][0], atol=1e-5)
        assert torch.allclose(hard_logp, all_g["goal_logp"][0, argmax_n], atol=1e-5)
        assert argmax_n == all_g["goal_logp"][0].argmax().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
