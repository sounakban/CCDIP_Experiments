"""
Checkpoint-free properties of V4's differentiable likelihood components.

WHAT RUNS HERE AND WHAT DOES NOT
    The full-path correctness gate — agreement with the numpy production likelihood at
    2.14e-16 on real episodes — needs the checkpoint and the GPU, so it lives in the spike
    scripts, not this suite. What CAN be pinned without a checkpoint is everything this
    package added around the policy: the 6-slot theta expansion.

A TRAP THIS FILE EXISTS TO AVOID
    The first gate's "can it fail" check perturbed the SHARED emission tables — both paths
    moved together, which proves consistency but not that the gate would catch a defect in
    the PORT itself. The known-bad test here breaks exactly one side (a scrambled theta
    order) and requires disagreement.
"""

import os
import sys
import unittest

import numpy as np
import torch

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.likelihood_grad import theta6_tensor

class ThetaExpansion(unittest.TestCase):
    """The 4 -> 6 slot expansion must respect the checkpoint's config order."""

    META = {"hlp_param_config": [
        {"name": "hlp_collision_penalty", "min": 0.0, "max": 10.0},
        {"name": "hlp_cognitive_cost_coef", "min": 0.0, "max": 0.15},
        {"name": "hlp_goal_proximity_cost_coef", "min": 0.0, "max": 2.0},
        {"name": "hlp_flag_miss_penalty", "min": 0.0, "max": 0.0},
        {"name": "hlp_flag_collect_reward", "min": 0.0, "max": 0.0},
        {"name": "hlp_goal_flag_dist_penalty", "min": 0.0, "max": 0.5},
    ]}

    def test_fitted_values_land_in_their_config_slots(self):
        t4 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t6 = theta6_tensor(t4, self.META)
        order = [d["name"] for d in self.META["hlp_param_config"]]
        for name, want in zip(THETA_NAMES, [1.0, 2.0, 3.0, 4.0]):
            self.assertEqual(float(t6[order.index(name)]), want)
        for pin in ("hlp_flag_miss_penalty", "hlp_flag_collect_reward"):
            self.assertEqual(float(t6[order.index(pin)]), 0.0)

    def test_a_scrambled_config_order_is_followed_not_assumed(self):
        """The expansion must READ the order; a reordered config must reorder the output —
        this is the known-bad direction for a hardcoded index map."""
        meta = {"hlp_param_config": list(reversed(self.META["hlp_param_config"]))}
        t6 = theta6_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0]), meta)
        self.assertEqual(float(t6[-1]), 1.0)      # collision now last
        self.assertEqual(float(t6[0]), 4.0)       # goal_flag now first

    def test_gradient_reaches_every_fitted_slot(self):
        t4 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        theta6_tensor(t4, self.META).sum().backward()
        self.assertTrue(bool((t4.grad == 1.0).all()),
                        "stacking lost a fitted element from the graph")


if __name__ == "__main__":
    unittest.main(verbosity=2)
