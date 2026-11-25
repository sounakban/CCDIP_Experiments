"""
Correctness gate, layer one: the torch recursion against the numpy reference.

WHAT IS BEING TESTED
    `inference_torch.forward_pass_torch` must reproduce `inference.forward_pass`
    numerically on geometry that exercises EVERY branch of the transition logic:
    one-to-one relabels, merging relabels (two attention patterns landing on the same
    target cell), steps where every subtask dies, and the crash-knockback direction (labels
    shifted upward off the grid).

    The geometry is synthetic -- a hand-built kernel stand-in -- so this layer runs on
    CPU in under a second with no checkpoint. The second layer of the gate, equality on
    REAL episode tables, runs in the step-2 gate script once the GPU is free.

    A test also proves the equality check CAN fail (one perturbed cell must break it),
    per the repo rule that a check which has never failed on known-bad input is not
    evidence.

No checkpoint and no GPU.
"""

import os
import unittest
from dataclasses import dataclass, field

import numpy as np
import torch

from human_modeling.inference.inference import _keep_plan, forward_pass
from human_modeling.inference.inference_torch import TorchPlan, forward_pass_torch

# ===========================================================================
# Module-level constants
# ===========================================================================

# | Small on purpose: 2 lanes x 4 bins = 8 goals, 2 cars = 4 attention patterns,
# | 7 steps. Big enough to hold every branch, small enough to reason about by hand.
NL, NB, K, N = 2, 4, 2, 7
G = NL * NB
M = 1 << K

# | Numerical agreement demanded between the two implementations. Both run float64 and
# | perform the same operations in the same order up to library-internal summation, so
# | anything looser than this would hide a real defect.
TOL = 1e-10


@dataclass
class FakeKernel:
    """
    A hand-built stand-in carrying exactly the attributes the transition plan reads.

    Built so the six transitions cover every branch: identity relabels, a MERGING
    relabel at step 2, total death at step 3 (a shift larger than the grid), the
    crash-knockback direction at step 4 (negative shift pushing bins UP), and scattered
    endings elsewhere.

    :ivar N: Trajectory steps.
    :vartype N: int
    :ivar G: Goals (lanes times bins).
    :vartype G: int
    :ivar NB: Distance bins per lane.
    :vartype NB: int
    """

    N:  int = N
    G:  int = G
    NB: int = NB
    n_valid:    np.ndarray = field(default_factory=lambda: np.full(N, K))
    goal_shift: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, NB, -1, 2]))
    goal_ended: np.ndarray = field(default_factory=lambda: _make_ended())
    mask_map:   list = field(default_factory=lambda: _make_mask_maps())

    def masks_at(self, t: int) -> np.ndarray:
        """All attention-pattern integers valid at step ``t`` (every one, here)."""
        return np.arange(M, dtype=np.int64)


def _make_ended() -> np.ndarray:
    """Goal endings per transition: none, a few, none, none, none, one."""
    ended = np.zeros((N - 1, G), dtype=bool)
    ended[1, [1, 5]] = True
    ended[5, [2]] = True
    return ended


def _make_mask_maps() -> list:
    """Identity relabels everywhere except a MERGE at transition 2 (1 and 2 collide)."""
    ident = np.arange(M, dtype=np.int64)
    maps = [ident.copy() for _ in range(N - 1)]
    maps[2] = np.array([0, 1, 1, 3], dtype=np.int64)
    return maps


def _plan(rho: list, kern: FakeKernel, device: str = "cpu") -> TorchPlan:
    """The torch transition plan for these tables, as `EpisodeGradContext` builds it."""
    return TorchPlan(kern, [a.shape for a in rho], device=device)


def make_tables(seed: int) -> tuple[list, list, FakeKernel]:
    """
    Random log-space tables on the fake geometry, with impossible cells sprinkled in.

    :param seed: Random seed; varied across tests so agreement is not one lucky draw.
    :type seed: int
    :return: Choice tables, emission tables, and the kernel they belong to.
    :rtype: tuple[list, list, FakeKernel]
    """
    rng = np.random.default_rng(seed)
    kern = FakeKernel()
    rho, emis = [], []
    for t in range(N):
        r = rng.standard_normal((G, M))
        b = 0.5 * rng.standard_normal((G, M))
        r[rng.random((G, M)) < 0.08] = -np.inf       # impossible choices exist
        rho.append(r)
        emis.append(b)
    return rho, emis, kern


class ForwardTorchMatchesNumpy(unittest.TestCase):
    """Numerical equality on geometry that provably exercises every branch."""

    def test_the_geometry_actually_covers_every_branch(self):
        """
        Prove the fake kernel hits each branch, so the equality below means something.

        Without this, all the agreement tests could be passing on geometry that never
        reaches the merge or total-death code at all.
        """
        rho, _, kern = make_tables(0)
        plan = _keep_plan(kern, [a.shape for a in rho])
        self.assertIn(False, plan.injective, "no merging relabel in the geometry")
        self.assertTrue(any(x is None for x in plan.flat_tgt),
                        "no step where every subtask dies")
        self.assertTrue(any(len(np.nonzero(~a)[0]) > 0 for a in plan.alive),
                        "no step with any dying subtask")
        self.assertTrue(any(len(np.nonzero(a)[0]) > 0 for a in plan.alive),
                        "no step with any surviving subtask")

    def test_matches_numpy_across_seeds(self):
        for seed in (0, 1, 2, 3, 4):
            rho, emis, kern = make_tables(seed)
            ref = forward_pass(rho, emis, kern)
            got = float(forward_pass_torch(rho, emis, kern, _plan(rho, kern)))
            self.assertTrue(np.isfinite(ref), f"seed {seed}: reference not finite")
            self.assertAlmostEqual(got, ref, delta=TOL * max(1.0, abs(ref)),
                                   msg=f"seed {seed}: torch {got!r} vs numpy {ref!r}")

    def test_gradient_flows_to_the_choice_tables(self):
        """
        The port's second purpose: exact gradients with no probing step.

        The choice tables are where the coefficients enter, so a gradient that reaches
        them (finite, not all zero) is the property the gradient-locating plan needs.
        """
        rho, emis, kern = make_tables(31)
        rho_t = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in rho]
        out = forward_pass_torch(rho_t, emis, kern, _plan(rho, kern))
        out.backward()
        grads = [a.grad for a in rho_t]
        self.assertTrue(all(g is not None for g in grads))
        finite_all = all(bool(torch.isfinite(g).all()) for g in grads)
        self.assertTrue(finite_all, "gradient not finite at some possible cell")
        total = sum(float(g.abs().sum()) for g in grads)
        self.assertGreater(total, 0.0, "gradient is identically zero")

    def test_the_equality_gate_can_fail(self):
        """One perturbed cell must break the agreement, or the gate proves nothing."""
        rho, emis, kern = make_tables(2)
        ref = forward_pass(rho, emis, kern)
        rho_bad = [a.copy() for a in rho]
        # | Two traps this test has already fallen into, both now avoided. The cell must
        # | be POSSIBLE (adding to -inf changes nothing), and it must be CONSULTED: when
        # | no subtask dies at a transition, no new one is drawn, and the next step's
        # | choice table is legitimately ignored -- in this geometry that makes step 3's
        # | table influence-free (verified: perturbing it moves the total by exactly 0 in
        # | both implementations). Step 1 is consulted, measured at ~5e-5 per 0.05 nudge.
        gi, mi = np.argwhere(np.isfinite(rho_bad[1]))[0]
        rho_bad[1][gi, mi] += 0.05
        got = float(forward_pass_torch(rho_bad, emis, kern, _plan(rho, kern)))
        self.assertNotAlmostEqual(got, ref, delta=TOL * max(1.0, abs(ref)))

    def test_gpu_matches_cpu_when_permitted(self):
        """
        Device equality; opt-in so unit runs never contend with a GPU job in flight.

        Enabled by ``ALLOW_GPU_TESTS=1``; the step-3 gate script sets it once the
        current chain has drained.
        """
        if not (torch.cuda.is_available() and os.environ.get("ALLOW_GPU_TESTS") == "1"):
            self.skipTest("GPU test disabled (set ALLOW_GPU_TESTS=1 with a free GPU)")
        rho, emis, kern = make_tables(5)
        cpu = float(forward_pass_torch(rho, emis, kern, _plan(rho, kern, "cpu")))
        gpu = float(forward_pass_torch(rho, emis, kern, _plan(rho, kern, "cuda")))
        self.assertAlmostEqual(gpu, cpu, delta=1e-9 * max(1.0, abs(cpu)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
